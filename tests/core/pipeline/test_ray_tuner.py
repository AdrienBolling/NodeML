"""Tests for :class:`nodeml.core.pipeline.tuners.ray_tuner.RayPipelineTuner`.

These tests exercise the tuner's internals and lifecycle without standing
up a real Ray cluster:

* ``_apply_hyperparameters`` — flat config merges, snapshot immutability,
  validation errors.
* ``_convert_metrics`` — scalar extraction from pandas and numpy backends,
  and one key per value for multi-value metrics.
* ``default_hyperparameter_space`` — key namespacing and omission of
  non-tunable nodes.
* ``_trainable`` — factory argument validation, and one in-process trial
  with the Ray session calls replaced.
* ``tune`` and ``get_best`` — with a fake ``ray.tune.Tuner``.
"""

from __future__ import annotations

import functools
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from ray.tune import TuneConfig

from nodeml.components.nodes.data_sources.inputs_passthrough import (
    InputsPassthroughConfig,
)
from nodeml.components.nodes.transforms.feature_selection.missing_rate_filter import (
    MissingRateFilterConfig,
)
from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    NumericalData,
    TabularDataContext,
)
from nodeml.core.common.exceptions import NodeConfigError
from nodeml.core.nodes.data_sink.sink import SinkConfig
from nodeml.core.nodes.node import Port
from nodeml.core.pipeline.pipeline import Edge, Pipeline, PipelineConfig
from nodeml.core.pipeline.runners.smart_runner import SmartRunner
from nodeml.core.pipeline.tuners import ray_tuner
from nodeml.core.pipeline.tuners.ray_tuner import (
    RayPipelineTuner,
    RayPipelineTunerConfig,
)
from tests.shims.pipelines import build_source_model_sink_pipeline


def _build_filter_pipeline() -> Pipeline:
    """Pipeline: source → MissingRateFilter → sink.

    ``MissingRateFilter`` exposes a ``threshold`` hyperparameter, which is
    what exercises the tuner's tunable-node discovery and HP injection.
    """
    mixed_port = Port(
        arr_type=ArrayLikeEnum.PANDAS,
        data_structure=DataStructureEnum.TABULAR,
        data_category=DataCategoryEnum.MIXED,
        data_shape="batch feature",
        desc="mixed features",
    )
    source_cfg = InputsPassthroughConfig(out_ports={"X": mixed_port})
    sink_cfg = SinkConfig(in_ports={"out": mixed_port})
    nodes: dict[str, tuple[str, object]] = {
        "source": ("InputsPassthrough", source_cfg),
        "filter": ("MissingRateFilter", MissingRateFilterConfig()),
        "sink": ("Sink", sink_cfg),
    }
    edges = [
        Edge(source="source", target="filter", ports_map=[("X", "input")]),
        Edge(source="filter", target="sink", ports_map=[("output", "out")]),
    ]
    pipe = Pipeline(config=PipelineConfig(nodes=nodes, edges=edges))
    pipe.compile()
    return pipe


def _build_non_tunable_pipeline() -> Pipeline:
    """Pipeline: source → sink.  No tunable nodes."""
    port = Port(
        arr_type=ArrayLikeEnum.PANDAS,
        data_structure=DataStructureEnum.TABULAR,
        data_category=DataCategoryEnum.NUMERICAL,
        data_shape="batch feature",
        desc="x",
    )
    source_cfg = InputsPassthroughConfig(out_ports={"X": port})
    sink_cfg = SinkConfig(in_ports={"X": port})
    nodes: dict[str, tuple[str, object]] = {
        "source": ("InputsPassthrough", source_cfg),
        "sink": ("Sink", sink_cfg),
    }
    edges = [Edge(source="source", target="sink", ports_map=[("X", "X")])]
    pipe = Pipeline(config=PipelineConfig(nodes=nodes, edges=edges))
    pipe.compile()
    return pipe


def _scalar_ctx() -> TabularDataContext:
    return TabularDataContext(
        columns=["metric"],
        dtypes=[np.dtype("float64")],
        categories=[NumericalData],
    )


class TestConstructor:
    def test_tuner_is_none_initially(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        assert tuner.tuner is None

    def test_snapshot_isolated_from_original_pipeline(self) -> None:
        """Keep the tuner snapshot independent of the original pipeline.

        The tuner takes a config snapshot; later changes to the original
        pipeline must not propagate to the snapshot.
        """
        pipe = _build_filter_pipeline()
        tuner = RayPipelineTuner(pipe, config=RayPipelineTunerConfig())
        pipe.remove_node("filter")
        # Snapshot still holds the filter node even after we removed it
        # from the original pipeline.
        assert "filter" in tuner._pipeline_config.nodes


class TestDefaultHyperparameterSpace:
    def test_returns_namespaced_keys_for_tunable_nodes(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        space = tuner.default_hyperparameter_space()
        assert "filter/threshold" in space

    def test_omits_nodes_without_hyperparameter_space(self) -> None:
        tuner = RayPipelineTuner(
            _build_non_tunable_pipeline(), config=RayPipelineTunerConfig()
        )
        assert tuner.default_hyperparameter_space() == {}


class TestApplyHyperparameters:
    def test_merges_sampled_value_into_node_config(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        new_conf = tuner._apply_hyperparameters({"filter/threshold": 0.3})
        _, filter_conf = new_conf.nodes["filter"]
        assert filter_conf.hyperparameters.threshold == pytest.approx(0.3)

    def test_other_nodes_untouched(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        new_conf = tuner._apply_hyperparameters({"filter/threshold": 0.7})
        assert set(new_conf.nodes.keys()) == {"source", "filter", "sink"}

    def test_snapshot_not_mutated(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        original = tuner._pipeline_config.nodes["filter"][1].hyperparameters.threshold
        _ = tuner._apply_hyperparameters({"filter/threshold": 0.9})
        assert (
            tuner._pipeline_config.nodes["filter"][1].hyperparameters.threshold
            == original
        )

    def test_unknown_node_raises(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        with pytest.raises(ValueError, match="unknown node"):
            tuner._apply_hyperparameters({"ghost/threshold": 0.1})

    def test_node_without_hyperparameters_raises(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        with pytest.raises(ValueError, match="no hyperparameters"):
            tuner._apply_hyperparameters({"source/anything": 1})


class TestConvertMetrics:
    def test_pandas_dataframe_scalar_extraction(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        out = tuner._convert_metrics({"mse": (pd.DataFrame([[0.42]]), _scalar_ctx())})
        assert out == {"mse": pytest.approx(0.42)}

    def test_numpy_array_scalar_extraction(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        out = tuner._convert_metrics({"mae": (np.array([[1.25]]), _scalar_ctx())})
        assert out == {"mae": pytest.approx(1.25)}

    def test_multiple_metrics_preserved(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        out = tuner._convert_metrics(
            {
                "mse": (pd.DataFrame([[0.1]]), _scalar_ctx()),
                "mae": (pd.DataFrame([[0.2]]), _scalar_ctx()),
            }
        )
        assert out == {
            "mse": pytest.approx(0.1),
            "mae": pytest.approx(0.2),
        }

    def test_returns_python_floats(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        out = tuner._convert_metrics({"mse": (pd.DataFrame([[0.5]]), _scalar_ctx())})
        assert type(out["mse"]) is float


class TestTrainableFactory:
    def test_requires_metric_or_aggregator(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        with pytest.raises(
            ValueError, match="optimization_metric or metric_aggregator"
        ):
            tuner._trainable()

    def test_accepts_optimization_metric(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        fn = tuner._trainable(optimization_metric="mse")
        assert callable(fn)

    def test_accepts_metric_aggregator(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        fn = tuner._trainable(metric_aggregator=lambda metrics: sum(metrics.values()))
        assert callable(fn)


class TestGetBest:
    def test_raises_when_tune_not_called(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        with pytest.raises(ValueError, match="not been run yet"):
            tuner.get_best()


# ---------------------------------------------------------------------------
# Fake Ray Tune objects: they let ``tune()`` run without a Ray cluster.
# ---------------------------------------------------------------------------


@dataclass
class _FakeResult:
    config: dict[str, Any]
    metrics: dict[str, float]
    error: Exception | None = None


class _FakeResultGrid:
    """Rank results like ``ray.tune.ResultGrid.get_best_result``."""

    def __init__(self, results: list[_FakeResult], tune_config: TuneConfig) -> None:
        self._results = results
        self._tune_config = tune_config

    def __iter__(self) -> Iterator[_FakeResult]:
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def get_best_result(
        self, metric: str | None = None, mode: str | None = None
    ) -> _FakeResult:
        metric = metric or self._tune_config.metric
        mode = mode or self._tune_config.mode
        pick = min if mode == "min" else max
        return pick(self._results, key=lambda result: result.metrics[metric])


class _FakeTuner:
    """Record the ``tune.Tuner`` arguments and return two fixed results."""

    last: _FakeTuner | None = None

    def __init__(
        self,
        trainable: Any,
        *,
        param_space: dict[str, Any],
        tune_config: TuneConfig,
        run_config: Any,
    ) -> None:
        self.trainable = trainable
        self.param_space = param_space
        self.tune_config = tune_config
        self.run_config = run_config
        _FakeTuner.last = self

    def fit(self) -> _FakeResultGrid:
        results = [
            _FakeResult({"filter/threshold": 0.1}, {"optimization_metric": 1.0}),
            _FakeResult({"filter/threshold": 0.9}, {"optimization_metric": 2.0}),
        ]
        return _FakeResultGrid(results, self.tune_config)


@pytest.fixture
def fake_tuner(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTuner]:
    """Replace ``ray.tune.Tuner`` with :class:`_FakeTuner`."""
    monkeypatch.setattr(ray_tuner.tune, "Tuner", _FakeTuner)
    return _FakeTuner


class TestTune:
    def test_requires_mode(self, fake_tuner: type[_FakeTuner]) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        with pytest.raises(ValueError, match="mode"):
            tuner.tune({}, optimization_metric="mse")
        with pytest.raises(ValueError, match="mode"):
            tuner.tune({}, optimization_metric="mse", tune_config=TuneConfig())

    def test_does_not_change_user_tune_config(
        self, fake_tuner: type[_FakeTuner]
    ) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        user_config = TuneConfig(mode="min", num_samples=2)
        tuner.tune({}, optimization_metric="mse", tune_config=user_config)

        assert user_config.metric is None
        assert fake_tuner.last is not None
        assert fake_tuner.last.tune_config is not user_config
        assert fake_tuner.last.tune_config.metric == "optimization_metric"
        assert fake_tuner.last.tune_config.num_samples == 2

    def test_accepts_aggregator_without_name(
        self, fake_tuner: type[_FakeTuner]
    ) -> None:
        def weighted(metrics: dict[str, float], weight: float) -> float:
            return weight * metrics["mse"]

        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        tuner.tune(
            {},
            metric_aggregator=functools.partial(weighted, weight=2.0),
            tune_config=TuneConfig(mode="min"),
        )
        assert tuner.results is not None


class TestGetBestMode:
    def test_default_mode_is_the_tuning_mode(
        self, fake_tuner: type[_FakeTuner]
    ) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        tuner.tune({}, optimization_metric="mse", tune_config=TuneConfig(mode="min"))
        best = tuner.get_best()
        assert best.metrics["optimization_metric"] == pytest.approx(1.0)

        tuner.tune({}, optimization_metric="mse", tune_config=TuneConfig(mode="max"))
        best = tuner.get_best()
        assert best.metrics["optimization_metric"] == pytest.approx(2.0)

    def test_explicit_mode_overrides_default(
        self, fake_tuner: type[_FakeTuner]
    ) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        tuner.tune({}, optimization_metric="mse", tune_config=TuneConfig(mode="min"))
        best = tuner.get_best(mode="max")
        assert best.metrics["optimization_metric"] == pytest.approx(2.0)

    def test_other_metric_requires_mode(self, fake_tuner: type[_FakeTuner]) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        tuner.tune({}, optimization_metric="mse", tune_config=TuneConfig(mode="min"))
        with pytest.raises(ValueError, match="mode"):
            tuner.get_best(metric="r2")


class TestApplyHyperparametersValidation:
    def test_out_of_range_value_raises(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        with pytest.raises(NodeConfigError, match="filter"):
            tuner._apply_hyperparameters({"filter/threshold": 2.0})

    def test_wrong_type_raises(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        with pytest.raises(NodeConfigError, match="threshold"):
            tuner._apply_hyperparameters({"filter/threshold": "high"})

    def test_unknown_hyperparameter_raises(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        with pytest.raises(NodeConfigError, match="unknown hyperparameter 'ghost'"):
            tuner._apply_hyperparameters({"filter/ghost": 0.1})

    def test_key_without_node_name_raises(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        with pytest.raises(NodeConfigError, match="node_name/hp_name"):
            tuner._apply_hyperparameters({"threshold": 0.1})

    def test_value_is_coerced_by_the_model(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        new_conf = tuner._apply_hyperparameters({"filter/threshold": 1})
        _, filter_conf = new_conf.nodes["filter"]
        assert type(filter_conf.hyperparameters.threshold) is float


class TestConvertMultiValueMetrics:
    def test_expands_one_key_per_column(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        frame = pd.DataFrame([[0.1, 0.3]], columns=["score_0", "score_1"])
        ctx = TabularDataContext(
            columns=["score_0", "score_1"],
            dtypes=[np.dtype("float64")] * 2,
            categories=[NumericalData] * 2,
        )
        out = tuner._convert_metrics(
            {"mse": (frame, ctx), "mae": (pd.DataFrame([[0.2]]), _scalar_ctx())}
        )
        assert out == {
            "mse.score_0": pytest.approx(0.1),
            "mse.score_1": pytest.approx(0.3),
            "mae": pytest.approx(0.2),
        }

    def test_numpy_metric_uses_context_columns(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        ctx = TabularDataContext(
            columns=["a", "b"],
            dtypes=[np.dtype("float64")] * 2,
            categories=[NumericalData] * 2,
        )
        out = tuner._convert_metrics({"m": (np.array([[1.0, 2.0]]), ctx)})
        assert out == {"m.a": pytest.approx(1.0), "m.b": pytest.approx(2.0)}

    def test_more_than_one_row_raises(self) -> None:
        tuner = RayPipelineTuner(
            _build_filter_pipeline(), config=RayPipelineTunerConfig()
        )
        with pytest.raises(ValueError, match="one row"):
            tuner._convert_metrics({"m": (pd.DataFrame([[1.0], [2.0]]), _scalar_ctx())})


class TestTrainable:
    """Run one trial in-process: the Ray session calls are replaced."""

    def test_trial_reports_metrics_and_loadable_checkpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
        regression_dataset: Any,
        tmp_path: Path,
    ) -> None:
        reports: list[dict[str, float]] = []

        def fake_report(metrics: dict[str, float], checkpoint: Any = None) -> None:
            reports.append(metrics)
            shutil.copytree(checkpoint.path, tmp_path / "checkpoint")

        monkeypatch.setattr(ray_tuner.tune, "report", fake_report)
        monkeypatch.setattr(ray_tuner.tune, "get_checkpoint", lambda: None)
        monkeypatch.setattr(ray_tuner.tune, "get_context", lambda: None)

        X_pair, y_pair, _ = regression_dataset
        pipe = build_source_model_sink_pipeline(
            model_node_type="RandomForestRegressor", with_metric=True
        )
        tuner = RayPipelineTuner(pipe, config=RayPipelineTunerConfig())
        trainable = tuner._trainable(optimization_metric="metric")
        input_data = {"source": {"X": X_pair, "y": y_pair}}
        trainable(
            {"model/n_estimators": 3, "model/max_depth": 2}, input_data=input_data
        )

        assert len(reports) == 1
        assert reports[0]["optimization_metric"] == reports[0]["metric"]

        loaded = Pipeline.load_from_dir(tmp_path / "checkpoint")
        _, model_conf = loaded.nodes["model"]
        assert model_conf.hyperparameters.n_estimators == 3
        preds = SmartRunner(loaded).infer(input_data={"source": {"X": X_pair}})
        assert preds["pred"][0].shape == (len(X_pair[0]), 1)

    def test_unknown_optimization_metric_names_available_metrics(
        self,
        monkeypatch: pytest.MonkeyPatch,
        regression_dataset: Any,
    ) -> None:
        monkeypatch.setattr(ray_tuner.tune, "get_checkpoint", lambda: None)
        monkeypatch.setattr(ray_tuner.tune, "get_context", lambda: None)
        X_pair, y_pair, _ = regression_dataset
        pipe = build_source_model_sink_pipeline(with_metric=True)
        tuner = RayPipelineTuner(pipe, config=RayPipelineTunerConfig())
        trainable = tuner._trainable(optimization_metric="r2")
        with pytest.raises(KeyError, match="Available metrics"):
            trainable({}, input_data={"source": {"X": X_pair, "y": y_pair}})
