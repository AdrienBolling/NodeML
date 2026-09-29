"""Tests for :class:`nodeml.core.pipeline.tuners.ray_tuner.RayPipelineTuner`.

These tests exercise the tuner's internals and lifecycle without standing
up a real Ray cluster:

* ``_apply_hyperparameters`` — flat config merges, snapshot immutability,
  validation errors.
* ``_convert_metrics`` — scalar extraction from pandas and numpy backends.
* ``default_hyperparameter_space`` — key namespacing and omission of
  non-tunable nodes.
* ``_trainable`` — factory argument validation.
* ``tuner`` property and ``get_best`` — initial state before :meth:`tune`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
from nodeml.core.nodes.data_sink.sink import SinkConfig
from nodeml.core.nodes.node import Port
from nodeml.core.pipeline.pipeline import Edge, Pipeline, PipelineConfig
from nodeml.core.pipeline.tuners.ray_tuner import (
    RayPipelineTuner,
    RayPipelineTunerConfig,
)


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
