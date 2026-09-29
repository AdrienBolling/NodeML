"""Tests for :mod:`nodeml.core.pipeline.serving.ray_serve`.

The tests do not start a Ray cluster.  They cover:

* the JSON conversion helpers (types, column order, dtypes);
* :class:`ServedPipeline`, the Ray-free logic of the deployment, built from
  a config or from a saved directory;
* :func:`build_pipeline_app`, which applies the deployment options.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from ray.serve import Application

from nodeml.core.common.data.data import (
    CategoricalData,
    NumericalData,
    TabularDataContext,
)
from nodeml.core.pipeline.pipeline import Pipeline
from nodeml.core.pipeline.runners.smart_runner import SmartRunner
from nodeml.core.pipeline.serving import ray_serve
from nodeml.core.pipeline.serving.ray_serve import (
    EvaluateRequest,
    InferRequest,
    PipelineServing,
    PipelineServingConfig,
    PortPayload,
    ServedPipeline,
    _payload_to_tuple,
    _tuple_to_response,
    build_pipeline_app,
)
from tests.shims.pipelines import build_source_model_sink_pipeline


@pytest.fixture
def trained_pipeline(regression_dataset: Any) -> Pipeline:
    """Return a trained source → LinearRegression → sink pipeline with a metric."""
    X_pair, y_pair, _ = regression_dataset
    pipeline = build_source_model_sink_pipeline(with_metric=True)
    SmartRunner(pipeline).train(input_data={"source": {"X": X_pair, "y": y_pair}})
    return pipeline


@pytest.fixture
def saved_dir(trained_pipeline: Pipeline, tmp_path: Path) -> Path:
    """Directory with the config and the params of ``trained_pipeline``."""
    trained_pipeline.save_config_to_dir(tmp_path)
    trained_pipeline.save_params_to_dir(tmp_path)
    return tmp_path


def _expected_predictions(pipeline: Pipeline, X_pair: Any) -> np.ndarray:
    preds = SmartRunner(pipeline).infer(input_data={"source": {"X": X_pair}})
    return preds["pred"][0].to_numpy()


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dict(orient="records")


class TestTupleToResponse:
    def test_keeps_json_native_types(self) -> None:
        frame = pd.DataFrame(
            {
                "label": pd.Series(["a", "b"], dtype="object"),
                "count": pd.Series([1, 2], dtype="int64"),
                "score": pd.Series([0.5, np.nan], dtype="float64"),
                "flag": pd.Series([True, False], dtype="bool"),
            }
        )
        ctx = TabularDataContext(
            columns=list(frame.columns),
            dtypes=list(frame.dtypes),
            categories=[
                CategoricalData,
                NumericalData,
                NumericalData,
                CategoricalData,
            ],
        )
        response = _tuple_to_response(frame, ctx)

        assert response.data == [
            {"label": "a", "count": 1, "score": 0.5, "flag": True},
            {"label": "b", "count": 2, "score": None, "flag": False},
        ]
        assert type(response.data[0]["count"]) is int
        assert type(response.data[0]["flag"]) is bool
        # Starlette serialises with allow_nan=False.
        json.dumps(response.model_dump(), allow_nan=False)

    def test_numpy_output_uses_context_columns(self) -> None:
        ctx = TabularDataContext(
            columns=["pred"],
            dtypes=[np.dtype("float64")],
            categories=[NumericalData],
        )
        response = _tuple_to_response(np.array([[1.5], [np.inf]]), ctx)
        assert response.data == [{"pred": 1.5}, {"pred": None}]
        assert response.context == ctx.dump_dict


class TestPayloadToTuple:
    def test_uses_context_order_and_dtypes(self) -> None:
        ctx = TabularDataContext(
            columns=["x0", "x1"],
            dtypes=[np.dtype("float64"), np.dtype("float64")],
            categories=[NumericalData, NumericalData],
        )
        payload = PortPayload(
            data=[{"x1": 2, "x0": 1}, {"x1": 4, "x0": 3}],
            context=ctx.dump_dict,
        )
        frame, out_ctx = _payload_to_tuple(payload)

        assert list(frame.columns) == ["x0", "x1"]
        assert list(frame.dtypes) == [np.dtype("float64")] * 2
        assert frame["x0"].tolist() == [1.0, 3.0]
        assert out_ctx.columns == ["x0", "x1"]

    def test_missing_column_raises(self) -> None:
        ctx = TabularDataContext(
            columns=["x0", "x1"],
            dtypes=[np.dtype("float64"), np.dtype("float64")],
            categories=[NumericalData, NumericalData],
        )
        payload = PortPayload(data=[{"x0": 1.0}], context=ctx.dump_dict)
        with pytest.raises(ValueError, match="Missing columns: \\['x1'\\]"):
            _payload_to_tuple(payload)


class TestServedPipeline:
    def test_from_config_and_params_dir(
        self, trained_pipeline: Pipeline, saved_dir: Path, regression_dataset: Any
    ) -> None:
        X_pair, _, _ = regression_dataset
        served = ServedPipeline(
            pipeline_config=trained_pipeline.config, params_dir=saved_dir
        )
        preds = served.predict({"source": {"X": X_pair}})
        np.testing.assert_allclose(
            preds["pred"][0].to_numpy(),
            _expected_predictions(trained_pipeline, X_pair),
        )

    def test_from_pipeline_dir(
        self, trained_pipeline: Pipeline, saved_dir: Path, regression_dataset: Any
    ) -> None:
        X_pair, _, _ = regression_dataset
        served = ServedPipeline(pipeline_dir=saved_dir)
        preds = served.predict({"source": {"X": X_pair}})
        np.testing.assert_allclose(
            preds["pred"][0].to_numpy(),
            _expected_predictions(trained_pipeline, X_pair),
        )

    def test_needs_exactly_one_source(
        self, trained_pipeline: Pipeline, saved_dir: Path
    ) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            ServedPipeline()
        with pytest.raises(ValueError, match="exactly one"):
            ServedPipeline(
                pipeline_config=trained_pipeline.config, pipeline_dir=saved_dir
            )

    def test_json_inference_and_evaluation(
        self, trained_pipeline: Pipeline, saved_dir: Path, regression_dataset: Any
    ) -> None:
        (X_df, X_ctx), (y_df, y_ctx), _ = regression_dataset
        served = ServedPipeline(pipeline_dir=saved_dir)
        x_payload = PortPayload(data=_records(X_df), context=X_ctx.dump_dict)
        y_payload = PortPayload(data=_records(y_df), context=y_ctx.dump_dict)

        response = served.infer_json(
            InferRequest(input_data={"source": {"X": x_payload}})
        )
        preds = [row["target"] for row in response["pred"].data]
        np.testing.assert_allclose(
            preds,
            _expected_predictions(trained_pipeline, (X_df, X_ctx)).reshape(-1),
        )

        metrics = served.evaluate_json(
            EvaluateRequest(input_data={"source": {"X": x_payload, "y": y_payload}})
        )
        (score,) = metrics["metric"].data[0].values()
        assert score == pytest.approx(0.0, abs=1e-3)

    def test_bad_payload_is_http_422(self, saved_dir: Path) -> None:
        served = ServedPipeline(pipeline_dir=saved_dir)
        ctx = TabularDataContext(
            columns=["x0"], dtypes=[np.dtype("float64")], categories=[NumericalData]
        )
        payload = PortPayload(data=[{"other": 1.0}], context=ctx.dump_dict)
        with pytest.raises(HTTPException) as excinfo:
            served.infer_json(InferRequest(input_data={"source": {"X": payload}}))
        assert excinfo.value.status_code == 422

    def test_deployment_routes_delegate(self, saved_dir: Path) -> None:
        served = ServedPipeline(pipeline_dir=saved_dir)
        deployment_class = PipelineServing.func_or_class
        assert deployment_class.health(served)["status"] == "ok"
        assert "model" in deployment_class.info(served)["nodes"]


class _FakeDeployment:
    """Record the ``options`` and ``bind`` calls of a Ray Serve deployment."""

    def __init__(self) -> None:
        self.options_kwargs: dict[str, Any] = {}
        self.bind_kwargs: dict[str, Any] = {}

    def options(self, **kwargs: Any) -> _FakeDeployment:
        self.options_kwargs = kwargs
        return self

    def bind(self, **kwargs: Any) -> str:
        self.bind_kwargs = kwargs
        return "app"


class TestBuildPipelineApp:
    def test_applies_replicas_and_actor_options(
        self, monkeypatch: pytest.MonkeyPatch, trained_pipeline: Pipeline
    ) -> None:
        fake = _FakeDeployment()
        monkeypatch.setattr(ray_serve, "PipelineServing", fake)
        config = PipelineServingConfig(
            num_replicas=3, ray_actor_options={"num_cpus": 0.5}
        )
        app = build_pipeline_app(
            pipeline_config=trained_pipeline.config, params_dir="p", config=config
        )

        assert app == "app"
        assert fake.options_kwargs == {
            "num_replicas": 3,
            "ray_actor_options": {"num_cpus": 0.5},
        }
        assert fake.bind_kwargs["params_dir"] == "p"
        assert fake.bind_kwargs["config"] is config

    def test_omits_actor_options_by_default(
        self, monkeypatch: pytest.MonkeyPatch, saved_dir: Path
    ) -> None:
        fake = _FakeDeployment()
        monkeypatch.setattr(ray_serve, "PipelineServing", fake)
        build_pipeline_app(pipeline_dir=saved_dir)
        assert fake.options_kwargs == {"num_replicas": 1}
        assert fake.bind_kwargs["pipeline_dir"] == saved_dir

    def test_returns_a_serve_application(self, saved_dir: Path) -> None:
        app = build_pipeline_app(
            pipeline_dir=saved_dir, config=PipelineServingConfig(num_replicas=2)
        )
        assert isinstance(app, Application)

    def test_needs_exactly_one_source(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            build_pipeline_app()
