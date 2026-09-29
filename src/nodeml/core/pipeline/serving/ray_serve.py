"""Ray Serve deployment module for NodeML Pipelines.

Exposes a trained pipeline as a Ray Serve deployment with both a
Ray-native ``predict`` method (zero-copy Arrow) and an HTTP endpoint
via FastAPI (JSON payloads).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from ray import serve

from nodeml.core.common.data.data import (
    ArrayLike,
    DataContext,
    TabularDataContext,
    tabular_context_from_dict_dump,
)
from nodeml.core.common.logging import Logger
from nodeml.core.pipeline.pipeline import Pipeline, PipelineConfig
from nodeml.core.pipeline.runners.smart_runner import (
    SmartRunner,
    SmartRunnerConfig,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class PipelineServingConfig(BaseModel):
    """Configuration for a pipeline serving deployment."""

    runner_config: SmartRunnerConfig = SmartRunnerConfig(verbose=False)
    num_replicas: int = 1
    ray_actor_options: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Request / Response models (FastAPI + validation)
# ---------------------------------------------------------------------------


class PortPayload(BaseModel):
    """A single port's data in JSON form.

    Attributes:
        data: Row-oriented records (each dict is one row).
        context: Serialized :class:`TabularDataContext` with ``columns``,
            ``dtypes``, and ``categories`` string lists.

    """

    data: list[dict[str, Any]]
    context: dict[str, list[str]]


class InferRequest(BaseModel):
    """Inference request body.

    Attributes:
        input_data: ``{node_name: {port_name: PortPayload}}``.

    """

    input_data: dict[str, dict[str, PortPayload]]


class EvaluateRequest(BaseModel):
    """Evaluation request body (same shape as inference)."""

    input_data: dict[str, dict[str, PortPayload]]


class PortResponse(BaseModel):
    """A single port's result in JSON form."""

    data: list[dict[str, Any]]
    context: dict[str, list[str]]


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _payload_to_tuple(
    payload: PortPayload,
) -> tuple[pd.DataFrame, TabularDataContext]:
    """Convert a JSON port payload to the ``(DataFrame, context)`` tuple."""
    df = pd.DataFrame.from_records(payload.data)
    ctx = tabular_context_from_dict_dump(payload.context)
    return df, ctx


def _tuple_to_response(array: ArrayLike, ctx: DataContext) -> PortResponse:
    """Convert a ``(array, context)`` tuple to a JSON-serializable response."""
    to_numpy = getattr(array, "to_numpy", None)
    np_arr = to_numpy() if callable(to_numpy) else np.asarray(array)
    if hasattr(array, "columns"):
        columns = list(array.columns)
    elif hasattr(ctx, "columns"):
        columns = ctx.columns
    else:
        columns = [f"col_{i}" for i in range(np_arr.shape[1] if np_arr.ndim > 1 else 1)]

    if np_arr.ndim == 1:
        np_arr = np_arr.reshape(-1, 1)
    records = [
        {col: float(val) for col, val in zip(columns, row, strict=True)}
        for row in np_arr
    ]
    ctx_dump = ctx.dump_dict if hasattr(ctx, "dump_dict") else {}
    return PortResponse(data=records, context=ctx_dump)


def _deserialize_input_data(
    raw: dict[str, dict[str, PortPayload]],
) -> dict[str, dict[str, tuple[pd.DataFrame, TabularDataContext]]]:
    """Convert nested request payload to the runner's ``input_data`` format."""
    return {
        node_name: {
            port_name: _payload_to_tuple(port_payload)
            for port_name, port_payload in ports.items()
        }
        for node_name, ports in raw.items()
    }


def _serialize_outputs(
    outputs: Mapping[str, tuple[ArrayLike, DataContext]],
) -> dict[str, PortResponse]:
    """Convert runner outputs to JSON-serializable responses."""
    return {
        name: _tuple_to_response(array, ctx) for name, (array, ctx) in outputs.items()
    }


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------


_app = FastAPI()


@serve.deployment
@serve.ingress(_app)
class PipelineServing:
    """Ray Serve deployment wrapping a trained NodeML pipeline.

    Provides two interfaces:

    - **Ray-native**: call :meth:`predict` / :meth:`evaluate` via
      ``handle.predict.remote(input_data)`` for zero-copy data transfer.
    - **HTTP** (FastAPI): ``POST /infer`` and ``POST /evaluate`` with JSON
      request bodies for external clients.

    Example::

        serving = PipelineServing.bind(
            pipeline_config=pipe.config,
            params_dir="/path/to/saved",
        )
        serve.run(serving)

    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        params_dir: str | None = None,
        config: PipelineServingConfig | None = None,
    ) -> None:
        """Build and compile the pipeline, load its parameters and create the runner."""
        serving_config = config or PipelineServingConfig()
        self._pipeline = Pipeline(config=pipeline_config)
        self._pipeline.compile()
        if params_dir is not None:
            self._pipeline.load_params_from_dir(params_dir)
        self._runner = SmartRunner(self._pipeline, config=serving_config.runner_config)
        self._log = Logger(
            "nodeml.serving",
            pipeline_name=self._pipeline.name,
            pipeline_version=self._pipeline.version,
        )
        self._log.info(
            "Deployment ready",
            num_nodes=len(self._pipeline.node_objects),
            params_loaded=params_dir is not None,
        )

    # ------------------------------------------------------------------
    # Ray-native interface (zero-copy Arrow via handle.remote())
    # ------------------------------------------------------------------

    def predict(
        self,
        input_data: Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]],
    ) -> Mapping[str, tuple[ArrayLike, DataContext]]:
        """Run inference and return sink outputs.

        This is the Ray-native entry point — call via
        ``handle.predict.remote(input_data)``.  Data stays in Arrow
        format through the object store with no JSON round-trip.

        Args:
            input_data: ``{node_name: {port_name: (array, context)}}``.

        Returns:
            Sink output port names mapped to ``(array, context)`` tuples.

        """
        return self._runner.infer(input_data=input_data)

    def evaluate(
        self,
        input_data: Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]],
    ) -> Mapping[str, tuple[ArrayLike, DataContext]]:
        """Run evaluation and return metric outputs.

        Args:
            input_data: ``{node_name: {port_name: (array, context)}}``.

        Returns:
            Metric names mapped to ``(array, context)`` tuples.

        """
        return self._runner.evaluate(input_data=input_data)

    # ------------------------------------------------------------------
    # HTTP interface (FastAPI — JSON payloads)
    # ------------------------------------------------------------------

    @_app.post("/infer")
    def http_infer(self, request: InferRequest) -> dict[str, PortResponse]:
        """Run inference via HTTP.

        Accepts a JSON body matching :class:`InferRequest` and returns
        predictions as :class:`PortResponse` dicts.
        """
        input_data = _deserialize_input_data(request.input_data)
        outputs = self._runner.infer(input_data=input_data)
        return _serialize_outputs(outputs)

    @_app.post("/evaluate")
    def http_evaluate(self, request: EvaluateRequest) -> dict[str, PortResponse]:
        """Run evaluation via HTTP.

        Accepts a JSON body matching :class:`EvaluateRequest` and returns
        metric scores as :class:`PortResponse` dicts.
        """
        input_data = _deserialize_input_data(request.input_data)
        outputs = self._runner.evaluate(input_data=input_data)
        return _serialize_outputs(outputs)

    @_app.get("/health")
    def health(self) -> dict[str, str]:
        """Health check endpoint."""
        return {
            "status": "ok",
            "pipeline": self._pipeline.name,
            "version": str(self._pipeline.version),
        }

    @_app.get("/info")
    def info(self) -> dict[str, Any]:
        """Pipeline metadata endpoint."""
        return {
            "name": self._pipeline.name,
            "version": str(self._pipeline.version),
            "nodes": list(self._pipeline.node_objects.keys()),
            "num_edges": len(self._pipeline.edges),
        }
