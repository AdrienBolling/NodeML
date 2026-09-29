"""Ray Serve deployment module for NodeML Pipelines.

Exposes a trained pipeline as a Ray Serve deployment with two interfaces:

* A Ray-native interface: ``handle.predict.remote(input_data)`` and
  ``handle.evaluate.remote(input_data)``.  Ray sends the Python objects
  with pickle (protocol 5).  There is no JSON conversion.  Numpy buffers
  can go out-of-band, so a replica on the same node can read them without
  a copy.  Ray pickles pandas DataFrames, so it copies their data.
* An HTTP interface via FastAPI: ``POST /infer`` and ``POST /evaluate``
  with JSON payloads, plus ``GET /health`` and ``GET /info``.

Use :func:`build_pipeline_app` to create the application.  It applies
:attr:`PipelineServingConfig.num_replicas` and
:attr:`PipelineServingConfig.ray_actor_options` to the deployment::

    app = build_pipeline_app(
        pipeline_dir="/path/to/saved",
        config=PipelineServingConfig(num_replicas=2),
    )
    serve.run(app)

:class:`ServedPipeline` holds the pipeline logic without Ray, so you can
use it (and test it) in a normal Python process.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ray import serve
from ray.serve import Application

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

# HTTP status code for a request body that does not match the pipeline.
_HTTP_UNPROCESSABLE = 422

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class PipelineServingConfig(BaseModel):
    """Configuration for a pipeline serving deployment.

    Attributes:
        runner_config: Configuration of the :class:`SmartRunner` in each
            replica.
        num_replicas: Number of replicas of the deployment.
            :func:`build_pipeline_app` applies it.
        ray_actor_options: Ray actor options of each replica, for example
            ``{"num_cpus": 1, "num_gpus": 0.5}``.  :func:`build_pipeline_app`
            applies it.  ``None`` keeps the Ray Serve defaults.

    """

    runner_config: SmartRunnerConfig = SmartRunnerConfig(verbose=False)
    num_replicas: int = 1
    ray_actor_options: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Request / Response models (FastAPI + validation)
# ---------------------------------------------------------------------------


class PortPayload(BaseModel):
    """A single port's data in JSON form.

    Attributes:
        data: Row-oriented records (each dict is one row).  The keys of
            each record must be the context columns.
        context: Serialized :class:`TabularDataContext` with ``columns``,
            ``dtypes``, and ``categories`` string lists.  The DataFrame gets
            the column order and the dtypes of the context.

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
    """A single port's result in JSON form.

    Attributes:
        data: Row-oriented records.  Each value has its JSON-native type:
            integers stay integers and strings stay strings.  NaN, infinite
            and missing values become ``null``.
        context: Serialized :class:`TabularDataContext` of the result.

    """

    data: list[dict[str, Any]]
    context: dict[str, list[str]]


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _payload_to_tuple(
    payload: PortPayload,
) -> tuple[pd.DataFrame, TabularDataContext]:
    """Convert a JSON port payload to the ``(DataFrame, context)`` tuple.

    The DataFrame gets the column order and the dtypes of the context.

    Args:
        payload: The JSON payload of one port.

    Returns:
        The ``(DataFrame, context)`` tuple.

    Raises:
        ValueError: If the records and the context have different columns,
            or a value cannot be converted to the context dtype.

    """
    ctx = tabular_context_from_dict_dump(payload.context)
    frame = pd.DataFrame.from_records(payload.data)
    if payload.data:
        missing = [column for column in ctx.columns if column not in frame.columns]
        unknown = [column for column in frame.columns if column not in ctx.columns]
        if missing or unknown:
            message = (
                "The records do not match the context columns. "
                f"Missing columns: {missing}. Unknown columns: {unknown}."
            )
            raise ValueError(message)
    frame = frame.reindex(columns=ctx.columns)
    frame = frame.astype(dict(zip(ctx.columns, ctx.dtypes, strict=True)))
    return frame, ctx


def _to_frame(array: ArrayLike, ctx: DataContext) -> pd.DataFrame:
    """Convert an output array to a DataFrame with named columns."""
    if isinstance(array, pd.DataFrame):
        return array
    values = (
        array.detach().cpu().numpy()
        if isinstance(array, torch.Tensor)
        else np.asarray(array)
    )
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    num_columns = values.shape[1]
    if isinstance(ctx, TabularDataContext) and len(ctx.columns) == num_columns:
        columns = list(ctx.columns)
    else:
        columns = [f"col_{i}" for i in range(num_columns)]
    return pd.DataFrame(values, columns=columns)


def _json_value(value: Any) -> Any:
    """Convert one cell value to its JSON-native Python type.

    Numpy scalars become Python scalars.  NaN, infinite and missing values
    become ``None``.  Dates and times become ISO 8601 strings.

    Args:
        value: A cell value of a DataFrame.

    Returns:
        A value that :func:`json.dumps` accepts with ``allow_nan=False``.

    """
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, datetime | date | time):  # pd.Timestamp included
        return value.isoformat()
    if isinstance(value, np.datetime64 | np.timedelta64):
        return None if np.isnat(value) else str(value)
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _tuple_to_response(array: ArrayLike, ctx: DataContext) -> PortResponse:
    """Convert a ``(array, context)`` tuple to a JSON-serializable response.

    Args:
        array: The output data.
        ctx: The data context of *array*.

    Returns:
        The response.  Each value keeps its JSON-native type (see
        :func:`_json_value`).

    """
    frame = _to_frame(array, ctx)
    columns = [str(column) for column in frame.columns]
    records = [
        {column: _json_value(value) for column, value in zip(columns, row, strict=True)}
        for row in frame.itertuples(index=False, name=None)
    ]
    ctx_dump = ctx.dump_dict if isinstance(ctx, TabularDataContext) else {}
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


def _check_pipeline_source(
    pipeline_config: PipelineConfig | None, pipeline_dir: str | Path | None
) -> None:
    """Make sure that exactly one pipeline source is given.

    Raises:
        ValueError: If both or none of the arguments are given.

    """
    if (pipeline_config is None) == (pipeline_dir is None):
        message = "Pass exactly one of pipeline_config or pipeline_dir."
        raise ValueError(message)


# ---------------------------------------------------------------------------
# Pipeline logic (no Ray)
# ---------------------------------------------------------------------------


class ServedPipeline:
    """A trained pipeline, its runner and the JSON conversion.

    This class does not use Ray.  :class:`PipelineServing` adds the Ray
    Serve deployment and the HTTP routes.

    Give the pipeline in one of two forms:

    * *pipeline_config*, with an optional *params_dir* that holds the
      trained parameters (see :meth:`Pipeline.save_params_to_dir`).
    * *pipeline_dir*, a directory written by
      :meth:`Pipeline.save_config_to_dir` and
      :meth:`Pipeline.save_params_to_dir`.  :meth:`Pipeline.load_from_dir`
      loads it.  A *params_dir* replaces the parameters of *pipeline_dir*.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig | None = None,
        params_dir: str | Path | None = None,
        config: PipelineServingConfig | None = None,
        *,
        pipeline_dir: str | Path | None = None,
        file_basename: str | None = None,
    ) -> None:
        """Build and compile the pipeline, load its parameters and create the runner.

        Args:
            pipeline_config: The pipeline configuration.
            params_dir: Directory with the trained parameters.
            config: Serving configuration.  Defaults are used when ``None``.
            pipeline_dir: Directory with a saved pipeline (config and
                parameters).
            file_basename: The file base name of the saved pipeline in
                *pipeline_dir*.  Required only when the directory holds more
                than one pipeline.

        Raises:
            ValueError: If both or none of *pipeline_config* and
                *pipeline_dir* are given.

        """
        _check_pipeline_source(pipeline_config, pipeline_dir)
        serving_config = config or PipelineServingConfig()
        if pipeline_dir is not None:
            self._pipeline = Pipeline.load_from_dir(
                pipeline_dir, file_basename, load_params=params_dir is None
            )
            params_path = (
                Path(params_dir or pipeline_dir)
                / f"{self._pipeline.file_basename}_params.pkl"
            )
            params_loaded = params_path.is_file()
        else:
            self._pipeline = Pipeline(config=pipeline_config)
            self._pipeline.compile()
            params_loaded = params_dir is not None
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
            params_loaded=params_loaded,
        )

    @property
    def pipeline(self) -> Pipeline:
        """The compiled pipeline that this object serves."""
        return self._pipeline

    # ------------------------------------------------------------------
    # Ray-native interface (Python objects, pickled by Ray)
    # ------------------------------------------------------------------

    def predict(
        self,
        input_data: Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]],
    ) -> Mapping[str, tuple[ArrayLike, DataContext]]:
        """Run inference and return sink outputs.

        This is the Ray-native entry point: call it with
        ``handle.predict.remote(input_data)``.  Ray pickles the data.  There
        is no JSON conversion.

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
    # JSON interface (used by the HTTP routes)
    # ------------------------------------------------------------------

    def infer_json(self, request: InferRequest) -> dict[str, PortResponse]:
        """Run inference on a JSON request.

        Args:
            request: The request body.

        Returns:
            Sink output port names mapped to JSON responses.

        Raises:
            HTTPException: With status 422 if the payload does not match
                its context.

        """
        input_data = _request_input_data(request.input_data)
        return _serialize_outputs(self._runner.infer(input_data=input_data))

    def evaluate_json(self, request: EvaluateRequest) -> dict[str, PortResponse]:
        """Run evaluation on a JSON request.

        Args:
            request: The request body.

        Returns:
            Metric names mapped to JSON responses.

        Raises:
            HTTPException: With status 422 if the payload does not match
                its context.

        """
        input_data = _request_input_data(request.input_data)
        return _serialize_outputs(self._runner.evaluate(input_data=input_data))

    def health_info(self) -> dict[str, str]:
        """Return the health status and the pipeline identity."""
        return {
            "status": "ok",
            "pipeline": self._pipeline.name,
            "version": str(self._pipeline.version),
        }

    def pipeline_info(self) -> dict[str, Any]:
        """Return the pipeline metadata."""
        return {
            "name": self._pipeline.name,
            "version": str(self._pipeline.version),
            "nodes": list(self._pipeline.node_objects.keys()),
            "num_edges": len(self._pipeline.edges),
        }


def _request_input_data(
    raw: dict[str, dict[str, PortPayload]],
) -> dict[str, dict[str, tuple[pd.DataFrame, TabularDataContext]]]:
    """Deserialize a request payload, with an HTTP 422 error on bad input."""
    try:
        return _deserialize_input_data(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=_HTTP_UNPROCESSABLE, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------


_app = FastAPI()


@serve.deployment
@serve.ingress(_app)
class PipelineServing(ServedPipeline):
    """Ray Serve deployment wrapping a trained NodeML pipeline.

    Provides two interfaces:

    - **Ray-native**: call :meth:`predict` / :meth:`evaluate` via
      ``handle.predict.remote(input_data)``.  Ray pickles the data; there
      is no JSON conversion.
    - **HTTP** (FastAPI): ``POST /infer`` and ``POST /evaluate`` with JSON
      request bodies for external clients.

    The constructor arguments are the arguments of :class:`ServedPipeline`.
    Use :func:`build_pipeline_app` to create the application: it applies
    ``num_replicas`` and ``ray_actor_options`` of the
    :class:`PipelineServingConfig`.  ``PipelineServing.bind(...)`` uses the
    Ray Serve defaults instead.

    Example::

        app = build_pipeline_app(
            pipeline_config=pipe.config,
            params_dir="/path/to/saved",
        )
        serve.run(app)

    """

    @_app.post("/infer")
    def http_infer(self, request: InferRequest) -> dict[str, PortResponse]:
        """Run inference via HTTP.

        Accepts a JSON body matching :class:`InferRequest` and returns
        predictions as :class:`PortResponse` dicts.
        """
        return self.infer_json(request)

    @_app.post("/evaluate")
    def http_evaluate(self, request: EvaluateRequest) -> dict[str, PortResponse]:
        """Run evaluation via HTTP.

        Accepts a JSON body matching :class:`EvaluateRequest` and returns
        metric scores as :class:`PortResponse` dicts.
        """
        return self.evaluate_json(request)

    @_app.get("/health")
    def health(self) -> dict[str, str]:
        """Health check endpoint."""
        return self.health_info()

    @_app.get("/info")
    def info(self) -> dict[str, Any]:
        """Pipeline metadata endpoint."""
        return self.pipeline_info()


def build_pipeline_app(
    pipeline_config: PipelineConfig | None = None,
    params_dir: str | Path | None = None,
    config: PipelineServingConfig | None = None,
    *,
    pipeline_dir: str | Path | None = None,
    file_basename: str | None = None,
) -> Application:
    """Build the Ray Serve application of a pipeline.

    The deployment gets ``num_replicas`` and ``ray_actor_options`` from
    *config*.  Pass the result to ``serve.run``.  Each replica reads
    *params_dir* and *pipeline_dir*, so these paths must exist on the nodes
    of the replicas.

    Args:
        pipeline_config: The pipeline configuration.
        params_dir: Directory with the trained parameters.
        config: Serving configuration.  Defaults are used when ``None``.
        pipeline_dir: Directory with a saved pipeline (config and
            parameters).
        file_basename: The file base name of the saved pipeline in
            *pipeline_dir*.

    Returns:
        The bound application.

    Raises:
        ValueError: If both or none of *pipeline_config* and *pipeline_dir*
            are given.

    """
    _check_pipeline_source(pipeline_config, pipeline_dir)
    serving_config = config or PipelineServingConfig()
    options: dict[str, Any] = {"num_replicas": serving_config.num_replicas}
    if serving_config.ray_actor_options is not None:
        options["ray_actor_options"] = serving_config.ray_actor_options
    return PipelineServing.options(**options).bind(
        pipeline_config=pipeline_config,
        params_dir=params_dir,
        config=serving_config,
        pipeline_dir=pipeline_dir,
        file_basename=file_basename,
    )
