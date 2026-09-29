"""MLflow logging wrapper for any :class:`PipelineRunner`.

:class:`MLFlowLoggerWrapper` wraps the ``train`` / ``evaluate`` / ``infer``
calls of a runner and logs these items to MLflow:

* **Pipeline metadata** (tags): name, version and node names, plus the
  user tags.
* **Pipeline config**: the full ``PipelineConfig`` as JSON (every field of
  every node config), before training.
* **Pipeline graph**: an interactive HTML render of the DAG, before
  training.
* **Pipeline parameters**: the trained parameters of every node as a
  pickle file, after training.
* **Metrics**: the metric outputs and ``evaluate_duration_s`` of each
  ``evaluate()`` call.
* **Training time**: ``train_duration_s``, only if ``log_train_time`` is
  ``True`` (the default is ``False``).

The pipeline artifacts are in one fixed artifact directory,
:data:`PIPELINE_ARTIFACT_DIR` (``pipeline/``), with the file names of
:meth:`Pipeline.save_config_to_dir`, :meth:`Pipeline.save_params_to_dir`
and :meth:`Pipeline.save_html_to_dir`.  :func:`load_pipeline_from_run`
loads the trained pipeline of a run.

The wrapper uses its own :class:`mlflow.MlflowClient`.  It does not change
the global MLflow state: the tracking URI, the active experiment and the
active run stay as they are.
"""

import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import Metric, Run, RunStatus
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from nodeml.core.common.data.data import ArrayLike, DataContext
from nodeml.core.common.logging import Logger
from nodeml.core.pipeline.pipeline import Pipeline
from nodeml.core.pipeline.runners.pipeline_runner import PipelineRunner
from nodeml.core.pipeline.tuners.metrics import metric_to_scalars

PIPELINE_ARTIFACT_DIR = "pipeline"
"""Artifact directory of a run that holds the pipeline config, params and graph."""

type InputData = Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]]

_FINISHED = RunStatus.to_string(RunStatus.FINISHED)
_FAILED = RunStatus.to_string(RunStatus.FAILED)
_KILLED = RunStatus.to_string(RunStatus.KILLED)


class MLFlowLoggerWrapperConfig(BaseSettings):
    """Configuration for the :class:`MLFlowLoggerWrapper`.

    Fields prefixed with ``MLFLOW_`` can be loaded from environment
    variables (e.g. ``MLFLOW_TRACKING_URI``, ``MLFLOW_EXPERIMENT_NAME``).
    Explicit constructor values take precedence over env vars, which in
    turn take precedence over values from an env file.

    To load from a ``.env`` file, pass the path at construction::

        config = MLFlowLoggerWrapperConfig(env_file=".env")
        config = MLFlowLoggerWrapperConfig(env_file="/etc/mlflow/.env")

    The env file can also hold other ``MLFLOW_*`` variables (for example
    ``MLFLOW_TRACKING_USERNAME``).  This class ignores them, and it does
    not export them to MLflow.
    """

    # "ignore": an env file can hold MLflow variables that are not fields.
    # __init__ rejects unknown keyword arguments.
    model_config = SettingsConfigDict(
        env_prefix="MLFLOW_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    tracking_uri: str | None = Field(
        default=None,
        description=(
            "MLflow tracking server URI. "
            "``None`` uses the MLflow default (``mlflow.get_tracking_uri()``). "
            "Env: ``MLFLOW_TRACKING_URI``."
        ),
    )
    experiment_name: str = Field(
        default="nodeml",
        description="MLflow experiment name. Created automatically if it does not exist. Env: ``MLFLOW_EXPERIMENT_NAME``.",
    )
    run_name: str | None = Field(
        default=None,
        description="Optional human-readable name for the MLflow run. Env: ``MLFLOW_RUN_NAME``.",
    )
    nested: bool = Field(
        default=False,
        description=(
            "If ``True``, the run is created as a nested child of the "
            "currently active run (``mlflow.active_run()``), for example "
            "inside Ray Tune trials. Env: ``MLFLOW_NESTED``."
        ),
    )
    log_pipeline_params: bool = Field(
        default=True,
        description="Log the pipeline's trained parameters (learned weights) as a pickle artifact ``pipeline/{basename}_params.pkl`` after training. Env: ``MLFLOW_LOG_PIPELINE_PARAMS``.",
    )
    log_pipeline_config: bool = Field(
        default=True,
        description="Log the full ``PipelineConfig`` as a JSON artifact ``pipeline/{basename}_config.json``. Env: ``MLFLOW_LOG_PIPELINE_CONFIG``.",
    )
    log_pipeline_graph: bool = Field(
        default=True,
        description="Log an interactive HTML render of the pipeline DAG as an artifact ``pipeline/{basename}_graph.html``. Env: ``MLFLOW_LOG_PIPELINE_GRAPH``.",
    )
    log_train_time: bool = Field(
        default=False,
        description="Log ``train_duration_s`` as an MLflow metric after training. Env: ``MLFLOW_LOG_TRAIN_TIME``.",
    )
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="Extra user-defined tags added to every MLflow run.",
    )

    _env_file_path: str | None = None

    def __init__(self, *, env_file: str | None = None, **kwargs: Any) -> None:
        """Initialize the config, optionally loading from an env file.

        Args:
            env_file: Path to a ``.env`` file containing ``MLFLOW_*``
                variables.  ``None`` (the default) skips file loading.
            **kwargs: Field values.  Explicit values take precedence over
                env vars and the env file.

        Raises:
            TypeError: If a keyword argument is not a field name.

        """
        unknown = sorted(set(kwargs) - set(type(self).model_fields))
        if unknown:
            message = f"Unknown MLFlowLoggerWrapperConfig field(s): {unknown}."
            raise TypeError(message)
        super().__init__(_env_file=env_file, **kwargs)
        self._env_file_path = env_file

    def _field_source(self, field_name: str) -> str:
        """Return a human-readable label for where a field's value came from."""
        field = type(self).model_fields[field_name]
        if getattr(self, field_name) == field.get_default(call_default_factory=True):
            return "default"
        if self._env_file_path is not None and Path(self._env_file_path).is_file():
            return f"env_file ({self._env_file_path}) or env/constructor"
        return "env or constructor"

    def log_resolved_config(self, logger: Logger) -> None:
        """Log every field with its resolved value and source."""
        env_file_status: str
        if self._env_file_path is None:
            env_file_status = "not set"
        elif Path(self._env_file_path).is_file():
            env_file_status = f"loaded ({self._env_file_path})"
        else:
            env_file_status = f"not found ({self._env_file_path})"
        logger.info(
            "MLflow config resolved",
            env_file=env_file_status,
            **{
                field_name: f"{getattr(self, field_name)!r} ({self._field_source(field_name)})"
                for field_name in type(self).model_fields
            },
        )


class MLFlowLoggerWrapper:
    """Wrap a :class:`PipelineRunner` with MLflow lifecycle logging.

    Run lifecycle:

    * :meth:`train` creates a run.  After a successful training, the run
      is ``FINISHED``.  If training raises, the run is ``FAILED``
      (``KILLED`` on ``KeyboardInterrupt``) and the error is raised again.
    * A failure to log an artifact or a metric does not change the result
      of training or evaluation.  The wrapper logs a warning instead.
    * :meth:`evaluate` logs into the run of the last successful
      :meth:`train` and does not change its status.  If there is no such
      run (no training, or a failed training), :meth:`evaluate` creates a
      new run.  Each :meth:`evaluate` call in a run has its own ``step``:
      0, 1, 2, and so on.

    Usage::

        runner = SmartRunner(pipeline, config=SmartRunnerConfig())
        mlflow_runner = MLFlowLoggerWrapper(
            runner,
            config=MLFlowLoggerWrapperConfig(experiment_name="my_experiment"),
        )
        mlflow_runner.train()
        mlflow_runner.evaluate(val_data, prefix="val")
        mlflow_runner.evaluate(test_data, prefix="test")
        pipeline = load_pipeline_from_run(
            mlflow_runner.run_id, tracking_uri=mlflow_runner.client.tracking_uri
        )
    """

    def __init__(
        self,
        pipeline_runner: PipelineRunner,
        *,
        config: MLFlowLoggerWrapperConfig | None = None,
    ) -> None:
        """Initialize the wrapper and resolve the MLflow experiment.

        The experiment is created if it does not exist.

        Args:
            pipeline_runner: The runner instance to wrap.
            config: MLflow logging configuration.  Uses defaults when
                ``None``.

        Raises:
            ValueError: If the experiment exists but is deleted.

        """
        self._runner = pipeline_runner
        self._config = config or MLFlowLoggerWrapperConfig()
        self._run_id: str | None = None
        # True when _run_id is the run of a failed training.
        self._run_failed = False
        # Step of the next evaluate() call in the current run.
        self._eval_step = 0
        self._log = Logger(
            "nodeml.mlflow",
            pipeline_name=pipeline_runner.pipeline.name,
            pipeline_version=pipeline_runner.pipeline.version,
        )

        self._config.log_resolved_config(self._log)

        self._client = MlflowClient(tracking_uri=self._config.tracking_uri)
        self._experiment_id = self._resolve_experiment_id()
        self._log.info(
            "MLflow experiment resolved",
            tracking_uri=self._client.tracking_uri,
            experiment_name=self._config.experiment_name,
            experiment_id=self._experiment_id,
        )

    # --- Public properties ------------------------------------------------

    @property
    def runner(self) -> PipelineRunner:
        """The underlying pipeline runner."""
        return self._runner

    @property
    def client(self) -> MlflowClient:
        """The MLflow client of this wrapper (bound to its tracking URI)."""
        return self._client

    @property
    def experiment_id(self) -> str:
        """ID of the MLflow experiment of this wrapper."""
        return self._experiment_id

    @property
    def run_id(self) -> str | None:
        """MLflow run ID of the most recent run (``None`` before the first run)."""
        return self._run_id

    # --- Lifecycle methods ------------------------------------------------

    def train(self, input_data: InputData | None = None) -> None:
        """Create an MLflow run, log the pipeline, train, and end the run.

        Args:
            input_data: Optional external data forwarded to the runner.

        Raises:
            MlflowException: If MLflow cannot create the run.  Training
                does not start.
            BaseException: Any error of the runner's ``train``, after the
                run is marked ``FAILED`` (or ``KILLED``).

        """
        run_id = self._start_run()
        pipeline = self._runner.pipeline
        if self._config.log_pipeline_config:
            self._log_pipeline_artifact(run_id, "config", pipeline.save_config_to_dir)
        if self._config.log_pipeline_graph:
            self._log_pipeline_artifact(run_id, "graph", pipeline.save_html_to_dir)

        t0 = time.perf_counter()
        try:
            self._runner.train(input_data=input_data)
        except BaseException as exc:
            self._run_failed = True
            status = _KILLED if isinstance(exc, KeyboardInterrupt) else _FAILED
            self._log.error("Training failed", run_id=run_id, run_status=status)
            self._end_run(run_id, status)
            raise
        duration_s = time.perf_counter() - t0
        self._log.info("Training complete", train_duration_s=round(duration_s, 3))

        if self._config.log_train_time:
            self._log_scalars(run_id, {"train_duration_s": duration_s}, step=0)
        if self._config.log_pipeline_params:
            self._log_pipeline_artifact(
                run_id, "params", self._runner.save_params_to_dir
            )
        self._end_run(run_id, _FINISHED)

    def evaluate(
        self,
        input_data: InputData | None = None,
        *,
        prefix: str | None = None,
    ) -> Mapping[str, tuple[ArrayLike, DataContext]]:
        """Evaluate the pipeline and log all resulting metrics.

        The metrics go to the run of the last successful :meth:`train`,
        whose status does not change.  Otherwise a new run is created.
        Each call logs its metrics with the next ``step`` of the run.

        A metric with one value is logged as ``name``.  A metric with one
        row of values is logged as one ``name.column`` key per value (see
        :func:`nodeml.core.pipeline.tuners.metrics.metric_to_scalars`).

        Args:
            input_data: Optional external data forwarded to the runner.
            prefix: Optional key prefix, for example ``"val"`` or
                ``"test"``.  The keys become ``f"{prefix}/{name}"``,
                ``evaluate_duration_s`` included.

        Returns:
            The raw metrics mapping returned by the runner.

        Raises:
            BaseException: Any error of the runner's ``evaluate``.  A run
                that this call created is marked ``FAILED`` first.

        """
        if self._run_id is None or self._run_failed:
            new_run = True
            run_id = self._start_run()
        else:
            new_run = False
            run_id = self._run_id
        step = self._eval_step
        self._eval_step += 1
        self._log.info("Evaluation started", run_id=run_id, step=step)

        t0 = time.perf_counter()
        try:
            metrics = self._runner.evaluate(input_data=input_data)
        except BaseException as exc:
            if new_run:
                status = _KILLED if isinstance(exc, KeyboardInterrupt) else _FAILED
                self._run_failed = True
                self._end_run(run_id, status)
            raise
        duration_s = time.perf_counter() - t0

        scalars = {"evaluate_duration_s": duration_s}
        scalars.update(self._metric_scalars(metrics))
        if prefix:
            scalars = {f"{prefix}/{key}": value for key, value in scalars.items()}
        self._log_scalars(run_id, scalars, step=step)
        self._log.info(
            "Evaluation complete",
            evaluate_duration_s=round(duration_s, 3),
            num_metrics=len(metrics),
            step=step,
        )
        if new_run:
            self._end_run(run_id, _FINISHED)
        return metrics

    def infer(
        self,
        input_data: InputData | None = None,
    ) -> Mapping[str, tuple[ArrayLike, DataContext]]:
        """Run inference via the underlying runner (no MLflow logging).

        Args:
            input_data: Optional external data forwarded to the runner.

        Returns:
            Sink-node outputs from the runner.

        """
        return self._runner.infer(input_data=input_data)

    # --- Delegation shortcuts ---------------------------------------------

    def get_params(self) -> dict[str, dict[str, Any]]:
        """Delegate to the underlying runner."""
        return self._runner.get_params()

    def set_params(self, params: dict[str, dict[str, Any]]) -> None:
        """Delegate to the underlying runner."""
        self._runner.set_params(params=params)

    def save_params_to_dir(self, dir_path: str | Path) -> None:
        """Delegate to the underlying runner."""
        self._runner.save_params_to_dir(dir_path=dir_path)

    def load_params_from_dir(self, dir_path: str | Path) -> None:
        """Delegate to the underlying runner."""
        self._runner.load_params_from_dir(dir_path=dir_path)

    # --- Internal helpers -------------------------------------------------

    def _resolve_experiment_id(self) -> str:
        """Return the ID of the configured experiment; create it if needed.

        Raises:
            ValueError: If the experiment exists but is deleted.

        """
        name = self._config.experiment_name
        experiment = self._client.get_experiment_by_name(name)
        if experiment is None:
            try:
                return self._client.create_experiment(name)
            except MlflowException:
                # Another process can create the experiment at the same time.
                experiment = self._client.get_experiment_by_name(name)
                if experiment is None:
                    raise
        if experiment.lifecycle_stage == "deleted":
            message = (
                f"The MLflow experiment '{name}' is deleted. Restore it, or "
                "set another experiment_name."
            )
            raise ValueError(message)
        return experiment.experiment_id

    def _start_run(self) -> str:
        """Create a new run with the pipeline tags, and make it current.

        Returns:
            The ID of the new run.

        """
        pipeline = self._runner.pipeline
        tags: dict[str, str] = {
            "pipeline_name": pipeline.name,
            "pipeline_version": str(pipeline.version),
            "pipeline_nodes": ", ".join(pipeline.node_objects.keys()),
            **self._config.tags,
        }
        if self._config.nested:
            parent = mlflow.active_run()
            if parent is not None:
                tags[MLFLOW_PARENT_RUN_ID] = parent.info.run_id
        run: Run = self._client.create_run(
            self._experiment_id,
            run_name=self._config.run_name,
            tags=context_registry.resolve_tags(tags),
        )
        self._run_id = run.info.run_id
        self._run_failed = False
        self._eval_step = 0
        self._log.info("MLflow run started", run_id=self._run_id)
        return self._run_id

    def _end_run(self, run_id: str, status: str) -> None:
        """Set the final status of a run; log a warning on failure."""
        try:
            self._client.set_terminated(run_id, status=status)
        except Exception as exc:  # noqa: BLE001 - logging must not change the result
            self._log.warning(
                "MLflow run status not set",
                run_id=run_id,
                status=status,
                error=repr(exc),
            )

    def _log_pipeline_artifact(
        self, run_id: str, kind: str, save_to_dir: Callable[[Path], None]
    ) -> None:
        """Save one pipeline file and log it under :data:`PIPELINE_ARTIFACT_DIR`.

        A failure logs a warning and does not stop the caller.

        Args:
            run_id: The run that gets the artifact.
            kind: Short name of the artifact, for the log messages.
            save_to_dir: Function that writes the file into a directory.

        """
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                save_to_dir(Path(tmpdir))
                self._client.log_artifacts(
                    run_id, tmpdir, artifact_path=PIPELINE_ARTIFACT_DIR
                )
        except Exception as exc:  # noqa: BLE001 - logging must not change the result
            self._log.warning(
                "Pipeline artifact not logged", artifact=kind, error=repr(exc)
            )
            return
        self._log.info("Pipeline artifact logged", artifact=kind)

    def _metric_scalars(
        self, metrics: Mapping[str, tuple[ArrayLike, DataContext]]
    ) -> dict[str, float]:
        """Convert runner metrics to float values; skip the invalid ones.

        Args:
            metrics: Dict mapping metric names to ``(array, context)`` tuples
                returned by the runner's ``evaluate``.

        Returns:
            Dict mapping metric keys to float values.

        """
        scalars: dict[str, float] = {}
        for name, (array, context) in metrics.items():
            try:
                scalars.update(metric_to_scalars(name, array, context))
            except ValueError as exc:
                self._log.warning("Metric not logged", metric=name, error=str(exc))
        return scalars

    def _log_scalars(self, run_id: str, scalars: dict[str, float], step: int) -> None:
        """Log float values as MLflow metrics; log a warning on failure."""
        if not scalars:
            return
        timestamp = int(time.time() * 1000)
        entries = [
            Metric(key=key, value=value, timestamp=timestamp, step=step)
            for key, value in scalars.items()
        ]
        try:
            self._client.log_batch(run_id, metrics=entries)
        except Exception as exc:  # noqa: BLE001 - logging must not change the result
            self._log.warning(
                "MLflow metrics not logged", keys=sorted(scalars), error=repr(exc)
            )


def load_pipeline_from_run(
    run_id: str,
    *,
    tracking_uri: str | None = None,
    load_params: bool = True,
) -> Pipeline:
    """Load the pipeline that :class:`MLFlowLoggerWrapper` logged in a run.

    The function downloads the :data:`PIPELINE_ARTIFACT_DIR` artifacts of
    the run and calls :meth:`Pipeline.load_from_dir`.  Only load runs that
    you trust: the parameters are a pickle file.

    Args:
        run_id: The MLflow run ID.
        tracking_uri: The tracking URI.  ``None`` uses the MLflow default.
        load_params: Load the trained parameters, if the run has them.

    Returns:
        A compiled :class:`Pipeline`.

    Raises:
        PipelineError: If the run has no pipeline config artifact.

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        local_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=PIPELINE_ARTIFACT_DIR,
            dst_path=tmpdir,
            tracking_uri=tracking_uri,
        )
        return Pipeline.load_from_dir(local_dir, load_params=load_params)
