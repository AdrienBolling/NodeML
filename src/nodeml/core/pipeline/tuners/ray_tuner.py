"""Ray Pipeline Tuner module.

:class:`RayPipelineTuner` searches the hyperparameters of a pipeline with
Ray Tune.  Each trial rebuilds the pipeline from a config snapshot, applies
the sampled hyperparameters, trains, evaluates, and reports one objective
value with the name ``optimization_metric``.

Each trial also reports a checkpoint.  The checkpoint holds the config of
the trial pipeline and its trained parameters, so
``Pipeline.load_from_dir(checkpoint_dir)`` rebuilds the trained pipeline of
a trial.
"""

import dataclasses
import tempfile
from collections.abc import Callable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ValidationError
from ray import tune
from ray.tune import Result, ResultGrid, RunConfig, TuneConfig

from nodeml.core.common.data.data import ArrayLike, DataContext
from nodeml.core.common.exceptions import NodeConfigError
from nodeml.core.common.logging import Logger
from nodeml.core.common.typechecking.typeguards import (
    has_hyperparameter_space,
    has_hyperparameters_config,
)
from nodeml.core.pipeline.pipeline import Pipeline, PipelineConfig
from nodeml.core.pipeline.runners.smart_runner import (
    SmartRunner,
    SmartRunnerConfig,
)
from nodeml.core.pipeline.tuners.metrics import metrics_to_scalars

OPTIMIZATION_METRIC = "optimization_metric"
"""Name of the objective value that each trial reports to Ray Tune."""

type TuneMode = Literal["min", "max"]
type InputData = Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]]

_TUNE_MODES: tuple[str, ...] = ("min", "max")


class RayPipelineTunerConfig(BaseModel):
    """Configuration for the RayPipelineTuner."""

    runner_config: SmartRunnerConfig = SmartRunnerConfig()


class RayPipelineTuner:
    """Hyperparameter tuner backed by Ray Tune.

    Builds a trainable function from a pipeline configuration and delegates
    the search to Ray Tune.  Each trial converts its metrics to float values
    (see :func:`nodeml.core.pipeline.tuners.metrics.metrics_to_scalars`) and
    reports one objective value, ``optimization_metric``.
    """

    def __init__(self, pipeline: Pipeline, *, config: RayPipelineTunerConfig) -> None:
        """Initialize the RayPipelineTuner.

        Args:
            pipeline: A pipeline whose config will be used as the base for
                each trial.  The pipeline is compiled internally for
                validation.
            config: Tuner-level configuration.

        """
        # Snapshot the fully-instantiated user-facing config so each trial
        # can rebuild an identical pipeline from scratch.
        self._pipeline_config = pipeline.config
        self._config = config
        self._tuner: tune.Tuner | None = None
        self._results: ResultGrid | None = None
        self._mode: TuneMode | None = None

        # Dummy Pipeline, for validation and node objects instantiation.
        self._dummy_pipe = Pipeline(config=self._pipeline_config)
        self._dummy_pipe.compile()

        self._log = Logger(
            "nodeml.pipeline.tuner.ray",
            pipeline_name=pipeline.name,
            pipeline_version=pipeline.version,
        )
        tunable_nodes = [
            name
            for name, obj in self._dummy_pipe.node_objects.items()
            if has_hyperparameter_space(obj)
        ]
        self._log.info(
            "RayPipelineTuner initialized",
            num_nodes=len(self._dummy_pipe.node_objects),
            num_tunable_nodes=len(tunable_nodes),
            tunable_nodes=tunable_nodes,
        )

    # --- Public API ---

    @property
    def tuner(self) -> tune.Tuner | None:
        """Get the Ray Tune Tuner object after tuning has been run."""
        return self._tuner

    @property
    def results(self) -> ResultGrid | None:
        """Get the :class:`ResultGrid` produced by the most recent tuning run."""
        return self._results

    def default_hyperparameter_space(self) -> dict[str, Any]:
        """Build the default hyperparameter space from all tunable nodes.

        Keys follow the ``"node_name/hp_name"`` convention and values are
        Ray Tune search-space objects (e.g. ``tune.choice``,
        ``tune.uniform``).  The returned dict can be customised before
        passing it to :meth:`tune`.

        Returns:
            Dict of ``{node_name/hp_name: ray_tune_definition}``.

        """
        hp_space: dict[str, Any] = {}
        for node_name, node_obj in self._dummy_pipe.node_objects.items():
            if has_hyperparameter_space(node_obj):
                for hp_name, hp_value in node_obj.hyperparameter_space.items():
                    hp_space[f"{node_name}/{hp_name}"] = hp_value
        self._log.debug(
            "Default hyperparameter space built",
            num_hyperparameters=len(hp_space),
            hyperparameter_keys=sorted(hp_space),
        )
        return hp_space

    def tune(
        self,
        param_space: dict[str, Any],
        *,
        optimization_metric: str | None = None,
        metric_aggregator: Callable[[dict[str, float]], float] | None = None,
        input_data: InputData | None = None,
        tune_config: TuneConfig | None = None,
        run_config: RunConfig | None = None,
    ) -> None:
        """Run the hyperparameter tuning process via Ray Tune.

        The tuner does not change *tune_config*.  It gives Ray Tune a copy
        whose ``metric`` is ``"optimization_metric"``.

        Args:
            param_space: Search space dict (``node_name/hp_name`` keys).
            optimization_metric: Name of the single metric to optimise.
                Mutually exclusive with *metric_aggregator* (one must be
                provided).  A multi-value metric has one key per value, for
                example ``"mse.score_0"``.
            metric_aggregator: Callable that reduces the full metrics dict
                to a scalar.  Takes precedence over *optimization_metric*.
            input_data: Optional external data passed to
                :meth:`SmartRunner.train` and :meth:`SmartRunner.evaluate`
                in every trial.  Required for pipelines that use
                ``InputsPassthrough``-style source nodes.
            tune_config: Ray Tune :class:`TuneConfig`.  It must set ``mode``
                to ``"min"`` or ``"max"``: the direction of the objective.
            run_config: Ray Tune :class:`RunConfig`.  Defaults are used
                when ``None``.

        Raises:
            ValueError: If *tune_config* does not set ``mode``, or if neither
                *optimization_metric* nor *metric_aggregator* is provided.

        """
        if tune_config is None or tune_config.mode not in _TUNE_MODES:
            given = None if tune_config is None else tune_config.mode
            msg = (
                "Set the direction of the objective: pass "
                "tune_config=TuneConfig(mode='min') or TuneConfig(mode='max'). "
                f"The given mode is {given!r}."
            )
            raise ValueError(msg)
        if tune_config.metric not in (None, OPTIMIZATION_METRIC):
            self._log.warning(
                "TuneConfig.metric is ignored",
                given_metric=tune_config.metric,
                used_metric=OPTIMIZATION_METRIC,
            )
        mode: TuneMode = "min" if tune_config.mode == "min" else "max"
        # Ray Tune ranks trials by one scalar key.  Give Ray a copy, so that
        # the object of the caller does not change.
        trial_tune_config = dataclasses.replace(tune_config, metric=OPTIMIZATION_METRIC)
        if run_config is None:
            run_config = RunConfig()
        trainable = self._trainable(
            optimization_metric=optimization_metric,
            metric_aggregator=metric_aggregator,
        )

        # Ship ``input_data`` into Ray's object store exactly once and let
        # every trial pick up a reference, rather than capturing the
        # payload in the trainable's closure (which re-pickles the full
        # arrays per trial and inflates the driver → worker transfer).
        if input_data is not None:
            trainable = tune.with_parameters(trainable, input_data=input_data)

        tuner = tune.Tuner(
            trainable,
            param_space=param_space,
            tune_config=trial_tune_config,
            run_config=run_config,
        )
        self._tuner = tuner
        with self._log.timed(
            "Tuning",
            num_hyperparameters=len(param_space),
            num_samples=trial_tune_config.num_samples,
            mode=mode,
            optimization_metric=optimization_metric,
            metric_aggregator=(
                _callable_name(metric_aggregator)
                if metric_aggregator is not None
                else None
            ),
            has_input_data=input_data is not None,
        ) as scoped:
            results = tuner.fit()
            num_errored = len([r for r in results if r.error is not None])
            scoped.info(
                "Tuning results summary",
                num_trials=len(results),
                num_errored=num_errored,
            )
        self._results = results
        self._mode = mode

    def get_best(
        self, metric: str | None = None, mode: TuneMode | None = None
    ) -> Result:
        """Return the best result from the most recent tuning run.

        Args:
            metric: Metric name to rank by.  Defaults to
                ``"optimization_metric"``.
            mode: ``"max"`` or ``"min"``.  For ``"optimization_metric"``,
                the default is the mode of the most recent :meth:`tune`
                call.  For all other metrics, *mode* is required.

        Returns:
            The best :class:`ray.tune.Result` object.

        Raises:
            ValueError: If :meth:`tune` has not been called yet, or if
                *mode* is missing for a metric other than
                ``"optimization_metric"``.

        """
        if self._results is None or self._mode is None:
            msg = "Tuning has not been run yet. Please run the tune() method first."
            raise ValueError(msg)
        if metric is None:
            metric = OPTIMIZATION_METRIC
        if mode is None:
            if metric != OPTIMIZATION_METRIC:
                msg = (
                    f"Pass mode='min' or mode='max' to rank the trials by "
                    f"'{metric}'. The default mode applies only to "
                    f"'{OPTIMIZATION_METRIC}'."
                )
                raise ValueError(msg)
            mode = self._mode
        best = self._results.get_best_result(metric=metric, mode=mode)
        self._log.info(
            "Best trial selected",
            metric=metric,
            mode=mode,
            trial_config=best.config,
            best_metrics={
                k: v
                for k, v in (best.metrics or {}).items()
                if isinstance(v, (int, float))
            },
        )
        return best

    # --- Internal methods ---

    def _trainable(
        self,
        *,
        optimization_metric: str | None = None,
        metric_aggregator: Callable[[dict[str, float]], float] | None = None,
    ) -> Callable[..., None]:
        """Build the trainable function passed to Ray Tune.

        The returned trainable accepts an optional ``input_data`` kwarg,
        which :meth:`tune` supplies via ``ray.tune.with_parameters`` so the
        payload is placed in the Ray object store once and shared across
        trials by reference.  Closing over the data directly would force
        Ray to re-pickle the full arrays on every trial.

        The trainable does not refer to the tuner object.  So Ray does not
        pickle the dummy pipeline or the results of an earlier run.

        Args:
            optimization_metric: Single metric name to extract.
            metric_aggregator: Callable reducing all metrics to a scalar.

        Returns:
            A function ``(config, *, input_data=None) -> None`` compatible
            with :class:`ray.tune.Tuner`.

        """
        objective = _objective_function(optimization_metric, metric_aggregator)
        base_config = self._pipeline_config
        tuner_log = self._log
        # Per-trial runners should never render a progress bar — a tuning
        # job typically fires dozens of short pipeline runs in parallel,
        # and multiplexed tqdm bars turn the output into noise.
        trial_runner_config = self._config.runner_config.model_copy(
            update={"verbose": False}
        )

        def trainable(
            config: dict[str, Any],
            input_data: InputData | None = None,
        ) -> None:
            trial_context = tune.get_context()
            trial_id = (
                trial_context.get_trial_id() if trial_context is not None else None
            )
            trial_log = tuner_log.bind(trial_id=trial_id or "?")
            with trial_log.timed("Trial", trial_config=config):
                pipe = Pipeline(config=_apply_hyperparameters(base_config, config))
                pipe.compile()
                runner = SmartRunner(pipeline=pipe, config=trial_runner_config)

                # Resume from checkpoint when Ray provides one (useful for
                # multi-step trainables; here the trainable is single-shot so
                # this is mainly forward-looking).
                checkpoint = tune.get_checkpoint()
                if checkpoint is not None:
                    with checkpoint.as_directory() as checkpoint_dir:
                        runner.load_params_from_dir(checkpoint_dir)

                runner.train(input_data=input_data)
                metrics = metrics_to_scalars(runner.evaluate(input_data=input_data))
                metrics[OPTIMIZATION_METRIC] = float(objective(metrics))
                trial_log.info("Trial metrics", metrics=metrics)

                # The checkpoint holds the config and the trained params, so
                # that Pipeline.load_from_dir rebuilds the trial pipeline.
                with tempfile.TemporaryDirectory() as tmpdir:
                    pipe.save_config_to_dir(tmpdir)
                    runner.save_params_to_dir(tmpdir)
                    checkpoint = tune.Checkpoint.from_directory(tmpdir)
                    tune.report(metrics=metrics, checkpoint=checkpoint)

        return trainable

    def _convert_metrics(
        self,
        metrics: Mapping[str, tuple[ArrayLike, DataContext]],
    ) -> dict[str, float]:
        """Convert pipeline metric outputs to float values for Ray Tune.

        A metric with one value keeps its name.  A metric with one row of
        more than one value gives one key per column,
        ``f"{name}.{column}"``.  See
        :func:`nodeml.core.pipeline.tuners.metrics.metric_to_scalars`.

        Args:
            metrics: Mapping of metric names to ``(array, context)`` tuples.

        Returns:
            Dict mapping metric keys to float values.

        Raises:
            ValueError: If a metric has more than one row, or a value that
                is not a number.

        """
        return metrics_to_scalars(metrics)

    def _apply_hyperparameters(self, config: Mapping[str, Any]) -> PipelineConfig:
        """Apply flat Ray Tune hyperparameters onto a fresh pipeline config.

        Args:
            config: Flat dict with ``"node_name/hp_name"`` keys and sampled
                values.

        Returns:
            A new :class:`PipelineConfig` with the hyperparameters merged
            into the matching node configs.

        Raises:
            NodeConfigError: If a key is not ``"node_name/hp_name"``,
                references an unknown node or hyperparameter, or if a value
                is not valid for the hyperparameter.

        """
        pipe_conf = _apply_hyperparameters(self._pipeline_config, config)
        patched_nodes = sorted({key.partition("/")[0] for key in config})
        self._log.debug(
            "Applied hyperparameters",
            num_nodes_patched=len(patched_nodes),
            patched_nodes=patched_nodes,
        )
        return pipe_conf


def _objective_function(
    optimization_metric: str | None,
    metric_aggregator: Callable[[dict[str, float]], float] | None,
) -> Callable[[dict[str, float]], float]:
    """Return the function that reduces the trial metrics to the objective.

    Args:
        optimization_metric: Single metric name to extract.
        metric_aggregator: Callable reducing all metrics to a scalar.  It
            takes precedence over *optimization_metric*.

    Returns:
        A function ``(metrics) -> float``.

    Raises:
        ValueError: If both arguments are ``None``.

    """
    if metric_aggregator is not None:
        return metric_aggregator
    if optimization_metric is None:
        msg = (
            "Either optimization_metric or metric_aggregator must be provided. "
            "If both are provided, metric_aggregator takes precedence."
        )
        raise ValueError(msg)
    target_metric = optimization_metric

    def objective(metrics: dict[str, float]) -> float:
        if target_metric not in metrics:
            msg = (
                f"The optimization metric '{target_metric}' is not in the trial "
                f"metrics. Available metrics: {sorted(metrics)}."
            )
            raise KeyError(msg)
        return metrics[target_metric]

    return objective


def _apply_hyperparameters(
    base_config: PipelineConfig, config: Mapping[str, Any]
) -> PipelineConfig:
    """Return a copy of *base_config* with the sampled hyperparameters.

    Pydantic validates each changed hyperparameters model.  So a value of
    the wrong type, or out of its range, does not reach the node.

    Args:
        base_config: The pipeline config to start from.  It does not change.
        config: Flat dict with ``"node_name/hp_name"`` keys and sampled
            values.

    Returns:
        A new :class:`PipelineConfig`.

    Raises:
        NodeConfigError: If a key is not ``"node_name/hp_name"``, references
            an unknown node or hyperparameter, or if a value is not valid.

    """
    hp_by_node: dict[str, dict[str, Any]] = {}
    for key, value in config.items():
        node_name, separator, hp_name = key.partition("/")
        if not separator or not node_name or not hp_name:
            msg = f"Hyperparameter key '{key}' is not valid. Use 'node_name/hp_name'."
            raise NodeConfigError(msg)
        hp_by_node.setdefault(node_name, {})[hp_name] = value

    # Start from a deep copy so we never mutate the snapshot taken in
    # __init__; subsequent trials must see the same base config.
    pipe_conf = base_config.model_copy(deep=True)
    updated_nodes = dict(pipe_conf.nodes)
    for node_name, updates in hp_by_node.items():
        if node_name not in updated_nodes:
            msg = (
                f"Hyperparameter key references unknown node '{node_name}'. "
                f"Known nodes: {sorted(updated_nodes)}."
            )
            raise NodeConfigError(msg)
        node_type, node_conf = updated_nodes[node_name]
        if node_conf is None or not has_hyperparameters_config(node_conf):
            msg = (
                f"Node '{node_name}' has no hyperparameters; cannot apply "
                f"updates {sorted(updates)}."
            )
            raise NodeConfigError(msg)
        new_hp = _validated_hyperparameters(
            node_name, node_conf.hyperparameters, updates
        )
        new_conf = node_conf.model_copy(update={"hyperparameters": new_hp})
        updated_nodes[node_name] = (node_type, new_conf)
    return pipe_conf.model_copy(update={"nodes": updated_nodes})


def _validated_hyperparameters(
    node_name: str, hyperparameters: BaseModel, updates: Mapping[str, Any]
) -> BaseModel:
    """Return a validated copy of *hyperparameters* with *updates* applied.

    Args:
        node_name: The node name, for the error messages.
        hyperparameters: The current hyperparameters model of the node.
        updates: The sampled values, keyed by hyperparameter name.

    Returns:
        A new model of the same class.

    Raises:
        NodeConfigError: If a name is unknown or a value is not valid.

    """
    model_class = type(hyperparameters)
    known = sorted(model_class.model_fields)
    unknown = sorted(set(updates) - set(known))
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        msg = (
            f"Node '{node_name}': unknown hyperparameter {names}. "
            f"Known hyperparameters: {known}."
        )
        raise NodeConfigError(msg)
    try:
        return model_class.model_validate({**hyperparameters.model_dump(), **updates})
    except ValidationError as exc:
        msg = (
            f"Node '{node_name}': the sampled hyperparameters {dict(updates)} "
            f"are not valid.\n{exc}"
        )
        raise NodeConfigError(msg) from exc


def _callable_name(function: Callable[..., Any]) -> str:
    """Return a readable name for *function*, also for ``functools.partial``."""
    name = getattr(function, "__qualname__", None) or getattr(
        function, "__name__", None
    )
    return name if isinstance(name, str) else repr(function)
