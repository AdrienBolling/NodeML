"""Ray Pipeline Tuner module."""

import tempfile
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from pydantic import BaseModel
from ray import tune
from ray.tune import Result, ResultGrid, RunConfig, TuneConfig

from nodeml.core.common.data.data import ArrayLike, DataContext
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


class RayPipelineTunerConfig(BaseModel):
    """Configuration for the RayPipelineTuner."""

    runner_config: SmartRunnerConfig = SmartRunnerConfig()


class RayPipelineTuner:
    """Hyperparameter tuner backed by Ray Tune.

    Builds a trainable function from a pipeline configuration and delegates
    the search to Ray Tune, converting metrics to a scalar objective.
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
        input_data: Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]]
        | None = None,
        tune_config: TuneConfig | None = None,
        run_config: RunConfig | None = None,
    ) -> None:
        """Run the hyperparameter tuning process via Ray Tune.

        Args:
            param_space: Search space dict (``node_name/hp_name`` keys).
            optimization_metric: Name of the single metric to optimise.
                Mutually exclusive with *metric_aggregator* (one must be
                provided).
            metric_aggregator: Callable that reduces the full metrics dict
                to a scalar.  Takes precedence over *optimization_metric*.
            input_data: Optional external data passed to
                :meth:`SmartRunner.train` and :meth:`SmartRunner.evaluate`
                in every trial.  Required for pipelines that use
                ``InputsPassthrough``-style source nodes.
            tune_config: Ray Tune :class:`TuneConfig`.  Defaults are used
                when ``None``.
            run_config: Ray Tune :class:`RunConfig`.  Defaults are used
                when ``None``.

        Raises:
            ValueError: If neither *optimization_metric* nor
                *metric_aggregator* is provided.

        """
        if tune_config is None:
            tune_config = TuneConfig()
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

        # Ray Tune needs a single scalar key to rank trials against.
        tune_config.metric = "optimization_metric"

        tuner = tune.Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config,
        )
        self._tuner = tuner
        with self._log.timed(
            "Tuning",
            num_hyperparameters=len(param_space),
            num_samples=tune_config.num_samples,
            mode=tune_config.mode,
            optimization_metric=optimization_metric,
            metric_aggregator=(
                metric_aggregator.__name__ if metric_aggregator is not None else None
            ),
            has_input_data=input_data is not None,
        ) as scoped:
            self._results = tuner.fit()
            num_errored = len([r for r in self._results if r.error is not None])
            scoped.info(
                "Tuning results summary",
                num_trials=len(self._results),
                num_errored=num_errored,
            )

    def get_best(self, metric: str | None = None, mode: str = "max") -> Result:
        """Return the best result from the most recent tuning run.

        Args:
            metric: Metric name to rank by.  Defaults to
                ``"optimization_metric"``.
            mode: ``"max"`` or ``"min"``.

        Returns:
            The best :class:`ray.tune.Result` object.

        Raises:
            ValueError: If :meth:`tune` has not been called yet.

        """
        if self._results is None:
            msg = "Tuning has not been run yet. Please run the tune() method first."
            raise ValueError(msg)
        best = self._results.get_best_result(metric=metric, mode=mode)
        self._log.info(
            "Best trial selected",
            metric=metric or "optimization_metric",
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

        Args:
            optimization_metric: Single metric name to extract.
            metric_aggregator: Callable reducing all metrics to a scalar.

        Returns:
            A function ``(config, *, input_data=None) -> None`` compatible
            with :class:`ray.tune.Tuner`.

        """
        if metric_aggregator is not None:
            scalar_fn = metric_aggregator
        elif optimization_metric is not None:
            target_metric = optimization_metric

            def scalar_fn(metrics: dict[str, float]) -> float:
                return metrics[target_metric]
        else:
            msg = (
                "Either optimization_metric or metric_aggregator must be provided. "
                "If both are provided, metric_aggregator takes precedence."
            )
            raise ValueError(msg)

        tuner_log = self._log
        # Per-trial runners should never render a progress bar — a tuning
        # job typically fires dozens of short pipeline runs in parallel,
        # and multiplexed tqdm bars turn the output into noise.
        trial_runner_config = self._config.runner_config.model_copy(
            update={"verbose": False}
        )

        def trainable(
            config: dict,
            input_data: Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]]
            | None = None,
        ) -> None:
            trial_context = tune.get_context()
            trial_id = (
                trial_context.get_trial_id() if trial_context is not None else "?"
            )
            trial_log = tuner_log.bind(trial_id=trial_id)
            with trial_log.timed("Trial", trial_config=config):
                pipe_conf = self._apply_hyperparameters(config)
                pipe = Pipeline(config=pipe_conf)
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
                raw_metrics = runner.evaluate(input_data=input_data)
                metrics = self._convert_metrics(raw_metrics)
                metrics["optimization_metric"] = scalar_fn(metrics)
                trial_log.info("Trial metrics", metrics=metrics)

                with tempfile.TemporaryDirectory() as tmpdir:
                    runner.save_params_to_dir(tmpdir)
                    checkpoint = tune.Checkpoint.from_directory(tmpdir)
                    tune.report(metrics=metrics, checkpoint=checkpoint)

        return trainable

    def _convert_metrics(
        self,
        metrics: Mapping[str, tuple[ArrayLike, DataContext]],
    ) -> dict[str, float]:
        """Convert pipeline metric outputs to scalar floats for Ray Tune.

        The SmartRunner returns each metric as a ``(array, context)`` tuple
        where ``array`` is a (1, 1)-shaped pandas DataFrame (see
        :meth:`SmartRunner._convert_data_to_tuple`).  Ray Tune only accepts
        plain Python numbers, so we extract the single scalar value.

        Args:
            metrics: Mapping of metric names to ``(array, context)`` tuples.

        Returns:
            Dict mapping metric names to float values.

        """
        converted: dict[str, float] = {}
        for metric_name, (array, _ctx) in metrics.items():
            # Be tolerant of either pandas or numpy/torch backends — the
            # runner is free to change which one it returns.
            to_numpy = getattr(array, "to_numpy", None)
            np_array = to_numpy() if callable(to_numpy) else np.asarray(array)
            converted[metric_name] = float(np_array.reshape(-1)[0])
        return converted

    def _apply_hyperparameters(self, config: dict) -> PipelineConfig:
        """Apply flat Ray Tune hyperparameters onto a fresh pipeline config.

        Args:
            config: Flat dict with ``"node_name/hp_name"`` keys and sampled
                values.

        Returns:
            A new :class:`PipelineConfig` with the hyperparameters merged
            into the matching node configs.

        Raises:
            ValueError: If a key references an unknown node or the node's
                config has no hyperparameters field.

        """
        hp_by_node: dict[str, dict[str, Any]] = {}
        for key, value in config.items():
            node_name, hp_name = key.split("/", 1)
            hp_by_node.setdefault(node_name, {})[hp_name] = value

        # Start from a deep copy so we never mutate the snapshot taken in
        # __init__; subsequent trials must see the same base config.
        pipe_conf = self._pipeline_config.model_copy(deep=True)
        updated_nodes = dict(pipe_conf.nodes)
        for node_name, updates in hp_by_node.items():
            if node_name not in updated_nodes:
                msg = (
                    f"Hyperparameter key references unknown node '{node_name}'. "
                    f"Known nodes: {sorted(updated_nodes)}."
                )
                raise ValueError(msg)
            node_type, node_conf = updated_nodes[node_name]
            if node_conf is None or not has_hyperparameters_config(node_conf):
                msg = (
                    f"Node '{node_name}' has no hyperparameters; cannot apply "
                    f"updates {sorted(updates)}."
                )
                raise ValueError(msg)
            new_hp = node_conf.hyperparameters.model_copy(update=updates)
            new_conf = node_conf.model_copy(update={"hyperparameters": new_hp})
            updated_nodes[node_name] = (node_type, new_conf)
        self._log.debug(
            "Applied hyperparameters",
            num_nodes_patched=len(hp_by_node),
            patched_nodes=sorted(hp_by_node),
        )
        return pipe_conf.model_copy(update={"nodes": updated_nodes})
