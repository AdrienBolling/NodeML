"""The CounterfactualEvaluator: counterfactual explanations for a trained pipeline.

The evaluator follows the pattern of the Ray tuner.  It keeps a copy of the
pipeline config and of the trained parameters.  For each evaluation it:

1. Rebuilds the pipeline and inserts a PerturbationNode at the insertion
   point (by default right after the data source).
2. Runs the pipeline once on reference data, and reads the rows that reach
   the PerturbationNode and the predictions.  CELIA uses these rows as the
   data distribution.
3. Runs one task for each (scenario, sample) pair, with Ray in parallel or
   in this process.  Each task rebuilds the pipeline with an injecting
   PerturbationNode, wraps it as a CELIA model, and asks the CELIA method of
   the scenario for counterfactuals.  Each prediction that the method asks
   for runs the part of the pipeline after the PerturbationNode.
4. Scores each counterfactual and returns a :class:`CounterfactualReport`,
   which includes the baseline prediction of each sample.

The original pipeline config and parameters never change.
"""

import copy
import logging
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ray

from nodeml.core.common.data.data import ArrayLike, DataContext, TabularDataContext
from nodeml.core.common.exceptions import NodeConfigError, NodeInputError
from nodeml.core.common.logging import Logger
from nodeml.core.nodes.transform.perturbation import PerturbationNode
from nodeml.core.pipeline.counterfactuals.celia_adapter import (
    CeliaView,
    PipelinePredictor,
    ViewPredictor,
    counterfactual_rows,
    guard_root_logger,
    import_celia,
    make_celia_data,
    make_celia_model,
    regression_explainer_class,
    split_columns,
)
from nodeml.core.pipeline.counterfactuals.config import (
    CounterfactualEvaluatorConfig,
    CounterfactualScenario,
    FeatureConstraints,
)
from nodeml.core.pipeline.counterfactuals.insertion import (
    OUTPUT_SINK_PORT,
    PERTURBATION_NODE_NAME,
    ResolvedPoints,
    build_counterfactual_pipeline,
    resolve_points,
)
from nodeml.core.pipeline.counterfactuals.report import (
    CounterfactualReport,
    SampleResult,
    SampleStatus,
    column_scales,
    frame_rows,
    row_to_dict,
    score_counterfactual,
)
from nodeml.core.pipeline.pipeline import Pipeline, PipelineConfig
from nodeml.core.pipeline.runners.smart_runner import SmartRunner, SmartRunnerConfig

type InputData = Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]]

_log = Logger("nodeml.counterfactuals")

# Name of the CELIA target when the prediction column is also a feature.
_FALLBACK_TARGET_NAME = "__nodeml_prediction__"


@dataclass(frozen=True)
class ReferenceData:
    """Rows at the insertion point and the pipeline predictions for them.

    Attributes:
        rows: The rows that reach the PerturbationNode.
        context: The context of *rows*.
        predictions: The prediction for each row.
        output_column: The name of the prediction column.

    """

    rows: pd.DataFrame
    context: TabularDataContext
    predictions: np.ndarray
    output_column: str


@dataclass(frozen=True)
class _TaskContext:
    """Everything that a task needs, shared by all tasks of one evaluation."""

    pipeline_config: PipelineConfig
    params: dict[str, Any]
    points: ResolvedPoints
    reference: ReferenceData
    target_name: str
    continuous: list[str]
    categorical: list[str]
    scales: dict[str, float]
    constraints: FeatureConstraints
    allow_missing: bool
    runner_config: SmartRunnerConfig


def _select_column(columns: Sequence[str], wanted: str | None) -> str:
    """Return the prediction column among the output *columns*."""
    if wanted is not None:
        if wanted not in columns:
            msg = f"The output has no column '{wanted}'. Columns: {list(columns)}."
            raise NodeConfigError(msg)
        return wanted
    if len(columns) != 1:
        msg = f"The output has the columns {list(columns)}. Set OutputPoint.column."
        raise NodeConfigError(msg)
    return str(columns[0])


def _is_empty_dice_result(exc: BaseException) -> bool:
    """Return whether *exc* is the CELIA DiCE regressor failure for an empty result.

    When DiCE finds no counterfactual for a sample, the CELIA 0.1.0 DiCE
    regressor calls ``.drop`` on ``None`` in ``_get_true_predictions`` and
    raises an AttributeError.  The evaluator reports this case as
    ``"not_found"``.
    """
    frames = traceback.extract_tb(exc.__traceback__)
    return (
        isinstance(exc, AttributeError)
        and bool(frames)
        and frames[-1].name == "_get_true_predictions"
    )


def _needs_category_codes(scenario: CounterfactualScenario) -> bool:
    """Return whether the method of *scenario* needs integer category codes.

    The DiCE ``"random"`` method of the pinned DiCE version and NNCE convert
    every value to float.  The DiCE ``"genetic"`` and ``"kdtree"`` methods
    handle string categories, and fail with codes when feasible categories
    are set, because they compare the categories as strings.
    """
    if scenario.method == "nnce":
        return True
    return scenario.explainer_kwargs.get("method", "random") == "random"


def _run_method(
    context: _TaskContext,
    scenario: CounterfactualScenario,
    constraints: FeatureConstraints,
    sample: pd.DataFrame,
    target_range: tuple[float, float],
) -> tuple[PipelinePredictor, CeliaView, Any]:
    """Rebuild the pipeline and run the CELIA method of *scenario* on *sample*.

    CELIA works on a :class:`CeliaView` of the rows: frozen columns hidden,
    missing reference values filled, and integer codes for string categories
    when the method needs them.

    Returns:
        The predictor of the rebuilt pipeline, the view, and the raw CELIA
        result (in the view).

    """
    pipeline = build_counterfactual_pipeline(
        context.pipeline_config, context.params, context.points, inject=True
    )
    predictor = PipelinePredictor(
        pipeline,
        context.reference.context,
        output_column=context.reference.output_column,
        runner_config=context.runner_config,
    )
    view = CeliaView(
        context.reference.rows,
        sample,
        continuous=context.continuous,
        categorical=context.categorical,
        frozen=constraints.frozen_columns,
        code_categories=_needs_category_codes(scenario),
        allow_missing=context.allow_missing,
    )
    data = make_celia_data(
        view.reference,
        context.reference.predictions,
        target_name=context.target_name,
        continuous=view.continuous,
        categorical=view.categorical,
        constraints=view.constraints(constraints),
    )
    explainer = regression_explainer_class(scenario.method)(
        make_celia_model(ViewPredictor(predictor, view)),
        data,
        **scenario.explainer_kwargs,
    )
    # Some CELIA methods (NNCE) accept the range only as a list.
    raw = explainer.generate_counterfactuals(
        view.sample, target_range=list(target_range), **scenario.generate_kwargs
    )
    return predictor, view, raw


def _error_result(
    scenario: CounterfactualScenario,
    sample_id: str,
    sample: pd.DataFrame,
    message: str | None,
    baseline: float | None = None,
) -> SampleResult:
    """Return an ``"error"`` result for a sample, without counterfactuals.

    Args:
        scenario: The scenario.
        sample_id: Identifier of the sample.
        sample: The sample, as one row.
        message: The error message.
        baseline: The baseline prediction, if the sample has one.

    """
    return SampleResult(
        scenario=scenario.name,
        method=scenario.method,
        sample_id=sample_id,
        original=row_to_dict(frame_rows(sample)[0]),
        baseline_prediction=baseline,
        target_range=None if baseline is None else scenario.target.resolve(baseline),
        status="error",
        message=message,
        runtime_s=0.0,
    )


def _task_error(
    task: tuple[CounterfactualScenario, str, pd.DataFrame, float], exc: BaseException
) -> SampleResult:
    """Return the ``"error"`` result of a task that raised *exc*."""
    scenario, sample_id, sample, baseline = task
    message = f"The task failed: {type(exc).__name__}: {exc}"
    _log.warning(
        "Counterfactual task failed",
        scenario=scenario.name,
        sample_id=sample_id,
        error=message,
    )
    return _error_result(scenario, sample_id, sample, message, baseline)


def explain_sample(
    context: _TaskContext,
    scenario: CounterfactualScenario,
    sample_id: str,
    sample: pd.DataFrame,
    baseline: float,
) -> SampleResult:
    """Run one scenario on one sample and score the counterfactuals.

    This function is the unit of work of the evaluator: a Ray task runs it
    in a worker process, or the evaluator calls it directly.  It rebuilds the
    pipeline, so it shares no state with other tasks.

    Args:
        context: The shared data of the evaluation.
        scenario: The scenario to run.
        sample_id: Identifier of the sample in the report.
        sample: The sample, as one row at the insertion point.
        baseline: The pipeline prediction for the sample.

    Returns:
        The result.  A failure of the method gives a result with the status
        ``"not_found"``, ``"already_in_range"`` or ``"error"``; it does not
        raise.

    """
    celia = import_celia()
    constraints = scenario.constraints or context.constraints
    target_range = scenario.target.resolve(baseline)
    original = frame_rows(sample)[0]

    frozen: list[str] = []

    def result(
        status: SampleStatus,
        runtime: float,
        message: str | None = None,
        records: list | None = None,
    ) -> SampleResult:
        return SampleResult(
            scenario=scenario.name,
            method=scenario.method,
            sample_id=sample_id,
            original=row_to_dict(original),
            baseline_prediction=baseline,
            target_range=target_range,
            status=status,
            message=message,
            frozen_columns=frozen,
            runtime_s=runtime,
            counterfactuals=records or [],
        )

    start = time.perf_counter()
    failure: tuple[SampleStatus, str] | None = None
    try:
        # BugDoc and other libraries configure the root logger during a run.
        with guard_root_logger():
            predictor, view, raw = _run_method(
                context, scenario, constraints, sample, target_range
            )
            frozen = view.frozen
    except celia.InstancesAreWithinRangeError as exc:
        failure = ("already_in_range", str(exc))
    except celia.NoCounterfactualsFoundError as exc:
        failure = ("not_found", str(exc))
    except Exception as exc:  # noqa: BLE001 - one failed task must not stop the evaluation
        if _is_empty_dice_result(exc):
            failure = ("not_found", "DiCE found no counterfactual.")
        else:
            failure = ("error", f"{type(exc).__name__}: {exc}")
            _log.warning(
                "Counterfactual method failed",
                scenario=scenario.name,
                sample_id=sample_id,
                error=failure[1],
            )
    runtime = time.perf_counter() - start
    if failure is not None:
        return result(failure[0], runtime, failure[1])

    frames = [frame for frame in counterfactual_rows(raw) if not frame.empty]
    if not frames:
        return result("not_found", runtime, "The method returned no counterfactual.")
    try:
        candidates = view.from_celia(pd.concat(frames, ignore_index=True))
        # Predict again with the pipeline: one rule for all methods.
        predictions = predictor.predict_frame(candidates)
        records = [
            score_counterfactual(
                index,
                original,
                candidate,
                float(prediction),
                target_range,
                continuous=context.continuous,
                scales=context.scales,
                constraints=constraints,
            )
            for index, (candidate, prediction) in enumerate(
                zip(frame_rows(candidates), predictions, strict=True)
            )
        ]
    except Exception as exc:  # noqa: BLE001 - one bad result must not stop the evaluation
        message = f"The counterfactuals cannot be scored: {type(exc).__name__}: {exc}"
        _log.warning(
            "Counterfactual scoring failed",
            scenario=scenario.name,
            sample_id=sample_id,
            error=message,
        )
        return result("error", runtime, message)
    return result("found", runtime, records=records)


class CounterfactualEvaluator:
    """Generate and score counterfactual explanations for a trained pipeline.

    Example::

        evaluator = CounterfactualEvaluator.from_pipeline(
            trained_pipeline,
            config=CounterfactualEvaluatorConfig(
                constraints=FeatureConstraints(immutable_columns=["age"]),
            ),
        )
        report = evaluator.evaluate(
            [
                CounterfactualScenario(
                    name="dice",
                    method="dice",
                    target=TargetRange(kind="relative", low=0.1, high=0.2),
                    generate_kwargs={"total_CFs": 3},
                )
            ],
            samples=X.iloc[:5],
            reference_input={"source": {"X": (X, X_context)}},
        )
        report.summary_frame()

    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        params: Mapping[str, Any],
        *,
        config: CounterfactualEvaluatorConfig | None = None,
    ) -> None:
        """Store copies of the pipeline config and of the trained parameters.

        Args:
            pipeline_config: The layout of the trained pipeline.
            params: The trained parameters (``Pipeline.get_params()``).
            config: The evaluator config.

        Raises:
            PipelineValidationError: If the insertion point or the output
                point is ambiguous or does not exist.

        """
        self._pipeline_config = pipeline_config.model_copy(deep=True)
        self._params = copy.deepcopy(dict(params))
        self._config = config or CounterfactualEvaluatorConfig()
        self._points = resolve_points(
            self._pipeline_config, self._config.insertion, self._config.output
        )
        pipeline = Pipeline(config=self._pipeline_config.model_copy(deep=True))
        self._pipeline_name = pipeline.name
        self._pipeline_version = str(pipeline.version)
        self._pipeline_hash = pipeline.hash()
        _log.info(
            "Counterfactual evaluator ready",
            pipeline_name=self._pipeline_name,
            insertion=f"{self._points.insertion_node}.{self._points.insertion_port}",
            output=f"{self._points.output_node}.{self._points.output_port}",
        )

    @classmethod
    def from_pipeline(
        cls, pipeline: Pipeline, *, config: CounterfactualEvaluatorConfig | None = None
    ) -> "CounterfactualEvaluator":
        """Create an evaluator from a compiled, trained pipeline.

        The pipeline does not change.

        Raises:
            PipelineCompilationError: If the pipeline is not compiled.

        """
        if not pipeline.compiled:
            from nodeml.core.common.exceptions import (  # noqa: PLC0415 - error path only
                PipelineCompilationError,
            )

            msg = "The pipeline must be compiled and trained."
            raise PipelineCompilationError(msg)
        return cls(pipeline.config, pipeline.get_params(), config=config)

    @classmethod
    def from_dir(
        cls,
        dir_path: str | Path,
        file_basename: str | None = None,
        *,
        config: CounterfactualEvaluatorConfig | None = None,
    ) -> "CounterfactualEvaluator":
        """Create an evaluator from a pipeline saved with ``save_config_to_dir`` and ``save_params_to_dir``."""
        pipeline = Pipeline.load_from_dir(dir_path, file_basename, load_params=True)
        return cls.from_pipeline(pipeline, config=config)

    @property
    def config(self) -> CounterfactualEvaluatorConfig:
        """The evaluator config."""
        return self._config

    @property
    def points(self) -> ResolvedPoints:
        """The resolved insertion point and output point."""
        return self._points

    @property
    def pipeline_config(self) -> PipelineConfig:
        """A copy of the pipeline config."""
        return self._pipeline_config.model_copy(deep=True)

    def reference_data(self, input_data: InputData | None = None) -> ReferenceData:
        """Run the pipeline in inference and read the rows at the insertion point.

        Args:
            input_data: Input for the runner, as for ``SmartRunner.infer``.
                ``None`` suits sources that load their own data (for example
                a TabularCSVFetcher).

        Returns:
            The rows that reach the PerturbationNode, and the pipeline
            prediction for each row.

        Raises:
            NodeInputError: If the pipeline removes or adds rows between the
                insertion point and the output.

        """
        pipeline = build_counterfactual_pipeline(
            self._pipeline_config, self._params, self._points, inject=False
        )
        runner = SmartRunner(pipeline, config=self._config.runner_config)
        outputs = runner.infer(input_data=input_data)
        node = pipeline.node_objects[PERTURBATION_NODE_NAME]
        if not isinstance(node, PerturbationNode) or node.last_input is None:
            msg = "The PerturbationNode received no rows. Check the input data."
            raise NodeInputError(msg)
        rows, context = node.last_input
        output, _ = outputs[OUTPUT_SINK_PORT]
        column = _select_column(list(output.columns), self._config.output.column)
        predictions = np.asarray(
            pd.to_numeric(output[column], errors="coerce"), dtype=np.float64
        )
        if len(predictions) != len(rows):
            msg = (
                f"The pipeline gives {len(predictions)} predictions for {len(rows)} rows "
                "at the insertion point. Counterfactuals need one prediction for each row."
            )
            raise NodeInputError(msg)
        return ReferenceData(
            rows=rows.reset_index(drop=True),
            context=context.copy(),
            predictions=predictions,
            output_column=column,
        )

    def evaluate(
        self,
        scenarios: Sequence[CounterfactualScenario],
        *,
        samples: pd.DataFrame | None = None,
        sample_indices: Sequence[int] | None = None,
        reference_input: InputData | None = None,
    ) -> CounterfactualReport:
        """Generate and score counterfactuals for each scenario and sample.

        Args:
            scenarios: The scenarios.  Their names must be unique.
            samples: The samples to explain, as rows at the insertion point.
                With the default insertion point, these are rows of the data
                source.
            sample_indices: Positions of the samples in the reference rows,
                instead of *samples*.
            reference_input: Runner input for the reference data (see
                :meth:`reference_data`).

        Returns:
            The report.

        Raises:
            NodeConfigError: If the scenarios or the samples are not valid.

        """
        self._check_scenarios(scenarios)
        reference = self.reference_data(reference_input)
        rows, ids = self._select_samples(reference, samples, sample_indices)
        continuous, categorical = split_columns(reference.context)
        target_name = (
            reference.output_column
            if reference.output_column not in reference.context.columns
            else _FALLBACK_TARGET_NAME
        )
        context = _TaskContext(
            pipeline_config=self._pipeline_config,
            params=self._params,
            points=self._points,
            reference=reference,
            target_name=target_name,
            continuous=continuous,
            categorical=categorical,
            scales=column_scales(reference.rows, continuous),
            constraints=self._config.constraints,
            allow_missing=self._config.missing_values == "freeze",
            runner_config=self._config.runner_config,
        )
        baselines = self._baselines(context, rows)
        # A sample that the pipeline cannot predict gets an error result in
        # each scenario; the other samples still run.
        results: list[SampleResult | None] = []
        tasks = []
        for scenario in scenarios:
            for position, sample_id in enumerate(ids):
                sample = rows.iloc[[position]]
                baseline, error = baselines[position]
                if baseline is None:
                    results.append(_error_result(scenario, sample_id, sample, error))
                else:
                    results.append(None)
                    tasks.append((scenario, sample_id, sample, baseline))
        with _log.timed(
            "Counterfactual evaluation",
            num_scenarios=len(scenarios),
            num_samples=len(ids),
            num_tasks=len(tasks),
            use_ray=self._config.use_ray,
        ):
            done = iter(self._run(context, tasks))
        results = [result if result is not None else next(done) for result in results]
        return CounterfactualReport(
            pipeline_name=self._pipeline_name,
            pipeline_version=self._pipeline_version,
            pipeline_hash=self._pipeline_hash,
            task=self._config.task,
            insertion=(self._points.insertion_node, self._points.insertion_port),
            output=(
                self._points.output_node,
                self._points.output_port,
                reference.output_column,
            ),
            feature_columns=list(reference.context.columns),
            continuous_columns=continuous,
            categorical_columns=categorical,
            constraints=self._config.constraints,
            scenarios=list(scenarios),
            results=results,
        )

    # --- Internal methods ---------------------------------------------------

    @staticmethod
    def _check_scenarios(scenarios: Sequence[CounterfactualScenario]) -> None:
        """Reject an empty list and duplicate scenario names."""
        if not scenarios:
            msg = "Give at least one CounterfactualScenario."
            raise NodeConfigError(msg)
        names = [scenario.name for scenario in scenarios]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            msg = f"Scenario names must be unique. Duplicates: {duplicates}."
            raise NodeConfigError(msg)

    @staticmethod
    def _select_samples(
        reference: ReferenceData,
        samples: pd.DataFrame | None,
        sample_indices: Sequence[int] | None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Return the sample rows (reference columns, in order) and their identifiers."""
        if (samples is None) == (sample_indices is None):
            msg = "Give exactly one of samples and sample_indices."
            raise NodeConfigError(msg)
        columns = reference.context.columns
        if samples is not None:
            missing = [column for column in columns if column not in samples.columns]
            if missing:
                msg = f"The samples have no columns {missing}. Expected the columns at the insertion point: {columns}."
                raise NodeConfigError(msg)
            if samples.empty:
                msg = "Give at least one sample."
                raise NodeConfigError(msg)
            return samples.loc[:, columns], [str(label) for label in samples.index]
        positions = list(sample_indices or [])
        if not positions:
            msg = "Give at least one sample index."
            raise NodeConfigError(msg)
        n_rows = len(reference.rows)
        outside = [position for position in positions if not 0 <= position < n_rows]
        if outside:
            msg = f"Sample indices {outside} are outside the {n_rows} reference rows."
            raise NodeConfigError(msg)
        return reference.rows.iloc[positions].reset_index(drop=True), [
            str(p) for p in positions
        ]

    @staticmethod
    def _baselines(
        context: _TaskContext, rows: pd.DataFrame
    ) -> list[tuple[float | None, str | None]]:
        """Return ``(prediction, None)`` or ``(None, error)`` for each sample."""
        pipeline = build_counterfactual_pipeline(
            context.pipeline_config, context.params, context.points, inject=True
        )
        predictor = PipelinePredictor(
            pipeline,
            context.reference.context,
            output_column=context.reference.output_column,
            runner_config=context.runner_config,
        )
        try:
            predictions = [float(value) for value in predictor.predict_frame(rows)]
        except Exception:  # noqa: BLE001 - find the failing samples one by one
            predictions = []
            for position in range(len(rows)):
                try:
                    predictions.append(
                        float(predictor.predict_frame(rows.iloc[[position]])[0])
                    )
                except Exception as exc:  # noqa: BLE001 - reported in the result
                    predictions.append(
                        f"The pipeline cannot predict the sample: {type(exc).__name__}: {exc}"
                    )
        out: list[tuple[float | None, str | None]] = []
        for value in predictions:
            if isinstance(value, str):
                out.append((None, value))
            elif not np.isfinite(value):
                out.append(
                    (
                        None,
                        "The pipeline prediction for the sample is not a finite number.",
                    )
                )
            else:
                out.append((value, None))
        return out

    def _run(
        self,
        context: _TaskContext,
        tasks: list[tuple[CounterfactualScenario, str, pd.DataFrame, float]],
    ) -> list[SampleResult]:
        """Run the tasks with Ray, or one after the other in this process.

        A task that raises gives an ``"error"`` result.  The other tasks keep
        their results.
        """
        if not self._config.use_ray:
            results = []
            for task in tasks:
                try:
                    results.append(explain_sample(context, *task))
                except Exception as exc:  # noqa: BLE001 - isolate the failing task
                    results.append(_task_error(task, exc))
            return results
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                logging_level=logging.WARNING,
            )
        context_ref = ray.put(context)
        remote = ray.remote(explain_sample).options(
            num_cpus=self._config.num_cpus_per_task
        )
        futures = [remote.remote(context_ref, *task) for task in tasks]
        results = []
        # ray.get on the whole list raises for the first failed task and
        # loses the other results, so get each result on its own.
        for task, future in zip(tasks, futures, strict=True):
            try:
                results.append(ray.get(future))
            except Exception as exc:  # noqa: BLE001 - isolate the failing task
                results.append(_task_error(task, exc))
        return results
