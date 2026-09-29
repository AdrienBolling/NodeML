"""Configuration models of the counterfactual evaluator."""

from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

from nodeml.core.pipeline.runners.smart_runner import SmartRunnerConfig

type Task = Literal["regression"]
"""Tasks that the evaluator supports.  Classification comes later."""

type RegressionMethod = Literal["dice", "nnce"]
"""CELIA regression methods that work with a pipeline as a black box.

BugDoc is not in the list: with CELIA 0.1.0 (commit 0b95d40), the BugDoc
regressor always fails with an IndexError, because CELIA gives BugDoc a list
of booleans where BugDoc expects rows.
"""


class InsertionPoint(BaseModel):
    """Where the evaluator inserts the PerturbationNode.

    The evaluator routes every edge that leaves ``node.port`` through the
    PerturbationNode.  The counterfactuals are then rows at this point of
    the pipeline.

    Attributes:
        node: Name of the node whose output port receives the
            PerturbationNode.  ``None`` selects the data source: the source
            node that the output node needs in inference.  There must be
            exactly one such source port.
        port: Name of the output port.  ``None`` selects the only output
            port of *node* that feeds the output node in inference.

    """

    node: str | None = None
    port: str | None = None

    @model_validator(mode="after")
    def _port_needs_node(self) -> Self:
        """Reject a port without a node."""
        if self.port is not None and self.node is None:
            msg = "InsertionPoint.port needs InsertionPoint.node."
            raise ValueError(msg)
        return self


class OutputPoint(BaseModel):
    """The model output that the counterfactuals must change.

    Attributes:
        node: Name of the node that produces the prediction.  ``None``
            selects the only node output that feeds the Sink.
        port: Name of the output port.  ``None`` selects the only port of
            *node* that feeds the Sink.
        column: Name of the prediction column.  ``None`` needs an output
            with exactly one column.

    """

    node: str | None = None
    port: str | None = None
    column: str | None = None

    @model_validator(mode="after")
    def _port_needs_node(self) -> Self:
        """Reject a port without a node."""
        if self.port is not None and self.node is None:
            msg = "OutputPoint.port needs OutputPoint.node."
            raise ValueError(msg)
        return self


class FeatureConstraints(BaseModel):
    """Real-world constraints on the features that a counterfactual may change.

    The column names refer to the columns at the insertion point.  CELIA
    uses these constraints (see ``celia.Data``); each method supports a
    subset of them.

    Attributes:
        immutable_columns: Columns that a counterfactual must not change.
        feasible_ranges: The allowed ``(min, max)`` range of numerical
            columns.
        feasible_categories: The allowed values of categorical columns.
        monotonic_increasing_columns: Columns that may only increase.
        correlated_features: ``(cause, effect, delta)`` triples: when
            *cause* increases, *effect* increases by *delta*.
        frozen_columns: Columns that CELIA does not see at all.  Every
            counterfactual keeps the value of the sample.  Use it for
            identifiers and for columns with many categories, which make
            the methods slow.

    """

    immutable_columns: list[str] = Field(default_factory=list)
    feasible_ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)
    feasible_categories: dict[str, list[Any]] = Field(default_factory=dict)
    monotonic_increasing_columns: list[str] = Field(default_factory=list)
    correlated_features: list[tuple[str, str, float]] = Field(default_factory=list)
    frozen_columns: list[str] = Field(default_factory=list)


class TargetRange(BaseModel):
    """The prediction range that a regression counterfactual must reach.

    ``kind`` gives the meaning of ``low`` and ``high`` for a sample with the
    baseline prediction ``p``:

    * ``"absolute"``: the range is ``[low, high]``.
    * ``"delta"``: the range is ``[p + low, p + high]``.
    * ``"relative"``: the range is ``[p * (1 + low), p * (1 + high)]``, for
      example ``low=0.1, high=0.2`` asks for 10 % to 20 % more.

    """

    kind: Literal["absolute", "delta", "relative"] = "delta"
    low: float
    high: float

    @model_validator(mode="after")
    def _low_below_high(self) -> Self:
        """Reject an empty range."""
        if not self.low < self.high:
            msg = f"TargetRange needs low < high, got low={self.low}, high={self.high}."
            raise ValueError(msg)
        return self

    def resolve(self, baseline: float) -> tuple[float, float]:
        """Return the absolute ``(low, high)`` range for a baseline prediction.

        Args:
            baseline: The prediction of the model for the original sample.

        Returns:
            The absolute range, with low <= high.

        """
        if self.kind == "absolute":
            return self.low, self.high
        if self.kind == "delta":
            return baseline + self.low, baseline + self.high
        bounds = (baseline * (1 + self.low), baseline * (1 + self.high))
        # A negative baseline swaps the bounds.
        return min(bounds), max(bounds)


class CounterfactualScenario(BaseModel):
    """One counterfactual method with its settings.

    Attributes:
        name: Unique name of the scenario in the report.
        method: The CELIA method.
        target: The prediction range that each counterfactual must reach.
        explainer_kwargs: Keyword arguments for the CELIA explainer
            constructor, for example ``{"method": "genetic"}`` for DiCE.
        generate_kwargs: Keyword arguments for ``generate_counterfactuals``,
            for example ``{"total_CFs": 3}`` for DiCE or
            ``{"n_counterfactuals": 3}`` for NNCE.
        constraints: Constraints for this scenario.  ``None`` uses the
            constraints of the evaluator config.

    """

    name: str = Field(min_length=1)
    method: RegressionMethod = "dice"
    target: TargetRange
    explainer_kwargs: dict[str, Any] = Field(default_factory=dict)
    generate_kwargs: dict[str, Any] = Field(default_factory=dict)
    constraints: FeatureConstraints | None = None


class CounterfactualEvaluatorConfig(BaseModel):
    """Configuration of the :class:`CounterfactualEvaluator`.

    Attributes:
        task: The prediction task.  Only ``"regression"`` is supported now.
        insertion: Where the PerturbationNode goes.  The default is right
            after the data source.
        output: The prediction that the counterfactuals must change.
        constraints: Default constraints for all scenarios.
        missing_values: What to do with missing values, which CELIA does
            not accept.  ``"freeze"``: a column that is missing in the sample
            is frozen (CELIA does not see it, and every counterfactual keeps
            the missing value), and the missing values of the reference rows
            are filled for CELIA with the median or the most frequent value.
            The pipeline always gets the real rows.  ``"error"``: a missing
            value gives the result status ``"error"``.
        use_ray: If ``True``, run one Ray task for each (scenario, sample)
            pair, in parallel.  If ``False``, run them one after the other
            in this process.
        num_cpus_per_task: CPUs that Ray reserves for each task.
        runner_config: Config of the runners that execute the pipelines.

    """

    task: Task = "regression"
    insertion: InsertionPoint = Field(default_factory=InsertionPoint)
    output: OutputPoint = Field(default_factory=OutputPoint)
    constraints: FeatureConstraints = Field(default_factory=FeatureConstraints)
    missing_values: Literal["freeze", "error"] = "freeze"
    use_ray: bool = True
    num_cpus_per_task: float = Field(default=1.0, gt=0)
    runner_config: SmartRunnerConfig = Field(
        default_factory=lambda: SmartRunnerConfig(verbose=False)
    )
