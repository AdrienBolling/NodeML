"""The counterfactual evaluation report and the metrics that it holds.

For each scenario and each sample, the report keeps the original row, the
baseline prediction, the target range, and every counterfactual with these
metrics:

* **validity**: the prediction of the counterfactual is in the target range.
* **proximity**: the L1 and L2 distances to the original row, over the
  continuous columns.  Each difference is divided by the range of the column
  in the reference data, so that columns with large values do not dominate.
* **sparsity**: the number of changed columns (continuous and categorical).
* **constraint violations**: changed immutable columns, and values outside
  the feasible values.

The report also has one summary for each scenario, and converts to pandas
tables and to JSON.
"""

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from nodeml.core.pipeline.counterfactuals.config import (
    CounterfactualScenario,
    FeatureConstraints,
)

type SampleStatus = Literal["found", "not_found", "already_in_range", "error"]

# Relative tolerance below which a continuous value counts as unchanged.
_CHANGE_TOLERANCE = 1e-9


def to_json_value(value: Any) -> Any:
    """Convert numpy and pandas scalars to plain Python values for JSON.

    NaN and infinite floats become ``None``.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def row_to_dict(row: Mapping[Any, Any]) -> dict[str, Any]:
    """Convert a row to a JSON-safe ``{column: value}`` dict."""
    return {str(column): to_json_value(value) for column, value in row.items()}


def frame_rows(df: pd.DataFrame) -> list[dict[Any, Any]]:
    """Return the rows of *df* as dicts.

    Unlike ``df.iloc[i]``, each value keeps the dtype of its column: an
    integer column does not become float next to a float column.
    """
    return df.to_dict(orient="records")


class CounterfactualRecord(BaseModel):
    """One counterfactual row and its metrics.

    Attributes:
        index: Position of the counterfactual among those of its sample.
        features: The full counterfactual row.
        changes: The changed columns, as ``{column: [original, new]}``.
        prediction: The prediction of the pipeline for the counterfactual.
        valid: Whether the prediction is in the target range.
        l1_distance: Scaled L1 distance to the original row (continuous
            columns).
        l2_distance: Scaled L2 distance to the original row (continuous
            columns).
        n_changed: Number of changed columns.
        violations: Broken constraints, as ``"immutable:<column>"`` or
            ``"feasible:<column>"``.

    """

    index: int
    features: dict[str, Any]
    changes: dict[str, list[Any]]
    prediction: float | None
    valid: bool
    l1_distance: float
    l2_distance: float
    n_changed: int
    violations: list[str]


class SampleResult(BaseModel):
    """The result of one scenario for one sample.

    Attributes:
        scenario: Name of the scenario.
        method: The CELIA method of the scenario.
        sample_id: Identifier of the sample (its index label).
        original: The original row.
        baseline_prediction: The prediction of the pipeline for the
            original row.
        target_range: The absolute ``[low, high]`` range for this sample,
            or ``None`` when the sample has no baseline prediction.
        status: ``"found"``, ``"not_found"``, ``"already_in_range"`` (the
            baseline is already in the target range), or ``"error"``.
        message: The error message, if any.
        frozen_columns: Columns that CELIA did not see for this sample (see
            ``FeatureConstraints.frozen_columns`` and
            ``CounterfactualEvaluatorConfig.missing_values``).
        runtime_s: Wall time of the method, in seconds.
        counterfactuals: The counterfactuals and their metrics.

    """

    scenario: str
    method: str
    sample_id: str
    original: dict[str, Any]
    baseline_prediction: float | None
    target_range: tuple[float, float] | None
    status: SampleStatus
    message: str | None = None
    frozen_columns: list[str] = Field(default_factory=list)
    runtime_s: float
    counterfactuals: list[CounterfactualRecord] = Field(default_factory=list)


class ScenarioSummary(BaseModel):
    """Aggregated metrics of one scenario over all samples.

    The mean distances and changes use the valid counterfactuals only.
    They are ``None`` when there is no valid counterfactual.
    """

    scenario: str
    method: str
    n_samples: int
    n_found: int
    n_not_found: int
    n_already_in_range: int
    n_errors: int
    n_counterfactuals: int
    n_valid: int
    validity_rate: float | None
    mean_l1_distance: float | None
    mean_l2_distance: float | None
    mean_n_changed: float | None
    n_violations: int
    total_runtime_s: float


class CounterfactualReport(BaseModel):
    """Structured result of a counterfactual evaluation.

    Attributes:
        created_at: UTC time of the evaluation.
        pipeline_name: Name of the evaluated pipeline.
        pipeline_version: Version of the evaluated pipeline.
        pipeline_hash: :meth:`Pipeline.hash` of the evaluated pipeline.
        task: The prediction task.
        insertion: ``[node, port]`` of the insertion point.
        output: ``[node, port, column]`` of the prediction.
        feature_columns: Columns at the insertion point, in order.
        continuous_columns: The continuous columns.
        categorical_columns: The categorical columns.
        constraints: The default constraints of the evaluation.
        scenarios: The evaluated scenarios.
        results: One result for each (scenario, sample) pair.
        summaries: One summary for each scenario.

    """

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    pipeline_name: str
    pipeline_version: str
    pipeline_hash: str
    task: str
    insertion: tuple[str, str]
    output: tuple[str, str, str]
    feature_columns: list[str]
    continuous_columns: list[str]
    categorical_columns: list[str]
    constraints: FeatureConstraints
    scenarios: list[CounterfactualScenario]
    results: list[SampleResult]
    summaries: list[ScenarioSummary] = Field(default_factory=list)

    def model_post_init(self, context: Any, /) -> None:
        """Compute the summaries if the caller gave none."""
        _ = context
        if not self.summaries:
            self.summaries = summarize(self.scenarios, self.results)

    # --- pandas views -----------------------------------------------------

    def summary_frame(self) -> pd.DataFrame:
        """Return one row for each scenario."""
        return pd.DataFrame([summary.model_dump() for summary in self.summaries])

    def samples_frame(self) -> pd.DataFrame:
        """Return one row for each (scenario, sample) pair, without the counterfactuals."""
        rows = [
            {
                "scenario": result.scenario,
                "method": result.method,
                "sample_id": result.sample_id,
                "baseline_prediction": result.baseline_prediction,
                "target_low": result.target_range[0] if result.target_range else None,
                "target_high": result.target_range[1] if result.target_range else None,
                "status": result.status,
                "n_counterfactuals": len(result.counterfactuals),
                "n_valid": sum(record.valid for record in result.counterfactuals),
                "runtime_s": result.runtime_s,
                "message": result.message,
            }
            for result in self.results
        ]
        return pd.DataFrame(rows)

    def counterfactuals_frame(self) -> pd.DataFrame:
        """Return one row for each counterfactual, with its metrics and features."""
        rows = [
            {
                "scenario": result.scenario,
                "method": result.method,
                "sample_id": result.sample_id,
                "cf_index": record.index,
                "baseline_prediction": result.baseline_prediction,
                "prediction": record.prediction,
                "valid": record.valid,
                "l1_distance": record.l1_distance,
                "l2_distance": record.l2_distance,
                "n_changed": record.n_changed,
                "changed_columns": sorted(record.changes),
                "violations": record.violations,
                **record.features,
            }
            for result in self.results
            for record in result.counterfactuals
        ]
        return pd.DataFrame(rows)

    def changes_frame(self, scenario: str, sample_id: str) -> pd.DataFrame:
        """Return the counterfactuals of one sample, with ``"-"`` for unchanged values.

        The first row is the original sample.

        Args:
            scenario: Name of the scenario.
            sample_id: Identifier of the sample.

        Raises:
            KeyError: If the pair does not exist.

        """
        for result in self.results:
            if result.scenario == scenario and result.sample_id == sample_id:
                rows = [{**result.original, "prediction": result.baseline_prediction}]
                rows.extend(
                    {
                        **{
                            column: (
                                record.features[column]
                                if column in record.changes
                                else "-"
                            )
                            for column in self.feature_columns
                        },
                        "prediction": record.prediction,
                    }
                    for record in result.counterfactuals
                )
                return pd.DataFrame(
                    rows,
                    index=["original"]
                    + [f"cf_{record.index}" for record in result.counterfactuals],
                )
        msg = f"No result for scenario '{scenario}' and sample '{sample_id}'."
        raise KeyError(msg)

    # --- JSON ---------------------------------------------------------------

    def save_json(self, path: str | Path) -> Path:
        """Write the report as JSON and return the path.

        The parent directory is created if needed.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2))
        return target

    @classmethod
    def load_json(cls, path: str | Path) -> Self:
        """Read a report written by :meth:`save_json`."""
        return cls.model_validate_json(Path(path).read_text())


# --- Metrics -------------------------------------------------------------------


def column_scales(
    reference: pd.DataFrame, continuous: Sequence[str]
) -> dict[str, float]:
    """Return the range of each continuous column, used to scale distances.

    A column with a zero or undefined range gets the scale 1.0.
    """
    scales: dict[str, float] = {}
    for column in continuous:
        values = np.asarray(
            pd.to_numeric(reference[column], errors="coerce"), dtype=float
        )
        finite = values[np.isfinite(values)]
        spread = float(finite.max() - finite.min()) if finite.size else 0.0
        scales[column] = spread if math.isfinite(spread) and spread > 0 else 1.0
    return scales


def _changed(original: Any, new: Any, *, continuous: bool) -> bool:
    """Return whether a value changed."""
    if pd.isna(original) and pd.isna(new):
        return False
    if continuous:
        try:
            a, b = float(original), float(new)
        except (TypeError, ValueError):
            return original != new
        return abs(a - b) > _CHANGE_TOLERANCE * max(1.0, abs(a))
    return original != new


def _in_range(value: Any, bounds: tuple[float, float]) -> bool:
    """Return whether *value* is a number in the closed range *bounds*."""
    low, high = bounds
    try:
        return low <= float(value) <= high
    except (TypeError, ValueError):
        return False


def score_counterfactual(
    index: int,
    original: Mapping[str, Any],
    counterfactual: Mapping[str, Any],
    prediction: float | None,
    target_range: tuple[float, float],
    *,
    continuous: Sequence[str],
    scales: Mapping[str, float],
    constraints: FeatureConstraints,
) -> CounterfactualRecord:
    """Compute the metrics of one counterfactual.

    Args:
        index: Position of the counterfactual among those of its sample.
        original: The original row, as ``{column: value}`` (see :func:`frame_rows`).
        counterfactual: The counterfactual row, with the same columns.
        prediction: The prediction for the counterfactual.
        target_range: The absolute ``(low, high)`` target range.
        continuous: The continuous columns.
        scales: The scale of each continuous column (see
            :func:`column_scales`).
        constraints: The constraints to check.

    Returns:
        The record with all metrics.

    """
    continuous_set = set(continuous)
    changes: dict[str, list[Any]] = {}
    l1 = 0.0
    l2 = 0.0
    for column, old in original.items():
        new = counterfactual[column]
        is_continuous = column in continuous_set
        if not _changed(old, new, continuous=is_continuous):
            continue
        changes[str(column)] = [to_json_value(old), to_json_value(new)]
        if is_continuous:
            try:
                step = abs(float(new) - float(old)) / scales.get(column, 1.0)
            except (TypeError, ValueError):
                continue
            l1 += step
            l2 += step * step
    violations = [
        f"immutable:{column}"
        for column in constraints.immutable_columns
        if column in changes
    ]
    violations.extend(
        f"feasible:{column}"
        for column, bounds in constraints.feasible_ranges.items()
        if column in counterfactual and not _in_range(counterfactual[column], bounds)
    )
    violations.extend(
        f"feasible:{column}"
        for column, allowed in constraints.feasible_categories.items()
        if column in counterfactual and counterfactual[column] not in allowed
    )
    low, high = target_range
    valid = (
        prediction is not None
        and math.isfinite(prediction)
        and low <= prediction <= high
    )
    return CounterfactualRecord(
        index=index,
        features=row_to_dict(counterfactual),
        changes=changes,
        prediction=to_json_value(prediction),
        valid=valid,
        l1_distance=l1,
        l2_distance=math.sqrt(l2),
        n_changed=len(changes),
        violations=violations,
    )


def _mean(values: list[float]) -> float | None:
    """Return the mean, or ``None`` for an empty list."""
    return float(np.mean(values)) if values else None


def summarize(
    scenarios: Sequence[CounterfactualScenario], results: Sequence[SampleResult]
) -> list[ScenarioSummary]:
    """Aggregate the results of each scenario."""
    summaries = []
    for scenario in scenarios:
        own = [result for result in results if result.scenario == scenario.name]
        records = [record for result in own for record in result.counterfactuals]
        valid = [record for record in records if record.valid]
        statuses = [result.status for result in own]
        summaries.append(
            ScenarioSummary(
                scenario=scenario.name,
                method=scenario.method,
                n_samples=len(own),
                n_found=statuses.count("found"),
                n_not_found=statuses.count("not_found"),
                n_already_in_range=statuses.count("already_in_range"),
                n_errors=statuses.count("error"),
                n_counterfactuals=len(records),
                n_valid=len(valid),
                validity_rate=len(valid) / len(records) if records else None,
                mean_l1_distance=_mean([record.l1_distance for record in valid]),
                mean_l2_distance=_mean([record.l2_distance for record in valid]),
                mean_n_changed=_mean([float(record.n_changed) for record in valid]),
                n_violations=sum(len(record.violations) for record in records),
                total_runtime_s=float(sum(result.runtime_s for result in own)),
            )
        )
    return summaries
