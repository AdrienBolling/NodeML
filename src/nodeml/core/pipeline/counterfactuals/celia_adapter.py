"""Connect a trained NodeML pipeline to the CELIA counterfactual library.

CELIA is an optional dependency: install it with
``pip install "nodeml[counterfactuals]"``.  This module imports it only when
a function needs it.

* :class:`PipelinePredictor` runs a pipeline that has an injecting
  PerturbationNode, and returns the predictions for rows at the insertion
  point.
* :func:`make_celia_model` wraps a predictor as a ``celia.BaseModel``, so
  that every CELIA method can use the pipeline as a black box.
* :func:`make_celia_data` builds a ``celia.Data`` object from the reference
  rows, their context and the feature constraints.
* :class:`CeliaView` gives CELIA rows without missing values and, where
  needed, integer codes instead of string categories (see
  :class:`CategoryCodes`), and converts CELIA rows back to pipeline rows.
"""

import functools
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from nodeml.core.common.data.data import (
    CategoricalData,
    NumericalData,
    TabularDataContext,
    infer_category,
)
from nodeml.core.common.exceptions import NodeInputError
from nodeml.core.nodes.transform.perturbation import PerturbationNode
from nodeml.core.pipeline.counterfactuals.config import (
    FeatureConstraints,
    RegressionMethod,
)
from nodeml.core.pipeline.counterfactuals.insertion import (
    OUTPUT_SINK_PORT,
    PERTURBATION_NODE_NAME,
)
from nodeml.core.pipeline.pipeline import Pipeline
from nodeml.core.pipeline.runners.smart_runner import SmartRunner, SmartRunnerConfig

INSTALL_HINT = (
    'Install the optional dependencies with: pip install "nodeml[counterfactuals]"'
)

REGRESSION_EXPLAINERS: dict[RegressionMethod, str] = {
    "dice": "DiceRegressorExplainer",
    "nnce": "NNCERegressorExplainer",
}
"""CELIA explainer class for each regression method."""


@contextmanager
def guard_root_logger() -> Iterator[None]:
    """Keep the root logger configuration while CELIA code runs.

    BugDoc, a dependency of CELIA, calls ``logging.basicConfig(level=DEBUG)``
    in modules that CELIA imports, also during a run.  Other libraries call
    ``logging.warning()``, which also configures the root logger.  Without a
    guard, every DEBUG message of the process then goes to stderr.

    ``basicConfig`` does nothing when the root logger has a handler.  So the
    guard adds a ``NullHandler`` to a root logger without handlers.  At the
    end, it removes the handlers that were added and restores the level.
    """
    root = logging.getLogger()
    level = root.level
    handlers = list(root.handlers)
    guard = None
    if not handlers:
        guard = logging.NullHandler()
        root.addHandler(guard)
    try:
        yield
    finally:
        for handler in list(root.handlers):
            if handler not in handlers:
                root.removeHandler(handler)
                if handler is not guard:
                    handler.close()
        root.setLevel(level)


def import_celia() -> Any:
    """Import CELIA and keep the logging configuration of the process.

    Returns:
        The ``celia`` module.

    Raises:
        ImportError: If CELIA is not installed.

    """
    with guard_root_logger():
        try:
            import celia  # noqa: PLC0415 - optional dependency, imported on demand
        except ImportError as exc:
            raise ImportError(INSTALL_HINT) from exc
    return celia


def _coerce_to_reference(df: pd.DataFrame, context: TabularDataContext) -> pd.DataFrame:
    """Return *df* with the reference columns and, where lossless, their dtypes.

    Counterfactual methods can return integral values as floats.  A column
    with an integer reference dtype goes back to that dtype when all values
    are integral.  Other columns keep their values.
    """
    missing = [column for column in context.columns if column not in df.columns]
    if missing:
        msg = f"The rows have no columns {missing}. Expected the columns {context.columns}."
        raise NodeInputError(msg)
    out = df.loc[:, context.columns].copy()
    for column, dtype in zip(context.columns, context.dtypes, strict=True):
        values = out[column]
        if values.dtype == dtype:
            continue
        if is_integer_dtype(dtype) and is_numeric_dtype(values):
            numeric = values.astype("float64")
            if numeric.notna().all() and np.allclose(numeric, np.round(numeric)):
                out[column] = np.round(numeric).astype(dtype)
            continue
        try:
            out[column] = values.astype(dtype)
        except (TypeError, ValueError):
            # Keep the values: the pipeline reports a clear error if needed.
            continue
    return out


class PipelinePredictor:
    """Predict with a trained pipeline from rows at the insertion point.

    The pipeline must have an injecting PerturbationNode named
    :data:`~nodeml.core.pipeline.counterfactuals.insertion.PERTURBATION_NODE_NAME`
    and a Sink port
    :data:`~nodeml.core.pipeline.counterfactuals.insertion.OUTPUT_SINK_PORT`
    (see :func:`~nodeml.core.pipeline.counterfactuals.insertion.build_counterfactual_pipeline`).
    """

    def __init__(
        self,
        pipeline: Pipeline,
        reference_context: TabularDataContext,
        *,
        output_column: str | None = None,
        runner_config: SmartRunnerConfig | None = None,
    ) -> None:
        """Create the runner of the pipeline.

        Args:
            pipeline: The compiled, trained pipeline with an injecting
                PerturbationNode.
            reference_context: Context of the rows at the insertion point.
                The categories of the injected columns come from it.
            output_column: The prediction column.  ``None`` needs an output
                with one column.
            runner_config: Config of the runner.  Default: no progress bar.

        """
        node = pipeline.node_objects.get(PERTURBATION_NODE_NAME)
        if not isinstance(node, PerturbationNode):
            msg = f"The pipeline has no PerturbationNode named '{PERTURBATION_NODE_NAME}'."
            raise NodeInputError(msg)
        self._node = node
        self._runner = SmartRunner(
            pipeline, config=runner_config or SmartRunnerConfig(verbose=False)
        )
        self._context = reference_context
        self._output_column = output_column
        self.n_calls = 0
        """Number of prediction calls, for diagnostics."""

    @property
    def feature_columns(self) -> list[str]:
        """The columns at the insertion point."""
        return list(self._context.columns)

    def predict_frame(self, rows: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
        """Return one float prediction for each row.

        Args:
            rows: Rows at the insertion point, with the reference columns.
                A Series is one row.  An array uses the reference column
                order.

        Returns:
            A 1-D float64 array.

        Raises:
            NodeInputError: If a reference column is missing, or if the
                prediction column cannot be selected.

        """
        if isinstance(rows, pd.Series):
            rows = rows.to_frame().T
        elif not isinstance(rows, pd.DataFrame):
            rows = pd.DataFrame(np.asarray(rows), columns=self._context.columns)
        df = _coerce_to_reference(rows, self._context).reset_index(drop=True)
        self._node.set_rows(df, self._context.aligned_to(df))
        self.n_calls += 1
        try:
            output, _ = self._runner.infer()[OUTPUT_SINK_PORT]
        finally:
            self._node.clear_rows()
        column = self._select_column(list(output.columns))
        return np.asarray(
            pd.to_numeric(output[column], errors="coerce"), dtype=np.float64
        )

    def _select_column(self, columns: Sequence[str]) -> str:
        """Return the prediction column among the output *columns*."""
        if self._output_column is not None:
            if self._output_column not in columns:
                msg = f"The output has no column '{self._output_column}'. Columns: {list(columns)}."
                raise NodeInputError(msg)
            return self._output_column
        if len(columns) != 1:
            msg = f"The output has the columns {list(columns)}. Set OutputPoint.column."
            raise NodeInputError(msg)
        self._output_column = str(columns[0])
        return self._output_column


@functools.cache
def _pipeline_model_class() -> type:
    """Return the ``celia.BaseModel`` subclass that wraps a predictor."""
    celia = import_celia()

    class PipelineModel(celia.BaseModel):
        """A ``celia.BaseModel`` that predicts with a NodeML pipeline."""

        def predict(self, x: Any) -> np.ndarray:
            """Return the pipeline predictions for rows at the insertion point."""
            return self.model.predict_frame(x)

        def predict_proba(self, x: Any) -> np.ndarray:
            """Fail: the evaluator supports regression only for now."""
            _ = x
            msg = "The counterfactual evaluator supports regression only for now."
            raise NotImplementedError(msg)

    return PipelineModel


class CategoryCodes:
    """Reversible integer codes for the non-numeric categorical columns.

    Some CELIA methods convert every value to float: the DiCE ``"random"``
    method of the DiCE version that CELIA pins, and NNCE.  For them, the
    evaluator gives CELIA an integer code for each category, and decodes the
    codes before each prediction and in the report.  DiCE keeps these
    columns categorical.  NNCE measures distances between codes, which is
    only an approximation: the codes have no real order.

    A missing value gets its own code and decodes to NaN.
    """

    def __init__(self, reference: pd.DataFrame, categorical: Sequence[str]) -> None:
        """Assign codes from the values of the reference rows.

        Args:
            reference: The reference rows.
            categorical: The categorical columns.  Numeric columns among
                them keep their values.

        """
        self._values: dict[str, list[Any]] = {}
        for column in categorical:
            series = reference[column]
            if is_numeric_dtype(series):
                continue
            values: list[Any] = sorted(pd.unique(series.dropna()), key=str)
            if series.isna().any():
                values.append(None)
            self._values[column] = values

    @property
    def columns(self) -> list[str]:
        """The coded columns."""
        return list(self._values)

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *df* with codes instead of categories.

        Raises:
            NodeInputError: If a value is not in the reference rows.

        """
        out = df.copy()
        for column, values in self._values.items():
            codes = {
                value: code for code, value in enumerate(values) if value is not None
            }
            missing = out[column].isna()
            unknown = sorted(
                {
                    str(value)
                    for value in out.loc[~missing, column]
                    if value not in codes
                }
            )
            if unknown:
                msg = f"Column '{column}' has values {unknown} that are not in the reference rows."
                raise NodeInputError(msg)
            if missing.any() and None not in values:
                msg = f"Column '{column}' has missing values, but the reference rows have none."
                raise NodeInputError(msg)
            mapped = out[column].map(codes)
            mapped[missing] = len(values) - 1
            out[column] = mapped.astype(np.int64)
        return out

    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *df* with categories instead of codes.

        A code is rounded and clipped to the valid codes first.
        """
        out = df.copy()
        for column, values in self._values.items():
            codes = (
                pd.to_numeric(out[column], errors="coerce")
                .fillna(0)
                .round()
                .clip(0, len(values) - 1)
                .astype(np.int64)
            )
            out[column] = pd.Series(
                [np.nan if values[code] is None else values[code] for code in codes],
                index=out.index,
                dtype=object,
            )
        return out

    def encode_constraints(self, constraints: FeatureConstraints) -> FeatureConstraints:
        """Return *constraints* with codes in the feasible categories."""
        feasible = dict(constraints.feasible_categories)
        for column, allowed in constraints.feasible_categories.items():
            if column in self._values:
                codes = {value: code for code, value in enumerate(self._values[column])}
                feasible[column] = [codes[value] for value in allowed if value in codes]
        return constraints.model_copy(update={"feasible_categories": feasible})


def fill_missing(reference: pd.DataFrame, continuous: Sequence[str]) -> pd.DataFrame:
    """Return a copy of *reference* without missing values.

    A continuous column gets its median, and another column gets its most
    frequent value.  A column with only missing values stays unchanged.
    """
    out = reference.copy()
    for column in out.columns:
        values = out[column]
        if not values.isna().any() or values.isna().all():
            continue
        if column in continuous:
            out[column] = values.fillna(values.median())
        else:
            out[column] = values.fillna(values.mode(dropna=True).iloc[0])
    return out


class CeliaView:
    """The rows that CELIA sees for one sample, and the way back.

    CELIA and the methods that it wraps do not accept missing values, and
    some methods do not accept string categories.  For one sample, the view:

    * **freezes** columns: CELIA does not see them, and every counterfactual
      keeps the value of the sample.  The frozen columns are the
      ``frozen_columns`` of the constraints, the columns that are missing in
      the sample, and the columns that are missing in all reference rows;
    * **fills** the missing values of the reference rows that CELIA sees
      (see :func:`fill_missing`).  The pipeline always gets the real rows;
    * gives CELIA **integer codes** instead of string categories when the
      method needs them (see :class:`CategoryCodes`).
    """

    def __init__(
        self,
        reference: pd.DataFrame,
        sample: pd.DataFrame,
        *,
        continuous: Sequence[str],
        categorical: Sequence[str],
        frozen: Sequence[str] = (),
        code_categories: bool = False,
        allow_missing: bool = True,
    ) -> None:
        """Build the view.

        Args:
            reference: Reference rows at the insertion point.
            sample: The sample, as one row with the reference columns.
            continuous: The continuous columns.
            categorical: The categorical columns.
            frozen: Columns to hide from CELIA in any case.
            code_categories: Give CELIA integer codes for string categories.
            allow_missing: If ``False``, a missing value in the sample or in
                the reference rows raises :class:`NodeInputError`.

        """
        columns = list(reference.columns)
        row = sample.iloc[0]
        missing_in_sample = [column for column in columns if pd.isna(row[column])]
        empty = [column for column in columns if reference[column].isna().all()]
        if not allow_missing and (missing_in_sample or reference.isna().any().any()):
            msg = (
                "The sample or the reference rows have missing values, and "
                "missing_values='error'. Set missing_values='freeze'."
            )
            raise NodeInputError(msg)
        frozen_set = {*frozen, *missing_in_sample, *empty}
        self.frozen: list[str] = [column for column in columns if column in frozen_set]
        self.visible: list[str] = [
            column for column in columns if column not in frozen_set
        ]
        self.continuous = [column for column in continuous if column in self.visible]
        self.categorical = [column for column in categorical if column in self.visible]
        self._columns = columns
        self._sample = sample.loc[:, columns].reset_index(drop=True)
        visible_reference = fill_missing(
            reference.loc[:, self.visible], self.continuous
        )
        self._codes = CategoryCodes(
            visible_reference, self.categorical if code_categories else []
        )
        self.reference = self._codes.encode(visible_reference).reset_index(drop=True)
        """The reference rows that CELIA sees."""
        self.sample = self.to_celia(self._sample)
        """The sample that CELIA sees."""

    def to_celia(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return *df* as CELIA sees it: visible columns, codes."""
        return self._codes.encode(df.loc[:, self.visible])

    def from_celia(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return rows from CELIA as pipeline rows: categories and frozen values back.

        Args:
            df: Rows with the visible columns (other columns are ignored).

        Returns:
            Rows with all the reference columns, in the reference order.

        """
        out = self._codes.decode(df.loc[:, self.visible]).reset_index(drop=True)
        for column in self.frozen:
            out[column] = pd.Series(
                [self._sample.loc[0, column]] * len(out), dtype=object
            )
        return out.loc[:, self._columns]

    def constraints(self, constraints: FeatureConstraints) -> FeatureConstraints:
        """Return *constraints* for CELIA: without the frozen columns, with codes.

        Raises:
            NodeInputError: If a constraint names a column that is not at the
                insertion point.

        """
        named = [
            *constraints.immutable_columns,
            *constraints.feasible_ranges,
            *constraints.feasible_categories,
            *constraints.monotonic_increasing_columns,
            *constraints.frozen_columns,
            *(
                column
                for triple in constraints.correlated_features
                for column in triple[:2]
            ),
        ]
        unknown = sorted({column for column in named if column not in self._columns})
        if unknown:
            msg = (
                f"The constraints name columns {unknown} that are not at the "
                f"insertion point. Columns: {self._columns}."
            )
            raise NodeInputError(msg)
        visible = set(self.visible)
        reduced = FeatureConstraints(
            immutable_columns=[
                c for c in constraints.immutable_columns if c in visible
            ],
            feasible_ranges={
                c: v for c, v in constraints.feasible_ranges.items() if c in visible
            },
            feasible_categories={
                c: v for c, v in constraints.feasible_categories.items() if c in visible
            },
            monotonic_increasing_columns=[
                c for c in constraints.monotonic_increasing_columns if c in visible
            ],
            correlated_features=[
                t
                for t in constraints.correlated_features
                if t[0] in visible and t[1] in visible
            ],
        )
        return self._codes.encode_constraints(reduced)


class ViewPredictor:
    """A predictor for rows from CELIA: it converts them with a :class:`CeliaView` first."""

    def __init__(self, predictor: PipelinePredictor, view: CeliaView) -> None:
        """Store the predictor and the view."""
        self._predictor = predictor
        self._view = view

    def predict_frame(self, rows: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
        """Convert *rows* to pipeline rows and return the pipeline predictions."""
        if isinstance(rows, pd.Series):
            rows = rows.to_frame().T
        elif not isinstance(rows, pd.DataFrame):
            rows = pd.DataFrame(np.asarray(rows), columns=self._view.visible)
        return self._predictor.predict_frame(self._view.from_celia(rows))


def make_celia_model(predictor: PipelinePredictor | ViewPredictor) -> Any:
    """Wrap *predictor* as a ``celia.BaseModel``."""
    return _pipeline_model_class()(predictor)


def split_columns(context: TabularDataContext) -> tuple[list[str], list[str]]:
    """Return the ``(continuous, categorical)`` columns of a context.

    Numerical columns are continuous and categorical columns are
    categorical.  A mixed column follows its dtype.
    """
    continuous: list[str] = []
    categorical: list[str] = []
    for column, dtype, category in zip(
        context.columns, context.dtypes, context.categories, strict=True
    ):
        resolved = (
            infer_category(dtype)
            if category not in (NumericalData, CategoricalData)
            else category
        )
        (continuous if resolved is NumericalData else categorical).append(column)
    return continuous, categorical


def _celia_feasible_values(constraints: FeatureConstraints) -> dict[str, Any] | None:
    """Return the feasible values in the CELIA format.

    CELIA reads a tuple as a ``(min, max)`` range and a list as the allowed
    categories.
    """
    converted: dict[str, Any] = {
        column: (float(low), float(high))
        for column, (low, high) in constraints.feasible_ranges.items()
    }
    converted.update(
        {
            column: list(allowed)
            for column, allowed in constraints.feasible_categories.items()
        }
    )
    return converted or None


def make_celia_data(
    reference: pd.DataFrame,
    predictions: np.ndarray,
    *,
    target_name: str,
    continuous: Sequence[str],
    categorical: Sequence[str],
    constraints: FeatureConstraints,
) -> Any:
    """Build the ``celia.Data`` object of an evaluation.

    The targets are the pipeline predictions for the reference rows, so the
    counterfactual methods explain the model and not the data.

    Args:
        reference: Reference rows at the insertion point.
        predictions: The pipeline prediction for each reference row.
        target_name: Name of the target in CELIA.  It must not be a column.
        continuous: The continuous columns.
        categorical: The categorical columns.
        constraints: The feature constraints.

    Returns:
        A ``celia.Data`` object.

    """
    celia = import_celia()
    return celia.Data(
        data=reference.reset_index(drop=True),
        targets=pd.Series(predictions, name=target_name),
        target_name=target_name,
        continuous_column_names=list(continuous),
        categorical_column_names=list(categorical),
        immutable_column_names=list(constraints.immutable_columns) or None,
        feasible_values=_celia_feasible_values(constraints),
        monotonic_increasing_column_names=list(constraints.monotonic_increasing_columns)
        or None,
        correlated_features=list(constraints.correlated_features) or None,
    )


def regression_explainer_class(method: RegressionMethod) -> type:
    """Return the CELIA regression explainer class of *method*."""
    return getattr(import_celia(), REGRESSION_EXPLAINERS[method])


def counterfactual_rows(result: Any) -> list[pd.DataFrame]:
    """Return the counterfactual rows of each sample in a CELIA result.

    CELIA returns one ``Counterfactual`` or a list of them.
    """
    items = result if isinstance(result, list) else [result]
    return [item.counterfactuals for item in items]
