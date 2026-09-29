"""DataFrame utility functions for the NodeML Framework."""

from collections.abc import Iterable

import pandas as pd

from nodeml.core.common.exceptions import NodeInputError


def filter_columns(
    data: pd.DataFrame,
    requested_columns: list[str] | None,
) -> pd.DataFrame:
    """Return a view of *data* restricted to *requested_columns*.

    Args:
        data: Source DataFrame.
        requested_columns: Columns to keep.  ``None`` keeps all columns and
            returns *data* unchanged.

    Returns:
        The requested columns of *data*, in the requested order.

    Raises:
        NodeInputError: If a requested column is not in *data*.

    """
    if requested_columns is None:
        return data

    missing = [col for col in requested_columns if col not in data.columns]
    if missing:
        msg = f"Requested columns not found in DataFrame: {missing}"
        raise NodeInputError(msg)

    return data[requested_columns]


def fill_missing(series: pd.Series, value: object, *, fallback_dtype: str) -> pd.Series:
    """Fill the missing values of *series* with *value*, and keep the dtype.

    * A pandas ``Categorical`` column gets *value* as a new category when
      it is not one of its categories.
    * When *value* does not fit the dtype (for example, ``2.5`` in an
      ``Int64`` column), the column is cast to *fallback_dtype* first.

    Args:
        series: The column to fill.
        value: The fill value.
        fallback_dtype: The dtype to use when *value* does not fit the dtype
            of *series* (for example ``"float64"`` or ``"object"``).

    Returns:
        The filled column.  It is *series* itself when no value is missing.

    """
    if not series.isna().any():
        return series
    if isinstance(series.dtype, pd.CategoricalDtype):
        if value not in series.cat.categories:
            series = series.cat.add_categories([value])
        return series.fillna(value)
    try:
        return series.fillna(value)
    except (TypeError, ValueError):
        # The value does not fit the dtype, for example 2.5 in Int64.
        return series.astype(fallback_dtype).fillna(value)


def check_fitted_columns(
    fitted: Iterable[object],
    columns: Iterable[object],
    *,
    node_name: str,
    allow_extra: bool = False,
) -> None:
    """Check that the input of a fitted node has the fitted columns.

    The column order is not checked.

    Args:
        fitted: The columns that the node saw during fit.
        columns: The columns of the input DataFrame.
        node_name: The name of the node, for the error message.
        allow_extra: If ``True``, the input can have columns that the node
            did not see during fit.

    Raises:
        NodeInputError: If a fitted column is not in *columns*, or if
            *allow_extra* is ``False`` and *columns* has a column that is
            not in *fitted*.

    """
    fitted = list(fitted)
    columns = list(columns)
    fitted_set = set(fitted)
    column_set = set(columns)
    missing = [col for col in fitted if col not in column_set]
    unknown = [] if allow_extra else [col for col in columns if col not in fitted_set]
    if not missing and not unknown:
        return
    problems = []
    if missing:
        problems.append(f"the input does not have the fitted columns {missing}")
    if unknown:
        problems.append(f"the input has columns {unknown} that fit did not see")
    msg = (
        f"{node_name}: {' and '.join(problems)}. "
        f"Fitted columns: {fitted}. Input columns: {columns}."
    )
    raise NodeInputError(msg)
