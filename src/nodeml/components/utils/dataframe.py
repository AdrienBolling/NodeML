"""DataFrame utility functions for the NodeML Framework."""

import pandas as pd


def filter_columns(
    data: pd.DataFrame,
    requested_columns: list[str] | None,
) -> pd.DataFrame:
    """Return a view of *data* restricted to *requested_columns*.

    Parameters
    ----------
    data:
        Source DataFrame.
    requested_columns:
        Columns to keep.  ``None`` means "keep all columns" and returns
        *data* unchanged.

    Raises
    ------
    ValueError
        If any of *requested_columns* are absent from *data*.

    """
    if requested_columns is None:
        return data

    missing = [col for col in requested_columns if col not in data.columns]
    if missing:
        msg = f"Requested columns not found in DataFrame: {missing}"
        raise ValueError(msg)

    return data[requested_columns]


def filter_dtypes(
    data: pd.DataFrame,
    requested_dtypes: list[str] | None,
) -> pd.DataFrame:
    """Return a view of *data* restricted to columns whose dtype matches *requested_dtypes*.

    Parameters
    ----------
    data:
        Source DataFrame.
    requested_dtypes:
        ``pandas.DataFrame.select_dtypes`` include-spec (e.g. ``["number"]``).
        ``None`` means "keep all columns" and returns *data* unchanged.

    Raises
    ------
    ValueError
        If no columns satisfy the dtype filter.

    """
    if requested_dtypes is None:
        return data

    selected = data.select_dtypes(include=requested_dtypes).columns.tolist()
    if not selected:
        msg = (
            f"No columns in the DataFrame have the requested dtypes: {requested_dtypes}"
        )
        raise ValueError(msg)

    return data[selected]
