"""Shared category helpers for the encoders of the NodeML Framework.

The encoders compare the real values of a column, not their string form.
An integral float is the same category as the equal integer: a column
stored as ``float64`` at fit (because of NaN) and as ``int64`` at inference
gives the same categories.
"""

from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd

# The code of a missing value or of a category that fit did not see.
UNKNOWN_CODE = -1


def category_key(value: object) -> Hashable:
    """Return the value that the encoders use to compare categories.

    Numpy scalars become Python scalars, and an integral float becomes an
    ``int`` (``1.0`` becomes ``1``).  Other values do not change.

    Args:
        value: A value of a column.  It must not be missing.

    Returns:
        The normalised value.

    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value  # type: ignore[return-value]


def learn_categories(series: pd.Series) -> list[Hashable]:
    """Return the sorted categories of *series*, without missing values.

    The categories are sorted by value when the values are comparable (for
    example, all numbers or all strings).  Otherwise, they are sorted by
    type name, then by string form.

    Args:
        series: A column of the training data.

    Returns:
        The distinct normalised values of *series*.

    """
    keys = {category_key(value) for value in series.dropna().astype(object).unique()}
    try:
        return sorted(keys)  # type: ignore[type-var]
    except TypeError:
        # The values cannot be compared with each other (for example, str and int).
        return sorted(keys, key=lambda key: (type(key).__name__, str(key)))


def encode_codes(series: pd.Series, categories: Sequence[Hashable]) -> np.ndarray:
    """Return the position of each value of *series* in *categories*.

    Args:
        series: A column to encode.
        categories: The categories learned during fit.

    Returns:
        An ``int64`` array.  A missing value, or a value that is not in
        *categories*, gets :data:`UNKNOWN_CODE`.

    """
    positions = {key: code for code, key in enumerate(categories)}
    values = series.astype(object)
    lookup = {
        value: positions.get(category_key(value), UNKNOWN_CODE)
        for value in values.dropna().unique()
    }
    return values.map(lookup).fillna(UNKNOWN_CODE).to_numpy(dtype=np.int64)
