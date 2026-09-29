"""VarianceFilter transform node for the NodeML Framework.

Removes the numerical features whose variance is at or below a
configurable threshold.  The filter computes the variance of each column
during :meth:`fit` with numpy and keeps the same columns at
:meth:`transform` time.

A threshold of ``0.0`` (the default) removes only the constant columns.
Higher values also remove near-constant, low-information features.

* Input  - a ``(batch, feature)`` **numerical** DataFrame.
* Output - the same DataFrame without the low-variance columns.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field
from ray import tune

from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
)
from nodeml.core.nodes.node import Port
from nodeml.core.nodes.transform.transform import (
    TransformConfig,
    TransformHyperParameters,
    TransformMetadata,
)

from ._column_filter import ColumnFilter, ColumnFilterRunningConfig


class VarianceFilterMetadata(TransformMetadata):
    """Metadata for the VarianceFilter node."""

    node_name: str = "VarianceFilter"
    description: str = (
        "Remove numerical features whose variance is at or below a "
        "configurable threshold.  A threshold of 0 removes only constant columns."
    )


class VarianceFilterRunningConfig(ColumnFilterRunningConfig):
    """Run-time knobs that do not affect the learned parameters."""


class VarianceFilterHyperParameters(TransformHyperParameters):
    """Tuneable hyperparameters for the VarianceFilter."""

    threshold: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "A column must have a variance strictly above this value to "
            "stay. ``0.0`` (default) removes only the constant columns. "
            "Higher values also remove low-information features."
        ),
    )


# Exposed at module level so external tuners can discover the search space.
hyperparameter_space: dict[str, Any] = {
    "threshold": tune.uniform(0.0, 10.0),
}


class VarianceFilterConfig(
    TransformConfig[
        VarianceFilterHyperParameters,
        VarianceFilterRunningConfig,
    ]
):
    """Full configuration for the VarianceFilter node."""

    hyperparameters: VarianceFilterHyperParameters = Field(
        default_factory=VarianceFilterHyperParameters,
        description="Tuneable hyperparameters (threshold).",
    )
    running_config: VarianceFilterRunningConfig = Field(
        default_factory=VarianceFilterRunningConfig,
        description="Run-time options (filtering_columns).",
    )
    in_ports: dict[str, Port] = Field(
        default={
            "input": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch feature",
                desc="Numerical DataFrame to filter by variance.",
            ),
        },
        description="Input ports: 'input' (numerical DataFrame).",
    )
    out_ports: dict[str, Port] = Field(
        default={
            "output": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch _",
                desc=(
                    "DataFrame with low-variance columns removed. "
                    "Feature dimension may shrink."
                ),
            ),
        },
        description="Output ports: 'output' (filtered DataFrame).",
    )


class VarianceFilter(ColumnFilter):
    """Remove numerical features with a variance at or below a threshold.

    During :meth:`fit`, the filter computes the variance of each candidate
    column with ``numpy.nanvar`` (NaN values are ignored).  It stores the
    columns that pass.  :meth:`transform` keeps these columns, in the input
    order.  A column with only NaN values has no variance, so the filter
    removes it.

    Example:
        >>> cfg = VarianceFilterConfig(
        ...     hyperparameters=VarianceFilterHyperParameters(threshold=0.01),
        ... )
        >>> node = VarianceFilter(config=cfg)

    """

    metadata = VarianceFilterMetadata()
    hyperparameter_space = hyperparameter_space

    def _surviving_columns(self, candidates: pd.DataFrame) -> list[str]:
        """Return the candidate columns with a variance > threshold."""
        threshold = self._config.hyperparameters.threshold
        if candidates.empty:
            return []  # No values: no column has a variance.
        values = candidates.to_numpy(dtype=np.float64, na_value=np.nan)
        with warnings.catch_warnings():
            # An all-NaN column gives NaN and a warning. NaN removes the column.
            warnings.simplefilter("ignore", RuntimeWarning)
            variances = np.nanvar(values, axis=0)
            spans = np.nanmax(values, axis=0) - np.nanmin(values, axis=0)
        # Rounding can give a small positive variance to a constant column
        # (for example, 0.1 repeated). A zero span marks a constant column.
        variances[spans == 0] = 0.0
        return [
            col
            for col, var in zip(candidates.columns, variances, strict=True)
            if var > threshold
        ]
