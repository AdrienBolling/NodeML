"""MissingRateFilter transform node for the NodeML Framework.

Removes the features whose rate of missing values is more than a
configurable threshold.  The filter computes the missing rate of each
column during :meth:`fit` and keeps the same columns at :meth:`transform`
time.

* Input  - a ``(batch, feature)`` DataFrame of **mixed** data category.
* Output - the same DataFrame without the columns with a high missing rate.

The missing values are the values that ``pandas.DataFrame.isna`` finds
(``NaN``, ``None``, ``pd.NA`` and ``NaT``), so all dtypes are supported.
"""

from typing import Any

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


class MissingRateFilterMetadata(TransformMetadata):
    """Metadata for the MissingRateFilter node."""

    node_name: str = "MissingRateFilter"
    description: str = (
        "Remove features whose fraction of missing values exceeds a "
        "configurable threshold.  Operates on mixed-category data."
    )


class MissingRateFilterRunningConfig(ColumnFilterRunningConfig):
    """Run-time knobs that do not affect the learned parameters."""


class MissingRateFilterHyperParameters(TransformHyperParameters):
    """Tuneable hyperparameters for the MissingRateFilter."""

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Maximum allowed fraction of missing values per column. "
            "Columns with a missing rate strictly above this value are dropped. "
            "``0.0`` removes any column with at least one missing value; "
            "``1.0`` keeps all columns regardless of missing values."
        ),
    )


# Exposed at module level so external tuners can discover the search space.
hyperparameter_space: dict[str, Any] = {
    "threshold": tune.uniform(0.0, 1.0),
}


class MissingRateFilterConfig(
    TransformConfig[
        MissingRateFilterHyperParameters,
        MissingRateFilterRunningConfig,
    ]
):
    """Full configuration for the MissingRateFilter node."""

    hyperparameters: MissingRateFilterHyperParameters = Field(
        default_factory=MissingRateFilterHyperParameters,
        description="Tuneable hyperparameters (threshold).",
    )
    running_config: MissingRateFilterRunningConfig = Field(
        default_factory=MissingRateFilterRunningConfig,
        description="Run-time options (filtering_columns).",
    )
    in_ports: dict[str, Port] = Field(
        default={
            "input": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.MIXED,
                data_shape="batch feature",
                desc="Input DataFrame (mixed category, may contain missing values).",
            ),
        },
        description="Input ports: 'input' (mixed-category DataFrame).",
    )
    out_ports: dict[str, Port] = Field(
        default={
            "output": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.MIXED,
                data_shape="batch _",
                desc=(
                    "DataFrame with high-missing-rate columns removed. "
                    "Feature dimension may shrink."
                ),
            ),
        },
        description="Output ports: 'output' (filtered DataFrame).",
    )


class MissingRateFilter(ColumnFilter):
    """Remove features whose missing rate exceeds a threshold.

    During :meth:`fit`, the filter computes the missing rate of each
    candidate column with ``pandas.DataFrame.isna``.  It stores the columns
    that pass.  :meth:`transform` keeps these columns, in the input order.

    Example:
        >>> cfg = MissingRateFilterConfig(
        ...     hyperparameters=MissingRateFilterHyperParameters(threshold=0.3),
        ... )
        >>> node = MissingRateFilter(config=cfg)

    """

    metadata = MissingRateFilterMetadata()
    hyperparameter_space = hyperparameter_space

    def _surviving_columns(self, candidates: pd.DataFrame) -> list[str]:
        """Return the candidate columns with a missing rate <= threshold."""
        threshold = self._config.hyperparameters.threshold
        missing_rates = candidates.isna().sum(axis=0) / max(len(candidates), 1)
        return [col for col, rate in missing_rates.items() if rate <= threshold]
