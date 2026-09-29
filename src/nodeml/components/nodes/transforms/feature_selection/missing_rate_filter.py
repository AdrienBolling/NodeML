"""MissingRateFilter transform node for the NodeML Framework.

Removes features whose missing-value rate exceeds a configurable threshold.
The missing rate per column is computed during :meth:`fit` and the resulting
column mask is reused at :meth:`transform` time.

* Input  - a ``(batch, feature)`` DataFrame of **mixed** data category.
* Output - the same DataFrame with high-missing-rate columns removed.

Implemented with numpy for performance (``np.isnan`` on the underlying
array after coercing to float where possible, with a fallback to pandas
``isna`` for object columns).
"""

from copy import deepcopy
from typing import Any, cast

import numpy as np
import pandas as pd
from pydantic import Field
from ray import tune

from nodeml.components.utils.dataframe import filter_columns
from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    TabularDataContext,
)
from nodeml.core.nodes.node import Port
from nodeml.core.nodes.transform.transform import (
    TransformConfig,
    TransformHyperParameters,
    TransformMetadata,
    TransformNode,
    TransformRunningConfig,
)

# Serialisable params: list of column names that survived the filter.
type _MissingRateParams = dict[str, list[str]]


class MissingRateFilterMetadata(TransformMetadata):
    """Metadata for the MissingRateFilter node."""

    node_name: str = "MissingRateFilter"
    description: str = (
        "Remove features whose fraction of missing values exceeds a "
        "configurable threshold.  Operates on mixed-category data."
    )


class MissingRateFilterRunningConfig(TransformRunningConfig):
    """Run-time knobs that do not affect the learned parameters."""

    filtering_columns: list[str] | None = Field(
        default=None,
        description=(
            "Subset of columns to evaluate for missing-rate filtering. "
            "``None`` (default) evaluates all columns. "
            "Columns not in this list are always kept in the output."
        ),
    )


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
        MissingRateFilterRunningConfig,
        MissingRateFilterHyperParameters,
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


class MissingRateFilter(
    TransformNode[
        pd.DataFrame,
        TabularDataContext,
        pd.DataFrame,
        TabularDataContext,
        _MissingRateParams,
    ]
):
    """Remove features whose missing rate exceeds a threshold.

    During :meth:`fit`, computes the missing rate of each candidate column
    using ``numpy`` and stores the list of columns that pass.  During
    :meth:`transform`, the stored column list is used to subset the
    DataFrame and its context.

    Example:
    -------
    >>> cfg = MissingRateFilterConfig(
    ...     hyperparameters=MissingRateFilterHyperParameters(threshold=0.3),
    ... )
    >>> node = MissingRateFilter(config=cfg)

    """

    metadata = MissingRateFilterMetadata()
    hyperparameter_space = hyperparameter_space

    def __init__(self, *, config: MissingRateFilterConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config
        self._params: _MissingRateParams = {"columns_to_keep": []}
        self._fitted = False

    # --- TransformNode interface ------------------------------------------

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Identify columns whose missing rate is within the threshold.

        Parameters
        ----------
        data:
            Must contain key ``"input"``.

        """
        df, _ = data["input"]
        candidates = filter_columns(df, self._config.running_config.filtering_columns)
        threshold = self._config.hyperparameters.threshold
        n_rows = len(df)

        # Compute missing rates using numpy for performance.
        missing_rates = np.asarray(candidates.isna().sum(axis=0)) / max(n_rows, 1)

        surviving_cols = [
            col
            for col, rate in zip(candidates.columns, missing_rates, strict=True)
            if rate <= threshold
        ]

        # Columns not in the candidate set are always kept.
        non_candidate_cols = [c for c in df.columns if c not in candidates.columns]
        self._params = {
            "columns_to_keep": non_candidate_cols + surviving_cols,
        }

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Subset the DataFrame to the columns identified during fit.

        Parameters
        ----------
        data:
            Must contain key ``"input"``.

        """
        df, ctx = data["input"]
        keep = self._params["columns_to_keep"]
        dropped = [c for c in df.columns if c not in set(keep)]

        out_ctx = deepcopy(ctx)
        out_ctx.remove_columns(dropped)
        return {"output": (cast("pd.DataFrame", df[keep]), out_ctx)}

    def get_params(self) -> _MissingRateParams:
        """Return the list of columns that survived filtering."""
        return self._params

    def set_params(self, params: _MissingRateParams) -> None:
        """Restore a previously fitted column list (checkpointing)."""
        self._params = params
        self._fitted = True
