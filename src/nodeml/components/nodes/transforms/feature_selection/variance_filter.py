"""VarianceFilter transform node for the NodeML Framework.

Removes numerical features whose variance falls below a configurable
threshold.  The per-column variance is computed during :meth:`fit` using
numpy and the resulting column mask is reused at :meth:`transform` time.

A threshold of ``0.0`` (the default) removes only constant columns.  Higher
values can be used to filter out near-constant or low-information features.

* Input  - a ``(batch, feature)`` **numerical** DataFrame.
* Output - the same DataFrame with low-variance columns removed.
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
type _VarianceFilterParams = dict[str, list[str]]


class VarianceFilterMetadata(TransformMetadata):
    """Metadata for the VarianceFilter node."""

    node_name: str = "VarianceFilter"
    description: str = (
        "Remove numerical features whose variance falls below a "
        "configurable threshold.  A threshold of 0 removes only constant columns."
    )


class VarianceFilterRunningConfig(TransformRunningConfig):
    """Run-time knobs that do not affect the learned parameters."""

    filtering_columns: list[str] | None = Field(
        default=None,
        description=(
            "Subset of columns to evaluate for variance filtering. "
            "``None`` (default) evaluates all columns. "
            "Columns not in this list are always kept in the output."
        ),
    )


class VarianceFilterHyperParameters(TransformHyperParameters):
    """Tuneable hyperparameters for the VarianceFilter."""

    threshold: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Minimum variance a column must have to be retained. "
            "``0.0`` (default) removes only strictly constant columns. "
            "Higher values are more aggressive and remove low-information features."
        ),
    )


# Exposed at module level so external tuners can discover the search space.
hyperparameter_space: dict[str, Any] = {
    "threshold": tune.uniform(0.0, 10.0),
}


class VarianceFilterConfig(
    TransformConfig[
        VarianceFilterRunningConfig,
        VarianceFilterHyperParameters,
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


class VarianceFilter(
    TransformNode[
        pd.DataFrame,
        TabularDataContext,
        pd.DataFrame,
        TabularDataContext,
        _VarianceFilterParams,
    ]
):
    """Remove numerical features with variance below a threshold.

    During :meth:`fit`, computes the variance of each candidate column
    using ``numpy.nanvar`` (ignoring NaN values) and stores the list of
    columns that pass.  During :meth:`transform`, the stored list is used
    to subset the DataFrame and its context.

    Example:
    -------
    >>> cfg = VarianceFilterConfig(
    ...     hyperparameters=VarianceFilterHyperParameters(threshold=0.01),
    ... )
    >>> node = VarianceFilter(config=cfg)

    """

    metadata = VarianceFilterMetadata()
    hyperparameter_space = hyperparameter_space

    def __init__(self, *, config: VarianceFilterConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config
        self._params: _VarianceFilterParams = {"columns_to_keep": []}
        self._fitted = False

    # --- TransformNode interface ------------------------------------------

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Identify columns whose variance meets the threshold.

        Parameters
        ----------
        data:
            Must contain key ``"input"``.

        """
        df, _ = data["input"]
        candidates = filter_columns(df, self._config.running_config.filtering_columns)
        threshold = self._config.hyperparameters.threshold

        # Compute variance using numpy, ignoring NaN values.
        variances = np.nanvar(
            candidates.to_numpy(dtype=np.float64, na_value=np.nan), axis=0
        )

        surviving_cols = [
            col
            for col, var in zip(candidates.columns, variances, strict=True)
            if var >= threshold
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

    def get_params(self) -> _VarianceFilterParams:
        """Return the list of columns that survived filtering."""
        return self._params

    def set_params(self, params: _VarianceFilterParams) -> None:
        """Restore a previously fitted column list (checkpointing)."""
        self._params = params
        self._fitted = True
