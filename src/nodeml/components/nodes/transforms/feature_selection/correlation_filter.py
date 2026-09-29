"""CorrelationFilter transform node for the NodeML Framework.

Removes highly correlated features by computing a pair-wise correlation
matrix during :meth:`fit` and greedily dropping one column from every pair
whose absolute correlation exceeds a configurable threshold.

Greedy strategy (order-dependent!)
----------------------------------
The filter iterates candidate columns left-to-right. For every pair
``(i, j)`` with ``i < j`` whose ``|corr| > threshold``, column *j* is
marked for removal (provided column *i* has not already been marked).

This means the **input column order directly determines which column of a
correlated pair survives**: earlier columns are retained, later columns
are dropped. Swapping the order of two highly-correlated columns in the
input DataFrame will swap which one the filter keeps.

Preferred columns
-----------------
To override this purely positional tie-breaking, the running config
exposes ``preferred_columns`` — columns that should be kept in priority
whenever possible. Internally the candidate set is reordered so that
preferred columns come first (in their original relative order), then the
rest (also in original relative order). The greedy pass then naturally
keeps preferred columns over non-preferred ones in every correlated pair.

When two preferred columns correlate with each other, the positional rule
still applies within the preferred group (the earlier one survives).
The final output DataFrame preserves the **original** column order of the
input — reordering is only used internally for drop selection.

* Input  - a ``(batch, feature)`` **numerical** DataFrame.
* Output - the same DataFrame with redundant features removed.

Implemented with numpy (``np.corrcoef``) for Pearson, or delegates to
``scipy.stats.spearmanr`` / ``scipy.stats.kendalltau`` for rank-based
methods.
"""

from copy import deepcopy
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from pydantic import Field
from ray import tune
from scipy.stats import kendalltau, spearmanr

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
type _CorrelationFilterParams = dict[str, list[str]]

# A correlation needs at least two paired observations.
_MIN_ROWS_FOR_CORRELATION = 2


class CorrelationFilterMetadata(TransformMetadata):
    """Metadata for the CorrelationFilter node."""

    node_name: str = "CorrelationFilter"
    description: str = (
        "Remove highly correlated numerical features. "
        "Keeps only one column from every correlated pair above the threshold."
    )


class CorrelationFilterRunningConfig(TransformRunningConfig):
    """Run-time knobs that do not affect the learned parameters."""

    filtering_columns: list[str] | None = Field(
        default=None,
        description=(
            "Subset of columns to evaluate for correlation filtering. "
            "``None`` (default) evaluates all columns. "
            "Columns not in this list are always kept in the output."
        ),
    )
    preferred_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Columns to keep in priority when resolving correlated pairs. "
            "Because the greedy drop strategy is order-dependent (later "
            "columns are dropped), listing a column here moves it to the "
            "front of the internal drop-selection order so it survives "
            "over non-preferred correlates. Unknown names (not present in "
            "the candidate set) are silently ignored. The output DataFrame "
            "still follows the original input column order."
        ),
    )


class CorrelationFilterHyperParameters(TransformHyperParameters):
    """Tuneable hyperparameters for the CorrelationFilter."""

    threshold: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description=(
            "Maximum allowed absolute correlation between any pair of features. "
            "When a pair exceeds this value the second column (in column order) "
            "is dropped. ``0.95`` is a common default; lower values are more "
            "aggressive."
        ),
    )
    method: Literal["pearson", "spearman", "kendall"] = Field(
        default="pearson",
        description=(
            "Correlation method. "
            "``'pearson'`` measures linear correlation (fast, via numpy). "
            "``'spearman'`` measures monotonic correlation (rank-based). "
            "``'kendall'`` measures ordinal association (rank-based, slower)."
        ),
    )


# Exposed at module level so external tuners can discover the search space.
hyperparameter_space: dict[str, Any] = {
    "threshold": tune.uniform(0.5, 1.0),
    "method": tune.choice(["pearson", "spearman", "kendall"]),
}


class CorrelationFilterConfig(
    TransformConfig[
        CorrelationFilterRunningConfig,
        CorrelationFilterHyperParameters,
    ]
):
    """Full configuration for the CorrelationFilter node."""

    hyperparameters: CorrelationFilterHyperParameters = Field(
        default_factory=CorrelationFilterHyperParameters,
        description="Tuneable hyperparameters (threshold, method).",
    )
    running_config: CorrelationFilterRunningConfig = Field(
        default_factory=CorrelationFilterRunningConfig,
        description="Run-time options (filtering_columns, preferred_columns).",
    )
    in_ports: dict[str, Port] = Field(
        default={
            "input": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch feature",
                desc="Numerical DataFrame to filter by correlation.",
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
                    "DataFrame with highly correlated features removed. "
                    "Feature dimension may shrink."
                ),
            ),
        },
        description="Output ports: 'output' (filtered DataFrame).",
    )


class CorrelationFilter(
    TransformNode[
        pd.DataFrame,
        TabularDataContext,
        pd.DataFrame,
        TabularDataContext,
        _CorrelationFilterParams,
    ]
):
    """Remove highly correlated numerical features.

    During :meth:`fit`, computes the pair-wise correlation matrix and
    greedily marks columns for removal when their absolute correlation
    with an already-kept column exceeds the threshold.

    Tie-breaking
    ------------
    The greedy pass is **order-dependent**: for every correlated pair the
    *earlier* column in the iteration order survives and the *later* one
    is dropped. By default that iteration order is the input column order,
    so input column order directly determines which column survives a
    correlated pair.

    Setting ``running_config.preferred_columns`` moves those names to the
    front of the internal iteration order (preserving their relative
    input order), so they are kept over non-preferred correlates. The
    output DataFrame still follows the original input column order.

    Example:
    -------
    >>> cfg = CorrelationFilterConfig(
    ...     hyperparameters=CorrelationFilterHyperParameters(
    ...         threshold=0.9, method="spearman"
    ...     ),
    ...     running_config=CorrelationFilterRunningConfig(
    ...         preferred_columns=["target_feature"],
    ...     ),
    ... )
    >>> node = CorrelationFilter(config=cfg)

    """

    metadata = CorrelationFilterMetadata()
    hyperparameter_space = hyperparameter_space

    def __init__(self, *, config: CorrelationFilterConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config
        self._params: _CorrelationFilterParams = {"columns_to_keep": []}
        self._fitted = False

    # --- TransformNode interface ------------------------------------------

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Compute the correlation matrix and identify redundant columns.

        Parameters
        ----------
        data:
            Must contain key ``"input"``.

        """
        df, _ = data["input"]
        candidates = filter_columns(df, self._config.running_config.filtering_columns)
        threshold = self._config.hyperparameters.threshold
        method = self._config.hyperparameters.method

        original_candidate_cols = candidates.columns.tolist()
        preference_ordered_cols = self._apply_preference(
            original_candidate_cols,
            self._config.running_config.preferred_columns,
        )
        # Reorder candidates so preferred columns come first; the greedy
        # pass keeps earlier columns, so preferred columns survive any
        # correlated pair involving a non-preferred column.
        reordered = candidates[preference_ordered_cols]

        corr_matrix = self._compute_correlation(reordered, method)
        cols_to_drop = self._greedy_drop(
            preference_ordered_cols, corr_matrix, threshold
        )

        # Output column order follows the ORIGINAL input order, not the
        # preference-reordered one used for drop selection.
        surviving_cols = [c for c in original_candidate_cols if c not in cols_to_drop]

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

    def get_params(self) -> _CorrelationFilterParams:
        """Return the list of columns that survived filtering."""
        return self._params

    def set_params(self, params: _CorrelationFilterParams) -> None:
        """Restore a previously fitted column list (checkpointing)."""
        self._params = params
        self._fitted = True

    # --- Private helpers --------------------------------------------------

    @staticmethod
    def _apply_preference(
        candidate_cols: list[str],
        preferred: list[str],
    ) -> list[str]:
        """Return *candidate_cols* reordered with preferred names first.

        Preserves the original relative order inside each group. Names in
        *preferred* that are not present in *candidate_cols* are ignored.
        """
        if not preferred:
            return list(candidate_cols)
        preferred_set = set(preferred)
        head = [c for c in candidate_cols if c in preferred_set]
        tail = [c for c in candidate_cols if c not in preferred_set]
        return head + tail

    @staticmethod
    def _compute_correlation(df: pd.DataFrame, method: str) -> np.ndarray:
        """Return an ``(n_features, n_features)`` absolute correlation matrix."""
        arr = df.to_numpy(dtype=np.float64, na_value=np.nan)

        if method == "pearson":
            # Drop rows with any NaN for corrcoef (it does not handle NaN).
            mask = ~np.isnan(arr).any(axis=1)
            clean = arr[mask]
            if clean.shape[0] < _MIN_ROWS_FOR_CORRELATION:
                return np.zeros((arr.shape[1], arr.shape[1]))
            return np.abs(np.corrcoef(clean, rowvar=False))

        if method == "spearman":
            corr, _ = spearmanr(arr, nan_policy="omit")
            # spearmanr returns a scalar when n_features == 1.
            corr = np.atleast_2d(corr)
            return np.abs(corr)

        # The remaining method is "kendall".

        n = arr.shape[1]
        corr = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                mask = ~(np.isnan(arr[:, i]) | np.isnan(arr[:, j]))
                if mask.sum() < _MIN_ROWS_FOR_CORRELATION:
                    corr[i, j] = corr[j, i] = 0.0
                else:
                    tau, _ = kendalltau(arr[mask, i], arr[mask, j])
                    corr[i, j] = corr[j, i] = abs(tau)
        return corr

    @staticmethod
    def _greedy_drop(
        columns: list[str],
        corr_matrix: np.ndarray,
        threshold: float,
    ) -> set[str]:
        """Greedily select columns to drop from a correlation matrix.

        Iterates in column order.  For every pair ``(i, j)`` with ``i < j``
        where ``|corr| > threshold``, column *j* is marked for removal
        (provided column *i* has not already been marked).
        """
        n = len(columns)
        to_drop: set[int] = set()

        for i in range(n):
            if i in to_drop:
                continue
            for j in range(i + 1, n):
                if j in to_drop:
                    continue
                if corr_matrix[i, j] > threshold:
                    to_drop.add(j)

        return {columns[idx] for idx in to_drop}
