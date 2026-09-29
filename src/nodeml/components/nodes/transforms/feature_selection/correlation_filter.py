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

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field
from ray import tune
from scipy.stats import kendalltau, spearmanr

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

# A correlation needs at least two paired observations.
_MIN_ROWS_FOR_CORRELATION = 2
# A correlation matrix needs at least two columns.
_MIN_COLUMNS_FOR_CORRELATION = 2


class CorrelationFilterMetadata(TransformMetadata):
    """Metadata for the CorrelationFilter node."""

    node_name: str = "CorrelationFilter"
    description: str = (
        "Remove highly correlated numerical features. "
        "Keeps only one column from every correlated pair above the threshold."
    )


class CorrelationFilterRunningConfig(ColumnFilterRunningConfig):
    """Run-time knobs that do not affect the learned parameters."""

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
        CorrelationFilterHyperParameters,
        CorrelationFilterRunningConfig,
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


class CorrelationFilter(ColumnFilter):
    """Remove highly correlated numerical features.

    During :meth:`fit`, the filter computes the pair-wise correlation
    matrix.  It greedily marks a column for removal when its absolute
    correlation with a kept column is more than the threshold.

    The greedy pass depends on the column order.  For each correlated
    pair, the *earlier* column survives and the *later* column is removed.
    By default, this order is the input column order.

    ``running_config.preferred_columns`` moves these names to the front of
    the internal order, with their relative input order.  Thus the filter
    keeps them over non-preferred correlated columns.  The output always
    follows the input column order.  A pair without a defined correlation
    (for example, a pair with a constant column) is not correlated.

    Example:
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

    def _surviving_columns(self, candidates: pd.DataFrame) -> list[str]:
        """Return the candidate columns that no kept column correlates with."""
        original_cols = candidates.columns.tolist()
        preference_ordered_cols = self._apply_preference(
            original_cols,
            self._config.running_config.preferred_columns,
        )
        # Reorder the candidates so that preferred columns come first. The
        # greedy pass keeps earlier columns, so preferred columns survive
        # each correlated pair with a non-preferred column.
        reordered = candidates[preference_ordered_cols]
        corr_matrix = self._compute_correlation(
            reordered, self._config.hyperparameters.method
        )
        cols_to_drop = self._greedy_drop(
            preference_ordered_cols,
            corr_matrix,
            self._config.hyperparameters.threshold,
        )
        return [col for col in original_cols if col not in cols_to_drop]

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
        """Return the ``(n_features, n_features)`` absolute correlation matrix.

        A pair without a defined correlation (for example, a pair with a
        constant column) gets 0, so the filter keeps both columns.
        """
        arr = df.to_numpy(dtype=np.float64, na_value=np.nan)
        n = arr.shape[1]
        if n < _MIN_COLUMNS_FOR_CORRELATION:
            return np.ones((n, n))  # One column or none: no pair to compare.

        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            # Constant columns make numpy and scipy warn; they get 0 below.
            warnings.simplefilter("ignore", RuntimeWarning)
            if method == "pearson":
                corr = CorrelationFilter._pearson(arr)
            elif method == "spearman":
                corr, _ = spearmanr(arr, nan_policy="omit")
                if np.ndim(corr) == 0:
                    # spearmanr returns one value for two columns.
                    corr = np.array([[1.0, corr], [corr, 1.0]])
            else:  # The remaining method is "kendall".
                corr = CorrelationFilter._kendall(arr)
        return np.nan_to_num(np.abs(np.asarray(corr, dtype=np.float64)), nan=0.0)

    @staticmethod
    def _pearson(arr: np.ndarray) -> np.ndarray:
        """Return the Pearson correlation matrix of the rows without NaN."""
        clean = arr[~np.isnan(arr).any(axis=1)]
        if clean.shape[0] < _MIN_ROWS_FOR_CORRELATION:
            return np.zeros((arr.shape[1], arr.shape[1]))
        return np.corrcoef(clean, rowvar=False)

    @staticmethod
    def _kendall(arr: np.ndarray) -> np.ndarray:
        """Return the Kendall tau matrix, with pairwise removal of NaN."""
        n = arr.shape[1]
        corr = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                mask = ~(np.isnan(arr[:, i]) | np.isnan(arr[:, j]))
                if mask.sum() < _MIN_ROWS_FOR_CORRELATION:
                    corr[i, j] = corr[j, i] = 0.0
                else:
                    tau, _ = kendalltau(arr[mask, i], arr[mask, j])
                    corr[i, j] = corr[j, i] = tau
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
