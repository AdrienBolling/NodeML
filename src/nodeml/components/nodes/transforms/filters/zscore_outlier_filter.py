"""Z-Score outlier filter transform node for the NodeML Framework.

Detects outliers as values whose absolute Z-score exceeds a configurable
threshold.  Two strategies are available once an outlier is detected:

* ``"remove"`` - drop the entire row (default).
* ``"cap"``    - clip the value to the ±threshold boundary.

Per-column **mean** and **std** are learned during :meth:`fit` and reused
at :meth:`transform` time.  Columns with zero standard deviation are skipped
(Z-score is always 0, so they never produce outliers).
"""

from typing import Any, Literal, cast

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

# Serialisable params type: two nested dicts keyed by column name.
type _ZScoreParams = dict[str, dict[str, float]]


class ZScoreOutlierFilterMetadata(TransformMetadata):
    """Metadata for the ZScoreOutlierFilter node."""

    node_name: str = "ZScoreOutlierFilter"
    description: str = (
        "Detect and handle outliers using the Z-score method. "
        "A value is an outlier if |z| > threshold."
    )


class ZScoreOutlierFilterRunningConfig(TransformRunningConfig):
    """Run-time knobs that do not affect the learned parameters."""

    filtering_columns: list[str] | None = Field(
        default=None,
        description=(
            "Subset of columns to apply the filter to. "
            "``None`` (default) applies the filter to all columns."
        ),
    )


class ZScoreOutlierFilterHyperParameters(TransformHyperParameters):
    """Tuneable hyperparameters for the ZScoreOutlierFilter."""

    zscore_cutoff: float = Field(
        default=3.0,
        gt=0.0,
        description=(
            "Absolute Z-score above which a value is considered an outlier. "
            "The standard choice is 3.0 (≈99.7 % of a Gaussian is within ±3 sigma). "
            "Lower values are more aggressive."
        ),
    )
    threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Proportion of features that must be flagged as outliers "
            "for a row to be removed (``strategy='remove'`` only). "
            "A row is dropped when ``outlier_features / total_features > threshold``. "
            "``0.0`` (default) drops any row with at least one outlier feature; "
            "``1.0`` drops only rows where every feature is an outlier."
        ),
    )
    strategy: Literal["remove", "cap"] = Field(
        default="remove",
        description=(
            "How to handle detected outliers. "
            "``'remove'`` drops rows whose outlier-feature proportion exceeds ``threshold``. "
            "``'cap'`` clips outlier values to ``mean ± zscore_cutoff * std`` (per-value, ignores ``threshold``)."
        ),
    )


# Exposed at module level so external tuners can discover the search space.
hyperparameter_space: dict[str, Any] = {
    "zscore_cutoff": tune.uniform(0.5, 10.0),
    "threshold": tune.uniform(0.0, 1.0),
    "strategy": tune.choice(["remove", "cap"]),
}


class ZScoreOutlierFilterConfig(
    TransformConfig[
        ZScoreOutlierFilterRunningConfig,
        ZScoreOutlierFilterHyperParameters,
    ]
):
    """Full configuration for the ZScoreOutlierFilter node."""

    hyperparameters: ZScoreOutlierFilterHyperParameters = Field(
        default_factory=ZScoreOutlierFilterHyperParameters,
        description="Tuneable hyperparameters (zscore_cutoff, threshold, strategy).",
    )
    running_config: ZScoreOutlierFilterRunningConfig = Field(
        default_factory=ZScoreOutlierFilterRunningConfig,
        description="Run-time options (filtering_columns).",
    )
    in_ports: dict[str, Port] = Field(
        default={
            "input": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch feature",
                desc="Numerical DataFrame to filter.",
            ),
            "target": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch _",
                desc="Target DataFrame; rows are kept in sync with 'input'.",
                mode=["training", "evaluation"],
            ),
            "sliced": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.MIXED,
                data_shape="batch _",
                desc="Auxiliary DataFrame; rows are kept in sync with 'input'.",
            ),
        },
        description="Input ports: 'input', 'target' (training/evaluation), 'sliced' (any DataFrame to keep in sync).",
    )
    out_ports: dict[str, Port] = Field(
        default={
            "output": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="_ feature",
                desc=(
                    "DataFrame with outliers handled. "
                    "Batch dimension may shrink with strategy='remove'."
                ),
            ),
            "target": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="_ _",
                desc="Target DataFrame with the same rows removed as 'output'.",
                mode=["training", "evaluation"],
            ),
            "sliced": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.MIXED,
                data_shape="_ _",
                desc="Auxiliary DataFrame with the same rows removed as 'output'.",
            ),
        },
        description="Output ports: 'output', 'target' (training/evaluation), 'sliced' (synced auxiliary).",
    )


class ZScoreOutlierFilter(
    TransformNode[
        pd.DataFrame,
        TabularDataContext,
        pd.DataFrame,
        TabularDataContext,
        _ZScoreParams,
    ]
):
    """Z-Score outlier filter.

    Learns per-column ``mean`` and ``std`` on the training split, then uses
    those statistics to detect and handle outliers at transform time.

    Example:
    -------
    >>> cfg = ZScoreOutlierFilterConfig(
    ...     hyperparameters=ZScoreOutlierFilterHyperParameters(zscore_cutoff=2.5),
    ... )
    >>> node = ZScoreOutlierFilter(config=cfg)

    """

    metadata = ZScoreOutlierFilterMetadata()
    hyperparameter_space = hyperparameter_space

    def __init__(self, *, config: ZScoreOutlierFilterConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config
        self._params: _ZScoreParams = {}
        # Must be set so node_transform blocks until fit has been called.
        self._fitted = False

    # --- TransformNode interface ------------------------------------------

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Learn per-column mean and standard deviation from *data["input"]*."""
        df, _ = data["input"]
        target = self._select_columns(df)

        means = target.mean()
        # Replace zero std with NaN so those columns are silently skipped.
        stds = target.std().replace(0.0, np.nan)

        self._params = {
            "mean": {col: float(means[col]) for col in target.columns},
            "std": {col: float(stds[col]) for col in target.columns},
        }

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Detect and handle outliers using the statistics learned during fit.

        Parameters
        ----------
        data:
            Must contain key ``"input"``.  May contain ``"target"``
            (training / evaluation); if present, the same row removal
            is applied to the target DataFrame.

        """
        df, ctx = data["input"]
        outlier_mask = self._detect_outliers(df)
        result = self._apply_strategy(df, outlier_mask)
        outputs: dict[str, tuple[pd.DataFrame, TabularDataContext]] = {
            "output": (result, ctx),
        }
        is_remove = self._config.hyperparameters.strategy == "remove"
        keep = ~self._row_exceeds_threshold(outlier_mask) if is_remove else None
        for port in ("target", "sliced"):
            if port in data:
                port_df, port_ctx = data[port]
                if keep is not None:
                    port_df = cast("pd.DataFrame", port_df.loc[keep]).reset_index(
                        drop=True
                    )
                outputs[port] = (port_df, port_ctx)
        return outputs

    def get_params(self) -> _ZScoreParams:
        """Return the fitted per-column statistics (``mean`` and ``std``)."""
        return self._params

    def set_params(self, params: _ZScoreParams) -> None:
        """Restore previously fitted statistics (used for checkpointing)."""
        self._params = params
        self._fitted = True

    # --- Private helpers --------------------------------------------------

    def _select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only the columns subject to filtering."""
        return filter_columns(df, self._config.running_config.filtering_columns)

    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a boolean DataFrame: ``True`` where a value is an outlier."""
        cols = list(self._params["mean"].keys())
        target = df[cols]

        mean = pd.Series(self._params["mean"])
        std = pd.Series(self._params["std"])

        # Columns with NaN std are treated as having Z-score 0 (no outliers).
        z_scores = target.sub(mean, axis=1).div(std, axis=1).fillna(0.0)
        return z_scores.abs().gt(self._config.hyperparameters.zscore_cutoff)

    def _row_exceeds_threshold(self, outlier_mask: pd.DataFrame) -> pd.Series:
        """Return a boolean Series: ``True`` where a row should be removed.

        A row is flagged when the proportion of outlier features strictly
        exceeds the configured ``threshold``.
        """
        num_cols = outlier_mask.shape[1]
        if num_cols == 0:
            return pd.Series(data=False, index=outlier_mask.index)
        proportion = outlier_mask.sum(axis=1) / num_cols
        return proportion > self._config.hyperparameters.threshold

    def _apply_strategy(
        self,
        df: pd.DataFrame,
        outlier_mask: pd.DataFrame,
    ) -> pd.DataFrame:
        """Handle detected outliers according to the configured strategy."""
        if self._config.hyperparameters.strategy == "remove":
            mask = self._row_exceeds_threshold(outlier_mask)
            filtered = cast("pd.DataFrame", df.loc[~mask])
            return filtered.reset_index(drop=True)

        # The remaining strategy is "cap".
        result = df.copy()
        k = self._config.hyperparameters.zscore_cutoff
        for col in outlier_mask.columns:
            mu = self._params["mean"][col]
            sig = self._params["std"].get(col, np.nan)
            if np.isnan(sig):
                continue  # Column with zero std — nothing to cap.
            result[col] = result[col].clip(lower=mu - k * sig, upper=mu + k * sig)
        return result
