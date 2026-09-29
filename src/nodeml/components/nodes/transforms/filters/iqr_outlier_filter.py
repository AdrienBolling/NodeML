"""IQR (Inter-Quartile Range) outlier filter transform node for the NodeML Framework.

Detects outliers as values outside the Tukey fences
``[Q1 - k·IQR, Q3 + k·IQR]`` for a configurable multiplier *k*.  Two
strategies are available:

* ``"remove"`` - drop the entire row (default).
* ``"cap"``    - clip the value to the fence boundary.

Per-column **Q1**, **Q3**, and **IQR** are learned during :meth:`fit` and
reused at :meth:`transform` time.

The standard Tukey fence uses ``k=1.5`` (mild outliers) or ``k=3.0``
(extreme outliers).
"""

from typing import Any, Literal, cast

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

# Serialisable params type: nested dicts keyed by statistic name → column name.
type _IQRParams = dict[str, dict[str, float]]


class IQROutlierFilterMetadata(TransformMetadata):
    """Metadata for the IQROutlierFilter node."""

    node_name: str = "IQROutlierFilter"
    description: str = (
        "Detect and handle outliers using the IQR (Tukey fence) method. "
        "A value is an outlier if it falls outside [Q1 - k·IQR, Q3 + k·IQR]."
    )


class IQROutlierFilterRunningConfig(TransformRunningConfig):
    """Run-time knobs that do not affect the learned parameters."""

    filtering_columns: list[str] | None = Field(
        default=None,
        description=(
            "Subset of columns to apply the filter to. "
            "``None`` (default) applies the filter to all columns."
        ),
    )


class IQROutlierFilterHyperParameters(TransformHyperParameters):
    """Tuneable hyperparameters for the IQROutlierFilter."""

    iqr_multiplier: float = Field(
        default=1.5,
        gt=0.0,
        description=(
            "IQR multiplier *k*. "
            "``k=1.5`` flags mild outliers (Tukey's default); "
            "``k=3.0`` flags only extreme outliers."
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
            "``'cap'`` clips outlier values to the fence boundary (per-value, ignores ``threshold``)."
        ),
    )


# Exposed at module level so external tuners can discover the search space.
hyperparameter_space: dict[str, Any] = {
    "iqr_multiplier": tune.uniform(0.5, 5.0),
    "threshold": tune.uniform(0.0, 1.0),
    "strategy": tune.choice(["remove", "cap"]),
}


class IQROutlierFilterConfig(
    TransformConfig[
        IQROutlierFilterRunningConfig,
        IQROutlierFilterHyperParameters,
    ]
):
    """Full configuration for the IQROutlierFilter node."""

    hyperparameters: IQROutlierFilterHyperParameters = Field(
        default_factory=IQROutlierFilterHyperParameters,
        description="Tuneable hyperparameters (iqr_multiplier, threshold, strategy).",
    )
    running_config: IQROutlierFilterRunningConfig = Field(
        default_factory=IQROutlierFilterRunningConfig,
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
                optional=True,
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
                optional=True,
            ),
        },
        description="Output ports: 'output', 'target' (training/evaluation), 'sliced' (synced auxiliary).",
    )


class IQROutlierFilter(
    TransformNode[
        pd.DataFrame,
        TabularDataContext,
        pd.DataFrame,
        TabularDataContext,
        _IQRParams,
    ]
):
    """IQR (Tukey fence) outlier filter.

    Learns per-column Q1, Q3, and IQR on the training split, then uses those
    statistics to detect and handle outliers at transform time.

    Example:
    -------
    >>> cfg = IQROutlierFilterConfig(
    ...     hyperparameters=IQROutlierFilterHyperParameters(
    ...         iqr_multiplier=3.0, strategy="cap"
    ...     ),
    ... )
    >>> node = IQROutlierFilter(config=cfg)

    """

    metadata = IQROutlierFilterMetadata()
    hyperparameter_space = hyperparameter_space

    def __init__(self, *, config: IQROutlierFilterConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config
        self._params: _IQRParams = {}
        # Must be set so node_transform blocks until fit has been called.
        self._fitted = False

    # --- TransformNode interface ------------------------------------------

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Learn per-column Q1, Q3, and IQR from *data["input"]*."""
        df, _ = data["input"]
        target = self._select_columns(df)

        q1 = target.quantile(0.25)
        q3 = target.quantile(0.75)
        iqr = q3 - q1

        self._params = {
            "q1": {col: float(q1[col]) for col in target.columns},
            "q3": {col: float(q3[col]) for col in target.columns},
            "iqr": {col: float(iqr[col]) for col in target.columns},
        }

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Detect and handle outliers using the IQR fences learned during fit.

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

    def get_params(self) -> _IQRParams:
        """Return the fitted per-column IQR statistics (``q1``, ``q3``, ``iqr``)."""
        return self._params

    def set_params(self, params: _IQRParams) -> None:
        """Restore previously fitted statistics (used for checkpointing)."""
        self._params = params
        self._fitted = True

    # --- Private helpers --------------------------------------------------

    def _select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only the columns subject to filtering."""
        return filter_columns(df, self._config.running_config.filtering_columns)

    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a boolean DataFrame: ``True`` where a value is outside the fence."""
        cols = list(self._params["q1"].keys())
        target = df[cols]
        k = self._config.hyperparameters.iqr_multiplier

        lower = pd.Series(
            {c: self._params["q1"][c] - k * self._params["iqr"][c] for c in cols}
        )
        upper = pd.Series(
            {c: self._params["q3"][c] + k * self._params["iqr"][c] for c in cols}
        )
        return target.lt(lower, axis=1) | target.gt(upper, axis=1)

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
        k = self._config.hyperparameters.iqr_multiplier
        result = df.copy()
        for col in outlier_mask.columns:
            lower = self._params["q1"][col] - k * self._params["iqr"][col]
            upper = self._params["q3"][col] + k * self._params["iqr"][col]
            result[col] = result[col].clip(lower=lower, upper=upper)
        return result
