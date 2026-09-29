"""Shared base for the outlier filters of the NodeML Framework.

An outlier filter learns a lower and an upper bound for each filtered
column during :meth:`OutlierFilter.fit`.  A value outside its bounds is an
outlier.  Two strategies handle the outliers:

* ``"remove"`` removes the rows that have too many outlier features.  Row
  removal happens only in the training mode.  In the inference and
  evaluation modes, the filter passes all rows through unchanged, so that
  each input row gets a prediction and a score.
* ``"cap"`` clips each outlier value to its bounds, in all modes.

The ``target`` and ``sliced`` ports carry frames whose rows follow the rows
of ``input`` by position.  The filter removes the same rows from them.
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from nodeml.components.utils.dataframe import check_fitted_columns, filter_columns
from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    TabularDataContext,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.common.exceptions import NodeInputError
from nodeml.core.nodes.node import Port
from nodeml.core.nodes.transform.transform import (
    TransformConfig,
    TransformHyperParameters,
    TransformNode,
    TransformRunningConfig,
)

# Serialisable params: statistic name -> column name -> value.
type OutlierFilterParams = dict[str, dict[str, float]]

# Ports that carry frames whose rows follow the rows of "input".
_ROW_ALIGNED_PORTS = ("target", "sliced")


class OutlierFilterRunningConfig(TransformRunningConfig):
    """Run-time options shared by the outlier filters."""

    filtering_columns: list[str] | None = Field(
        default=None,
        description=(
            "Subset of columns to apply the filter to. "
            "``None`` (default) applies the filter to all columns."
        ),
    )


class OutlierFilterHyperParameters(TransformHyperParameters):
    """Hyperparameters shared by the outlier filters."""

    threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum fraction of outlier features for the removal of a row "
            "(``strategy='remove'`` only). A row is removed when it has at "
            "least one outlier feature and ``outlier_features / "
            "total_features >= threshold``. ``0.0`` (default) removes each "
            "row that has at least one outlier feature. ``1.0`` removes only "
            "the rows where all features are outliers."
        ),
    )
    strategy: Literal["remove", "cap"] = Field(
        default="remove",
        description=(
            "How to handle the outliers. ``'remove'`` removes the rows that "
            "``threshold`` selects, in the training mode only. In the other "
            "modes, all rows pass through. ``'cap'`` clips each outlier value "
            "to its fitted bounds in all modes, and ignores ``threshold``."
        ),
    )


def outlier_filter_in_ports() -> dict[str, Port]:
    """Return the default input ports of an outlier filter.

    Returns:
        The ``input``, ``target`` and ``sliced`` ports.

    """
    return {
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
            desc="Target DataFrame; its rows follow the rows of 'input'.",
            mode=[NodeExecutionMode.TRAINING, NodeExecutionMode.EVALUATION],
        ),
        "sliced": Port(
            arr_type=ArrayLikeEnum.PANDAS,
            data_structure=DataStructureEnum.TABULAR,
            data_category=DataCategoryEnum.MIXED,
            data_shape="batch _",
            optional=True,
            desc="Auxiliary DataFrame; its rows follow the rows of 'input'.",
        ),
    }


def outlier_filter_out_ports() -> dict[str, Port]:
    """Return the default output ports of an outlier filter.

    Returns:
        The ``output``, ``target`` and ``sliced`` ports.

    """
    return {
        "output": Port(
            arr_type=ArrayLikeEnum.PANDAS,
            data_structure=DataStructureEnum.TABULAR,
            data_category=DataCategoryEnum.NUMERICAL,
            data_shape="_ feature",
            desc=(
                "DataFrame with the outliers handled. In the training mode, "
                "strategy='remove' can make the batch dimension smaller."
            ),
        ),
        "target": Port(
            arr_type=ArrayLikeEnum.PANDAS,
            data_structure=DataStructureEnum.TABULAR,
            data_category=DataCategoryEnum.NUMERICAL,
            data_shape="_ _",
            desc="Target DataFrame with the same rows removed as 'output'.",
            mode=[NodeExecutionMode.TRAINING, NodeExecutionMode.EVALUATION],
        ),
        "sliced": Port(
            arr_type=ArrayLikeEnum.PANDAS,
            data_structure=DataStructureEnum.TABULAR,
            data_category=DataCategoryEnum.MIXED,
            data_shape="_ _",
            optional=True,
            desc="Auxiliary DataFrame with the same rows removed as 'output'.",
        ),
    }


class OutlierFilter(
    TransformNode[
        pd.DataFrame,
        TabularDataContext,
        pd.DataFrame,
        TabularDataContext,
        OutlierFilterParams,
    ],
    ABC,
):
    """Base class for the outlier filters.

    A subclass learns its statistics in :meth:`_fit_statistics` and gives
    the bounds of each column in :meth:`_bounds`.  This class does the
    detection, the row removal and the capping.
    """

    def __init__(self, *, config: TransformConfig) -> None:
        """Initialise the node with its configuration.

        Args:
            config: The configuration of the filter.

        """
        self._config = config
        self._params: OutlierFilterParams = {}
        # node_transform refuses to run until fit or set_params is called.
        self._fitted = False

    # --- Subclass interface -------------------------------------------------

    @abstractmethod
    def _fit_statistics(self, candidates: pd.DataFrame) -> OutlierFilterParams:
        """Learn the statistics of the filtered columns.

        Args:
            candidates: The filtered columns of the training data.

        Returns:
            The params, keyed by statistic name, then by column name.

        """

    @abstractmethod
    def _raw_bounds(self) -> tuple[pd.Series, pd.Series]:
        """Return the lower and upper bounds of each fitted column.

        A NaN bound means that the column has no bound on that side.

        Returns:
            Two float Series indexed by the fitted column names.

        """

    # --- TransformNode interface --------------------------------------------

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Learn the statistics of the filtered columns of ``data["input"]``.

        Args:
            data: Must contain the key ``"input"``.

        """
        df, _ = data["input"]
        candidates = filter_columns(df, self._config.running_config.filtering_columns)
        self._params = self._fit_statistics(candidates)

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Handle the outliers with the bounds learned during fit.

        Args:
            data: Must contain the key ``"input"``.  It can also contain
                ``"target"`` and ``"sliced"``.  These frames must have the
                same number of rows as ``"input"``.

        Returns:
            The ``"output"`` frame, and the ``"target"`` and ``"sliced"``
            frames when they are in *data*.

        Raises:
            NodeInputError: If ``"input"`` does not have a fitted column,
                or if a row-aligned frame has another number of rows.

        """
        df, ctx = data["input"]
        lower, upper = self._bounds()
        check_fitted_columns(
            lower.index, df.columns, node_name=type(self).__name__, allow_extra=True
        )
        passthrough = {port: data[port] for port in _ROW_ALIGNED_PORTS if port in data}
        for port, (port_df, _) in passthrough.items():
            if len(port_df) != len(df):
                msg = (
                    f"{type(self).__name__}: port '{port}' has {len(port_df)} rows, "
                    f"but port 'input' has {len(df)} rows. The rows of '{port}' "
                    "must follow the rows of 'input'."
                )
                raise NodeInputError(msg)

        if self._config.hyperparameters.strategy == "cap":
            result = self._cap(df, lower, upper)
            return {"output": (result, ctx.aligned_to(result)), **passthrough}

        if self.execution_mode != NodeExecutionMode.TRAINING:
            # Rows are removed only in training: pass all rows through.
            return {"output": (df, ctx), **passthrough}

        keep = ~self._rows_to_remove(df, lower, upper)
        outputs = {"output": (self._take(df, keep), ctx)}
        for port, (port_df, port_ctx) in passthrough.items():
            outputs[port] = (self._take(port_df, keep), port_ctx)
        return outputs

    def get_params(self) -> OutlierFilterParams:
        """Return the fitted statistics.

        Returns:
            The params, keyed by statistic name, then by column name.

        """
        return self._params

    def set_params(self, params: OutlierFilterParams) -> None:
        """Restore the statistics returned by :meth:`get_params`.

        Args:
            params: The params of a fitted filter of the same class.

        """
        self._params = params
        self._fitted = True

    # --- Private helpers ----------------------------------------------------

    def _bounds(self) -> tuple[pd.Series, pd.Series]:
        """Return the bounds of each fitted column, with infinity for no bound."""
        lower, upper = self._raw_bounds()
        return (
            lower.astype(np.float64).fillna(-np.inf),
            upper.astype(np.float64).fillna(np.inf),
        )

    def _rows_to_remove(
        self, df: pd.DataFrame, lower: pd.Series, upper: pd.Series
    ) -> np.ndarray:
        """Return a boolean mask of the rows to remove, by position.

        A row is removed when it has at least one outlier feature and the
        fraction of outlier features is at least ``threshold``.
        """
        n_columns = len(lower)
        if n_columns == 0:
            return np.zeros(len(df), dtype=bool)
        values = df[list(lower.index)].to_numpy(dtype=np.float64, na_value=np.nan)
        # A comparison with NaN is False, so a missing value is not an outlier.
        outliers = (values < lower.to_numpy()) | (values > upper.to_numpy())
        counts = outliers.sum(axis=1)
        fraction = counts / n_columns
        threshold = self._config.hyperparameters.threshold
        return (counts > 0) & (fraction >= threshold)

    @staticmethod
    def _cap(df: pd.DataFrame, lower: pd.Series, upper: pd.Series) -> pd.DataFrame:
        """Clip each fitted column to its bounds."""
        result = df.copy()
        for col in lower.index:
            low, high = lower[col], upper[col]
            if np.isinf(low) and np.isinf(high):
                continue  # No bound: keep the column and its dtype unchanged.
            result[col] = df[col].clip(lower=low, upper=high)
        return result

    @staticmethod
    def _take(df: pd.DataFrame, keep: np.ndarray) -> pd.DataFrame:
        """Keep the rows of *df* where *keep* is True, and reset the index."""
        return df.iloc[keep].reset_index(drop=True)
