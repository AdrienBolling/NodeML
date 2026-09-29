"""Shared base for the scalers of the NodeML Framework.

Each scaler computes ``x_scaled = (x - center) / scale`` per column.  It
learns the center and the scale of each column during
:meth:`AffineScaler.fit`.

* The output keeps the column order and the index of the input.
* The input must have exactly the fitted columns, in any order.
* A column with a zero or undefined scale (for example, a fit on one row
  or a constant column) is only centred: its scale becomes 1.
* The output columns are ``float64``, and the output context says so.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from nodeml.components.utils.dataframe import check_fitted_columns
from nodeml.core.common.data.data import TabularDataContext
from nodeml.core.nodes.transform.transform import TransformConfig, TransformNode

# Serialisable params: statistic name -> column name -> value.
type ScalerParams = dict[str, dict[str, float]]


class AffineScaler(
    TransformNode[
        pd.DataFrame,
        TabularDataContext,
        pd.DataFrame,
        TabularDataContext,
        ScalerParams,
    ],
    ABC,
):
    """Base class for the scalers of the form ``(x - center) / scale``.

    A subclass learns its statistics in :meth:`_fit_statistics` and gives
    the center and the scale of each column in :meth:`_center_and_scale`.
    """

    def __init__(self, *, config: TransformConfig) -> None:
        """Initialise the node with its configuration.

        Args:
            config: The configuration of the scaler.

        """
        self._config = config
        self._params: ScalerParams = self._empty_params()
        # node_transform refuses to run until fit or set_params is called.
        self._fitted = False

    # --- Subclass interface -------------------------------------------------

    @staticmethod
    @abstractmethod
    def _empty_params() -> ScalerParams:
        """Return the params of a scaler that is not fitted."""

    @staticmethod
    @abstractmethod
    def _fit_statistics(df: pd.DataFrame) -> ScalerParams:
        """Learn the statistics of each column of *df*.

        Args:
            df: The training data.

        Returns:
            The params, keyed by statistic name, then by column name.

        """

    @abstractmethod
    def _center_and_scale(self) -> tuple[pd.Series, pd.Series]:
        """Return the center and the scale of each fitted column.

        Returns:
            Two float Series indexed by the fitted column names.

        """

    # --- TransformNode interface --------------------------------------------

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Learn the statistics of each column of ``data["input"]``.

        Args:
            data: Must contain the key ``"input"``.

        """
        df, _ = data["input"]
        self._params = self._fit_statistics(df)

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Scale each column with the statistics learned during fit.

        Args:
            data: Must contain the key ``"input"``.

        Returns:
            ``{"output": (df, ctx)}`` with the scaled ``float64`` columns, in
            the input order.

        Raises:
            NodeInputError: If the input columns are not the fitted columns.

        """
        df, ctx = data["input"]
        center, scale = self._center_and_scale()
        check_fitted_columns(center.index, df.columns, node_name=type(self).__name__)
        columns = list(df.columns)
        center_values = center[columns].to_numpy(dtype=np.float64)
        scale_values = scale[columns].to_numpy(dtype=np.float64)
        # A zero or NaN scale would give inf or NaN: only centre the column.
        scale_values = np.where(
            np.isnan(scale_values) | (scale_values == 0.0), 1.0, scale_values
        )
        values = df.to_numpy(dtype=np.float64, na_value=np.nan)
        result = pd.DataFrame(
            (values - center_values) / scale_values,
            index=df.index,
            columns=df.columns,
        )
        return {"output": (result, ctx.aligned_to(result))}

    def get_params(self) -> ScalerParams:
        """Return the fitted statistics.

        Returns:
            The params, keyed by statistic name, then by column name.

        """
        return self._params

    def set_params(self, params: ScalerParams) -> None:
        """Restore the statistics returned by :meth:`get_params`.

        Args:
            params: The params of a fitted scaler of the same class.

        """
        self._params = params
        self._fitted = True
