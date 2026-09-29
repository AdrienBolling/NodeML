"""Shared base for the column filters of the NodeML Framework.

A column filter learns during :meth:`ColumnFilter.fit` which columns to
keep.  :meth:`ColumnFilter.transform` then keeps these columns.

* ``running_config.filtering_columns`` limits the candidate columns.  The
  filter always keeps the other columns.
* The output keeps the column order of the input.
* The output context comes from the input context, selected by column
  name, so the categories stay with their columns.
"""

from abc import ABC, abstractmethod

import pandas as pd
from pydantic import Field

from nodeml.components.utils.dataframe import check_fitted_columns, filter_columns
from nodeml.core.common.data.data import TabularDataContext
from nodeml.core.nodes.transform.transform import (
    TransformConfig,
    TransformNode,
    TransformRunningConfig,
)

# Serialisable params: {"columns_to_keep": [...]}, in the input order at fit.
type ColumnFilterParams = dict[str, list[str]]


class ColumnFilterRunningConfig(TransformRunningConfig):
    """Run-time options shared by the column filters."""

    filtering_columns: list[str] | None = Field(
        default=None,
        description=(
            "Subset of columns that the filter can remove. "
            "``None`` (default) evaluates all columns. "
            "Columns not in this list are always kept in the output."
        ),
    )


class ColumnFilter(
    TransformNode[
        pd.DataFrame,
        TabularDataContext,
        pd.DataFrame,
        TabularDataContext,
        ColumnFilterParams,
    ],
    ABC,
):
    """Base class for the filters that learn which columns to keep.

    A subclass gives the candidate columns that pass its test in
    :meth:`_surviving_columns`.  This class handles ``filtering_columns``,
    the column order and the context.
    """

    def __init__(self, *, config: TransformConfig) -> None:
        """Initialise the node with its configuration.

        Args:
            config: The configuration of the filter.

        """
        self._config = config
        self._params: ColumnFilterParams = {"columns_to_keep": []}
        # node_transform refuses to run until fit or set_params is called.
        self._fitted = False

    # --- Subclass interface -------------------------------------------------

    @abstractmethod
    def _surviving_columns(self, candidates: pd.DataFrame) -> list[str]:
        """Return the candidate columns that pass the filter.

        Args:
            candidates: The candidate columns of the training data.

        Returns:
            The names of the columns to keep, in any order.

        """

    # --- TransformNode interface --------------------------------------------

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Learn which columns of ``data["input"]`` to keep.

        Args:
            data: Must contain the key ``"input"``.

        Raises:
            NodeInputError: If a column of ``filtering_columns`` is not in
                the input.

        """
        df, _ = data["input"]
        candidates = filter_columns(df, self._config.running_config.filtering_columns)
        candidate_names = set(candidates.columns)
        surviving = set(self._surviving_columns(candidates))
        self._params = {
            "columns_to_keep": [
                col
                for col in df.columns
                if col not in candidate_names or col in surviving
            ],
        }

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Keep the columns selected during fit, in the input order.

        Args:
            data: Must contain the key ``"input"``.

        Returns:
            ``{"output": (df, ctx)}`` with the kept columns.

        Raises:
            NodeInputError: If the input does not have a kept column.

        """
        df, ctx = data["input"]
        keep = self._params["columns_to_keep"]
        check_fitted_columns(
            keep, df.columns, node_name=type(self).__name__, allow_extra=True
        )
        kept = set(keep)
        columns = [col for col in df.columns if col in kept]
        return {"output": (df[columns], ctx.select(columns))}

    def get_params(self) -> ColumnFilterParams:
        """Return the columns to keep.

        Returns:
            ``{"columns_to_keep": [...]}``.

        """
        return self._params

    def set_params(self, params: ColumnFilterParams) -> None:
        """Restore the columns returned by :meth:`get_params`.

        Args:
            params: The params of a fitted filter of the same class.

        """
        self._params = params
        self._fitted = True
