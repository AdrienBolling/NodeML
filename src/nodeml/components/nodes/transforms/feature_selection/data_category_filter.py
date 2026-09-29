"""DataCategoryFilter transform node for the NodeML Framework.

Splits a mixed-category DataFrame into two streams by reading the column
semantics from the :class:`~nodeml.core.common.data.data.TabularDataContext`:

* ``categorical`` - columns tagged as :class:`~nodeml.core.common.data.data.CategoricalData`
* ``numerical``   - columns tagged as :class:`~nodeml.core.common.data.data.NumericalData`

Columns tagged as :class:`~nodeml.core.common.data.data.MixedData` are
excluded from both outputs.  Using the context (not pandas dtype
introspection) as the source of truth ensures correctness after transforms
such as label-encoding, which assign a numeric dtype to a categorical column.

The two retained column lists are persisted via :meth:`get_params` /
:meth:`set_params` so the node can be checkpointed and restored without
re-fitting.  Each output keeps the column order of the input.
"""

import pandas as pd
from pydantic import Field

from nodeml.components.utils.dataframe import check_fitted_columns
from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    CategoricalData,
    DataCategoryEnum,
    DataStructureEnum,
    NumericalData,
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


class DataCategoryFilterMetadata(TransformMetadata):
    """Metadata for the DataCategoryFilter node."""

    node_name: str = "DataCategoryFilter"
    description: str = (
        "Split a mixed-category DataFrame into categorical and numerical streams "
        "using the TabularDataContext as the authoritative source of column semantics."
    )
    trainable: bool = True


class DataCategoryFilterRunningConfig(TransformRunningConfig):
    """No run-time knobs for this node."""


class DataCategoryFilterHyperParameters(TransformHyperParameters):
    """No learnable hyperparameters."""


# Serialisable params: per-category lists of column names captured during fit.
type _DataCategoryFilterParams = dict[str, list[str]]


class DataCategoryFilterConfig(
    TransformConfig[
        DataCategoryFilterHyperParameters,
        DataCategoryFilterRunningConfig,
    ]
):
    """Full configuration for the DataCategoryFilter node."""

    hyperparameters: DataCategoryFilterHyperParameters = Field(
        default_factory=DataCategoryFilterHyperParameters,
        description="No tuneable hyperparameters for this node.",
    )
    running_config: DataCategoryFilterRunningConfig = Field(
        default_factory=DataCategoryFilterRunningConfig,
        description="No run-time knobs for this node.",
    )
    in_ports: dict[str, Port] = Field(
        default={
            "input": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.MIXED,
                data_shape="batch feature",
                desc="Mixed-category input DataFrame.",
            ),
        },
        description="Input port: 'input' (mixed-category DataFrame).",
    )
    out_ports: dict[str, Port] = Field(
        default={
            "categorical": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.CATEGORICAL,
                data_shape="batch _",
                desc="Columns tagged as CategoricalData in the input context.",
            ),
            "numerical": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch _",
                desc="Columns tagged as NumericalData in the input context.",
            ),
        },
        description=(
            "Output ports: 'categorical' and 'numerical'. "
            "MixedData columns are excluded from both."
        ),
    )


class DataCategoryFilter(
    TransformNode[
        pd.DataFrame,
        TabularDataContext,
        pd.DataFrame,
        TabularDataContext,
        _DataCategoryFilterParams,
    ]
):
    """Split a mixed DataFrame into categorical and numerical output ports.

    The column assignment is determined once during :meth:`fit` by reading
    ``ctx.categories`` and is reused on every subsequent :meth:`transform`
    call.  :class:`~nodeml.core.common.data.data.MixedData` columns are not
    routed to either output.  Each output keeps the input column order.

    Example:
        >>> node = DataCategoryFilter(config=DataCategoryFilterConfig())
        >>> node.node_fit_transform({"input": (df, ctx)})
        {"categorical": (df_cat, ctx_cat), "numerical": (df_num, ctx_num)}

    """

    metadata = DataCategoryFilterMetadata()

    def __init__(self, *, config: DataCategoryFilterConfig) -> None:
        """Initialise the node with its configuration.

        Args:
            config: The configuration of the node.

        """
        self._config = config
        self._params: _DataCategoryFilterParams = {
            "categorical": [],
            "numerical": [],
        }
        # Guard: node_transform raises until fit has been called.
        self._fitted = False

    # --- TransformNode interface ------------------------------------------

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Identify categorical and numerical columns from the input context.

        Args:
            data: Must contain the key ``"input"``.  Only the
                :class:`~nodeml.core.common.data.data.TabularDataContext` is
                read; the DataFrame values are not inspected.

        """
        _, ctx = data["input"]
        self._params = {
            "categorical": [
                col
                for col, cat in zip(ctx.columns, ctx.categories, strict=True)
                if cat is CategoricalData
            ],
            "numerical": [
                col
                for col, cat in zip(ctx.columns, ctx.categories, strict=True)
                if cat is NumericalData
            ],
        }

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Route columns to the ``categorical`` and ``numerical`` output ports.

        Args:
            data: Must contain the key ``"input"``.

        Returns:
            Always both ``"categorical"`` and ``"numerical"`` keys.  Either
            DataFrame can have zero columns if :meth:`fit` found no matching
            column.

        Raises:
            NodeInputError: If the input does not have a fitted column.

        """
        df, ctx = data["input"]
        check_fitted_columns(
            [*self._params["categorical"], *self._params["numerical"]],
            df.columns,
            node_name="DataCategoryFilter",
            allow_extra=True,
        )
        return {
            "categorical": self._select(df, ctx, self._params["categorical"]),
            "numerical": self._select(df, ctx, self._params["numerical"]),
        }

    def get_params(self) -> _DataCategoryFilterParams:
        """Return the column assignment learned during fit.

        Returns:
            ``{"categorical": [...], "numerical": [...]}``.

        """
        return self._params

    def set_params(self, params: _DataCategoryFilterParams) -> None:
        """Restore a previously fitted column assignment (checkpointing).

        Args:
            params: Dict with keys ``"categorical"`` and ``"numerical"``,
                each mapping to a list of column names.

        """
        self._params = params
        self._fitted = True

    # --- Private helpers --------------------------------------------------

    @staticmethod
    def _select(
        df: pd.DataFrame,
        ctx: TabularDataContext,
        columns: list[str],
    ) -> tuple[pd.DataFrame, TabularDataContext]:
        """Keep the *columns* of *df* and *ctx*, in the input order."""
        keep = set(columns)
        ordered = [col for col in df.columns if col in keep]
        return df[ordered], ctx.select(ordered)
