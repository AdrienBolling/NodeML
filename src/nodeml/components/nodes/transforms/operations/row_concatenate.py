"""RowConcatenate transform node for the NodeML Framework.

Concatenates two DataFrames along the row axis (batch dimension).  Both
inputs must share exactly the same schema — columns, dtypes, and data
categories must all match.  The output row index is reset so that
downstream nodes receive a contiguous index starting at 0.
"""

import pandas as pd
from pydantic import Field

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


class RowConcatenateMetadata(TransformMetadata):
    """Metadata for the RowConcatenate node."""

    node_name: str = "RowConcatenate"
    description: str = (
        "Concatenate two DataFrames row-wise (batch axis). "
        "Both inputs must share the same schema."
    )
    trainable: bool = False


class RowConcatenateRunningConfig(TransformRunningConfig):
    """No run-time knobs needed for row concatenation."""


class RowConcatenateHyperParameters(TransformHyperParameters):
    """No learnable hyperparameters."""


class RowConcatenateConfig(
    TransformConfig[RowConcatenateRunningConfig, RowConcatenateHyperParameters]
):
    """Full configuration for the RowConcatenate node."""

    hyperparameters: RowConcatenateHyperParameters = Field(
        default_factory=RowConcatenateHyperParameters,
        description="No tuneable hyperparameters for this node.",
    )
    running_config: RowConcatenateRunningConfig = Field(
        default_factory=RowConcatenateRunningConfig,
        description="No run-time knobs for this node.",
    )
    in_ports: dict[str, Port] = Field(
        default={
            "input_1": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.MIXED,
                data_shape="batch1 feature",
                desc="First DataFrame (batch1 x feature).",
            ),
            "input_2": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.MIXED,
                data_shape="batch2 feature",
                desc="Second DataFrame (batch2 x feature).",
            ),
        },
        description="Input ports: 'input_1' and 'input_2' (same-schema DataFrames).",
    )
    out_ports: dict[str, Port] = Field(
        default={
            "output": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.MIXED,
                data_shape="_ feature",
                desc="Row-concatenated DataFrame ((batch1+batch2) x feature).",
            ),
        },
        description="Output ports: 'output' (concatenated DataFrame).",
    )


class RowConcatenate(
    TransformNode[
        pd.DataFrame,
        TabularDataContext,
        pd.DataFrame,
        TabularDataContext,
        None,
    ]
):
    """Concatenate two DataFrames along the row axis.

    The output context is inherited from ``input_1``.  The row index of the
    output is reset to avoid duplicate index values.

    **Schema validation** - both inputs must have identical column names,
    dtypes, and data categories.  A ``ValueError`` is raised otherwise.
    This is checked in both :meth:`fit` (early feedback) and
    :meth:`transform` (runtime safety).
    """

    metadata = RowConcatenateMetadata()

    def __init__(self, *, config: RowConcatenateConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config

    # --- TransformNode interface ------------------------------------------

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Validate schema compatibility at fit time (no parameters to learn)."""
        _, ctx1 = data["input_1"]
        _, ctx2 = data["input_2"]
        self._assert_schema_compatible(ctx1, ctx2)

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Concatenate the two DataFrames row-wise.

        Parameters
        ----------
        data:
            Must contain keys ``"input_1"`` and ``"input_2"``.

        """
        df1, ctx1 = data["input_1"]
        df2, ctx2 = data["input_2"]
        self._assert_schema_compatible(ctx1, ctx2)

        concatenated = pd.concat([df1, df2], axis=0, ignore_index=True)
        return {"output": (concatenated, ctx1)}

    def get_params(self) -> None:
        """No learned parameters — returns ``None``."""
        return

    def set_params(self, params: None) -> None:
        """No learned parameters to restore."""

    # --- Private helpers --------------------------------------------------

    @staticmethod
    def _assert_schema_compatible(
        ctx1: TabularDataContext, ctx2: TabularDataContext
    ) -> None:
        """Raise ``ValueError`` if the two contexts have incompatible schemas."""
        if ctx1.columns != ctx2.columns:
            msg = (
                f"RowConcatenate: column mismatch between inputs.\n"
                f"  input_1 columns: {ctx1.columns}\n"
                f"  input_2 columns: {ctx2.columns}"
            )
            raise ValueError(msg)
        if ctx1.dtypes != ctx2.dtypes:
            msg = (
                f"RowConcatenate: dtype mismatch between inputs.\n"
                f"  input_1 dtypes: {ctx1.dtypes}\n"
                f"  input_2 dtypes: {ctx2.dtypes}"
            )
            raise ValueError(msg)
        if ctx1.categories != ctx2.categories:
            msg = (
                f"RowConcatenate: data-category mismatch between inputs.\n"
                f"  input_1 categories: {ctx1.categories}\n"
                f"  input_2 categories: {ctx2.categories}"
            )
            raise ValueError(msg)
