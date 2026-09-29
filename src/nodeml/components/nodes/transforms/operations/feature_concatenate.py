"""FeatureConcatenate transform node for the NodeML Framework.

Concatenates two DataFrames along the column axis (feature dimension).
The rows pair by position, not by index label, and the output keeps the
index of ``input_1``.  Duplicate column names across the two inputs are
rejected to prevent silent data corruption.
"""

import pandas as pd
from pydantic import Field

from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    TabularDataContext,
)
from nodeml.core.common.exceptions import NodeInputError
from nodeml.core.nodes.node import Port
from nodeml.core.nodes.transform.transform import (
    TransformConfig,
    TransformHyperParameters,
    TransformMetadata,
    TransformNode,
    TransformRunningConfig,
)


class FeatureConcatenateMetadata(TransformMetadata):
    """Metadata for the FeatureConcatenate node."""

    node_name: str = "FeatureConcatenate"
    description: str = (
        "Concatenate two DataFrames column-wise (feature axis). "
        "Duplicate column names across the two inputs are rejected."
    )
    trainable: bool = False


class FeatureConcatenateRunningConfig(TransformRunningConfig):
    """Run-time options for feature concatenation."""

    check_row_count: bool = Field(
        default=True,
        description=(
            "When ``True`` (default), raise if the two inputs have different "
            "numbers of rows. Set to ``False`` to allow unequal row counts: "
            "the shorter input is padded with ``NaN`` at the end. The runner "
            "checks that the 'batch' dimension of the two input ports has "
            "the same size, so ``False`` applies only outside a runner or "
            "with other port shapes."
        ),
    )


class FeatureConcatenateHyperParameters(TransformHyperParameters):
    """No learnable hyperparameters."""


class FeatureConcatenateConfig(
    TransformConfig[
        FeatureConcatenateHyperParameters,
        FeatureConcatenateRunningConfig,
    ]
):
    """Full configuration for the FeatureConcatenate node."""

    hyperparameters: FeatureConcatenateHyperParameters = Field(
        default_factory=FeatureConcatenateHyperParameters,
        description="No tuneable hyperparameters for this node.",
    )
    running_config: FeatureConcatenateRunningConfig = Field(
        default_factory=FeatureConcatenateRunningConfig,
        description="Run-time options (check_row_count).",
    )
    in_ports: dict[str, Port] = Field(
        default={
            "input_1": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.MIXED,
                data_shape="batch feature1",
                desc="First DataFrame (batch x feature1).",
            ),
            "input_2": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.MIXED,
                data_shape="batch feature2",
                desc="Second DataFrame (batch x feature2).",
            ),
        },
        description="Input ports: 'input_1' and 'input_2' (DataFrames to concatenate).",
    )
    out_ports: dict[str, Port] = Field(
        default={
            "output": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.MIXED,
                data_shape="batch _",
                desc="Column-concatenated DataFrame (batch x (feature1+feature2)).",
            ),
        },
        description="Output ports: 'output' (concatenated DataFrame).",
    )


class FeatureConcatenate(
    TransformNode[
        pd.DataFrame,
        TabularDataContext,
        pd.DataFrame,
        TabularDataContext,
        None,
    ]
):
    """Concatenate two DataFrames along the column axis.

    The rows pair by position: row ``i`` of ``input_1`` goes with row ``i``
    of ``input_2``, whatever their index labels.  The output keeps the
    index of ``input_1``.  When ``input_2`` is longer (only possible with
    ``check_row_count=False``), the output gets a new ``RangeIndex``.

    The output :class:`~nodeml.core.common.data.data.TabularDataContext`
    merges the two input contexts in order (``input_1`` columns first,
    then ``input_2`` columns).  Its dtypes describe the output, which can
    differ from the inputs when NaN padding occurs.

    **Duplicate column guard** - if both inputs share any column name the
    operation raises a ``NodeInputError`` rather than silently producing
    ambiguous columns.

    **Row-count check** - controlled by
    ``running_config.check_row_count`` (enabled by default).
    """

    metadata = FeatureConcatenateMetadata()

    def __init__(self, *, config: FeatureConcatenateConfig) -> None:
        """Initialise the node with its configuration.

        Args:
            config: The configuration of the node.

        """
        self._config = config

    # --- TransformNode interface ------------------------------------------

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Validate the inputs at fit time (no parameters to learn).

        Args:
            data: Must contain the keys ``"input_1"`` and ``"input_2"``.

        Raises:
            NodeInputError: If the inputs share a column name, or if their
                row counts differ and ``check_row_count`` is ``True``.

        """
        df1, _ = data["input_1"]
        df2, _ = data["input_2"]
        self._assert_inputs_valid(df1, df2)

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Concatenate the two DataFrames column-wise, by row position.

        Args:
            data: Must contain the keys ``"input_1"`` and ``"input_2"``.

        Returns:
            ``{"output": (df, ctx)}``.

        Raises:
            NodeInputError: If the inputs share a column name, or if their
                row counts differ and ``check_row_count`` is ``True``.

        """
        df1, ctx1 = data["input_1"]
        df2, ctx2 = data["input_2"]
        self._assert_inputs_valid(df1, df2)

        concatenated = pd.concat(
            [df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1
        )
        if len(concatenated) == len(df1):
            concatenated.index = df1.index
        merged_ctx = TabularDataContext(
            columns=ctx1.columns + ctx2.columns,
            dtypes=ctx1.dtypes + ctx2.dtypes,
            categories=ctx1.categories + ctx2.categories,
        )
        return {"output": (concatenated, merged_ctx.aligned_to(concatenated))}

    def get_params(self) -> None:
        """Return ``None``: the node has no learned parameters.

        Returns:
            ``None``.

        """
        return

    def set_params(self, params: None) -> None:
        """Do nothing: the node has no learned parameters.

        Args:
            params: Ignored.

        """

    # --- Private helpers --------------------------------------------------

    def _assert_inputs_valid(self, df1: pd.DataFrame, df2: pd.DataFrame) -> None:
        """Raise ``NodeInputError`` on duplicate columns or row-count mismatch."""
        duplicates = set(df1.columns) & set(df2.columns)
        if duplicates:
            msg = (
                f"FeatureConcatenate: duplicate column names across inputs: "
                f"{sorted(duplicates)}. Rename columns before concatenating."
            )
            raise NodeInputError(msg)
        if self._config.running_config.check_row_count and df1.shape[0] != df2.shape[0]:
            msg = (
                f"FeatureConcatenate: row-count mismatch between inputs: "
                f"{df1.shape[0]} vs {df2.shape[0]}. "
                "Set running_config.check_row_count=False to allow this."
            )
            raise NodeInputError(msg)
