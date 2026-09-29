"""Label Encoding transform node for the NodeML Framework.

Takes categorical columns and returns them as ordinal integer-encoded
numerical columns.  During :meth:`fit` the node finds the sorted distinct
categories of each column.  During :meth:`transform` it maps each category
to its position (0, 1, 2, …).  A missing value, or a category that fit did
not see, becomes ``-1``.

The node compares real values, not strings: ``1.0`` and ``1`` are the same
category.  Numbers are sorted by value.  See :mod:`._categories`.

:meth:`get_params` / :meth:`set_params` save and restore the categories.
"""

from collections.abc import Hashable

import numpy as np
import pandas as pd
from pydantic import Field

from nodeml.components.utils.dataframe import check_fitted_columns
from nodeml.core.common.data.data import (
    ArrayLikeEnum,
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

from ._categories import encode_codes, learn_categories

# Serialisable params: column name -> sorted list of categories seen at fit.
type _LabelEncodingParams = dict[str, list[Hashable]]


class LabelEncodingMetadata(TransformMetadata):
    """Metadata for the LabelEncoding node."""

    node_name: str = "LabelEncoding"
    description: str = "Encode categorical columns as ordinal integers (0, 1, 2, …)."
    trainable: bool = True


class LabelEncodingRunningConfig(TransformRunningConfig):
    """No run-time knobs for this node."""


class LabelEncodingHyperParameters(TransformHyperParameters):
    """No tuneable hyperparameters."""


class LabelEncodingConfig(
    TransformConfig[
        LabelEncodingHyperParameters,
        LabelEncodingRunningConfig,
    ],
):
    """Full configuration for the LabelEncoding node."""

    hyperparameters: LabelEncodingHyperParameters = Field(
        default_factory=LabelEncodingHyperParameters,
        description="No tuneable hyperparameters for this node.",
    )
    running_config: LabelEncodingRunningConfig = Field(
        default_factory=LabelEncodingRunningConfig,
        description="No run-time knobs for this node.",
    )
    in_ports: dict[str, Port] = Field(
        default={
            "input": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.CATEGORICAL,
                data_shape="batch feature",
                desc="Categorical DataFrame to label-encode.",
            ),
        },
        description="Input port: 'input' (categorical DataFrame).",
    )
    out_ports: dict[str, Port] = Field(
        default={
            "output": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch feature",
                desc="Integer-encoded numerical DataFrame.",
            ),
        },
        description="Output port: 'output' (integer-encoded columns).",
    )


class LabelEncoding(
    TransformNode[
        pd.DataFrame,
        TabularDataContext,
        pd.DataFrame,
        TabularDataContext,
        _LabelEncodingParams,
    ],
):
    """Encode categorical columns as ordinal integers.

    During :meth:`fit`, the node captures the sorted categories of each
    column.  :meth:`transform` maps each value to its position in that
    list.  Missing values and unseen categories become ``-1``.  The output
    keeps the column order and the index of the input.

    Example:
        >>> node = LabelEncoding(config=LabelEncodingConfig())
        >>> out = node.node_fit_transform({"input": (df_cat, ctx_cat)})
        >>> encoded_df, encoded_ctx = out["output"]

    """

    metadata = LabelEncodingMetadata()

    def __init__(self, *, config: LabelEncodingConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config
        self._params: _LabelEncodingParams = {}
        self._fitted = False

    # --- TransformNode interface ------------------------------------------

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Learn the sorted categories of each column of ``data["input"]``.

        Args:
            data: Must contain the key ``"input"``.

        """
        df, _ = data["input"]
        self._params = {col: learn_categories(df[col]) for col in df.columns}

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Encode each column with the categories learned during fit.

        Args:
            data: Must contain the key ``"input"``.

        Returns:
            ``{"output": (df, ctx)}`` with one ``int64`` column per input
            column, in the input order.

        Raises:
            NodeInputError: If the input columns are not the fitted columns.

        """
        df, _ = data["input"]
        check_fitted_columns(self._params, df.columns, node_name="LabelEncoding")
        result = pd.DataFrame(
            {col: encode_codes(df[col], self._params[col]) for col in df.columns},
            index=df.index,
            columns=df.columns,
        )
        ctx = TabularDataContext(
            columns=list(result.columns),
            dtypes=[np.dtype("int64")] * result.shape[1],
            categories=[NumericalData] * result.shape[1],
        )
        return {"output": (result, ctx)}

    def get_params(self) -> _LabelEncodingParams:
        """Return the categories learned during fit.

        Returns:
            A mapping of column name to its sorted categories.

        """
        return self._params

    def set_params(self, params: _LabelEncodingParams) -> None:
        """Restore the categories returned by :meth:`get_params`.

        Args:
            params: The params of a fitted LabelEncoding node.

        """
        self._params = params
        self._fitted = True
