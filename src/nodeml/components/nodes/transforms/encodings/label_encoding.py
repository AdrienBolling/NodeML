"""Label Encoding transform node for the NodeML Framework.

Takes categorical columns and returns them as ordinal integer-encoded
numerical columns.  During :meth:`fit` the node discovers the sorted unique
categories for every column.  During :meth:`transform` it maps each category
to its positional index (0, 1, 2, …).  Categories that were not seen at fit
time are mapped to ``-1``.

The fitted category mapping is persisted via :meth:`get_params` /
:meth:`set_params` for checkpointing.
"""

import numpy as np
import pandas as pd
from pydantic import Field

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

# Serialisable params: column name -> sorted list of categories seen at fit.
type _LabelEncodingParams = dict[str, list[str]]


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
        LabelEncodingRunningConfig,
        LabelEncodingHyperParameters,
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

    During :meth:`fit` the sorted unique categories per column are captured.
    :meth:`transform` maps each value to its index in that sorted list;
    unseen categories are mapped to ``-1``.

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
        """Learn sorted unique categories for every column."""
        df, _ = data["input"]
        self._params = {
            col: sorted(df[col].dropna().unique().astype(str).tolist())
            for col in df.columns
        }

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Apply label encoding using the categories learned at fit time."""
        df, _ = data["input"]

        encoded_columns: dict[str, pd.Series] = {}
        column_names: list[str] = []

        for col in df.columns:
            categories = self._params.get(col, [])
            mapping: dict[str, int] = {cat: idx for idx, cat in enumerate(categories)}
            encoded_columns[col] = (
                df[col].astype(str).map(mapping).fillna(-1).astype(np.int64)
            )
            column_names.append(col)

        result = pd.DataFrame(encoded_columns)

        ctx = TabularDataContext(
            columns=column_names,
            dtypes=[np.dtype("int64")] * len(column_names),
            categories=[NumericalData] * len(column_names),
        )
        return {"output": (result, ctx)}

    def get_params(self) -> _LabelEncodingParams:
        """Return the per-column category mapping learned during fit."""
        return self._params

    def set_params(self, params: _LabelEncodingParams) -> None:
        """Restore a previously fitted category mapping."""
        self._params = params
        self._fitted = True
