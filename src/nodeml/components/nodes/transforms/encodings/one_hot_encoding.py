"""One-Hot Encoding transform node for the NodeML Framework.

Takes categorical columns and returns them as one-hot-encoded numerical
columns.  During :meth:`fit` the node discovers the unique categories for
every column.  During :meth:`transform` it applies the same mapping, silently
ignoring categories that were not seen at fit time (they produce all-zero
rows for their original column's dummies).

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
type _OneHotParams = dict[str, list[str]]


class OneHotEncodingMetadata(TransformMetadata):
    """Metadata for the OneHotEncoding node."""

    node_name: str = "OneHotEncoding"
    description: str = (
        "One-hot encode categorical columns into numerical dummy columns."
    )
    trainable: bool = True


class OneHotEncodingRunningConfig(TransformRunningConfig):
    """No run-time knobs for this node."""


class OneHotEncodingHyperParameters(TransformHyperParameters):
    """No tuneable hyperparameters."""


class OneHotEncodingConfig(
    TransformConfig[
        OneHotEncodingRunningConfig,
        OneHotEncodingHyperParameters,
    ],
):
    """Full configuration for the OneHotEncoding node."""

    hyperparameters: OneHotEncodingHyperParameters = Field(
        default_factory=OneHotEncodingHyperParameters,
        description="No tuneable hyperparameters for this node.",
    )
    running_config: OneHotEncodingRunningConfig = Field(
        default_factory=OneHotEncodingRunningConfig,
        description="No run-time knobs for this node.",
    )
    in_ports: dict[str, Port] = Field(
        default={
            "input": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.CATEGORICAL,
                data_shape="batch feature",
                desc="Categorical-only DataFrame to one-hot encode.",
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
                data_shape="batch _",
                desc="One-hot encoded numerical DataFrame.",
            ),
        },
        description="Output port: 'output' (numerical dummy columns).",
    )


class OneHotEncoding(
    TransformNode[
        pd.DataFrame,
        TabularDataContext,
        pd.DataFrame,
        TabularDataContext,
        _OneHotParams,
    ],
):
    """One-hot encode categorical columns into numerical dummies.

    During :meth:`fit` the unique categories per column are captured.
    :meth:`transform` applies the same encoding deterministically, producing
    columns named ``<original_col>_<category>``.

    Example:
    -------
    >>> node = OneHotEncoding(config=OneHotEncodingConfig())
    >>> out = node.node_fit_transform({"input": (df_cat, ctx_cat)})
    >>> encoded_df, encoded_ctx = out["output"]

    """

    metadata = OneHotEncodingMetadata()

    def __init__(self, *, config: OneHotEncodingConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config
        self._params: _OneHotParams = {}
        self._fitted = False

    # --- TransformNode interface ------------------------------------------

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Learn unique categories for every column."""
        df, _ = data["input"]
        self._params = {
            col: sorted(df[col].dropna().unique().astype(str).tolist())
            for col in df.columns
        }

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Apply one-hot encoding using the categories learned at fit time."""
        df, _ = data["input"]

        encoded_frames: list[pd.DataFrame] = []
        encoded_columns: list[str] = []
        for col in df.columns:
            categories = self._params.get(col, [])
            dummies = pd.DataFrame(
                0,
                index=df.index,
                columns=[f"{col}_{cat}" for cat in categories],
                dtype=np.uint8,
            )
            for cat in categories:
                mask = df[col].astype(str) == cat
                dummies.loc[mask, f"{col}_{cat}"] = 1
            encoded_frames.append(dummies)
            encoded_columns.extend(dummies.columns.tolist())

        result = pd.concat(encoded_frames, axis=1) if encoded_frames else pd.DataFrame()

        ctx = TabularDataContext(
            columns=encoded_columns,
            dtypes=[np.dtype("uint8")] * len(encoded_columns),
            categories=[NumericalData] * len(encoded_columns),
        )
        return {"output": (result, ctx)}

    def get_params(self) -> _OneHotParams:
        """Return the per-column category mapping learned during fit."""
        return self._params

    def set_params(self, params: _OneHotParams) -> None:
        """Restore a previously fitted category mapping."""
        self._params = params
        self._fitted = True
