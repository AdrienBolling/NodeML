"""One-Hot Encoding transform node for the NodeML Framework.

Takes categorical columns and returns them as one-hot-encoded numerical
columns.  During :meth:`fit` the node finds the sorted distinct categories
of each column.  During :meth:`transform` it applies the same mapping.  A
missing value, or a category that fit did not see, gives zeros in all the
dummy columns of its column.

The node compares real values, not strings: ``1.0`` and ``1`` are the same
category, and its dummy column is ``<column>_1``.  Numbers are sorted by
value.  See :mod:`._categories`.

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
from nodeml.core.common.exceptions import NodeInputError
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
type _OneHotParams = dict[str, list[Hashable]]


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
        OneHotEncodingHyperParameters,
        OneHotEncodingRunningConfig,
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

    During :meth:`fit`, the node captures the sorted categories of each
    column.  :meth:`transform` applies the same encoding.  It makes one
    ``uint8`` column ``<original_col>_<category>`` per category.  The output
    keeps the input index, and follows the input column order.

    Example:
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
        """Learn the sorted categories of each column of ``data["input"]``.

        Args:
            data: Must contain the key ``"input"``.

        Raises:
            NodeInputError: If two dummy columns get the same name (for
                example, the string ``"1"`` and the integer ``1``).

        """
        df, _ = data["input"]
        params = {col: learn_categories(df[col]) for col in df.columns}
        names = [
            name
            for col, categories in params.items()
            for name in _dummy_names(col, categories)
        ]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            msg = (
                f"OneHotEncoding: the dummy column names {duplicates} are not unique. "
                "Rename the columns or the categories before the encoding."
            )
            raise NodeInputError(msg)
        self._params = params

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Encode each column with the categories learned during fit.

        Args:
            data: Must contain the key ``"input"``.

        Returns:
            ``{"output": (df, ctx)}`` with the ``uint8`` dummy columns.

        Raises:
            NodeInputError: If the input columns are not the fitted columns.

        """
        df, _ = data["input"]
        check_fitted_columns(self._params, df.columns, node_name="OneHotEncoding")
        dummies: dict[str, np.ndarray] = {}
        for col in df.columns:
            categories = self._params[col]
            codes = encode_codes(df[col], categories)
            one_hot = codes[:, None] == np.arange(len(categories))[None, :]
            for position, name in enumerate(_dummy_names(col, categories)):
                dummies[name] = one_hot[:, position].astype(np.uint8)
        result = pd.DataFrame(dummies, index=df.index, columns=list(dummies))
        ctx = TabularDataContext(
            columns=list(result.columns),
            dtypes=[np.dtype("uint8")] * result.shape[1],
            categories=[NumericalData] * result.shape[1],
        )
        return {"output": (result, ctx)}

    def get_params(self) -> _OneHotParams:
        """Return the categories learned during fit.

        Returns:
            A mapping of column name to its sorted categories.

        """
        return self._params

    def set_params(self, params: _OneHotParams) -> None:
        """Restore the categories returned by :meth:`get_params`.

        Args:
            params: The params of a fitted OneHotEncoding node.

        """
        self._params = params
        self._fitted = True


def _dummy_names(col: object, categories: list[Hashable]) -> list[str]:
    """Return the dummy column names of *col*, one per category."""
    return [f"{col}_{category}" for category in categories]
