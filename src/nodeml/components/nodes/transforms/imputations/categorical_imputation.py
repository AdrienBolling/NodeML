"""Categorical imputation transform node for the NodeML Framework.

Fills missing values in categorical columns using one of two strategies:

* ``"most_frequent"`` - replace NaNs with the most frequent value per column
  (learned at fit time).  A column with only missing values gets ``value``.
* ``"constant"``      - replace NaNs with a user-supplied constant string.

The output keeps the dtype of each column when the fill value fits it.  A
pandas ``Categorical`` column gets the fill value as a new category when
necessary.  When the fill value does not fit the dtype (for example, the
string ``"missing"`` in an ``Int64`` column), the column becomes ``object``.

Per-column fill values are persisted via :meth:`get_params` /
:meth:`set_params` for checkpointing.
"""

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from nodeml.components.utils.dataframe import check_fitted_columns, fill_missing
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

# Serialisable params: column name -> fill value (a Python scalar).
type _CategoricalImputationParams = dict[str, Any]


class CategoricalImputationMetadata(TransformMetadata):
    """Metadata for the CategoricalImputation node."""

    node_name: str = "CategoricalImputation"
    description: str = (
        "Fill missing values in categorical columns using most_frequent or a constant."
    )
    trainable: bool = True


class CategoricalImputationRunningConfig(TransformRunningConfig):
    """No run-time knobs for this node."""


class CategoricalImputationHyperParameters(TransformHyperParameters):
    """Tuneable hyperparameters for the CategoricalImputation node."""

    strategy: Literal["most_frequent", "constant"] = Field(
        default="most_frequent",
        description=(
            "Imputation strategy. "
            "``'most_frequent'`` uses the mode per column (learned at fit). "
            "``'constant'`` uses the ``value`` field."
        ),
    )
    value: str = Field(
        default="missing",
        description="Fill value used when strategy is ``'constant'``.",
    )


class CategoricalImputationConfig(
    TransformConfig[
        CategoricalImputationHyperParameters,
        CategoricalImputationRunningConfig,
    ],
):
    """Full configuration for the CategoricalImputation node."""

    hyperparameters: CategoricalImputationHyperParameters = Field(
        default_factory=CategoricalImputationHyperParameters,
        description="Imputation strategy and constant value.",
    )
    running_config: CategoricalImputationRunningConfig = Field(
        default_factory=CategoricalImputationRunningConfig,
        description="No run-time knobs for this node.",
    )
    in_ports: dict[str, Port] = Field(
        default={
            "input": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.CATEGORICAL,
                data_shape="batch feature",
                desc="Categorical DataFrame with potential missing values.",
            ),
        },
        description="Input port: 'input' (categorical DataFrame).",
    )
    out_ports: dict[str, Port] = Field(
        default={
            "output": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.CATEGORICAL,
                data_shape="batch feature",
                desc="Categorical DataFrame with missing values filled.",
            ),
        },
        description="Output port: 'output' (imputed categorical DataFrame).",
    )


class CategoricalImputation(
    TransformNode[
        pd.DataFrame,
        TabularDataContext,
        pd.DataFrame,
        TabularDataContext,
        _CategoricalImputationParams,
    ],
):
    """Fill missing values in categorical columns.

    Example:
        >>> cfg = CategoricalImputationConfig(
        ...     hyperparameters=CategoricalImputationHyperParameters(
        ...         strategy="constant", value="N/A"
        ...     ),
        ... )
        >>> node = CategoricalImputation(config=cfg)
        >>> out = node.node_fit_transform({"input": (df, ctx)})

    """

    metadata = CategoricalImputationMetadata()

    def __init__(self, *, config: CategoricalImputationConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config
        self._params: _CategoricalImputationParams = {}
        self._fitted = False

    # --- TransformNode interface ------------------------------------------

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Learn the fill value of each column of ``data["input"]``.

        Args:
            data: Must contain the key ``"input"``.

        """
        df, _ = data["input"]
        hp = self._config.hyperparameters

        match hp.strategy:
            case "most_frequent":
                self._params = {
                    col: _most_frequent(df[col], default=hp.value) for col in df.columns
                }
            case "constant":
                self._params = dict.fromkeys(df.columns, hp.value)

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Fill the missing values with the values learned during fit.

        Args:
            data: Must contain the key ``"input"``.

        Returns:
            ``{"output": (df, ctx)}``.  The context describes the output
            dtypes.

        Raises:
            NodeInputError: If the input columns are not the fitted columns.

        """
        df, ctx = data["input"]
        check_fitted_columns(
            self._params, df.columns, node_name="CategoricalImputation"
        )
        result = df.copy()
        for col in df.columns:
            result[col] = fill_missing(
                df[col], self._params[col], fallback_dtype="object"
            )
        return {"output": (result, ctx.aligned_to(result))}

    def get_params(self) -> _CategoricalImputationParams:
        """Return the fill value of each column.

        Returns:
            A mapping of column name to fill value.

        """
        return self._params

    def set_params(self, params: _CategoricalImputationParams) -> None:
        """Restore the fill values returned by :meth:`get_params`.

        Args:
            params: The params of a fitted CategoricalImputation node.

        """
        self._params = params
        self._fitted = True


def _most_frequent(series: pd.Series, *, default: str) -> Any:
    """Return the most frequent value of *series*, or *default* if it has none.

    A numpy scalar becomes a Python scalar, so that the params stay simple.
    """
    mode = series.mode(dropna=True)
    if mode.empty:
        return default
    value = mode.iloc[0]
    return value.item() if isinstance(value, np.generic) else value
