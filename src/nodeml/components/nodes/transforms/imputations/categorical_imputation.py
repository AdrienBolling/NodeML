"""Categorical imputation transform node for the NodeML Framework.

Fills missing values in categorical columns using one of two strategies:

* ``"most_frequent"`` - replace NaNs with the most frequent value per column
  (learned at fit time).
* ``"constant"``      - replace NaNs with a user-supplied constant string.

Per-column fill values are persisted via :meth:`get_params` /
:meth:`set_params` for checkpointing.
"""

from typing import Literal

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

# Serialisable params: column name -> fill value.
type _CategoricalImputationParams = dict[str, str]


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
        CategoricalImputationRunningConfig,
        CategoricalImputationHyperParameters,
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
    -------
    >>> cfg = CategoricalImputationConfig(
    ...     hyperparameters=CategoricalImputationHyperParameters(strategy="constant", value="N/A"),
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
        """Learn per-column fill values from the training data."""
        df, _ = data["input"]
        hp = self._config.hyperparameters

        match hp.strategy:
            case "most_frequent":
                self._params = {
                    col: str(df[col].mode().iloc[0])
                    if not df[col].mode().empty
                    else hp.value
                    for col in df.columns
                }
            case "constant":
                self._params = dict.fromkeys(df.columns, hp.value)

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Fill missing values using the values learned at fit time."""
        df, ctx = data["input"]
        result = df.fillna(self._params)
        return {"output": (result, ctx)}

    def get_params(self) -> _CategoricalImputationParams:
        """Return the per-column fill values."""
        return self._params

    def set_params(self, params: _CategoricalImputationParams) -> None:
        """Restore previously fitted fill values."""
        self._params = params
        self._fitted = True
