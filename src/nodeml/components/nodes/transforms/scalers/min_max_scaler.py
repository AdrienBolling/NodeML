"""Min-max scaler transform node for the NodeML Framework.

Scales numerical columns to the [0, 1] range::

    x_scaled = (x - min) / (max - min)
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

type _MinMaxScalerParams = dict[str, dict[str, float]]


class MinMaxScalerMetadata(TransformMetadata):
    """Metadata for the MinMaxScaler node."""

    node_name: str = "MinMaxScaler"
    description: str = "Scale numerical columns to the [0, 1] range."
    trainable: bool = True


class MinMaxScalerRunningConfig(TransformRunningConfig):
    """No run-time knobs."""


class MinMaxScalerHyperParameters(TransformHyperParameters):
    """No tuneable hyperparameters."""


class MinMaxScalerConfig(
    TransformConfig[MinMaxScalerRunningConfig, MinMaxScalerHyperParameters]
):
    """Configuration for the MinMaxScaler node."""

    hyperparameters: MinMaxScalerHyperParameters = Field(
        default_factory=MinMaxScalerHyperParameters
    )
    running_config: MinMaxScalerRunningConfig = Field(
        default_factory=MinMaxScalerRunningConfig
    )
    in_ports: dict[str, Port] = Field(
        default={
            "input": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch feature",
                desc="Numerical DataFrame to scale.",
            )
        },
    )
    out_ports: dict[str, Port] = Field(
        default={
            "output": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch feature",
                desc="Min-max scaled numerical DataFrame.",
            )
        },
    )


class MinMaxScaler(
    TransformNode[
        pd.DataFrame,
        TabularDataContext,
        pd.DataFrame,
        TabularDataContext,
        _MinMaxScalerParams,
    ]
):
    """Scale numerical columns to the [0, 1] range using per-column min and max."""

    metadata = MinMaxScalerMetadata()

    def __init__(self, *, config: MinMaxScalerConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config
        self._params: _MinMaxScalerParams = {"min": {}, "max": {}}
        self._fitted = False

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Learn per-column min and max from the input data.

        Args:
            data: Dictionary mapping port names to (DataFrame, context) tuples.

        """
        df, _ = data["input"]
        mins = df.min()
        maxs = df.max()
        self._params = {
            "min": {col: float(mins[col]) for col in df.columns},
            "max": {col: float(maxs[col]) for col in df.columns},
        }

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Apply min-max scaling to the input data.

        Args:
            data: Dictionary mapping port names to (DataFrame, context) tuples.

        Returns:
            Dictionary mapping port names to (scaled DataFrame, context) tuples.

        """
        df, ctx = data["input"]
        col_min = pd.Series(self._params["min"])
        col_max = pd.Series(self._params["max"])
        col_range = col_max - col_min
        col_range = col_range.replace(0.0, 1.0)
        result = df.sub(col_min, axis=1).div(col_range, axis=1)
        return {"output": (result, ctx)}

    def get_params(self) -> _MinMaxScalerParams:
        """Return the learned min and max parameters."""
        return self._params

    def set_params(self, params: _MinMaxScalerParams) -> None:
        """Set the min and max parameters.

        Args:
            params: Dictionary with 'min' and 'max' keys mapping column names to values.

        """
        self._params = params
        self._fitted = True
