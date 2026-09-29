"""Robust scaler transform node for the NodeML Framework.

Scales numerical columns using median and interquartile range (IQR),
making the transformation robust to outliers::

    x_scaled = (x - median) / IQR

where IQR = Q3 - Q1.
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

type _RobustScalerParams = dict[str, dict[str, float]]


class RobustScalerMetadata(TransformMetadata):
    """Metadata for the RobustScaler node."""

    node_name: str = "RobustScaler"
    description: str = (
        "Scale numerical columns using median and IQR, robust to outliers."
    )
    trainable: bool = True


class RobustScalerRunningConfig(TransformRunningConfig):
    """No run-time knobs."""


class RobustScalerHyperParameters(TransformHyperParameters):
    """No tuneable hyperparameters."""


class RobustScalerConfig(
    TransformConfig[RobustScalerRunningConfig, RobustScalerHyperParameters]
):
    """Configuration for the RobustScaler node."""

    hyperparameters: RobustScalerHyperParameters = Field(
        default_factory=RobustScalerHyperParameters
    )
    running_config: RobustScalerRunningConfig = Field(
        default_factory=RobustScalerRunningConfig
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
                desc="Robust-scaled numerical DataFrame.",
            )
        },
    )


class RobustScaler(
    TransformNode[
        pd.DataFrame,
        TabularDataContext,
        pd.DataFrame,
        TabularDataContext,
        _RobustScalerParams,
    ]
):
    """Scale numerical columns using median and IQR, robust to outliers."""

    metadata = RobustScalerMetadata()

    def __init__(self, *, config: RobustScalerConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config
        self._params: _RobustScalerParams = {"median": {}, "iqr": {}}
        self._fitted = False

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Learn per-column median and IQR from the input data.

        Args:
            data: Dictionary mapping port names to (DataFrame, context) tuples.

        """
        df, _ = data["input"]
        medians = df.median()
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqrs = q3 - q1
        self._params = {
            "median": {col: float(medians[col]) for col in df.columns},
            "iqr": {col: float(iqrs[col]) for col in df.columns},
        }

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Apply robust scaling to the input data.

        Args:
            data: Dictionary mapping port names to (DataFrame, context) tuples.

        Returns:
            Dictionary mapping port names to (scaled DataFrame, context) tuples.

        """
        df, ctx = data["input"]
        median = pd.Series(self._params["median"])
        iqr = pd.Series(self._params["iqr"])
        iqr = iqr.replace(0.0, 1.0)
        result = df.sub(median, axis=1).div(iqr, axis=1)
        return {"output": (result, ctx)}

    def get_params(self) -> _RobustScalerParams:
        """Return the learned median and IQR parameters."""
        return self._params

    def set_params(self, params: _RobustScalerParams) -> None:
        """Set the median and IQR parameters.

        Args:
            params: Dictionary with 'median' and 'iqr' keys mapping column names to values.

        """
        self._params = params
        self._fitted = True
