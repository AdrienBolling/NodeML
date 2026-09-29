"""Min-max scaler transform node for the NodeML Framework.

Scales numerical columns to the [0, 1] range::

    x_scaled = (x - min) / (max - min)

The scaler learns the **min** and the **max** of each column during
:meth:`fit` and uses them again at :meth:`transform` time.  A column with
a zero range (for example, a constant column) is only centred on its min.
See :mod:`._affine_scaler` for the shared behaviour.
"""

import pandas as pd
from pydantic import Field

from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
)
from nodeml.core.nodes.node import Port
from nodeml.core.nodes.transform.transform import (
    TransformConfig,
    TransformHyperParameters,
    TransformMetadata,
    TransformRunningConfig,
)

from ._affine_scaler import AffineScaler, ScalerParams


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
    TransformConfig[MinMaxScalerHyperParameters, MinMaxScalerRunningConfig]
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


class MinMaxScaler(AffineScaler):
    """Scale numerical columns to the [0, 1] range with the min and max of each column.

    Example:
        >>> node = MinMaxScaler(config=MinMaxScalerConfig())
        >>> out = node.node_fit_transform({"input": (df, ctx)})

    """

    metadata = MinMaxScalerMetadata()

    @staticmethod
    def _empty_params() -> ScalerParams:
        """Return the params of a scaler that is not fitted."""
        return {"min": {}, "max": {}}

    @staticmethod
    def _fit_statistics(df: pd.DataFrame) -> ScalerParams:
        """Return the min and the max of each column of *df*."""
        mins = df.min()
        maxs = df.max()
        return {
            "min": {col: float(mins[col]) for col in df.columns},
            "max": {col: float(maxs[col]) for col in df.columns},
        }

    def _center_and_scale(self) -> tuple[pd.Series, pd.Series]:
        """Return the min as the center and the range as the scale."""
        col_min = pd.Series(self._params["min"], dtype="float64")
        col_max = pd.Series(self._params["max"], dtype="float64")
        return col_min, col_max - col_min
