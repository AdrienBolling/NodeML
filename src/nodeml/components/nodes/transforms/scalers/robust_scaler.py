"""Robust scaler transform node for the NodeML Framework.

Scales numerical columns with the median and the interquartile range
(IQR), so that outliers have a small effect on the transformation::

    x_scaled = (x - median) / IQR

where IQR = Q3 - Q1.  The scaler learns the **median** and the **IQR** of
each column during :meth:`fit` and uses them again at :meth:`transform`
time.  A column with a zero IQR is only centred on its median.  See
:mod:`._affine_scaler` for the shared behaviour.
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
    TransformConfig[RobustScalerHyperParameters, RobustScalerRunningConfig]
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


class RobustScaler(AffineScaler):
    """Scale numerical columns with the median and the IQR, robust to outliers.

    Example:
        >>> node = RobustScaler(config=RobustScalerConfig())
        >>> out = node.node_fit_transform({"input": (df, ctx)})

    """

    metadata = RobustScalerMetadata()

    @staticmethod
    def _empty_params() -> ScalerParams:
        """Return the params of a scaler that is not fitted."""
        return {"median": {}, "iqr": {}}

    @staticmethod
    def _fit_statistics(df: pd.DataFrame) -> ScalerParams:
        """Return the median and the IQR of each column of *df*."""
        medians = df.median()
        iqrs = df.quantile(0.75) - df.quantile(0.25)
        return {
            "median": {col: float(medians[col]) for col in df.columns},
            "iqr": {col: float(iqrs[col]) for col in df.columns},
        }

    def _center_and_scale(self) -> tuple[pd.Series, pd.Series]:
        """Return the median as the center and the IQR as the scale."""
        return (
            pd.Series(self._params["median"], dtype="float64"),
            pd.Series(self._params["iqr"], dtype="float64"),
        )
