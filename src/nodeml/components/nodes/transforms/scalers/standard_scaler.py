"""Standard scaler transform node for the NodeML Framework.

Standardises numerical columns to zero mean and unit variance::

    x_scaled = (x - mean) / std

The scaler learns the **mean** and the **std** (``ddof=1``) of each column
during :meth:`fit` and uses them again at :meth:`transform` time.  A column
with a zero or undefined std (for example, a fit on one row) is only
centred, to prevent a division by zero.  See :mod:`._affine_scaler` for the
shared behaviour.
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


class StandardScalerMetadata(TransformMetadata):
    """Metadata for the StandardScaler node."""

    node_name: str = "StandardScaler"
    description: str = "Standardise numerical columns to zero mean and unit variance."
    trainable: bool = True


class StandardScalerRunningConfig(TransformRunningConfig):
    """No run-time knobs for this node."""


class StandardScalerHyperParameters(TransformHyperParameters):
    """No tuneable hyperparameters."""


class StandardScalerConfig(
    TransformConfig[
        StandardScalerHyperParameters,
        StandardScalerRunningConfig,
    ],
):
    """Full configuration for the StandardScaler node."""

    hyperparameters: StandardScalerHyperParameters = Field(
        default_factory=StandardScalerHyperParameters,
        description="No tuneable hyperparameters for this node.",
    )
    running_config: StandardScalerRunningConfig = Field(
        default_factory=StandardScalerRunningConfig,
        description="No run-time knobs for this node.",
    )
    in_ports: dict[str, Port] = Field(
        default={
            "input": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch feature",
                desc="Numerical DataFrame to standardise.",
            ),
        },
        description="Input port: 'input' (numerical DataFrame).",
    )
    out_ports: dict[str, Port] = Field(
        default={
            "output": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch feature",
                desc="Standardised numerical DataFrame.",
            ),
        },
        description="Output port: 'output' (scaled numerical DataFrame).",
    )


class StandardScaler(AffineScaler):
    """Standardise numerical columns to zero mean and unit variance.

    Example:
        >>> node = StandardScaler(config=StandardScalerConfig())
        >>> out = node.node_fit_transform({"input": (df, ctx)})

    """

    metadata = StandardScalerMetadata()

    @staticmethod
    def _empty_params() -> ScalerParams:
        """Return the params of a scaler that is not fitted."""
        return {"mean": {}, "std": {}}

    @staticmethod
    def _fit_statistics(df: pd.DataFrame) -> ScalerParams:
        """Return the mean and the std of each column of *df*."""
        means = df.mean()
        stds = df.std()
        return {
            "mean": {col: float(means[col]) for col in df.columns},
            "std": {col: float(stds[col]) for col in df.columns},
        }

    def _center_and_scale(self) -> tuple[pd.Series, pd.Series]:
        """Return the mean as the center and the std as the scale."""
        return (
            pd.Series(self._params["mean"], dtype="float64"),
            pd.Series(self._params["std"], dtype="float64"),
        )
