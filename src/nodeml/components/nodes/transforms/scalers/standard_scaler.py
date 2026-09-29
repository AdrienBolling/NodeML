"""Standard scaler transform node for the NodeML Framework.

Standardises numerical columns to zero mean and unit variance::

    x_scaled = (x - mean) / std

Per-column **mean** and **std** are learned during :meth:`fit` and reused
at :meth:`transform` time.  Columns with zero standard deviation are left
untouched (shifted to zero mean only) to avoid division by zero.
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

# Serialisable params: {"mean": {col: float}, "std": {col: float}}
type _StandardScalerParams = dict[str, dict[str, float]]


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
        StandardScalerRunningConfig,
        StandardScalerHyperParameters,
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


class StandardScaler(
    TransformNode[
        pd.DataFrame,
        TabularDataContext,
        pd.DataFrame,
        TabularDataContext,
        _StandardScalerParams,
    ],
):
    """Standardise numerical columns to zero mean and unit variance.

    Example:
    -------
    >>> node = StandardScaler(config=StandardScalerConfig())
    >>> out = node.node_fit_transform({"input": (df, ctx)})

    """

    metadata = StandardScalerMetadata()

    def __init__(self, *, config: StandardScalerConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config
        self._params: _StandardScalerParams = {"mean": {}, "std": {}}
        self._fitted = False

    # --- TransformNode interface ------------------------------------------

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Learn per-column mean and standard deviation."""
        df, _ = data["input"]
        means = df.mean()
        stds = df.std()
        self._params = {
            "mean": {col: float(means[col]) for col in df.columns},
            "std": {col: float(stds[col]) for col in df.columns},
        }

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Apply (x - mean) / std using the statistics learned at fit time."""
        df, ctx = data["input"]
        mean = pd.Series(self._params["mean"])
        std = pd.Series(self._params["std"])
        # Replace zero std with 1 so those columns are only mean-centred.
        std = std.replace(0.0, 1.0)
        result = df.sub(mean, axis=1).div(std, axis=1)
        return {"output": (result, ctx)}

    def get_params(self) -> _StandardScalerParams:
        """Return the per-column mean and std."""
        return self._params

    def set_params(self, params: _StandardScalerParams) -> None:
        """Restore previously fitted statistics."""
        self._params = params
        self._fitted = True
