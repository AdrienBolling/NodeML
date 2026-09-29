"""LinearRegression model node for the NodeML Framework.

Wraps ``sklearn.linear_model.LinearRegression`` and exposes it as a NodeML
:class:`~nodeml.core.nodes.models.model.Model` node.  The node expects two
input ports:

* ``X`` - numerical feature matrix ``(batch, features)`` as a numpy array
* ``y`` - target matrix ``(batch, targets)`` as a numpy array (training / evaluation only)

and emits one output port:

* ``pred`` - float64 predictions ``(batch, targets)`` as a numpy array

scikit-learn supports multi-output regression when ``y`` has more than one
column.
"""

from typing import Any

from pydantic import Field
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from nodeml.components.nodes.models._sklearn_base import SklearnModelNode
from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.nodes.models.model import (
    ModelConfig,
    ModelHyperParameters,
    ModelMetadata,
    ModelRunningConfig,
)
from nodeml.core.nodes.node import Port


class LinearRegressionMetadata(ModelMetadata):
    """Registry metadata for the LinearRegression node."""

    node_name: str = "LinearRegression"
    description: str = (
        "Linear Regression based on scikit-learn. "
        "Supports both single- and multi-output regression."
    )


class LinearRegressionHyperParameters(ModelHyperParameters):
    """LinearRegression has no tunable hyperparameters."""


class LinearRegressionRunningConfig(ModelRunningConfig):
    """Runtime options for the LinearRegression node."""

    fit_intercept: bool = Field(
        default=True, description="Whether to calculate the intercept for this model."
    )


hyperparameter_space: dict[str, Any] = {}


class LinearRegressionConfig(
    ModelConfig[LinearRegressionHyperParameters, LinearRegressionRunningConfig]
):
    """Configuration of the LinearRegression node: X and y in, pred out."""

    hyperparameters: LinearRegressionHyperParameters = Field(
        default_factory=LinearRegressionHyperParameters
    )
    running_config: LinearRegressionRunningConfig = Field(
        default_factory=LinearRegressionRunningConfig
    )
    in_ports: dict[str, Port] = Field(
        default={
            "X": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch features",
                desc="Numerical input feature matrix.",
            ),
            "y": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch targets",
                mode=[NodeExecutionMode.TRAINING, NodeExecutionMode.EVALUATION],
                desc="Target values.",
            ),
        },
    )
    out_ports: dict[str, Port] = Field(
        default={
            "pred": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch targets",
                desc="Predicted values (float64), one column for each target.",
            )
        },
    )


class LinearRegressionNode(SklearnModelNode):
    """Ordinary least squares regression (scikit-learn ``LinearRegression``)."""

    metadata = LinearRegressionMetadata()
    hyperparameter_space = hyperparameter_space
    estimator_class = SklearnLinearRegression

    def _estimator_kwargs(self) -> dict[str, Any]:
        """Return the estimator arguments from the running config."""
        return {"fit_intercept": self._config.running_config.fit_intercept}
