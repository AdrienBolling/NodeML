"""GradientBoostingRegressor model node for the NodeML Framework.

Wraps ``sklearn.ensemble.GradientBoostingRegressor`` and exposes it as a NodeML
:class:`~nodeml.core.nodes.models.model.Model` node.  The node expects two
input ports:

* ``X`` - numerical feature matrix ``(batch, features)`` as a numpy array
* ``y`` - one target column ``(batch, 1)`` as a numpy array (training / evaluation only)

and emits one output port:

* ``pred`` - float64 predictions ``(batch, 1)`` as a numpy array

scikit-learn's GradientBoostingRegressor supports one target only.  A ``y``
with more than one column causes a
:class:`~nodeml.core.common.exceptions.NodeInputError`.
"""

from typing import Any, Literal

from pydantic import Field
from ray import tune
from sklearn.ensemble import (
    GradientBoostingRegressor as SklearnGradientBoostingRegressor,
)

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


class GradientBoostingRegressorMetadata(ModelMetadata):
    """Metadata for the GradientBoostingRegressor model node."""

    node_name: str = "GradientBoostingRegressor"
    description: str = (
        "Gradient Boosting Regressor based on scikit-learn. "
        "Builds an additive model of shallow decision trees trained "
        "sequentially to correct the residuals of the ensemble."
    )


class GradientBoostingRegressorHyperParameters(ModelHyperParameters):
    """Tuneable hyperparameters for the GradientBoostingRegressor.

    These are the parameters that make sense to explore during
    hyperparameter search.  All other sklearn knobs live in
    :class:`GradientBoostingRegressorRunningConfig`.
    """

    n_estimators: int = Field(
        default=100,
        ge=1,
        description=(
            "Number of boosting stages (trees) to fit. "
            "More stages reduce bias but increase the risk of overfitting "
            "and training time."
        ),
    )
    max_depth: int = Field(
        default=3,
        ge=1,
        description=(
            "Maximum depth of each individual regression tree. "
            "Shallow trees (3-5) act as weak learners and are the typical "
            "choice for gradient boosting."
        ),
    )
    learning_rate: float = Field(
        default=0.1,
        gt=0,
        description=(
            "Shrinkage factor applied to each tree's contribution. "
            "Smaller values require more boosting stages but often "
            "yield better generalisation."
        ),
    )


class GradientBoostingRegressorRunningConfig(ModelRunningConfig):
    """Execution-time options for the GradientBoostingRegressor.

    These affect training behaviour but are usually held fixed during
    hyperparameter search.
    """

    loss: Literal[
        "squared_error",
        "absolute_error",
        "huber",
        "quantile",
    ] = Field(
        default="squared_error",
        description=(
            "Loss function to optimise. "
            "``'squared_error'`` minimises MSE. "
            "``'absolute_error'`` minimises MAD. "
            "``'huber'`` is a combination of both, robust to outliers. "
            "``'quantile'`` allows quantile regression."
        ),
    )
    random_state: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Seed for the internal random-number generator. "
            "Set to a non-negative integer for fully reproducible training runs. "
            "``None`` uses the global numpy random state."
        ),
    )


# Exposed at module level so external tuners can discover the search space.
hyperparameter_space: dict[str, Any] = {
    "n_estimators": tune.choice([50, 100, 200, 500]),
    "max_depth": tune.choice([3, 5, 7, 10]),
    "learning_rate": tune.choice([0.01, 0.05, 0.1, 0.2]),
}


class GradientBoostingRegressorConfig(
    ModelConfig[
        GradientBoostingRegressorHyperParameters,
        GradientBoostingRegressorRunningConfig,
    ]
):
    """Full configuration for the GradientBoostingRegressor node."""

    hyperparameters: GradientBoostingRegressorHyperParameters = Field(
        default_factory=GradientBoostingRegressorHyperParameters,
        description="Tuneable hyperparameters (n_estimators, max_depth, learning_rate).",
    )
    running_config: GradientBoostingRegressorRunningConfig = Field(
        default_factory=GradientBoostingRegressorRunningConfig,
        description="Execution-time options (loss, random_state).",
    )
    in_ports: dict[str, Port] = Field(
        default={
            "X": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch features",
                desc="Numerical input feature matrix (batch, features).",
            ),
            "y": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch 1",
                mode=[NodeExecutionMode.TRAINING, NodeExecutionMode.EVALUATION],
                desc="Target values (batch, 1). Required during training and evaluation only.",
            ),
        },
        description="Input ports: 'X' (features, all modes) and 'y' (targets, training/evaluation).",
    )
    out_ports: dict[str, Port] = Field(
        default={
            "pred": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch 1",
                desc="Predicted values (float64), one column.",
            ),
        },
        description="Output ports: 'pred' (predicted targets).",
    )


class GradientBoostingRegressorNode(SklearnModelNode):
    """Gradient Boosting Regressor model node.

    :class:`SklearnModelNode` builds the estimator from
    :class:`GradientBoostingRegressorConfig` at each :meth:`fit`.  The
    estimator supports one target column only.  The prediction is float64
    and its column takes the name of the training target.
    """

    metadata = GradientBoostingRegressorMetadata()
    hyperparameter_space = hyperparameter_space
    estimator_class = SklearnGradientBoostingRegressor
    multi_output = False

    def _estimator_kwargs(self) -> dict[str, Any]:
        """Return the estimator arguments from the node config."""
        hp = self._config.hyperparameters
        rc = self._config.running_config
        return {
            "n_estimators": hp.n_estimators,
            "max_depth": hp.max_depth,
            "learning_rate": hp.learning_rate,
            "loss": rc.loss,
            "random_state": rc.random_state,
        }
