"""RandomForestRegressor model node for the NodeML Framework.

Wraps ``sklearn.ensemble.RandomForestRegressor`` and exposes it as a NodeML
:class:`~nodeml.core.nodes.models.model.Model` node.  The node expects two
input ports:

* ``X`` - feature matrix ``(batch, features)`` as a numpy array
* ``y`` - target matrix ``(batch, targets)`` as a numpy array (training / evaluation only)

and emits one output port:

* ``pred`` - float64 predictions ``(batch, targets)`` as a numpy array

scikit-learn supports multi-output regression when ``y`` has more than one
column.
"""

from typing import Any, Literal

from pydantic import Field
from ray import tune
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor

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


class RandomForestRegressorMetadata(ModelMetadata):
    """Metadata for the RandomForestRegressor model node."""

    node_name: str = "RandomForestRegressor"
    description: str = (
        "Random Forest Regressor based on scikit-learn. "
        "Supports both single- and multi-output regression."
    )


class RandomForestRegressorHyperParameters(ModelHyperParameters):
    """Tuneable hyperparameters for the RandomForestRegressor.

    These are the parameters that make sense to explore during
    hyperparameter search.  All other sklearn knobs live in
    :class:`RandomForestRegressorRunningConfig`.
    """

    n_estimators: int = Field(
        default=100,
        ge=1,
        description=(
            "Number of trees in the forest. "
            "More trees reduce variance but increase memory usage and training time."
        ),
    )
    max_depth: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum depth of each tree. "
            "``None`` grows trees until all leaves are pure or contain fewer than "
            "``min_samples_split`` samples. "
            "Shallower trees regularise more aggressively."
        ),
    )


class RandomForestRegressorRunningConfig(ModelRunningConfig):
    """Execution-time options for the RandomForestRegressor.

    These affect training behaviour but are usually held fixed during
    hyperparameter search.
    """

    criterion: Literal[
        "squared_error",
        "absolute_error",
        "friedman_mse",
        "poisson",
    ] = Field(
        default="squared_error",
        description=(
            "Impurity measure used to evaluate the quality of a split. "
            "``'squared_error'`` minimises MSE (variance reduction). "
            "``'absolute_error'`` minimises MAD. "
            "``'friedman_mse'`` uses Friedman's improvement score. "
            "``'poisson'`` is suited to non-negative count targets."
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
    "max_depth": tune.choice([None, 5, 10, 20, 30]),
}


class RandomForestRegressorConfig(
    ModelConfig[
        RandomForestRegressorHyperParameters,
        RandomForestRegressorRunningConfig,
    ]
):
    """Full configuration for the RandomForestRegressor node."""

    hyperparameters: RandomForestRegressorHyperParameters = Field(
        default_factory=RandomForestRegressorHyperParameters,
        description="Tuneable hyperparameters (n_estimators, max_depth).",
    )
    running_config: RandomForestRegressorRunningConfig = Field(
        default_factory=RandomForestRegressorRunningConfig,
        description="Execution-time options (criterion, random_state).",
    )
    in_ports: dict[str, Port] = Field(
        default={
            "X": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.MIXED,
                data_shape="batch features",
                desc="Input feature matrix (batch, features).",
            ),
            "y": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch targets",
                mode=[NodeExecutionMode.TRAINING, NodeExecutionMode.EVALUATION],
                desc="Target values (batch, targets). Required during training and evaluation only.",
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
                data_shape="batch targets",
                desc="Predicted values (float64), one column for each target.",
            ),
        },
        description="Output ports: 'pred' (predicted targets).",
    )


class RandomForestRegressorNode(SklearnModelNode):
    """Random Forest Regressor model node.

    :class:`SklearnModelNode` builds the estimator from
    :class:`RandomForestRegressorConfig` at each :meth:`fit`.  The
    predictions are float64 and the output columns take the names of the
    training targets.
    """

    metadata = RandomForestRegressorMetadata()
    hyperparameter_space = hyperparameter_space
    estimator_class = SklearnRandomForestRegressor

    def _estimator_kwargs(self) -> dict[str, Any]:
        """Return the estimator arguments from the node config."""
        hp = self._config.hyperparameters
        rc = self._config.running_config
        return {
            "n_estimators": hp.n_estimators,
            "max_depth": hp.max_depth,
            "criterion": rc.criterion,
            "random_state": rc.random_state,
        }
