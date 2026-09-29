"""GradientBoostingClassifier model node for the NodeML Framework.

Wraps ``sklearn.ensemble.GradientBoostingClassifier`` and exposes it as a NodeML
:class:`~nodeml.core.nodes.models.model.Model` node.  The node expects two
input ports:

* ``X`` - numerical feature matrix ``(batch, features)`` as a numpy array
* ``y`` - class labels ``(batch, 1)`` as a numpy array (training / evaluation only)

and emits one output port:

* ``pred`` - float64 class probabilities ``(batch, classes)`` as a numpy array

``pred`` has one column ``proba_<class>`` for each class, in the order of
the ``classes_`` attribute of the estimator.  A binary problem also gives
two columns.  The classification metrics expect integer class labels
``0 .. C-1``.
"""

from typing import Any, Literal

from pydantic import Field
from ray import tune
from sklearn.ensemble import (
    GradientBoostingClassifier as SklearnGradientBoostingClassifier,
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


class GradientBoostingClassifierMetadata(ModelMetadata):
    """Metadata for the GradientBoostingClassifier model node."""

    node_name: str = "GradientBoostingClassifier"
    description: str = (
        "Gradient Boosting Classifier based on scikit-learn. "
        "Builds an additive model of shallow decision trees trained "
        "sequentially to minimise a classification loss function."
    )


class GradientBoostingClassifierHyperParameters(ModelHyperParameters):
    """Tuneable hyperparameters for the GradientBoostingClassifier.

    These are the parameters that make sense to explore during
    hyperparameter search.  All other sklearn knobs live in
    :class:`GradientBoostingClassifierRunningConfig`.
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
            "Maximum depth of each individual decision tree. "
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


class GradientBoostingClassifierRunningConfig(ModelRunningConfig):
    """Execution-time options for the GradientBoostingClassifier.

    These affect training behaviour but are usually held fixed during
    hyperparameter search.
    """

    loss: Literal[
        "log_loss",
        "exponential",
    ] = Field(
        default="log_loss",
        description=(
            "Loss function to optimise. "
            "``'log_loss'`` refers to binomial / multinomial deviance "
            "(logistic regression loss). "
            "``'exponential'`` recovers the AdaBoost algorithm."
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


class GradientBoostingClassifierConfig(
    ModelConfig[
        GradientBoostingClassifierHyperParameters,
        GradientBoostingClassifierRunningConfig,
    ]
):
    """Full configuration for the GradientBoostingClassifier node."""

    hyperparameters: GradientBoostingClassifierHyperParameters = Field(
        default_factory=GradientBoostingClassifierHyperParameters,
        description="Tuneable hyperparameters (n_estimators, max_depth, learning_rate).",
    )
    running_config: GradientBoostingClassifierRunningConfig = Field(
        default_factory=GradientBoostingClassifierRunningConfig,
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
                desc=(
                    "Class labels (batch, 1). Use integers 0 .. C-1 for the "
                    "classification metrics. Required during training and "
                    "evaluation only."
                ),
            ),
        },
        description="Input ports: 'X' (features, all modes) and 'y' (labels, training/evaluation).",
    )
    out_ports: dict[str, Port] = Field(
        default={
            "pred": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch classes",
                desc=(
                    "Float64 class probabilities (batch, classes). One column "
                    "'proba_<class>' for each class, in the order of classes_."
                ),
            ),
        },
        description="Output ports: 'pred' (predicted class probabilities).",
    )


class GradientBoostingClassifierNode(SklearnModelNode):
    """Gradient Boosting Classifier model node.

    :class:`SklearnModelNode` builds the estimator from
    :class:`GradientBoostingClassifierConfig` at each :meth:`fit`.  The
    ``pred`` port holds float64 class probabilities, one column
    ``proba_<class>`` for each class, in the order of ``classes_``.  A binary
    problem also gets one column for each class.  Use integer class labels
    ``0 .. C-1`` for the classification metrics.
    """

    metadata = GradientBoostingClassifierMetadata()
    hyperparameter_space = hyperparameter_space
    estimator_class = SklearnGradientBoostingClassifier
    output_probabilities = True
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
