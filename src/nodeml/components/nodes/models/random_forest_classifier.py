"""RandomForestClassifier model node for the NodeML Framework.

Wraps ``sklearn.ensemble.RandomForestClassifier`` and exposes it as a NodeML
:class:`~nodeml.core.nodes.models.model.Model` node.  The node expects two
input ports:

* ``X`` - feature matrix ``(batch, features)`` as a numpy array
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
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier

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


class RandomForestClassifierMetadata(ModelMetadata):
    """Metadata for the RandomForestClassifier model node."""

    node_name: str = "RandomForestClassifier"
    description: str = (
        "Random Forest Classifier based on scikit-learn. "
        "Supports multi-class classification and returns class probabilities."
    )


class RandomForestClassifierHyperParameters(ModelHyperParameters):
    """Tuneable hyperparameters for the RandomForestClassifier.

    These are the parameters that make sense to explore during
    hyperparameter search.  All other sklearn knobs live in
    :class:`RandomForestClassifierRunningConfig`.
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


class RandomForestClassifierRunningConfig(ModelRunningConfig):
    """Execution-time options for the RandomForestClassifier.

    These affect training behaviour but are usually held fixed during
    hyperparameter search.
    """

    criterion: Literal[
        "gini",
        "entropy",
        "log_loss",
    ] = Field(
        default="gini",
        description=(
            "Impurity measure used to evaluate the quality of a split. "
            "``'gini'`` uses the Gini impurity. "
            "``'entropy'`` uses the Shannon information gain. "
            "``'log_loss'`` uses the log-loss (cross-entropy)."
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


class RandomForestClassifierConfig(
    ModelConfig[
        RandomForestClassifierHyperParameters,
        RandomForestClassifierRunningConfig,
    ]
):
    """Full configuration for the RandomForestClassifier node."""

    hyperparameters: RandomForestClassifierHyperParameters = Field(
        default_factory=RandomForestClassifierHyperParameters,
        description="Tuneable hyperparameters (n_estimators, max_depth).",
    )
    running_config: RandomForestClassifierRunningConfig = Field(
        default_factory=RandomForestClassifierRunningConfig,
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
                data_shape="batch 1",
                mode=[NodeExecutionMode.TRAINING, NodeExecutionMode.EVALUATION],
                desc=(
                    "Class labels (batch, 1). Use integers 0 .. C-1 for the "
                    "classification metrics. Required during training and "
                    "evaluation only."
                ),
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
                data_shape="batch classes",
                desc=(
                    "Float64 class probabilities (batch, classes). One column "
                    "'proba_<class>' for each class, in the order of classes_."
                ),
            ),
        },
        description="Output ports: 'pred' (predicted class probabilities).",
    )


class RandomForestClassifierNode(SklearnModelNode):
    """Random Forest Classifier model node.

    :class:`SklearnModelNode` builds the estimator from
    :class:`RandomForestClassifierConfig` at each :meth:`fit`.  The ``pred``
    port holds float64 class probabilities, one column ``proba_<class>``
    for each class, in the order of ``classes_``.  A binary problem also
    gets one column for each class.  Use integer class labels ``0 .. C-1``
    for the classification metrics.
    """

    metadata = RandomForestClassifierMetadata()
    hyperparameter_space = hyperparameter_space
    estimator_class = SklearnRandomForestClassifier
    output_probabilities = True
    multi_output = False

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
