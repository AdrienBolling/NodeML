"""GradientBoostingClassifier model node for the NodeML Framework.

Wraps ``sklearn.ensemble.GradientBoostingClassifier`` and exposes it as a NodeML
:class:`~nodeml.core.nodes.models.model.Model` node.  The node expects two
input ports:

* ``X`` - feature matrix ``(batch, features)`` as a numpy array
* ``y`` - label vector ``(batch, 1)`` as a numpy array (training / evaluation only)

and emits one output port:

* ``pred`` - predicted class probabilities ``(batch, num_classes)`` as a numpy array

The classifier outputs probabilities via ``predict_proba``, which always
returns a 2-D array ``(batch, num_classes)``.
"""

from typing import Any, Literal

import numpy as np
from pydantic import Field
from ray import tune
from sklearn.ensemble import (
    GradientBoostingClassifier as SklearnGradientBoostingClassifier,
)

from nodeml.components.utils.sklearn_params import (
    get_sklearn_fitted_params,
    set_sklearn_fitted_params,
)
from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    TabularDataContext,
    tabular_context_from_dict_dump,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.nodes.models.model import (
    Model,
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

# Type alias for the serialisable param dict used by get_params / set_params.
type _GBCParams = dict[str, Any]


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
                desc="Class labels (batch, 1). Required during training and evaluation only.",
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
                data_shape="batch num_classes",
                desc="Predicted class probabilities (batch, num_classes).",
            ),
        },
        description="Output ports: 'pred' (predicted class probabilities).",
    )


class GradientBoostingClassifierNode(
    Model[
        np.ndarray,
        TabularDataContext,
        np.ndarray,
        TabularDataContext,
        _GBCParams,
    ]
):
    """Gradient Boosting Classifier model node.

    The underlying sklearn estimator is built from
    :class:`GradientBoostingClassifierConfig` on initialisation.  The target
    context (column names, dtypes, categories) is captured during
    :meth:`fit` and replayed on every :meth:`predict` call so that the
    output :class:`~nodeml.core.common.data.data.TabularDataContext` is
    always consistent with the training labels.
    """

    metadata = GradientBoostingClassifierMetadata()
    hyperparameter_space = hyperparameter_space

    def __init__(self, *, config: GradientBoostingClassifierConfig) -> None:
        """Construct the sklearn estimator from *config*."""
        self._config = config
        self._model = SklearnGradientBoostingClassifier(
            n_estimators=config.hyperparameters.n_estimators,
            max_depth=config.hyperparameters.max_depth,
            learning_rate=config.hyperparameters.learning_rate,
            loss=config.running_config.loss,
            random_state=config.running_config.random_state,
        )
        # Populated during fit; used to rebuild the output context at
        # prediction time without requiring access to the training data.
        self._target_context_dump: dict[str, list[str]] = {}

    # --- Model interface --------------------------------------------------

    def fit(self, data: dict[str, tuple[np.ndarray, TabularDataContext]]) -> None:
        """Fit the Gradient Boosting Classifier on the provided *(X, y)* pair.

        Args:
            data: Must contain keys ``"X"`` (features) and ``"y"`` (labels).

        """
        X, _ = data["X"]
        y, y_ctx = data["y"]
        # sklearn GBC expects a 1-D label array.
        self._model.fit(X, y.ravel())
        self._target_context_dump = y_ctx.dump_dict

    def predict(
        self, data: dict[str, tuple[np.ndarray, TabularDataContext]]
    ) -> dict[str, tuple[np.ndarray, TabularDataContext]]:
        """Predict class probabilities using the fitted Gradient Boosting Classifier.

        Args:
            data: Must contain key ``"X"`` (features).  ``"y"`` is ignored if
                present (inference / evaluation phases).

        Returns:
            ``{"pred": (probabilities, context)}`` where probabilities is a
            2-D array ``(batch, num_classes)``.

        """
        X, _ = data["X"]
        pred: np.ndarray = self._model.predict_proba(X)
        # predict_proba always returns 2-D (batch, num_classes).
        pred_ctx = tabular_context_from_dict_dump(self._target_context_dump)
        return {"pred": (pred, pred_ctx)}

    def get_params(self) -> _GBCParams:
        """Return the serialisable state of this node.

        Includes both the sklearn estimator's internal parameters and the
        captured target context needed to reconstruct predictions.
        """
        return {
            "fitted_params": get_sklearn_fitted_params(self._model),
            "target_context": self._target_context_dump,
        }

    def set_params(self, params: _GBCParams) -> None:
        """Restore node state from a previously serialised param dict."""
        set_sklearn_fitted_params(self._model, params["fitted_params"])
        self._target_context_dump = params["target_context"]
