"""RandomForestClassifier model node for the NodeML Framework.

Wraps ``sklearn.ensemble.RandomForestClassifier`` and exposes it as a NodeML
:class:`~nodeml.core.nodes.models.model.Model` node.  The node expects two
input ports:

* ``X`` - feature matrix ``(batch, features)`` as a numpy array
* ``y`` - target vector ``(batch, 1)`` as a numpy array (training / evaluation only)

and emits one output port:

* ``pred`` - predicted class probabilities ``(batch, num_classes)`` as a numpy array

The ``predict`` method returns class probabilities via ``predict_proba`` when
available, falling back to ``predict`` otherwise.
"""

from typing import Any, Literal

import numpy as np
from pydantic import Field
from ray import tune
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier

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

# Type alias for the serialisable param dict used by get_params / set_params.
type _RFCParams = dict[str, Any]


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
                desc="Target class labels (batch, 1). Required during training and evaluation only.",
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
                data_shape="batch num_classes",
                desc="Predicted class probabilities (batch, num_classes).",
            ),
        },
        description="Output ports: 'pred' (predicted class probabilities).",
    )


class RandomForestClassifierNode(
    Model[
        np.ndarray,
        TabularDataContext,
        np.ndarray,
        TabularDataContext,
        _RFCParams,
    ]
):
    """Random Forest Classifier model node.

    The underlying sklearn estimator is built from
    :class:`RandomForestClassifierConfig` on initialisation.  The target
    context (column names, dtypes, categories) is captured during
    :meth:`fit` and replayed on every :meth:`predict` call so that the
    output :class:`~nodeml.core.common.data.data.TabularDataContext` is
    always consistent with the training labels.
    """

    metadata = RandomForestClassifierMetadata()
    hyperparameter_space = hyperparameter_space

    def __init__(self, *, config: RandomForestClassifierConfig) -> None:
        """Construct the sklearn estimator from *config*."""
        self._config = config
        self._model = SklearnRandomForestClassifier(
            n_estimators=config.hyperparameters.n_estimators,
            max_depth=config.hyperparameters.max_depth,
            criterion=config.running_config.criterion,
            random_state=config.running_config.random_state,
        )
        # Populated during fit; used to rebuild the output context at
        # prediction time without requiring access to the training data.
        self._target_context_dump: dict[str, list[str]] = {}

    # --- Model interface --------------------------------------------------

    def fit(self, data: dict[str, tuple[np.ndarray, TabularDataContext]]) -> None:
        """Fit the Random Forest on the provided *(X, y)* pair.

        Parameters
        ----------
        data:
            Must contain keys ``"X"`` (features) and ``"y"`` (targets).

        """
        X, _ = data["X"]
        y, y_ctx = data["y"]
        # sklearn classifiers expect a 1-D target array.
        self._model.fit(X, y.ravel())
        self._target_context_dump = y_ctx.dump_dict

    def predict(
        self, data: dict[str, tuple[np.ndarray, TabularDataContext]]
    ) -> dict[str, tuple[np.ndarray, TabularDataContext]]:
        """Predict using the fitted Random Forest.

        Parameters
        ----------
        data:
            Must contain key ``"X"`` (features).  ``"y"`` is ignored if
            present (inference / evaluation phases).

        Returns
        -------
        dict
            ``{"pred": (probabilities, context)}`` where probabilities is a
            2-D array ``(batch, num_classes)``.

        """
        X, _ = data["X"]
        # Prefer predict_proba for class probabilities; fall back to predict.
        if hasattr(self._model, "predict_proba"):
            pred: np.ndarray = self._model.predict_proba(X)
        else:
            pred = self._model.predict(X)
        # Ensure output is always 2-D so downstream nodes see a consistent shape.
        if pred.ndim == 1:
            pred = pred[:, np.newaxis]
        pred_ctx = tabular_context_from_dict_dump(self._target_context_dump)
        return {"pred": (pred, pred_ctx)}

    def get_params(self) -> _RFCParams:
        """Return the serialisable state of this node.

        Includes both the sklearn estimator's internal parameters and the
        captured target context needed to reconstruct predictions.
        """
        return {
            "fitted_params": get_sklearn_fitted_params(self._model),
            "target_context": self._target_context_dump,
        }

    def set_params(self, params: _RFCParams) -> None:
        """Restore node state from a previously serialised param dict."""
        set_sklearn_fitted_params(self._model, params["fitted_params"])
        self._target_context_dump = params["target_context"]
