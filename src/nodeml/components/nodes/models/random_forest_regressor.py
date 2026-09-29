"""RandomForestRegressor model node for the NodeML Framework.

Wraps ``sklearn.ensemble.RandomForestRegressor`` and exposes it as a NodeML
:class:`~nodeml.core.nodes.models.model.Model` node.  The node expects two
input ports:

* ``X`` - feature matrix ``(batch, features)`` as a numpy array
* ``y`` - target matrix ``(batch, targets)`` as a numpy array (training / evaluation only)

and emits one output port:

* ``pred`` - predicted values ``(batch, targets)`` as a numpy array

sklearn handles multi-output regression transparently when ``y`` is 2-D, so
no extra wrapping is needed.
"""

from typing import Any, Literal

import numpy as np
from pydantic import Field
from ray import tune
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor

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

# Type alias for the serialisable param dict used by get_params / set_params.
type _RFParams = dict[str, Any]


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
                desc="Predicted values (batch, targets).",
            ),
        },
        description="Output ports: 'pred' (predicted targets).",
    )


class RandomForestRegressorNode(
    Model[
        np.ndarray,
        TabularDataContext,
        np.ndarray,
        TabularDataContext,
        _RFParams,
    ]
):
    """Random Forest Regressor model node.

    The underlying sklearn estimator is built from
    :class:`RandomForestRegressorConfig` on initialisation.  The target
    context (column names, dtypes, categories) is captured during
    :meth:`fit` and replayed on every :meth:`predict` call so that the
    output :class:`~nodeml.core.common.data.data.TabularDataContext` is
    always consistent with the training labels.
    """

    metadata = RandomForestRegressorMetadata()
    hyperparameter_space = hyperparameter_space

    def __init__(self, *, config: RandomForestRegressorConfig) -> None:
        """Construct the sklearn estimator from *config*."""
        self._config = config
        self._model = SklearnRandomForestRegressor(
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
        # sklearn accepts a 2-D y for multi-output regression.
        self._model.fit(X, y)
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
            ``{"pred": (predictions, context)}`` where predictions is a
            2-D array ``(batch, targets)``.

        """
        X, _ = data["X"]
        pred: np.ndarray = self._model.predict(X)
        # sklearn returns 1-D output for single-target problems; normalise
        # to 2-D so downstream nodes always see a consistent shape.
        if pred.ndim == 1:
            pred = pred[:, np.newaxis]
        pred_ctx = tabular_context_from_dict_dump(self._target_context_dump)
        return {"pred": (pred, pred_ctx)}

    def get_params(self) -> _RFParams:
        """Return the serialisable state of this node.

        Includes both the sklearn estimator's internal parameters and the
        captured target context needed to reconstruct predictions.
        """
        return {
            "fitted_params": get_sklearn_fitted_params(self._model),
            "target_context": self._target_context_dump,
        }

    def set_params(self, params: _RFParams) -> None:
        """Restore node state from a previously serialised param dict."""
        self._model = set_sklearn_fitted_params(self._model, params["fitted_params"])
        self._target_context_dump = params["target_context"]
