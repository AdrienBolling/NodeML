"""Shared base class for the model nodes that wrap a scikit-learn estimator."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
from sklearn.base import BaseEstimator

from nodeml.components.nodes.models._common import (
    not_fitted_error,
    prediction_context,
)
from nodeml.components.utils.sklearn_params import (
    get_sklearn_fitted_params,
    set_sklearn_fitted_params,
)
from nodeml.core.common.data.data import TabularDataContext
from nodeml.core.common.exceptions import NodeInputError
from nodeml.core.nodes.models.model import Model, ModelConfig

# A target of shape (batch, targets) has two dimensions.
_MATRIX_NDIM = 2

type SklearnParams = dict[str, Any]


class SklearnModelNode(
    Model[
        np.ndarray,
        TabularDataContext,
        np.ndarray,
        TabularDataContext,
        SklearnParams,
    ],
    ABC,
):
    """Base class for a model node that wraps one scikit-learn estimator.

    A subclass sets :attr:`estimator_class` and implements
    :meth:`_estimator_kwargs`.  This class does the rest:

    * :meth:`fit` builds a new estimator from the config and fits it on the
      ``X`` and ``y`` ports.  A ``y`` with one column goes to scikit-learn
      as a vector, which is the scikit-learn convention.
    * :meth:`predict` returns float64 predictions on the ``pred`` port.  A
      regressor names the columns after the target columns.  A classifier
      returns one probability column ``proba_<class>`` for each class, in
      the order of the ``classes_`` attribute of the estimator.
    * :meth:`get_params` returns the fitted estimator object.  Thus
      :meth:`set_params` restores the full estimator on a new node, private
      attributes included.

    Attributes:
        estimator_class: The scikit-learn estimator class.
        output_probabilities: If ``True``, ``pred`` holds the class
            probabilities from ``predict_proba``.  If ``False``, ``pred``
            holds the output of ``predict``.
        multi_output: If ``False``, ``y`` must have exactly one column.

    """

    estimator_class: ClassVar[type[BaseEstimator]]
    output_probabilities: ClassVar[bool] = False
    multi_output: ClassVar[bool] = True

    def __init__(self, *, config: ModelConfig) -> None:
        """Store the config.  The estimator is built in :meth:`fit`.

        Args:
            config: The node configuration.

        """
        super().__init__(config=config)
        self._estimator: BaseEstimator | None = None
        # Context of the training targets.  The regressor output columns
        # take the target column names.
        self._target_context_dump: dict[str, list[str]] = {}

    @abstractmethod
    def _estimator_kwargs(self) -> dict[str, Any]:
        """Return the constructor arguments of the estimator.

        Returns:
            Keyword arguments for :attr:`estimator_class`, taken from the
            node config.

        """

    def _new_estimator(self) -> BaseEstimator:
        """Return a new, unfitted estimator built from the node config."""
        return self.estimator_class(**self._estimator_kwargs())

    def _target_for_fit(self, y: np.ndarray, y_ctx: TabularDataContext) -> np.ndarray:
        """Return *y* in the shape that scikit-learn expects.

        Args:
            y: The target array.
            y_ctx: The context of *y*, for the error message.

        Returns:
            A vector if *y* has one column, else *y* unchanged.

        Raises:
            NodeInputError: If *y* has more than one column and the node
                supports one target column only.

        """
        if y.ndim != _MATRIX_NDIM:
            return y
        if y.shape[1] == 1:
            return y.ravel()
        if not self.multi_output:
            msg = (
                f"{type(self).__name__} supports one target column, but y has "
                f"{y.shape[1]} columns: {y_ctx.columns}."
            )
            raise NodeInputError(msg)
        return y

    # --- Model interface --------------------------------------------------

    def fit(self, data: dict[str, tuple[np.ndarray, TabularDataContext]]) -> None:
        """Fit a new estimator on the ``X`` and ``y`` ports.

        Args:
            data: Must contain the keys ``"X"`` (features) and ``"y"``
                (targets).

        Raises:
            NodeInputError: If ``y`` has more than one column and the node
                supports one target column only.

        """
        X, _ = data["X"]
        y, y_ctx = data["y"]
        # A new estimator for each fit: params taken before stay unchanged.
        estimator = self._new_estimator()
        estimator.fit(X, self._target_for_fit(y, y_ctx))
        self._estimator = estimator
        self._target_context_dump = y_ctx.dump_dict

    def predict(
        self, data: dict[str, tuple[np.ndarray, TabularDataContext]]
    ) -> dict[str, tuple[np.ndarray, TabularDataContext]]:
        """Predict on the ``X`` port.

        Args:
            data: Must contain the key ``"X"`` (features).  The node ignores
                ``"y"``.

        Returns:
            ``{"pred": (predictions, context)}``.  The predictions are a
            float64 array of shape ``(batch, targets)`` for a regressor, or
            ``(batch, classes)`` for a classifier.

        Raises:
            NodeNotFittedError: If the node is not fitted.

        """
        if self._estimator is None:
            raise not_fitted_error(self)
        X, _ = data["X"]
        if self.output_probabilities:
            pred = self._estimator.predict_proba(X)
            columns = [f"proba_{label}" for label in self._estimator.classes_]
        else:
            pred = self._estimator.predict(X)
            # scikit-learn returns a vector for a single target.
            if pred.ndim == 1:
                pred = pred[:, np.newaxis]
            columns = self._target_context_dump["columns"]
        pred = np.asarray(pred, dtype=np.float64)
        return {"pred": (pred, prediction_context(columns))}

    def get_params(self) -> SklearnParams:
        """Return the fitted state of the node.

        Returns:
            A picklable dict with these keys:

            * ``"estimator"``: the fitted estimator, or ``None`` if the node
              is not fitted.
            * ``"fitted_params"``: the public fitted attributes of the
              estimator (names that end with ``_``), for inspection.
            * ``"target_context"``: the dumped context of the training
              targets.

        """
        fitted_params = (
            get_sklearn_fitted_params(self._estimator)
            if self._estimator is not None
            else {}
        )
        return {
            "estimator": self._estimator,
            "fitted_params": fitted_params,
            "target_context": self._target_context_dump,
        }

    def set_params(self, params: SklearnParams) -> None:
        """Restore the fitted state from :meth:`get_params` output.

        Params saved by NodeML 0.1.0 have no ``"estimator"`` key.  For them,
        the public fitted attributes go onto a new estimator.

        Args:
            params: The output of :meth:`get_params`.

        Raises:
            NodeInputError: If the estimator is not an instance of
                :attr:`estimator_class`.

        """
        estimator = params.get("estimator")
        fitted_params = params.get("fitted_params")
        if estimator is None and fitted_params:
            estimator = set_sklearn_fitted_params(self._new_estimator(), fitted_params)
        if estimator is not None and not isinstance(estimator, self.estimator_class):
            msg = (
                f"{type(self).__name__} needs a {self.estimator_class.__name__} "
                f"estimator, but the params hold a {type(estimator).__name__}."
            )
            raise NodeInputError(msg)
        self._estimator = estimator
        self._target_context_dump = dict(params.get("target_context", {}))
