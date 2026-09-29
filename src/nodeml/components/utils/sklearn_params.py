"""Helpers for the public fitted attributes of scikit-learn estimators.

scikit-learn stores learned state on attributes whose names end with an
underscore (for example ``coef_``, ``classes_`` or ``estimators_``).
``BaseEstimator.get_params`` / ``set_params`` cover only the constructor
arguments, not this fitted state.

These helpers do not capture the full fitted state: some estimators also
keep private attributes (for example ``_loss`` in gradient boosting).  To
save and restore an estimator, pickle the estimator object itself.  The
model nodes use these helpers only to show the fitted attributes and to
read params saved by NodeML 0.1.
"""

from typing import Any

from sklearn.base import BaseEstimator


def get_sklearn_fitted_params(estimator: BaseEstimator) -> dict[str, Any]:
    """Return the public fitted attributes of a scikit-learn estimator.

    Args:
        estimator: A scikit-learn estimator.

    Returns:
        The attributes whose names end with one underscore.  Private and
        dunder attributes are excluded.

    """
    return {
        name: value
        for name, value in vars(estimator).items()
        if name.endswith("_") and not name.startswith("_") and not name.endswith("__")
    }


def set_sklearn_fitted_params(
    estimator: BaseEstimator, fitted_params: dict[str, Any]
) -> BaseEstimator:
    """Set fitted attributes on a scikit-learn estimator in place.

    Args:
        estimator: The estimator to change.
        fitted_params: Attribute names and values, as returned by
            :func:`get_sklearn_fitted_params`.

    Returns:
        The same estimator.

    """
    for name, value in fitted_params.items():
        setattr(estimator, name, value)
    return estimator
