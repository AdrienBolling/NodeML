"""Helpers that the scikit-learn and PyTorch model bases share."""

from collections.abc import Sequence

import numpy as np

from nodeml.core.common.data.data import NumericalData, TabularDataContext
from nodeml.core.common.exceptions import NodeNotFittedError


def prediction_context(columns: Sequence[object]) -> TabularDataContext:
    """Return the context of a float64 prediction array.

    All model nodes return float64 predictions.  The dtype does not come
    from the target: an integer target would round the predictions.

    Args:
        columns: One name for each prediction column.

    Returns:
        A context with float64 dtypes and numerical categories.

    """
    names = [str(column) for column in columns]
    return TabularDataContext(
        columns=names,
        dtypes=[np.dtype(np.float64)] * len(names),
        categories=[NumericalData] * len(names),
    )


def not_fitted_error(node: object) -> NodeNotFittedError:
    """Return the error for a prediction request on an unfitted node.

    Args:
        node: The node that is not fitted.

    Returns:
        A :class:`NodeNotFittedError` with a clear message.

    """
    msg = (
        f"{type(node).__name__} is not fitted. Train the pipeline, call fit, "
        "or load trained parameters with set_params before you predict."
    )
    return NodeNotFittedError(msg)
