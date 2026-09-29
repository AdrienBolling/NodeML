"""Convert runner metric outputs to named float values.

:meth:`PipelineRunner.evaluate` returns each metric as an
``(array, context)`` tuple.  Tools such as Ray Tune and MLflow accept only
plain numbers.  The functions in this module convert the tuples with one
fixed rule:

* A metric with one value gives one key: ``{name: value}``.
* A metric with one row of more than one value gives one key per column:
  ``{f"{name}.{column}": value}``.  A multi-output regression metric with
  the columns ``score_0`` and ``score_1`` gives ``mse.score_0`` and
  ``mse.score_1``.
* Any other shape is an error.

The module does not import Ray, so the MLflow logger can use it too.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch

from nodeml.core.common.data.data import ArrayLike, DataContext, TabularDataContext

# Metric values are a table: (rows, columns).
_TABLE_NDIM = 2


def metric_to_scalars(
    name: str,
    array: ArrayLike,
    context: DataContext | None = None,
) -> dict[str, float]:
    """Convert one metric output to named float values.

    The column names come from the DataFrame columns.  For a numpy array or
    a torch tensor, they come from the :class:`TabularDataContext` when its
    length matches, else from the column index.

    Args:
        name: The metric name, as returned by the runner.
        array: The metric values: a pandas DataFrame, a numpy array or a
            torch tensor.
        context: The data context of *array*.

    Returns:
        ``{name: value}`` for a metric with one value, else
        ``{f"{name}.{column}": value}`` for each column.

    Raises:
        ValueError: If the metric has no value, more than one row, or a
            value that is not a number.

    """
    columns: list[str] | None = None
    if isinstance(array, pd.DataFrame):
        columns = [str(column) for column in array.columns]
        values = array.to_numpy()
    elif isinstance(array, torch.Tensor):
        values = array.detach().cpu().numpy()
    else:
        values = np.asarray(array)
    if values.ndim < _TABLE_NDIM:
        # A scalar, or one row of values.
        values = values.reshape(1, -1)

    if values.size == 1:
        return {name: _to_float(name, values.reshape(-1)[0])}
    if values.size == 0 or values.ndim != _TABLE_NDIM or values.shape[0] != 1:
        message = (
            f"Metric '{name}' has the shape {values.shape}. A metric must have "
            "one value, or one row of values."
        )
        raise ValueError(message)

    num_values = values.shape[1]
    if columns is None:
        if (
            isinstance(context, TabularDataContext)
            and len(context.columns) == num_values
        ):
            columns = [str(column) for column in context.columns]
        else:
            columns = [str(index) for index in range(num_values)]
    return {
        f"{name}.{column}": _to_float(name, value)
        for column, value in zip(columns, values[0], strict=True)
    }


def metrics_to_scalars(
    metrics: Mapping[str, tuple[ArrayLike, DataContext]],
) -> dict[str, float]:
    """Convert all the metric outputs of a runner to named float values.

    See :func:`metric_to_scalars` for the naming rule.

    Args:
        metrics: The mapping that :meth:`PipelineRunner.evaluate` returns.

    Returns:
        A flat mapping of metric keys to float values.

    Raises:
        ValueError: If a metric cannot be converted.

    """
    scalars: dict[str, float] = {}
    for name, (array, context) in metrics.items():
        scalars.update(metric_to_scalars(name, array, context))
    return scalars


def _to_float(name: str, value: Any) -> float:
    """Convert one metric value to a float, with a clear error."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        message = f"Metric '{name}' has a value that is not a number: {value!r}."
        raise ValueError(message) from exc
