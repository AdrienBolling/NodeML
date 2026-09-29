"""Shared base class and helpers for the torchmetrics metric nodes.

Every built-in metric node wraps one ``torchmetrics`` metric.  The node
takes two numpy input ports, ``pred`` and ``target``, and returns one numpy
output port, ``score``.  :class:`TorchMetricNode` holds the parts that all
these nodes share: the port I/O, the NaN policy, the tensor conversion hook
and the output format.
"""

from abc import abstractmethod
from typing import Literal

import numpy as np
import torch
from pydantic import Field
from torchmetrics import Metric

from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    NumericalData,
    TabularDataContext,
)
from nodeml.core.common.exceptions import NodeInputError
from nodeml.core.common.logging import Logger
from nodeml.core.nodes.metrics.metric_node import (
    MetricNode,
    MetricNodeConfig,
    MetricNodeRunningConfig,
)
from nodeml.core.nodes.node import Port

type MetricData = dict[str, tuple[np.ndarray, TabularDataContext]]

_log = Logger("nodeml.components.metrics")


class TorchMetricRunningConfig(MetricNodeRunningConfig):
    """Run-time options that all torchmetrics metric nodes share."""

    nan_policy: Literal["omit", "propagate"] = Field(
        default="omit",
        description=(
            "What to do with rows that have a NaN in the prediction or the "
            "target. "
            "``'omit'`` drops these rows before scoring and logs a warning "
            "with the number of dropped rows. "
            "If all rows have a NaN, the node raises an error. "
            "``'propagate'`` keeps these rows, so the score can be NaN."
        ),
    )


def _nan_rows(values: np.ndarray) -> np.ndarray:
    """Return a boolean mask of the rows of *values* that have a NaN.

    Args:
        values: A ``(batch, columns)`` array.

    Returns:
        A ``(batch,)`` boolean array.

    """
    if not np.issubdtype(values.dtype, np.inexact):
        # Integer and boolean arrays cannot hold NaN.
        return np.zeros(len(values), dtype=bool)
    return np.isnan(values).any(axis=1)


def score_out_ports(desc: str, *, per_output: bool = False) -> dict[str, Port]:
    """Return the output ports of a metric node: one ``score`` port.

    Args:
        desc: Description of the score.
        per_output: If ``True``, the score can have one column per output,
            so the port shape is ``"1 _"``.  Else it is ``"1 1"``.

    Returns:
        Mapping with the ``score`` port.

    """
    # "_" is an anonymous dimension.  A named dimension would be shared by
    # all input ports of a node that receives several scores (for example
    # the Sink), and scores with other column counts would fail the check.
    return {
        "score": Port(
            arr_type=ArrayLikeEnum.NUMPY,
            data_structure=DataStructureEnum.TABULAR,
            data_category=DataCategoryEnum.NUMERICAL,
            data_shape="1 _" if per_output else "1 1",
            desc=desc,
        ),
    }


def score_result(value: torch.Tensor | float, name: str) -> MetricData:
    """Wrap a metric value into the output of the ``score`` port.

    Args:
        value: The metric value: a scalar, or one value per output.
        name: Name of the score column.  With more than one value, the
            columns are ``name_0``, ``name_1``, ...

    Returns:
        Mapping from ``"score"`` to a ``(1, n_values)`` float64 array and
        its context.

    """
    # np.array copies the values, so a later reset() cannot change them.
    values = np.array(torch.as_tensor(value).detach().cpu().numpy(), dtype=np.float64)
    values = values.reshape(1, -1)
    n_values = values.shape[1]
    columns = [name] if n_values == 1 else [f"{name}_{i}" for i in range(n_values)]
    return {
        "score": (
            values,
            TabularDataContext(
                columns=columns,
                dtypes=[np.dtype("float64")] * n_values,
                categories=[NumericalData] * n_values,
            ),
        ),
    }


class TorchMetricNode(
    MetricNode[
        np.ndarray,
        TabularDataContext,
        np.ndarray,
        TabularDataContext,
    ]
):
    """Base class for the metric nodes that wrap a torchmetrics metric.

    A subclass sets :attr:`score_name` and implements :meth:`_build_metric`
    and :meth:`_to_tensors`.  Its running config must derive from
    :class:`TorchMetricRunningConfig`.

    Attributes:
        score_name: Name of the score column in the output.

    """

    score_name: str = "score"

    def __init__(self, *, config: MetricNodeConfig) -> None:
        """Initialise the node and build its torchmetrics metric.

        Args:
            config: Full node configuration.

        """
        self._config = config
        self._metric = self._build_metric()

    @abstractmethod
    def _build_metric(self) -> Metric:
        """Build the torchmetrics metric from the running config.

        Returns:
            The torchmetrics metric that accumulates the state.

        """

    @abstractmethod
    def _to_tensors(
        self, pred: np.ndarray, target: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert the port arrays to the tensors that the metric expects.

        Args:
            pred: Array of the ``pred`` port.
            target: Array of the ``target`` port.

        Returns:
            The ``(pred, target)`` tensors.

        """

    def _score_name(self) -> str:
        """Return the name of the score column.

        Returns:
            :attr:`score_name`.  A subclass can derive the name from its
            config.

        """
        return self.score_name

    # --- MetricNode interface ------------------------------------------------

    def update(self, data: MetricData) -> None:
        """Feed predictions and targets into the torchmetrics accumulator.

        With ``nan_policy='omit'``, the rows with a NaN are dropped first
        (see :meth:`_drop_nan_rows`).

        Args:
            data: Mapping with the ``"pred"`` and ``"target"`` ports.

        """
        pred, _ = data["pred"]
        target, _ = data["target"]
        pred, target = np.asarray(pred), np.asarray(target)
        if self._config.running_config.nan_policy == "omit":
            pred, target = self._drop_nan_rows(pred, target)
        self._metric.update(*self._to_tensors(pred, target))

    def _drop_nan_rows(
        self, pred: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Drop the rows where *pred* or *target* has a NaN in any column.

        The rows of *pred* and *target* stay aligned by position.  When rows
        are dropped, the node logs one warning with the number of rows.

        Args:
            pred: Array of the ``pred`` port.
            target: Array of the ``target`` port.

        Returns:
            The ``(pred, target)`` arrays without the NaN rows.

        Raises:
            NodeInputError: If *pred* and *target* have different numbers of
                rows, or if all rows have a NaN.

        """
        if len(pred) != len(target):
            msg = (
                f"{type(self).__name__}: pred has {len(pred)} rows, but target "
                f"has {len(target)} rows."
            )
            raise NodeInputError(msg)
        nan_rows = _nan_rows(pred.reshape(len(pred), -1)) | _nan_rows(
            target.reshape(len(target), -1)
        )
        n_dropped = int(nan_rows.sum())
        if n_dropped == 0:
            return pred, target
        if n_dropped == len(pred):
            msg = (
                f"{type(self).__name__}: all {n_dropped} rows have a NaN in pred "
                "or target, so no row is left to score."
            )
            raise NodeInputError(msg)
        _log.warning(
            "Rows with a NaN in pred or target are dropped before scoring.",
            node_class=type(self).__name__,
            dropped_rows=n_dropped,
            total_rows=len(pred),
        )
        return pred[~nan_rows], target[~nan_rows]

    def compute(self) -> MetricData:
        """Compute the metric and return it as one row.

        The row has one column, or one column per output for a metric that
        returns one value per output.  The accumulated state does not
        change.

        Returns:
            Mapping from ``"score"`` to the values and their context.

        """
        return score_result(self._metric.compute(), self._score_name())

    def reset(self) -> None:
        """Clear the state of the torchmetrics metric."""
        self._metric.reset()
