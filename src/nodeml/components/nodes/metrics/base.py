"""Shared base class and helpers for the torchmetrics metric nodes.

Every built-in metric node wraps one ``torchmetrics`` metric.  The node
takes two numpy input ports, ``pred`` and ``target``, and returns one numpy
output port, ``score``.  :class:`TorchMetricNode` holds the parts that all
these nodes share: the port I/O, the tensor conversion hook and the output
format.
"""

from abc import abstractmethod

import numpy as np
import torch
from torchmetrics import Metric

from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    NumericalData,
    TabularDataContext,
)
from nodeml.core.nodes.metrics.metric_node import MetricNode, MetricNodeConfig
from nodeml.core.nodes.node import Port

type MetricData = dict[str, tuple[np.ndarray, TabularDataContext]]


def score_out_ports(desc: str) -> dict[str, Port]:
    """Return the output ports of a metric node: one ``score`` port.

    Args:
        desc: Description of the score.

    Returns:
        Mapping with the ``score`` port.

    """
    return {
        "score": Port(
            arr_type=ArrayLikeEnum.NUMPY,
            data_structure=DataStructureEnum.TABULAR,
            data_category=DataCategoryEnum.NUMERICAL,
            data_shape="1 1",
            desc=desc,
        ),
    }


def score_result(value: float, name: str) -> MetricData:
    """Wrap a metric value into the output of the ``score`` port.

    Args:
        value: The metric value.
        name: Name of the score column.

    Returns:
        Mapping from ``"score"`` to a ``(1, 1)`` float64 array and its context.

    """
    return {
        "score": (
            np.array([[value]], dtype=np.float64),
            TabularDataContext(
                columns=[name],
                dtypes=[np.dtype("float64")],
                categories=[NumericalData],
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
    and :meth:`_to_tensors`.

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

        Args:
            data: Mapping with the ``"pred"`` and ``"target"`` ports.

        """
        pred, _ = data["pred"]
        target, _ = data["target"]
        self._metric.update(*self._to_tensors(np.asarray(pred), np.asarray(target)))

    def compute(self) -> MetricData:
        """Compute the metric and return it as a ``(1, 1)`` array.

        The accumulated state does not change.

        Returns:
            Mapping from ``"score"`` to the value and its context.

        """
        value = self._metric.compute().item()
        return score_result(value, self._score_name())

    def reset(self) -> None:
        """Clear the state of the torchmetrics metric."""
        self._metric.reset()
