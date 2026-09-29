"""Shared base class for the regression metric nodes."""

import numpy as np
import torch

from nodeml.components.nodes.metrics.base import TorchMetricNode
from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
)
from nodeml.core.nodes.metrics.metric_node import MetricNodeConfig
from nodeml.core.nodes.node import Port


def regression_in_ports() -> dict[str, Port]:
    """Return the input ports of a regression metric node.

    Returns:
        Mapping with the ``pred`` and ``target`` ports.  Both ports have the
        shape ``(batch, targets)``.

    """
    return {
        "pred": Port(
            arr_type=ArrayLikeEnum.NUMPY,
            data_structure=DataStructureEnum.TABULAR,
            data_category=DataCategoryEnum.NUMERICAL,
            data_shape="batch targets",
            desc="Predicted values (batch, targets).",
        ),
        "target": Port(
            arr_type=ArrayLikeEnum.NUMPY,
            data_structure=DataStructureEnum.TABULAR,
            data_category=DataCategoryEnum.NUMERICAL,
            data_shape="batch targets",
            desc="Ground-truth target values (batch, targets).",
        ),
    }


class RegressionMetricNode(TorchMetricNode):
    """Base class for the regression metric nodes.

    The metric state and the inputs are float64.  In float32, large targets
    lose their small errors (for example, a target of 1e4 has a float32
    resolution of about 1e-3).
    """

    def __init__(self, *, config: MetricNodeConfig) -> None:
        """Initialise the node and set the metric state to float64.

        Args:
            config: Full node configuration.

        """
        super().__init__(config=config)
        self._metric.set_dtype(torch.float64)

    def _to_tensors(
        self, pred: np.ndarray, target: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert predictions and targets to float64 tensors.

        A single-output ``(batch, 1)`` pair becomes 1-D, as some
        torchmetrics regression metrics (for example ``R2Score``) expect.

        Args:
            pred: Predictions, ``(batch, targets)``.
            target: Targets, ``(batch, targets)``.

        Returns:
            The ``(pred, target)`` tensors.

        """
        pred_t = torch.from_numpy(np.ascontiguousarray(pred, dtype=np.float64))
        target_t = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float64))
        if pred_t.ndim > 1 and pred_t.shape[-1] == 1:
            pred_t = pred_t.squeeze(-1)
            target_t = target_t.squeeze(-1)
        return pred_t, target_t
