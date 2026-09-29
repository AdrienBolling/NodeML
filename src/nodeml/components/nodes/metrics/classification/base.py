"""Shared base classes for the classification metric nodes.

The classification metric nodes differ only in the torchmetrics class that
they wrap, the options that they pass to it, and the name of the score.
:class:`ClassificationMetricNode` holds everything else.
"""

from typing import ClassVar, Literal

import numpy as np
import torch
from pydantic import Field
from torchmetrics import Metric

from nodeml.components.nodes.metrics.base import TorchMetricNode
from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
)
from nodeml.core.nodes.metrics.metric_node import MetricNodeRunningConfig
from nodeml.core.nodes.node import Port


class ClassificationRunningConfig(MetricNodeRunningConfig):
    """Run-time options that all classification metric nodes share."""

    task: Literal["binary", "multiclass"] = Field(
        default="binary",
        description=(
            "Classification task type. "
            "``'binary'`` expects predictions in ``(batch, 1)``. "
            "``'multiclass'`` expects predictions in ``(batch, num_classes)``."
        ),
    )
    num_classes: int | None = Field(
        default=None,
        ge=2,
        description=(
            "Number of classes. Required for ``task='multiclass'``. "
            "Ignored for binary tasks."
        ),
    )


class ThresholdRunningConfig(ClassificationRunningConfig):
    """Classification options with a decision threshold for binary tasks."""

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Decision threshold for binary classification. "
            "Predictions above this value are assigned to the positive class. "
            "Only used when ``task='binary'``."
        ),
    )


class StatScoresRunningConfig(ThresholdRunningConfig):
    """Options of the precision, recall and F1 metric nodes."""

    average: Literal["micro", "macro", "weighted"] = Field(
        default="macro",
        description=(
            "Averaging strategy for the multiclass score. "
            "``'micro'`` computes the score over all samples. "
            "``'macro'`` averages the per-class scores. "
            "``'weighted'`` weights the per-class scores by class support."
        ),
    )


def classification_in_ports() -> dict[str, Port]:
    """Return the input ports of a classification metric node.

    Returns:
        Mapping with the ``pred`` and ``target`` ports.

    """
    return {
        "pred": Port(
            arr_type=ArrayLikeEnum.NUMPY,
            data_structure=DataStructureEnum.TABULAR,
            data_category=DataCategoryEnum.NUMERICAL,
            data_shape="batch _",
            desc=(
                "Predicted class probabilities or logits. "
                "Shape (batch, 1) for binary, (batch, num_classes) for multiclass."
            ),
        ),
        "target": Port(
            arr_type=ArrayLikeEnum.NUMPY,
            data_structure=DataStructureEnum.TABULAR,
            data_category=DataCategoryEnum.NUMERICAL,
            data_shape="batch 1",
            desc="Ground-truth integer class labels (batch, 1).",
        ),
    }


class ClassificationMetricNode(TorchMetricNode):
    """Base class for the classification metric nodes.

    A subclass sets the class attributes below.  The node builds
    ``metric_class(task=..., **options)``, where the options are the running
    config fields named in :attr:`common_options` and in the options of the
    task.

    Attributes:
        metric_class: The torchmetrics class, for example
            ``torchmetrics.Accuracy``.
        common_options: Running config fields passed for every task.
        binary_options: Running config fields passed for ``task='binary'``.
        multiclass_options: Running config fields passed for
            ``task='multiclass'``.

    """

    metric_class: ClassVar[type[Metric]]
    common_options: ClassVar[tuple[str, ...]] = ()
    binary_options: ClassVar[tuple[str, ...]] = ()
    multiclass_options: ClassVar[tuple[str, ...]] = ("num_classes",)

    def _build_metric(self) -> Metric:
        """Build the torchmetrics metric for the configured task.

        Returns:
            The torchmetrics metric.

        """
        rc = self._config.running_config
        task_options = (
            self.binary_options if rc.task == "binary" else self.multiclass_options
        )
        options = {name: getattr(rc, name) for name in self.common_options}
        options.update({name: getattr(rc, name) for name in task_options})
        return self.metric_class(task=rc.task, **options)

    def _to_tensors(
        self, pred: np.ndarray, target: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert predictions and labels to the tensors of torchmetrics.

        Args:
            pred: Predictions, ``(batch, 1)`` or ``(batch, num_classes)``.
            target: Integer class labels, ``(batch, 1)``.

        Returns:
            The ``(pred, target)`` tensors.

        """
        pred_t = torch.from_numpy(pred).float()
        target_t = torch.from_numpy(target).long().squeeze(-1)
        # Binary task expects 1-D predictions.
        if self._config.running_config.task == "binary":
            pred_t = pred_t.squeeze(-1)
        return pred_t, target_t
