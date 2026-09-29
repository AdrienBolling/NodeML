"""Precision metric node for the NodeML Framework.

Wraps ``torchmetrics.Precision`` and exposes it as a NodeML
:class:`~nodeml.core.nodes.metrics.metric_node.MetricNode`.  The node expects
two input ports:

* ``pred``   - predicted class probabilities or logits ``(batch, num_classes)``
               as a numpy array (for multiclass), or ``(batch, 1)`` for binary
* ``target`` - ground-truth class labels ``(batch, 1)`` as a numpy integer array

and emits one output port:

* ``score`` - scalar metric value ``(1, 1)`` as a numpy array

Precision (positive predictive value) measures the fraction of positive
predictions that are actually correct.  Supports both binary and multiclass
tasks via the ``task`` running config option.  For binary tasks, predictions
are thresholded at ``threshold``.
"""

from typing import Literal

import numpy as np
import torch
from pydantic import Field
from torchmetrics import Precision as TorchPrecision

from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    NumericalData,
    TabularDataContext,
)
from nodeml.core.nodes.metrics.metric_node import (
    MetricNode,
    MetricNodeConfig,
    MetricNodeMetadata,
    MetricNodeRunningConfig,
)
from nodeml.core.nodes.node import Port


class PrecisionMetadata(MetricNodeMetadata):
    """Metadata for the Precision metric node."""

    node_name: str = "Precision"
    description: str = (
        "Classification precision (positive predictive value) based on torchmetrics. "
        "Fraction of positive predictions that are truly positive. "
        "Supports binary and multiclass tasks."
    )


class PrecisionRunningConfig(MetricNodeRunningConfig):
    """Run-time options for the Precision metric node."""

    task: Literal["binary", "multiclass"] = Field(
        default="binary",
        description=(
            "Classification task type. "
            "``'binary'`` expects predictions in ``(batch, 1)`` and thresholds at ``threshold``. "
            "``'multiclass'`` expects predictions in ``(batch, num_classes)`` and picks argmax."
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
    average: Literal["micro", "macro", "weighted"] = Field(
        default="macro",
        description=(
            "Averaging strategy for multiclass precision. "
            "``'micro'`` computes global precision over all samples. "
            "``'macro'`` averages per-class precision. "
            "``'weighted'`` weights per-class precision by class support."
        ),
    )


class PrecisionConfig(MetricNodeConfig[PrecisionRunningConfig]):
    """Full configuration for the Precision metric node."""

    running_config: PrecisionRunningConfig = Field(
        default_factory=PrecisionRunningConfig,
        description="Run-time options (task, num_classes, threshold, average).",
    )
    in_ports: dict[str, Port] = Field(
        default={
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
        },
        description="Input ports: 'pred' (class probabilities) and 'target' (class labels).",
    )
    out_ports: dict[str, Port] = Field(
        default={
            "score": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="1 1",
                desc="Scalar precision value in [0, 1].",
            ),
        },
        description="Output ports: 'score' (scalar precision).",
    )


class PrecisionNode(
    MetricNode[
        np.ndarray,
        TabularDataContext,
        np.ndarray,
        TabularDataContext,
    ]
):
    """Classification precision metric node.

    Converts numpy inputs to torch tensors, delegates to
    ``torchmetrics.Precision``, and returns the scalar result as a
    ``(1, 1)`` numpy array with a :class:`TabularDataContext`.
    """

    metadata = PrecisionMetadata()

    def __init__(self, *, config: PrecisionConfig) -> None:
        """Initialise the Precision metric node.

        Args:
            config: Full node configuration including running config,
                input ports, and output ports.

        """
        self._config = config
        rc = config.running_config
        metric_kwargs: dict = {"task": rc.task}
        if rc.task == "binary":
            metric_kwargs["threshold"] = rc.threshold
        if rc.task == "multiclass":
            metric_kwargs["num_classes"] = rc.num_classes
            metric_kwargs["average"] = rc.average
        self._metric = TorchPrecision(**metric_kwargs)

    # --- MetricNode interface ------------------------------------------------

    def update(self, data: dict[str, tuple[np.ndarray, TabularDataContext]]) -> None:
        """Feed predictions and targets into the torchmetrics accumulator.

        Args:
            data: Dictionary mapping port names to ``(array, context)``
                tuples.  Expected keys are ``"pred"`` and ``"target"``.

        """
        pred, _ = data["pred"]
        target, _ = data["target"]
        pred_t = torch.from_numpy(pred).float()
        target_t = torch.from_numpy(target).long().squeeze(-1)
        # Binary task expects 1-D predictions.
        if self._config.running_config.task == "binary":
            pred_t = pred_t.squeeze(-1)
        self._metric.update(pred_t, target_t)

    def compute(self) -> dict[str, tuple[np.ndarray, TabularDataContext]]:
        """Compute precision and return a ``(1, 1)`` result array.

        Returns:
            Dictionary with a single ``"score"`` key mapping to a tuple of
            the scalar precision value as a ``(1, 1)`` numpy array and the
            corresponding :class:`TabularDataContext`.

        """
        value = self._metric.compute().item()
        self._metric.reset()
        return {
            "score": (
                np.array([[value]], dtype=np.float64),
                TabularDataContext(
                    columns=["precision"],
                    dtypes=[np.dtype("float64")],
                    categories=[NumericalData],
                ),
            ),
        }
