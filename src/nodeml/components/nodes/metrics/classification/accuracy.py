"""Accuracy metric node for the NodeML Framework.

Wraps ``torchmetrics.Accuracy`` and exposes it as a NodeML
:class:`~nodeml.core.nodes.metrics.metric_node.MetricNode`.  The node expects
two input ports:

* ``pred``   - predicted class probabilities or logits ``(batch, num_classes)``
               as a numpy array (for multiclass), or ``(batch, 1)`` for binary
* ``target`` - ground-truth class labels ``(batch, 1)`` as a numpy integer array

and emits one output port:

* ``score`` - scalar metric value ``(1, 1)`` as a numpy array

Supports both binary and multiclass tasks via the ``task`` running config
option.  For binary tasks, predictions are thresholded at ``threshold``.
"""

from typing import Literal

import numpy as np
import torch
from pydantic import Field
from torchmetrics import Accuracy

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


class AccuracyMetadata(MetricNodeMetadata):
    """Metadata for the Accuracy metric node."""

    node_name: str = "Accuracy"
    description: str = (
        "Classification accuracy based on torchmetrics. "
        "Fraction of correctly predicted samples. "
        "Supports binary and multiclass tasks."
    )


class AccuracyRunningConfig(MetricNodeRunningConfig):
    """Run-time options for the Accuracy metric node."""

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
    top_k: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of top predictions to consider for a correct match. "
            "``top_k=1`` is standard accuracy; ``top_k=5`` gives top-5 accuracy. "
            "Only used when ``task='multiclass'``."
        ),
    )
    average: Literal["micro", "macro", "weighted"] = Field(
        default="micro",
        description=(
            "Averaging strategy for multiclass accuracy. "
            "``'micro'`` computes global accuracy (equivalent to standard accuracy). "
            "``'macro'`` averages per-class accuracy. "
            "``'weighted'`` weights per-class accuracy by class support."
        ),
    )


class AccuracyConfig(MetricNodeConfig[AccuracyRunningConfig]):
    """Full configuration for the Accuracy metric node."""

    running_config: AccuracyRunningConfig = Field(
        default_factory=AccuracyRunningConfig,
        description="Run-time options (task, num_classes, threshold, top_k, average).",
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
                desc="Scalar accuracy value in [0, 1].",
            ),
        },
        description="Output ports: 'score' (scalar accuracy).",
    )


class AccuracyNode(
    MetricNode[
        np.ndarray,
        TabularDataContext,
        np.ndarray,
        TabularDataContext,
    ]
):
    """Classification accuracy metric node.

    Converts numpy inputs to torch tensors, delegates to
    ``torchmetrics.Accuracy``, and returns the scalar result as a
    ``(1, 1)`` numpy array with a :class:`TabularDataContext`.
    """

    metadata = AccuracyMetadata()

    def __init__(self, *, config: AccuracyConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config
        rc = config.running_config
        metric_kwargs: dict = {"task": rc.task}
        if rc.task == "binary":
            metric_kwargs["threshold"] = rc.threshold
        if rc.task == "multiclass":
            metric_kwargs["num_classes"] = rc.num_classes
            metric_kwargs["top_k"] = rc.top_k
            metric_kwargs["average"] = rc.average
        self._metric = Accuracy(**metric_kwargs)

    # --- MetricNode interface ------------------------------------------------

    def update(self, data: dict[str, tuple[np.ndarray, TabularDataContext]]) -> None:
        """Feed predictions and targets into the torchmetrics accumulator."""
        pred, _ = data["pred"]
        target, _ = data["target"]
        pred_t = torch.from_numpy(pred).float()
        target_t = torch.from_numpy(target).long().squeeze(-1)
        # Binary task expects 1-D predictions.
        if self._config.running_config.task == "binary":
            pred_t = pred_t.squeeze(-1)
        self._metric.update(pred_t, target_t)

    def compute(self) -> dict[str, tuple[np.ndarray, TabularDataContext]]:
        """Compute accuracy and return a ``(1, 1)`` result array."""
        value = self._metric.compute().item()
        self._metric.reset()
        return {
            "score": (
                np.array([[value]], dtype=np.float64),
                TabularDataContext(
                    columns=["accuracy"],
                    dtypes=[np.dtype("float64")],
                    categories=[NumericalData],
                ),
            ),
        }
