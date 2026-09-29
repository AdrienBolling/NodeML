"""Area Under the ROC Curve metric node for the NodeML Framework.

Wraps ``torchmetrics.AUROC`` and exposes it as a NodeML
:class:`~nodeml.core.nodes.metrics.metric_node.MetricNode`.  The node expects
two input ports:

* ``pred``   - predicted class probabilities or logits ``(batch, num_classes)``
               as a numpy array (for multiclass), or ``(batch, 1)`` for binary
* ``target`` - ground-truth class labels ``(batch, 1)`` as a numpy integer array

and emits one output port:

* ``score`` - scalar metric value ``(1, 1)`` as a numpy array

AUROC measures the model's ability to discriminate between classes across
all possible decision thresholds.  A score of 0.5 corresponds to a random
classifier; 1.0 is a perfect classifier.
"""

from typing import Literal

import numpy as np
import torch
from pydantic import Field
from torchmetrics import AUROC

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


class AUROCMetadata(MetricNodeMetadata):
    """Metadata for the AUROC metric node."""

    node_name: str = "AUROC"
    description: str = (
        "Area Under the ROC Curve based on torchmetrics. "
        "Measures the model's discriminative ability across all thresholds. "
        "Supports binary and multiclass tasks."
    )


class AUROCRunningConfig(MetricNodeRunningConfig):
    """Run-time options for the AUROC metric node."""

    task: Literal["binary", "multiclass"] = Field(
        default="binary",
        description=(
            "Classification task type. "
            "``'binary'`` expects predictions in ``(batch, 1)`` (probability of positive class). "
            "``'multiclass'`` expects predictions in ``(batch, num_classes)`` (class probabilities)."
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
    average: Literal["macro", "weighted"] = Field(
        default="macro",
        description=(
            "Averaging strategy for multiclass AUROC. "
            "``'macro'`` averages per-class AUROC (treats all classes equally). "
            "``'weighted'`` weights per-class AUROC by class support."
        ),
    )
    max_fpr: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description=(
            "Maximum false positive rate for partial AUROC computation. "
            "Only used when ``task='binary'``. "
            "``None`` computes the full AUROC."
        ),
    )


class AUROCConfig(MetricNodeConfig[AUROCRunningConfig]):
    """Full configuration for the AUROC metric node."""

    running_config: AUROCRunningConfig = Field(
        default_factory=AUROCRunningConfig,
        description="Run-time options (task, num_classes, average, max_fpr).",
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
                desc="Scalar AUROC value in [0, 1].",
            ),
        },
        description="Output ports: 'score' (scalar AUROC).",
    )


class AUROCNode(
    MetricNode[
        np.ndarray,
        TabularDataContext,
        np.ndarray,
        TabularDataContext,
    ]
):
    """AUROC metric node.

    Converts numpy inputs to torch tensors, delegates to
    ``torchmetrics.AUROC``, and returns the scalar result as a
    ``(1, 1)`` numpy array with a :class:`TabularDataContext`.
    """

    metadata = AUROCMetadata()

    def __init__(self, *, config: AUROCConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config
        rc = config.running_config
        metric_kwargs: dict = {"task": rc.task}
        if rc.task == "binary":
            metric_kwargs["max_fpr"] = rc.max_fpr
        if rc.task == "multiclass":
            metric_kwargs["num_classes"] = rc.num_classes
            metric_kwargs["average"] = rc.average
        self._metric = AUROC(**metric_kwargs)

    # --- MetricNode interface ------------------------------------------------

    def update(self, data: dict[str, tuple[np.ndarray, TabularDataContext]]) -> None:
        """Feed predictions and targets into the torchmetrics accumulator."""
        pred, _ = data["pred"]
        target, _ = data["target"]
        pred_t = torch.from_numpy(pred).float()
        target_t = torch.from_numpy(target).long().squeeze(-1)
        if self._config.running_config.task == "binary":
            pred_t = pred_t.squeeze(-1)
        self._metric.update(pred_t, target_t)

    def compute(self) -> dict[str, tuple[np.ndarray, TabularDataContext]]:
        """Compute AUROC and return a ``(1, 1)`` result array."""
        value = self._metric.compute().item()
        self._metric.reset()
        return {
            "score": (
                np.array([[value]], dtype=np.float64),
                TabularDataContext(
                    columns=["auroc"],
                    dtypes=[np.dtype("float64")],
                    categories=[NumericalData],
                ),
            ),
        }
