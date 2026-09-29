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

from pydantic import Field
from torchmetrics import AUROC

from nodeml.components.nodes.metrics.base import score_out_ports
from nodeml.components.nodes.metrics.classification.base import (
    ClassificationMetricNode,
    ClassificationRunningConfig,
    classification_in_ports,
)
from nodeml.core.nodes.metrics.metric_node import (
    MetricNodeConfig,
    MetricNodeMetadata,
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


class AUROCRunningConfig(ClassificationRunningConfig):
    """Run-time options for the AUROC metric node."""

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
        default=classification_in_ports(),
        description="Input ports: 'pred' (class probabilities) and 'target' (class labels).",
    )
    out_ports: dict[str, Port] = Field(
        default=score_out_ports("Scalar AUROC value in [0, 1]."),
        description="Output ports: 'score' (scalar AUROC).",
    )


class AUROCNode(ClassificationMetricNode):
    """AUROC metric node.

    Converts numpy inputs to torch tensors, delegates to
    ``torchmetrics.AUROC``, and returns the scalar result as a
    ``(1, 1)`` numpy array with a :class:`TabularDataContext`.
    """

    metadata = AUROCMetadata()
    metric_class = AUROC
    score_name = "auroc"
    binary_options = ("max_fpr",)
    multiclass_options = ("num_classes", "average")
