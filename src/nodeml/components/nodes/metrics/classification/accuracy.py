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

from pydantic import Field
from torchmetrics import Accuracy

from nodeml.components.nodes.metrics.base import score_out_ports
from nodeml.components.nodes.metrics.classification.base import (
    ClassificationMetricNode,
    ThresholdRunningConfig,
    classification_in_ports,
)
from nodeml.core.nodes.metrics.metric_node import (
    MetricNodeConfig,
    MetricNodeMetadata,
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


class AccuracyRunningConfig(ThresholdRunningConfig):
    """Run-time options for the Accuracy metric node."""

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
        default=classification_in_ports(),
        description="Input ports: 'pred' (class probabilities) and 'target' (class labels).",
    )
    out_ports: dict[str, Port] = Field(
        default=score_out_ports("Scalar accuracy value in [0, 1]."),
        description="Output ports: 'score' (scalar accuracy).",
    )


class AccuracyNode(ClassificationMetricNode):
    """Classification accuracy metric node.

    Converts numpy inputs to torch tensors, delegates to
    ``torchmetrics.Accuracy``, and returns the scalar result as a
    ``(1, 1)`` numpy array with a :class:`TabularDataContext`.
    """

    metadata = AccuracyMetadata()
    metric_class = Accuracy
    score_name = "accuracy"
    binary_options = ("threshold",)
    multiclass_options = ("num_classes", "top_k", "average")
