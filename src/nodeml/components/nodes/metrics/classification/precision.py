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

from pydantic import Field
from torchmetrics import Precision as TorchPrecision

from nodeml.components.nodes.metrics.base import score_out_ports
from nodeml.components.nodes.metrics.classification.base import (
    ClassificationMetricNode,
    StatScoresRunningConfig,
    classification_in_ports,
)
from nodeml.core.nodes.metrics.metric_node import (
    MetricNodeConfig,
    MetricNodeMetadata,
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


class PrecisionRunningConfig(StatScoresRunningConfig):
    """Run-time options for the Precision metric node."""


class PrecisionConfig(MetricNodeConfig[PrecisionRunningConfig]):
    """Full configuration for the Precision metric node."""

    running_config: PrecisionRunningConfig = Field(
        default_factory=PrecisionRunningConfig,
        description="Run-time options (task, num_classes, threshold, average).",
    )
    in_ports: dict[str, Port] = Field(
        default=classification_in_ports(),
        description="Input ports: 'pred' (class probabilities) and 'target' (class labels).",
    )
    out_ports: dict[str, Port] = Field(
        default=score_out_ports("Scalar precision value in [0, 1]."),
        description="Output ports: 'score' (scalar precision).",
    )


class PrecisionNode(ClassificationMetricNode):
    """Classification precision metric node.

    Converts numpy inputs to torch tensors, delegates to
    ``torchmetrics.Precision``, and returns the scalar result as a
    ``(1, 1)`` numpy array with a :class:`TabularDataContext`.
    """

    metadata = PrecisionMetadata()
    metric_class = TorchPrecision
    score_name = "precision"
    binary_options = ("threshold",)
    multiclass_options = ("num_classes", "average")
