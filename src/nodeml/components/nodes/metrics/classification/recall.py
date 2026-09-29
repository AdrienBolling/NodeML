"""Recall metric node for the NodeML Framework.

Wraps ``torchmetrics.Recall`` and exposes it as a NodeML
:class:`~nodeml.core.nodes.metrics.metric_node.MetricNode`.  The node expects
two input ports:

* ``pred``   - predicted class probabilities or logits ``(batch, num_classes)``
               as a numpy array (for multiclass), or ``(batch, 1)`` for binary
* ``target`` - ground-truth class labels ``(batch, 1)`` as a numpy integer array

and emits one output port:

* ``score`` - scalar metric value ``(1, 1)`` as a numpy array

Recall (sensitivity / true positive rate) measures the fraction of actual
positives that are correctly identified.  Supports both binary and multiclass
tasks via the ``task`` running config option.  For binary tasks, predictions
are thresholded at ``threshold``.
"""

from pydantic import Field
from torchmetrics import Recall as TorchRecall

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


class RecallMetadata(MetricNodeMetadata):
    """Metadata for the Recall metric node."""

    node_name: str = "Recall"
    description: str = (
        "Classification recall (sensitivity / true positive rate) based on torchmetrics. "
        "Fraction of actual positives that are correctly identified. "
        "Supports binary and multiclass tasks."
    )


class RecallRunningConfig(StatScoresRunningConfig):
    """Run-time options for the Recall metric node."""


class RecallConfig(MetricNodeConfig[RecallRunningConfig]):
    """Full configuration for the Recall metric node."""

    running_config: RecallRunningConfig = Field(
        default_factory=RecallRunningConfig,
        description="Run-time options (task, num_classes, threshold, average).",
    )
    in_ports: dict[str, Port] = Field(
        default=classification_in_ports(),
        description="Input ports: 'pred' (class probabilities) and 'target' (class labels).",
    )
    out_ports: dict[str, Port] = Field(
        default=score_out_ports("Scalar recall value in [0, 1]."),
        description="Output ports: 'score' (scalar recall).",
    )


class RecallNode(ClassificationMetricNode):
    """Classification recall metric node.

    Converts numpy inputs to torch tensors, delegates to
    ``torchmetrics.Recall``, and returns the scalar result as a
    ``(1, 1)`` numpy array with a :class:`TabularDataContext`.
    """

    metadata = RecallMetadata()
    metric_class = TorchRecall
    score_name = "recall"
    binary_options = ("threshold",)
    multiclass_options = ("num_classes", "average")
