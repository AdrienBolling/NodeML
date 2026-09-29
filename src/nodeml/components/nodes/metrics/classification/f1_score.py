"""F1 Score metric node for the NodeML Framework.

Wraps ``torchmetrics.F1Score`` and exposes it as a NodeML
:class:`~nodeml.core.nodes.metrics.metric_node.MetricNode`.  The node expects
two input ports:

* ``pred``   - class probabilities ``(batch, num_classes)`` (also for binary),
               positive-class probabilities ``(batch, 1)`` (binary only), or
               hard class labels ``(batch, 1)``
* ``target`` - ground-truth class labels ``(batch, 1)`` as a numpy integer array

and emits one output port:

* ``score`` - scalar metric value ``(1, 1)`` as a numpy array

F1 is the harmonic mean of precision and recall, providing a single measure
that balances both.  Supports binary and multiclass tasks with configurable
averaging strategies.
"""

from pydantic import Field
from torchmetrics import F1Score

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


class F1ScoreMetadata(MetricNodeMetadata):
    """Metadata for the F1Score metric node."""

    node_name: str = "F1Score"
    description: str = (
        "F1 Score based on torchmetrics. "
        "Harmonic mean of precision and recall. "
        "Supports binary and multiclass tasks with configurable averaging."
    )


class F1ScoreRunningConfig(StatScoresRunningConfig):
    """Run-time options for the F1Score metric node."""

    zero_division: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Value to return when there is a zero division "
            "(i.e., all predictions and labels are negative). "
            "Defaults to 0."
        ),
    )


class F1ScoreConfig(MetricNodeConfig[F1ScoreRunningConfig]):
    """Full configuration for the F1Score metric node."""

    running_config: F1ScoreRunningConfig = Field(
        default_factory=F1ScoreRunningConfig,
        description="Run-time options (task, num_classes, threshold, average, zero_division).",
    )
    in_ports: dict[str, Port] = Field(
        default=classification_in_ports(),
        description="Input ports: 'pred' (class probabilities) and 'target' (class labels).",
    )
    out_ports: dict[str, Port] = Field(
        default=score_out_ports("Scalar F1 score value in [0, 1]."),
        description="Output ports: 'score' (scalar F1 score).",
    )


class F1(ClassificationMetricNode):
    """F1 Score metric node.

    Converts numpy inputs to torch tensors, delegates to
    ``torchmetrics.F1Score``, and returns the scalar result as a
    ``(1, 1)`` numpy array with a :class:`TabularDataContext`.
    """

    metadata = F1ScoreMetadata()
    metric_class = F1Score
    score_name = "f1"
    common_options = ("zero_division",)
    binary_options = ("threshold",)
    multiclass_options = ("num_classes", "average")
