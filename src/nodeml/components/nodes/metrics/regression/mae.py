"""Mean Absolute Error metric node for the NodeML Framework.

Wraps ``torchmetrics.MeanAbsoluteError`` and exposes it as a NodeML
:class:`~nodeml.core.nodes.metrics.metric_node.MetricNode`.  The node expects
two input ports:

* ``pred``   - predicted values ``(batch, targets)`` as a numpy array
* ``target`` - ground-truth values ``(batch, targets)`` as a numpy array

and emits one output port:

* ``score`` - metric value ``(1, 1)`` as a numpy array, or one column per
              output ``(1, targets)`` with ``num_outputs > 1``

MAE is more robust to outliers than MSE and reports error in the same
units as the target variable.
"""

from pydantic import Field
from torchmetrics import MeanAbsoluteError, Metric

from nodeml.components.nodes.metrics.base import score_out_ports
from nodeml.components.nodes.metrics.regression.base import (
    RegressionMetricNode,
    regression_in_ports,
)
from nodeml.core.nodes.metrics.metric_node import (
    MetricNodeConfig,
    MetricNodeMetadata,
    MetricNodeRunningConfig,
)
from nodeml.core.nodes.node import Port


class MAEMetadata(MetricNodeMetadata):
    """Metadata for the MAE metric node."""

    node_name: str = "MAE"
    description: str = (
        "Mean Absolute Error based on torchmetrics. "
        "Measures average absolute deviation between predictions and targets. "
        "More robust to outliers than MSE."
    )


class MAERunningConfig(MetricNodeRunningConfig):
    """Run-time options for the MAE metric node."""

    num_outputs: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of outputs (target columns). "
            "With 1, the score is the mean over all outputs. "
            "With more than 1, the score has one column per output, "
            "and the inputs must have exactly this number of columns."
        ),
    )


class MAEConfig(MetricNodeConfig[MAERunningConfig]):
    """Full configuration for the MAE metric node."""

    running_config: MAERunningConfig = Field(
        default_factory=MAERunningConfig,
        description="Run-time options (num_outputs).",
    )
    in_ports: dict[str, Port] = Field(
        default=regression_in_ports(),
        description="Input ports: 'pred' (predictions) and 'target' (ground truth).",
    )
    out_ports: dict[str, Port] = Field(
        default=score_out_ports(
            "Metric value (1, 1), or one value per output (1, targets) "
            "with num_outputs > 1.",
            per_output=True,
        ),
        description="Output ports: 'score' (metric value).",
    )


class MAE(RegressionMetricNode):
    """Mean Absolute Error metric node.

    Converts numpy inputs to torch tensors, delegates to
    ``torchmetrics.MeanAbsoluteError``, and returns the result as one row
    of float64 values: one column ``mae``, or one column per output
    ``mae_0``, ``mae_1``, ... with ``num_outputs > 1``.
    """

    metadata = MAEMetadata()
    score_name = "mae"

    def _build_metric(self) -> Metric:
        """Build the torchmetrics metric.

        Returns:
            A ``MeanAbsoluteError`` metric.

        """
        return MeanAbsoluteError(num_outputs=self._config.running_config.num_outputs)
