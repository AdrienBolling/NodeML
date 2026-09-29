"""Mean Absolute Error metric node for the NodeML Framework.

Wraps ``torchmetrics.MeanAbsoluteError`` and exposes it as a NodeML
:class:`~nodeml.core.nodes.metrics.metric_node.MetricNode`.  The node expects
two input ports:

* ``pred``   - predicted values ``(batch, targets)`` as a numpy array
* ``target`` - ground-truth values ``(batch, targets)`` as a numpy array

and emits one output port:

* ``score`` - scalar metric value ``(1, 1)`` as a numpy array

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
            "Number of output targets. "
            "Set to >1 for multi-output regression so that the metric "
            "averages across outputs correctly."
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
        default=score_out_ports("Scalar metric value."),
        description="Output ports: 'score' (scalar metric value).",
    )


class MAE(RegressionMetricNode):
    """Mean Absolute Error metric node.

    Converts numpy inputs to torch tensors, delegates to
    ``torchmetrics.MeanAbsoluteError``, and returns the scalar result as a
    ``(1, 1)`` numpy array with a :class:`TabularDataContext`.
    """

    metadata = MAEMetadata()
    score_name = "mae"

    def _build_metric(self) -> Metric:
        """Build the torchmetrics metric.

        Returns:
            A ``MeanAbsoluteError`` metric.

        """
        return MeanAbsoluteError(num_outputs=self._config.running_config.num_outputs)
