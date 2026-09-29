"""Mean Absolute Percentage Error metric node for the NodeML Framework.

Wraps ``torchmetrics.MeanAbsolutePercentageError`` and exposes it as a NodeML
:class:`~nodeml.core.nodes.metrics.metric_node.MetricNode`.  The node measures
the average absolute percentage deviation between predictions and targets.
The result lies in [0, inf), where 0 indicates a perfect fit.  The value is
**not** multiplied by 100 (this is the torchmetrics default).

The node expects two input ports:

* ``pred``   - predicted values ``(batch, targets)`` as a numpy array
* ``target`` - ground-truth values ``(batch, targets)`` as a numpy array

and emits one output port:

* ``score`` - scalar metric value ``(1, 1)`` as a numpy array
"""

from pydantic import Field
from torchmetrics import MeanAbsolutePercentageError, Metric

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


class MAPEMetadata(MetricNodeMetadata):
    """Metadata for the MAPE metric node."""

    node_name: str = "MAPE"
    description: str = (
        "Mean Absolute Percentage Error based on torchmetrics. "
        "Measures the average absolute percentage deviation between "
        "predictions and targets. Returns a value in [0, inf) where "
        "0 is perfect."
    )


class MAPERunningConfig(MetricNodeRunningConfig):
    """Run-time options for the MAPE metric node."""


class MAPEConfig(MetricNodeConfig[MAPERunningConfig]):
    """Full configuration for the MAPE metric node."""

    running_config: MAPERunningConfig = Field(
        default_factory=MAPERunningConfig,
        description="Run-time options.",
    )
    in_ports: dict[str, Port] = Field(
        default=regression_in_ports(),
        description="Input ports: 'pred' (predictions) and 'target' (ground truth).",
    )
    out_ports: dict[str, Port] = Field(
        default=score_out_ports("Scalar metric value."),
        description="Output ports: 'score' (scalar metric value).",
    )


class MAPE(RegressionMetricNode):
    """Mean Absolute Percentage Error metric node.

    Converts numpy inputs to torch tensors, delegates to
    ``torchmetrics.MeanAbsolutePercentageError``, and returns the scalar
    result as a ``(1, 1)`` numpy array with a :class:`TabularDataContext`.
    """

    metadata = MAPEMetadata()
    score_name = "mape"

    def _build_metric(self) -> Metric:
        """Build the torchmetrics metric.

        Returns:
            A ``MeanAbsolutePercentageError`` metric.

        """
        return MeanAbsolutePercentageError()
