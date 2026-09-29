"""Mean Squared Error metric node for the NodeML Framework.

Wraps ``torchmetrics.MeanSquaredError`` and exposes it as a NodeML
:class:`~nodeml.core.nodes.metrics.metric_node.MetricNode`.  The node expects
two input ports:

* ``pred``   - predicted values ``(batch, targets)`` as a numpy array
* ``target`` - ground-truth values ``(batch, targets)`` as a numpy array

and emits one output port:

* ``score`` - metric value ``(1, 1)`` as a numpy array, or one column per
              output ``(1, targets)`` with ``num_outputs > 1``

Setting ``squared=False`` in the running config turns this into RMSE.
"""

from pydantic import Field
from torchmetrics import MeanSquaredError, Metric

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


class MSEMetadata(MetricNodeMetadata):
    """Metadata for the MSE metric node."""

    node_name: str = "MSE"
    description: str = (
        "Mean Squared Error (or Root Mean Squared Error when squared=False) "
        "based on torchmetrics. Measures average squared deviation between "
        "predictions and targets."
    )


class MSERunningConfig(MetricNodeRunningConfig):
    """Run-time options for the MSE metric node."""

    squared: bool = Field(
        default=True,
        description=(
            "If ``True``, return MSE. "
            "If ``False``, return RMSE (the square root of MSE). "
            "RMSE is in the same units as the target, which can be easier to interpret."
        ),
    )
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


class MSEConfig(MetricNodeConfig[MSERunningConfig]):
    """Full configuration for the MSE metric node."""

    running_config: MSERunningConfig = Field(
        default_factory=MSERunningConfig,
        description="Run-time options (squared, num_outputs).",
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


class MSE(RegressionMetricNode):
    """Mean Squared Error metric node.

    Converts numpy inputs to torch tensors, delegates to
    ``torchmetrics.MeanSquaredError``, and returns the result as one row
    of float64 values: one column ``mse``, or one column per output
    ``mse_0``, ``mse_1``, ... with ``num_outputs > 1``.  With
    ``squared=False``, the prefix is ``rmse``.
    """

    metadata = MSEMetadata()
    score_name = "mse"

    def _build_metric(self) -> Metric:
        """Build the torchmetrics metric.

        Returns:
            A ``MeanSquaredError`` metric.

        """
        rc = self._config.running_config
        return MeanSquaredError(squared=rc.squared, num_outputs=rc.num_outputs)

    def _score_name(self) -> str:
        """Return ``"mse"``, or ``"rmse"`` when ``squared`` is ``False``.

        Returns:
            The name of the score column.

        """
        return "mse" if self._config.running_config.squared else "rmse"
