"""R-squared (coefficient of determination) metric node for the NodeML Framework.

Wraps ``torchmetrics.R2Score`` and exposes it as a NodeML
:class:`~nodeml.core.nodes.metrics.metric_node.MetricNode`.  The node expects
two input ports:

* ``pred``   - predicted values ``(batch, targets)`` as a numpy array
* ``target`` - ground-truth values ``(batch, targets)`` as a numpy array

and emits one output port:

* ``score`` - metric value ``(1, 1)`` as a numpy array, or one column per
              output ``(1, targets)`` with ``multioutput='raw_values'``

R² = 1 indicates a perfect fit; R² = 0 means the model predicts no better
than the target mean.  Negative values are possible when the model is
arbitrarily worse than the mean predictor.
"""

from typing import Literal

from pydantic import Field
from torchmetrics import Metric, R2Score

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


class R2ScoreMetadata(MetricNodeMetadata):
    """Metadata for the R2Score metric node."""

    node_name: str = "R2Score"
    description: str = (
        "Coefficient of determination (R²) based on torchmetrics. "
        "Measures the proportion of variance in the target that is "
        "explained by the predictions. "
        "R²=1 is a perfect fit, R²=0 matches the mean predictor."
    )


class R2ScoreRunningConfig(MetricNodeRunningConfig):
    """Run-time options for the R2Score metric node."""

    adjusted: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of independent variables used to calculate the adjusted R². "
            "When set to 0 (default), the standard R² is returned. "
            "Adjusted R² penalises model complexity."
        ),
    )
    multioutput: Literal[
        "uniform_average",
        "raw_values",
        "variance_weighted",
    ] = Field(
        default="uniform_average",
        description=(
            "Strategy for aggregating across multiple outputs. "
            "``'uniform_average'`` averages scores equally. "
            "``'variance_weighted'`` weights by target variance. "
            "``'raw_values'`` returns one score per output "
            "(columns ``r2_0``, ``r2_1``, ...)."
        ),
    )


class R2ScoreConfig(MetricNodeConfig[R2ScoreRunningConfig]):
    """Full configuration for the R2Score metric node."""

    running_config: R2ScoreRunningConfig = Field(
        default_factory=R2ScoreRunningConfig,
        description="Run-time options (adjusted, multioutput).",
    )
    in_ports: dict[str, Port] = Field(
        default=regression_in_ports(),
        description="Input ports: 'pred' (predictions) and 'target' (ground truth).",
    )
    out_ports: dict[str, Port] = Field(
        default=score_out_ports(
            "Metric value (1, 1), or one value per output (1, targets) "
            "with multioutput='raw_values'.",
            per_output=True,
        ),
        description="Output ports: 'score' (metric value).",
    )


class R2(RegressionMetricNode):
    """R-squared metric node.

    Converts numpy inputs to torch tensors, delegates to
    ``torchmetrics.R2Score``, and returns the result as one row
    of float64 values: one column ``r2``, or one column per output
    ``r2_0``, ``r2_1``, ... with ``multioutput='raw_values'``.
    """

    metadata = R2ScoreMetadata()
    score_name = "r2"

    def _build_metric(self) -> Metric:
        """Build the torchmetrics metric.

        Returns:
            An ``R2Score`` metric.

        """
        rc = self._config.running_config
        return R2Score(adjusted=rc.adjusted, multioutput=rc.multioutput)
