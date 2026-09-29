"""R-squared (coefficient of determination) metric node for the NodeML Framework.

Wraps ``torchmetrics.R2Score`` and exposes it as a NodeML
:class:`~nodeml.core.nodes.metrics.metric_node.MetricNode`.  The node expects
two input ports:

* ``pred``   - predicted values ``(batch, targets)`` as a numpy array
* ``target`` - ground-truth values ``(batch, targets)`` as a numpy array

and emits one output port:

* ``score`` - scalar metric value ``(1, 1)`` as a numpy array

R² = 1 indicates a perfect fit; R² = 0 means the model predicts no better
than the target mean.  Negative values are possible when the model is
arbitrarily worse than the mean predictor.
"""

from typing import Literal

import numpy as np
import torch
from pydantic import Field
from torchmetrics import R2Score

from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    NumericalData,
    TabularDataContext,
)
from nodeml.core.nodes.metrics.metric_node import (
    MetricNode,
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
            "``'raw_values'`` returns per-output scores (not reduced to scalar)."
        ),
    )


class R2ScoreConfig(MetricNodeConfig[R2ScoreRunningConfig]):
    """Full configuration for the R2Score metric node."""

    running_config: R2ScoreRunningConfig = Field(
        default_factory=R2ScoreRunningConfig,
        description="Run-time options (adjusted, multioutput).",
    )
    in_ports: dict[str, Port] = Field(
        default={
            "pred": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch targets",
                desc="Predicted values (batch, targets).",
            ),
            "target": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch targets",
                desc="Ground-truth target values (batch, targets).",
            ),
        },
        description="Input ports: 'pred' (predictions) and 'target' (ground truth).",
    )
    out_ports: dict[str, Port] = Field(
        default={
            "score": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="1 1",
                desc="Scalar metric value.",
            ),
        },
        description="Output ports: 'score' (scalar metric value).",
    )


class R2(
    MetricNode[
        np.ndarray,
        TabularDataContext,
        np.ndarray,
        TabularDataContext,
    ]
):
    """R-squared metric node.

    Converts numpy inputs to torch tensors, delegates to
    ``torchmetrics.R2Score``, and returns the scalar result as a
    ``(1, 1)`` numpy array with a :class:`TabularDataContext`.

    Note: torchmetrics ``R2Score`` expects 1-D input for single-output
    regression.  This node squeezes the last dimension when
    ``targets == 1`` before forwarding to the metric.
    """

    metadata = R2ScoreMetadata()

    def __init__(self, *, config: R2ScoreConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config
        rc = config.running_config
        self._metric = R2Score(
            adjusted=rc.adjusted,
            multioutput=rc.multioutput,
        )

    # --- MetricNode interface ------------------------------------------------

    def update(self, data: dict[str, tuple[np.ndarray, TabularDataContext]]) -> None:
        """Feed predictions and targets into the torchmetrics accumulator."""
        pred, _ = data["pred"]
        target, _ = data["target"]
        pred_t = torch.from_numpy(pred).float()
        target_t = torch.from_numpy(target).float()
        # R2Score expects 1-D tensors for single-output regression.
        if pred_t.shape[-1] == 1:
            pred_t = pred_t.squeeze(-1)
            target_t = target_t.squeeze(-1)
        self._metric.update(pred_t, target_t)

    def compute(self) -> dict[str, tuple[np.ndarray, TabularDataContext]]:
        """Compute R² and return a ``(1, 1)`` result array."""
        value = self._metric.compute().item()
        self._metric.reset()
        return {
            "score": (
                np.array([[value]], dtype=np.float64),
                TabularDataContext(
                    columns=["r2"],
                    dtypes=[np.dtype("float64")],
                    categories=[NumericalData],
                ),
            ),
        }
