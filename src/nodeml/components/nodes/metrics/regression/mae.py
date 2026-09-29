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

import numpy as np
import torch
from pydantic import Field
from torchmetrics import MeanAbsoluteError

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


class MAE(
    MetricNode[
        np.ndarray,
        TabularDataContext,
        np.ndarray,
        TabularDataContext,
    ]
):
    """Mean Absolute Error metric node.

    Converts numpy inputs to torch tensors, delegates to
    ``torchmetrics.MeanAbsoluteError``, and returns the scalar result as a
    ``(1, 1)`` numpy array with a :class:`TabularDataContext`.
    """

    metadata = MAEMetadata()

    def __init__(self, *, config: MAEConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config
        rc = config.running_config
        self._metric = MeanAbsoluteError(
            num_outputs=rc.num_outputs,
        )

    # --- MetricNode interface ------------------------------------------------

    def update(self, data: dict[str, tuple[np.ndarray, TabularDataContext]]) -> None:
        """Feed predictions and targets into the torchmetrics accumulator."""
        pred, _ = data["pred"]
        target, _ = data["target"]
        self._metric.update(
            torch.from_numpy(pred).float(),
            torch.from_numpy(target).float(),
        )

    def compute(self) -> dict[str, tuple[np.ndarray, TabularDataContext]]:
        """Compute MAE and return a ``(1, 1)`` result array."""
        value = self._metric.compute().item()
        self._metric.reset()
        return {
            "score": (
                np.array([[value]], dtype=np.float64),
                TabularDataContext(
                    columns=["mae"],
                    dtypes=[np.dtype("float64")],
                    categories=[NumericalData],
                ),
            ),
        }
