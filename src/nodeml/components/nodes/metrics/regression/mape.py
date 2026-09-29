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

import numpy as np
import torch
from pydantic import Field
from torchmetrics import MeanAbsolutePercentageError

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


class MAPE(
    MetricNode[
        np.ndarray,
        TabularDataContext,
        np.ndarray,
        TabularDataContext,
    ]
):
    """Mean Absolute Percentage Error metric node.

    Converts numpy inputs to torch tensors, delegates to
    ``torchmetrics.MeanAbsolutePercentageError``, and returns the scalar
    result as a ``(1, 1)`` numpy array with a :class:`TabularDataContext`.
    """

    metadata = MAPEMetadata()

    def __init__(self, *, config: MAPEConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config
        self._metric = MeanAbsolutePercentageError()

    # --- MetricNode interface ------------------------------------------------

    def update(self, data: dict[str, tuple[np.ndarray, TabularDataContext]]) -> None:
        """Feed predictions and targets into the torchmetrics accumulator."""
        pred, _ = data["pred"]
        target, _ = data["target"]
        self._metric.update(
            torch.from_numpy(np.ascontiguousarray(pred)).float(),
            torch.from_numpy(np.ascontiguousarray(target)).float(),
        )

    def compute(self) -> dict[str, tuple[np.ndarray, TabularDataContext]]:
        """Compute MAPE and return a ``(1, 1)`` result array."""
        value = self._metric.compute().item()
        self._metric.reset()
        return _scalar_result(value, "mape")


def _scalar_result(
    value: float, col_name: str
) -> dict[str, tuple[np.ndarray, TabularDataContext]]:
    """Wrap a scalar metric value into the port-compatible output format."""
    return {
        "score": (
            np.array([[value]], dtype=np.float64),
            TabularDataContext(
                columns=[col_name],
                dtypes=[np.dtype("float64")],
                categories=[NumericalData],
            ),
        ),
    }
