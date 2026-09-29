"""Mean Squared Error metric node for the NodeML Framework.

Wraps ``torchmetrics.MeanSquaredError`` and exposes it as a NodeML
:class:`~nodeml.core.nodes.metrics.metric_node.MetricNode`.  The node expects
two input ports:

* ``pred``   - predicted values ``(batch, targets)`` as a numpy array
* ``target`` - ground-truth values ``(batch, targets)`` as a numpy array

and emits one output port:

* ``score`` - scalar metric value ``(1, 1)`` as a numpy array

Setting ``squared=False`` in the running config turns this into RMSE.
"""

import numpy as np
import torch
from pydantic import Field
from torchmetrics import MeanSquaredError

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
            "Number of output targets. "
            "Set to >1 for multi-output regression so that the metric "
            "averages across outputs correctly."
        ),
    )


class MSEConfig(MetricNodeConfig[MSERunningConfig]):
    """Full configuration for the MSE metric node."""

    running_config: MSERunningConfig = Field(
        default_factory=MSERunningConfig,
        description="Run-time options (squared, num_outputs).",
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


class MSE(
    MetricNode[
        np.ndarray,
        TabularDataContext,
        np.ndarray,
        TabularDataContext,
    ]
):
    """Mean Squared Error metric node.

    Converts numpy inputs to torch tensors, delegates to
    ``torchmetrics.MeanSquaredError``, and returns the scalar result as a
    ``(1, 1)`` numpy array with a :class:`TabularDataContext`.
    """

    metadata = MSEMetadata()

    def __init__(self, *, config: MSEConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config
        rc = config.running_config
        self._metric = MeanSquaredError(
            squared=rc.squared,
            num_outputs=rc.num_outputs,
        )

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
        """Compute MSE (or RMSE) and return a ``(1, 1)`` result array."""
        value = self._metric.compute().item()
        self._metric.reset()
        return _scalar_result(
            value, "mse" if self._config.running_config.squared else "rmse"
        )


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
