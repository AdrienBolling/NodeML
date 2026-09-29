"""F1 Score metric node for the NodeML Framework.

Wraps ``torchmetrics.F1Score`` and exposes it as a NodeML
:class:`~nodeml.core.nodes.metrics.metric_node.MetricNode`.  The node expects
two input ports:

* ``pred``   - predicted class probabilities or logits ``(batch, num_classes)``
               as a numpy array (for multiclass), or ``(batch, 1)`` for binary
* ``target`` - ground-truth class labels ``(batch, 1)`` as a numpy integer array

and emits one output port:

* ``score`` - scalar metric value ``(1, 1)`` as a numpy array

F1 is the harmonic mean of precision and recall, providing a single measure
that balances both.  Supports binary and multiclass tasks with configurable
averaging strategies.
"""

from typing import Literal

import numpy as np
import torch
from pydantic import Field
from torchmetrics import F1Score

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


class F1ScoreMetadata(MetricNodeMetadata):
    """Metadata for the F1Score metric node."""

    node_name: str = "F1Score"
    description: str = (
        "F1 Score based on torchmetrics. "
        "Harmonic mean of precision and recall. "
        "Supports binary and multiclass tasks with configurable averaging."
    )


class F1ScoreRunningConfig(MetricNodeRunningConfig):
    """Run-time options for the F1Score metric node."""

    task: Literal["binary", "multiclass"] = Field(
        default="binary",
        description=(
            "Classification task type. "
            "``'binary'`` expects predictions in ``(batch, 1)`` and thresholds at ``threshold``. "
            "``'multiclass'`` expects predictions in ``(batch, num_classes)`` and picks argmax."
        ),
    )
    num_classes: int | None = Field(
        default=None,
        ge=2,
        description=(
            "Number of classes. Required for ``task='multiclass'``. "
            "Ignored for binary tasks."
        ),
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Decision threshold for binary classification. "
            "Predictions above this value are assigned to the positive class. "
            "Only used when ``task='binary'``."
        ),
    )
    average: Literal["micro", "macro", "weighted"] = Field(
        default="macro",
        description=(
            "Averaging strategy for multiclass F1. "
            "``'micro'`` computes global F1 over all samples. "
            "``'macro'`` averages per-class F1 (treats all classes equally). "
            "``'weighted'`` weights per-class F1 by class support."
        ),
    )
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
        default={
            "pred": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch _",
                desc=(
                    "Predicted class probabilities or logits. "
                    "Shape (batch, 1) for binary, (batch, num_classes) for multiclass."
                ),
            ),
            "target": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch 1",
                desc="Ground-truth integer class labels (batch, 1).",
            ),
        },
        description="Input ports: 'pred' (class probabilities) and 'target' (class labels).",
    )
    out_ports: dict[str, Port] = Field(
        default={
            "score": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="1 1",
                desc="Scalar F1 score value in [0, 1].",
            ),
        },
        description="Output ports: 'score' (scalar F1 score).",
    )


class F1(
    MetricNode[
        np.ndarray,
        TabularDataContext,
        np.ndarray,
        TabularDataContext,
    ]
):
    """F1 Score metric node.

    Converts numpy inputs to torch tensors, delegates to
    ``torchmetrics.F1Score``, and returns the scalar result as a
    ``(1, 1)`` numpy array with a :class:`TabularDataContext`.
    """

    metadata = F1ScoreMetadata()

    def __init__(self, *, config: F1ScoreConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config
        rc = config.running_config
        metric_kwargs: dict = {"task": rc.task, "zero_division": rc.zero_division}
        if rc.task == "binary":
            metric_kwargs["threshold"] = rc.threshold
        if rc.task == "multiclass":
            metric_kwargs["num_classes"] = rc.num_classes
            metric_kwargs["average"] = rc.average
        self._metric = F1Score(**metric_kwargs)

    # --- MetricNode interface ------------------------------------------------

    def update(self, data: dict[str, tuple[np.ndarray, TabularDataContext]]) -> None:
        """Feed predictions and targets into the torchmetrics accumulator."""
        pred, _ = data["pred"]
        target, _ = data["target"]
        pred_t = torch.from_numpy(pred).float()
        target_t = torch.from_numpy(target).long().squeeze(-1)
        if self._config.running_config.task == "binary":
            pred_t = pred_t.squeeze(-1)
        self._metric.update(pred_t, target_t)

    def compute(self) -> dict[str, tuple[np.ndarray, TabularDataContext]]:
        """Compute F1 score and return a ``(1, 1)`` result array."""
        value = self._metric.compute().item()
        self._metric.reset()
        return {
            "score": (
                np.array([[value]], dtype=np.float64),
                TabularDataContext(
                    columns=["f1"],
                    dtypes=[np.dtype("float64")],
                    categories=[NumericalData],
                ),
            ),
        }
