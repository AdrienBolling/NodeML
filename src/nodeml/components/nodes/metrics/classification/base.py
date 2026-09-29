"""Shared base classes for the classification metric nodes.

The classification metric nodes differ only in the torchmetrics class that
they wrap, the options that they pass to it, and the name of the score.
:class:`ClassificationMetricNode` holds everything else, including the
conversion of the prediction formats below.

Accepted ``pred`` formats (``target`` is always ``(batch, 1)`` integer
labels ``0 .. num_classes - 1``):

* ``task='binary'``: positive-class probabilities ``(batch, 1)``, or class
  probabilities ``(batch, 2)`` (the node uses column 1, the positive
  class).
* ``task='multiclass'``: class probabilities ``(batch, num_classes)``, one
  column per class, in the order of the labels.
* Accuracy, F1, precision and recall also accept hard class labels
  ``(batch, 1)`` for both tasks.

This is the output format of the classifier nodes: one float64 column
``proba_<class>`` per class, also for two classes.
"""

from typing import ClassVar, Literal

import numpy as np
import torch
from pydantic import Field
from torchmetrics import Metric

from nodeml.components.nodes.metrics.base import (
    TorchMetricNode,
    TorchMetricRunningConfig,
)
from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
)
from nodeml.core.common.exceptions import NodeConfigError, NodeInputError
from nodeml.core.nodes.node import Port

# Class probabilities of a binary classifier: negative, positive.
_BINARY_PROBA_COLUMNS = 2


class ClassificationRunningConfig(TorchMetricRunningConfig):
    """Run-time options that all classification metric nodes share."""

    task: Literal["binary", "multiclass"] = Field(
        default="binary",
        description=(
            "Classification task type. "
            "``'binary'`` expects positive-class probabilities ``(batch, 1)`` "
            "or class probabilities ``(batch, 2)``. "
            "``'multiclass'`` expects class probabilities "
            "``(batch, num_classes)``."
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


class ThresholdRunningConfig(ClassificationRunningConfig):
    """Classification options with a decision threshold for binary tasks."""

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Decision threshold for binary classification. "
            "Probabilities above this value are assigned to the positive class. "
            "Integer hard labels are used as they are. "
            "Only used when ``task='binary'``."
        ),
    )


class StatScoresRunningConfig(ThresholdRunningConfig):
    """Options of the precision, recall and F1 metric nodes."""

    average: Literal["micro", "macro", "weighted"] = Field(
        default="macro",
        description=(
            "Averaging strategy for the multiclass score. "
            "``'micro'`` computes the score over all samples. "
            "``'macro'`` averages the per-class scores. "
            "``'weighted'`` weights the per-class scores by class support."
        ),
    )
    zero_division: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Score to return when the score divides by zero: precision "
            "without positive predictions, recall without positive labels, "
            "F1 without both. Defaults to 0."
        ),
    )


def classification_in_ports(*, accepts_labels: bool = True) -> dict[str, Port]:
    """Return the input ports of a classification metric node.

    The ``batch`` dimension is the same on both ports.

    Args:
        accepts_labels: If ``True``, the ``pred`` description also names
            hard class labels.

    Returns:
        Mapping with the ``pred`` and ``target`` ports.

    """
    pred_desc = (
        "Class probabilities (batch, num_classes), one column per class. "
        "For binary also positive-class probabilities (batch, 1)."
    )
    if accepts_labels:
        pred_desc += " Hard class labels (batch, 1) are also accepted."
    return {
        "pred": Port(
            arr_type=ArrayLikeEnum.NUMPY,
            data_structure=DataStructureEnum.TABULAR,
            data_category=DataCategoryEnum.NUMERICAL,
            data_shape="batch _",
            desc=pred_desc,
        ),
        "target": Port(
            arr_type=ArrayLikeEnum.NUMPY,
            data_structure=DataStructureEnum.TABULAR,
            data_category=DataCategoryEnum.NUMERICAL,
            data_shape="batch 1",
            desc="Ground-truth integer class labels (batch, 1).",
        ),
    }


def _is_integer(values: np.ndarray) -> bool:
    """Return whether *values* has an integer or boolean dtype."""
    return np.issubdtype(values.dtype, np.integer) or values.dtype == np.bool_


def _as_columns(values: np.ndarray) -> np.ndarray:
    """Return *values* as a 2-D ``(batch, columns)`` array."""
    return values.reshape(len(values), -1)


class ClassificationMetricNode(TorchMetricNode):
    """Base class for the classification metric nodes.

    A subclass sets the class attributes below.  The node builds
    ``metric_class(task=..., **options)``, where the options are the running
    config fields named in :attr:`common_options` and in the options of the
    task.

    Attributes:
        metric_class: The torchmetrics class, for example
            ``torchmetrics.Accuracy``.
        common_options: Running config fields passed for every task.
        binary_options: Running config fields passed for ``task='binary'``.
        multiclass_options: Running config fields passed for
            ``task='multiclass'``.
        accepts_labels: If ``True``, ``pred`` can also hold hard class
            labels ``(batch, 1)``.

    """

    metric_class: ClassVar[type[Metric]]
    common_options: ClassVar[tuple[str, ...]] = ()
    binary_options: ClassVar[tuple[str, ...]] = ()
    multiclass_options: ClassVar[tuple[str, ...]] = ("num_classes",)
    accepts_labels: ClassVar[bool] = True

    def _build_metric(self) -> Metric:
        """Build the torchmetrics metric for the configured task.

        Returns:
            The torchmetrics metric.

        Raises:
            NodeConfigError: If ``task='multiclass'`` and ``num_classes`` is
                not set.

        """
        rc = self._config.running_config
        if rc.task == "multiclass" and rc.num_classes is None:
            msg = f"{type(self).__name__}: task='multiclass' requires num_classes."
            raise NodeConfigError(msg)
        task_options = (
            self.binary_options if rc.task == "binary" else self.multiclass_options
        )
        options = {name: getattr(rc, name) for name in self.common_options}
        options.update({name: getattr(rc, name) for name in task_options})
        return self.metric_class(task=rc.task, **options)

    def _to_tensors(
        self, pred: np.ndarray, target: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert predictions and labels to the tensors of torchmetrics.

        Args:
            pred: Predictions in one of the formats of the module docstring.
            target: Integer class labels, ``(batch, 1)``.

        Returns:
            The ``(pred, target)`` tensors.  ``target`` is 1-D.  ``pred`` is
            1-D for binary tasks and for hard labels, else
            ``(batch, num_classes)``.

        """
        target_t = self._labels(target, port="target")
        pred = _as_columns(pred)
        if self._config.running_config.task == "binary":
            return self._binary_pred(pred), target_t
        return self._multiclass_pred(pred), target_t

    def _binary_pred(self, pred: np.ndarray) -> torch.Tensor:
        """Return the positive-class scores of a binary task.

        Args:
            pred: Predictions, ``(batch, 1)`` or ``(batch, 2)``.

        Returns:
            A 1-D tensor: float probabilities, or integer labels.

        Raises:
            NodeInputError: If *pred* has more than two columns.

        """
        n_columns = pred.shape[1]
        if n_columns > _BINARY_PROBA_COLUMNS:
            msg = (
                f"{type(self).__name__}: task 'binary' expects 1 or 2 prediction "
                f"columns, got {n_columns}. Use task='multiclass' for more classes."
            )
            raise NodeInputError(msg)
        # With two columns, column 1 is the probability of the positive class.
        positive = pred[:, n_columns - 1]
        if self.accepts_labels and _is_integer(positive):
            # Integer labels are used as they are, without the threshold.
            return torch.from_numpy(positive.astype(np.int64))
        return torch.from_numpy(positive.astype(np.float64))

    def _multiclass_pred(self, pred: np.ndarray) -> torch.Tensor:
        """Return the class probabilities or the labels of a multiclass task.

        Args:
            pred: Predictions, ``(batch, num_classes)`` or labels
                ``(batch, 1)``.

        Returns:
            A ``(batch, num_classes)`` float tensor, or 1-D integer labels.

        Raises:
            NodeInputError: If the number of columns does not match
                ``num_classes``, or if the node needs probabilities and gets
                labels.

        """
        num_classes = self._config.running_config.num_classes
        n_columns = pred.shape[1]
        if n_columns == 1:
            if not self.accepts_labels:
                msg = (
                    f"{type(self).__name__}: task 'multiclass' needs class "
                    f"probabilities (batch, {num_classes}), not hard labels."
                )
                raise NodeInputError(msg)
            return self._labels(pred, port="pred")
        if n_columns != num_classes:
            msg = (
                f"{type(self).__name__}: got {n_columns} prediction columns, but "
                f"num_classes is {num_classes}. Give one column per class."
            )
            raise NodeInputError(msg)
        return torch.from_numpy(np.ascontiguousarray(pred, dtype=np.float64))

    def _labels(self, values: np.ndarray, *, port: str) -> torch.Tensor:
        """Convert one column of class labels to a 1-D integer tensor.

        Args:
            values: Class labels, ``(batch, 1)``.  Float values must be whole
                numbers.
            port: Name of the port, for the error messages.

        Returns:
            A 1-D ``int64`` tensor.

        Raises:
            NodeInputError: If *values* has more than one column or holds
                values that are not whole numbers.

        """
        values = _as_columns(values)
        if values.shape[1] != 1:
            msg = (
                f"{type(self).__name__}: the '{port}' port expects one column of "
                f"class labels, got {values.shape[1]}."
            )
            raise NodeInputError(msg)
        labels = values[:, 0]
        if _is_integer(labels):
            return torch.from_numpy(labels.astype(np.int64))
        if np.isnan(labels).any():
            msg = (
                f"{type(self).__name__}: the '{port}' port has NaN class labels. "
                "Set nan_policy='omit' to drop these rows."
            )
            raise NodeInputError(msg)
        if not np.array_equal(labels, np.round(labels)):
            msg = (
                f"{type(self).__name__}: the '{port}' port expects integer class "
                "labels."
            )
            raise NodeInputError(msg)
        return torch.from_numpy(labels.astype(np.int64))
