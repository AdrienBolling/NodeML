"""Enumerations shared across the NodeML framework."""

from enum import StrEnum


class NodeExecutionMode(StrEnum):
    """Execution modes that control when a pipeline node is invoked.

    Members:
        TRAINING: Node runs only during the training phase.
        INFERENCE: Node runs only during the inference phase.
        EVALUATION: Node runs only during the evaluation phase.
        ALL: Node runs in every phase.
        DEFAULT: Placeholder for an unset mode; raises an error if used at runtime.
    """

    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
    ALL = "all"
    DEFAULT = "default"
