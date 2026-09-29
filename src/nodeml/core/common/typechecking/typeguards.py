"""Runtime type-guard functions for NodeML node and config introspection."""

from typing import TypeGuard

from pydantic import BaseModel

from nodeml.core.common.typechecking.protocols import (
    AcceptsInputsSourceNode,
    HasHyperparametersConfig,
    HasHyperparametersNode,
    HasHyperparameterSpace,
    HasParamsNode,
    HasRunningConfigConfig,
    HasRunningConfigNode,
)
from nodeml.core.nodes.data_sink.sink import Sink
from nodeml.core.nodes.node import Node, NodeConfig, NodeType


def _is_non_empty_model(value: object) -> bool:
    """Return ``True`` if *value* is a pydantic model with at least one field."""
    # model_fields is read from the class: reading it from an instance is
    # deprecated since pydantic 2.11.
    return isinstance(value, BaseModel) and len(type(value).model_fields) > 0


# has Hyperparameters typeguards
def has_hyperparameters(obj: Node) -> TypeGuard[HasHyperparametersNode]:
    """Check whether a Node has non-empty hyperparameters in its config.

    Args:
        obj: The node to inspect.

    Returns:
        ``True`` if ``obj.config.hyperparameters`` is a ``BaseModel`` with at
        least one field.

    """
    return has_hyperparameters_config(obj.config)


def has_hyperparameters_config(obj: NodeConfig) -> TypeGuard[HasHyperparametersConfig]:
    """Check whether a NodeConfig has non-empty hyperparameters.

    Args:
        obj: The node configuration to inspect.

    Returns:
        ``True`` if ``obj.hyperparameters`` is a ``BaseModel`` with at least
        one field.

    """
    return _is_non_empty_model(getattr(obj, "hyperparameters", None))


# has RunningConfig typeguards
def has_running_config(obj: Node) -> TypeGuard[HasRunningConfigNode]:
    """Check whether a Node has a non-empty running configuration.

    Args:
        obj: The node to inspect.

    Returns:
        ``True`` if ``obj.config.running_config`` is a ``BaseModel`` with at
        least one field.

    """
    return has_running_config_config(obj.config)


def has_running_config_config(obj: NodeConfig) -> TypeGuard[HasRunningConfigConfig]:
    """Check whether a NodeConfig has a non-empty running configuration.

    Args:
        obj: The node configuration to inspect.

    Returns:
        ``True`` if ``obj.running_config`` is a ``BaseModel`` with at least
        one field.

    """
    return _is_non_empty_model(getattr(obj, "running_config", None))


# has Hyperparameter_space typeguards
def has_hyperparameter_space(obj: Node) -> TypeGuard[HasHyperparameterSpace]:
    """Check whether a Node exposes a hyperparameter search space.

    Args:
        obj: The node to inspect.

    Returns:
        ``True`` if ``obj.hyperparameter_space`` exists and is a ``dict``.

    """
    return hasattr(obj, "hyperparameter_space") and isinstance(
        obj.hyperparameter_space,  # type: ignore[attr-defined]
        dict,
    )


# has Params typeguards
def has_params(obj: Node) -> TypeGuard[HasParamsNode]:
    """Check whether a Node implements ``get_params`` and ``set_params``.

    Args:
        obj: The node to inspect.

    Returns:
        ``True`` if both methods exist and are callable.

    """
    return (
        hasattr(obj, "get_params")
        and callable(obj.get_params)  # type: ignore[attr-defined]
        and hasattr(obj, "set_params")
        and callable(obj.set_params)  # type: ignore[attr-defined]
    )


# is Sink Node typeguard
def is_sink_node(obj: Node) -> TypeGuard[Sink]:
    """Check whether a Node is a Sink node.

    Args:
        obj: The node to inspect.

    Returns:
        ``True`` if *obj* has node type ``"SINK"`` and is an instance of :class:`Sink`.

    """
    return obj.config.node_type == NodeType.SINK and isinstance(obj, Sink)


# is List typeguard
def is_list(obj: object) -> TypeGuard[list]:
    """Check whether *obj* is a ``list``.

    Args:
        obj: Any Python object.

    Returns:
        ``True`` if *obj* is a list instance.

    """
    return isinstance(obj, list)


# accepts inputs typeguard
def accepts_inputs_source_node(node: Node) -> TypeGuard[AcceptsInputsSourceNode]:
    """Check whether a source Node accepts inputs from other nodes.

    Args:
        node: The node to inspect.

    Returns:
        ``True`` if *node* is a ``SOURCE`` node with ``accepts_inputs`` set.

    """
    if hasattr(node, "accepts_inputs") and node.config.node_type == NodeType.SOURCE:
        return node.accepts_inputs  # type: ignore[attr-defined]
    return False
