"""Runtime type-guard functions for NodeML node and config introspection."""

from typing import TypeGuard

import torch
from pydantic import BaseModel

from nodeml.core.common.typechecking.protocols import (
    AcceptsInputsSourceNode,
    HasHyperparametersConfig,
    HasHyperparametersNode,
    HasHyperparameterSpace,
    HasParamsNode,
    HasRunningConfigConfig,
    HasRunningConfigNode,
    IsValidTorchModule,
)
from nodeml.core.nodes.data_sink.sink import Sink
from nodeml.core.nodes.node import Node, NodeConfig, NodeType


# has Hyperparameters typeguards
def has_hyperparameters(obj: Node) -> TypeGuard[HasHyperparametersNode]:
    """Check whether a Node has non-empty hyperparameters in its config.

    Args:
        obj: The node to inspect.

    Returns:
        ``True`` if ``obj.config.hyperparameters`` exists, is a non-empty
        ``BaseModel``, and has at least one populated field.

    """
    hp = getattr(obj.config, "hyperparameters", None)
    # Check if hp exists and is not None
    if hp is None:
        return False
    # Check if hp is a BaseModel and is not empty
    if len(hp.model_fields) == 0:
        return False
    # If hp is a BaseModel, check if it is not empty by checking if its model_dump() is not empty
    return not (isinstance(hp, BaseModel) and not hp.model_dump())


def has_hyperparameters_config(obj: NodeConfig) -> TypeGuard[HasHyperparametersConfig]:
    """Check whether a NodeConfig has non-empty hyperparameters.

    Args:
        obj: The node configuration to inspect.

    Returns:
        ``True`` if ``obj.hyperparameters`` is a non-empty ``BaseModel``.

    """
    hp = getattr(obj, "hyperparameters", None)
    # Check if hp exists and is not None
    if hp is None:
        return False
    # Check if hp is a BaseModel and is not empty
    if len(hp.model_fields) == 0:
        return False
    # If hp is a BaseModel, check if it is not empty by checking if its model_dump() is not empty
    return isinstance(hp, BaseModel) and bool(hp.model_dump())


# has RunningConfig typeguards
def has_running_config(obj: Node) -> TypeGuard[HasRunningConfigNode]:
    """Check whether a Node has a non-empty running configuration.

    Args:
        obj: The node to inspect.

    Returns:
        ``True`` if ``obj.config.running_config`` is a non-empty ``BaseModel``.

    """
    rc = getattr(obj.config, "running_config", None)
    # Check if rc exists and is not None
    if rc is None:
        return False
    # Check if rc is a BaseModel and is not empty
    if len(rc.model_fields) == 0:
        return False
    # If rc is a BaseModel, check if it is not empty by checking if its model_dump() is not empty
    return isinstance(rc, BaseModel) and bool(rc.model_dump())


def has_running_config_config(obj: NodeConfig) -> TypeGuard[HasRunningConfigConfig]:
    """Check whether a NodeConfig has a non-empty running configuration.

    Args:
        obj: The node configuration to inspect.

    Returns:
        ``True`` if ``obj.running_config`` is a non-empty ``BaseModel``.

    """
    rc = getattr(obj, "running_config", None)
    # Check if rc exists and is not None
    if rc is None:
        return False
    # Check if rc is a BaseModel and is not empty
    if len(rc.model_fields) == 0:
        return False
    # If rc is a BaseModel, check if it is not empty by checking if its model_dump() is not empty
    return isinstance(rc, BaseModel) and bool(rc.model_dump())


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


# is Valid Torch Module typeguard
def is_valid_torch_module(obj: type) -> TypeGuard[IsValidTorchModule]:
    """Check whether *obj* is a valid torch.nn.Module with a train_module method.

    Args:
        obj: The object to inspect.

    Returns:
        ``True`` if *obj* is a subclass of torch.nn.Module and has a callable train_module method.

    """
    return (
        issubclass(obj, torch.nn.Module)
        and hasattr(obj, "train_module")
        and callable(obj.train_module)
    )
