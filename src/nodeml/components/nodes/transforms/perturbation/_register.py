"""Register the PerturbationNode into the NODE_REGISTRY."""

from nodeml.core.nodes.registry.node_registry import NODE_REGISTRY
from nodeml.core.nodes.transform.perturbation import (
    PerturbationHyperParameters,
    PerturbationNode,
    PerturbationNodeConfig,
    PerturbationRunningConfig,
)


def register_nodes() -> None:
    """Register the PerturbationNode, which the core defines."""
    NODE_REGISTRY.register(
        name="PerturbationNode",
        node_class=PerturbationNode,
        node_config_class=PerturbationNodeConfig,
        running_config_class=PerturbationRunningConfig,
        hyperparameters_class=PerturbationHyperParameters,
    )
