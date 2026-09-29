"""Node Registry implementation."""

from pydantic import BaseModel

from nodeml.core.common.registry import Registry
from nodeml.core.nodes.node import Node, NodeConfig


class NodeRegistry(Registry):
    """Registry for Nodes in the NodeML framework.

    This registry allows for registering and retrieving nodes by name, as well as listing all registered nodes.
    """

    _column_order: list[str] = [
        "name",
        "node_type",
        "node_class",
        "node_config_class",
        "running_config",
        "hyperparameters",
    ]

    def __init__(self) -> None:
        """Initialise an empty node registry."""
        super().__init__(entity="node")

    def register(
        self,
        name: str,
        node_class: type[Node],
        node_config_class: type[NodeConfig],
        running_config_class: type[BaseModel] | None = None,
        hyperparameters_class: type[BaseModel] | None = None,
    ) -> None:
        """Register a node in the registry.

        Args:
            name: The name of the node to register.
            node_class: The class of the node to register.
            node_config_class: The class of the node's configuration to register.
            running_config_class: The class of the node's running configuration to register.
            hyperparameters_class: The class of the node's hyperparameters to register.

        """
        # Read node_type from the field default; do not build a config,
        # because a config class can have required fields.
        node_type = str(node_config_class.model_fields["node_type"].default)

        self._register(
            name=name,
            class_callable=node_class,
            config_class_callable=node_config_class,
            node_type=node_type,
            running_config=running_config_class,
            hyperparameters=hyperparameters_class,
            metadata={
                **node_class.metadata.model_dump(),
            },
        )

    def get_node_config_class(self, name: str) -> type[NodeConfig]:
        """Get the configuration class for a registered node.

        Args:
            name: The name of the node to get the configuration class for.

        Returns:
            The configuration class for the registered node.

        Raises:
            RegistryError: If *name* is not registered.

        """
        return self.get(name)["node_config_class"]

    def get_node_class(self, name: str) -> type[Node]:
        """Get the class for a registered node.

        Args:
            name: The name of the node to get the class for.

        Returns:
            The class for the registered node.

        Raises:
            RegistryError: If *name* is not registered.

        """
        return self.get(name)["node_class"]


NODE_REGISTRY = NodeRegistry()
