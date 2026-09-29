"""NodeML — Node-based ML Pipeline Framework."""

from nodeml.components import _auto_discovery
from nodeml.core.nodes.registry.node_registry import NODE_REGISTRY as NODE_REGISTRY

# Register every built-in component in NODE_REGISTRY.
_auto_discovery()

__all__ = ["NODE_REGISTRY"]
