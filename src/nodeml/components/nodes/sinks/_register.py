"""Register sink nodes from this package into the NODE_REGISTRY."""

from nodeml.core.nodes.data_sink.sink import (
    Sink,
    SinkConfig,
)
from nodeml.core.nodes.registry.node_registry import NODE_REGISTRY


# --- Registering the nodes in the registry ---
# Do this for all the nodes defined in this module, so that they can be easily accessed and used in the NodeML Framework.
def register_nodes() -> None:
    """Register all sink nodes defined in this package."""
    NODE_REGISTRY.register(
        name="Sink",
        node_class=Sink,
        node_config_class=SinkConfig,
    )
