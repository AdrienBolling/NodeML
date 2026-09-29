"""Define the base DataSource class for the NodeML Framework."""

from abc import ABC, abstractmethod

from pydantic import BaseModel

from nodeml.core.nodes.node import Node, NodeConfig, NodeMetadata, NodeType


class DataSourceMetadata(NodeMetadata):
    """Metadata for a DataSource node."""

    _node_type: NodeType = NodeType.SOURCE


class DataSourceRunningConfig(BaseModel):
    """Runtime parameters for executing a DataSource node.

    Subclass this to add source-specific execution settings (e.g. connection
    timeouts, batch sizes) that are not part of the source's identity.
    """


class DataSourceConfig[R: DataSourceRunningConfig](NodeConfig):
    """Configuration for all DataSource nodes in the NodeML Framework.

    Generic over ``R``, which must be a :class:`DataSourceRunningConfig`
    subclass carrying source-specific runtime parameters.

    Attributes:
        node_type: Always ``NodeType.SOURCE``.
        running_config: Optional runtime parameters for source execution.

    """

    node_type: NodeType = NodeType.SOURCE
    running_config: R | None = None


class DataSourceNode[D_I, D_C_I, D_O, D_C_O](Node[D_I, D_C_I, D_O, D_C_O], ABC):
    """Base class for all data source nodes in the NodeML Framework."""

    metadata = DataSourceMetadata()

    def __init__(self, *, config: DataSourceConfig) -> None:
        """Initialize the DataSourceNode with the given configuration."""
        self._config = config

    @abstractmethod
    def setup_source(self) -> None:
        """Set up the data source (e.g. establish connections, load resources).

        Called by :meth:`node_fit` during the pipeline's fit phase.
        """
        ...

    @abstractmethod
    def fetch_data(
        self, data: dict[str, tuple[D_I, D_C_I]]
    ) -> dict[str, tuple[D_O, D_C_O]]:
        """Fetch data from the source.

        Args:
            data: Mapping of port name to ``(data, context)`` tuples. Typically
                unused for data sources but provided for API consistency.

        Returns:
            Mapping of output port name to ``(data, context)`` tuples.

        """
        ...

    # --- Node API implementation --- Don't touch these unless you know what you're doing ---

    def node_fit(self, data: dict[str, tuple[D_I, D_C_I]]) -> None:
        """Fit the data source node with the given data.

        Args:
            data: Dictionary (unused for data sources)

        """
        _ = data  # Unused for data sources
        self.setup_source()

    def node_transform(
        self, data: dict[str, tuple[D_I, D_C_I]]
    ) -> dict[str, tuple[D_O, D_C_O]]:
        """Transform data through the Node by fetching data.

        Args:
            data: Dictionary (unused for data sources)

        Returns:
            Fetched data

        """
        return self.fetch_data(data)

    # --- API convenience ---

    @property
    def running_config(self) -> DataSourceRunningConfig | None:
        """Convenience property to access the running configuration of the data source."""
        return self._config.running_config

    @property
    def config(self) -> DataSourceConfig:
        """Convenience property to access the full configuration of the data source."""
        return self._config
