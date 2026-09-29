"""Sink Node module."""

import pandas as pd

from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    TabularDataContext,
)
from nodeml.core.nodes.node import Node, NodeConfig, NodeMetadata, NodeType, Port


class SinkConfig(NodeConfig):
    """Configuration for a Sink node.

    A pipeline has one Sink.  Send every output that you need to the Sink
    with an edge: ``Edge(source="model", target="sink", ports_map=[("pred",
    "pred")])``.  At compile time, the pipeline creates one Sink input port
    and one Sink output port for each ``(source_port, sink_port)`` pair, with
    the definition of the source port.  The Sink ports follow the order of
    the edges.  So the default config declares no ports.
    """

    node_type: NodeType = NodeType.SINK
    in_ports: dict[str, Port] = {}
    out_ports: dict[str, Port] = {}


class SinkMetadata(NodeMetadata):
    """Sink Node metadata."""


class Sink(Node[pd.DataFrame, TabularDataContext, pd.DataFrame, TabularDataContext]):
    """Terminal node that collects pipeline outputs.

    A Sink consumes data on its input ports and passes it through unchanged,
    in the order of its output ports.  The ports are created during pipeline
    compilation (see :class:`SinkConfig`).
    """

    metadata = SinkMetadata()

    def __init__(self, *, config: SinkConfig) -> None:
        """Initialize the Sink Node with the given configuration."""
        self._config = config

    def node_fit(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> None:
        """Fit the Sink Node. This method will be called during the training phase of the pipeline."""
        # Nothing to be done here

    def node_transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Return the inputs unchanged, ordered like the output ports.

        Inputs without a matching output port come last, in their input
        order.
        """
        ordered = {name: data[name] for name in self.out_ports if name in data}
        ordered.update(
            {name: value for name, value in data.items() if name not in ordered}
        )
        return ordered

    def add_port(self, port_name: str, port: Port | None = None) -> None:
        """Add a matching input/output port pair to the Sink.

        Args:
            port_name: Name for the new port pair.
            port: Port definition to copy.  When ``None``, the port accepts
                mixed-category pandas data of any shape.

        """
        if port is None:
            port = Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_category=DataCategoryEnum.MIXED,
                data_shape="_ _",
                desc=f"Auto port for {port_name}.",
            )
        self.in_ports[port_name] = port.model_copy(deep=True)
        self.out_ports[port_name] = port.model_copy(deep=True)
