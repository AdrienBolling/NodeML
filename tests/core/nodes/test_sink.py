"""Tests for :class:`nodeml.core.nodes.data_sink.sink.Sink`."""

from __future__ import annotations

from nodeml.core.nodes.data_sink.sink import Sink, SinkConfig
from nodeml.core.nodes.node import NodeType
from tests.shims.tabular import numerical_pair


class TestSink:
    def test_default_config_declares_dump_port(self) -> None:
        cfg = SinkConfig()
        assert "dump" in cfg.in_ports
        assert cfg.node_type == NodeType.SINK

    def test_add_port_adds_paired_in_out_ports(self) -> None:
        cfg = SinkConfig()
        sink = Sink(config=cfg)

        sink.add_port("pred")
        assert "pred" in sink.in_ports
        assert "pred" in sink.out_ports

    def test_node_transform_is_pass_through(self) -> None:
        sink = Sink(config=SinkConfig())
        payload = {"dump": numerical_pair()}
        # Sinks do not mutate their inputs.
        assert sink.node_transform(payload) == payload
