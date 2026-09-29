"""Tests for :class:`nodeml.core.nodes.data_sink.sink.Sink`."""

from __future__ import annotations

from nodeml.core.common.data.data import ArrayLikeEnum, DataCategoryEnum
from nodeml.core.nodes.data_sink.sink import Sink, SinkConfig
from nodeml.core.nodes.node import NodeType, Port
from tests.shims.tabular import numerical_pair


class TestSink:
    def test_default_config_declares_no_ports(self) -> None:
        # The pipeline creates the Sink ports from the edges at compile time.
        cfg = SinkConfig()
        assert cfg.in_ports == {}
        assert cfg.out_ports == {}
        assert cfg.node_type == NodeType.SINK

    def test_add_port_adds_paired_in_out_ports(self) -> None:
        cfg = SinkConfig()
        sink = Sink(config=cfg)

        sink.add_port("pred")
        assert "pred" in sink.in_ports
        assert "pred" in sink.out_ports

    def test_add_port_copies_the_given_port(self) -> None:
        port = Port(
            arr_type=ArrayLikeEnum.NUMPY,
            data_category=DataCategoryEnum.NUMERICAL,
            data_shape="batch 1",
            desc="pred",
        )
        sink = Sink(config=SinkConfig())
        sink.add_port("pred", port)
        assert sink.in_ports["pred"] == port
        assert sink.in_ports["pred"] is not port

    def test_node_transform_is_pass_through(self) -> None:
        sink = Sink(config=SinkConfig())
        payload = {"dump": numerical_pair()}
        # Sinks do not mutate their inputs.
        assert sink.node_transform(payload) == payload

    def test_node_transform_follows_the_output_port_order(self) -> None:
        sink = Sink(config=SinkConfig())
        sink.add_port("first")
        sink.add_port("second")
        pair = numerical_pair()
        out = sink.node_transform({"second": pair, "first": pair})
        assert list(out) == ["first", "second"]
