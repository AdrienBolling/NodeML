"""Tests for :class:`nodeml.core.pipeline.pipeline.Pipeline`.

We exercise:

* the ``add_node`` / ``remove_node`` / ``add_edge`` / ``remove_edge`` CRUD,
* edge validation (port existence, structure/category compatibility,
  execution-mode compatibility),
* pipeline compilation and the autoprune path,
* the sink-node invariants (exactly one, no outgoing edges).
"""

from __future__ import annotations

import pytest

from nodeml.components.nodes.data_sources.inputs_passthrough import (
    InputsPassthroughConfig,
)
from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
)
from nodeml.core.nodes.data_sink.sink import SinkConfig
from nodeml.core.nodes.node import Port
from nodeml.core.pipeline.pipeline import Edge, Pipeline, PipelineConfig


def _source_cfg() -> InputsPassthroughConfig:
    return InputsPassthroughConfig(
        out_ports={
            "output": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch feature",
                desc="test",
            ),
        }
    )


def _sink_cfg_with_output_port() -> SinkConfig:
    return SinkConfig(
        in_ports={
            "output": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch feature",
                desc="test",
            ),
        }
    )


class TestPipelineCRUD:
    def test_add_node_registers_in_config(self) -> None:
        p = Pipeline()
        p.add_node("source", "InputsPassthrough", _source_cfg())

        assert "source" in p.nodes

    def test_add_duplicate_node_raises(self) -> None:
        p = Pipeline()
        p.add_node("source", "InputsPassthrough", _source_cfg())
        with pytest.raises(ValueError, match="already exists"):
            p.add_node("source", "InputsPassthrough", _source_cfg())

    def test_second_sink_is_rejected(self) -> None:
        p = Pipeline()
        p.add_node("sink_a", "Sink", _sink_cfg_with_output_port())
        with pytest.raises(ValueError, match="Sink node"):
            p.add_node("sink_b", "Sink", _sink_cfg_with_output_port())

    def test_remove_node_removes_from_config_and_graph(self) -> None:
        p = Pipeline()
        p.add_node("source", "InputsPassthrough", _source_cfg())
        p.remove_node("source")
        assert "source" not in p.nodes
        assert "source" not in p.graph.nodes


class TestEdgeValidation:
    def test_missing_source_port_raises(self) -> None:
        p = Pipeline()
        p.add_node("source", "InputsPassthrough", _source_cfg())
        p.add_node("sink", "Sink", _sink_cfg_with_output_port())
        with pytest.raises(ValueError, match="does not exist"):
            p.add_edge(
                Edge(
                    source="source",
                    target="sink",
                    ports_map=[("missing", "output")],
                )
            )

    def test_sink_cannot_be_edge_source(self) -> None:
        from nodeml.components.nodes.transforms.scalers.standard_scaler import (
            StandardScalerConfig,
        )

        p = Pipeline()
        p.add_node(
            "sink",
            "Sink",
            SinkConfig(
                in_ports={
                    "dump": Port(
                        arr_type=ArrayLikeEnum.PANDAS,
                        data_structure=DataStructureEnum.TABULAR,
                        data_category=DataCategoryEnum.MIXED,
                        data_shape="_ _",
                        desc="dump",
                    ),
                },
                out_ports={
                    "output": Port(
                        arr_type=ArrayLikeEnum.PANDAS,
                        data_structure=DataStructureEnum.TABULAR,
                        data_category=DataCategoryEnum.NUMERICAL,
                        data_shape="batch feature",
                        desc="test",
                    ),
                },
            ),
        )
        # A downstream node whose in_port is compatible with the sink's
        # out_port — so port-existence and compatibility checks pass and
        # the Sink-as-source check is actually exercised.
        p.add_node("downstream", "StandardScaler", StandardScalerConfig())
        with pytest.raises(ValueError, match="originates from a Sink"):
            p.add_edge(
                Edge(
                    source="sink",
                    target="downstream",
                    ports_map=[("output", "input")],
                )
            )


class TestCompile:
    def test_compile_sets_compiled_flag(self) -> None:
        p = Pipeline()
        p.add_node("source", "InputsPassthrough", _source_cfg())
        p.add_node("sink", "Sink", _sink_cfg_with_output_port())
        p.add_edge(
            Edge(
                source="source",
                target="sink",
                ports_map=[("output", "output")],
            )
        )
        p.compile()
        assert p.compiled is True
        assert "source" in p.node_objects
        assert "sink" in p.node_objects

    def test_compile_without_sink_raises(self) -> None:
        p = Pipeline()
        p.add_node("source", "InputsPassthrough", _source_cfg())
        with pytest.raises(ValueError, match="Sink"):
            p.compile()

    def test_compile_twice_raises(self) -> None:
        p = Pipeline()
        p.add_node("source", "InputsPassthrough", _source_cfg())
        p.add_node("sink", "Sink", _sink_cfg_with_output_port())
        p.add_edge(
            Edge(
                source="source",
                target="sink",
                ports_map=[("output", "output")],
            )
        )
        p.compile()
        with pytest.raises(ValueError, match="already compiled"):
            p.compile()

    def test_autoprune_removes_disconnected_nodes(self) -> None:
        p = Pipeline(config=PipelineConfig(name="autoprune_test"))
        p.add_node("source", "InputsPassthrough", _source_cfg())
        p.add_node("orphan", "InputsPassthrough", _source_cfg())
        p.add_node("sink", "Sink", _sink_cfg_with_output_port())
        p.add_edge(
            Edge(
                source="source",
                target="sink",
                ports_map=[("output", "output")],
            )
        )
        p.compile()
        assert "orphan" not in p.node_objects


class TestConfigSnapshot:
    def test_config_round_trip_preserves_edges(self) -> None:
        p = Pipeline()
        p.add_node("source", "InputsPassthrough", _source_cfg())
        p.add_node("sink", "Sink", _sink_cfg_with_output_port())
        p.add_edge(
            Edge(
                source="source",
                target="sink",
                ports_map=[("output", "output")],
            )
        )

        snapshot = p.config
        assert len(snapshot.edges) == 1
        assert snapshot.edges[0].source == "source"


def _build_linear_pipeline() -> Pipeline:
    p = Pipeline()
    p.add_node("source", "InputsPassthrough", _source_cfg())
    p.add_node("sink", "Sink", _sink_cfg_with_output_port())
    p.add_edge(
        Edge(source="source", target="sink", ports_map=[("output", "output")]),
    )
    return p


class TestPipelineHash:
    def test_hash_is_64_char_hex_string(self) -> None:
        h = _build_linear_pipeline().hash()
        assert isinstance(h, str)
        assert len(h) == 64
        int(h, 16)  # must be valid hex

    def test_same_config_yields_same_hash(self) -> None:
        assert _build_linear_pipeline().hash() == _build_linear_pipeline().hash()

    def test_hash_is_insertion_order_invariant(self) -> None:
        p_a = Pipeline()
        p_a.add_node("source", "InputsPassthrough", _source_cfg())
        p_a.add_node("sink", "Sink", _sink_cfg_with_output_port())
        p_a.add_edge(
            Edge(source="source", target="sink", ports_map=[("output", "output")]),
        )

        p_b = Pipeline()
        p_b.add_node("sink", "Sink", _sink_cfg_with_output_port())
        p_b.add_node("source", "InputsPassthrough", _source_cfg())
        p_b.add_edge(
            Edge(source="source", target="sink", ports_map=[("output", "output")]),
        )

        assert p_a.hash() == p_b.hash()

    def test_hash_changes_when_layout_changes(self) -> None:
        p_one_edge = _build_linear_pipeline()

        p_two_nodes = Pipeline()
        p_two_nodes.add_node("source", "InputsPassthrough", _source_cfg())
        p_two_nodes.add_node("other", "InputsPassthrough", _source_cfg())
        p_two_nodes.add_node("sink", "Sink", _sink_cfg_with_output_port())
        p_two_nodes.add_edge(
            Edge(source="source", target="sink", ports_map=[("output", "output")]),
        )

        assert p_one_edge.hash() != p_two_nodes.hash()

    def test_hash_changes_when_pipeline_name_changes(self) -> None:
        p_default = _build_linear_pipeline()
        p_renamed = Pipeline(
            config=PipelineConfig(name="other", nodes={}, edges=[]),
        )
        p_renamed.add_node("source", "InputsPassthrough", _source_cfg())
        p_renamed.add_node("sink", "Sink", _sink_cfg_with_output_port())
        p_renamed.add_edge(
            Edge(source="source", target="sink", ports_map=[("output", "output")]),
        )
        assert p_default.hash() != p_renamed.hash()

    def test_hash_ignores_node_config_instance_uuid(self) -> None:
        """Ignore the per-instance UUID of ``NodeConfig`` in the hash.

        Two ``NodeConfig`` instances with identical fields have different
        UUIDs; the pipeline hash must not change because of that.
        """
        assert _build_linear_pipeline().hash() == _build_linear_pipeline().hash()
