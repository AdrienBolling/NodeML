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


# ---------------------------------------------------------------------------
# Config round trip (save -> load keeps every node config field)
# ---------------------------------------------------------------------------


def _rf_pipeline(n_estimators: int = 10) -> Pipeline:
    from nodeml.components.nodes.models.random_forest_regressor import (
        RandomForestRegressorConfig,
    )
    from tests.shims.pipelines import build_source_model_sink_pipeline

    cfg = RandomForestRegressorConfig()
    cfg = cfg.model_copy(
        update={
            "hyperparameters": cfg.hyperparameters.model_copy(
                update={"n_estimators": n_estimators}
            )
        }
    )
    return build_source_model_sink_pipeline(
        model_node_type="RandomForestRegressor", model_config=cfg
    )


class TestConfigRoundTrip:
    def test_dump_contains_hyperparameters_and_running_config(self) -> None:
        dumped = _rf_pipeline().config.model_dump(mode="json")
        model_cfg = dumped["nodes"]["model"][1]
        assert model_cfg["hyperparameters"]["n_estimators"] == 10
        assert "running_config" in model_cfg

    def test_hash_changes_when_hyperparameters_change(self) -> None:
        assert _rf_pipeline(10).hash() != _rf_pipeline(20).hash()

    def test_validate_json_rebuilds_concrete_config_classes(self) -> None:
        from nodeml.components.nodes.models.random_forest_regressor import (
            RandomForestRegressorConfig,
        )

        original = _rf_pipeline(7)
        loaded = PipelineConfig.model_validate_json(original.config.model_dump_json())
        _, model_cfg = loaded.nodes["model"]
        assert isinstance(model_cfg, RandomForestRegressorConfig)
        assert model_cfg.hyperparameters.n_estimators == 7

    def test_save_then_from_config_file_keeps_the_hash(self, tmp_path) -> None:
        original = _rf_pipeline(7)
        original.save_config_to_dir(tmp_path)
        loaded = Pipeline.from_config_file(
            tmp_path / f"{original.file_basename}_config.json"
        )
        assert loaded.hash() == original.hash()
        assert loaded.compiled is False

    def test_load_from_dir_compiles_and_restores_params(
        self, tmp_path, regression_dataset
    ) -> None:
        from nodeml.core.pipeline.runners.smart_runner import (
            SmartRunner,
            SmartRunnerConfig,
        )

        X_pair, y_pair, _ = regression_dataset
        original = _rf_pipeline(5)
        runner = SmartRunner(original, config=SmartRunnerConfig(verbose=False))
        runner.train(input_data={"source": {"X": X_pair, "y": y_pair}})
        expected = runner.infer(input_data={"source": {"X": X_pair}})["pred"][0]
        original.save_config_to_dir(tmp_path)
        original.save_params_to_dir(tmp_path)

        loaded = Pipeline.load_from_dir(tmp_path)
        assert loaded.compiled is True
        loaded_runner = SmartRunner(loaded, config=SmartRunnerConfig(verbose=False))
        actual = loaded_runner.infer(input_data={"source": {"X": X_pair}})["pred"][0]
        assert (actual.to_numpy() == expected.to_numpy()).all()

    def test_load_from_dir_needs_a_single_config_file(self, tmp_path) -> None:
        from nodeml.core.common.exceptions import PipelineError

        with pytest.raises(PipelineError, match="exactly one"):
            Pipeline.load_from_dir(tmp_path)

    def test_save_methods_create_the_directory(self, tmp_path) -> None:
        target = tmp_path / "new" / "dir"
        _build_linear_pipeline().save_config_to_dir(target)
        assert any(target.glob("*_config.json"))


# ---------------------------------------------------------------------------
# Editing a compiled pipeline
# ---------------------------------------------------------------------------


class TestEditing:
    def test_edit_after_compile_marks_uncompiled(self) -> None:
        p = _build_linear_pipeline()
        p.compile()
        p.add_node("other", "InputsPassthrough", _source_cfg())
        assert p.compiled is False

    def test_recompile_after_edit_reuses_unchanged_node_objects(self) -> None:
        p = _build_linear_pipeline()
        p.compile()
        source_object = p.node_objects["source"]
        p.add_node("other", "InputsPassthrough", _source_cfg())
        p.add_edge(Edge(source="other", target="sink", ports_map=[("output", "other")]))
        p.compile()
        assert p.node_objects["source"] is source_object
        assert "other" in p.node_objects

    def test_failed_edit_keeps_compiled_state(self) -> None:
        p = _build_linear_pipeline()
        p.compile()
        with pytest.raises(ValueError, match="does not exist"):
            p.remove_node("missing")
        assert p.compiled is True

    def test_remove_node_removes_its_edges(self) -> None:
        p = _build_linear_pipeline()
        p.remove_node("source")
        assert p.edges == []
        assert not p.graph.has_node("source")

    def test_compile_works_after_removing_a_connected_node(
        self,
    ) -> None:
        p = _build_linear_pipeline()
        p.add_node("other", "InputsPassthrough", _source_cfg())
        p.add_edge(Edge(source="other", target="sink", ports_map=[("output", "other")]))
        p.remove_node("source")
        p.compile()
        assert "source" not in p.node_objects

    def test_adding_the_same_sink_edge_twice_then_removing_it(self) -> None:
        p = _build_linear_pipeline()
        p.add_edge(
            Edge(source="source", target="sink", ports_map=[("output", "output")])
        )
        assert len(p.sink_edges_dict) == 1
        p.remove_edge("source", "sink")
        assert p.sink_edges_dict == {}

    def test_merged_ports_map_keeps_insertion_order(self) -> None:
        p = _build_linear_pipeline()
        p.add_edge(Edge(source="source", target="sink", ports_map=[("output", "b")]))
        p.add_edge(Edge(source="source", target="sink", ports_map=[("output", "a")]))
        assert p.get_edge("source", "sink").ports_map == [
            ("output", "output"),
            ("output", "b"),
            ("output", "a"),
        ]

    def test_add_edge_does_not_alias_the_callers_edge(self) -> None:
        p = _build_linear_pipeline()
        edge = Edge(source="source", target="sink", ports_map=[("output", "b")])
        p.add_edge(edge)
        assert edge.ports_map == [("output", "b")]

    def test_update_node_rejects_another_node_type(self) -> None:
        p = _build_linear_pipeline()
        with pytest.raises(ValueError, match="Cannot change the type"):
            p.update_node("source", SinkConfig())

    def test_update_edge_validates_ports(self) -> None:
        from nodeml.components.nodes.transforms.scalers.standard_scaler import (
            StandardScalerConfig,
        )

        p = _build_linear_pipeline()
        p.add_node("scaler", "StandardScaler", StandardScalerConfig())
        p.add_edge(
            Edge(source="source", target="scaler", ports_map=[("output", "input")])
        )
        with pytest.raises(ValueError, match="does not exist"):
            p.update_edge(
                Edge(source="source", target="scaler", ports_map=[("output", "nope")])
            )
        assert p.get_edge("source", "scaler").ports_map == [("output", "input")]


# ---------------------------------------------------------------------------
# Sink ports are created at compile time
# ---------------------------------------------------------------------------


class TestSinkPorts:
    def test_default_sink_gets_ports_from_edges_in_edge_order(self) -> None:
        p = Pipeline()
        p.add_node("a", "InputsPassthrough", _source_cfg())
        p.add_node("b", "InputsPassthrough", _source_cfg())
        p.add_node("sink", "Sink")
        p.add_edge(Edge(source="b", target="sink", ports_map=[("output", "second")]))
        p.add_edge(Edge(source="a", target="sink", ports_map=[("output", "first")]))
        p.compile()
        sink = p.node_objects["sink"]
        assert list(sink.in_ports) == ["second", "first"]
        assert list(sink.out_ports) == ["second", "first"]
        assert sink.in_ports["first"].data_category == DataCategoryEnum.NUMERICAL
        # The stored config does not change, so the hash stays stable.
        assert p.get_node_config("sink").in_ports == {}

    def test_sink_port_copies_the_source_port_modes(self) -> None:
        from tests.shims.pipelines import build_source_model_sink_pipeline

        p = build_source_model_sink_pipeline()
        p.add_edge(Edge(source="source", target="sink", ports_map=[("y", "target")]))
        p.compile()
        sink = p.node_objects["sink"]
        assert (
            sink.in_ports["target"].mode
            == p.get_node_config("source").out_ports["y"].mode
        )

    def test_two_outputs_on_one_sink_port_are_rejected(self) -> None:
        p = _build_linear_pipeline()
        p.add_node("other", "InputsPassthrough", _source_cfg())
        with pytest.raises(ValueError, match="receives both"):
            p.add_edge(
                Edge(source="other", target="sink", ports_map=[("output", "output")])
            )
        assert ("other", "sink") not in p.edges_dict

    def test_hash_is_the_same_before_and_after_compile(self) -> None:
        p = _build_linear_pipeline()
        before = p.hash()
        p.compile()
        assert p.hash() == before


class TestGraphWithoutMetrics:
    def test_metric_nodes_are_excluded(self) -> None:
        from tests.shims.pipelines import build_source_model_sink_pipeline

        p = build_source_model_sink_pipeline(with_metric=True)
        view = p.graph_wo_metrics
        assert "metric" in p.graph.nodes
        assert "metric" not in view.nodes
        assert set(view.nodes) == {"source", "model", "sink"}
