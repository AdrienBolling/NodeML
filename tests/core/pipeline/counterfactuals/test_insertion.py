"""Tests for :mod:`nodeml.core.pipeline.counterfactuals.insertion`."""

from __future__ import annotations

import numpy as np
import pytest

from nodeml.core.common.data.data import DataCategoryEnum
from nodeml.core.common.exceptions import PipelineValidationError
from nodeml.core.pipeline.counterfactuals.config import InsertionPoint, OutputPoint
from nodeml.core.pipeline.counterfactuals.insertion import (
    OUTPUT_SINK_PORT,
    PERTURBATION_NODE_NAME,
    ResolvedPoints,
    build_counterfactual_pipeline,
    insert_perturbation_node,
    resolve_points,
)
from nodeml.core.pipeline.pipeline import Edge, Pipeline
from tests.core.pipeline.counterfactuals.conftest import TrainedCase, source_config


def _points(case: TrainedCase) -> ResolvedPoints:
    return resolve_points(case.pipeline.config, InsertionPoint(), OutputPoint())


class TestResolvePoints:
    def test_default_insertion_is_the_source_port_used_in_inference(
        self, numeric_case: TrainedCase
    ) -> None:
        # source.y feeds the model only in training and evaluation.
        assert _points(numeric_case) == ResolvedPoints("source", "X", "model", "pred")

    def test_default_output_needs_one_sink_input(
        self, numeric_case: TrainedCase
    ) -> None:
        pipeline = numeric_case.pipeline
        pipeline.add_edge(
            Edge(source="scaler", target="sink", ports_map=[("output", "scaled")])
        )
        with pytest.raises(PipelineValidationError, match="Set OutputPoint.node"):
            resolve_points(pipeline.config, InsertionPoint(), OutputPoint())
        points = resolve_points(
            pipeline.config, InsertionPoint(), OutputPoint(node="model")
        )
        assert points.output_port == "pred"

    def test_two_sources_need_an_explicit_insertion_point(self) -> None:
        p = Pipeline()
        p.add_node(
            "left", "InputsPassthrough", source_config(DataCategoryEnum.NUMERICAL)
        )
        p.add_node(
            "right", "InputsPassthrough", source_config(DataCategoryEnum.NUMERICAL)
        )
        p.add_node("concat", "FeatureConcatenate")
        p.add_node("model", "LinearRegression")
        p.add_node("sink", "Sink")
        p.add_edge(Edge(source="left", target="concat", ports_map=[("X", "input_1")]))
        p.add_edge(Edge(source="right", target="concat", ports_map=[("X", "input_2")]))
        p.add_edge(Edge(source="concat", target="model", ports_map=[("output", "X")]))
        p.add_edge(Edge(source="left", target="model", ports_map=[("y", "y")]))
        p.add_edge(Edge(source="model", target="sink", ports_map=[("pred", "pred")]))
        with pytest.raises(PipelineValidationError, match="Set InsertionPoint.node"):
            resolve_points(p.config, InsertionPoint(), OutputPoint())
        points = resolve_points(p.config, InsertionPoint(node="concat"), OutputPoint())
        assert (points.insertion_node, points.insertion_port) == ("concat", "output")

    def test_unknown_ports_are_rejected(self, numeric_case: TrainedCase) -> None:
        config = numeric_case.pipeline.config
        with pytest.raises(PipelineValidationError, match="no output port"):
            resolve_points(
                config, InsertionPoint(node="source", port="nope"), OutputPoint()
            )
        with pytest.raises(PipelineValidationError, match="no output port"):
            resolve_points(
                config, InsertionPoint(), OutputPoint(node="model", port="nope")
            )

    def test_port_without_node_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="needs InsertionPoint.node"):
            InsertionPoint(port="X")


class TestInsertPerturbationNode:
    def test_reroutes_the_insertion_port_and_keeps_the_original(
        self, numeric_case: TrainedCase
    ) -> None:
        original_hash = numeric_case.pipeline.hash()
        original_edges = [edge.model_dump() for edge in numeric_case.pipeline.edges]
        pipeline = insert_perturbation_node(
            numeric_case.pipeline.config, _points(numeric_case), inject=True
        )
        edges = {(e.source, e.target): e.ports_map for e in pipeline.edges}
        assert edges[("source", PERTURBATION_NODE_NAME)] == [("X", "input")]
        assert edges[(PERTURBATION_NODE_NAME, "scaler")] == [("output", "input")]
        assert ("source", "scaler") not in edges
        # The y pair of the source -> model edge stays in place.
        assert edges[("source", "model")] == [("y", "y")]
        assert ("pred", OUTPUT_SINK_PORT) in edges[("model", "sink")]
        assert numeric_case.pipeline.hash() == original_hash
        assert [
            edge.model_dump() for edge in numeric_case.pipeline.edges
        ] == original_edges

    def test_perturbation_ports_copy_the_source_port(
        self, mixed_case: TrainedCase
    ) -> None:
        pipeline = insert_perturbation_node(
            mixed_case.pipeline.config, _points(mixed_case), inject=False
        )
        config = pipeline.get_node_config(PERTURBATION_NODE_NAME)
        assert config.out_ports["output"].data_category == DataCategoryEnum.MIXED
        assert config.out_ports["output"].data_shape == "batch feature"


class TestBuildCounterfactualPipeline:
    @pytest.mark.parametrize("case_name", ["numeric_case", "mixed_case"])
    def test_injected_rows_give_the_original_predictions(
        self, case_name: str, request: pytest.FixtureRequest
    ) -> None:
        from nodeml.core.pipeline.counterfactuals.celia_adapter import PipelinePredictor

        case: TrainedCase = request.getfixturevalue(case_name)
        pipeline = build_counterfactual_pipeline(
            case.pipeline.config, case.pipeline.get_params(), _points(case), inject=True
        )
        predictor = PipelinePredictor(pipeline, case.X_context)
        rows = case.X.iloc[:7]
        np.testing.assert_allclose(predictor.predict_frame(rows), case.infer(rows))

    def test_injection_does_not_run_the_source(self, numeric_case: TrainedCase) -> None:
        from nodeml.core.pipeline.counterfactuals.celia_adapter import PipelinePredictor

        pipeline = build_counterfactual_pipeline(
            numeric_case.pipeline.config,
            numeric_case.pipeline.get_params(),
            _points(numeric_case),
            inject=True,
        )
        # No input_data at all: an InputsPassthrough source that runs would fail.
        predictor = PipelinePredictor(pipeline, numeric_case.X_context)
        assert predictor.predict_frame(numeric_case.X.iloc[:2]).shape == (2,)

    def test_a_source_that_bypasses_the_insertion_point_is_rejected(self) -> None:
        from nodeml.core.pipeline.runners.smart_runner import (
            SmartRunner,
            SmartRunnerConfig,
        )
        from tests.shims.tabular import numerical_pair

        p = Pipeline()
        p.add_node(
            "left", "InputsPassthrough", source_config(DataCategoryEnum.NUMERICAL)
        )
        p.add_node(
            "right", "InputsPassthrough", source_config(DataCategoryEnum.NUMERICAL)
        )
        p.add_node("concat", "FeatureConcatenate")
        p.add_node("model", "LinearRegression")
        p.add_node("sink", "Sink")
        p.add_edge(Edge(source="left", target="concat", ports_map=[("X", "input_1")]))
        p.add_edge(Edge(source="right", target="concat", ports_map=[("X", "input_2")]))
        p.add_edge(Edge(source="concat", target="model", ports_map=[("output", "X")]))
        p.add_edge(Edge(source="left", target="model", ports_map=[("y", "y")]))
        p.add_edge(Edge(source="model", target="sink", ports_map=[("pred", "pred")]))
        df, ctx = numerical_pair(n_rows=30)
        left = df.rename(columns=lambda c: f"l_{c}")
        right = df.rename(columns=lambda c: f"r_{c}")
        y = df[[df.columns[0]]].rename(columns=lambda _: "t")
        p.compile()
        SmartRunner(p, config=SmartRunnerConfig(verbose=False)).train(
            input_data={
                "left": {
                    "X": (left, ctx.aligned_to(left)),
                    "y": (y, ctx.aligned_to(y)),
                },
                "right": {
                    "X": (right, ctx.aligned_to(right)),
                    "y": (y, ctx.aligned_to(y)),
                },
            }
        )
        points = resolve_points(
            p.config, InsertionPoint(node="left", port="X"), OutputPoint()
        )
        with pytest.raises(PipelineValidationError, match="still needs the sources"):
            build_counterfactual_pipeline(p.config, p.get_params(), points, inject=True)
