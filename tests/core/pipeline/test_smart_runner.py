"""Tests for :class:`nodeml.core.pipeline.runners.smart_runner.SmartRunner`.

These tests run an end-to-end pipeline
(``InputsPassthrough`` → ``LinearRegression`` → ``Sink`` + optional ``MSE``
metric) and verify that ``train`` / ``evaluate`` / ``infer`` behave as
documented.  Small deterministic datasets keep the tests fast.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from nodeml.core.pipeline.pipeline import Pipeline
from nodeml.core.pipeline.runners.smart_runner import SmartRunner
from tests.shims.pipelines import build_source_model_sink_pipeline

# ---------------------------------------------------------------------------
# Pre-compilation guard
# ---------------------------------------------------------------------------


class TestSmartRunnerPrecompilation:
    def test_runner_requires_compiled_pipeline(self) -> None:
        # A brand-new, non-compiled pipeline must be rejected.
        with pytest.raises(ValueError, match="compiled"):
            SmartRunner(Pipeline())


# ---------------------------------------------------------------------------
# Full lifecycle with a simple regression pipeline
# ---------------------------------------------------------------------------


class TestSmartRunnerLifecycle:
    """End-to-end training → evaluation → inference over LinearRegression."""

    def test_train_and_infer_recover_linear_targets(self, regression_dataset) -> None:
        (X_pair, y_pair, coefs) = regression_dataset

        pipeline = build_source_model_sink_pipeline(
            model_node_type="LinearRegression",
            with_metric=False,
            name="smart_runner_train_infer",
        )
        runner = SmartRunner(pipeline)

        # Train with y — infer with the same X.
        runner.train(input_data={"source": {"X": X_pair, "y": y_pair}})
        preds = runner.infer(input_data={"source": {"X": X_pair, "y": y_pair}})

        assert "pred" in preds
        pred_df, _ = preds["pred"]
        # Noise scale in the shim is 0.01 → predictions should be close.
        np.testing.assert_allclose(
            pred_df.to_numpy().flatten(),
            y_pair[0].to_numpy().flatten(),
            atol=0.1,
        )
        # True coefficients should be roughly recovered.
        fitted = pipeline.node_objects["model"].get_params()["fitted_params"]
        np.testing.assert_allclose(fitted["coef_"].flatten(), coefs, atol=0.1)

    def test_evaluate_returns_low_mse(self, regression_dataset) -> None:
        (X_pair, y_pair, _coefs) = regression_dataset

        pipeline = build_source_model_sink_pipeline(
            model_node_type="LinearRegression",
            with_metric=True,
            name="smart_runner_evaluate",
        )
        runner = SmartRunner(pipeline)

        runner.train(input_data={"source": {"X": X_pair, "y": y_pair}})
        metrics = runner.evaluate(input_data={"source": {"X": X_pair, "y": y_pair}})

        assert "metric" in metrics
        score_df, _ = metrics["metric"]
        score = float(score_df.to_numpy().reshape(-1)[0])
        # With noise_scale=0.01, MSE should be ≈ 1e-4.
        assert score < 1e-2


class TestSmartRunnerCaches:
    def test_caches_are_cleared_between_phases(self, regression_dataset) -> None:
        (X_pair, y_pair, _coefs) = regression_dataset
        pipeline = build_source_model_sink_pipeline(with_metric=True)
        runner = SmartRunner(pipeline)

        runner.train(input_data={"source": {"X": X_pair, "y": y_pair}})
        # Node outputs from the train phase live on the runner until the next
        # phase starts. We check the reset happens at the start of ``evaluate``.
        assert runner._node_outputs  # populated by train
        runner.evaluate(input_data={"source": {"X": X_pair, "y": y_pair}})
        # After evaluate, only the metric outputs survive on
        # ``_metric_node_outputs``; other node outputs were re-computed.
        assert "metric" in runner._metric_node_outputs


# ---------------------------------------------------------------------------
# Runner checks: external inputs, shapes across ports, context drift
# ---------------------------------------------------------------------------


def _register(name: str, node_class: type, config_class: type) -> None:
    from nodeml.core.nodes.registry.node_registry import NODE_REGISTRY

    if name in NODE_REGISTRY:
        NODE_REGISTRY.unregister(name)
    NODE_REGISTRY.register(
        name=name, node_class=node_class, node_config_class=config_class
    )


@pytest.fixture
def shim_nodes() -> Iterator[None]:
    """Register the test shims in the global registry for one test."""
    from nodeml.core.nodes.registry.node_registry import NODE_REGISTRY
    from tests.shims.nodes import (
        ConstantSource,
        ConstantSourceConfig,
        IdentityTransform,
        IdentityTransformConfig,
    )

    class ReorderingTransform(IdentityTransform):
        """Reverse the columns but keep the input context (a node bug)."""

        def transform(self, data: dict) -> dict:
            df, ctx = data["input"]
            return {"output": (df[list(reversed(df.columns))], ctx)}

    class ExtraPortTransform(IdentityTransform):
        """Return a port that the config does not declare (a node bug)."""

        def transform(self, data: dict) -> dict:
            return {"output": data["input"], "extra": data["input"]}

    names = {
        "ShimConstantSource": (ConstantSource, ConstantSourceConfig),
        "ShimReordering": (ReorderingTransform, IdentityTransformConfig),
        "ShimExtraPort": (ExtraPortTransform, IdentityTransformConfig),
    }
    for name, (node_class, config_class) in names.items():
        _register(name, node_class, config_class)
    yield
    for name in names:
        NODE_REGISTRY.unregister(name)


def _source_transform_sink(transform_type: str) -> Pipeline:
    from nodeml.core.pipeline.pipeline import Edge
    from tests.shims.tabular import numerical_pair

    p = Pipeline()
    p.add_node("source", "ShimConstantSource")
    p.add_node("transform", transform_type)
    p.add_node("sink", "Sink")
    p.add_edge(
        Edge(source="source", target="transform", ports_map=[("output", "input")])
    )
    p.add_edge(Edge(source="transform", target="sink", ports_map=[("output", "out")]))
    p.compile()
    p.node_objects["source"].set_payload(numerical_pair())
    return p


class TestRunnerChecks:
    def test_unknown_input_data_node_is_rejected(self, regression_dataset) -> None:
        from nodeml.core.common.exceptions import NodeInputError

        X_pair, y_pair, _ = regression_dataset
        runner = SmartRunner(build_source_model_sink_pipeline())
        with pytest.raises(NodeInputError, match="not nodes of the pipeline"):
            runner.train(input_data={"typo": {"X": X_pair, "y": y_pair}})

    def test_input_data_for_a_node_without_inputs_is_rejected(
        self, regression_dataset
    ) -> None:
        from nodeml.core.common.exceptions import NodeInputError

        X_pair, _, _ = regression_dataset
        runner = SmartRunner(build_source_model_sink_pipeline())
        with pytest.raises(NodeInputError, match="do not accept external inputs"):
            runner.infer(input_data={"model": {"X": X_pair}})

    def test_x_and_y_with_different_row_counts_are_rejected(
        self, regression_dataset
    ) -> None:
        from nodeml.core.common.exceptions import DataTypeError

        X_pair, (y_df, y_ctx), _ = regression_dataset
        runner = SmartRunner(build_source_model_sink_pipeline())
        with pytest.raises(DataTypeError, match="same size on every"):
            runner.train(
                input_data={"source": {"X": X_pair, "y": (y_df.iloc[:-1], y_ctx)}}
            )

    def test_context_drift_names_the_node(self, shim_nodes) -> None:
        from nodeml.core.common.exceptions import DataContextError

        runner = SmartRunner(_source_transform_sink("ShimReordering"))
        with pytest.raises(DataContextError, match="node 'transform'"):
            runner.train()

    def test_undeclared_output_port_is_rejected(self, shim_nodes) -> None:
        from nodeml.core.common.exceptions import NodeError

        runner = SmartRunner(_source_transform_sink("ShimExtraPort"))
        with pytest.raises(NodeError, match="does not declare"):
            runner.train()

    def test_sources_set_up_once_per_runner_call(self, shim_nodes) -> None:
        from nodeml.core.pipeline.pipeline import Edge
        from tests.shims.tabular import numerical_pair

        p = Pipeline()
        p.add_node("source", "ShimConstantSource")
        p.add_node("sink", "Sink")
        p.add_edge(Edge(source="source", target="sink", ports_map=[("output", "out")]))
        p.compile()
        source = p.node_objects["source"]
        source.set_payload(numerical_pair())
        runner = SmartRunner(p)
        runner.train()
        assert source.setup_calls == 1
        runner.infer()
        assert source.setup_calls == 2
        runner.infer()
        assert source.setup_calls == 3
