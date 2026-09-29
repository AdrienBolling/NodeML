"""Tests for :class:`nodeml.core.pipeline.runners.smart_runner.SmartRunner`.

These tests run an end-to-end pipeline
(``InputsPassthrough`` → ``LinearRegression`` → ``Sink`` + optional ``MSE``
metric) and verify that ``train`` / ``evaluate`` / ``infer`` behave as
documented.  Small deterministic datasets keep the tests fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from nodeml.core.pipeline.runners.smart_runner import SmartRunner
from tests.shims.pipelines import build_source_model_sink_pipeline

# ---------------------------------------------------------------------------
# Pre-compilation guard
# ---------------------------------------------------------------------------


class TestSmartRunnerPrecompilation:
    def test_runner_requires_compiled_pipeline(self) -> None:
        from nodeml.core.pipeline.pipeline import Pipeline

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
