"""Tests for the regression metric nodes (MSE, MAE, MAPE, R2Score)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from sklearn import metrics as skm

from nodeml.components.nodes.metrics.regression.mse import MSE, MSEConfig
from nodeml.components.nodes.metrics.regression.r2_score import R2, R2ScoreConfig
from nodeml.core.nodes.registry.node_registry import NODE_REGISTRY
from nodeml.core.pipeline.runners.smart_runner import SmartRunner
from tests.shims.pipelines import build_source_model_sink_pipeline
from tests.shims.tabular import linear_regression_dataset


def _data(pred: np.ndarray, target: np.ndarray) -> dict:
    return {"pred": (pred, None), "target": (target, None)}


def _score(node: object, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    score, _ = node.node_transform(_data(pred, target))["score"]
    return score


def _node(name: str, **running: Any) -> Any:
    """Build a registered metric node with the given running options."""
    config_class = NODE_REGISTRY.get_node_config_class(name)
    running_class = NODE_REGISTRY.get(name)["running_config"]
    config = config_class(running_config=running_class(**running))
    return NODE_REGISTRY.get_node_class(name)(config=config)


def _adjusted_r2(target: np.ndarray, pred: np.ndarray, n_features: int) -> float:
    n_rows = len(target)
    r2 = skm.r2_score(target, pred)
    return 1 - (1 - r2) * (n_rows - 1) / (n_rows - n_features - 1)


@pytest.fixture
def regression_pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    target = rng.normal(size=(20, 1))
    pred = target + rng.normal(scale=0.5, size=(20, 1))
    return pred, target


@pytest.fixture
def large_values_pair() -> tuple[np.ndarray, np.ndarray]:
    """Targets near 1e4 with errors near 1e-2: float32 loses the errors.

    The errors stay above 1e-4 in total: torchmetrics R2Score gives 1.0
    when the residual sum of squares is below 1e-4.
    """
    rng = np.random.default_rng(3)
    target = 1e4 + rng.normal(size=(50, 1))
    pred = target + rng.normal(scale=1e-2, size=(50, 1))
    return pred, target


# (registered name, running options, sklearn function on (y_true, y_pred)).
SINGLE_OUTPUT_CASES: list[tuple[str, dict, Callable[..., float]]] = [
    ("MSE", {}, skm.mean_squared_error),
    ("MSE", {"squared": False}, skm.root_mean_squared_error),
    ("MAE", {}, skm.mean_absolute_error),
    ("MAPE", {}, skm.mean_absolute_percentage_error),
    ("R2Score", {}, skm.r2_score),
    ("R2Score", {"adjusted": 3}, lambda t, p: _adjusted_r2(t, p, 3)),
]


class TestValues:
    @pytest.mark.parametrize(("name", "running", "reference"), SINGLE_OUTPUT_CASES)
    def test_single_output_matches_sklearn(
        self, name: str, running: dict, reference: Callable, regression_pair
    ) -> None:
        pred, target = regression_pair

        score = _score(_node(name, **running), pred, target)

        np.testing.assert_allclose(score, [[reference(target, pred)]], rtol=1e-12)

    @pytest.mark.parametrize(("name", "running", "reference"), SINGLE_OUTPUT_CASES)
    def test_large_values_keep_float64_precision(
        self, name: str, running: dict, reference: Callable, large_values_pair
    ) -> None:
        pred, target = large_values_pair

        score = _score(_node(name, **running), pred, target)

        np.testing.assert_allclose(score, [[reference(target, pred)]], rtol=1e-9)


class TestMetricState:
    def test_state_is_reset_after_a_compute_error(self, regression_pair) -> None:
        pred, target = regression_pair
        node = R2(config=R2ScoreConfig())
        # R2 needs two samples: update succeeds, compute fails.
        with pytest.raises(ValueError, match="two samples"):
            node.node_transform(_data(pred[:1], target[:1]))

        # A batched evaluation after the error sees only its own data.
        node.update(_data(pred, target))
        score, _ = node.compute()["score"]

        expected = _score(R2(config=R2ScoreConfig()), pred, target)
        np.testing.assert_array_equal(score, expected)

    def test_compute_does_not_clear_the_state(self, regression_pair) -> None:
        pred, target = regression_pair
        node = MSE(config=MSEConfig())
        node.update(_data(pred[:10], target[:10]))
        node.update(_data(pred[10:], target[10:]))

        first, _ = node.compute()["score"]
        second, _ = node.compute()["score"]

        np.testing.assert_array_equal(first, second)
        expected = _score(MSE(config=MSEConfig()), pred, target)
        np.testing.assert_allclose(first, expected, rtol=1e-6)

    def test_reset_clears_the_state(self, regression_pair) -> None:
        pred, target = regression_pair
        node = MSE(config=MSEConfig())
        node.update(_data(pred[:10], target[:10]))

        node.reset()
        node.update(_data(pred[10:], target[10:]))
        score, _ = node.compute()["score"]

        expected = _score(MSE(config=MSEConfig()), pred[10:], target[10:])
        np.testing.assert_array_equal(score, expected)


class TestRunnerEvaluate:
    def test_evaluate_after_a_failed_evaluate_is_not_accumulated(self) -> None:
        X_pair, y_pair, _ = linear_regression_dataset(noise_scale=0.5)
        (X_df, X_ctx), (y_df, y_ctx) = X_pair, y_pair
        pipeline = build_source_model_sink_pipeline(
            with_metric=True, metric_node_type="R2Score"
        )
        runner = SmartRunner(pipeline)
        runner.train(input_data={"source": {"X": X_pair, "y": y_pair}})
        first = runner.evaluate(input_data={"source": {"X": X_pair, "y": y_pair}})

        # One row: R2 fails in compute().
        one_row = {"X": (X_df.iloc[:1], X_ctx), "y": (y_df.iloc[:1], y_ctx)}
        with pytest.raises(ValueError, match="two samples"):
            runner.evaluate(input_data={"source": one_row})
        second = runner.evaluate(input_data={"source": {"X": X_pair, "y": y_pair}})

        first_df, _ = first["metric"]
        second_df, _ = second["metric"]
        np.testing.assert_array_equal(first_df.to_numpy(), second_df.to_numpy())
        assert list(second_df.columns) == ["r2"]
