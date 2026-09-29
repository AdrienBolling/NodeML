"""Tests for :class:`nodeml.components.nodes.metrics.regression.mse.MSE`."""

from __future__ import annotations

import numpy as np

from nodeml.components.nodes.metrics.regression.mse import (
    MSE,
    MSEConfig,
    MSERunningConfig,
)


class TestMSE:
    def test_zero_error_on_identical_inputs(self) -> None:
        node = MSE(config=MSEConfig())
        data = {
            "pred": (np.ones((8, 1), dtype=np.float64), None),
            "target": (np.ones((8, 1), dtype=np.float64), None),
        }
        node.update(data)
        out = node.compute()
        score, ctx = out["score"]

        assert score.shape == (1, 1)
        np.testing.assert_allclose(score, np.array([[0.0]]))
        assert ctx.columns == ["mse"]

    def test_mse_value_matches_manual_calculation(self) -> None:
        node = MSE(config=MSEConfig())
        pred = np.array([[1.0], [2.0], [3.0]])
        target = np.array([[1.5], [2.5], [3.5]])
        # Each error is 0.5, so the mean squared error is 0.25.
        node.update({"pred": (pred, None), "target": (target, None)})
        score, _ = node.compute()["score"]
        np.testing.assert_allclose(score, np.array([[0.25]]))

    def test_rmse_mode_labels_column_as_rmse(self) -> None:
        node = MSE(
            config=MSEConfig(
                running_config=MSERunningConfig(squared=False),
            )
        )
        data = {
            "pred": (np.zeros((4, 1)), None),
            "target": (np.ones((4, 1)), None),
        }
        node.update(data)
        score, ctx = node.compute()["score"]
        np.testing.assert_allclose(score, np.array([[1.0]]))
        assert ctx.columns == ["rmse"]
