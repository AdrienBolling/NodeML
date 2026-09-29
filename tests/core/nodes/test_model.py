"""Tests for :class:`nodeml.core.nodes.models.model.Model`."""

from __future__ import annotations

import numpy as np

from nodeml.core.common.data.data import NumericalData, TabularDataContext
from tests.shims.nodes import MeanModel, MeanModelConfig


def _numeric_ctx(cols: list[str]) -> TabularDataContext:
    return TabularDataContext(
        columns=cols,
        dtypes=[np.dtype("float64") for _ in cols],
        categories=[NumericalData for _ in cols],
    )


class TestModelLifecycle:
    """End-to-end fit/predict on the ``MeanModel`` shim."""

    def test_fit_learns_training_mean(self) -> None:
        model = MeanModel(config=MeanModelConfig())
        X = np.arange(6, dtype=np.float64).reshape(3, 2)
        y = np.array([[1.0], [2.0], [3.0]])
        data = {
            "X": (X, _numeric_ctx(["a", "b"])),
            "y": (y, _numeric_ctx(["target"])),
        }

        model.node_fit(data)
        out = model.node_transform(data)
        pred, _ = out["pred"]

        np.testing.assert_allclose(pred, np.full((3, 1), 2.0))

    def test_params_round_trip(self) -> None:
        model = MeanModel(config=MeanModelConfig())
        X = np.zeros((5, 1))
        y = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        data = {
            "X": (X, _numeric_ctx(["x"])),
            "y": (y, _numeric_ctx(["target"])),
        }
        model.node_fit(data)

        restored = MeanModel(config=MeanModelConfig())
        restored.set_params(model.get_params())

        assert restored.get_params() == model.get_params()
