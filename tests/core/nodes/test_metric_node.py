"""Tests for :class:`nodeml.core.nodes.metrics.metric_node.MetricNode`.

The abstract :class:`MetricNode` only defines the ``update`` / ``compute``
contract.  We exercise it through the :class:`SumCountMetric` shim so that
we stay insulated from the concrete torchmetrics-backed implementations.
"""

from __future__ import annotations

import numpy as np

from nodeml.core.common.data.data import NumericalData, TabularDataContext
from tests.shims.nodes import SumCountMetric, SumCountMetricConfig


def _ctx(cols: list[str]) -> TabularDataContext:
    return TabularDataContext(
        columns=cols,
        dtypes=[np.dtype("float64") for _ in cols],
        categories=[NumericalData for _ in cols],
    )


class TestMetricNodeFlow:
    def test_node_transform_runs_update_then_compute(self) -> None:
        metric = SumCountMetric(config=SumCountMetricConfig())
        data = {
            "pred": (np.zeros((4, 1)), _ctx(["pred"])),
            "target": (np.zeros((4, 1)), _ctx(["target"])),
        }
        out = metric.node_transform(data)

        score, _ = out["score"]
        np.testing.assert_array_equal(score, np.array([[4.0]]))

    def test_repeated_updates_accumulate(self) -> None:
        metric = SumCountMetric(config=SumCountMetricConfig())
        batch_a = {
            "pred": (np.zeros((3, 1)), _ctx(["pred"])),
            "target": (np.zeros((3, 1)), _ctx(["target"])),
        }
        batch_b = {
            "pred": (np.zeros((5, 1)), _ctx(["pred"])),
            "target": (np.zeros((5, 1)), _ctx(["target"])),
        }

        metric.update(batch_a)
        metric.update(batch_b)
        score, _ = metric.compute()["score"]
        np.testing.assert_array_equal(score, np.array([[8.0]]))
