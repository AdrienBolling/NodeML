"""Tests for :class:`nodeml.core.nodes.metrics.metric_node.MetricNode`.

The abstract :class:`MetricNode` only defines the ``reset`` / ``update`` /
``compute`` contract.  We exercise it through the :class:`SumCountMetric` shim so that
we stay insulated from the concrete torchmetrics-backed implementations.
"""

from __future__ import annotations

import numpy as np
import pytest

from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    NumericalData,
    TabularDataContext,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.nodes.metrics.metric_node import MetricNode
from nodeml.core.nodes.node import Port
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


def _batch(n_rows: int) -> dict[str, tuple[np.ndarray, TabularDataContext]]:
    return {
        "pred": (np.zeros((n_rows, 1)), _ctx(["pred"])),
        "target": (np.zeros((n_rows, 1)), _ctx(["target"])),
    }


class _FailingComputeMetric(SumCountMetric):
    """Metric whose compute always fails after a successful update."""

    def compute(self) -> dict:
        msg = "compute failed"
        raise RuntimeError(msg)


class TestMetricNodeReset:
    def test_each_node_transform_starts_from_a_clean_state(self) -> None:
        metric = SumCountMetric(config=SumCountMetricConfig())

        first, _ = metric.node_transform(_batch(4))["score"]
        second, _ = metric.node_transform(_batch(4))["score"]

        np.testing.assert_array_equal(first, np.array([[4.0]]))
        np.testing.assert_array_equal(second, np.array([[4.0]]))

    def test_node_transform_drops_earlier_manual_updates(self) -> None:
        metric = SumCountMetric(config=SumCountMetricConfig())
        metric.update(_batch(3))

        score, _ = metric.node_transform(_batch(4))["score"]

        np.testing.assert_array_equal(score, np.array([[4.0]]))

    def test_state_is_reset_when_compute_fails(self) -> None:
        metric = _FailingComputeMetric(config=SumCountMetricConfig())

        with pytest.raises(RuntimeError, match="compute failed"):
            metric.node_transform(_batch(4))

        assert metric._count == 0

    def test_reset_clears_the_accumulated_state(self) -> None:
        metric = SumCountMetric(config=SumCountMetricConfig())
        metric.update(_batch(3))

        metric.reset()
        metric.update(_batch(2))

        score, _ = metric.compute()["score"]
        np.testing.assert_array_equal(score, np.array([[2.0]]))

    def test_default_reset_does_nothing(self) -> None:
        metric = _SuperInitMetric(config=SumCountMetricConfig())

        metric.reset()

        assert metric.node_transform(_batch(2)) == {}


class _SuperInitMetric(MetricNode):
    """Metric that uses the base constructor, as other node types do."""

    def __init__(self, *, config: SumCountMetricConfig) -> None:
        super().__init__(config=config)

    def update(self, data: dict) -> None:
        pass

    def compute(self) -> dict:
        return {}


def _port(**kwargs: object) -> Port:
    return Port(
        arr_type=ArrayLikeEnum.NUMPY,
        data_structure=DataStructureEnum.TABULAR,
        data_category=DataCategoryEnum.NUMERICAL,
        data_shape="batch 1",
        desc="A port.",
        **kwargs,
    )


class TestMetricNodeConfigPortModes:
    def test_default_ports_are_evaluation_only(self) -> None:
        config = SumCountMetricConfig()

        for port in [*config.in_ports.values(), *config.out_ports.values()]:
            assert port.mode == [NodeExecutionMode.EVALUATION]

    def test_ports_without_a_mode_become_evaluation_only(self) -> None:
        config = SumCountMetricConfig(in_ports={"pred": _port(), "target": _port()})

        assert config.in_ports["pred"].mode == [NodeExecutionMode.EVALUATION]

    def test_explicit_all_mode_is_kept(self) -> None:
        port = _port(mode=[NodeExecutionMode.ALL])
        config = SumCountMetricConfig(in_ports={"pred": port, "target": _port()})

        assert config.in_ports["pred"].mode == [NodeExecutionMode.ALL]
        assert config.in_ports["target"].mode == [NodeExecutionMode.EVALUATION]

    def test_caller_port_is_not_changed(self) -> None:
        shared = _port()
        SumCountMetricConfig(in_ports={"pred": shared, "target": shared})

        # The same Port object can also belong to another node config.
        assert shared.mode == [NodeExecutionMode.ALL]

    def test_json_round_trip_keeps_the_modes(self) -> None:
        port = _port(mode=[NodeExecutionMode.ALL])
        config = SumCountMetricConfig(in_ports={"pred": port, "target": _port()})

        loaded = SumCountMetricConfig.model_validate_json(config.model_dump_json())

        assert loaded.in_ports["pred"].mode == [NodeExecutionMode.ALL]
        assert loaded.in_ports["target"].mode == [NodeExecutionMode.EVALUATION]
        assert loaded.out_ports["score"].mode == [NodeExecutionMode.EVALUATION]


class TestMetricNodeInit:
    def test_base_init_takes_only_the_config(self) -> None:
        config = SumCountMetricConfig()
        metric = _SuperInitMetric(config=config)

        assert metric.config is config
        assert metric.running_config is config.running_config
