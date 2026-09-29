"""Check that every metric node returns exactly what its ports declare.

The SmartRunner rejects undeclared output ports and contexts that do not
describe their data.  These tests check the nine built-in metric nodes
directly and through a runner.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from nodeml.components.nodes.data_sources.inputs_passthrough import (
    InputsPassthroughConfig,
)
from nodeml.core.common.data.data import (
    DATA_CATEGORY_MAPPING,
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    NumericalData,
    TabularData,
    TabularDataContext,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.nodes.node import Port
from nodeml.core.nodes.registry.node_registry import NODE_REGISTRY
from nodeml.core.pipeline.pipeline import Edge, Pipeline, PipelineConfig
from nodeml.core.pipeline.runners.smart_runner import SmartRunner

N_ROWS = 24

# Registered name -> node class name.  Both names are public API.
METRIC_CLASSES = {
    "Accuracy": "AccuracyNode",
    "AUROC": "AUROCNode",
    "F1Score": "F1",
    "Precision": "PrecisionNode",
    "Recall": "RecallNode",
    "MSE": "MSE",
    "MAE": "MAE",
    "MAPE": "MAPE",
    "R2Score": "R2",
}
# Classification metric name -> score column.
CLASSIFICATION = {
    "Accuracy": "accuracy",
    "AUROC": "auroc",
    "F1Score": "f1",
    "Precision": "precision",
    "Recall": "recall",
}


def _node(name: str, **running: Any) -> Any:
    config_class = NODE_REGISTRY.get_node_config_class(name)
    running_class = NODE_REGISTRY.get(name)["running_config"]
    config = config_class(running_config=running_class(**running))
    return NODE_REGISTRY.get_node_class(name)(config=config)


def _binary() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    positive = rng.random((N_ROWS, 1))
    target = (positive + rng.normal(scale=0.3, size=(N_ROWS, 1)) > 0.5).astype(int)
    return np.hstack([1.0 - positive, positive]), target


def _multiclass() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    proba = rng.random((N_ROWS, 3))
    proba /= proba.sum(axis=1, keepdims=True)
    return proba, rng.integers(0, 3, size=(N_ROWS, 1))


def _regression(n_outputs: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    target = rng.normal(loc=5.0, size=(N_ROWS, n_outputs))
    return target + rng.normal(scale=0.2, size=target.shape), target


# (registered name, running options, (pred, target), expected columns).
CASES: list[tuple[str, dict, tuple[np.ndarray, np.ndarray], list[str]]] = [
    *[(name, {}, _binary(), [col]) for name, col in CLASSIFICATION.items()],
    *[
        (name, {"task": "multiclass", "num_classes": 3}, _multiclass(), [col])
        for name, col in CLASSIFICATION.items()
    ],
    ("MSE", {}, _regression(1), ["mse"]),
    ("MSE", {"squared": False}, _regression(1), ["rmse"]),
    ("MSE", {"num_outputs": 2}, _regression(2), ["mse_0", "mse_1"]),
    ("MAE", {}, _regression(1), ["mae"]),
    ("MAE", {"num_outputs": 2}, _regression(2), ["mae_0", "mae_1"]),
    ("MAPE", {}, _regression(1), ["mape"]),
    ("MAPE", {}, _regression(2), ["mape"]),
    ("R2Score", {}, _regression(1), ["r2"]),
    ("R2Score", {}, _regression(2), ["r2"]),
    ("R2Score", {"multioutput": "raw_values"}, _regression(2), ["r2_0", "r2_1"]),
]


def test_registered_names_and_classes_do_not_change() -> None:
    for name, class_name in METRIC_CLASSES.items():
        assert NODE_REGISTRY.get_node_class(name).__name__ == class_name


@pytest.mark.parametrize("name", list(METRIC_CLASSES))
def test_ports_are_evaluation_only(name: str) -> None:
    config = NODE_REGISTRY.get_node_config_class(name)()

    assert set(config.in_ports) == {"pred", "target"}
    assert set(config.out_ports) == {"score"}
    for port in [*config.in_ports.values(), *config.out_ports.values()]:
        assert port.mode == [NodeExecutionMode.EVALUATION]


@pytest.mark.parametrize(("name", "running", "data", "columns"), CASES)
def test_output_matches_the_declared_port(
    name: str,
    running: dict,
    data: tuple[np.ndarray, np.ndarray],
    columns: list[str],
) -> None:
    node = _node(name, **running)
    pred, target = data

    out = node.node_transform({"pred": (pred, None), "target": (target, None)})

    assert set(out) == set(node.out_ports)
    array, ctx = out["score"]
    port = node.out_ports["score"]
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float64
    assert array.shape == (1, len(columns))
    assert isinstance(ctx, TabularDataContext)
    assert ctx.columns == columns
    assert ctx.dtypes == [np.dtype("float64")] * len(columns)
    assert ctx.categories == [NumericalData] * len(columns)
    # The data must satisfy the category and the shape of the port.
    tabular = TabularData(
        data=array, columns=ctx.columns, dtypes=ctx.dtypes, categories=ctx.categories
    )
    expected = DATA_CATEGORY_MAPPING[port.data_category][TabularData, port.data_shape]
    assert isinstance(tabular, expected)


def _numerical_pair(array: np.ndarray, prefix: str) -> tuple[pd.DataFrame, Any]:
    df = pd.DataFrame(array, columns=[f"{prefix}{i}" for i in range(array.shape[1])])
    ctx = TabularDataContext(
        columns=list(df.columns),
        dtypes=list(df.dtypes),
        categories=[NumericalData] * len(df.columns),
    )
    return df, ctx


def _metrics_pipeline(metrics: dict[str, tuple[str, dict]]) -> Pipeline:
    """Build a pipeline: InputsPassthrough (pred, y) -> metrics, plus a Sink."""

    def port(desc: str) -> Port:
        return Port(
            arr_type=ArrayLikeEnum.PANDAS,
            data_structure=DataStructureEnum.TABULAR,
            data_category=DataCategoryEnum.NUMERICAL,
            data_shape="batch targets",
            desc=desc,
        )

    source = InputsPassthroughConfig(
        out_ports={"pred": port("Predictions."), "y": port("Targets.")}
    )
    nodes: dict[str, tuple[str, Any]] = {
        "source": ("InputsPassthrough", source),
        "sink": ("Sink", NODE_REGISTRY.get_node_config_class("Sink")()),
    }
    edges = [Edge(source="source", target="sink", ports_map=[("pred", "pred")])]
    for node_name, (metric, running) in metrics.items():
        config_class = NODE_REGISTRY.get_node_config_class(metric)
        running_class = NODE_REGISTRY.get(metric)["running_config"]
        nodes[node_name] = (
            metric,
            config_class(running_config=running_class(**running)),
        )
        edges.append(
            Edge(
                source="source",
                target=node_name,
                ports_map=[("pred", "pred"), ("y", "target")],
            )
        )
    pipeline = Pipeline(config=PipelineConfig(name="metrics", nodes=nodes, edges=edges))
    pipeline.compile()
    return pipeline


def test_multi_output_regression_through_a_runner() -> None:
    pred, target = _regression(2)
    metrics = {
        "mse": ("MSE", {"num_outputs": 2}),
        "rmse": ("MSE", {"num_outputs": 2, "squared": False}),
        "mae": ("MAE", {"num_outputs": 2}),
        "mape": ("MAPE", {}),
        "r2": ("R2Score", {"multioutput": "raw_values"}),
        "r2_mean": ("R2Score", {}),
    }
    runner = SmartRunner(_metrics_pipeline(metrics))
    inputs = {
        "source": {
            "pred": _numerical_pair(pred, "p"),
            "y": _numerical_pair(target, "t"),
        }
    }

    first = runner.evaluate(input_data=inputs)
    second = runner.evaluate(input_data=inputs)

    expected_columns = {
        "mse": ["mse_0", "mse_1"],
        "rmse": ["rmse_0", "rmse_1"],
        "mae": ["mae_0", "mae_1"],
        "mape": ["mape"],
        "r2": ["r2_0", "r2_1"],
        "r2_mean": ["r2"],
    }
    assert set(first) == set(metrics)
    for node_name, columns in expected_columns.items():
        df, ctx = first[node_name]
        assert list(df.columns) == columns
        assert ctx.columns == columns
        assert (df.dtypes == np.float64).all()
        pd.testing.assert_frame_equal(df, second[node_name][0])
