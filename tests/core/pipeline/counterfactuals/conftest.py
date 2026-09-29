"""Fixtures for the counterfactual evaluator tests.

Two small regression pipelines, trained on deterministic data:

* ``numeric``: source.X -> StandardScaler -> LinearRegression.
* ``mixed``: source.X -> DataCategoryFilter; the numerical columns go
  through a StandardScaler, the categorical column through a OneHotEncoding;
  FeatureConcatenate joins them for a LinearRegression.

In both, ``source.y`` feeds the model only in training and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from nodeml.components.nodes.data_sources.inputs_passthrough import (
    InputsPassthroughConfig,
)
from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    CategoricalData,
    DataCategoryEnum,
    DataStructureEnum,
    NumericalData,
    TabularDataContext,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.nodes.node import Port
from nodeml.core.pipeline.pipeline import Edge, Pipeline
from nodeml.core.pipeline.runners.smart_runner import SmartRunner, SmartRunnerConfig


@dataclass
class TrainedCase:
    """A trained pipeline and its data."""

    pipeline: Pipeline
    X: pd.DataFrame
    X_context: TabularDataContext
    y: pd.DataFrame
    y_context: TabularDataContext

    @property
    def reference_input(self) -> dict:
        """Runner input with the features only (inference)."""
        return {"source": {"X": (self.X, self.X_context)}}

    def infer(self, X: pd.DataFrame) -> np.ndarray:
        """Return the predictions of the original pipeline for *X*."""
        runner = SmartRunner(self.pipeline, config=SmartRunnerConfig(verbose=False))
        ctx = self.X_context.aligned_to(X)
        out, _ = runner.infer(input_data={"source": {"X": (X, ctx)}})["pred"]
        return out.to_numpy()[:, 0]


def source_config(category: DataCategoryEnum) -> InputsPassthroughConfig:
    """Return a source with an ``X`` port (all modes) and a ``y`` port (training, evaluation)."""
    return InputsPassthroughConfig(
        out_ports={
            "X": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=category,
                data_shape="batch feature",
                desc="Features.",
            ),
            "y": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch 1",
                desc="Target.",
                mode=[NodeExecutionMode.TRAINING, NodeExecutionMode.EVALUATION],
            ),
        }
    )


def _train(pipeline: Pipeline, X, X_ctx, y, y_ctx) -> TrainedCase:
    pipeline.compile()
    SmartRunner(pipeline, config=SmartRunnerConfig(verbose=False)).train(
        input_data={"source": {"X": (X, X_ctx), "y": (y, y_ctx)}}
    )
    return TrainedCase(pipeline, X, X_ctx, y, y_ctx)


def build_numeric_case(n_rows: int = 120) -> TrainedCase:
    """Train ``t = 3a + 0.5b - c`` with a LinearRegression."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n_rows),
            "b": rng.uniform(0, 10, size=n_rows),
            "c": rng.integers(0, 5, size=n_rows).astype(np.int64),
        }
    )
    y = pd.DataFrame({"t": 3 * X["a"] + 0.5 * X["b"] - X["c"]})
    X_ctx = TabularDataContext(list(X.columns), list(X.dtypes), [NumericalData] * 3)
    y_ctx = TabularDataContext(["t"], list(y.dtypes), [NumericalData])
    p = Pipeline()
    p.add_node("source", "InputsPassthrough", source_config(DataCategoryEnum.NUMERICAL))
    p.add_node("scaler", "StandardScaler")
    p.add_node("model", "LinearRegression")
    p.add_node("sink", "Sink")
    p.add_edge(Edge(source="source", target="scaler", ports_map=[("X", "input")]))
    p.add_edge(Edge(source="scaler", target="model", ports_map=[("output", "X")]))
    p.add_edge(Edge(source="source", target="model", ports_map=[("y", "y")]))
    p.add_edge(Edge(source="model", target="sink", ports_map=[("pred", "pred")]))
    return _train(p, X, X_ctx, y, y_ctx)


def build_mixed_case(n_rows: int = 120) -> TrainedCase:
    """Train ``t = 2a + b + offset(color)`` with a categorical column."""
    rng = np.random.default_rng(1)
    colors = np.array(["red", "green", "blue"])
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n_rows),
            "b": rng.uniform(0, 5, size=n_rows),
            "color": colors[rng.integers(0, 3, size=n_rows)],
        }
    )
    offset = X["color"].map({"red": 0.0, "green": 3.0, "blue": 6.0})
    y = pd.DataFrame({"t": 2 * X["a"] + X["b"] + offset})
    X_ctx = TabularDataContext(
        list(X.columns), list(X.dtypes), [NumericalData, NumericalData, CategoricalData]
    )
    y_ctx = TabularDataContext(["t"], list(y.dtypes), [NumericalData])
    p = Pipeline()
    p.add_node("source", "InputsPassthrough", source_config(DataCategoryEnum.MIXED))
    p.add_node("split", "DataCategoryFilter")
    p.add_node("scaler", "StandardScaler")
    p.add_node("onehot", "OneHotEncoding")
    p.add_node("concat", "FeatureConcatenate")
    p.add_node("model", "LinearRegression")
    p.add_node("sink", "Sink")
    p.add_edge(Edge(source="source", target="split", ports_map=[("X", "input")]))
    p.add_edge(
        Edge(source="split", target="scaler", ports_map=[("numerical", "input")])
    )
    p.add_edge(
        Edge(source="split", target="onehot", ports_map=[("categorical", "input")])
    )
    p.add_edge(
        Edge(source="scaler", target="concat", ports_map=[("output", "input_1")])
    )
    p.add_edge(
        Edge(source="onehot", target="concat", ports_map=[("output", "input_2")])
    )
    p.add_edge(Edge(source="concat", target="model", ports_map=[("output", "X")]))
    p.add_edge(Edge(source="source", target="model", ports_map=[("y", "y")]))
    p.add_edge(Edge(source="model", target="sink", ports_map=[("pred", "pred")]))
    return _train(p, X, X_ctx, y, y_ctx)


@pytest.fixture
def numeric_case() -> TrainedCase:
    """Return a trained numeric regression pipeline."""
    return build_numeric_case()


@pytest.fixture
def mixed_case() -> TrainedCase:
    """Return a trained regression pipeline with a categorical column."""
    return build_mixed_case()


def build_imputed_case(n_rows: int = 120) -> TrainedCase:
    """Train the numeric case with a NumericalImputation first, so NaN is accepted."""
    case = build_numeric_case(n_rows)
    p = Pipeline()
    p.add_node("source", "InputsPassthrough", source_config(DataCategoryEnum.NUMERICAL))
    p.add_node("imputer", "NumericalImputation")
    p.add_node("scaler", "StandardScaler")
    p.add_node("model", "LinearRegression")
    p.add_node("sink", "Sink")
    p.add_edge(Edge(source="source", target="imputer", ports_map=[("X", "input")]))
    p.add_edge(Edge(source="imputer", target="scaler", ports_map=[("output", "input")]))
    p.add_edge(Edge(source="scaler", target="model", ports_map=[("output", "X")]))
    p.add_edge(Edge(source="source", target="model", ports_map=[("y", "y")]))
    p.add_edge(Edge(source="model", target="sink", ports_map=[("pred", "pred")]))
    X = case.X.astype({"c": "float64"})
    X.loc[X.index[::7], "b"] = np.nan  # some missing values in the reference rows
    X_ctx = case.X_context.aligned_to(X)
    return _train(p, X, X_ctx, case.y, case.y_context)


@pytest.fixture
def imputed_case() -> TrainedCase:
    """Return a trained pipeline that imputes missing values."""
    return build_imputed_case()
