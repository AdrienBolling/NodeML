"""Tests for the classification metric nodes.

The nodes are Accuracy, AUROC, F1Score, Precision and Recall.  The values
are checked against ``sklearn.metrics`` on small seeded datasets.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import Field
from sklearn import metrics as skm

from nodeml.components.nodes.data_sources.inputs_passthrough import (
    InputsPassthroughConfig,
)
from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    NumericalData,
    TabularDataContext,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.common.exceptions import NodeConfigError, NodeInputError
from nodeml.core.nodes.models.model import (
    Model,
    ModelConfig,
    ModelHyperParameters,
    ModelRunningConfig,
)
from nodeml.core.nodes.node import Port
from nodeml.core.nodes.registry.node_registry import NODE_REGISTRY
from nodeml.core.pipeline.pipeline import Edge, Pipeline, PipelineConfig
from nodeml.core.pipeline.runners.smart_runner import SmartRunner

N_ROWS = 40
N_CLASSES = 3

# Registered name -> sklearn function on (y_true, y_pred) for label metrics.
LABEL_METRICS: dict[str, Callable[..., float]] = {
    "Accuracy": skm.accuracy_score,
    "F1Score": skm.f1_score,
    "Precision": skm.precision_score,
    "Recall": skm.recall_score,
}
ALL_METRICS = [*LABEL_METRICS, "AUROC"]


def _node(name: str, **running: Any) -> Any:
    """Build a registered metric node with the given running options."""
    config_class = NODE_REGISTRY.get_node_config_class(name)
    running_class = NODE_REGISTRY.get(name)["running_config"]
    config = config_class(running_config=running_class(**running))
    return NODE_REGISTRY.get_node_class(name)(config=config)


def _score(node: Any, pred: np.ndarray, target: np.ndarray) -> float:
    out = node.node_transform({"pred": (pred, None), "target": (target, None)})
    score, _ = out["score"]
    assert score.shape == (1, 1)
    return float(score[0, 0])


@pytest.fixture
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    """Positive-class probabilities ``(n, 1)`` and labels ``(n, 1)``."""
    rng = np.random.default_rng(0)
    target = rng.integers(0, 2, size=(N_ROWS, 1))
    noise = rng.normal(scale=0.35, size=(N_ROWS, 1))
    positive = np.clip(0.5 + (target - 0.5) * 0.6 + noise, 0.01, 0.99)
    return positive, target


@pytest.fixture
def multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    """Class probabilities ``(n, 3)`` and labels ``(n, 1)``."""
    rng = np.random.default_rng(1)
    target = rng.integers(0, N_CLASSES, size=(N_ROWS, 1))
    logits = rng.normal(size=(N_ROWS, N_CLASSES))
    logits[np.arange(N_ROWS), target[:, 0]] += 1.0
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return proba, target


def _sklearn_binary(name: str, positive: np.ndarray, target: np.ndarray) -> float:
    if name == "AUROC":
        return skm.roc_auc_score(target[:, 0], positive[:, 0])
    labels = (positive[:, 0] > 0.5).astype(int)
    return LABEL_METRICS[name](target[:, 0], labels)


def _sklearn_multiclass(name: str, proba: np.ndarray, target: np.ndarray) -> float:
    if name == "AUROC":
        return skm.roc_auc_score(target[:, 0], proba, multi_class="ovr")
    labels = proba.argmax(axis=1)
    if name == "Accuracy":
        return skm.accuracy_score(target[:, 0], labels)
    return LABEL_METRICS[name](target[:, 0], labels, average="macro")


# ---------------------------------------------------------------------------
# Binary task
# ---------------------------------------------------------------------------


class TestBinary:
    @pytest.mark.parametrize("name", ALL_METRICS)
    def test_two_column_probabilities_use_the_positive_column(
        self, name: str, binary_data
    ) -> None:
        positive, target = binary_data
        proba = np.hstack([1.0 - positive, positive])

        score = _score(_node(name), proba, target)

        assert score == pytest.approx(_sklearn_binary(name, positive, target))

    @pytest.mark.parametrize("name", ALL_METRICS)
    def test_one_column_probabilities(self, name: str, binary_data) -> None:
        positive, target = binary_data

        score = _score(_node(name), positive, target)

        assert score == pytest.approx(_sklearn_binary(name, positive, target))

    @pytest.mark.parametrize("name", list(LABEL_METRICS))
    @pytest.mark.parametrize("dtype", [np.int64, np.float64])
    def test_hard_labels(self, name: str, dtype: type, binary_data) -> None:
        positive, target = binary_data
        labels = (positive > 0.5).astype(dtype)

        score = _score(_node(name), labels, target)

        assert score == pytest.approx(LABEL_METRICS[name](target, labels))

    def test_integer_labels_ignore_the_threshold(self, binary_data) -> None:
        positive, target = binary_data
        labels = (positive > 0.5).astype(np.int64)

        score = _score(_node("Accuracy", threshold=1.0), labels, target)

        assert score == pytest.approx(skm.accuracy_score(target, labels))

    @pytest.mark.parametrize("name", ALL_METRICS)
    def test_more_than_two_columns_are_rejected(
        self, name: str, multiclass_data
    ) -> None:
        proba, target = multiclass_data

        with pytest.raises(NodeInputError, match="task 'binary'"):
            _score(_node(name), proba, target)


# ---------------------------------------------------------------------------
# Multiclass task
# ---------------------------------------------------------------------------


class TestMulticlass:
    @pytest.mark.parametrize("name", ALL_METRICS)
    def test_probabilities(self, name: str, multiclass_data) -> None:
        proba, target = multiclass_data
        node = _node(name, task="multiclass", num_classes=N_CLASSES)

        score = _score(node, proba, target)

        assert score == pytest.approx(_sklearn_multiclass(name, proba, target))

    @pytest.mark.parametrize("name", list(LABEL_METRICS))
    @pytest.mark.parametrize("dtype", [np.int64, np.float64])
    def test_hard_labels(self, name: str, dtype: type, multiclass_data) -> None:
        proba, target = multiclass_data
        labels = proba.argmax(axis=1).reshape(-1, 1).astype(dtype)
        node = _node(name, task="multiclass", num_classes=N_CLASSES)

        score = _score(node, labels, target)

        assert score == pytest.approx(_sklearn_multiclass(name, proba, target))

    def test_auroc_rejects_hard_labels(self, multiclass_data) -> None:
        proba, target = multiclass_data
        labels = proba.argmax(axis=1).reshape(-1, 1)
        node = _node("AUROC", task="multiclass", num_classes=N_CLASSES)

        with pytest.raises(NodeInputError, match="probabilities"):
            _score(node, labels, target)

    @pytest.mark.parametrize("name", ALL_METRICS)
    def test_column_count_must_match_num_classes(
        self, name: str, multiclass_data
    ) -> None:
        proba, target = multiclass_data
        node = _node(name, task="multiclass", num_classes=N_CLASSES + 1)

        with pytest.raises(NodeInputError, match="num_classes"):
            _score(node, proba, target)

    @pytest.mark.parametrize("name", ALL_METRICS)
    def test_num_classes_is_required(self, name: str) -> None:
        with pytest.raises(NodeConfigError, match="num_classes"):
            _node(name, task="multiclass")


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


class TestZeroDivision:
    @pytest.mark.parametrize("name", ["F1Score", "Precision", "Recall"])
    def test_default_is_zero(self, name: str) -> None:
        running_class = NODE_REGISTRY.get(name)["running_config"]

        assert running_class().zero_division == 0.0

    @pytest.mark.parametrize("name", ["F1Score", "Precision", "Recall"])
    @pytest.mark.parametrize("zero_division", [0.0, 1.0])
    def test_score_without_positives(self, name: str, zero_division: float) -> None:
        # No positive predictions and no positive labels: every score divides
        # by zero.
        pred = np.zeros((N_ROWS, 1))
        target = np.zeros((N_ROWS, 1), dtype=np.int64)

        score = _score(_node(name, zero_division=zero_division), pred, target)

        reference = LABEL_METRICS[name](
            target[:, 0], pred[:, 0].astype(int), zero_division=zero_division
        )
        assert score == pytest.approx(reference)
        assert score == zero_division


class TestTargets:
    def test_float_labels_are_accepted(self, binary_data) -> None:
        positive, target = binary_data

        score = _score(_node("Accuracy"), positive, target.astype(np.float64))

        assert score == pytest.approx(_sklearn_binary("Accuracy", positive, target))

    def test_non_integer_labels_are_rejected(self, binary_data) -> None:
        positive, _ = binary_data
        target = np.full((N_ROWS, 1), 0.5)

        with pytest.raises(NodeInputError, match="integer class labels"):
            _score(_node("Accuracy"), positive, target)

    def test_more_than_one_target_column_is_rejected(self, binary_data) -> None:
        positive, target = binary_data

        with pytest.raises(NodeInputError, match="one column"):
            _score(_node("Accuracy"), positive, np.hstack([target, target]))


# ---------------------------------------------------------------------------
# End to end: a probability model feeds all metrics through a SmartRunner
# ---------------------------------------------------------------------------


class _ProbaHyperParameters(ModelHyperParameters):
    pass


class _ProbaRunningConfig(ModelRunningConfig):
    pass


class _ProbaModelConfig(ModelConfig[_ProbaHyperParameters, _ProbaRunningConfig]):
    """Config of a fixed binary classifier with two probability columns."""

    hyperparameters: _ProbaHyperParameters = Field(
        default_factory=_ProbaHyperParameters
    )
    running_config: _ProbaRunningConfig = Field(default_factory=_ProbaRunningConfig)
    in_ports: dict[str, Port] = Field(
        default={
            "X": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch features",
                desc="Features.",
            ),
        }
    )
    out_ports: dict[str, Port] = Field(
        default={
            "pred": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch classes",
                desc="Class probabilities, one column per class.",
            ),
        }
    )


class _ProbaModel(
    Model[np.ndarray, TabularDataContext, np.ndarray, TabularDataContext, None]
):
    """Binary classifier: the first feature is the positive probability.

    The output has the format of the sklearn classifier nodes: float64
    columns ``proba_<class>``.
    """

    def __init__(self, *, config: _ProbaModelConfig) -> None:
        self._config = config

    def fit(self, data: dict) -> None:
        _ = data

    def predict(self, data: dict) -> dict:
        X, _ = data["X"]
        positive = X[:, :1].astype(np.float64)
        proba = np.hstack([1.0 - positive, positive])
        ctx = TabularDataContext(
            columns=["proba_0", "proba_1"],
            dtypes=[np.dtype("float64")] * 2,
            categories=[NumericalData] * 2,
        )
        return {"pred": (proba, ctx)}

    def get_params(self) -> None:
        return None

    def set_params(self, params: None) -> None:
        _ = params


@pytest.fixture
def proba_model() -> Iterator[str]:
    """Register the test-local probability model for one test."""
    name = "TestProbaModel"
    NODE_REGISTRY.register(
        name=name, node_class=_ProbaModel, node_config_class=_ProbaModelConfig
    )
    yield name
    NODE_REGISTRY.unregister(name)


def _numerical_pair(df: pd.DataFrame) -> tuple[pd.DataFrame, TabularDataContext]:
    ctx = TabularDataContext(
        columns=list(df.columns),
        dtypes=list(df.dtypes),
        categories=[NumericalData] * len(df.columns),
    )
    return df, ctx


def _classification_pipeline(model_name: str) -> Pipeline:
    source = InputsPassthroughConfig(
        out_ports={
            "X": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch features",
                desc="Features.",
            ),
            "y": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch 1",
                desc="Integer class labels.",
                mode=[NodeExecutionMode.TRAINING, NodeExecutionMode.EVALUATION],
            ),
        }
    )
    nodes = {
        "source": ("InputsPassthrough", source),
        "model": (model_name, _ProbaModelConfig()),
        "sink": ("Sink", NODE_REGISTRY.get_node_config_class("Sink")()),
    }
    edges = [
        Edge(source="source", target="model", ports_map=[("X", "X")]),
        Edge(source="model", target="sink", ports_map=[("pred", "pred")]),
    ]
    for name in ALL_METRICS:
        nodes[name] = (name, NODE_REGISTRY.get_node_config_class(name)())
        edges.append(Edge(source="model", target=name, ports_map=[("pred", "pred")]))
        edges.append(Edge(source="source", target=name, ports_map=[("y", "target")]))
    pipeline = Pipeline(
        config=PipelineConfig(name="classification", nodes=nodes, edges=edges)
    )
    pipeline.compile()
    return pipeline


class TestPipeline:
    def test_binary_probabilities_through_a_runner(
        self, proba_model: str, binary_data
    ) -> None:
        positive, target = binary_data
        X = _numerical_pair(pd.DataFrame({"p": positive[:, 0]}))
        y = _numerical_pair(pd.DataFrame({"label": target[:, 0]}))
        runner = SmartRunner(_classification_pipeline(proba_model))
        runner.train(input_data={"source": {"X": X, "y": y}})

        first = runner.evaluate(input_data={"source": {"X": X, "y": y}})
        second = runner.evaluate(input_data={"source": {"X": X, "y": y}})

        assert set(first) == set(ALL_METRICS)
        for name in ALL_METRICS:
            first_df, _ = first[name]
            second_df, _ = second[name]
            assert first_df.shape == (1, 1)
            assert first_df.iloc[0, 0] == pytest.approx(
                _sklearn_binary(name, positive, target)
            )
            # A second evaluation does not accumulate the first one.
            pd.testing.assert_frame_equal(first_df, second_df)
