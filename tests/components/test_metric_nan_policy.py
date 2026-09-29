"""Tests for the ``nan_policy`` option of the metric nodes.

With ``nan_policy="omit"`` (the default), a metric drops the rows where
the prediction or the target has a NaN, and logs one warning.  With
``"propagate"``, the rows stay and the score can be NaN.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import pytest
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
from nodeml.core.common.exceptions import NodeInputError
from nodeml.core.nodes.node import Port
from nodeml.core.nodes.registry.node_registry import NODE_REGISTRY
from nodeml.core.pipeline.pipeline import Edge, Pipeline, PipelineConfig
from nodeml.core.pipeline.runners.smart_runner import SmartRunner

ALL_METRICS = [
    "Accuracy",
    "AUROC",
    "F1Score",
    "Precision",
    "Recall",
    "MSE",
    "MAE",
    "MAPE",
    "R2Score",
]
N_ROWS = 30


def _node(name: str, **running: Any) -> Any:
    config_class = NODE_REGISTRY.get_node_config_class(name)
    running_class = NODE_REGISTRY.get(name)["running_config"]
    config = config_class(running_config=running_class(**running))
    return NODE_REGISTRY.get_node_class(name)(config=config)


def _score(node: Any, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    score, _ = node.node_transform({"pred": (pred, None), "target": (target, None)})[
        "score"
    ]
    return score


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == logging.WARNING]


@pytest.fixture
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    target = rng.normal(loc=5.0, size=(N_ROWS, 2))
    pred = target + rng.normal(scale=0.3, size=(N_ROWS, 2))
    return pred, target


@pytest.fixture
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    """Class probabilities ``(n, 2)`` and float labels ``(n, 1)``."""
    rng = np.random.default_rng(8)
    positive = rng.random((N_ROWS, 1))
    target = (positive + rng.normal(scale=0.3, size=(N_ROWS, 1)) > 0.5).astype(float)
    return np.hstack([1.0 - positive, positive]), target


@pytest.mark.parametrize("name", ALL_METRICS)
def test_default_policy_is_omit(name: str) -> None:
    running_class = NODE_REGISTRY.get(name)["running_config"]

    assert running_class().nan_policy == "omit"


class TestOmit:
    def test_nan_rows_are_dropped_with_one_warning(
        self, caplog: pytest.LogCaptureFixture, regression_data
    ) -> None:
        pred, target = regression_data
        pred, target = pred.copy(), target.copy()
        target[[1, 4], 0] = np.nan
        pred[7, 1] = np.nan
        keep = np.setdiff1d(np.arange(N_ROWS), [1, 4, 7])

        with caplog.at_level(logging.WARNING, logger="nodeml"):
            score = _score(_node("MSE", num_outputs=2), pred, target)

        expected = skm.mean_squared_error(
            target[keep], pred[keep], multioutput="raw_values"
        )
        np.testing.assert_allclose(score[0], expected, rtol=1e-12)
        warnings = _warnings(caplog)
        assert len(warnings) == 1
        assert warnings[0].context["node_class"] == "MSE"
        assert warnings[0].context["dropped_rows"] == 3

    def test_no_warning_without_nan(
        self, caplog: pytest.LogCaptureFixture, regression_data
    ) -> None:
        pred, target = regression_data

        with caplog.at_level(logging.WARNING, logger="nodeml"):
            _score(_node("R2Score"), pred, target)

        assert not _warnings(caplog)

    @pytest.mark.parametrize("name", ALL_METRICS)
    def test_all_nan_rows_raise(self, name: str, binary_data) -> None:
        pred, target = binary_data
        target = np.full_like(target, np.nan)

        with pytest.raises(NodeInputError, match="NaN"):
            _score(_node(name), pred, target)

    @pytest.mark.parametrize(
        ("name", "reference"),
        [
            ("Accuracy", lambda t, p: skm.accuracy_score(t, p[:, 1] > 0.5)),
            ("F1Score", lambda t, p: skm.f1_score(t, p[:, 1] > 0.5)),
            ("AUROC", lambda t, p: skm.roc_auc_score(t, p[:, 1])),
        ],
    )
    def test_classification_with_nan_targets(
        self,
        caplog: pytest.LogCaptureFixture,
        name: str,
        reference: Any,
        binary_data,
    ) -> None:
        proba, target = binary_data
        target = target.copy()
        target[[0, 5, 9]] = np.nan
        keep = np.setdiff1d(np.arange(N_ROWS), [0, 5, 9])

        with caplog.at_level(logging.WARNING, logger="nodeml"):
            score = _score(_node(name), proba, target)

        expected = reference(target[keep, 0].astype(int), proba[keep])
        assert float(score[0, 0]) == pytest.approx(expected)
        assert _warnings(caplog)[0].context["dropped_rows"] == 3


class TestPropagate:
    @pytest.mark.parametrize("name", ["MSE", "MAE", "MAPE", "R2Score"])
    def test_regression_score_is_nan(self, name: str, regression_data) -> None:
        pred, target = regression_data
        target = target.copy()
        target[3, 0] = np.nan

        score = _score(_node(name, nan_policy="propagate"), pred, target)

        assert np.isnan(score).all()

    def test_no_rows_are_dropped(
        self, caplog: pytest.LogCaptureFixture, regression_data
    ) -> None:
        pred, target = regression_data
        target = target.copy()
        target[3, 0] = np.nan

        with caplog.at_level(logging.WARNING, logger="nodeml"):
            _score(_node("MSE", nan_policy="propagate"), pred, target)

        assert not _warnings(caplog)

    def test_classification_nan_target_is_rejected(self, binary_data) -> None:
        proba, target = binary_data
        target = target.copy()
        target[2] = np.nan

        with pytest.raises(NodeInputError, match="nan_policy"):
            _score(_node("Accuracy", nan_policy="propagate"), proba, target)


def _pair(array: np.ndarray, prefix: str) -> tuple[pd.DataFrame, TabularDataContext]:
    df = pd.DataFrame(array, columns=[f"{prefix}{i}" for i in range(array.shape[1])])
    ctx = TabularDataContext(
        columns=list(df.columns),
        dtypes=list(df.dtypes),
        categories=[NumericalData] * len(df.columns),
    )
    return df, ctx


def test_nan_targets_through_a_runner(regression_data) -> None:
    pred, target = regression_data
    port = Port(
        arr_type=ArrayLikeEnum.PANDAS,
        data_structure=DataStructureEnum.TABULAR,
        data_category=DataCategoryEnum.NUMERICAL,
        data_shape="batch targets",
        desc="Data.",
    )
    source = InputsPassthroughConfig(out_ports={"pred": port, "y": port})
    nodes = {
        "source": ("InputsPassthrough", source),
        "sink": ("Sink", NODE_REGISTRY.get_node_config_class("Sink")()),
        "r2": ("R2Score", NODE_REGISTRY.get_node_config_class("R2Score")()),
    }
    edges = [
        Edge(source="source", target="sink", ports_map=[("pred", "pred")]),
        Edge(
            source="source", target="r2", ports_map=[("pred", "pred"), ("y", "target")]
        ),
    ]
    pipeline = Pipeline(config=PipelineConfig(name="nan", nodes=nodes, edges=edges))
    pipeline.compile()
    runner = SmartRunner(pipeline)
    with_nan = target.copy()
    with_nan[[2, 11], 1] = np.nan
    keep = np.setdiff1d(np.arange(N_ROWS), [2, 11])

    metrics = runner.evaluate(
        input_data={"source": {"pred": _pair(pred, "p"), "y": _pair(with_nan, "t")}}
    )

    score_df, _ = metrics["r2"]
    assert score_df.iloc[0, 0] == pytest.approx(skm.r2_score(target[keep], pred[keep]))
