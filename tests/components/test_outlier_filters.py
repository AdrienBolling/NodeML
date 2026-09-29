"""Tests for the outlier filters (IQR and z-score)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from nodeml.components.nodes.transforms.filters._outlier_filter import OutlierFilter
from nodeml.components.nodes.transforms.filters.iqr_outlier_filter import (
    IQROutlierFilter,
    IQROutlierFilterConfig,
    IQROutlierFilterHyperParameters,
)
from nodeml.components.nodes.transforms.filters.zscore_outlier_filter import (
    ZScoreOutlierFilter,
    ZScoreOutlierFilterConfig,
    ZScoreOutlierFilterHyperParameters,
)
from nodeml.core.common.data.data import (
    CategoricalData,
    NumericalData,
    TabularDataContext,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.common.exceptions import NodeInputError
from tests.shims.tabular import numerical_context

FILTERS = {
    "iqr": (IQROutlierFilter, IQROutlierFilterConfig, IQROutlierFilterHyperParameters),
    "zscore": (
        ZScoreOutlierFilter,
        ZScoreOutlierFilterConfig,
        ZScoreOutlierFilterHyperParameters,
    ),
}


def _make(kind: str, **hyperparameters: object) -> OutlierFilter:
    node_class, config_class, hp_class = FILTERS[kind]
    if kind == "zscore":
        hyperparameters.setdefault("zscore_cutoff", 1.5)
    return node_class(config=config_class(hyperparameters=hp_class(**hyperparameters)))


def _frame() -> pd.DataFrame:
    """Return six rows; the last row is an outlier in column ``a`` only."""
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


def _pair(df: pd.DataFrame) -> tuple[pd.DataFrame, TabularDataContext]:
    return df, numerical_context(df)


def _train(
    node: OutlierFilter, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
    node.set_execution_mode(NodeExecutionMode.TRAINING)
    return node.node_fit_transform(data)


@pytest.fixture(params=sorted(FILTERS))
def kind(request) -> str:
    return request.param


class TestRowRemovalOnlyInTraining:
    def test_training_removes_the_outlier_row(self, kind) -> None:
        out = _train(_make(kind), {"input": _pair(_frame())})
        assert out["output"][0]["a"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]

    @pytest.mark.parametrize(
        "mode", [NodeExecutionMode.INFERENCE, NodeExecutionMode.EVALUATION]
    )
    def test_other_modes_keep_every_row(self, kind, mode) -> None:
        node = _make(kind)
        _train(node, {"input": _pair(_frame())})
        node.set_execution_mode(mode)
        target = pd.DataFrame({"y": np.arange(6.0)})
        out = node.node_transform({"input": _pair(_frame()), "target": _pair(target)})
        pd.testing.assert_frame_equal(out["output"][0], _frame())
        pd.testing.assert_frame_equal(out["target"][0], target)

    def test_cap_applies_in_inference(self, kind) -> None:
        node = _make(kind, strategy="cap")
        _train(node, {"input": _pair(_frame())})
        node.set_execution_mode(NodeExecutionMode.INFERENCE)
        out, _ = node.node_transform({"input": _pair(_frame())})["output"]
        assert len(out) == 6
        assert out["a"].iloc[-1] < 100.0


class TestTargetAndSlicedRows:
    def test_target_with_another_index_follows_input_by_position(self, kind) -> None:
        target = pd.DataFrame({"y": np.arange(6.0)}, index=range(10, 16))
        sliced = pd.DataFrame({"s": list("uvwxyz")}, index=list("pqrstu"))
        sliced_ctx = TabularDataContext(
            columns=["s"], dtypes=[np.dtype(object)], categories=[CategoricalData]
        )
        out = _train(
            _make(kind),
            {
                "input": _pair(_frame()),
                "target": _pair(target),
                "sliced": (sliced, sliced_ctx),
            },
        )
        assert out["target"][0]["y"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert out["sliced"][0]["s"].tolist() == list("uvwxy")

    def test_target_with_another_row_count_raises(self, kind) -> None:
        target = pd.DataFrame({"y": np.arange(5.0)})
        with pytest.raises(NodeInputError, match="rows"):
            _train(_make(kind), {"input": _pair(_frame()), "target": _pair(target)})


class TestSlicedPortIsOptional:
    @pytest.mark.parametrize(
        "config_class", [IQROutlierFilterConfig, ZScoreOutlierFilterConfig]
    )
    def test_sliced_ports_are_optional(self, config_class) -> None:
        config = config_class()
        assert config.in_ports["sliced"].optional
        assert config.out_ports["sliced"].optional


class TestThreshold:
    """Rule: remove a row with at least one outlier feature and outlier fraction >= threshold."""

    @staticmethod
    def _two_feature_frame() -> pd.DataFrame:
        # Row 10 is an outlier in "a" only; row 11 is an outlier in "a" and "b".
        normal = [1.0, 2.0, 3.0, 4.0, 5.0] * 2
        return pd.DataFrame({"a": [*normal, 100.0, 100.0], "b": [*normal, 3.0, 100.0]})

    @pytest.mark.parametrize(
        ("threshold", "expected_rows"), [(0.0, 10), (0.5, 10), (0.6, 11), (1.0, 11)]
    )
    def test_threshold_rule(self, kind, threshold, expected_rows) -> None:
        df = self._two_feature_frame()
        node = _make(kind, threshold=threshold)
        if kind == "zscore":
            node = _make(kind, threshold=threshold, zscore_cutoff=1.0)
        out = _train(node, {"input": _pair(df)})
        assert len(out["output"][0]) == expected_rows

    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_threshold_out_of_range_is_rejected(self, kind, threshold) -> None:
        _, _, hp_class = FILTERS[kind]
        with pytest.raises(ValidationError):
            hp_class(threshold=threshold)


class TestZeroSpread:
    @pytest.mark.parametrize("strategy", ["remove", "cap"])
    def test_zscore_one_row_fit_skips_the_column(self, strategy) -> None:
        df = pd.DataFrame({"a": [3.0]})
        node = _make("zscore", strategy=strategy)
        out = _train(node, {"input": _pair(df)})
        assert out["output"][0]["a"].tolist() == [3.0]
        out = node.node_transform({"input": _pair(pd.DataFrame({"a": [1e6, 3.0]}))})
        assert out["output"][0]["a"].tolist() == [1e6, 3.0]

    def test_zscore_constant_column_is_skipped_with_cap(self) -> None:
        df = pd.DataFrame({"a": [2.0, 2.0, 2.0]})
        node = _make("zscore", strategy="cap")
        _train(node, {"input": _pair(df)})
        out = node.node_transform({"input": _pair(pd.DataFrame({"a": [50.0]}))})
        assert out["output"][0]["a"].tolist() == [50.0]


class TestContext:
    def test_cap_on_integer_input_updates_the_context_dtype(self, kind) -> None:
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 100], "b": [1, 2, 3, 4, 5, 6]})
        out_df, out_ctx = _train(_make(kind, strategy="cap"), {"input": _pair(df)})[
            "output"
        ]
        assert out_ctx.columns == list(out_df.columns)
        assert out_ctx.dtypes == list(out_df.dtypes)
        assert out_ctx.categories == [NumericalData, NumericalData]

    def test_filtering_columns_limits_the_detection(self, kind) -> None:
        node_class, config_class, hp_class = FILTERS[kind]
        running_class = config_class.model_fields["running_config"].annotation
        hp = hp_class(zscore_cutoff=1.5) if kind == "zscore" else hp_class()
        node = node_class(
            config=config_class(
                hyperparameters=hp,
                running_config=running_class(filtering_columns=["b"]),
            )
        )
        out = _train(node, {"input": _pair(_frame())})
        assert len(out["output"][0]) == 6

    def test_missing_fitted_column_raises(self, kind) -> None:
        node = _make(kind)
        _train(node, {"input": _pair(_frame())})
        df = _frame()[["b"]]
        with pytest.raises(NodeInputError, match="'a'"):
            node.node_transform({"input": _pair(df)})


class TestSaveLoad:
    @pytest.mark.parametrize("strategy", ["remove", "cap"])
    @pytest.mark.parametrize(
        "mode", [NodeExecutionMode.TRAINING, NodeExecutionMode.INFERENCE]
    )
    def test_set_params_restores_the_node(self, kind, strategy, mode) -> None:
        fitted = _make(kind, strategy=strategy)
        _train(fitted, {"input": _pair(_frame())})

        reborn = _make(kind, strategy=strategy)
        reborn.set_params(fitted.get_params())
        for node in (fitted, reborn):
            node.set_execution_mode(mode)
        expected = fitted.node_transform({"input": _pair(_frame())})
        actual = reborn.node_transform({"input": _pair(_frame())})
        pd.testing.assert_frame_equal(actual["output"][0], expected["output"][0])
