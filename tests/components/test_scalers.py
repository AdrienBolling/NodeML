"""Tests for the scalers (MinMax, Robust and Standard)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nodeml.components.nodes.transforms.scalers.min_max_scaler import (
    MinMaxScaler,
    MinMaxScalerConfig,
)
from nodeml.components.nodes.transforms.scalers.robust_scaler import (
    RobustScaler,
    RobustScalerConfig,
)
from nodeml.components.nodes.transforms.scalers.standard_scaler import (
    StandardScaler,
    StandardScalerConfig,
)
from nodeml.core.common.data.data import NumericalData
from nodeml.core.common.exceptions import NodeInputError
from tests.shims.tabular import numerical_context

SCALERS = {
    "min_max": (MinMaxScaler, MinMaxScalerConfig),
    "robust": (RobustScaler, RobustScalerConfig),
    "standard": (StandardScaler, StandardScalerConfig),
}


def _make(kind: str) -> MinMaxScaler | RobustScaler | StandardScaler:
    node_class, config_class = SCALERS[kind]
    return node_class(config=config_class())


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 10.0],
            "b": [-5.0, 0.0, 5.0, 10.0, 15.0],
            "c": [0.5, 0.25, 0.75, 1.0, 0.0],
        }
    )


@pytest.fixture(params=sorted(SCALERS))
def kind(request) -> str:
    return request.param


class TestColumnOrder:
    def test_other_input_order_is_kept(self, kind) -> None:
        df = _frame()
        node = _make(kind)
        expected, _ = node.node_fit_transform({"input": (df, numerical_context(df))})[
            "output"
        ]
        shuffled = df[["c", "a", "b"]]
        out_df, out_ctx = node.node_transform(
            {"input": (shuffled, numerical_context(shuffled))}
        )["output"]
        assert list(out_df.columns) == ["c", "a", "b"]
        assert out_ctx.columns == ["c", "a", "b"]
        pd.testing.assert_frame_equal(out_df, expected[["c", "a", "b"]])

    def test_missing_fitted_column_raises(self, kind) -> None:
        df = _frame()
        node = _make(kind)
        node.node_fit({"input": (df, numerical_context(df))})
        part = df[["a", "b"]]
        with pytest.raises(NodeInputError, match="'c'"):
            node.node_transform({"input": (part, numerical_context(part))})

    def test_unknown_column_raises(self, kind) -> None:
        df = _frame()
        node = _make(kind)
        node.node_fit({"input": (df, numerical_context(df))})
        more = df.assign(d=1.0)
        with pytest.raises(NodeInputError, match="'d'"):
            node.node_transform({"input": (more, numerical_context(more))})


class TestZeroSpread:
    def test_one_row_fit_gives_finite_output(self, kind) -> None:
        df = pd.DataFrame({"a": [3.0], "b": [-1.0]})
        node = _make(kind)
        out_df, _ = node.node_fit_transform({"input": (df, numerical_context(df))})[
            "output"
        ]
        # With no spread, the scaler only centres the column.
        np.testing.assert_allclose(out_df.to_numpy(), [[0.0, 0.0]])
        more = pd.DataFrame({"a": [5.0], "b": [-1.0]})
        out_df, _ = node.node_transform({"input": (more, numerical_context(more))})[
            "output"
        ]
        np.testing.assert_allclose(out_df.to_numpy(), [[2.0, 0.0]])

    def test_constant_column_is_only_centred(self, kind) -> None:
        df = pd.DataFrame({"const": [4.0] * 5, "var": np.arange(5.0)})
        node = _make(kind)
        out_df, _ = node.node_fit_transform({"input": (df, numerical_context(df))})[
            "output"
        ]
        np.testing.assert_allclose(out_df["const"].to_numpy(), np.zeros(5))


class TestContext:
    def test_integer_input_gives_a_float_context(self, kind) -> None:
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]}, dtype="int64")
        out_df, out_ctx = _make(kind).node_fit_transform(
            {"input": (df, numerical_context(df))}
        )["output"]
        assert out_ctx.columns == ["a", "b"]
        assert out_ctx.dtypes == list(out_df.dtypes)
        assert out_ctx.dtypes == [np.dtype("float64"), np.dtype("float64")]
        assert out_ctx.categories == [NumericalData, NumericalData]

    def test_output_keeps_the_input_index(self, kind) -> None:
        df = _frame().set_axis(list("vwxyz"))
        out_df, _ = _make(kind).node_fit_transform(
            {"input": (df, numerical_context(df))}
        )["output"]
        assert list(out_df.index) == list("vwxyz")


class TestValues:
    def test_min_max_range(self) -> None:
        df = _frame()
        out_df, _ = _make("min_max").node_fit_transform(
            {"input": (df, numerical_context(df))}
        )["output"]
        np.testing.assert_allclose(out_df.min().to_numpy(), 0.0)
        np.testing.assert_allclose(out_df.max().to_numpy(), 1.0)

    def test_robust_median_and_iqr(self) -> None:
        df = _frame()
        out_df, _ = _make("robust").node_fit_transform(
            {"input": (df, numerical_context(df))}
        )["output"]
        np.testing.assert_allclose(out_df.median().to_numpy(), 0.0)
        iqr = out_df.quantile(0.75) - out_df.quantile(0.25)
        np.testing.assert_allclose(iqr.to_numpy(), 1.0)


class TestSaveLoad:
    def test_set_params_restores_the_node(self, kind) -> None:
        df = _frame()
        ctx = numerical_context(df)
        fitted = _make(kind)
        expected = fitted.node_fit_transform({"input": (df, ctx)})

        reborn = _make(kind)
        reborn.set_params(fitted.get_params())
        actual = reborn.node_transform({"input": (df, ctx)})
        pd.testing.assert_frame_equal(actual["output"][0], expected["output"][0])
        assert actual["output"][1] == expected["output"][1]
