"""Tests for the imputers (categorical and numerical)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nodeml.components.nodes.transforms.imputations.categorical_imputation import (
    CategoricalImputation,
    CategoricalImputationConfig,
    CategoricalImputationHyperParameters,
)
from nodeml.components.nodes.transforms.imputations.numerical_imputation import (
    NumericalImputation,
    NumericalImputationConfig,
    NumericalImputationHyperParameters,
)
from nodeml.core.common.data.data import (
    CategoricalData,
    NumericalData,
    TabularDataContext,
)
from nodeml.core.common.exceptions import NodeInputError


def _pair(
    df: pd.DataFrame, category: type = CategoricalData
) -> tuple[pd.DataFrame, TabularDataContext]:
    return df, TabularDataContext(
        columns=list(df.columns),
        dtypes=list(df.dtypes),
        categories=[category] * df.shape[1],
    )


def _categorical(
    strategy: str = "most_frequent", value: str = "missing"
) -> CategoricalImputation:
    hp = CategoricalImputationHyperParameters(strategy=strategy, value=value)
    return CategoricalImputation(config=CategoricalImputationConfig(hyperparameters=hp))


def _numerical(strategy: str = "mean", value: float = 0.0) -> NumericalImputation:
    hp = NumericalImputationHyperParameters(strategy=strategy, value=value)
    return NumericalImputation(config=NumericalImputationConfig(hyperparameters=hp))


def _check_context(
    out: tuple[pd.DataFrame, TabularDataContext], category: type
) -> None:
    df, ctx = out
    assert ctx.columns == list(df.columns)
    assert ctx.dtypes == list(df.dtypes)
    assert ctx.categories == [category] * df.shape[1]


class TestCategoricalImputation:
    @pytest.mark.parametrize("strategy", ["most_frequent", "constant"])
    def test_pandas_categorical_column_keeps_its_dtype(self, strategy) -> None:
        df = pd.DataFrame({"c": pd.Categorical(["x", None, "x", "y"])})
        out = _categorical(strategy).node_fit_transform({"input": _pair(df)})["output"]
        expected = "x" if strategy == "most_frequent" else "missing"
        assert out[0]["c"].tolist() == ["x", expected, "x", "y"]
        assert isinstance(out[0]["c"].dtype, pd.CategoricalDtype)
        _check_context(out, CategoricalData)

    def test_most_frequent_keeps_a_nullable_integer_dtype(self) -> None:
        df = pd.DataFrame({"c": pd.array([1, None, 1, 2], dtype="Int64")})
        out = _categorical().node_fit_transform({"input": _pair(df)})["output"]
        assert out[0]["c"].tolist() == [1, 1, 1, 2]
        assert out[0]["c"].dtype == "Int64"
        _check_context(out, CategoricalData)

    def test_most_frequent_keeps_a_float_code_column_numeric(self) -> None:
        df = pd.DataFrame({"c": [3.0, np.nan, 3.0, 4.0]})
        out = _categorical().node_fit_transform({"input": _pair(df)})["output"]
        assert out[0]["c"].tolist() == [3.0, 3.0, 3.0, 4.0]
        assert out[0]["c"].dtype == np.float64

    def test_constant_string_in_a_numeric_column_gives_object(self) -> None:
        df = pd.DataFrame({"c": pd.array([1, None], dtype="Int64")})
        out = _categorical("constant").node_fit_transform({"input": _pair(df)})[
            "output"
        ]
        assert out[0]["c"].tolist() == [1, "missing"]
        _check_context(out, CategoricalData)

    def test_all_missing_column_gets_the_constant(self) -> None:
        df = pd.DataFrame({"c": [np.nan, np.nan]})
        out = _categorical().node_fit_transform({"input": _pair(df)})["output"]
        assert out[0]["c"].tolist() == ["missing", "missing"]
        _check_context(out, CategoricalData)

    def test_missing_fitted_column_raises(self) -> None:
        node = _categorical()
        node.node_fit({"input": _pair(pd.DataFrame({"a": ["x"], "b": ["y"]}))})
        with pytest.raises(NodeInputError, match="'b'"):
            node.node_transform({"input": _pair(pd.DataFrame({"a": [None]}))})


class TestNumericalImputation:
    def test_non_integral_mean_in_a_nullable_integer_column(self) -> None:
        df = pd.DataFrame({"n": pd.array([1, None, 2], dtype="Int64")})
        out = _numerical().node_fit_transform({"input": _pair(df, NumericalData)})[
            "output"
        ]
        assert out[0]["n"].tolist() == [1.0, 1.5, 2.0]
        _check_context(out, NumericalData)

    @pytest.mark.parametrize(
        ("strategy", "expected"), [("mean", 2.0), ("median", 1.5), ("constant", -1.0)]
    )
    def test_strategies(self, strategy, expected) -> None:
        df = pd.DataFrame({"n": [1.0, np.nan, 1.5, 3.5]})
        node = _numerical(strategy, value=-1.0)
        out = node.node_fit_transform({"input": _pair(df, NumericalData)})["output"]
        assert out[0]["n"].tolist()[1] == pytest.approx(expected)
        _check_context(out, NumericalData)

    def test_missing_fitted_column_raises(self) -> None:
        node = _numerical()
        node.node_fit({"input": _pair(pd.DataFrame({"a": [1.0], "b": [2.0]}))})
        with pytest.raises(NodeInputError, match="'b'"):
            node.node_transform({"input": _pair(pd.DataFrame({"a": [np.nan]}))})


class TestSaveLoad:
    @pytest.mark.parametrize(
        ("make", "df"),
        [
            (
                _categorical,
                pd.DataFrame({"c": ["x", None, "x"], "d": [1.0, 2.0, None]}),
            ),
            (_numerical, pd.DataFrame({"n": [1.0, None, 3.0], "m": [None, 2.0, 4.0]})),
        ],
    )
    def test_set_params_restores_the_node(self, make, df) -> None:
        fitted = make()
        expected = fitted.node_fit_transform({"input": _pair(df)})

        reborn = make()
        reborn.set_params(fitted.get_params())
        actual = reborn.node_transform({"input": _pair(df)})
        pd.testing.assert_frame_equal(actual["output"][0], expected["output"][0])
        assert actual["output"][1] == expected["output"][1]
