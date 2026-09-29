"""Tests for the encoders (label encoding and one-hot encoding)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nodeml.components.nodes.transforms.encodings.label_encoding import (
    LabelEncoding,
    LabelEncodingConfig,
)
from nodeml.components.nodes.transforms.encodings.one_hot_encoding import (
    OneHotEncoding,
    OneHotEncodingConfig,
)
from nodeml.core.common.data.data import (
    CategoricalData,
    NumericalData,
    TabularDataContext,
)
from nodeml.core.common.exceptions import NodeInputError


def _pair(df: pd.DataFrame) -> tuple[pd.DataFrame, TabularDataContext]:
    return df, TabularDataContext(
        columns=list(df.columns),
        dtypes=list(df.dtypes),
        categories=[CategoricalData] * df.shape[1],
    )


def _label() -> LabelEncoding:
    return LabelEncoding(config=LabelEncodingConfig())


def _one_hot() -> OneHotEncoding:
    return OneHotEncoding(config=OneHotEncodingConfig())


class TestLabelEncoding:
    def test_float_categories_match_integer_values_at_inference(self) -> None:
        # NaN makes the fit column float64; inference sees int64 values.
        node = _label()
        node.node_fit({"input": _pair(pd.DataFrame({"k": [1.0, 2.0, np.nan, 10.0]}))})
        out_df, _ = node.node_transform(
            {"input": _pair(pd.DataFrame({"k": [1, 2, 10]}))}
        )["output"]
        assert out_df["k"].tolist() == [0, 1, 2]

    def test_numbers_are_ordered_by_value(self) -> None:
        df = pd.DataFrame({"k": [10, 2, 1, 2]})
        out_df, _ = _label().node_fit_transform({"input": _pair(df)})["output"]
        assert out_df["k"].tolist() == [2, 1, 0, 1]

    def test_strings_are_ordered_and_unseen_values_get_minus_one(self) -> None:
        node = _label()
        node.node_fit({"input": _pair(pd.DataFrame({"k": ["b", "a", None, "c"]}))})
        df = pd.DataFrame({"k": ["c", "a", "z", None]})
        out_df, _ = node.node_transform({"input": _pair(df)})["output"]
        assert out_df["k"].tolist() == [2, 0, -1, -1]

    def test_mixed_types_do_not_fail(self) -> None:
        df = pd.DataFrame({"k": pd.Series(["a", 1, 2.5, "a"], dtype=object)})
        out_df, _ = _label().node_fit_transform({"input": _pair(df)})["output"]
        assert out_df["k"].tolist()[0] == out_df["k"].tolist()[3]
        assert sorted(out_df["k"].tolist()[:3]) == [0, 1, 2]

    def test_categorical_dtype(self) -> None:
        df = pd.DataFrame({"k": pd.Categorical(["y", "x", None, "y"])})
        out_df, out_ctx = _label().node_fit_transform({"input": _pair(df)})["output"]
        assert out_df["k"].tolist() == [1, 0, -1, 1]
        assert out_ctx.dtypes == [np.dtype("int64")]
        assert out_ctx.categories == [NumericalData]

    def test_zero_columns_keep_the_rows(self) -> None:
        df = pd.DataFrame(index=[5, 6, 7])
        out_df, out_ctx = _label().node_fit_transform({"input": _pair(df)})["output"]
        assert out_df.shape == (3, 0)
        assert list(out_df.index) == [5, 6, 7]
        assert out_ctx.columns == []

    def test_output_keeps_the_input_index_and_order(self) -> None:
        df = pd.DataFrame({"a": ["x", "y"], "b": ["u", "v"]}, index=[3, 1])
        node = _label()
        node.node_fit({"input": _pair(df)})
        out_df, out_ctx = node.node_transform({"input": _pair(df[["b", "a"]])})[
            "output"
        ]
        assert list(out_df.index) == [3, 1]
        assert list(out_df.columns) == ["b", "a"]
        assert out_ctx.columns == ["b", "a"]

    def test_missing_fitted_column_raises(self) -> None:
        node = _label()
        node.node_fit({"input": _pair(pd.DataFrame({"a": ["x"], "b": ["y"]}))})
        with pytest.raises(NodeInputError, match="'b'"):
            node.node_transform({"input": _pair(pd.DataFrame({"a": ["x"]}))})


class TestOneHotEncoding:
    def test_float_categories_match_integer_values_at_inference(self) -> None:
        node = _one_hot()
        node.node_fit({"input": _pair(pd.DataFrame({"k": [1.0, 2.0, np.nan]}))})
        out_df, out_ctx = node.node_transform(
            {"input": _pair(pd.DataFrame({"k": [2, 1, 3]}))}
        )["output"]
        assert list(out_df.columns) == ["k_1", "k_2"]
        assert out_df.to_numpy().tolist() == [[0, 1], [1, 0], [0, 0]]
        assert out_ctx.columns == ["k_1", "k_2"]
        assert out_ctx.dtypes == [np.dtype("uint8")] * 2
        assert out_ctx.categories == [NumericalData] * 2

    def test_numbers_are_ordered_by_value(self) -> None:
        df = pd.DataFrame({"k": [10, 2, 1]})
        out_df, _ = _one_hot().node_fit_transform({"input": _pair(df)})["output"]
        assert list(out_df.columns) == ["k_1", "k_2", "k_10"]

    def test_zero_columns_keep_the_rows(self) -> None:
        df = pd.DataFrame(index=[5, 6, 7])
        out_df, out_ctx = _one_hot().node_fit_transform({"input": _pair(df)})["output"]
        assert out_df.shape == (3, 0)
        assert list(out_df.index) == [5, 6, 7]
        assert out_ctx.columns == []

    def test_output_keeps_the_input_index(self) -> None:
        df = pd.DataFrame({"a": ["x", "y", "x"]}, index=[9, 8, 7])
        out_df, _ = _one_hot().node_fit_transform({"input": _pair(df)})["output"]
        assert list(out_df.index) == [9, 8, 7]
        assert out_df.to_numpy().tolist() == [[1, 0], [0, 1], [1, 0]]

    def test_duplicate_output_names_raise(self) -> None:
        df = pd.DataFrame({"k": pd.Series(["1", 1], dtype=object)})
        with pytest.raises(NodeInputError, match="k_1"):
            _one_hot().node_fit({"input": _pair(df)})


class TestSaveLoad:
    @pytest.mark.parametrize("make", [_label, _one_hot])
    def test_set_params_restores_the_node(self, make) -> None:
        df = pd.DataFrame({"a": ["x", "y", None, "x"], "b": [1.0, np.nan, 3.0, 1.0]})
        fitted = make()
        expected = fitted.node_fit_transform({"input": _pair(df)})

        reborn = make()
        reborn.set_params(fitted.get_params())
        actual = reborn.node_transform({"input": _pair(df)})
        pd.testing.assert_frame_equal(actual["output"][0], expected["output"][0])
        assert actual["output"][1] == expected["output"][1]
