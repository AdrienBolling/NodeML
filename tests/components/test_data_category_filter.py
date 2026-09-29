"""Tests for the DataCategoryFilter transform node."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nodeml.components.nodes.transforms.feature_selection.data_category_filter import (
    DataCategoryFilter,
    DataCategoryFilterConfig,
)
from nodeml.core.common.data.data import (
    CategoricalData,
    MixedData,
    NumericalData,
    TabularDataContext,
)
from nodeml.core.common.exceptions import NodeInputError

# The first letter of a column name gives its category.
_CATEGORIES = {"n": NumericalData, "c": CategoricalData, "m": MixedData}


def _pair(df: pd.DataFrame) -> tuple[pd.DataFrame, TabularDataContext]:
    return df, TabularDataContext(
        columns=list(df.columns),
        dtypes=list(df.dtypes),
        categories=[_CATEGORIES[col[0]] for col in df.columns],
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n1a": [1.0, 2.0],
            "c1a": ["x", "y"],
            "m": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "n1b": [3, 4],
            "c1b": ["u", "v"],
        }
    )


class TestDataCategoryFilter:
    def test_split_by_context_category(self) -> None:
        out = DataCategoryFilter(config=DataCategoryFilterConfig()).node_fit_transform(
            {"input": _pair(_frame())}
        )
        num_df, num_ctx = out["numerical"]
        cat_df, cat_ctx = out["categorical"]
        assert list(num_df.columns) == num_ctx.columns == ["n1a", "n1b"]
        assert list(cat_df.columns) == cat_ctx.columns == ["c1a", "c1b"]
        assert num_ctx.dtypes == [np.dtype("float64"), np.dtype("int64")]

    def test_other_input_order_keeps_output_and_context_aligned(self) -> None:
        node = DataCategoryFilter(config=DataCategoryFilterConfig())
        node.node_fit({"input": _pair(_frame())})
        shuffled = _frame()[["c1b", "n1b", "m", "c1a", "n1a"]]
        out = node.node_transform({"input": _pair(shuffled)})
        for port in ("numerical", "categorical"):
            df, ctx = out[port]
            assert ctx.columns == list(df.columns)
            assert ctx.dtypes == list(df.dtypes)
        assert list(out["numerical"][0].columns) == ["n1b", "n1a"]

    def test_missing_fitted_column_raises(self) -> None:
        node = DataCategoryFilter(config=DataCategoryFilterConfig())
        node.node_fit({"input": _pair(_frame())})
        part = _frame().drop(columns=["c1b"])
        with pytest.raises(NodeInputError, match="c1b"):
            node.node_transform({"input": _pair(part)})

    def test_set_params_restores_the_node(self) -> None:
        fitted = DataCategoryFilter(config=DataCategoryFilterConfig())
        expected = fitted.node_fit_transform({"input": _pair(_frame())})
        reborn = DataCategoryFilter(config=DataCategoryFilterConfig())
        reborn.set_params(fitted.get_params())
        actual = reborn.node_transform({"input": _pair(_frame())})
        for port in ("numerical", "categorical"):
            pd.testing.assert_frame_equal(actual[port][0], expected[port][0])
            assert actual[port][1] == expected[port][1]
