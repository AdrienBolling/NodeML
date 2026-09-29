"""Tests for the ColumnOrder transform node."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nodeml.components.nodes.transforms.operations.column_order import (
    ColumnOrder,
    ColumnOrderConfig,
    ColumnOrderRunningConfig,
)
from nodeml.core.common.data.data import (
    CategoricalData,
    NumericalData,
    TabularDataContext,
)
from nodeml.core.common.exceptions import NodeInputError


def _pair(df: pd.DataFrame) -> tuple[pd.DataFrame, TabularDataContext]:
    categories = [
        NumericalData if pd.api.types.is_numeric_dtype(dtype) else CategoricalData
        for dtype in df.dtypes
    ]
    return df, TabularDataContext(
        columns=list(df.columns), dtypes=list(df.dtypes), categories=categories
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"], "c": [1, 2]})


class TestColumnOrder:
    def test_learned_order_reorders_the_context_too(self) -> None:
        node = ColumnOrder(config=ColumnOrderConfig())
        node.node_fit({"input": _pair(_frame())})
        shuffled = _frame()[["c", "a", "b"]]
        out_df, out_ctx = node.node_transform({"input": _pair(shuffled)})["output"]
        assert list(out_df.columns) == ["a", "b", "c"]
        assert out_ctx.columns == ["a", "b", "c"]
        assert out_ctx.dtypes == [
            np.dtype("float64"),
            np.dtype(object),
            np.dtype("int64"),
        ]
        assert out_ctx.categories == [NumericalData, CategoricalData, NumericalData]

    def test_explicit_order_drops_extra_columns(self) -> None:
        config = ColumnOrderConfig(
            running_config=ColumnOrderRunningConfig(column_order=["c", "a"])
        )
        out_df, out_ctx = ColumnOrder(config=config).node_fit_transform(
            {"input": _pair(_frame())}
        )["output"]
        assert list(out_df.columns) == ["c", "a"]
        assert out_ctx.columns == ["c", "a"]
        assert out_ctx.categories == [NumericalData, NumericalData]

    def test_strict_mode_rejects_extra_columns(self) -> None:
        config = ColumnOrderConfig(
            running_config=ColumnOrderRunningConfig(column_order=["a"], strict=True)
        )
        with pytest.raises(NodeInputError, match="unexpected"):
            ColumnOrder(config=config).node_fit_transform({"input": _pair(_frame())})

    def test_missing_column_raises(self) -> None:
        node = ColumnOrder(config=ColumnOrderConfig())
        node.node_fit({"input": _pair(_frame())})
        with pytest.raises(NodeInputError, match="missing"):
            node.node_transform({"input": _pair(_frame()[["a", "b"]])})

    def test_no_hyperparameter_space(self) -> None:
        # The node has no hyperparameters, so the tuner must skip it.
        assert not hasattr(ColumnOrder, "hyperparameter_space")

    def test_set_params_restores_the_node(self) -> None:
        fitted = ColumnOrder(config=ColumnOrderConfig())
        fitted.node_fit({"input": _pair(_frame())})
        reborn = ColumnOrder(config=ColumnOrderConfig())
        reborn.set_params(fitted.get_params())
        shuffled = _frame()[["b", "c", "a"]]
        expected = fitted.node_transform({"input": _pair(shuffled)})["output"]
        actual = reborn.node_transform({"input": _pair(shuffled)})["output"]
        pd.testing.assert_frame_equal(actual[0], expected[0])
        assert actual[1] == expected[1]
