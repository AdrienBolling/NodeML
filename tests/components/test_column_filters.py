"""Tests for the column filters (correlation, missing rate and variance)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nodeml.components.nodes.transforms.feature_selection.correlation_filter import (
    CorrelationFilter,
    CorrelationFilterConfig,
    CorrelationFilterHyperParameters,
    CorrelationFilterRunningConfig,
)
from nodeml.components.nodes.transforms.feature_selection.missing_rate_filter import (
    MissingRateFilter,
    MissingRateFilterConfig,
    MissingRateFilterHyperParameters,
    MissingRateFilterRunningConfig,
)
from nodeml.components.nodes.transforms.feature_selection.variance_filter import (
    VarianceFilter,
    VarianceFilterConfig,
    VarianceFilterHyperParameters,
    VarianceFilterRunningConfig,
)
from nodeml.core.common.data.data import (
    CategoricalData,
    NumericalData,
    TabularDataContext,
)
from nodeml.core.common.exceptions import NodeInputError
from tests.shims.tabular import numerical_context

FILTERS = {
    "correlation": (
        CorrelationFilter,
        CorrelationFilterConfig,
        CorrelationFilterHyperParameters,
        CorrelationFilterRunningConfig,
    ),
    "missing_rate": (
        MissingRateFilter,
        MissingRateFilterConfig,
        MissingRateFilterHyperParameters,
        MissingRateFilterRunningConfig,
    ),
    "variance": (
        VarianceFilter,
        VarianceFilterConfig,
        VarianceFilterHyperParameters,
        VarianceFilterRunningConfig,
    ),
}


def _make(
    kind: str,
    hyperparameters: dict[str, object] | None = None,
    running_config: dict[str, object] | None = None,
) -> CorrelationFilter | MissingRateFilter | VarianceFilter:
    node_class, config_class, hp_class, rc_class = FILTERS[kind]
    config = config_class(
        hyperparameters=hp_class(**(hyperparameters or {})),
        running_config=rc_class(**(running_config or {})),
    )
    return node_class(config=config)


def _frame() -> pd.DataFrame:
    """Return a frame from which no filter removes a column by default."""
    return pd.DataFrame(
        {
            "keep_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "other": [3.0, 1.0, 4.0, 1.0, 5.0, 9.0],
            "keep_2": [2.0, 7.0, 1.0, 8.0, 2.0, 8.0],
        }
    )


@pytest.fixture(params=sorted(FILTERS))
def kind(request) -> str:
    return request.param


class TestFilteringColumnsKeepTheInputOrder:
    """Columns outside filtering_columns are kept, in the input order."""

    def test_output_and_context_follow_the_input_order(self, kind) -> None:
        df = _frame()
        node = _make(kind, running_config={"filtering_columns": ["keep_1", "other"]})
        out_df, out_ctx = node.node_fit_transform(
            {"input": (df, numerical_context(df))}
        )["output"]
        assert list(out_df.columns) == ["keep_1", "other", "keep_2"]
        assert out_ctx.columns == list(out_df.columns)

    def test_context_categories_follow_their_columns(self, kind) -> None:
        df = _frame()
        ctx = TabularDataContext(
            columns=list(df.columns),
            dtypes=list(df.dtypes),
            categories=[NumericalData, CategoricalData, NumericalData],
        )
        node = _make(kind, running_config={"filtering_columns": ["keep_1", "other"]})
        _, out_ctx = node.node_fit_transform({"input": (df, ctx)})["output"]
        assert dict(zip(out_ctx.columns, out_ctx.categories, strict=True)) == {
            "keep_1": NumericalData,
            "other": CategoricalData,
            "keep_2": NumericalData,
        }

    def test_transform_follows_the_order_of_its_own_input(self, kind) -> None:
        df = _frame()
        node = _make(kind)
        node.node_fit({"input": (df, numerical_context(df))})
        shuffled = df[["keep_2", "keep_1", "other"]]
        out_df, out_ctx = node.node_transform(
            {"input": (shuffled, numerical_context(shuffled))}
        )["output"]
        assert list(out_df.columns) == ["keep_2", "keep_1", "other"]
        assert out_ctx.columns == list(out_df.columns)

    def test_missing_fitted_column_raises(self, kind) -> None:
        df = _frame()
        node = _make(kind)
        node.node_fit({"input": (df, numerical_context(df))})
        part = df[["keep_1"]]
        with pytest.raises(NodeInputError, match="keep_2"):
            node.node_transform({"input": (part, numerical_context(part))})


class TestVarianceFilter:
    def test_default_threshold_removes_constant_columns(self) -> None:
        df = _frame().assign(const=0.1)
        node = _make("variance")
        out_df, _ = node.node_fit_transform({"input": (df, numerical_context(df))})[
            "output"
        ]
        assert "const" not in out_df.columns
        assert list(out_df.columns) == ["keep_1", "other", "keep_2"]

    def test_column_at_the_threshold_is_removed(self) -> None:
        df = pd.DataFrame({"a": [0.0, 2.0], "b": [0.0, 4.0]})  # variances 1 and 4
        node = _make("variance", hyperparameters={"threshold": 1.0})
        out_df, _ = node.node_fit_transform({"input": (df, numerical_context(df))})[
            "output"
        ]
        assert list(out_df.columns) == ["b"]


class TestCorrelationFilter:
    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_two_correlated_columns_keep_the_first(self, method) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.1]})
        node = _make("correlation", hyperparameters={"method": method})
        out_df, _ = node.node_fit_transform({"input": (df, numerical_context(df))})[
            "output"
        ]
        assert list(out_df.columns) == ["a"]

    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_one_column_is_kept(self, method) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        node = _make("correlation", hyperparameters={"method": method})
        out_df, _ = node.node_fit_transform({"input": (df, numerical_context(df))})[
            "output"
        ]
        assert list(out_df.columns) == ["a"]

    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_three_columns(self, method) -> None:
        rng = np.random.default_rng(0)
        base = rng.standard_normal(30)
        df = pd.DataFrame(
            {"a": base, "noise": rng.standard_normal(30), "b": base * 3.0 + 1.0}
        )
        node = _make("correlation", hyperparameters={"method": method})
        out_df, _ = node.node_fit_transform({"input": (df, numerical_context(df))})[
            "output"
        ]
        assert list(out_df.columns) == ["a", "noise"]

    def test_preferred_column_survives(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.1]})
        node = _make("correlation", running_config={"preferred_columns": ["b"]})
        out_df, _ = node.node_fit_transform({"input": (df, numerical_context(df))})[
            "output"
        ]
        assert list(out_df.columns) == ["b"]


class TestMissingRateFilter:
    def test_column_above_the_threshold_is_removed(self) -> None:
        df = _frame()
        df.loc[:3, "other"] = np.nan  # 4 of 6 values are missing.
        node = _make("missing_rate", hyperparameters={"threshold": 0.5})
        out_df, out_ctx = node.node_fit_transform(
            {"input": (df, numerical_context(df))}
        )["output"]
        assert list(out_df.columns) == ["keep_1", "keep_2"]
        assert out_ctx.columns == ["keep_1", "keep_2"]


class TestSaveLoad:
    def test_set_params_restores_the_node(self, kind) -> None:
        df = _frame().assign(const=1.0)
        df.loc[:4, "const"] = np.nan
        ctx = numerical_context(df)
        fitted = _make(kind)
        expected = fitted.node_fit_transform({"input": (df, ctx)})

        reborn = _make(kind)
        reborn.set_params(fitted.get_params())
        actual = reborn.node_transform({"input": (df, ctx)})
        pd.testing.assert_frame_equal(actual["output"][0], expected["output"][0])
        assert actual["output"][1] == expected["output"][1]
