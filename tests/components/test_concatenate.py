"""Tests for the RowConcatenate and FeatureConcatenate transform nodes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nodeml.components.nodes.transforms.operations.feature_concatenate import (
    FeatureConcatenate,
    FeatureConcatenateConfig,
    FeatureConcatenateRunningConfig,
)
from nodeml.components.nodes.transforms.operations.row_concatenate import (
    RowConcatenate,
    RowConcatenateConfig,
)
from nodeml.core.common.data.data import (
    CategoricalData,
    NumericalData,
    TabularDataContext,
)
from nodeml.core.common.exceptions import NodeInputError


def _pair(
    df: pd.DataFrame, category: type = NumericalData
) -> tuple[pd.DataFrame, TabularDataContext]:
    return df, TabularDataContext(
        columns=list(df.columns),
        dtypes=list(df.dtypes),
        categories=[category] * df.shape[1],
    )


def _feature_concatenate(*, check_row_count: bool = True) -> FeatureConcatenate:
    config = FeatureConcatenateConfig(
        running_config=FeatureConcatenateRunningConfig(check_row_count=check_row_count)
    )
    return FeatureConcatenate(config=config)


class TestFeatureConcatenate:
    def test_rows_pair_by_position_when_the_labels_differ_in_order(self) -> None:
        left = pd.DataFrame({"x": [1, 2, 3]}, index=[0, 1, 2])
        right = pd.DataFrame({"y": [10, 20, 30]}, index=[2, 1, 0])
        out_df, _ = _feature_concatenate().node_fit_transform(
            {"input_1": _pair(left), "input_2": _pair(right)}
        )["output"]
        assert out_df.to_dict("list") == {"x": [1, 2, 3], "y": [10, 20, 30]}
        assert list(out_df.index) == [0, 1, 2]

    def test_rows_pair_by_position_when_the_labels_differ(self) -> None:
        left = pd.DataFrame({"x": [1, 2, 3]}, index=["a", "b", "c"])
        right = pd.DataFrame({"y": [10, 20, 30]}, index=[5, 6, 7])
        out_df, out_ctx = _feature_concatenate().node_fit_transform(
            {"input_1": _pair(left), "input_2": _pair(right, CategoricalData)}
        )["output"]
        assert out_df.shape == (3, 2)
        assert list(out_df.index) == ["a", "b", "c"]
        assert out_df["y"].tolist() == [10, 20, 30]
        assert out_ctx.columns == ["x", "y"]
        assert out_ctx.categories == [NumericalData, CategoricalData]

    def test_unequal_row_counts_pad_with_nan_when_allowed(self) -> None:
        left = pd.DataFrame({"x": [1, 2, 3]}, index=[7, 8, 9])
        right = pd.DataFrame({"y": [10, 20]}, index=[0, 1])
        out_df, out_ctx = _feature_concatenate(
            check_row_count=False
        ).node_fit_transform({"input_1": _pair(left), "input_2": _pair(right)})[
            "output"
        ]
        assert list(out_df.index) == [7, 8, 9]
        assert out_df["y"].tolist()[:2] == [10.0, 20.0]
        assert np.isnan(out_df["y"].iloc[2])
        assert out_ctx.dtypes == list(out_df.dtypes)

    def test_unequal_row_counts_raise_by_default(self) -> None:
        left = pd.DataFrame({"x": [1, 2, 3]})
        right = pd.DataFrame({"y": [10, 20]})
        with pytest.raises(NodeInputError, match="row-count"):
            _feature_concatenate().node_fit_transform(
                {"input_1": _pair(left), "input_2": _pair(right)}
            )

    def test_duplicate_columns_raise(self) -> None:
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(NodeInputError, match="duplicate"):
            _feature_concatenate().node_fit_transform(
                {"input_1": _pair(df), "input_2": _pair(df)}
            )


class TestRowConcatenate:
    def test_rows_are_stacked_and_the_context_matches(self) -> None:
        top = pd.DataFrame({"c": ["a", "b"]}, index=[4, 5])
        bottom = pd.DataFrame({"c": ["c"]}, index=[4])
        out_df, out_ctx = RowConcatenate(
            config=RowConcatenateConfig()
        ).node_fit_transform(
            {
                "input_1": _pair(top, CategoricalData),
                "input_2": _pair(bottom, CategoricalData),
            }
        )["output"]
        assert out_df["c"].tolist() == ["a", "b", "c"]
        assert list(out_df.index) == [0, 1, 2]
        assert out_ctx.dtypes == list(out_df.dtypes)
        assert out_ctx.categories == [CategoricalData]

    def test_schema_mismatch_raises(self) -> None:
        with pytest.raises(NodeInputError, match="column mismatch"):
            RowConcatenate(config=RowConcatenateConfig()).node_fit_transform(
                {
                    "input_1": _pair(pd.DataFrame({"a": [1]})),
                    "input_2": _pair(pd.DataFrame({"b": [1]})),
                }
            )


class TestStatelessParams:
    @pytest.mark.parametrize(
        "node",
        [
            RowConcatenate(config=RowConcatenateConfig()),
            FeatureConcatenate(config=FeatureConcatenateConfig()),
        ],
    )
    def test_get_params_is_none_and_transform_needs_no_fit(self, node) -> None:
        assert node.get_params() is None
        reborn = type(node)(config=node.config)
        reborn.set_params(node.get_params())
        df = pd.DataFrame({"a": [1]})
        other = (
            df if isinstance(node, RowConcatenate) else df.rename(columns={"a": "b"})
        )
        out = reborn.node_transform({"input_1": _pair(df), "input_2": _pair(other)})
        assert set(out) == {"output"}
