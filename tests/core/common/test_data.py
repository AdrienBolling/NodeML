"""Tests for ``nodeml.core.common.data.data``.

Covers the :class:`TabularData` wrapper (conversion between pandas, numpy,
and torch backends) and the :class:`TabularDataContext` (serialisation,
column removal, round-trip through ``dump_dict`` / ``tabular_context_from_dict_dump``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from nodeml.core.common.data.data import (
    DATA_CATEGORY_MAPPING,
    INVERSE_DATA_CATEGORY_MAPPING,
    ArrayLikeEnum,
    CategoricalData,
    DataCategoryEnum,
    MixedData,
    NumericalData,
    TabularData,
    TabularDataContext,
    tabular_context_from_dict_dump,
)
from tests.shims import tabular as tabular_shim

# ---------------------------------------------------------------------------
# TabularDataContext
# ---------------------------------------------------------------------------


class TestTabularDataContext:
    """Exercises the :class:`TabularDataContext` serialisation contract."""

    def test_dump_dict_round_trips(self) -> None:
        df = tabular_shim.numerical_df()
        ctx = tabular_shim.numerical_context(df)

        dumped = ctx.dump_dict
        restored = tabular_context_from_dict_dump(dumped)

        assert restored.columns == ctx.columns
        assert [str(d) for d in restored.dtypes] == [str(d) for d in ctx.dtypes]
        assert restored.categories == ctx.categories

    def test_dump_tuple_returns_three_parallel_lists(self) -> None:
        df = tabular_shim.mixed_df()
        ctx = tabular_shim.mixed_context(df)

        cols, dtypes, cats = ctx.dump_tuple
        assert cols == ctx.columns
        assert dtypes == ctx.dtypes
        assert cats == ctx.categories

    def test_remove_columns_preserves_alignment(self) -> None:
        ctx = TabularDataContext(
            columns=["a", "b", "c"],
            dtypes=[np.dtype("float64"), np.dtype("int64"), np.dtype("object")],
            categories=[NumericalData, NumericalData, CategoricalData],
        )

        ctx.remove_columns(["b"])

        assert ctx.columns == ["a", "c"]
        assert [str(d) for d in ctx.dtypes] == ["float64", "object"]
        assert ctx.categories == [NumericalData, CategoricalData]

    def test_remove_columns_silently_ignores_unknown(self) -> None:
        ctx = TabularDataContext(
            columns=["a"],
            dtypes=[np.dtype("float64")],
            categories=[NumericalData],
        )
        ctx.remove_columns(["does_not_exist"])
        assert ctx.columns == ["a"]


# ---------------------------------------------------------------------------
# TabularData — construction, conversion, validation
# ---------------------------------------------------------------------------


class TestTabularDataFromPandas:
    """Covers :meth:`TabularData.from_pandas` and its category inference."""

    def test_infers_numerical_categories(self) -> None:
        df = tabular_shim.numerical_df()
        td = TabularData(
            data=df,
            columns=list(df.columns),
            dtypes=list(df.dtypes),
        )
        assert td.is_initialized
        assert td.category is NumericalData
        assert td.shape == df.shape

    def test_infers_categorical_for_object_column(self) -> None:
        df = tabular_shim.mixed_df()
        td = TabularData(
            data=df,
            columns=list(df.columns),
            dtypes=list(df.dtypes),
        )
        assert td.category is MixedData
        assert td.categories == [NumericalData, CategoricalData]

    def test_raises_when_categories_length_wrong(self) -> None:
        df = tabular_shim.numerical_df()
        with pytest.raises(ValueError, match="match number of columns"):
            TabularData(
                data=df,
                columns=list(df.columns),
                dtypes=list(df.dtypes),
                categories=[NumericalData],  # wrong length
            )


class TestTabularDataFromNumpy:
    """Covers :meth:`TabularData.from_numpy`."""

    def test_requires_categories_for_numpy(self) -> None:
        arr = np.zeros((4, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="Categories must be provided"):
            TabularData(
                data=arr,
                columns=["a", "b"],
                dtypes=[np.dtype("float64"), np.dtype("float64")],
                infer_categories=False,
            )

    def test_accepts_valid_numpy_input(self) -> None:
        arr = np.arange(12, dtype=np.float64).reshape(6, 2)
        td = TabularData(
            data=arr,
            columns=["a", "b"],
            dtypes=[np.dtype("float64"), np.dtype("float64")],
            categories=[NumericalData, NumericalData],
        )
        assert td.shape == (6, 2)
        assert td.category is NumericalData

    def test_rejects_non_2d_arrays(self) -> None:
        arr = np.zeros(8, dtype=np.float64)
        with pytest.raises(ValueError, match="must be 2D"):
            TabularData(
                data=arr,
                columns=["a"],
                dtypes=[np.dtype("float64")],
                categories=[NumericalData],
            )


class TestTabularDataConversions:
    """Round-trip conversions between pandas / numpy / torch backends."""

    def test_to_pandas_returns_original_frame(self) -> None:
        df = tabular_shim.numerical_df()
        td = TabularData(
            data=df,
            columns=list(df.columns),
            dtypes=list(df.dtypes),
        )
        out_df, out_ctx = td.to_pandas()
        pd.testing.assert_frame_equal(out_df, df)
        assert out_ctx.columns == list(df.columns)

    def test_to_numpy_matches_values(self) -> None:
        df = tabular_shim.numerical_df()
        td = TabularData(
            data=df,
            columns=list(df.columns),
            dtypes=list(df.dtypes),
        )
        arr, ctx = td.to_numpy()
        np.testing.assert_array_equal(arr, df.to_numpy())
        assert ctx.columns == list(df.columns)

    def test_to_tensor_produces_matching_values(self) -> None:
        df = tabular_shim.numerical_df()
        td = TabularData(
            data=df,
            columns=list(df.columns),
            dtypes=list(df.dtypes),
        )
        tensor, ctx = td.to_tensor()
        assert isinstance(tensor, torch.Tensor)
        np.testing.assert_array_equal(tensor.numpy(), df.to_numpy())
        assert ctx.columns == list(df.columns)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestCategoryMappings:
    """Spot-check the ``DataCategoryEnum`` ↔ ``DataCategory`` mappings."""

    def test_forward_and_inverse_are_consistent(self) -> None:
        for enum_value, cls in DATA_CATEGORY_MAPPING.items():
            assert INVERSE_DATA_CATEGORY_MAPPING[cls] == enum_value

    def test_enum_values_are_strings(self) -> None:
        assert str(DataCategoryEnum.NUMERICAL) == "numerical_data"
        assert str(DataCategoryEnum.CATEGORICAL) == "categorical_data"
        assert str(DataCategoryEnum.MIXED) == "mixed_data"

    def test_array_like_enum_values(self) -> None:
        assert ArrayLikeEnum.PANDAS == "pd.DataFrame"
        assert ArrayLikeEnum.NUMPY == "np.ndarray"
        assert ArrayLikeEnum.TORCH == "torch.Tensor"
