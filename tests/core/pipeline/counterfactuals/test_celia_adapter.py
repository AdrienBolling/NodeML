"""Tests for :mod:`nodeml.core.pipeline.counterfactuals.celia_adapter`."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from nodeml.core.common.data.data import (
    CategoricalData,
    MixedData,
    NumericalData,
    TabularDataContext,
)
from nodeml.core.common.exceptions import NodeInputError
from nodeml.core.pipeline.counterfactuals.celia_adapter import (
    CategoryCodes,
    CeliaView,
    _coerce_to_reference,
    fill_missing,
    guard_root_logger,
    split_columns,
)
from nodeml.core.pipeline.counterfactuals.config import FeatureConstraints

_CONTEXT = TabularDataContext(
    columns=["a", "c", "color"],
    dtypes=[np.dtype("float64"), np.dtype("int64"), np.dtype("object")],
    categories=[NumericalData, NumericalData, CategoricalData],
)


class TestGuardRootLogger:
    def test_restores_level_and_handlers(self) -> None:
        root = logging.getLogger()
        level, handlers = root.level, list(root.handlers)
        with guard_root_logger():
            logging.basicConfig(level=logging.DEBUG)
            logging.getLogger().setLevel(logging.DEBUG)
        assert root.level == level
        assert root.handlers == handlers

    def test_basic_config_does_nothing_inside_the_guard(self) -> None:
        root = logging.getLogger()
        saved = list(root.handlers)
        for handler in saved:  # pytest adds a capture handler to the root logger
            root.removeHandler(handler)
        try:
            with guard_root_logger():
                logging.basicConfig(level=logging.DEBUG)
                assert all(isinstance(h, logging.NullHandler) for h in root.handlers)
            assert root.handlers == []
        finally:
            for handler in saved:
                root.addHandler(handler)


class TestCoerceToReference:
    def test_integral_floats_go_back_to_int(self) -> None:
        df = pd.DataFrame({"color": ["red"], "a": [0.5], "c": [3.0]})
        out = _coerce_to_reference(df, _CONTEXT)
        assert list(out.columns) == ["a", "c", "color"]
        assert out["c"].dtype == np.int64

    def test_non_integral_floats_stay_float(self) -> None:
        out = _coerce_to_reference(
            pd.DataFrame({"a": [0.5], "c": [2.5], "color": ["red"]}), _CONTEXT
        )
        assert out["c"].dtype == np.float64

    def test_unknown_category_is_not_turned_into_nan(self) -> None:
        # A cast to a categorical dtype turns an unknown category into NaN.
        context = TabularDataContext(
            columns=["color"],
            dtypes=[pd.CategoricalDtype(["red", "blue"])],
            categories=[CategoricalData],
        )
        unknown = _coerce_to_reference(pd.DataFrame({"color": ["green"]}), context)
        assert unknown["color"].tolist() == ["green"]
        known = _coerce_to_reference(pd.DataFrame({"color": ["blue"]}), context)
        assert isinstance(known["color"].dtype, pd.CategoricalDtype)

    def test_missing_column_raises(self) -> None:
        with pytest.raises(NodeInputError, match="no columns"):
            _coerce_to_reference(pd.DataFrame({"a": [0.5]}), _CONTEXT)


class TestCategoryCodes:
    _REFERENCE = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "color": ["red", "blue", None], "n": [1, 2, 3]}
    )

    def _codes(self) -> CategoryCodes:
        return CategoryCodes(self._REFERENCE, ["color", "n"])

    def test_only_non_numeric_columns_get_codes(self) -> None:
        assert self._codes().columns == ["color"]

    def test_round_trip(self) -> None:
        codes = self._codes()
        encoded = codes.encode(self._REFERENCE)
        assert encoded["color"].dtype == np.int64
        assert list(encoded["color"]) == [1, 0, 2]  # blue, red, then missing
        decoded = codes.decode(encoded)
        assert decoded["color"].iloc[0] == "red"
        assert pd.isna(decoded["color"].iloc[2])
        assert list(decoded["n"]) == [1, 2, 3]

    def test_decode_rounds_and_clips_codes(self) -> None:
        decoded = self._codes().decode(pd.DataFrame({"color": [0.4, 7.0, -3.0]}))
        assert list(decoded["color"][:1]) == ["blue"]
        assert pd.isna(decoded["color"].iloc[1])  # clipped to the missing code
        assert decoded["color"].iloc[2] == "blue"

    def test_unknown_value_raises(self) -> None:
        with pytest.raises(NodeInputError, match="not in the reference rows"):
            self._codes().encode(pd.DataFrame({"color": ["green"]}))

    def test_constraints_get_codes(self) -> None:
        constraints = FeatureConstraints(
            feasible_categories={"color": ["red", "purple"]},
            feasible_ranges={"a": (0.0, 1.0)},
        )
        encoded = self._codes().encode_constraints(constraints)
        assert encoded.feasible_categories == {"color": [1]}
        assert encoded.feasible_ranges == {"a": (0.0, 1.0)}


def test_split_columns_resolves_mixed_columns_by_dtype() -> None:
    context = TabularDataContext(
        columns=["a", "b", "s"],
        dtypes=[np.dtype("float64"), np.dtype("int64"), np.dtype("object")],
        categories=[NumericalData, MixedData, MixedData],
    )
    assert split_columns(context) == (["a", "b"], ["s"])


class TestWithCelia:
    @pytest.fixture(autouse=True)
    def _celia(self) -> None:
        pytest.importorskip("celia")

    def test_import_keeps_the_root_logger(self) -> None:
        from nodeml.core.pipeline.counterfactuals.celia_adapter import import_celia

        root = logging.getLogger()
        level, handlers = root.level, list(root.handlers)
        import_celia()
        assert root.level == level
        assert root.handlers == handlers

    def test_make_celia_data_converts_feasible_ranges(self) -> None:
        from nodeml.core.pipeline.counterfactuals.celia_adapter import make_celia_data

        reference = pd.DataFrame(
            {"a": [0.0, 1.0], "c": [1, 2], "color": ["red", "blue"]}
        )
        data = make_celia_data(
            reference,
            np.array([1.0, 2.0]),
            target_name="t",
            continuous=["a", "c"],
            categorical=["color"],
            constraints=FeatureConstraints(
                immutable_columns=["c"],
                feasible_ranges={"a": (0.0, 5.0)},
                feasible_categories={"color": ["red", "blue"]},
            ),
        )
        assert data.feasible_values == {"a": (0.0, 5.0), "color": ["red", "blue"]}
        assert data.immutable_column_names == ["c"]
        assert list(data.targets) == [1.0, 2.0]


class TestCeliaView:
    _REFERENCE = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, 5.0],
            "b": [np.nan, np.nan, np.nan, np.nan],
            "color": ["red", "blue", None, "red"],
            "id": ["x1", "x2", "x3", "x4"],
        }
    )

    def _view(self, sample: pd.DataFrame, **kwargs: object) -> CeliaView:
        return CeliaView(
            self._REFERENCE,
            sample,
            continuous=["a", "b"],
            categorical=["color", "id"],
            **kwargs,
        )

    def test_frozen_columns(self) -> None:
        sample = pd.DataFrame(
            {"a": [np.nan], "b": [np.nan], "color": ["red"], "id": ["x1"]}
        )
        view = self._view(sample, frozen=["id"])
        # a: missing in the sample; b: missing in all reference rows; id: asked.
        assert view.frozen == ["a", "b", "id"]
        assert view.visible == ["color"]
        assert list(view.sample.columns) == ["color"]

    def test_reference_is_filled_for_celia(self) -> None:
        sample = pd.DataFrame(
            {"a": [2.0], "b": [np.nan], "color": ["red"], "id": ["x1"]}
        )
        view = self._view(sample, frozen=["id"])
        assert not view.reference.isna().any().any()
        assert view.reference["a"].tolist() == [1.0, 3.0, 3.0, 5.0]  # median 3.0
        assert view.reference["color"].tolist() == ["red", "blue", "red", "red"]

    def test_from_celia_restores_frozen_values_and_order(self) -> None:
        sample = pd.DataFrame(
            {"a": [2.0], "b": [np.nan], "color": ["red"], "id": ["x9"]}
        )
        view = self._view(sample, frozen=["id"], code_categories=True)
        assert view.sample["color"].dtype == np.int64
        rows = view.from_celia(pd.DataFrame({"color": [0, 1], "a": [7.0, 8.0]}))
        assert list(rows.columns) == ["a", "b", "color", "id"]
        assert rows["id"].tolist() == ["x9", "x9"]
        assert rows["color"].tolist() == ["blue", "red"]
        assert rows["b"].isna().all()

    def test_missing_values_can_be_an_error(self) -> None:
        sample = pd.DataFrame({"a": [2.0], "b": [1.0], "color": ["red"], "id": ["x1"]})
        with pytest.raises(NodeInputError, match="missing values"):
            self._view(sample, allow_missing=False)

    def test_constraints_drop_frozen_columns_and_reject_unknown_ones(self) -> None:
        sample = pd.DataFrame(
            {"a": [np.nan], "b": [1.0], "color": ["red"], "id": ["x1"]}
        )
        view = self._view(sample)
        reduced = view.constraints(
            FeatureConstraints(
                immutable_columns=["a", "color"], feasible_ranges={"a": (0, 1)}
            )
        )
        assert reduced.immutable_columns == ["color"]
        assert reduced.feasible_ranges == {}
        with pytest.raises(NodeInputError, match="not at the insertion point"):
            view.constraints(FeatureConstraints(immutable_columns=["missing"]))


def test_fill_missing_leaves_empty_columns() -> None:
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, np.nan], "s": ["x", None]})
    out = fill_missing(df, ["a", "b"])
    assert out["a"].tolist() == [1.0, 1.0]
    assert out["b"].isna().all()
    assert out["s"].tolist() == ["x", "x"]
