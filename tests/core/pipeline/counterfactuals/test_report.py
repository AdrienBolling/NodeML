"""Tests for :mod:`nodeml.core.pipeline.counterfactuals.report`."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nodeml.core.pipeline.counterfactuals.config import (
    CounterfactualScenario,
    FeatureConstraints,
    TargetRange,
)
from nodeml.core.pipeline.counterfactuals.report import (
    CounterfactualRecord,
    CounterfactualReport,
    SampleResult,
    column_scales,
    frame_rows,
    score_counterfactual,
    summarize,
    to_json_value,
)

_ORIGINAL = {"a": 1.0, "b": 10.0, "c": 2, "color": "red"}


def _score(
    counterfactual: dict, prediction: float = 5.0, **constraints: object
) -> CounterfactualRecord:
    return score_counterfactual(
        0,
        _ORIGINAL,
        counterfactual,
        prediction,
        (4.0, 6.0),
        continuous=["a", "b", "c"],
        scales={"a": 2.0, "b": 10.0, "c": 4.0},
        constraints=FeatureConstraints(**constraints),
    )


class TestScoreCounterfactual:
    def test_changes_and_scaled_distances(self) -> None:
        record = _score({**_ORIGINAL, "a": 2.0, "b": 15.0, "color": "blue"})
        assert record.changes == {
            "a": [1.0, 2.0],
            "b": [10.0, 15.0],
            "color": ["red", "blue"],
        }
        assert record.n_changed == 3
        # |2-1|/2 + |15-10|/10; the categorical change adds no distance.
        assert record.l1_distance == pytest.approx(1.0)
        assert record.l2_distance == pytest.approx(math.sqrt(0.5))

    def test_validity_follows_the_target_range(self) -> None:
        assert _score(_ORIGINAL, prediction=5.0).valid is True
        assert _score(_ORIGINAL, prediction=7.0).valid is False
        assert _score(_ORIGINAL, prediction=float("nan")).valid is False

    def test_tiny_float_noise_is_not_a_change(self) -> None:
        record = _score({**_ORIGINAL, "b": 10.0 + 1e-12})
        assert record.changes == {}

    def test_constraint_violations(self) -> None:
        record = _score(
            {**_ORIGINAL, "c": 3, "b": 50.0, "color": "pink"},
            immutable_columns=["c"],
            feasible_ranges={"b": (0.0, 20.0)},
            feasible_categories={"color": ["red", "blue"]},
        )
        assert sorted(record.violations) == [
            "feasible:b",
            "feasible:color",
            "immutable:c",
        ]

    def test_values_are_json_safe(self) -> None:
        record = _score({**_ORIGINAL, "a": np.float64(3.0), "c": np.int64(2)})
        assert type(record.features["a"]) is float
        assert type(record.features["c"]) is int


class TestHelpers:
    def test_column_scales_use_the_range(self) -> None:
        df = pd.DataFrame({"a": [0.0, 4.0], "flat": [1.0, 1.0]})
        assert column_scales(df, ["a", "flat"]) == {"a": 4.0, "flat": 1.0}

    def test_frame_rows_keep_integer_types(self) -> None:
        df = pd.DataFrame({"a": [0.5], "c": np.array([3], dtype=np.int64)})
        assert frame_rows(df)[0]["c"] == 3
        assert isinstance(frame_rows(df)[0]["c"], (int, np.integer))

    def test_to_json_value(self) -> None:
        assert to_json_value(np.float64(1.5)) == 1.5
        assert to_json_value(float("inf")) is None
        assert to_json_value(pd.NA) is None


def _result(scenario: str, status: str, prediction: float | None = 5.0) -> SampleResult:
    records = (
        [] if prediction is None else [_score({**_ORIGINAL, "a": 2.0}, prediction)]
    )
    return SampleResult(
        scenario=scenario,
        method="dice",
        sample_id="0",
        original=_ORIGINAL,
        baseline_prediction=1.0,
        target_range=(4.0, 6.0),
        status=status,
        runtime_s=0.5,
        counterfactuals=records,
    )


def _report() -> CounterfactualReport:
    scenario = CounterfactualScenario(name="dice", target=TargetRange(low=3, high=5))
    return CounterfactualReport(
        pipeline_name="p",
        pipeline_version="0.1.0",
        pipeline_hash="h",
        task="regression",
        insertion=("source", "X"),
        output=("model", "pred", "t"),
        feature_columns=list(_ORIGINAL),
        continuous_columns=["a", "b", "c"],
        categorical_columns=["color"],
        constraints=FeatureConstraints(),
        scenarios=[scenario],
        results=[
            _result("dice", "found"),
            _result("dice", "not_found", prediction=None),
        ],
    )


class TestReport:
    def test_summary(self) -> None:
        (summary,) = _report().summaries
        assert summary.n_samples == 2
        assert summary.n_found == 1
        assert summary.n_not_found == 1
        assert summary.validity_rate == 1.0
        assert summary.mean_n_changed == 1.0
        assert summary.total_runtime_s == pytest.approx(1.0)

    def test_summary_without_counterfactuals(self) -> None:
        scenario = CounterfactualScenario(name="s", target=TargetRange(low=3, high=5))
        (summary,) = summarize([scenario], [_result("s", "error", prediction=None)])
        assert summary.validity_rate is None
        assert summary.mean_l1_distance is None
        assert summary.n_errors == 1

    def test_frames(self) -> None:
        report = _report()
        assert list(report.summary_frame()["scenario"]) == ["dice"]
        assert list(report.samples_frame()["status"]) == ["found", "not_found"]
        cf = report.counterfactuals_frame()
        assert len(cf) == 1
        assert cf.loc[0, "a"] == 2.0

    def test_changes_frame_marks_unchanged_values(self) -> None:
        frame = _report().changes_frame("dice", "0")
        assert list(frame.index) == ["original", "cf_0"]
        assert frame.loc["cf_0", "a"] == 2.0
        assert frame.loc["cf_0", "b"] == "-"
        with pytest.raises(KeyError):
            _report().changes_frame("dice", "missing")

    def test_json_round_trip(self, tmp_path) -> None:
        report = _report()
        path = report.save_json(tmp_path / "out" / "report.json")
        loaded = CounterfactualReport.load_json(path)
        assert loaded == report
