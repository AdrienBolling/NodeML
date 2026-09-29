"""Tests for :mod:`nodeml.core.pipeline.counterfactuals.config`."""

from __future__ import annotations

import pytest

from nodeml.core.pipeline.counterfactuals.config import (
    CounterfactualEvaluatorConfig,
    CounterfactualScenario,
    TargetRange,
)


class TestTargetRange:
    def test_absolute(self) -> None:
        assert TargetRange(kind="absolute", low=1, high=2).resolve(10.0) == (1, 2)

    def test_delta(self) -> None:
        assert TargetRange(kind="delta", low=1, high=2).resolve(10.0) == (11.0, 12.0)

    def test_relative(self) -> None:
        low, high = TargetRange(kind="relative", low=0.1, high=0.2).resolve(10.0)
        assert low == pytest.approx(11.0)
        assert high == pytest.approx(12.0)

    def test_relative_keeps_the_order_for_a_negative_baseline(self) -> None:
        low, high = TargetRange(kind="relative", low=0.1, high=0.2).resolve(-10.0)
        assert low == pytest.approx(-12.0)
        assert high == pytest.approx(-11.0)

    def test_empty_range_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="low < high"):
            TargetRange(low=2, high=2)


class TestScenario:
    def test_default_method_is_dice(self) -> None:
        scenario = CounterfactualScenario(name="s", target=TargetRange(low=1, high=2))
        assert scenario.method == "dice"

    def test_unsupported_method_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="method"):
            CounterfactualScenario(
                name="s", method="bugdoc", target=TargetRange(low=1, high=2)
            )

    def test_classification_is_not_supported_yet(self) -> None:
        with pytest.raises(ValueError, match="task"):
            CounterfactualEvaluatorConfig(task="classification")
