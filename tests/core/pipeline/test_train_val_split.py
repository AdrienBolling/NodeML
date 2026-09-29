"""Tests for :mod:`nodeml.core.pipeline.validation.train_val_split`."""

from __future__ import annotations

import pytest

from nodeml.core.common.data.data import TabularData
from nodeml.core.pipeline.runners.smart_runner import SmartRunner
from nodeml.core.pipeline.validation import (
    TabularTrainValSplit,
    TabularTrainValSplitConfig,
)
from tests.shims.pipelines import build_source_model_sink_pipeline


def _splitter(**kwargs: object) -> TabularTrainValSplit:
    return TabularTrainValSplit(
        TabularTrainValSplitConfig(
            data_nodes=[("source", ["X"])],
            target_nodes=[("source", ["y"])],
            seed=0,
            **kwargs,
        )
    )


class TestTabularTrainValSplit:
    def test_output_can_be_passed_to_the_runner(self, regression_dataset) -> None:
        X_pair, y_pair, _ = regression_dataset
        train, val = _splitter(randomize=True).split(
            {"source": {"X": X_pair, "y": y_pair}}
        )
        runner = SmartRunner(build_source_model_sink_pipeline(with_metric=True))
        runner.train(input_data=train)
        metrics = runner.evaluate(input_data=val)
        assert "metric" in metrics

    def test_rows_stay_aligned_across_ports(self, regression_dataset) -> None:
        X_pair, y_pair, _ = regression_dataset
        train, val = _splitter(randomize=True).split(
            {"source": {"X": X_pair, "y": y_pair}}
        )
        n_rows = X_pair[0].shape[0]
        assert (
            train["source"]["X"][0].shape[0] + val["source"]["X"][0].shape[0] == n_rows
        )
        assert train["source"]["X"][0].shape[0] == train["source"]["y"][0].shape[0]

    def test_unreferenced_entries_are_copied(self, regression_dataset) -> None:
        X_pair, y_pair, _ = regression_dataset
        extra = {"other": {"Z": X_pair}}
        train, val = _splitter().split({"source": {"X": X_pair, "y": y_pair}, **extra})
        assert train["other"]["Z"] is X_pair
        assert val["other"]["Z"] is X_pair

    def test_tabular_data_values_are_accepted(self, regression_dataset) -> None:
        (X_df, X_ctx), (y_df, y_ctx), _ = regression_dataset
        data = {
            "source": {
                "X": TabularData(X_df, X_ctx.columns, X_ctx.dtypes, X_ctx.categories),
                "y": TabularData(y_df, y_ctx.columns, y_ctx.dtypes, y_ctx.categories),
            }
        }
        train, _ = _splitter(best_kldiv=True, kldiv_trials=5).split(data)
        df, ctx = train["source"]["X"]
        assert list(df.columns) == ctx.columns

    def test_row_count_mismatch_is_rejected(self, regression_dataset) -> None:
        X_pair, (y_df, y_ctx), _ = regression_dataset
        with pytest.raises(ValueError, match="Row count mismatch"):
            _splitter().split({"source": {"X": X_pair, "y": (y_df.iloc[:-1], y_ctx)}})
