"""Tests for the :class:`TabularCSVFetcher` data-source node."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nodeml.components.nodes.data_sources.tabular_csv_fetcher import (
    TabularCSVFetcher,
    TabularCSVFetcherConfig,
    TabularCSVFetcherRunningConfig,
)
from nodeml.core.pipeline.pipeline import Edge, Pipeline


class TestTabularCSVFetcherLifecycle:
    def test_setup_and_fetch_returns_dataframe(
        self, csv_and_context_files, numerical_dataframe
    ) -> None:
        csv_path, ctx_path = csv_and_context_files

        cfg = TabularCSVFetcherConfig(
            running_config=TabularCSVFetcherRunningConfig(
                csv_path=str(csv_path),
                context_path=str(ctx_path),
            ),
        )
        node = TabularCSVFetcher(config=cfg)
        node.setup_source()

        out = node.fetch_data()
        df, ctx = out["output"]
        pd.testing.assert_frame_equal(df, numerical_dataframe)
        assert ctx.columns == list(numerical_dataframe.columns)

    def test_missing_csv_raises(self, tmp_path) -> None:
        cfg = TabularCSVFetcherConfig(
            running_config=TabularCSVFetcherRunningConfig(
                csv_path=str(tmp_path / "does_not_exist.csv"),
                context_path=str(tmp_path / "does_not_exist.json"),
            ),
        )
        node = TabularCSVFetcher(config=cfg)
        with pytest.raises(FileNotFoundError, match="CSV"):
            node.setup_source()

    def test_missing_context_raises(self, tmp_path, numerical_dataframe) -> None:
        csv_path = tmp_path / "data.csv"
        numerical_dataframe.to_csv(csv_path, index=False)
        cfg = TabularCSVFetcherConfig(
            running_config=TabularCSVFetcherRunningConfig(
                csv_path=str(csv_path),
                context_path=str(tmp_path / "missing.json"),
            ),
        )
        node = TabularCSVFetcher(config=cfg)
        with pytest.raises(FileNotFoundError, match="Context"):
            node.setup_source()

    def test_context_length_mismatch_raises(
        self, tmp_path, numerical_dataframe
    ) -> None:
        csv_path = tmp_path / "data.csv"
        ctx_path = tmp_path / "ctx.json"
        numerical_dataframe.to_csv(csv_path, index=False)
        # Context with fewer columns than the CSV.
        ctx_path.write_text(
            json.dumps(
                {
                    "columns": ["f0"],
                    "dtypes": ["float64"],
                    "categories": ["numerical_data"],
                }
            )
        )
        cfg = TabularCSVFetcherConfig(
            running_config=TabularCSVFetcherRunningConfig(
                csv_path=str(csv_path),
                context_path=str(ctx_path),
            ),
        )
        node = TabularCSVFetcher(config=cfg)
        with pytest.raises(ValueError, match="does not match"):
            node.setup_source()

    def test_fetch_before_setup_raises(self, tmp_path) -> None:
        cfg = TabularCSVFetcherConfig(
            running_config=TabularCSVFetcherRunningConfig(
                csv_path=str(tmp_path / "noop.csv"),
                context_path=str(tmp_path / "noop.json"),
            ),
        )
        node = TabularCSVFetcher(config=cfg)
        with pytest.raises(RuntimeError, match="Data not loaded"):
            node.fetch_data()


class TestTabularCSVFetcherPhases:
    def _write(self, tmp_path, name, df, ctx) -> tuple[Path, Path]:
        csv_path = tmp_path / f"{name}.csv"
        ctx_path = tmp_path / f"{name}.json"
        df.to_csv(csv_path, index=False)
        ctx_path.write_text(json.dumps(ctx.dump_dict))
        return csv_path, ctx_path

    def _pipeline(self, running_config) -> Pipeline:
        p = Pipeline(config=None)
        p.add_node(
            "source",
            "TabularCSVFetcher",
            TabularCSVFetcherConfig(running_config=running_config),
        )
        p.add_node("sink", "Sink")
        p.add_edge(Edge(source="source", target="sink", ports_map=[("output", "out")]))
        p.compile()
        return p

    def test_inference_reads_its_own_file(
        self, tmp_path, numerical_dataframe, numerical_context
    ) -> None:
        from nodeml.core.pipeline.runners.smart_runner import SmartRunner

        train_csv, ctx = self._write(
            tmp_path, "train", numerical_dataframe, numerical_context
        )
        infer_df = numerical_dataframe.head(5)
        infer_csv, _ = self._write(tmp_path, "infer", infer_df, numerical_context)
        rc = TabularCSVFetcherRunningConfig(
            csv_path=str(train_csv),
            context_path=str(ctx),
            inference_csv_path=str(infer_csv),
        )
        runner = SmartRunner(self._pipeline(rc))
        runner.train()
        out = runner.infer()["out"][0]
        assert out.shape == (5, numerical_dataframe.shape[1])

    def test_inference_works_without_training(
        self, tmp_path, numerical_dataframe, numerical_context
    ) -> None:
        from nodeml.core.pipeline.runners.smart_runner import SmartRunner

        csv, ctx = self._write(tmp_path, "data", numerical_dataframe, numerical_context)
        rc = TabularCSVFetcherRunningConfig(csv_path=str(csv), context_path=str(ctx))
        out = SmartRunner(self._pipeline(rc)).infer()["out"][0]
        assert out.shape == numerical_dataframe.shape

    def test_context_with_other_column_names_is_rejected(
        self, tmp_path, numerical_dataframe, numerical_context
    ) -> None:
        renamed = numerical_dataframe.rename(columns=lambda c: f"{c}_x")
        csv, ctx = self._write(tmp_path, "data", renamed, numerical_context)
        node = TabularCSVFetcher(
            config=TabularCSVFetcherConfig(
                running_config=TabularCSVFetcherRunningConfig(
                    csv_path=str(csv), context_path=str(ctx)
                )
            )
        )
        with pytest.raises(ValueError, match="does not match the data columns"):
            node.setup_source()
