"""Tests for :mod:`nodeml.components.pipe_runner_wrappers.mlflow_logger`.

Every test uses a temporary MLflow tracking store (SQLite, one per module)
and its own experiment, with the artifacts in the temporary directory of
the test.  The working directory is the temporary directory too, so that
MLflow writes nothing into the repository.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient, fluent
from mlflow.tracking._tracking_service import utils as tracking_utils

from nodeml.components.pipe_runner_wrappers.mlflow_logger import (
    PIPELINE_ARTIFACT_DIR,
    MLFlowLoggerWrapper,
    MLFlowLoggerWrapperConfig,
    load_pipeline_from_run,
)
from nodeml.core.common.data.data import NumericalData, TabularDataContext
from nodeml.core.pipeline.pipeline import Pipeline
from nodeml.core.pipeline.runners.smart_runner import SmartRunner, SmartRunnerConfig
from tests.shims.pipelines import build_source_model_sink_pipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tracking_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Return a temporary SQLite tracking URI, shared by the tests of the module."""
    db_path = tmp_path_factory.mktemp("mlflow") / "mlflow.db"
    return f"sqlite:///{db_path}"


@pytest.fixture(autouse=True)
def _isolated_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Isolate the global MLflow state and the working directory."""
    monkeypatch.chdir(tmp_path)
    for key in [key for key in os.environ if key.startswith("MLFLOW_")]:
        monkeypatch.delenv(key)
    # monkeypatch restores this variable, also if MLflow sets it.
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    saved_tracking_uri = tracking_utils._tracking_uri
    yield
    tracking_utils._tracking_uri = saved_tracking_uri
    # A test can replace mlflow.active_run, so use the fluent module.
    while fluent.active_run() is not None:
        fluent.end_run()


@pytest.fixture
def client(tracking_uri: str) -> MlflowClient:
    return MlflowClient(tracking_uri=tracking_uri)


def _new_experiment(client: MlflowClient, tmp_path: Path, name: str) -> str:
    """Create an experiment whose artifacts go to *tmp_path*."""
    return client.create_experiment(
        name, artifact_location=(tmp_path / f"artifacts_{name}").as_uri()
    )


@pytest.fixture
def experiment_name(
    request: pytest.FixtureRequest, client: MlflowClient, tmp_path: Path
) -> str:
    """Create a new experiment for the test, named after the test."""
    name = request.node.name
    _new_experiment(client, tmp_path, name)
    return name


@pytest.fixture
def input_data(regression_dataset: Any) -> dict[str, Any]:
    X_pair, y_pair, _ = regression_dataset
    return {"source": {"X": X_pair, "y": y_pair}}


def _runner() -> SmartRunner:
    pipeline = build_source_model_sink_pipeline(with_metric=True)
    return SmartRunner(pipeline, config=SmartRunnerConfig(verbose=False))


def _wrapper(
    tracking_uri: str, experiment_name: str, **config: Any
) -> MLFlowLoggerWrapper:
    return MLFlowLoggerWrapper(
        _runner(),
        config=MLFlowLoggerWrapperConfig(
            tracking_uri=tracking_uri, experiment_name=experiment_name, **config
        ),
    )


def _run_id(wrapper: MLFlowLoggerWrapper) -> str:
    assert wrapper.run_id is not None
    return wrapper.run_id


# ---------------------------------------------------------------------------
# Experiment and global state
# ---------------------------------------------------------------------------


class TestExperiment:
    def test_each_wrapper_logs_into_its_own_experiment(
        self,
        tracking_uri: str,
        client: MlflowClient,
        tmp_path: Path,
        input_data: dict[str, Any],
    ) -> None:
        first_id = _new_experiment(client, tmp_path, "first")
        second_id = _new_experiment(client, tmp_path, "second")
        first = _wrapper(tracking_uri, "first", log_pipeline_graph=False)
        second = _wrapper(tracking_uri, "second", log_pipeline_graph=False)

        first.train(input_data)
        second.train(input_data)

        assert first.experiment_id == first_id
        assert client.get_run(_run_id(first)).info.experiment_id == first_id
        assert client.get_run(_run_id(second)).info.experiment_id == second_id

    def test_creates_a_missing_experiment(
        self, tracking_uri: str, client: MlflowClient
    ) -> None:
        wrapper = _wrapper(tracking_uri, "created_by_wrapper")
        experiment = client.get_experiment_by_name("created_by_wrapper")
        assert experiment is not None
        assert wrapper.experiment_id == experiment.experiment_id

    def test_does_not_change_global_mlflow_state(
        self, tracking_uri: str, experiment_name: str, input_data: dict[str, Any]
    ) -> None:
        uri_before = mlflow.get_tracking_uri()
        wrapper = _wrapper(tracking_uri, experiment_name, log_pipeline_graph=False)
        wrapper.train(input_data)
        wrapper.evaluate(input_data)

        assert mlflow.get_tracking_uri() == uri_before
        assert "MLFLOW_TRACKING_URI" not in os.environ
        assert mlflow.active_run() is None


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


class TestRunLifecycle:
    def test_successful_training_finishes_the_run(
        self,
        tracking_uri: str,
        client: MlflowClient,
        experiment_name: str,
        input_data: dict[str, Any],
    ) -> None:
        wrapper = _wrapper(
            tracking_uri, experiment_name, run_name="my_run", tags={"team": "a"}
        )
        wrapper.train(input_data)

        run = client.get_run(_run_id(wrapper))
        assert run.info.status == "FINISHED"
        assert run.info.run_name == "my_run"
        assert run.data.tags["pipeline_name"] == "test_pipeline"
        assert run.data.tags["pipeline_nodes"] == "source, model, sink, metric"
        assert run.data.tags["team"] == "a"
        assert "train_duration_s" not in run.data.metrics

    def test_train_time_is_logged_when_enabled(
        self,
        tracking_uri: str,
        client: MlflowClient,
        experiment_name: str,
        input_data: dict[str, Any],
    ) -> None:
        wrapper = _wrapper(tracking_uri, experiment_name, log_train_time=True)
        wrapper.train(input_data)
        assert "train_duration_s" in client.get_run(_run_id(wrapper)).data.metrics

    def test_failed_training_marks_the_run_failed(
        self, tracking_uri: str, client: MlflowClient, experiment_name: str
    ) -> None:
        wrapper = _wrapper(tracking_uri, experiment_name)
        with pytest.raises(ValueError, match="missing required inputs"):
            wrapper.train()  # The source needs input data.
        assert client.get_run(_run_id(wrapper)).info.status == "FAILED"

    def test_evaluate_does_not_change_a_failed_run(
        self,
        tracking_uri: str,
        client: MlflowClient,
        experiment_name: str,
        input_data: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        wrapper = _wrapper(tracking_uri, experiment_name)
        wrapper.runner.train(input_data)  # The runner itself is trained.

        def failing_train(**_kwargs: Any) -> None:
            message = "training failed"
            raise RuntimeError(message)

        monkeypatch.setattr(wrapper.runner, "train", failing_train)
        with pytest.raises(RuntimeError, match="training failed"):
            wrapper.train(input_data)
        failed_run_id = _run_id(wrapper)

        wrapper.evaluate(input_data)

        failed_run = client.get_run(failed_run_id)
        assert failed_run.info.status == "FAILED"
        assert "metric" not in failed_run.data.metrics
        assert wrapper.run_id != failed_run_id
        eval_run = client.get_run(_run_id(wrapper))
        assert eval_run.info.status == "FINISHED"
        assert "metric" in eval_run.data.metrics

    def test_evaluate_without_train_creates_a_run(
        self,
        tracking_uri: str,
        client: MlflowClient,
        experiment_name: str,
        input_data: dict[str, Any],
    ) -> None:
        wrapper = _wrapper(tracking_uri, experiment_name)
        wrapper.runner.train(input_data)
        wrapper.evaluate(input_data)

        run = client.get_run(_run_id(wrapper))
        assert run.info.status == "FINISHED"
        assert run.data.tags["pipeline_name"] == "test_pipeline"
        assert "metric" in run.data.metrics

    def test_nested_run_has_the_active_run_as_parent(
        self,
        tracking_uri: str,
        client: MlflowClient,
        experiment_name: str,
        input_data: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        experiment = client.get_experiment_by_name(experiment_name)
        assert experiment is not None
        parent = client.create_run(experiment.experiment_id)
        monkeypatch.setattr(mlflow, "active_run", lambda: parent)

        wrapper = _wrapper(
            tracking_uri, experiment_name, nested=True, log_pipeline_graph=False
        )
        wrapper.train(input_data)

        tags = client.get_run(_run_id(wrapper)).data.tags
        assert tags["mlflow.parentRunId"] == parent.info.run_id


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


class TestArtifacts:
    def test_pipeline_artifacts_are_in_one_fixed_directory(
        self,
        tracking_uri: str,
        client: MlflowClient,
        experiment_name: str,
        input_data: dict[str, Any],
    ) -> None:
        wrapper = _wrapper(tracking_uri, experiment_name)
        wrapper.train(input_data)

        run_id = _run_id(wrapper)
        assert [a.path for a in client.list_artifacts(run_id)] == [
            PIPELINE_ARTIFACT_DIR
        ]
        names = sorted(
            Path(a.path).name
            for a in client.list_artifacts(run_id, PIPELINE_ARTIFACT_DIR)
        )
        assert names == [
            "test_pipeline_v0.1.0_config.json",
            "test_pipeline_v0.1.0_graph.html",
            "test_pipeline_v0.1.0_params.pkl",
        ]

    def test_load_pipeline_from_run(
        self,
        tracking_uri: str,
        experiment_name: str,
        input_data: dict[str, Any],
    ) -> None:
        wrapper = _wrapper(tracking_uri, experiment_name, log_pipeline_graph=False)
        wrapper.train(input_data)
        expected = wrapper.infer({"source": {"X": input_data["source"]["X"]}})

        loaded = load_pipeline_from_run(_run_id(wrapper), tracking_uri=tracking_uri)

        assert isinstance(loaded, Pipeline)
        preds = SmartRunner(loaded, config=SmartRunnerConfig(verbose=False)).infer(
            {"source": {"X": input_data["source"]["X"]}}
        )
        np.testing.assert_allclose(
            preds["pred"][0].to_numpy(), expected["pred"][0].to_numpy()
        )

    def test_artifact_failure_before_training_is_a_warning(
        self,
        tracking_uri: str,
        client: MlflowClient,
        experiment_name: str,
        input_data: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        wrapper = _wrapper(tracking_uri, experiment_name)

        def broken_render(*_args: Any, **_kwargs: Any) -> None:
            message = "no renderer"
            raise RuntimeError(message)

        monkeypatch.setattr(wrapper.runner.pipeline, "save_html_to_dir", broken_render)
        with caplog.at_level(logging.WARNING, logger="nodeml"):
            wrapper.train(input_data)

        run = client.get_run(_run_id(wrapper))
        assert run.info.status == "FINISHED"
        assert "Pipeline artifact not logged" in caplog.text
        assert wrapper.runner.get_params()  # The model is trained.

    def test_params_failure_after_training_keeps_the_run_finished(
        self,
        tracking_uri: str,
        client: MlflowClient,
        experiment_name: str,
        input_data: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        wrapper = _wrapper(tracking_uri, experiment_name, log_pipeline_graph=False)

        def broken_save(*_args: Any, **_kwargs: Any) -> None:
            message = "disk full"
            raise OSError(message)

        monkeypatch.setattr(wrapper.runner, "save_params_to_dir", broken_save)
        with caplog.at_level(logging.WARNING, logger="nodeml"):
            wrapper.train(input_data)

        assert client.get_run(_run_id(wrapper)).info.status == "FINISHED"
        warnings = [
            record
            for record in caplog.records
            if record.getMessage() == "Pipeline artifact not logged"
        ]
        assert len(warnings) == 1
        assert warnings[0].context["artifact"] == "params"
        assert "disk full" in warnings[0].context["error"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_each_evaluate_call_has_its_own_step(
        self,
        tracking_uri: str,
        client: MlflowClient,
        experiment_name: str,
        input_data: dict[str, Any],
    ) -> None:
        wrapper = _wrapper(tracking_uri, experiment_name, log_pipeline_graph=False)
        wrapper.train(input_data)
        wrapper.evaluate(input_data)
        wrapper.evaluate(input_data)

        history = client.get_metric_history(_run_id(wrapper), "metric")
        assert sorted(m.step for m in history) == [0, 1]
        run = client.get_run(_run_id(wrapper))
        assert run.info.status == "FINISHED"

    def test_prefix_separates_the_metric_keys(
        self,
        tracking_uri: str,
        client: MlflowClient,
        experiment_name: str,
        input_data: dict[str, Any],
    ) -> None:
        wrapper = _wrapper(tracking_uri, experiment_name, log_pipeline_graph=False)
        wrapper.train(input_data)
        wrapper.evaluate(input_data, prefix="val")
        wrapper.evaluate(input_data, prefix="test")

        metrics = client.get_run(_run_id(wrapper)).data.metrics
        assert {
            "val/metric",
            "val/evaluate_duration_s",
            "test/metric",
            "test/evaluate_duration_s",
        } <= set(metrics)
        assert "metric" not in metrics

    def test_multi_value_metric_has_one_key_per_value(
        self,
        tracking_uri: str,
        client: MlflowClient,
        experiment_name: str,
        input_data: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        wrapper = _wrapper(tracking_uri, experiment_name, log_pipeline_graph=False)
        wrapper.train(input_data)
        frame = pd.DataFrame([[0.1, 0.3]], columns=["score_0", "score_1"])
        ctx = TabularDataContext(
            columns=["score_0", "score_1"],
            dtypes=[np.dtype("float64")] * 2,
            categories=[NumericalData] * 2,
        )
        monkeypatch.setattr(
            wrapper.runner, "evaluate", lambda **_kwargs: {"mse": (frame, ctx)}
        )
        returned = wrapper.evaluate(input_data)

        metrics = client.get_run(_run_id(wrapper)).data.metrics
        assert metrics["mse.score_0"] == pytest.approx(0.1)
        assert metrics["mse.score_1"] == pytest.approx(0.3)
        assert returned["mse"][0] is frame


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_env_file_may_hold_other_mlflow_variables(self, tmp_path: Path) -> None:
        env_file = tmp_path / "mlflow.env"
        env_file.write_text(
            "MLFLOW_EXPERIMENT_NAME=from_file\n"
            "MLFLOW_TRACKING_USERNAME=someone\n"
            "MLFLOW_LOG_TRAIN_TIME=true\n"
        )
        config = MLFlowLoggerWrapperConfig(env_file=str(env_file))
        assert config.experiment_name == "from_file"
        assert config.log_train_time is True

    def test_unknown_keyword_raises(self) -> None:
        with pytest.raises(TypeError, match="experment_name"):
            MLFlowLoggerWrapperConfig(experment_name="typo")

    def test_explicit_value_beats_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "from_env")
        assert MLFlowLoggerWrapperConfig().experiment_name == "from_env"
        config = MLFlowLoggerWrapperConfig(experiment_name="explicit")
        assert config.experiment_name == "explicit"
