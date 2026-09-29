"""End-to-end tests for :class:`CounterfactualEvaluator` with CELIA."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from nodeml.core.common.exceptions import NodeConfigError
from nodeml.core.pipeline.counterfactuals import (
    CounterfactualEvaluator,
    CounterfactualEvaluatorConfig,
    CounterfactualReport,
    CounterfactualScenario,
    FeatureConstraints,
    TargetRange,
)
from tests.core.pipeline.counterfactuals.conftest import TrainedCase

pytest.importorskip("celia")


def _dice(name: str = "dice", **kwargs: object) -> CounterfactualScenario:
    return CounterfactualScenario(
        name=name,
        method="dice",
        target=kwargs.pop("target", TargetRange(kind="delta", low=2.0, high=4.0)),
        # Only the DiCE "random" method accepts random_seed.
        generate_kwargs=kwargs.pop(
            "generate_kwargs", {"total_CFs": 2, "random_seed": 0}
        ),
        **kwargs,
    )


def _evaluator(case: TrainedCase, **config: object) -> CounterfactualEvaluator:
    settings = {"use_ray": False, **config}
    return CounterfactualEvaluator.from_pipeline(
        case.pipeline, config=CounterfactualEvaluatorConfig(**settings)
    )


class TestDiceRegression:
    def test_finds_valid_counterfactuals(self, numeric_case: TrainedCase) -> None:
        report = _evaluator(numeric_case).evaluate(
            [_dice()],
            samples=numeric_case.X.iloc[:3],
            reference_input=numeric_case.reference_input,
        )
        assert [result.status for result in report.results] == ["found"] * 3
        for result in report.results:
            low, high = result.target_range
            assert low == pytest.approx(result.baseline_prediction + 2.0)
            assert len(result.counterfactuals) == 2
            for record in result.counterfactuals:
                assert record.valid
                assert low <= record.prediction <= high
        assert report.summaries[0].validity_rate == 1.0
        assert report.insertion == ("source", "X")
        assert report.output == ("model", "pred", "t")

    def test_predictions_in_the_report_come_from_the_pipeline(
        self, numeric_case: TrainedCase
    ) -> None:
        report = _evaluator(numeric_case).evaluate(
            [_dice()],
            samples=numeric_case.X.iloc[:1],
            reference_input=numeric_case.reference_input,
        )
        frame = report.counterfactuals_frame()
        expected = numeric_case.infer(frame[["a", "b", "c"]])
        assert list(frame["prediction"]) == pytest.approx(list(expected))

    def test_immutable_columns_do_not_change(self, numeric_case: TrainedCase) -> None:
        report = _evaluator(
            numeric_case, constraints=FeatureConstraints(immutable_columns=["c"])
        ).evaluate(
            [_dice()],
            samples=numeric_case.X.iloc[:3],
            reference_input=numeric_case.reference_input,
        )
        records = [r for result in report.results for r in result.counterfactuals]
        assert records
        assert all("c" not in record.changes for record in records)
        assert report.summaries[0].n_violations == 0

    @pytest.mark.parametrize("dice_method", ["random", "genetic", "kdtree"])
    def test_categorical_column_through_one_hot_encoding(
        self, mixed_case: TrainedCase, dice_method: str
    ) -> None:
        # The DiCE "random" method needs the integer category codes.
        report = _evaluator(
            mixed_case,
            constraints=FeatureConstraints(
                feasible_categories={"color": ["green", "blue"]}
            ),
        ).evaluate(
            [
                _dice(
                    target=TargetRange(kind="delta", low=3.0, high=5.0),
                    explainer_kwargs={"method": dice_method},
                    generate_kwargs={"total_CFs": 2}
                    | ({"random_seed": 0} if dice_method == "random" else {}),
                )
            ],
            samples=mixed_case.X.iloc[:2],
            reference_input=mixed_case.reference_input,
        )
        assert report.categorical_columns == ["color"]
        assert [result.status for result in report.results] == ["found", "found"]
        records = [r for res in report.results for r in res.counterfactuals]
        assert all(record.valid for record in records)
        assert {record.features["color"] for record in records} <= {"green", "blue"}
        assert report.summaries[0].n_violations == 0

    def test_already_in_range(self, numeric_case: TrainedCase) -> None:
        report = _evaluator(numeric_case).evaluate(
            [_dice(target=TargetRange(kind="delta", low=-1.0, high=1.0))],
            samples=numeric_case.X.iloc[:1],
            reference_input=numeric_case.reference_input,
        )
        assert report.results[0].status == "already_in_range"


class TestOtherMethodsAndFailures:
    def test_nnce(self, numeric_case: TrainedCase) -> None:
        scenario = CounterfactualScenario(
            name="nnce",
            method="nnce",
            target=TargetRange(kind="delta", low=2.0, high=4.0),
            generate_kwargs={"n_counterfactuals": 2},
        )
        report = _evaluator(numeric_case).evaluate(
            [scenario],
            samples=numeric_case.X.iloc[:2],
            reference_input=numeric_case.reference_input,
        )
        assert [result.status for result in report.results] == ["found", "found"]
        assert all(r.valid for res in report.results for r in res.counterfactuals)

    def test_nnce_with_a_categorical_column(self, mixed_case: TrainedCase) -> None:
        scenario = CounterfactualScenario(
            name="nnce",
            method="nnce",
            target=TargetRange(kind="delta", low=3.0, high=5.0),
            generate_kwargs={"n_counterfactuals": 2},
        )
        report = _evaluator(mixed_case).evaluate(
            [scenario],
            samples=mixed_case.X.iloc[:2],
            reference_input=mixed_case.reference_input,
        )
        assert [result.status for result in report.results] == ["found", "found"]
        colors = {
            r.features["color"] for res in report.results for r in res.counterfactuals
        }
        assert colors <= {"red", "green", "blue"}

    def test_a_failing_scenario_does_not_stop_the_others(
        self, numeric_case: TrainedCase
    ) -> None:
        broken = _dice(name="broken", explainer_kwargs={"method": "no-such-method"})
        report = _evaluator(numeric_case).evaluate(
            [broken, _dice()],
            samples=numeric_case.X.iloc[:1],
            reference_input=numeric_case.reference_input,
        )
        statuses = {result.scenario: result.status for result in report.results}
        assert statuses == {"broken": "error", "dice": "found"}
        assert report.results[0].message

    def test_sample_indices(self, numeric_case: TrainedCase) -> None:
        report = _evaluator(numeric_case).evaluate(
            [_dice()],
            sample_indices=[4, 5],
            reference_input=numeric_case.reference_input,
        )
        assert [result.sample_id for result in report.results] == ["4", "5"]
        assert report.results[0].original["a"] == pytest.approx(
            numeric_case.X.loc[4, "a"]
        )


class TestMissingValuesAndEmptyResults:
    def test_frozen_columns_do_not_change(self, numeric_case: TrainedCase) -> None:
        report = _evaluator(
            numeric_case, constraints=FeatureConstraints(frozen_columns=["c"])
        ).evaluate(
            [_dice()],
            samples=numeric_case.X.iloc[:1],
            reference_input=numeric_case.reference_input,
        )
        result = report.results[0]
        assert result.frozen_columns == ["c"]
        assert result.status == "found"
        assert all("c" not in record.changes for record in result.counterfactuals)

    def test_missing_sample_value_is_frozen(self, imputed_case: TrainedCase) -> None:
        samples = imputed_case.X.iloc[1:2].copy()
        samples.loc[samples.index[0], "a"] = np.nan
        report = _evaluator(imputed_case).evaluate(
            [_dice()], samples=samples, reference_input=imputed_case.reference_input
        )
        result = report.results[0]
        assert result.frozen_columns == ["a"]
        assert result.status == "found"
        for record in result.counterfactuals:
            assert record.features["a"] is None
            assert record.valid

    def test_missing_values_can_be_an_error(self, imputed_case: TrainedCase) -> None:
        report = _evaluator(imputed_case, missing_values="error").evaluate(
            [_dice()],
            samples=imputed_case.X.iloc[1:2],
            reference_input=imputed_case.reference_input,
        )
        assert report.results[0].status == "error"
        assert "missing values" in (report.results[0].message or "")

    def test_a_sample_that_the_pipeline_cannot_predict(
        self, numeric_case: TrainedCase
    ) -> None:
        samples = numeric_case.X.iloc[:2].copy()
        samples.loc[samples.index[0], "a"] = np.nan  # LinearRegression rejects NaN
        report = _evaluator(numeric_case).evaluate(
            [_dice()], samples=samples, reference_input=numeric_case.reference_input
        )
        statuses = [result.status for result in report.results]
        assert statuses == ["error", "found"]
        assert report.results[0].baseline_prediction is None
        assert "cannot predict" in (report.results[0].message or "")
        assert report.samples_frame()["target_low"].isna().iloc[0]

    def test_unreachable_target_is_not_found(self, numeric_case: TrainedCase) -> None:
        report = _evaluator(numeric_case).evaluate(
            [_dice(target=TargetRange(kind="absolute", low=1e6, high=2e6))],
            samples=numeric_case.X.iloc[:1],
            reference_input=numeric_case.reference_input,
        )
        assert report.results[0].status == "not_found"


class TestFailureIsolation:
    def test_a_scoring_failure_is_an_error_result(
        self, numeric_case: TrainedCase, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nodeml.core.pipeline.counterfactuals import evaluator as evaluator_module

        def broken(*args: object, **kwargs: object) -> None:
            _ = args, kwargs
            msg = "the pipeline rejects a generated value"
            raise ValueError(msg)

        monkeypatch.setattr(evaluator_module, "score_counterfactual", broken)
        report = _evaluator(numeric_case).evaluate(
            [_dice()],
            samples=numeric_case.X.iloc[:2],
            reference_input=numeric_case.reference_input,
        )
        assert [result.status for result in report.results] == ["error", "error"]
        assert "cannot be scored" in (report.results[0].message or "")

    def test_a_failing_task_does_not_stop_the_others(
        self, numeric_case: TrainedCase, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nodeml.core.pipeline.counterfactuals import evaluator as evaluator_module

        original = evaluator_module.explain_sample

        def flaky(context, scenario, sample_id, sample, baseline):  # noqa: ANN202
            if sample_id == "0":
                msg = "worker crashed"
                raise RuntimeError(msg)
            return original(context, scenario, sample_id, sample, baseline)

        monkeypatch.setattr(evaluator_module, "explain_sample", flaky)
        report = _evaluator(numeric_case).evaluate(
            [_dice()],
            samples=numeric_case.X.iloc[:2],
            reference_input=numeric_case.reference_input,
        )
        assert [result.status for result in report.results] == ["error", "found"]
        failed = report.results[0]
        assert "worker crashed" in (failed.message or "")
        assert failed.baseline_prediction is not None
        assert failed.target_range is not None


class TestInputChecks:
    def test_samples_or_indices(self, numeric_case: TrainedCase) -> None:
        evaluator = _evaluator(numeric_case)
        kwargs = {"reference_input": numeric_case.reference_input}
        with pytest.raises(NodeConfigError, match="exactly one"):
            evaluator.evaluate([_dice()], **kwargs)
        with pytest.raises(NodeConfigError, match="exactly one"):
            evaluator.evaluate(
                [_dice()], samples=numeric_case.X.iloc[:1], sample_indices=[0], **kwargs
            )
        with pytest.raises(NodeConfigError, match="outside"):
            evaluator.evaluate([_dice()], sample_indices=[10_000], **kwargs)
        with pytest.raises(NodeConfigError, match="no columns"):
            evaluator.evaluate(
                [_dice()], samples=numeric_case.X[["a"]].iloc[:1], **kwargs
            )

    def test_scenario_names_must_be_unique(self, numeric_case: TrainedCase) -> None:
        with pytest.raises(NodeConfigError, match="unique"):
            _evaluator(numeric_case).evaluate(
                [_dice(), _dice()],
                samples=numeric_case.X.iloc[:1],
                reference_input=numeric_case.reference_input,
            )

    def test_unknown_constraint_column_is_an_error_result(
        self, numeric_case: TrainedCase
    ) -> None:
        scenario = _dice(constraints=FeatureConstraints(immutable_columns=["missing"]))
        report = _evaluator(numeric_case).evaluate(
            [scenario],
            samples=numeric_case.X.iloc[:1],
            reference_input=numeric_case.reference_input,
        )
        assert report.results[0].status == "error"
        assert "missing" in (report.results[0].message or "")


class TestPipelineIsPreserved:
    def test_original_pipeline_and_params_do_not_change(
        self, numeric_case: TrainedCase
    ) -> None:
        pipeline = numeric_case.pipeline
        before_hash = pipeline.hash()
        before_nodes = set(pipeline.node_objects)
        before = numeric_case.infer(numeric_case.X.iloc[:5])
        _evaluator(numeric_case).evaluate(
            [_dice()],
            samples=numeric_case.X.iloc[:2],
            reference_input=numeric_case.reference_input,
        )
        assert pipeline.hash() == before_hash
        assert set(pipeline.node_objects) == before_nodes
        assert pipeline.compiled
        assert list(numeric_case.infer(numeric_case.X.iloc[:5])) == pytest.approx(
            list(before)
        )

    def test_from_dir(self, numeric_case: TrainedCase, tmp_path) -> None:
        numeric_case.pipeline.save_config_to_dir(tmp_path)
        numeric_case.pipeline.save_params_to_dir(tmp_path)
        evaluator = CounterfactualEvaluator.from_dir(
            tmp_path, config=CounterfactualEvaluatorConfig(use_ray=False)
        )
        report = evaluator.evaluate(
            [_dice()],
            samples=numeric_case.X.iloc[:1],
            reference_input=numeric_case.reference_input,
        )
        assert report.results[0].status == "found"

    def test_report_json_round_trip(self, numeric_case: TrainedCase, tmp_path) -> None:
        # A range is a tuple; JSON has no tuples, so this checks the round trip.
        constraints = FeatureConstraints(feasible_ranges={"b": (0.0, 10.0)})
        report = _evaluator(numeric_case, constraints=constraints).evaluate(
            [_dice()],
            samples=numeric_case.X.iloc[:1],
            reference_input=numeric_case.reference_input,
        )
        loaded = CounterfactualReport.load_json(report.save_json(tmp_path / "r.json"))
        assert loaded == report


@pytest.fixture
def ray_cluster() -> Iterator[None]:
    """Start a small local Ray cluster for one test, and stop it after."""
    import ray

    started = not ray.is_initialized()
    if started:
        ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    yield
    if started:
        ray.shutdown()


def test_ray_gives_the_same_results(
    numeric_case: TrainedCase, ray_cluster: None
) -> None:
    samples = numeric_case.X.iloc[:2]
    local = _evaluator(numeric_case).evaluate(
        [_dice()], samples=samples, reference_input=numeric_case.reference_input
    )
    remote = _evaluator(numeric_case, use_ray=True).evaluate(
        [_dice()], samples=samples, reference_input=numeric_case.reference_input
    )
    _ = ray_cluster

    def strip(report: CounterfactualReport) -> list:
        return [
            [record.model_dump() for record in result.counterfactuals]
            for result in report.results
        ]

    assert strip(remote) == strip(local)


def test_a_failed_ray_task_does_not_lose_the_other_results(
    numeric_case: TrainedCase, ray_cluster: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ray

    real_get = ray.get
    calls: list[int] = []

    def get_with_one_failure(ref: object, **kwargs: object) -> object:
        calls.append(1)
        if len(calls) == 1:
            msg = "task lost"
            raise RuntimeError(msg)
        return real_get(ref, **kwargs)

    monkeypatch.setattr(ray, "get", get_with_one_failure)
    report = _evaluator(numeric_case, use_ray=True).evaluate(
        [_dice()],
        samples=numeric_case.X.iloc[:2],
        reference_input=numeric_case.reference_input,
    )
    _ = ray_cluster
    assert [result.status for result in report.results] == ["error", "found"]
    assert "task lost" in (report.results[0].message or "")
