"""Tests for the scikit-learn model nodes.

The nodes are ``LinearRegression``, ``RandomForestRegressor``,
``GradientBoostingRegressor``, ``RandomForestClassifier`` and
``GradientBoostingClassifier``.  All tests use small, seeded data.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nodeml.components.nodes.models._sklearn_base import SklearnModelNode
from nodeml.core.common.data.data import (
    CategoricalData,
    DataCategoryEnum,
    NumericalData,
    TabularDataContext,
)
from nodeml.core.common.exceptions import (
    DataTypeError,
    NodeInputError,
    NodeNotFittedError,
)
from nodeml.core.nodes.models.model import ModelConfig
from nodeml.core.nodes.registry.node_registry import NODE_REGISTRY
from nodeml.core.pipeline.pipeline import Pipeline
from nodeml.core.pipeline.runners.smart_runner import SmartRunner, SmartRunnerConfig
from tests.shims.pipelines import build_source_model_sink_pipeline
from tests.shims.tabular import numerical_context

REGRESSORS = ["LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor"]
CLASSIFIERS = ["RandomForestClassifier", "GradientBoostingClassifier"]
ALL_MODELS = REGRESSORS + CLASSIFIERS
SINGLE_TARGET_MODELS = [
    "GradientBoostingRegressor",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
]

type _Pair = tuple[pd.DataFrame, TabularDataContext]

# Small and seeded settings keep the tests fast and deterministic.
_OVERRIDES: dict[str, dict] = {
    "RandomForestRegressor": {
        "hyperparameters": {"n_estimators": 5},
        "running_config": {"random_state": 0},
    },
    "RandomForestClassifier": {
        "hyperparameters": {"n_estimators": 5},
        "running_config": {"random_state": 0},
    },
    "GradientBoostingRegressor": {
        "hyperparameters": {"n_estimators": 5},
        "running_config": {"random_state": 0},
    },
    "GradientBoostingClassifier": {
        "hyperparameters": {"n_estimators": 5},
        "running_config": {"random_state": 0},
    },
}


def _config(name: str) -> ModelConfig:
    config_class = NODE_REGISTRY.get_node_config_class(name)
    return config_class.model_validate(_OVERRIDES.get(name, {}))


def _node(name: str) -> SklearnModelNode:
    return NODE_REGISTRY.get_node_class(name)(config=_config(name))


def _features(n_rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((n_rows, 3)), columns=["a", "b", "c"])


def _targets(name: str, X: pd.DataFrame, n_classes: int = 3) -> pd.DataFrame:
    """Return int64 targets: class labels for classifiers, counts for regressors."""
    if name in CLASSIFIERS:
        bins = np.quantile(X["a"], np.linspace(0, 1, n_classes + 1)[1:-1])
        return pd.DataFrame({"label": np.digitize(X["a"], bins).astype(np.int64)})
    values = np.round(2 * X["a"] - X["b"]).astype(np.int64)
    return pd.DataFrame({"t": values})


def _pairs(name: str, n_classes: int = 3) -> tuple[_Pair, _Pair]:
    X = _features()
    y = _targets(name, X, n_classes)
    return (X, numerical_context(X)), (y, numerical_context(y))


def _numpy_inputs(
    X_pair: _Pair, y_pair: _Pair | None = None
) -> dict[str, tuple[Any, TabularDataContext]]:
    data = {"X": (X_pair[0].to_numpy(), X_pair[1])}
    if y_pair is not None:
        data["y"] = (y_pair[0].to_numpy(), y_pair[1])
    return data


def _pipeline(name: str) -> Pipeline:
    return build_source_model_sink_pipeline(
        model_node_type=name, model_config=_config(name)
    )


def _runner(pipeline: Pipeline) -> SmartRunner:
    return SmartRunner(pipeline, config=SmartRunnerConfig(verbose=False))


# ---------------------------------------------------------------------------
# Classifier output contract
# ---------------------------------------------------------------------------


class TestClassifierProbabilities:
    @pytest.mark.parametrize("n_classes", [2, 3])
    @pytest.mark.parametrize("name", CLASSIFIERS)
    def test_runner_returns_one_probability_column_per_class(
        self, name: str, n_classes: int
    ) -> None:
        X_pair, y_pair = _pairs(name, n_classes)
        runner = _runner(_pipeline(name))
        runner.train(input_data={"source": {"X": X_pair, "y": y_pair}})
        pred_df, pred_ctx = runner.infer(input_data={"source": {"X": X_pair}})["pred"]

        expected_columns = [f"proba_{label}" for label in range(n_classes)]
        assert list(pred_df.columns) == expected_columns
        assert pred_ctx.columns == expected_columns
        assert pred_ctx.categories == [NumericalData] * n_classes
        assert (pred_df.dtypes == np.float64).all()
        np.testing.assert_allclose(pred_df.sum(axis=1), 1.0)

    @pytest.mark.parametrize("name", CLASSIFIERS)
    def test_columns_follow_the_estimator_classes(self, name: str) -> None:
        X_pair, y_pair = _pairs(name)
        node = _node(name)
        node.fit(_numpy_inputs(X_pair, y_pair))
        pred, ctx = node.predict(_numpy_inputs(X_pair))["pred"]
        classes = node.get_params()["estimator"].classes_
        assert ctx.columns == [f"proba_{label}" for label in classes]
        assert pred.dtype == np.float64
        assert pred.shape == (len(X_pair[0]), len(classes))

    @pytest.mark.parametrize("name", CLASSIFIERS)
    def test_pred_port_declares_one_column_per_class(self, name: str) -> None:
        assert _node(name).out_ports["pred"].data_shape == "batch classes"


# ---------------------------------------------------------------------------
# Regression output contract
# ---------------------------------------------------------------------------


class TestRegressionPredictions:
    @pytest.mark.parametrize("name", REGRESSORS)
    def test_int_target_gives_float64_predictions(self, name: str) -> None:
        X_pair, y_pair = _pairs(name)
        runner = _runner(_pipeline(name))
        runner.train(input_data={"source": {"X": X_pair, "y": y_pair}})
        pred_df, pred_ctx = runner.infer(input_data={"source": {"X": X_pair}})["pred"]

        node = _node(name)
        node.fit(_numpy_inputs(X_pair, y_pair))
        expected, _ = node.predict(_numpy_inputs(X_pair))["pred"]

        assert list(pred_df.columns) == ["t"]
        assert pred_ctx.dtypes == [np.dtype(np.float64)]
        assert pred_ctx.categories == [NumericalData]
        np.testing.assert_allclose(pred_df.to_numpy(), expected)
        # The runner must not round the predictions to the int target dtype.
        assert not np.allclose(expected, np.round(expected))

    @pytest.mark.parametrize("name", ["LinearRegression", "RandomForestRegressor"])
    def test_multi_target_predictions_keep_the_target_names(self, name: str) -> None:
        X = _features()
        y = pd.DataFrame({"t1": X["a"] + X["b"], "t2": X["b"] - X["c"]})
        node = _node(name)
        node.fit(_numpy_inputs((X, numerical_context(X)), (y, numerical_context(y))))
        pred, ctx = node.predict(_numpy_inputs((X, numerical_context(X))))["pred"]
        assert pred.shape == (len(X), 2)
        assert ctx.columns == ["t1", "t2"]

    @pytest.mark.parametrize("name", REGRESSORS)
    def test_single_target_is_passed_as_a_vector(self, name: str) -> None:
        X_pair, y_pair = _pairs(name)
        node = _node(name)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            node.fit(_numpy_inputs(X_pair, y_pair))


class TestSingleTargetModels:
    @pytest.mark.parametrize("name", SINGLE_TARGET_MODELS)
    def test_ports_declare_one_target_column(self, name: str) -> None:
        node = _node(name)
        assert node.in_ports["y"].data_shape == "batch 1"

    def test_gradient_boosting_regressor_predicts_one_column(self) -> None:
        node = _node("GradientBoostingRegressor")
        assert node.out_ports["pred"].data_shape == "batch 1"

    @pytest.mark.parametrize("name", SINGLE_TARGET_MODELS)
    def test_multi_column_target_raises(self, name: str) -> None:
        X = _features()
        y = pd.DataFrame({"t1": np.arange(len(X)) % 2, "t2": np.arange(len(X)) % 3})
        node = _node(name)
        with pytest.raises(NodeInputError, match="one target column"):
            node.fit(
                _numpy_inputs((X, numerical_context(X)), (y, numerical_context(y)))
            )


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_MODELS)
def test_feature_port_accepts_only_numerical_data(name: str) -> None:
    assert _node(name).in_ports["X"].data_category == DataCategoryEnum.NUMERICAL


def test_runner_rejects_categorical_features() -> None:
    """scikit-learn cannot use strings, so the runner must stop them."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": ["u", "v", "u", "v"]})
    X_ctx = TabularDataContext(
        columns=["a", "b"],
        dtypes=list(X.dtypes),
        categories=[NumericalData, CategoricalData],
    )
    y = pd.DataFrame({"t": [1.0, 2.0, 3.0, 4.0]})
    runner = _runner(_pipeline("LinearRegression"))
    with pytest.raises(DataTypeError, match="target port 'X'"):
        runner.train(
            input_data={"source": {"X": (X, X_ctx), "y": (y, numerical_context(y))}}
        )


# ---------------------------------------------------------------------------
# Not fitted
# ---------------------------------------------------------------------------


class TestNotFitted:
    @pytest.mark.parametrize("name", ALL_MODELS)
    def test_predict_before_fit_raises(self, name: str) -> None:
        X_pair, _ = _pairs(name)
        with pytest.raises(NodeNotFittedError, match="not fitted"):
            _node(name).predict(_numpy_inputs(X_pair))

    @pytest.mark.parametrize("name", ALL_MODELS)
    def test_params_of_an_unfitted_node_give_an_unfitted_node(self, name: str) -> None:
        X_pair, _ = _pairs(name)
        reborn = _node(name)
        reborn.set_params(_node(name).get_params())
        with pytest.raises(NodeNotFittedError):
            reborn.predict(_numpy_inputs(X_pair))


# ---------------------------------------------------------------------------
# Save and load
# ---------------------------------------------------------------------------


class TestSaveAndLoad:
    @pytest.mark.parametrize("name", ALL_MODELS)
    def test_set_params_on_a_new_node_restores_predictions(self, name: str) -> None:
        X_pair, y_pair = _pairs(name)
        node = _node(name)
        node.fit(_numpy_inputs(X_pair, y_pair))
        expected, expected_ctx = node.predict(_numpy_inputs(X_pair))["pred"]

        reborn = _node(name)
        reborn.set_params(node.get_params())
        actual, actual_ctx = reborn.predict(_numpy_inputs(X_pair))["pred"]
        np.testing.assert_array_equal(actual, expected)
        assert actual_ctx == expected_ctx

    @pytest.mark.parametrize("name", ALL_MODELS)
    def test_load_from_dir_then_infer_gives_the_same_predictions(
        self, name: str, tmp_path
    ) -> None:
        X_pair, y_pair = _pairs(name)
        original = _pipeline(name)
        runner = _runner(original)
        runner.train(input_data={"source": {"X": X_pair, "y": y_pair}})
        expected = runner.infer(input_data={"source": {"X": X_pair}})["pred"][0]
        original.save_config_to_dir(tmp_path)
        original.save_params_to_dir(tmp_path)

        loaded = Pipeline.load_from_dir(tmp_path)
        actual = _runner(loaded).infer(input_data={"source": {"X": X_pair}})["pred"][0]
        pd.testing.assert_frame_equal(actual, expected)

    def test_set_params_reads_the_format_of_older_versions(self) -> None:
        """Params saved by NodeML 0.1.0 hold only the public fitted attributes."""
        X_pair, y_pair = _pairs("LinearRegression")
        node = _node("LinearRegression")
        node.fit(_numpy_inputs(X_pair, y_pair))
        estimator = node.get_params()["estimator"]
        old_params = {
            "fitted_params": {
                name: value
                for name, value in vars(estimator).items()
                if name.endswith("_") and not name.startswith("_")
            },
            "target_context": y_pair[1].dump_dict,
        }

        reborn = _node("LinearRegression")
        reborn.set_params(old_params)
        np.testing.assert_array_equal(
            reborn.predict(_numpy_inputs(X_pair))["pred"][0],
            node.predict(_numpy_inputs(X_pair))["pred"][0],
        )

    def test_set_params_rejects_an_estimator_of_another_type(self) -> None:
        X_pair, y_pair = _pairs("LinearRegression")
        node = _node("LinearRegression")
        node.fit(_numpy_inputs(X_pair, y_pair))
        with pytest.raises(NodeInputError, match="RandomForestRegressor"):
            _node("RandomForestRegressor").set_params(node.get_params())

    @pytest.mark.parametrize("name", ALL_MODELS)
    def test_a_new_fit_does_not_change_params_taken_before(self, name: str) -> None:
        X_pair, y_pair = _pairs(name)
        node = _node(name)
        node.fit(_numpy_inputs(X_pair, y_pair))
        params = node.get_params()
        expected, _ = node.predict(_numpy_inputs(X_pair))["pred"]

        X_other = X_pair[0] * 3.0 + 1.0
        node.fit(_numpy_inputs((X_other, X_pair[1]), y_pair))
        reborn = _node(name)
        reborn.set_params(params)
        np.testing.assert_array_equal(
            reborn.predict(_numpy_inputs(X_pair))["pred"][0], expected
        )
