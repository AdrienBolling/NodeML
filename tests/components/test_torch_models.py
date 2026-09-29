"""Tests for the PyTorch model nodes ``MLP`` and ``CNN``.

All tests use tiny networks, few epochs and seeded training, so that they
are fast and deterministic.  The CUDA tests run only on a machine with a
GPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError

from nodeml.components.nodes.models._torch_base import TorchModelNode
from nodeml.components.nodes.models.cnn import CNNMetadata
from nodeml.components.nodes.models.mlp import MLPMetadata
from nodeml.core.common.data.data import NumericalData, TabularDataContext
from nodeml.core.common.exceptions import (
    NodeConfigError,
    NodeInputError,
    NodeNotFittedError,
)
from nodeml.core.nodes.models.model import ModelConfig
from nodeml.core.nodes.registry.node_registry import NODE_REGISTRY
from nodeml.core.pipeline.pipeline import Pipeline
from nodeml.core.pipeline.runners.smart_runner import SmartRunner, SmartRunnerConfig
from tests.shims.pipelines import build_source_model_sink_pipeline
from tests.shims.tabular import numerical_context

NETWORKS = ["MLP", "CNN"]
needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

_SMALL_HYPERPARAMETERS: dict[str, dict[str, Any]] = {
    "MLP": {"hidden_layers": 1, "hidden_units": 8},
    "CNN": {"num_conv_layers": 1, "num_filters": 4, "fc_units": 8},
}

type _Pair = tuple[pd.DataFrame, TabularDataContext]


def _config(
    name: str,
    hyperparameters: dict[str, Any] | None = None,
    **running: Any,
) -> ModelConfig:
    config_class = NODE_REGISTRY.get_node_config_class(name)
    return config_class.model_validate(
        {
            "hyperparameters": {
                **_SMALL_HYPERPARAMETERS[name],
                **(hyperparameters or {}),
            },
            "running_config": {
                "epochs": 3,
                "batch_size": 8,
                "random_state": 0,
                **running,
            },
        }
    )


def _node(
    name: str, hyperparameters: dict[str, Any] | None = None, **running: Any
) -> TorchModelNode:
    config = _config(name, hyperparameters, **running)
    return NODE_REGISTRY.get_node_class(name)(config=config)


def _pairs(n_rows: int = 32) -> tuple[_Pair, _Pair]:
    """Return float features and an int64 target."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.standard_normal((n_rows, 6)), columns=[f"x{i}" for i in range(6)]
    )
    y = pd.DataFrame({"t": np.round(2 * X["x0"] - X["x1"]).astype(np.int64)})
    return (X, numerical_context(X)), (y, numerical_context(y))


def _tensor_inputs(
    X_pair: _Pair, y_pair: _Pair | None = None, device: str = "cpu"
) -> dict[str, tuple[torch.Tensor, TabularDataContext]]:
    data = {"X": (torch.from_numpy(X_pair[0].to_numpy()).to(device), X_pair[1])}
    if y_pair is not None:
        data["y"] = (torch.from_numpy(y_pair[0].to_numpy()).to(device), y_pair[1])
    return data


def _fitted(name: str, **running: Any) -> TorchModelNode:
    X_pair, y_pair = _pairs()
    node = _node(name, **running)
    node.fit(_tensor_inputs(X_pair, y_pair))
    return node


def _predict(node: TorchModelNode, device: str = "cpu") -> torch.Tensor:
    X_pair, _ = _pairs()
    return node.predict(_tensor_inputs(X_pair, device=device))["pred"][0]


def _state(node: TorchModelNode) -> dict[str, torch.Tensor]:
    return node.get_params()["model_state_dict"]


def _assert_same_state(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> None:
    assert first.keys() == second.keys()
    for key, value in first.items():
        torch.testing.assert_close(value, second[key], rtol=0, atol=0)


def _pipeline(name: str) -> Pipeline:
    return build_source_model_sink_pipeline(
        model_node_type=name, model_config=_config(name)
    )


def _runner(pipeline: Pipeline) -> SmartRunner:
    return SmartRunner(pipeline, config=SmartRunnerConfig(verbose=False))


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


class TestPredictions:
    @pytest.mark.parametrize("name", NETWORKS)
    def test_int_target_gives_float64_predictions(self, name: str) -> None:
        X_pair, y_pair = _pairs()
        runner = _runner(_pipeline(name))
        runner.train(input_data={"source": {"X": X_pair, "y": y_pair}})
        pred_df, pred_ctx = runner.infer(input_data={"source": {"X": X_pair}})["pred"]

        assert list(pred_df.columns) == ["t"]
        assert pred_ctx.dtypes == [np.dtype(np.float64)]
        assert pred_ctx.categories == [NumericalData]
        values = pred_df.to_numpy()
        # The runner must not round the predictions to the int target dtype.
        assert not np.allclose(values, np.round(values))

    @pytest.mark.parametrize("name", NETWORKS)
    def test_predictions_are_float64_on_cpu(self, name: str) -> None:
        pred = _predict(_fitted(name))
        assert pred.dtype == torch.float64
        assert pred.device.type == "cpu"
        assert pred.shape == (32, 1)

    @pytest.mark.parametrize("name", NETWORKS)
    def test_predict_before_fit_raises(self, name: str) -> None:
        with pytest.raises(NodeNotFittedError, match="not fitted"):
            _predict(_node(name))

    @pytest.mark.parametrize("name", NETWORKS)
    def test_predict_with_another_feature_count_raises(self, name: str) -> None:
        node = _fitted(name)
        X_pair, _ = _pairs()
        X = X_pair[0].iloc[:, :4]
        with pytest.raises(NodeInputError, match="6 features"):
            node.predict(_tensor_inputs((X, numerical_context(X))))


class TestDocumentation:
    @pytest.mark.parametrize("metadata", [MLPMetadata(), CNNMetadata()])
    def test_description_says_regression_only(self, metadata) -> None:
        assert "regression" in metadata.description
        assert "classification" not in metadata.description


# ---------------------------------------------------------------------------
# Save and load
# ---------------------------------------------------------------------------


class TestSaveAndLoad:
    @pytest.mark.parametrize("name", NETWORKS)
    def test_set_params_on_a_new_node_gives_the_same_predictions(
        self, name: str
    ) -> None:
        node = _fitted(name)
        reborn = _node(name)
        reborn.set_params(node.get_params())
        torch.testing.assert_close(_predict(reborn), _predict(node), rtol=0, atol=0)

    @pytest.mark.parametrize("name", NETWORKS)
    def test_load_from_dir_then_infer_gives_the_same_predictions(
        self, name: str, tmp_path
    ) -> None:
        X_pair, y_pair = _pairs()
        original = _pipeline(name)
        runner = _runner(original)
        runner.train(input_data={"source": {"X": X_pair, "y": y_pair}})
        expected = runner.infer(input_data={"source": {"X": X_pair}})["pred"][0]
        original.save_config_to_dir(tmp_path)
        original.save_params_to_dir(tmp_path)

        loaded = Pipeline.load_from_dir(tmp_path)
        actual = _runner(loaded).infer(input_data={"source": {"X": X_pair}})["pred"][0]
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize("name", NETWORKS)
    def test_params_of_an_unfitted_node_give_an_unfitted_node(self, name: str) -> None:
        reborn = _node(name)
        reborn.set_params(_node(name).get_params())
        with pytest.raises(NodeNotFittedError):
            _predict(reborn)

    @pytest.mark.parametrize("name", NETWORKS)
    def test_params_without_the_network_size_raise(self, name: str) -> None:
        """NodeML 0.1 saved no network size, so these params cannot load."""
        params = _fitted(name).get_params()
        old_params = {
            "model_state_dict": params["model_state_dict"],
            "target_context": params["target_context"],
        }
        with pytest.raises(NodeInputError, match="Train the node again"):
            _node(name).set_params(old_params)

    @pytest.mark.parametrize("name", NETWORKS)
    def test_set_params_does_not_change_the_global_rng(self, name: str) -> None:
        params = _fitted(name).get_params()
        before = torch.random.get_rng_state()
        _node(name).set_params(params)
        assert torch.equal(torch.random.get_rng_state(), before)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


class TestSeeding:
    @pytest.mark.parametrize("name", NETWORKS)
    def test_init_does_not_change_the_global_rng(self, name: str) -> None:
        torch.rand(3)  # Move the global RNG away from any seeded state.
        before = torch.random.get_rng_state()
        _node(name)
        assert torch.equal(torch.random.get_rng_state(), before)

    @pytest.mark.parametrize("name", NETWORKS)
    def test_seeded_fit_does_not_change_the_global_rng(self, name: str) -> None:
        X_pair, y_pair = _pairs()
        node = _node(name)
        before = torch.random.get_rng_state()
        node.fit(_tensor_inputs(X_pair, y_pair))
        assert torch.equal(torch.random.get_rng_state(), before)

    @pytest.mark.parametrize("dropout", [0.0, 0.3])
    @pytest.mark.parametrize("name", NETWORKS)
    def test_a_second_fit_gives_the_same_weights(self, name: str, dropout) -> None:
        X_pair, y_pair = _pairs()
        node = _node(name, {"dropout": dropout})
        node.fit(_tensor_inputs(X_pair, y_pair))
        first = _state(node)
        torch.rand(5)  # Other code uses the global RNG between the fits.
        node.fit(_tensor_inputs(X_pair, y_pair))
        _assert_same_state(first, _state(node))

    @pytest.mark.parametrize("name", NETWORKS)
    def test_two_nodes_with_the_same_seed_give_the_same_weights(
        self, name: str
    ) -> None:
        X_pair, y_pair = _pairs()
        first, second = _node(name), _node(name)
        first.fit(_tensor_inputs(X_pair, y_pair))
        second.fit(_tensor_inputs(X_pair, y_pair))
        _assert_same_state(_state(first), _state(second))

    @pytest.mark.parametrize("name", NETWORKS)
    def test_unseeded_fits_differ(self, name: str) -> None:
        X_pair, y_pair = _pairs()
        node = _node(name, random_state=None)
        node.fit(_tensor_inputs(X_pair, y_pair))
        first = _state(node)
        node.fit(_tensor_inputs(X_pair, y_pair))
        second = _state(node)
        assert any(not torch.equal(first[key], second[key]) for key in first)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


class TestDevice:
    @pytest.mark.parametrize("name", NETWORKS)
    def test_device_defaults_to_cpu(self, name: str) -> None:
        config = NODE_REGISTRY.get_node_config_class(name)()
        assert config.running_config.device == "cpu"

    @pytest.mark.parametrize("name", NETWORKS)
    def test_unknown_device_is_rejected(self, name: str) -> None:
        with pytest.raises(ValidationError):
            _config(name, device="tpu")

    @pytest.mark.parametrize("name", NETWORKS)
    def test_auto_device_trains_and_returns_cpu_predictions(self, name: str) -> None:
        pred = _predict(_fitted(name, device="auto"))
        assert pred.device.type == "cpu"
        assert pred.dtype == torch.float64

    @pytest.mark.parametrize("name", NETWORKS)
    def test_cuda_device_without_cuda_raises(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(NodeConfigError, match="CUDA is not available"):
            _fitted(name, device="cuda")

    @needs_cuda
    @pytest.mark.parametrize("name", NETWORKS)
    def test_cuda_input_on_the_cpu_device(self, name: str) -> None:
        X_pair, y_pair = _pairs()
        node = _node(name)
        node.fit(_tensor_inputs(X_pair, y_pair, device="cuda"))
        pred = _predict(node, device="cuda")
        assert pred.device.type == "cpu"
        torch.testing.assert_close(pred, _predict(_fitted(name)))

    @needs_cuda
    @pytest.mark.parametrize("name", NETWORKS)
    def test_cuda_device_returns_cpu_predictions(self, name: str) -> None:
        node = _fitted(name, device="cuda")
        pred = _predict(node)
        assert pred.device.type == "cpu"
        assert all(t.device.type == "cpu" for t in _state(node).values())
        reborn = _node(name)
        reborn.set_params(node.get_params())
        torch.testing.assert_close(_predict(reborn), pred, rtol=1e-4, atol=1e-5)
