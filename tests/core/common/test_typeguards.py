"""Tests for the runtime typeguards in ``nodeml.core.common.typechecking``.

These tests lean heavily on the Node shims in :mod:`tests.shims.nodes` so
we exercise the typeguards against real Node instances (not Mocks).
"""

from __future__ import annotations

from nodeml.components.nodes.data_sources.inputs_passthrough import (
    InputsPassthrough,
    InputsPassthroughConfig,
)
from nodeml.core.common.typechecking.typeguards import (
    accepts_inputs_source_node,
    has_hyperparameters,
    has_hyperparameters_config,
    has_params,
    has_running_config,
    is_list,
)
from tests.shims.nodes import (
    ConstantSource,
    ConstantSourceConfig,
    IdentityTransform,
    IdentityTransformConfig,
    MeanModel,
    MeanModelConfig,
)


class TestHasParams:
    def test_true_for_transform_with_get_set_params(self) -> None:
        node = IdentityTransform(config=IdentityTransformConfig())
        assert has_params(node) is True

    def test_false_for_source_without_params_methods(self) -> None:
        node = ConstantSource(config=ConstantSourceConfig())
        assert has_params(node) is False


class TestHasHyperparameters:
    def test_false_for_empty_hyperparameters(self) -> None:
        """Shim ``IdentityTransform`` has an empty hyperparameters BaseModel."""
        node = IdentityTransform(config=IdentityTransformConfig())
        assert has_hyperparameters(node) is False

    def test_config_level_check_matches_node_check(self) -> None:
        cfg = IdentityTransformConfig()
        assert has_hyperparameters_config(cfg) == has_hyperparameters(
            IdentityTransform(config=cfg)
        )


class TestHasRunningConfig:
    def test_false_for_empty_running_config(self) -> None:
        node = MeanModel(config=MeanModelConfig())
        assert has_running_config(node) is False


class TestAcceptsInputsSourceNode:
    def test_true_for_inputs_passthrough(self) -> None:
        node = InputsPassthrough(config=InputsPassthroughConfig())
        assert accepts_inputs_source_node(node) is True

    def test_false_for_plain_data_source(self) -> None:
        node = ConstantSource(config=ConstantSourceConfig())
        assert accepts_inputs_source_node(node) is False


class TestIsList:
    def test_positive(self) -> None:
        assert is_list([1, 2, 3]) is True

    def test_negative(self) -> None:
        assert is_list((1, 2, 3)) is False
        assert is_list("abc") is False
        assert is_list(None) is False
