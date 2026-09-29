"""Tests for :class:`nodeml.core.nodes.registry.node_registry.NodeRegistry`.

These tests operate on a fresh :class:`NodeRegistry` instance so that they
cannot interact with the module-level :data:`NODE_REGISTRY` populated by
``nodeml`` auto-discovery.
"""

from __future__ import annotations

import pytest

from nodeml.core.nodes.registry.node_registry import NodeRegistry
from tests.shims.nodes import (
    IdentityTransform,
    IdentityTransformConfig,
    MeanModel,
    MeanModelConfig,
)


def _fresh_registry() -> NodeRegistry:
    """Return an isolated registry to keep tests independent of global state."""
    return NodeRegistry()


class TestNodeRegistration:
    def test_register_round_trips_class_and_config(self) -> None:
        reg = _fresh_registry()
        reg.register(
            name="Identity",
            node_class=IdentityTransform,
            node_config_class=IdentityTransformConfig,
        )
        assert reg.get_node_class("Identity") is IdentityTransform
        assert reg.get_node_config_class("Identity") is IdentityTransformConfig

    def test_register_stores_node_type_string(self) -> None:
        reg = _fresh_registry()
        reg.register(
            name="MeanModel",
            node_class=MeanModel,
            node_config_class=MeanModelConfig,
        )
        assert reg["MeanModel"]["node_type"] == "model"

    def test_duplicate_registration_raises(self) -> None:
        reg = _fresh_registry()
        reg.register(
            name="Identity",
            node_class=IdentityTransform,
            node_config_class=IdentityTransformConfig,
        )
        with pytest.raises(ValueError, match="already registered"):
            reg.register(
                name="Identity",
                node_class=IdentityTransform,
                node_config_class=IdentityTransformConfig,
            )


class TestGlobalRegistryHasExpectedComponents:
    """Sanity-check auto-discovery via ``nodeml`` import (fixture-triggered)."""

    def test_core_components_are_registered(self) -> None:
        from nodeml.core.nodes.registry.node_registry import NODE_REGISTRY

        # A minimal baseline of components that should always be present.
        expected = {
            "Sink",
            "InputsPassthrough",
            "TabularCSVFetcher",
            "StandardScaler",
            "LinearRegression",
            "MSE",
        }
        missing = expected - set(NODE_REGISTRY.keys())
        assert not missing, f"Missing registered components: {missing}"


class TestRegistryLookups:
    def test_register_does_not_instantiate_the_config(self) -> None:
        from pydantic import Field

        class _RequiredFieldConfig(IdentityTransformConfig):
            required: int = Field()

        reg = _fresh_registry()
        reg.register(
            name="Required",
            node_class=IdentityTransform,
            node_config_class=_RequiredFieldConfig,
        )
        assert reg.get("Required")["node_type"] == "transform"

    def test_unknown_name_raises_registry_error(self) -> None:
        from nodeml.core.common.exceptions import RegistryError

        reg = _fresh_registry()
        with pytest.raises(RegistryError, match="not registered"):
            reg.get_node_config_class("missing")
        with pytest.raises(RegistryError, match="not registered"):
            reg.get_node_class("missing")
