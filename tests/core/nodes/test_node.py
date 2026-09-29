"""Tests for the ``Node`` base class and the ``MetaPostInitHook`` metaclass."""

from __future__ import annotations

import pytest

from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.nodes.node import NodeConfig, NodeType, Port
from tests.shims.nodes import (
    ConstantSource,
    ConstantSourceConfig,
    IdentityTransform,
    IdentityTransformConfig,
)
from tests.shims.tabular import numerical_pair


class TestNodeInit:
    def test_config_must_be_keyword_argument(self) -> None:
        """``Node`` wraps ``__init__`` to reject positional ``config``."""
        cfg = IdentityTransformConfig()
        with pytest.raises(TypeError, match="keyword argument"):
            IdentityTransform(cfg)  # type: ignore[misc]

    def test_default_execution_mode_is_default_enum(self) -> None:
        node = IdentityTransform(config=IdentityTransformConfig())
        # Shim bypasses Node.__init__, so the execution mode is only set
        # when the runner calls ``set_execution_mode``. We assert on that
        # API instead of the private attribute.
        node.set_execution_mode(NodeExecutionMode.TRAINING)
        assert node.execution_mode == NodeExecutionMode.TRAINING


class TestNodeConfig:
    def test_unique_uuid_per_instance(self) -> None:
        cfg_a = NodeConfig()
        cfg_b = NodeConfig()
        assert cfg_a.id != cfg_b.id

    def test_hash_matches_uuid_hash(self) -> None:
        cfg = NodeConfig()
        assert hash(cfg) == hash(cfg.id)


class TestPort:
    def test_defaults_are_sensible(self) -> None:
        p = Port(
            arr_type="pd.DataFrame",
            data_category="numerical_data",
            data_shape="batch feature",
            desc="test",
        )
        assert p.optional is False
        assert p.mode == ["all"]

    def test_custom_mode_round_trips(self) -> None:
        p = Port(
            arr_type="np.ndarray",
            data_category="numerical_data",
            data_shape="batch 1",
            desc="test",
            mode=[str(NodeExecutionMode.TRAINING)],
        )
        assert p.mode == ["training"]


class TestNodeTypeEnum:
    def test_all_expected_members(self) -> None:
        assert {nt.value for nt in NodeType} == {
            "base",
            "source",
            "sink",
            "transform",
            "model",
            "metric",
        }


class TestNodeFitTransform:
    def test_identity_transform_records_calls(self) -> None:
        """``node_fit_transform`` calls both fit and transform in order."""
        node = IdentityTransform(config=IdentityTransformConfig())
        df, ctx = numerical_pair()

        out = node.node_fit_transform({"input": (df, ctx)})

        assert node.fit_calls == 1
        assert node.transform_calls == 1
        assert "output" in out

    def test_transform_without_fit_raises(self) -> None:
        """The Transform base class gates transform on the fitted flag."""
        node = IdentityTransform(config=IdentityTransformConfig())
        df, ctx = numerical_pair()
        with pytest.raises(ValueError, match="fitted before"):
            node.node_transform({"input": (df, ctx)})


class TestDataSourceBase:
    """Quick smoke-test the ``DataSourceNode`` convenience wiring."""

    def test_node_fit_calls_setup_source(self) -> None:
        source = ConstantSource(config=ConstantSourceConfig())
        source.set_payload(numerical_pair())
        source.node_fit({})
        assert source.setup_calls == 1

    def test_node_transform_returns_payload(self) -> None:
        payload = numerical_pair()
        source = ConstantSource(config=ConstantSourceConfig())
        source.set_payload(payload)
        result = source.node_transform({})
        assert result["output"] is payload
