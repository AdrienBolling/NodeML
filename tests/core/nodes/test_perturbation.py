"""Tests for :class:`nodeml.core.nodes.transform.perturbation.PerturbationNode`."""

from __future__ import annotations

import pytest

from nodeml.core.common.data.data import ArrayLikeEnum, DataCategoryEnum
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.common.exceptions import DataContextError, NodeInputError
from nodeml.core.nodes.node import Port
from nodeml.core.nodes.registry.node_registry import NODE_REGISTRY
from nodeml.core.nodes.transform.perturbation import (
    PerturbationNode,
    PerturbationNodeConfig,
    PerturbationRunningConfig,
)
from tests.shims.tabular import numerical_pair


def _node(*, inject: bool) -> PerturbationNode:
    return PerturbationNode(
        config=PerturbationNodeConfig(
            running_config=PerturbationRunningConfig(inject_in_inference=inject)
        )
    )


class TestPassThrough:
    def test_returns_the_input_and_records_it(self) -> None:
        node = _node(inject=False)
        pair = numerical_pair()
        node.set_execution_mode(NodeExecutionMode.INFERENCE)
        df, ctx = node.node_transform({"input": pair})["output"]
        assert df is pair[0]
        assert ctx is pair[1]
        assert node.last_input == pair

    def test_injection_is_not_used_outside_inference(self) -> None:
        node = _node(inject=True)
        pair = numerical_pair()
        node.set_rows(*numerical_pair(n_rows=3))
        node.set_execution_mode(NodeExecutionMode.EVALUATION)
        assert node.node_transform({"input": pair})["output"][0] is pair[0]

    def test_missing_input_raises(self) -> None:
        node = _node(inject=False)
        node.set_execution_mode(NodeExecutionMode.INFERENCE)
        with pytest.raises(NodeInputError, match="needs its 'input' port"):
            node.node_transform({})

    def test_transform_works_without_fit(self) -> None:
        # Stateless: a reloaded node must transform without training.
        node = _node(inject=False)
        node.set_params(node.get_params())
        assert node.node_transform({"input": numerical_pair()})


class TestInjection:
    def test_returns_the_injected_rows_in_inference(self) -> None:
        node = _node(inject=True)
        rows = numerical_pair(n_rows=3)
        node.set_rows(*rows)
        node.set_execution_mode(NodeExecutionMode.INFERENCE)
        out = node.node_transform({})
        assert out["output"][0] is rows[0]

    def test_inference_without_rows_raises(self) -> None:
        node = _node(inject=True)
        node.set_execution_mode(NodeExecutionMode.INFERENCE)
        with pytest.raises(NodeInputError, match="no rows are set"):
            node.node_transform({})

    def test_clear_rows_forgets_the_rows(self) -> None:
        node = _node(inject=True)
        node.set_rows(*numerical_pair())
        node.clear_rows()
        node.set_execution_mode(NodeExecutionMode.INFERENCE)
        with pytest.raises(NodeInputError):
            node.node_transform({})

    def test_set_rows_checks_the_context(self) -> None:
        node = _node(inject=True)
        df, ctx = numerical_pair()
        with pytest.raises(DataContextError):
            node.set_rows(df[list(reversed(df.columns))], ctx)


class TestConfig:
    def test_injection_makes_the_input_port_inactive_in_inference(self) -> None:
        port = PerturbationNodeConfig(
            running_config=PerturbationRunningConfig(inject_in_inference=True)
        ).in_ports["input"]
        assert port.optional is True
        assert port.mode == [NodeExecutionMode.TRAINING, NodeExecutionMode.EVALUATION]

    def test_injection_does_not_change_a_shared_port(self) -> None:
        shared = Port(
            arr_type=ArrayLikeEnum.PANDAS,
            data_category=DataCategoryEnum.NUMERICAL,
            data_shape="batch feature",
            desc="shared",
        )
        PerturbationNodeConfig(
            in_ports={"input": shared},
            running_config=PerturbationRunningConfig(inject_in_inference=True),
        )
        assert shared.mode == [NodeExecutionMode.ALL]
        assert shared.optional is False

    def test_other_port_names_are_rejected(self) -> None:
        port = PerturbationNodeConfig().in_ports["input"]
        with pytest.raises(ValueError, match="exactly one input port"):
            PerturbationNodeConfig(in_ports={"X": port})

    def test_node_is_registered(self) -> None:
        assert NODE_REGISTRY.get_node_class("PerturbationNode") is PerturbationNode
        assert NODE_REGISTRY.get("PerturbationNode")["node_type"] == "transform"
