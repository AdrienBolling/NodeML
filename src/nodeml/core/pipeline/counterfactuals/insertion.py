"""Insert a PerturbationNode into a copy of a trained pipeline.

The functions here never change the config or the parameters that they
receive.  They work on a deep copy, so the original pipeline stays the same.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nodeml.core.common.data.data import ArrayLikeEnum
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.common.exceptions import PipelineValidationError
from nodeml.core.nodes.node import NodeType
from nodeml.core.nodes.transform.perturbation import (
    PERTURBATION_INPUT_PORT,
    PERTURBATION_OUTPUT_PORT,
    PerturbationNodeConfig,
    PerturbationRunningConfig,
)
from nodeml.core.pipeline.counterfactuals.config import InsertionPoint, OutputPoint
from nodeml.core.pipeline.pipeline import Edge, Pipeline, PipelineConfig

PERTURBATION_NODE_NAME = "__perturbation__"
"""Name of the PerturbationNode in the rebuilt pipeline."""

OUTPUT_SINK_PORT = "__counterfactual_output__"
"""Sink port that receives the prediction in the rebuilt pipeline."""


@dataclass(frozen=True)
class ResolvedPoints:
    """The insertion point and the output point, with all names resolved.

    Attributes:
        insertion_node: Node whose output port feeds the PerturbationNode.
        insertion_port: That output port.
        output_node: Node that produces the prediction.
        output_port: That output port.

    """

    insertion_node: str
    insertion_port: str
    output_node: str
    output_port: str


def _is_active_in_inference(modes: list[NodeExecutionMode]) -> bool:
    """Return whether a port with *modes* runs in inference."""
    return NodeExecutionMode.ALL in modes or NodeExecutionMode.INFERENCE in modes


def _inference_inputs(pipeline: Pipeline, node_name: str) -> list[tuple[str, str]]:
    """Return the ``(source node, source port)`` pairs that feed *node_name* in inference."""
    nodes = pipeline.resolved_nodes
    in_ports = nodes[node_name][1].in_ports
    return [
        (edge.source, source_port)
        for edge in pipeline.edges
        if edge.target == node_name
        for source_port, target_port in edge.ports_map
        if target_port in in_ports
        and _is_active_in_inference(in_ports[target_port].mode)
    ]


def _inference_ancestors(pipeline: Pipeline, start: str) -> set[tuple[str, str]]:
    """Return every ``(node, port)`` output that *start* needs in inference."""
    seen: set[tuple[str, str]] = set()
    stack = [start]
    visited: set[str] = set()
    while stack:
        name = stack.pop()
        if name in visited:
            continue
        visited.add(name)
        for pair in _inference_inputs(pipeline, name):
            seen.add(pair)
            stack.append(pair[0])
    return seen


def _resolve_output(pipeline: Pipeline, output: OutputPoint) -> tuple[str, str]:
    """Return the ``(node, port)`` of the prediction."""
    sink = pipeline.get_validated_sink_node_name()
    sink_inputs = [
        (edge.source, source_port)
        for edge in pipeline.sink_edges_dict.values()
        for source_port, _ in edge.ports_map
    ]
    if output.node is None:
        if len(sink_inputs) != 1:
            msg = (
                f"The Sink '{sink}' receives {len(sink_inputs)} outputs "
                f"{sink_inputs}. Set OutputPoint.node and OutputPoint.port."
            )
            raise PipelineValidationError(msg)
        return sink_inputs[0]
    out_ports = pipeline.get_node_config(output.node).out_ports
    if output.port is not None:
        if output.port not in out_ports:
            msg = f"Node '{output.node}' has no output port '{output.port}'. Ports: {list(out_ports)}."
            raise PipelineValidationError(msg)
        return output.node, output.port
    if len(out_ports) != 1:
        msg = f"Node '{output.node}' has the output ports {list(out_ports)}. Set OutputPoint.port."
        raise PipelineValidationError(msg)
    return output.node, next(iter(out_ports))


def _resolve_insertion(
    pipeline: Pipeline, insertion: InsertionPoint, output_node: str
) -> tuple[str, str]:
    """Return the ``(node, port)`` that receives the PerturbationNode."""
    needed = _inference_ancestors(pipeline, output_node)
    if insertion.node is None:
        sources = sorted(
            (node, port)
            for node, port in needed
            if pipeline.get_node_config(node).node_type == NodeType.SOURCE
        )
        if len(sources) != 1:
            msg = (
                f"The node '{output_node}' needs {len(sources)} source outputs in "
                f"inference: {sources}. Set InsertionPoint.node and InsertionPoint.port."
            )
            raise PipelineValidationError(msg)
        return sources[0]
    out_ports = pipeline.get_node_config(insertion.node).out_ports
    if insertion.port is not None:
        if insertion.port not in out_ports:
            msg = f"Node '{insertion.node}' has no output port '{insertion.port}'. Ports: {list(out_ports)}."
            raise PipelineValidationError(msg)
        return insertion.node, insertion.port
    used = sorted(port for node, port in needed if node == insertion.node)
    if len(used) != 1:
        msg = (
            f"The node '{output_node}' needs {len(used)} outputs of '{insertion.node}' "
            f"in inference: {used}. Set InsertionPoint.port."
        )
        raise PipelineValidationError(msg)
    return insertion.node, used[0]


def resolve_points(
    config: PipelineConfig, insertion: InsertionPoint, output: OutputPoint
) -> ResolvedPoints:
    """Resolve the insertion point and the output point of a pipeline.

    Args:
        config: The pipeline config.  It does not change.
        insertion: The requested insertion point.
        output: The requested output point.

    Returns:
        The resolved points.

    Raises:
        PipelineValidationError: If a point is ambiguous or does not exist.

    """
    pipeline = Pipeline(config=config.model_copy(deep=True))
    output_node, output_port = _resolve_output(pipeline, output)
    insertion_node, insertion_port = _resolve_insertion(
        pipeline, insertion, output_node
    )
    return ResolvedPoints(
        insertion_node=insertion_node,
        insertion_port=insertion_port,
        output_node=output_node,
        output_port=output_port,
    )


def insert_perturbation_node(
    config: PipelineConfig, points: ResolvedPoints, *, inject: bool
) -> Pipeline:
    """Return an uncompiled copy of the pipeline with a PerturbationNode.

    Every edge pair that leaves ``insertion_node.insertion_port`` goes
    through the PerturbationNode.  An extra edge sends the prediction to the
    Sink port :data:`OUTPUT_SINK_PORT`.

    Args:
        config: The pipeline config.  It does not change.
        points: The resolved points.
        inject: The ``inject_in_inference`` option of the PerturbationNode.

    Returns:
        A new, uncompiled :class:`Pipeline`.

    """
    pipeline = Pipeline(config=config.model_copy(deep=True))
    source_port = pipeline.get_node_config(points.insertion_node).out_ports[
        points.insertion_port
    ]
    # The node copies the source port, so the downstream checks stay the same.
    port = source_port.model_copy(
        update={"arr_type": ArrayLikeEnum.PANDAS, "optional": False}, deep=True
    )
    pipeline.add_node(
        PERTURBATION_NODE_NAME,
        "PerturbationNode",
        PerturbationNodeConfig(
            in_ports={PERTURBATION_INPUT_PORT: port},
            out_ports={PERTURBATION_OUTPUT_PORT: port.model_copy(deep=True)},
            running_config=PerturbationRunningConfig(inject_in_inference=inject),
        ),
    )
    for edge in list(pipeline.edges):
        if edge.source != points.insertion_node:
            continue
        moved = [pair for pair in edge.ports_map if pair[0] == points.insertion_port]
        if not moved:
            continue
        kept = [pair for pair in edge.ports_map if pair[0] != points.insertion_port]
        if kept:
            pipeline.update_edge(
                Edge(source=edge.source, target=edge.target, ports_map=kept)
            )
        else:
            pipeline.remove_edge(edge.source, edge.target)
        pipeline.add_edge(
            Edge(
                source=PERTURBATION_NODE_NAME,
                target=edge.target,
                ports_map=[(PERTURBATION_OUTPUT_PORT, target) for _, target in moved],
            )
        )
    pipeline.add_edge(
        Edge(
            source=points.insertion_node,
            target=PERTURBATION_NODE_NAME,
            ports_map=[(points.insertion_port, PERTURBATION_INPUT_PORT)],
        )
    )
    pipeline.add_edge(
        Edge(
            source=points.output_node,
            target=pipeline.get_validated_sink_node_name(),
            ports_map=[(points.output_port, OUTPUT_SINK_PORT)],
        )
    )
    return pipeline


def build_counterfactual_pipeline(
    config: PipelineConfig,
    params: Mapping[str, Any],
    points: ResolvedPoints,
    *,
    inject: bool,
) -> Pipeline:
    """Return a compiled, trained copy of the pipeline with a PerturbationNode.

    Args:
        config: The pipeline config.  It does not change.
        params: The trained parameters (``Pipeline.get_params()``).  They do
            not change.
        points: The resolved points.
        inject: If ``True``, the PerturbationNode returns injected rows in
            inference, and the nodes before it do not run.

    Returns:
        A compiled pipeline with the trained parameters loaded.

    Raises:
        PipelineValidationError: If, with injection, the prediction still
            needs a data source in inference.  The PerturbationNode must then
            be inserted after a node that all the inputs go through.

    """
    pipeline = insert_perturbation_node(config, points, inject=inject)
    pipeline.compile()
    # The nodes can share objects with *params* (for example a fitted
    # estimator).  The evaluator only predicts, so these objects do not change.
    pipeline.set_params(dict(params))
    if inject:
        leaks = sorted(
            {
                node
                for node, _ in _inference_ancestors(pipeline, points.output_node)
                if pipeline.get_node_config(node).node_type == NodeType.SOURCE
            }
        )
        if leaks:
            msg = (
                f"With the PerturbationNode after '{points.insertion_node}."
                f"{points.insertion_port}', the node '{points.output_node}' still "
                f"needs the sources {leaks} in inference. Insert the "
                "PerturbationNode after a node that all the inputs go through."
            )
            raise PipelineValidationError(msg)
    return pipeline
