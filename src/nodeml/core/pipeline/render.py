"""Rendering utilities for NodeML pipeline graphs."""

from __future__ import annotations

import html
import math
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.nodes.node import NodeConfig, NodeType

# Edge labels are flipped outside [-limit, limit] degrees to stay upright.
_UPRIGHT_LIMIT_DEG = 90

if TYPE_CHECKING:
    from nodeml.core.pipeline.pipeline import Edge, Pipeline


# =============================================================================
# Graph structure helpers
# =============================================================================


def _longest_path_length_to(graph: nx.DiGraph, target: str) -> dict[str, int]:
    """Longest-path distance in a DAG from every reachable node to ``target``.

    Uses DP over reverse topological order: ``dist[target] = 0`` and
    ``dist[node] = 1 + max(dist[succ])`` over successors with a known
    distance.  ``nx.shortest_path_length`` collapses parallel chains to
    the shortest branch, which lets edges skip backwards over shorter
    siblings during layout; the longest-path variant pushes each node
    as far right as its deepest successor allows so every ancestor
    lands strictly left of every descendant.
    """
    dist: dict[str, int] = {target: 0}
    for node in reversed(list(nx.topological_sort(graph))):
        if node == target:
            continue
        successor_depths = [
            dist[succ] + 1 for succ in graph.successors(node) if succ in dist
        ]
        if successor_depths:
            dist[node] = max(successor_depths)
    return dist


def split_execution_graph_into_columns(
    pipeline: Pipeline,
) -> dict[int, set[str]]:
    """Group nodes into layout columns for visualization.

    The main execution graph is laid out from left to right using
    longest-path distance to the sink node — this guarantees that a
    node always sits to the left of every one of its successors, so
    edges never skip backwards across columns.  Source nodes are
    forced to the leftmost column and metric nodes into their own
    column on the right.

    Args:
        pipeline: Compiled pipeline whose graph will be partitioned.

    Returns:
        Dict mapping column indices to sets of node names.

    Raises:
        ValueError: If the pipeline has no sink node.

    """
    graph = pipeline.graph
    main_graph = pipeline.graph_wo_metrics
    sink_node = pipeline.sink_node_name
    node_columns: dict[str, int | None] = dict.fromkeys(main_graph.nodes)

    if sink_node is None:
        msg = "Pipeline must have a sink node to compute layout."
        raise ValueError(msg)

    longest_paths_to_sink = _longest_path_length_to(main_graph, sink_node)
    max_depth = max(longest_paths_to_sink.values())

    for node, depth in longest_paths_to_sink.items():
        if node is not None:
            node_columns[node] = -int(depth)

    # Force all source nodes to the far left.
    source_node_names = pipeline.get_source_node_names()
    if source_node_names:
        for source_node in source_node_names:
            node_columns[source_node] = -int(max_depth) - 1

    # Place metric nodes in their own column.
    for node in graph.nodes:
        if pipeline.internal_config.nodes[node][1].node_type == NodeType.METRIC:
            node_columns[node] = 1

    columns: dict[int, set[str]] = {}
    for node, col in node_columns.items():
        if col is not None:
            columns.setdefault(col, set()).add(node)

    return columns


# =============================================================================
# Node mappings
# =============================================================================


def node_name_to_color_mapping(
    node_configs: dict[str, tuple[str, NodeConfig]],
) -> dict[str, str]:
    """Map node names to colors based on node type.

    Args:
        node_configs: Mapping of node names to ``(class_name, NodeConfig)``.

    Returns:
        Dict mapping node names to CSS color strings.

    """
    type_to_color = {
        NodeType.SOURCE: "lightblue",
        NodeType.SINK: "lightcoral",
        NodeType.METRIC: "lightgreen",
        NodeType.TRANSFORM: "orange",
        NodeType.MODEL: "lightgray",
    }
    return {
        node_name: type_to_color.get(node_config.node_type, "white")
        for node_name, (_, node_config) in node_configs.items()
    }


def chosen_data_from_node(node_config: NodeConfig) -> Any:
    """Extract the data displayed for a node in hover/tooltip views.

    Args:
        node_config: Configuration of the node.

    Returns:
        Serialized dict of the node configuration.

    """
    return node_config.model_dump()


def node_name_to_node_data_mapping(
    node_configs: dict[str, tuple[str, NodeConfig]],
) -> dict[str, dict[str, Any]]:
    """Map node names to serialized node data for tooltips.

    Args:
        node_configs: Mapping of node names to ``(class_name, NodeConfig)``.

    Returns:
        Dict mapping node names to their serialized config dicts.

    """
    return {
        node_name: chosen_data_from_node(node_config)
        for node_name, (_, node_config) in node_configs.items()
    }


def node_name_to_marker_style_mapping(
    node_configs: dict[str, tuple[str, NodeConfig]],
) -> dict[str, str]:
    """Map node names to matplotlib marker styles based on node type.

    Args:
        node_configs: Mapping of node names to ``(class_name, NodeConfig)``.

    Returns:
        Dict mapping node names to matplotlib marker character codes.

    """
    type_to_marker = {
        NodeType.SOURCE: "p",
        NodeType.SINK: "p",
        NodeType.METRIC: "d",
        NodeType.TRANSFORM: "^",
        NodeType.MODEL: "o",
    }
    return {
        node_name: type_to_marker.get(node_config.node_type, "x")
        for node_name, (_, node_config) in node_configs.items()
    }


def node_name_to_hover_text_mapping(
    node_configs: dict[str, tuple[str, NodeConfig]],
) -> dict[str, str]:
    """Map node names to plain-text hover content.

    Args:
        node_configs: Mapping of node names to ``(class_name, NodeConfig)``.

    Returns:
        Dict mapping node names to multi-line hover strings.

    """
    hover_text_mapping: dict[str, str] = {}
    node_data_mapping = node_name_to_node_data_mapping(node_configs)

    for node_name, node_data in node_data_mapping.items():
        meta_dict = (
            node_data if isinstance(node_data, dict) else {"value": str(node_data)}
        )
        lines = [f"node: {node_name}"]

        if meta_dict:
            for key, value in meta_dict.items():
                lines.append(f"{key}: {value}")
        else:
            lines.append("(no metadata)")

        hover_text_mapping[node_name] = "\n".join(lines)

    return hover_text_mapping


# =============================================================================
# Edge mappings
# =============================================================================


def edge_to_linestyle_mapping(
    edges: list[Edge],
    node_configs: dict[str, tuple[str, NodeConfig]],
) -> dict[tuple[str, str], str]:
    """Map edges to line styles based on connected node types.

    Args:
        edges: List of pipeline edges.
        node_configs: Mapping of node names to ``(class_name, NodeConfig)``.

    Returns:
        Dict mapping ``(source, target)`` to matplotlib line-style strings.

    """
    mapping: dict[tuple[str, str], str] = {}

    for edge in edges:
        source_type = node_configs[edge.source][1].node_type
        target_type = node_configs[edge.target][1].node_type

        if source_type == NodeType.SOURCE or target_type == NodeType.SINK:
            mapping[(edge.source, edge.target)] = "-"
        elif NodeType.METRIC in (source_type, target_type):
            mapping[(edge.source, edge.target)] = "--"
        else:
            mapping[(edge.source, edge.target)] = "-"

    return mapping


def select_edges_to_display(
    edges: list[Edge],
    node_configs: dict[str, tuple[str, NodeConfig]],
    execution_mode: str | None = None,
) -> list[Edge]:
    """Return the subset of edges to display for a given execution mode.

    Args:
        edges: All edges in the pipeline.
        node_configs: Mapping of node names to ``(class_name, NodeConfig)``.
        execution_mode: ``None`` to display all edges, or a mode name to
            filter to edges active in that mode.

    Returns:
        Filtered list of edges.

    Notes:
        This function delegates to :func:`edge_matches_execution_mode` to
        determine whether an edge is active in the requested mode.

    """
    if execution_mode is None or execution_mode == NodeExecutionMode.ALL:
        return edges

    source_configs = {node: config for node, (_, config) in node_configs.items()}
    target_configs = {node: config for node, (_, config) in node_configs.items()}

    return [
        edge
        for edge in edges
        if edge_matches_execution_mode(
            edge,
            execution_mode,
            source_configs[edge.source],
            target_configs[edge.target],
        )
    ]


def edge_matches_execution_mode(
    edge: Edge,
    execution_mode: str,
    source_config: NodeConfig,
    target_config: NodeConfig,
) -> bool:
    """Check whether an edge is active in the given execution mode.

    Args:
        edge: The edge to check.
        execution_mode: The execution mode to match against.
        source_config: Config of the source node.
        target_config: Config of the target node.

    Returns:
        ``True`` if at least one port mapping matches *execution_mode*.

    """
    exec_modes = [
        (
            source_config.out_ports[source_port].mode,
            target_config.in_ports[target_port].mode,
        )
        for (source_port, target_port) in edge.ports_map
    ]
    for source_modes, target_modes in exec_modes:
        if "all" in source_modes and "all" in target_modes:
            return True
        if execution_mode in source_modes or execution_mode in target_modes:
            return True
    return False


def edge_to_hover_text_mapping(edges: list[Edge]) -> dict[tuple[str, str], str]:
    """Map edges to plain-text hover content.

    Args:
        edges: List of pipeline edges.

    Returns:
        Dict mapping ``(source, target)`` to multi-line hover strings.

    """
    mapping: dict[tuple[str, str], str] = {}

    for edge in edges:
        lines = [f"edge: {edge.source} -> {edge.target}", "Ports mapping:"]
        lines.extend(
            f"  {target_port}: {source_port}"
            for (source_port, target_port) in edge.ports_map
        )
        mapping[(edge.source, edge.target)] = "\n".join(lines)

    return mapping


def edge_curved_mapping(
    edges: list[Edge],
    node_configs: dict[str, tuple[str, NodeConfig]],
) -> dict[tuple[str, str], bool]:
    """Map edges to whether they should be rendered as curved.

    Edges targeting metric nodes are curved to visually separate them from
    the main execution flow.

    Args:
        edges: List of pipeline edges.
        node_configs: Mapping of node names to ``(class_name, NodeConfig)``.

    Returns:
        Dict mapping ``(source, target)`` to a boolean (``True`` = curved).

    """
    return {
        (edge.source, edge.target): node_configs[edge.target][1].node_type
        == NodeType.METRIC
        for edge in edges
    }


# =============================================================================
# Hover HTML helpers for Plotly
# =============================================================================


def _dict_to_hover_html(data: Any, indent: int = 0) -> str:
    """Format nested dict/list data as HTML for Plotly hover content.

    Args:
        data: Arbitrary data (dict, list, or scalar) to format.
        indent: Current nesting level for indentation.

    Returns:
        HTML-escaped string with ``<br>`` line breaks.

    """
    pad = "&nbsp;" * 4 * indent

    if isinstance(data, dict):
        lines: list[str] = []
        for key, value in data.items():
            escaped_key = html.escape(str(key))
            if isinstance(value, (dict, list, tuple)):
                lines.append(f"{pad}<b>{escaped_key}</b>:")
                lines.append(_dict_to_hover_html(value, indent + 1))
            else:
                lines.append(f"{pad}<b>{escaped_key}</b>: {html.escape(str(value))}")
        return "<br>".join(lines)

    if isinstance(data, (list, tuple)):
        lines = []
        for item in data:
            if isinstance(item, (dict, list, tuple)):
                lines.append(f"{pad}•")
                lines.append(_dict_to_hover_html(item, indent + 1))
            else:
                lines.append(f"{pad}• {html.escape(str(item))}")
        return "<br>".join(lines)

    return f"{pad}{html.escape(str(data))}"


def _node_hover_html(node_name: str, node_data: Any) -> str:
    """Build an HTML hover snippet for a single node."""
    title = f"<b>{html.escape(str(node_name))}</b>"
    body = _dict_to_hover_html(node_data)
    return f"{title}<br>{body}" if body else title


def _edge_hover_html(source: str, target: str, ports_map: list[tuple[str, str]]) -> str:
    """Build an HTML hover snippet for a single edge."""
    lines = [
        f"<b>{html.escape(source)} → {html.escape(target)}</b>",
        "<b>Ports mapping</b>",
    ]
    lines.extend(
        f"{html.escape(str(source_port))} → {html.escape(str(target_port))}"
        for source_port, target_port in ports_map
    )
    return "<br>".join(lines)


# =============================================================================
# Geometry helpers for curved edges
# =============================================================================


def _vertical_center(pos: dict[str, Any]) -> float:
    """Return the vertical midpoint of all node positions."""
    ys = [float(coords[1]) for coords in pos.values()]
    return (min(ys) + max(ys)) / 2.0


def _edge_curvature(
    source: str,
    pos: dict[str, Any],
    y_center: float,
    min_curvature: float = 0.18,
    max_curvature: float = 0.45,
    power: float = 1.2,
) -> float:
    """Compute signed curvature for a curved edge from source position.

    Edges above the layout center bend upward, edges below bend downward.
    The farther the source from the center, the stronger the curvature.
    """
    source_y = float(pos[source][1])
    ys = [float(coords[1]) for coords in pos.values()]
    max_dist = max(*(abs(y - y_center) for y in ys), 1e-12)

    norm_dist = min(max(abs(source_y - y_center) / max_dist, 0.0), 1.0)
    magnitude = min_curvature + (max_curvature - min_curvature) * (norm_dist**power)
    sign = 1.0 if source_y >= y_center else -1.0
    return sign * magnitude


def _quadratic_bezier_control_point(
    p0: tuple[float, float],
    p1: tuple[float, float],
    curvature: float,
) -> tuple[float, float]:
    """Return the control point for a quadratic Bézier curve."""
    x0, y0 = p0
    x1, y1 = p1

    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy) or 1.0

    nx_, ny_ = -dy / dist, dx / dist
    return mx + curvature * dist * nx_, my + curvature * dist * ny_


def _quadratic_bezier_point(
    p0: tuple[float, float],
    p1: tuple[float, float],
    control: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    """Evaluate a quadratic Bézier curve at parameter t."""
    x0, y0 = p0
    x1, y1 = p1
    cx, cy = control

    x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t**2 * x1
    y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t**2 * y1
    return x, y


def _quadratic_bezier_derivative(
    p0: tuple[float, float],
    p1: tuple[float, float],
    control: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    """Evaluate the derivative of a quadratic Bézier curve at parameter t."""
    x0, y0 = p0
    x1, y1 = p1
    cx, cy = control

    dx = 2 * (1 - t) * (cx - x0) + 2 * t * (x1 - cx)
    dy = 2 * (1 - t) * (cy - y0) + 2 * t * (y1 - cy)
    return dx, dy


def _quadratic_bezier_polyline(
    p0: tuple[float, float],
    p1: tuple[float, float],
    curvature: float,
    steps: int = 200,
) -> tuple[list[float], list[float], tuple[float, float]]:
    """Sample a quadratic Bézier curve into x/y coordinate lists."""
    control = _quadratic_bezier_control_point(p0, p1, curvature)
    ts = np.linspace(0.0, 1.0, steps + 1)

    xs: list[float] = []
    ys: list[float] = []
    for t in ts:
        x, y = _quadratic_bezier_point(p0, p1, control, float(t))
        xs.append(x)
        ys.append(y)

    return xs, ys, control


def _bezier_t_for_arc_fraction(
    p0: tuple[float, float],
    p1: tuple[float, float],
    control: tuple[float, float],
    steps: int = 400,
    arc_fraction: float = 0.5,
) -> float:
    """Approximate the parameter t corresponding to a target arc-length fraction."""
    ts = np.linspace(0.0, 1.0, steps + 1)
    points = np.array([_quadratic_bezier_point(p0, p1, control, float(t)) for t in ts])

    seg_lengths = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cum_lengths[-1]

    if total == 0:
        return 0.5

    target = arc_fraction * total
    idx = int(np.searchsorted(cum_lengths, target))
    idx = max(1, min(idx, len(ts) - 1))

    prev_len = cum_lengths[idx - 1]
    next_len = cum_lengths[idx]
    alpha = 0.0 if next_len == prev_len else (target - prev_len) / (next_len - prev_len)
    return float(ts[idx - 1] + alpha * (ts[idx] - ts[idx - 1]))


def _sample_quadratic_bezier_with_exact_label_pose(
    p0: tuple[float, float],
    p1: tuple[float, float],
    curvature: float,
    steps: int = 200,
    label_arc_fraction: float = 0.5,
) -> tuple[list[float], list[float], float, float, float]:
    """Sample a Bézier curve and compute an exact label position and angle."""
    xs, ys, control = _quadratic_bezier_polyline(
        p0=p0,
        p1=p1,
        curvature=curvature,
        steps=steps,
    )

    t_label = _bezier_t_for_arc_fraction(
        p0=p0,
        p1=p1,
        control=control,
        steps=max(400, steps),
        arc_fraction=label_arc_fraction,
    )

    lx, ly = _quadratic_bezier_point(p0, p1, control, t_label)
    dx, dy = _quadratic_bezier_derivative(p0, p1, control, t_label)

    # Keep edge labels upright: flip angles outside [-90, 90] degrees.
    angle = math.degrees(math.atan2(dy, dx))
    if angle > _UPRIGHT_LIMIT_DEG:
        angle -= 180
    elif angle < -_UPRIGHT_LIMIT_DEG:
        angle += 180

    return xs, ys, lx, ly, angle


# =============================================================================
# Plotly rendering helpers
# =============================================================================


def _add_edge_shapes(
    fig: go.Figure,
    edges: list[Edge],
    pos: dict[str, Any],
    edge_color_mapping: dict[tuple[str, str], str],
    edge_linestyle_mapping: dict[tuple[str, str], str],
    edge_curved_mapping_: dict[tuple[str, str], bool],
) -> None:
    """Add visible straight edge shapes to a Plotly figure.

    Hover is handled separately with transparent scatter traces.
    """
    dash_map = {"-": "solid", "--": "dash", "-.": "dashdot", ":": "dot"}

    for edge in edges:
        source, target = edge.source, edge.target
        if edge_curved_mapping_.get((source, target), False):
            continue

        x0, y0 = float(pos[source][0]), float(pos[source][1])
        x1, y1 = float(pos[target][0]), float(pos[target][1])

        fig.add_shape(
            type="line",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line={
                "color": edge_color_mapping.get((source, target), "gray"),
                "width": 2,
                "dash": dash_map.get(
                    edge_linestyle_mapping.get((source, target), "-"), "solid"
                ),
            },
            layer="below",
        )


def _add_legend(
    fig: go.Figure,
    graph: nx.Graph,
    node_configs: dict[str, tuple[str, NodeConfig]],
    node_color_mapping: dict[str, str],
    node_marker_mapping: dict[str, str],
    edge_color_mapping: dict[tuple[str, str], str],
) -> None:
    """Add node and edge legend entries for the styles present in the graph."""
    marker_symbol_map = {
        "o": "circle",
        "s": "square",
        "^": "triangle-up",
        "v": "triangle-down",
        "d": "diamond",
        "p": "pentagon",
        "x": "x",
        "*": "star",
    }

    seen_node_types: dict[NodeType, str] = {}
    for node in graph.nodes:
        node_type = node_configs[node][1].node_type
        if node_type in seen_node_types:
            continue
        seen_node_types[node_type] = node

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "size": 12,
                    "color": node_color_mapping[node],
                    "symbol": marker_symbol_map.get(
                        node_marker_mapping[node], "circle"
                    ),
                    "line": {"color": "black", "width": 1},
                },
                name=node_type.name,
                legendgroup="nodes",
                showlegend=True,
            )
        )

    color_to_label = {
        "black": "All modes",
        "blue": "Multiple modes",
        "green": "Train",
        "orange": "Inference",
        "purple": "Evaluation",
        "gray": "Other",
    }

    seen_colors: set[str] = set()
    for color in edge_color_mapping.values():
        if color in seen_colors:
            continue
        seen_colors.add(color)

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line={"color": color, "width": 3},
                name=color_to_label.get(color, color),
                legendgroup="edges",
                showlegend=True,
            )
        )


def edge_to_color_mapping(
    edges: list[Edge],
    execution_mode: str | None = None,
) -> dict[tuple[str, str], str]:
    """Map edges to colors based on execution mode.

    Args:
        edges: List of pipeline edges.
        execution_mode: Mode name.  Defaults to ``"all"`` when ``None``.

    Returns:
        Dict mapping ``(source, target)`` to CSS color strings.

    """
    if execution_mode is None:
        execution_mode = str(NodeExecutionMode.ALL)
    exec_to_color_mapping = {
        NodeExecutionMode.ALL: "black",
        NodeExecutionMode.TRAINING: "green",
        NodeExecutionMode.INFERENCE: "orange",
        NodeExecutionMode.EVALUATION: "purple",
    }
    return {
        (edge.source, edge.target): exec_to_color_mapping[
            NodeExecutionMode(execution_mode)
        ]
        for edge in edges
    }


def _straight_edge_polyline(
    p0: tuple[float, float],
    p1: tuple[float, float],
    steps: int = 100,
) -> tuple[list[float], list[float]]:
    """Sample a straight line into many points for reliable Plotly hover."""
    x0, y0 = p0
    x1, y1 = p1
    ts = np.linspace(0.0, 1.0, steps + 1)

    xs = [float((1 - t) * x0 + t * x1) for t in ts]
    ys = [float((1 - t) * y0 + t * y1) for t in ts]
    return xs, ys


# =============================================================================
# Plotly rendering
# =============================================================================


def _filter_node_data_for_mode(data: dict[str, Any], mode: str) -> dict[str, Any]:
    """Return a copy of serialized node data with ports filtered by mode.

    Ports whose ``mode`` list does not include the requested execution
    mode (or ``"all"``) are dropped so that hover tooltips only show
    the ports relevant to the currently displayed execution mode.
    When *mode* is ``"all"`` the data is returned unchanged.
    """
    all_mode = str(NodeExecutionMode.ALL)
    if mode == all_mode:
        return data
    filtered = dict(data)
    for port_key in ("in_ports", "out_ports"):
        ports = filtered.get(port_key)
        if not isinstance(ports, dict):
            continue
        filtered[port_key] = {
            pname: pdata
            for pname, pdata in ports.items()
            if _port_active(pdata, mode, all_mode)
        }
    return filtered


def _port_active(port_data: Any, mode: str, all_mode: str) -> bool:
    """Check whether a serialized port is active in the given mode."""
    if not isinstance(port_data, dict):
        return True
    port_modes = port_data.get("mode", [])
    if not isinstance(port_modes, (list, tuple)):
        return True
    str_modes = [str(m) for m in port_modes]
    return all_mode in str_modes or mode in str_modes


def _build_node_label_annotations(
    names: list[str],
    xs: list[float],
    ys: list[float],
    hover_html: list[str],
) -> list[dict[str, Any]]:
    """Build annotation dicts for node label boxes.

    Returns a list of plain dicts (not Plotly objects) so they can be
    swapped via relayout when the execution-mode dropdown changes.
    """
    annotations: list[dict[str, Any]] = []
    for name, x, y, hover in zip(names, xs, ys, hover_html, strict=True):
        annotations.append(
            {
                "x": x,
                "y": y,
                "text": str(name),
                "showarrow": False,
                "yshift": 22,
                "bgcolor": "rgba(245,245,245,0.92)",
                "bordercolor": "rgba(0,0,0,0.35)",
                "borderwidth": 1,
                "borderpad": 3,
                "font": {"size": 11, "color": "black"},
                "hovertext": hover,
                "hoverlabel": {
                    "bgcolor": "rgba(30,30,30,0.95)",
                    "font": {"color": "white", "size": 12},
                },
            }
        )
    return annotations


def _auto_figsize(columns: dict[int, set[str]]) -> tuple[int, int]:
    """Pick a ``(width, height)`` in inches that fits the layout shape.

    Width scales with the number of columns (so long pipelines get wider
    canvases), height with the widest column (so fan-outs have room to
    spread vertically).  The floors keep tiny pipelines from rendering
    into a sliver.
    """
    num_columns = max(len(columns), 1)
    max_rows = max((len(s) for s in columns.values()), default=1)
    width = max(8, num_columns * 2)
    height = max(12, int(max_rows * 4))
    return (width, height)


def render_pipeline_graph_plotly(  # noqa: C901, PLR0912, PLR0915 - one figure, built step by step
    pipeline: Pipeline,
    title: str = "Pipeline Graph",
    figsize: tuple[int, int] | None = None,
    execution_mode: str | None = None,
) -> go.Figure:
    """Render the pipeline graph with Plotly and a dropdown to select execution mode.

    Args:
        pipeline: Compiled pipeline to render.
        title: Figure title.
        figsize: ``(width, height)`` in logical inches (multiplied by 100 for
            pixel dimensions).  ``None`` (the default) scales width with
            the number of layout columns and height with the widest
            column so long pipelines don't get crammed into a fixed box.
        execution_mode: Initial execution mode for the dropdown.  ``None``
            defaults to ``"all"``.

    Returns:
        A Plotly :class:`~plotly.graph_objects.Figure`.

    """
    version = pipeline.version

    graph = pipeline.graph
    node_configs = pipeline.internal_config.nodes

    columns = split_execution_graph_into_columns(pipeline=pipeline)
    if figsize is None:
        figsize = _auto_figsize(columns)
    pos = nx.multipartite_layout(graph, subset_key=columns)
    y_center = _vertical_center(pos)

    node_color_mapping = node_name_to_color_mapping(node_configs)
    node_marker_mapping = node_name_to_marker_style_mapping(node_configs)
    node_data_mapping = node_name_to_node_data_mapping(node_configs)

    marker_symbol_map = {
        "o": "circle",
        "s": "square",
        "^": "triangle-up",
        "v": "triangle-down",
        "d": "diamond",
        "p": "pentagon",
        "x": "x",
        "*": "star",
    }
    dash_map = {"-": "solid", "--": "dash", "-.": "dashdot", ":": "dot"}

    fig = go.Figure()

    # -------------------------------------------------------------------------
    # Build edge traces for each execution mode
    # -------------------------------------------------------------------------
    mode_order = [
        str(NodeExecutionMode.ALL),
        str(NodeExecutionMode.TRAINING),
        str(NodeExecutionMode.INFERENCE),
        str(NodeExecutionMode.EVALUATION),
    ]

    if execution_mode is None:
        initial_mode = str(NodeExecutionMode.ALL)
    else:
        initial_mode = str(NodeExecutionMode(execution_mode))

    edge_trace_indices_by_mode: dict[str, list[int]] = {mode: [] for mode in mode_order}

    for mode in mode_order:
        visible_edges = select_edges_to_display(
            pipeline.edges,
            execution_mode=mode,
            node_configs=node_configs,
        )
        edge_color_mapping = edge_to_color_mapping(visible_edges, mode)
        edge_linestyle_mapping = edge_to_linestyle_mapping(visible_edges, node_configs)
        edge_curved_mapping_ = edge_curved_mapping(visible_edges, node_configs)

        for edge in visible_edges:
            source, target = edge.source, edge.target
            x0, y0 = float(pos[source][0]), float(pos[source][1])
            x1, y1 = float(pos[target][0]), float(pos[target][1])
            curved = edge_curved_mapping_.get((source, target), False)

            if curved:
                curvature = _edge_curvature(
                    source=source,
                    pos=pos,
                    y_center=y_center,
                    min_curvature=0.10,
                    max_curvature=0.45,
                    power=1.2,
                )

                xs, ys, _, _, _ = _sample_quadratic_bezier_with_exact_label_pose(
                    (x0, y0),
                    (x1, y1),
                    curvature=curvature,
                    steps=200,
                    label_arc_fraction=0.5,
                )
            else:
                xs, ys = _straight_edge_polyline(
                    (x0, y0),
                    (x1, y1),
                    steps=100,
                )

            # Arrow marker only on the last sample (``angleref="previous"``
            # rotates it along the tangent so curved edges still point right).
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    line={
                        "color": edge_color_mapping.get((source, target), "gray"),
                        "width": 2,
                        "dash": dash_map.get(
                            edge_linestyle_mapping.get((source, target), "-"),
                            "solid",
                        ),
                    },
                    marker={
                        "size": [0] * (len(xs) - 1) + [14],
                        "symbol": "arrow",
                        "angleref": "previous",
                        "color": edge_color_mapping.get((source, target), "gray"),
                        "standoff": 10,
                    },
                    opacity=0.6,
                    hovertemplate=_edge_hover_html(source, target, edge.ports_map)
                    + "<extra></extra>",
                    showlegend=False,
                    visible=(mode == initial_mode),
                )
            )
            edge_trace_indices_by_mode[mode].append(len(fig.data) - 1)

    # -------------------------------------------------------------------------
    # Node traces (one per execution mode for mode-filtered hover)
    # -------------------------------------------------------------------------
    nodes_in_order = list(graph.nodes)
    node_x = [float(pos[node][0]) for node in nodes_in_order]
    node_y = [float(pos[node][1]) for node in nodes_in_order]
    node_symbols = [
        marker_symbol_map.get(node_marker_mapping.get(node, "o"), "circle")
        for node in nodes_in_order
    ]
    node_colors = [node_color_mapping.get(node, "white") for node in nodes_in_order]

    node_hover_by_mode: dict[str, list[str]] = {}
    for mode in mode_order:
        node_hover_by_mode[mode] = [
            _node_hover_html(
                node,
                _filter_node_data_for_mode(node_data_mapping.get(node, {}), mode),
            )
            for node in nodes_in_order
        ]

    annotations_by_mode: dict[str, list[dict[str, Any]]] = {
        mode: _build_node_label_annotations(
            nodes_in_order, node_x, node_y, node_hover_by_mode[mode]
        )
        for mode in mode_order
    }

    node_trace_indices_by_mode: dict[str, int] = {}
    for mode in mode_order:
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                hovertemplate="%{customdata}<extra></extra>",
                customdata=node_hover_by_mode[mode],
                marker={
                    "size": 28,
                    "color": node_colors,
                    "symbol": node_symbols,
                    "line": {"color": "black", "width": 1},
                },
                showlegend=False,
                visible=(mode == initial_mode),
            )
        )
        node_trace_indices_by_mode[mode] = len(fig.data) - 1

    # -------------------------------------------------------------------------
    # Legend entries
    # Keep them always visible
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"size": 0},
            name="— Nodes —",
            showlegend=True,
            visible=True,
        )
    )
    separator_trace_index = len(fig.data) - 1

    # Node legend
    seen_node_types: dict[NodeType, str] = {}
    node_legend_trace_indices: list[int] = []

    for node in graph.nodes:
        node_type = node_configs[node][1].node_type
        if node_type in seen_node_types:
            continue
        seen_node_types[node_type] = node

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "size": 12,
                    "color": node_color_mapping[node],
                    "symbol": marker_symbol_map.get(
                        node_marker_mapping[node], "circle"
                    ),
                    "line": {"color": "black", "width": 1},
                },
                name=node_type.name,
                legendgroup="nodes",
                showlegend=True,
                visible=True,
            )
        )
        node_legend_trace_indices.append(len(fig.data) - 1)

    # Edge legend
    color_to_label = {
        "black": "All modes",
        "green": "Training",
        "orange": "Inference",
        "purple": "Evaluation",
        "gray": "Other",
    }

    edge_legend_trace_indices: list[int] = []
    for color in ["black", "green", "orange", "purple"]:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line={"color": color, "width": 3},
                name=color_to_label[color],
                legendgroup="edges",
                showlegend=True,
                visible=True,
            )
        )
        edge_legend_trace_indices.append(len(fig.data) - 1)

    # -------------------------------------------------------------------------
    # Dropdown visibility masks
    # -------------------------------------------------------------------------
    always_visible = {
        separator_trace_index,
        *node_legend_trace_indices,
        *edge_legend_trace_indices,
    }

    buttons = []
    total_traces = len(fig.data)

    for mode in mode_order:
        visible_mask = [False] * total_traces

        for idx in always_visible:
            visible_mask[idx] = True

        for idx in edge_trace_indices_by_mode[mode]:
            visible_mask[idx] = True

        visible_mask[node_trace_indices_by_mode[mode]] = True

        label = mode.capitalize()

        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [
                    {"visible": visible_mask},
                    {
                        "title": {
                            "text": (
                                f"{title} — {label}"
                                f"<br><span style='font-size:12px;color:gray'>{version!s}</span>"
                            ),
                            "x": 0.5,
                            "xanchor": "center",
                        },
                        "annotations": annotations_by_mode[mode],
                    },
                ],
            }
        )

    width = int(figsize[0] * 100)
    height = int(figsize[1] * 100)

    fig.update_layout(
        title={
            "text": (
                f"{title} — {initial_mode.capitalize()}"
                f"<br><span style='font-size:12px;color:gray'>{version!s}</span>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        annotations=annotations_by_mode[initial_mode],
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hoverlabel={
            "bgcolor": "rgba(30,30,30,0.95)",
            "font": {"color": "white", "size": 12},
        },
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
        legend={
            "title": "Legend",
            "orientation": "v",
            "x": 1.02,
            "y": 1,
            "xanchor": "left",
            "yanchor": "top",
            "bordercolor": "rgba(0,0,0,0.1)",
            "borderwidth": 1,
        },
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.01,
                "xanchor": "left",
                "y": 1.12,
                "yanchor": "top",
            }
        ],
    )

    return fig
