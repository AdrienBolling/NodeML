"""Define the base classes for NodeML Pipelines.

NodeML Pipelines are akin to graphs. The nodes are the components of the pipeline (Models, Data sources, Transforms, etc.), and edges represent the data flow between these nodes.
"""

import hashlib
import json
import pickle
import re
from collections.abc import Callable, Mapping
from functools import wraps
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from nodeml.core.common.data.data import (
    DATA_CATEGORY_MAPPING,
    DATA_STRUCTURE_MAPPING,
    Data,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.common.logging import Logger
from nodeml.core.common.typechecking.typeguards import has_params
from nodeml.core.common.version import Version
from nodeml.core.nodes.node import Node, NodeConfig, NodeType, Port
from nodeml.core.nodes.registry.node_registry import NODE_REGISTRY
from nodeml.core.pipeline.render import render_pipeline_graph_plotly

_log = Logger("nodeml.pipeline")

_ = Data  # For type checking purposes


def decompile(method: Callable[..., Any]):  # noqa: ANN201
    """Mark a Pipeline method so that calling it resets the compiled state.

    Any method decorated with ``@decompile`` will automatically set the
    pipeline back to *uncompiled* after execution, forcing a re-compilation
    before the next run.

    Args:
        method: The Pipeline method to wrap.

    Returns:
        A wrapper that calls *method* and then marks the pipeline as
        uncompiled.

    """

    @wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(self, Pipeline):
            message = f"Expected 'self' to be an instance of Pipeline, got {type(self)} instead."
            raise TypeError(message)
        result: Any = method(self, *args, **kwargs)
        self._uncompile()
        return result

    return wrapper


class PipelineSettings(BaseSettings):
    """Settings for a NodeML Pipeline.

    Attributes:
        autoprune: When ``True``, unconnected nodes are automatically removed
            during compilation.

    """

    autoprune: bool = True


class Edge(BaseModel):
    """Define a node-to-node connection in a Pipeline.

    Attributes:
        source: Name of the source node.
        target: Name of the target node.
        ports_map: List of ``(source_port, target_port)`` tuples describing
            which output port of the source feeds into which input port of
            the target.  Ignored when the target is the Sink node.

    """

    source: str
    target: str
    ports_map: list[tuple[str, str]]


class PipelineConfig(BaseModel):
    """Define the global, user-facing configuration for a NodeML Pipeline.

    Attributes:
        nodes: Mapping of node names to ``(node_class_name, NodeConfig | None)``
            tuples.  A ``None`` config causes the default config to be used.
        edges: List of edges describing connections between nodes.
        name: Human-readable pipeline name.
        version: Semantic version of the pipeline.
        settings: Pipeline-level settings (e.g. autoprune).

    """

    nodes: Mapping[str, tuple[str, NodeConfig | None]] = {}
    edges: list[Edge] = []
    name: str = "My Pipeline"
    version: Version = Version(major=0, minor=1, patch=0)
    settings: PipelineSettings = PipelineSettings()


class _InternalPipelineConfig(BaseModel):
    """Internal configuration for a NodeML Pipeline.

    Stores fully-instantiated node configs and edge data keyed by
    ``(source, target)`` for fast lookup.  This is not part of the
    user-facing API.

    Attributes:
        nodes: Mapping of node names to ``(node_class_name, NodeConfig)``
            tuples.  All configs are guaranteed to be fully instantiated.
        edges: Edges keyed by ``(source_name, target_name)`` for O(1)
            lookup.
        name: Human-readable pipeline name.
        version: Semantic version of the pipeline.
        settings: Pipeline-level settings.

    """

    nodes: dict[str, tuple[str, NodeConfig]] = {}
    edges: dict[tuple[str, str], Edge] = {}
    name: str = "My Pipeline"
    version: Version = Version(major=0, minor=1, patch=0)
    settings: PipelineSettings = PipelineSettings()


def _sanitize_name(name: str) -> str:
    """Convert a pipeline name to a file-safe string.

    Spaces and runs of non-alphanumeric characters (except ``-`` and
    ``_``) are replaced by a single underscore, then leading/trailing
    underscores are stripped.
    """
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")


class Pipeline:
    """Base class for a NodeML Pipeline.

    A Pipeline is a directed graph where nodes represent processing units (like models, data sources, transforms) and edges represent the flow of data between these nodes.
    """

    def __init__(
        self,
        *,
        config: PipelineConfig | None = None,
    ) -> None:
        """Initialize the Pipeline with the given configuration.

        Pipelines are typically created empty and built programmatically, or
        loaded from a saved config.  Call :meth:`compile` before running the
        pipeline to instantiate all node objects.

        Args:
            config: Global configuration for the pipeline.  When ``None``,
                an empty :class:`PipelineConfig` is used.

        """
        # Initialize the pipeline internals
        self._compiled = False
        user_provided_config = config if config is not None else PipelineConfig()

        # Fully instantiate the user_provided_config
        self._config = self._full_init_config(user_provided_config)

        # Initialize the node objects dictionary (node name to node instance)
        self._node_objects: dict[str, Node] = {}
        # Initialize the graph structure (adjacency list representation)
        self._graph: nx.DiGraph = nx.DiGraph()
        # Initialize the sink node name
        self._sink_node_name: str | None = self._init_sink_node_name()
        # Instantiate the Sink Edges indexes
        self._sink_edges_idx: list[tuple[str, str]] = self._init_sink_edges_idx()

        # Validate the edges
        for edge in self._config.edges.values():
            self._validate_edge(edge)

        # Initialize the graph representation according to the provided config
        self._init_graph_repr()

    #### Properties ####
    # --- Properties for pipeline attributes ---
    @property
    def config(self) -> PipelineConfig:
        """Get the user-facing pipeline configuration."""
        return PipelineConfig(
            nodes=self._config.nodes,
            edges=list(self._config.edges.values()),
            name=self._config.name,
            version=self._config.version,
            settings=self._config.settings,
        )

    @property
    def internal_config(self) -> _InternalPipelineConfig:
        """Get the internal pipeline configuration."""
        return self._config

    @property
    def node_objects(self) -> dict[str, Node]:
        """Get the dictionary of node objects in the pipeline."""
        return self._node_objects

    @property
    def graph(self) -> nx.DiGraph:
        """Get the graph structure of the pipeline."""
        return self._graph

    # --- Properties for quick access ---
    @property
    def nodes(self) -> dict[str, tuple[str, NodeConfig]]:
        """Get the nodes in the pipeline."""
        return self._config.nodes

    @property
    def edges(self) -> list[Edge]:
        """Get the edges in the pipeline."""
        return list(self._config.edges.values())

    @property
    def edges_dict(self) -> dict[tuple[str, str], Edge]:
        """Get the internal edges of the pipeline."""
        return self._config.edges

    @property
    def name(self) -> str:
        """Get the name of the pipeline."""
        return self._config.name

    @property
    def version(self) -> Version:
        """Get the version of the pipeline."""
        return self._config.version

    @property
    def settings(self) -> PipelineSettings:
        """Get the settings of the pipeline."""
        return self._config.settings

    @property
    def file_basename(self) -> str:
        """File-safe base name: ``{sanitized_name}_v{version}``.

        Used by :meth:`save_params_to_dir`, :meth:`save_config_to_dir`,
        and :meth:`save_html_to_dir` so every artefact shares the same
        prefix.

        """
        return f"{_sanitize_name(self.name)}_v{self.version}"

    @property
    def compiled(self) -> bool:
        """Check if the pipeline is compiled."""
        return self._compiled

    @property
    def sink_node_name(self) -> str | None:
        """Get the name of the sink node in the pipeline, if it exists."""
        return self._sink_node_name

    @property
    def sink_edges_dict(self) -> dict[tuple[str, str], Edge]:
        """Get the edges connected to the sink node."""
        if self.sink_node_name is None:
            return {}
        return {idx: self._config.edges[idx] for idx in self._sink_edges_idx}

    @property
    def graph_wo_metrics(self) -> nx.DiGraph:
        """Return a filtered graph view excluding metric nodes and their edges."""

        def filter_nodes(node_name: str) -> bool:
            node_type, _ = self._config.nodes[node_name]
            return node_type != NodeType.METRIC

        def filter_edges(source: str, target: str) -> bool:
            edge = self._config.edges.get((source, target))
            if edge is None:
                return True
            source_node_type, _ = self._config.nodes[source]
            target_node_type, _ = self._config.nodes[target]
            return NodeType.METRIC not in (source_node_type, target_node_type)

        return nx.subgraph_view(
            self._graph, filter_node=filter_nodes, filter_edge=filter_edges
        )

    #### Public API ####
    # --- Public API for pipeline management ---

    def add_node(
        self, node_name: str, node_type: str, node_config: NodeConfig | None = None
    ) -> None:
        """Add a node to the pipeline.

        Args:
            node_name: Unique name for the node within this pipeline.
            node_type: Registered node class name (looked up in the node
                registry).
            node_config: Optional configuration.  When ``None``, the default
                config for *node_type* is used.

        Raises:
            ValueError: If *node_name* already exists or a second Sink node
                is added.

        """
        if node_name in self._config.nodes:
            message = f"Node '{node_name}' already exists in the pipeline."
            raise ValueError(message)
        if node_config is None:
            node_config = self._get_default_node_config(node_type)
        if node_config.node_type == NodeType.SINK and self.sink_node_name is not None:
            message = f"Sink node '{self.sink_node_name}' already exists in the pipeline. Cannot add another sink node '{node_name}'."
            raise ValueError(message)
        # Add the node to the pipeline configuration
        self._config.nodes[node_name] = (node_type, node_config)
        # Track the sink node name so compile() and related checks see it.
        if node_config.node_type == NodeType.SINK:
            self._sink_node_name = node_name
        # Add the node to the graph structure
        self._add_node_to_nx(node_name)

    def remove_node(self, node_name: str) -> None:
        """Remove a node and all its connected edges from the pipeline.

        Args:
            node_name: Name of the node to remove.

        Raises:
            ValueError: If *node_name* does not exist.

        """
        if node_name not in self._config.nodes:
            message = f"Node '{node_name}' does not exist in the pipeline."
            raise ValueError(message)
        # If this node is the Sink node, we need to remove all edges connected to it from the list of sink edges
        if node_name == self.sink_node_name:
            for idx in self._sink_edges_idx:
                del self._config.edges[idx]
            self._sink_edges_idx = []
            self._sink_node_name = None
        # Remove the node from the pipeline configuration
        del self._config.nodes[node_name]
        # Remove the node from the graph structure
        self._remove_node_from_nx(node_name)

    def update_node(self, node_name: str, node_config: NodeConfig) -> None:
        """Update a node's configuration in the pipeline.

        Args:
            node_name: Name of the existing node to update.
            node_config: New configuration to assign to the node.

        Raises:
            ValueError: If *node_name* does not exist.

        """
        if node_name not in self._config.nodes:
            message = f"Node '{node_name}' does not exist in the pipeline."
            raise ValueError(message)
        node_type, _ = self._config.nodes[node_name]
        # Update the node configuration in the pipeline configuration
        self._config.nodes[node_name] = (node_type, node_config)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the pipeline.

        If an edge between the same source and target already exists, the
        port mappings are merged (duplicates removed).

        Args:
            edge: The edge to add.

        Raises:
            ValueError: If the edge originates from the Sink node or fails
                validation.

        """
        # Validate the edge before adding it to the pipeline
        self._validate_edge(
            edge
        )  # The edges are also checked at compile time just in case, but we want to raise errors early mainly for rendering
        # If an edge already exists between the source and target nodes, we will just update the ports_map
        if edge.target == self.sink_node_name:
            # if the target is the Sink node, we need to add this edge to the list of sink edges
            self._sink_edges_idx.append((edge.source, edge.target))
        if edge.source == self.sink_node_name:
            message = f"Edges with source as the Sink node '{self.sink_node_name}' are not allowed. Cannot add edge from '{edge.source}' to '{edge.target}'."
            raise ValueError(message)
        if self._graph.has_edge(edge.source, edge.target):
            existing_edge = self._config.edges[(edge.source, edge.target)]
            existing_edge.ports_map = list(
                set(existing_edge.ports_map + edge.ports_map)
            )  # Merge the ports_map while avoiding duplicates
            self._config.edges[(edge.source, edge.target)] = existing_edge
            self._update_edge_in_nx(existing_edge)
        else:
            # Add the edge to the pipeline configuration
            self._config.edges[(edge.source, edge.target)] = edge
            # Add the edge to the graph structure
            self._add_edge_to_nx(edge)

    def remove_edge(self, source: str, target: str) -> None:
        """Remove an edge from the pipeline.

        Args:
            source: Name of the source node.
            target: Name of the target node.

        Raises:
            ValueError: If the edge does not exist.

        """
        if (source, target) not in self._config.edges:
            message = (
                f"Edge from '{source}' to '{target}' does not exist in the pipeline."
            )
            raise ValueError(message)
        # If this edge is connected to the Sink node, we need to remove it from the list of sink edges
        if target == self.sink_node_name:
            self._sink_edges_idx.remove((source, target))
        # Remove the edge from the pipeline configuration
        del self._config.edges[(source, target)]
        # Remove the edge from the graph structure
        self._remove_edge_from_nx(Edge(source=source, target=target, ports_map=[]))

    def update_edge(self, edge: Edge) -> None:
        """Update an edge's ports_map in the pipeline.

        Args:
            edge: Edge with the updated ``ports_map``.  The ``source`` and
                ``target`` fields identify the existing edge to update.

        Raises:
            ValueError: If the edge does not exist.

        """
        if (edge.source, edge.target) not in self._config.edges:
            message = f"Edge from '{edge.source}' to '{edge.target}' does not exist in the pipeline."
            raise ValueError(message)
        if edge.target == self.sink_node_name:
            # if the target is the Sink node, we need to update this edge in the list of sink edges
            idx = self._sink_edges_idx.index((edge.source, edge.target))
            self._sink_edges_idx[idx] = (edge.source, edge.target)
        # Update the edge in the pipeline configuration
        self._config.edges[(edge.source, edge.target)] = edge
        # Update the edge in the graph structure
        self._update_edge_in_nx(edge)

    def compile(self) -> None:
        """Compile the pipeline by instantiating all node objects and validating the graph structure.

        Raises:
            ValueError: If the pipeline is already compiled or validation
                fails (e.g. missing Sink node, cycles, incompatible ports).

        """
        if self.compiled:
            message = "Pipeline is already compiled."
            raise ValueError(message)
        self._compile()

    ## --- Public API for getters ---
    def get_node_config(self, node_name: str) -> NodeConfig:
        """Get a node's configuration from the pipeline.

        Args:
            node_name: Name of the node.

        Returns:
            The :class:`NodeConfig` associated with *node_name*.

        Raises:
            ValueError: If *node_name* does not exist.

        """
        if node_name not in self._config.nodes:
            message = f"Node '{node_name}' does not exist in the pipeline."
            raise ValueError(message)
        return self._config.nodes[node_name][1]

    def get_validated_sink_node_name(self) -> str:
        """Get the name of the sink node, raising if none is configured.

        Returns:
            The sink node name.

        Raises:
            ValueError: If no Sink node exists in the pipeline.

        """
        if self.sink_node_name is None:
            message = "No Sink node found in the pipeline. A pipeline must have one Sink node to be valid."
            raise ValueError(message)
        return self.sink_node_name

    def get_edge(self, source: str, target: str) -> Edge:
        """Get an edge's configuration from the pipeline.

        Args:
            source: Name of the source node.
            target: Name of the target node.

        Returns:
            The :class:`Edge` between *source* and *target*.

        Raises:
            ValueError: If the edge does not exist.

        """
        if (source, target) not in self._config.edges:
            message = (
                f"Edge from '{source}' to '{target}' does not exist in the pipeline."
            )
            raise ValueError(message)
        return self._config.edges[(source, target)]

    def get_source_node_names(self) -> list[str]:
        """Get the names of all source nodes in the pipeline.

        Returns:
            List of node names whose type is :attr:`NodeType.SOURCE`.

        """
        source_node_names = []
        for node_name, (_, node_config) in self._config.nodes.items():
            if node_config.node_type == NodeType.SOURCE:
                source_node_names.append(node_name)
        return source_node_names

    def get_metric_node_names(self) -> list[str]:
        """Get the names of all metric nodes in the pipeline.

        Returns:
            List of node names whose type is :attr:`NodeType.METRIC`.

        """
        metric_node_names = []
        for node_name, (_, node_config) in self._config.nodes.items():
            if node_config.node_type == NodeType.METRIC:
                metric_node_names.append(node_name)
        return metric_node_names

    ## --- Public API for params management ---
    def get_params(self) -> dict[str, Any]:
        """Get the parameters of all parameterised nodes in the pipeline.

        Returns:
            Dict mapping node names to their parameter dicts.

        """
        return {
            node_name: node_object.get_params()
            for node_name, node_object in self.node_objects.items()
            if has_params(node_object)
        }

    def set_params(self, params: dict[str, dict[str, Any]]) -> None:
        """Set node parameters from a nested dict.

        Args:
            params: Mapping of ``{node_name: {param_name: value}}``.

        Raises:
            ValueError: If a node does not exist or does not support
                parameters.

        """
        for node_name, node_params in params.items():
            if node_name not in self.node_objects:
                message = f"Node '{node_name}' does not exist in the pipeline. Cannot set parameters for non-existing node."
                raise ValueError(message)
            node_object = self.node_objects[node_name]
            if not has_params(node_object):
                message = f"Node '{node_name}' does not have parameters. Cannot set parameters for node that does not have parameters."
                raise ValueError(message)
            node_object.set_params(node_params)

    def save_params_to_dir(self, dir_path: str) -> None:
        """Save all node parameters to a pickle file in the given directory.

        Pickle is used so that arbitrary Python objects inside the aggregated
        params dict (numpy arrays, sklearn internals, torch tensors, etc.) are
        handled transparently.

        Args:
            dir_path: Directory in which the pickle file will be written.

        """
        params = self.get_params()
        file_name = f"{dir_path}/{self.file_basename}_params.pkl"
        with Path(file_name).open("wb") as f:
            pickle.dump(params, f)

    def load_params_from_dir(self, dir_path: str) -> None:
        """Load node parameters from a previously saved pickle file.

        Args:
            dir_path: Directory containing the parameter pickle file.

        """
        file_name = f"{dir_path}/{self.file_basename}_params.pkl"
        with Path(file_name).open("rb") as f:
            params = pickle.load(f)  # noqa: S301 - only load files you trust
        self.set_params(params)

    def save_config_to_dir(self, dir_path: str) -> None:
        """Save the pipeline configuration as a JSON file.

        Args:
            dir_path: Directory in which the JSON file will be written.

        """
        file_name = f"{dir_path}/{self.file_basename}_config.json"
        Path(file_name).write_text(self.config.model_dump_json(indent=2))

    def save_html_to_dir(self, dir_path: str, **render_kwargs: Any) -> None:
        """Render the pipeline graph and save it as an HTML file.

        Args:
            dir_path: Directory in which the HTML file will be written.
            **render_kwargs: Forwarded to :meth:`render_to_html`
                (``title``, ``backend``, ``figsize``, ``full_html``).

        """
        html = self.render_to_html(**render_kwargs)
        file_name = f"{dir_path}/{self.file_basename}_graph.html"
        Path(file_name).write_text(html)

    ## --- Public API for identity ---
    def hash(self) -> str:
        """Return a stable hex digest identifying this pipeline's layout and config.

        The digest is a SHA-256 hash derived from the user-facing
        :class:`PipelineConfig`: node names, node types, node configs,
        edges (with port mappings), plus pipeline name, version, and
        settings.  Nodes, edges, and port mappings are sorted canonically
        so construction order does not perturb the result.

        Per-instance UUIDs on :class:`NodeConfig` are declared as
        ``PrivateAttr`` and are therefore excluded from :meth:`model_dump`,
        keeping the hash content-addressable: two pipelines built from the
        same :class:`PipelineConfig` produce the same hash.

        Returns:
            Hex-encoded SHA-256 digest of the canonical pipeline config.

        """
        payload = self.config.model_dump(mode="json")
        payload["nodes"] = dict(sorted(payload["nodes"].items()))
        payload["edges"] = sorted(
            payload["edges"], key=lambda e: (e["source"], e["target"])
        )
        for edge in payload["edges"]:
            edge["ports_map"] = sorted(edge["ports_map"])
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    #### Internal methods ####
    # --- Internal methods for pipeline management ---
    def _full_init_config(self, config: PipelineConfig) -> _InternalPipelineConfig:
        """Convert a user-facing :class:`PipelineConfig` into an internal one.

        Ensures every :class:`NodeConfig` is fully instantiated (defaults
        filled in) and edges are keyed by ``(source, target)`` for fast
        lookup.

        Args:
            config: The user-provided pipeline configuration.

        Returns:
            A fully instantiated :class:`_InternalPipelineConfig`.

        """
        fully_instantiated_nodes = self._full_init_node_configs(config.nodes)
        edges_dict = {}
        for edge in config.edges:
            if (edge.source, edge.target) in edges_dict:
                # Merge the ports_map of the edge if an edge between the same source and target already exists
                existing_edge = edges_dict[(edge.source, edge.target)]
                existing_edge.ports_map = list(
                    set(existing_edge.ports_map + edge.ports_map)
                )  # Merge the ports_map while avoiding duplicates
                edges_dict[(edge.source, edge.target)] = existing_edge
            else:
                edges_dict[(edge.source, edge.target)] = edge
        return _InternalPipelineConfig(
            nodes=fully_instantiated_nodes,
            edges=edges_dict,
            name=config.name,
            version=config.version,
            settings=config.settings,
        )

    def _instantiate_node_objects(self) -> None:
        """Instantiate the node objects based on the current pipeline configuration."""
        for node_name, (node_type, node_config) in self._config.nodes.items():
            node_class = NODE_REGISTRY.get_node_class(node_type)
            node_object = node_class(config=node_config)
            self._node_objects[node_name] = node_object
            _log.debug(
                "Node instantiated",
                node_name=node_name,
                node_type=node_config.node_type,
            )

    def _prune(self) -> None:
        """Prune the pipeline by removing any nodes that are not connected to any Node."""
        # Get the set of all nodes that are connected to at least one edge
        connected_nodes = set()
        for edge in self.edges:
            connected_nodes.add(edge.source)
            connected_nodes.add(edge.target)

        # Get the set of all nodes in the pipeline
        all_nodes = set(self.nodes.keys())

        # Get the set of unconnected nodes
        unconnected_nodes = all_nodes - connected_nodes

        # Remove unconnected nodes from the pipeline
        for node_name in unconnected_nodes:
            self.remove_node(node_name)

    # --- Internal methods for Nodes ---
    def _full_init_node_configs(
        self, nodes: Mapping[str, tuple[str, NodeConfig | None]]
    ) -> dict[str, tuple[str, NodeConfig]]:
        """Ensure every node has a fully instantiated :class:`NodeConfig`.

        Args:
            nodes: Raw node mapping from the user config.

        Returns:
            A new dict with ``None`` configs replaced by defaults.

        """
        fully_instantiated_nodes = {}
        for node_name, (node_type, node_config) in nodes.items():
            if node_config is not None:
                fully_instantiated_nodes[node_name] = (node_type, node_config)
            else:
                # If node_config is None, we need to create a default NodeConfig based on the node_type
                default_node_config = self._get_default_node_config(node_type)
                fully_instantiated_nodes[node_name] = (node_type, default_node_config)
        return fully_instantiated_nodes

    def _get_default_node_config(self, node_class_name: str) -> NodeConfig:
        """Return a default :class:`NodeConfig` for the given registered class name.

        Args:
            node_class_name: Registered node class name.

        """
        node_conf_class = NODE_REGISTRY.get_node_config_class(node_class_name)
        return node_conf_class()

    def _init_sink_node_name(self) -> str | None:
        """Initialize the sink node name based on the current pipeline configuration."""
        sink_node_name = None
        for node_name, (_, node_config) in self._config.nodes.items():
            if node_config.node_type == NodeType.SINK:
                if sink_node_name is not None:
                    message = f"Multiple Sink nodes found in the pipeline ('{sink_node_name}' and '{node_name}'). A pipeline can only have one Sink node."
                    raise ValueError(message)
                sink_node_name = node_name
        return sink_node_name

    # --- Internal methods for edges ---
    def _init_sink_edges_idx(self) -> list[tuple[str, str]]:
        """Initialize the list of sink edges indexes based on the current pipeline configuration."""
        sink_edges_idx: list[tuple[str, str]] = []
        if self.sink_node_name is not None:
            sink_edges_idx = [
                (edge.source, edge.target)
                for edge in self.edges
                if edge.target == self.sink_node_name
            ]
        return sink_edges_idx

    # --- Internal methods for NX graph management ---
    def _init_graph_repr(self) -> None:
        """Initialize the graph structure of the pipeline based on the current configuration."""
        for node_name in self._config.nodes.keys():
            self._add_node_to_nx(node_name)
        for edge in self._config.edges.values():
            self._add_edge_to_nx(edge)

    def _add_edge_to_nx(self, edge: Edge) -> None:
        """Add an edge to the graph structure of the pipeline."""
        source_node_name = edge.source
        target_node_name = edge.target

        # Add the edge to the graph structure
        self._graph.add_edge(source_node_name, target_node_name)

        # Set the ports map as edge attributes in the graph
        self._graph.edges[source_node_name, target_node_name]["ports_map"] = (
            edge.ports_map
        )

    def _add_node_to_nx(self, node_name: str) -> None:
        """Add a node to the graph structure of the pipeline."""
        self._graph.add_node(node_name)

    def _update_edge_in_nx(self, edge: Edge) -> None:
        """Update an edge in the graph structure of the pipeline."""
        source_node_name = edge.source
        target_node_name = edge.target

        # Update the ports map as edge attributes in the graph
        if self._graph.has_edge(source_node_name, target_node_name):
            self._graph.edges[source_node_name, target_node_name]["ports_map"] = (
                edge.ports_map
            )
        else:
            message = f"Edge from '{source_node_name}' to '{target_node_name}' does not exist in the graph. Cannot update non-existing edge."
            raise ValueError(message)

    def _remove_edge_from_nx(self, edge: Edge) -> None:
        """Remove an edge from the graph structure of the pipeline."""
        source_node_name = edge.source
        target_node_name = edge.target

        # Remove the edge from the graph structure
        if self._graph.has_edge(source_node_name, target_node_name):
            self._graph.remove_edge(source_node_name, target_node_name)
        else:
            message = f"Edge from '{source_node_name}' to '{target_node_name}' does not exist in the graph. Cannot remove non-existing edge."
            raise ValueError(message)

    def _remove_node_from_nx(self, node_name: str) -> None:
        """Remove a node from the graph structure of the pipeline."""
        # Remove the node from the graph structure
        if self._graph.has_node(node_name):
            self._graph.remove_node(node_name)
        else:
            message = f"Node '{node_name}' does not exist in the graph. Cannot remove non-existing node."
            raise ValueError(message)

    # --- Internal methods for validation ---
    def _validate(self) -> None:
        """Validate the pipeline layout and configuration.

        This method checks for issues such as:
        - Presence of a Sink node
        - Validity of edges (source and target nodes should exist, ports_map should be valid, etc.)
        - Absence of cycles in the graph (if not allowed)

        Raises:
            ValueError: If any validation check fails.

        """
        # Check for the presence of a Sink node
        if self.sink_node_name is None:
            message = "Pipeline must contain a Sink node, but no Sink node was found."
            raise ValueError(message)

        # Check the validity of edges
        for edge in self.edges:
            self._validate_edge(edge)

        # Check for cycles in the graph (if not allowed)
        if not nx.is_directed_acyclic_graph(self._graph):
            message = "Pipeline graph contains cycles, but cycles are not allowed."
            raise ValueError(message)

    def _validate_edge(self, edge: Edge) -> None:
        """Validate an edge's configuration and port compatibility.

        Checks performed:
        - Source and target nodes exist in the pipeline.
        - All ports in ``ports_map`` exist on their respective nodes.
        - Data structures and categories are compatible between ports.
        - Execution modes of connected ports are compatible.
        - No edge originates from the Sink node.

        Args:
            edge: The edge to validate.

        Raises:
            ValueError: If any validation check fails.

        """
        source_node_conf = self.get_node_config(edge.source)
        target_node_conf = self.get_node_config(edge.target)

        # For now only do a limited check of the compatibility of the ports
        source_out_ports = source_node_conf.out_ports
        target_in_ports = target_node_conf.in_ports

        for source_port, target_port in edge.ports_map:
            # Check that all source ports in the ports_map exist in the source node's out_ports
            if source_port not in source_out_ports:
                message = f"Source port '{source_port}' in edge from '{edge.source}' to '{edge.target}' does not exist in the source node's out_ports."
                message += f"\nAvailable source ports: {list(source_out_ports.keys())}"
                raise ValueError(message)
            if target_port not in target_in_ports:
                message = f"Target port '{target_port}' in edge from '{edge.source}' to '{edge.target}' does not exist in the target node's in_ports."
                message += f"\nAvailable target ports: {list(target_in_ports.keys())}"
                raise ValueError(message)

            # Check that data structures match between the source and target ports
            self._is_compatible_structures(
                source_out_ports,
                target_in_ports,
                edge.ports_map,
                edge.source,
                edge.target,
            )

            # Check that data categories match between the source and target ports
            self._is_compatible_categories(
                source_out_ports,
                target_in_ports,
                edge.ports_map,
                edge.source,
                edge.target,
            )

            # Check that the execution modes of the ports are compatible (can't have an edge from a port that only executes in "train" mode to a port that only executes in "inference" mode for example)
            self._are_execution_modes_compatible(
                source_out_ports,
                target_in_ports,
                edge.ports_map,
                edge.source,
                edge.target,
            )

            # Finally, check that no edge originates from the Sink node
            if source_node_conf.node_type == NodeType.SINK:
                message = f"Edge from '{edge.source}' to '{edge.target}' is not valid because it originates from a Sink node, which cannot have outgoing edges."
                raise ValueError(message)

    def _are_execution_modes_compatible(
        self,
        source_out_ports: dict[str, Port],
        target_in_ports: dict[str, Port],
        ports_map: list[tuple[str, str]],
        source: str,
        target: str,
    ) -> None:
        """Check that the execution modes of the source and target ports in the given ports_map are compatible."""
        for source_port, target_port in ports_map:
            source_modes = source_out_ports[source_port].mode
            target_modes = target_in_ports[target_port].mode
            if not (set(source_modes).intersection(set(target_modes))) and not (
                NodeExecutionMode.ALL in source_modes
                or NodeExecutionMode.ALL in target_modes
            ):
                message = f"Execution modes of source port '{source_port}' in edge from '{source}' to '{target}' are not compatible with execution modes of target port '{target_port}'.\n"
                message += (
                    f"Source port '{source_port}' execution modes: {source_modes}\n"
                )
                message += (
                    f"Target port '{target_port}' execution modes: {target_modes}\n"
                )
                message += "Please ensure that the execution modes are compatible."
                raise ValueError(message)

    def _is_compatible_structures(
        self,
        source_out_ports: dict[str, Port],
        target_in_ports: dict[str, Port],
        ports_map: list[tuple[str, str]],
        source: str,
        target: str,
    ) -> None:
        """Check that the data structures of the source and target ports in the given ports_map are compatible."""
        for source_port, target_port in ports_map:
            source_structure = DATA_STRUCTURE_MAPPING[
                source_out_ports[source_port].data_structure
            ]
            target_structure = DATA_STRUCTURE_MAPPING[
                target_in_ports[target_port].data_structure
            ]
            target_sub_source = issubclass(target_structure, source_structure)
            source_sub_target = issubclass(source_structure, target_structure)
            if not (target_sub_source or source_sub_target):
                message = f"Data structure of source port '{source_port}' in edge from '{source}' to '{target}' is not compatible with data structure of target port '{target_port}'."
                raise ValueError(message)
            if not (target_sub_source and source_sub_target):
                _log.warning(
                    "Data structure mismatch (compatible but not identical)",
                    source=source,
                    target=target,
                    source_port=source_port,
                    target_port=target_port,
                )

    def _is_compatible_categories(
        self,
        source_out_ports: dict[str, Port],
        target_in_ports: dict[str, Port],
        ports_map: list[tuple[str, str]],
        source: str,
        target: str,
    ) -> None:
        """Check that the data categories of the source and target ports in the given ports_map are compatible."""
        for source_port, target_port in ports_map:
            source_categories = set(
                DATA_CATEGORY_MAPPING[
                    source_out_ports[source_port].data_category
                ].dtypes
            )
            target_categories = set(
                DATA_CATEGORY_MAPPING[target_in_ports[target_port].data_category].dtypes
            )
            if not (
                source_categories.issubset(target_categories)
                or target_categories.issubset(source_categories)
            ):
                message = f"Data category of source port '{source_port}' in edge from '{source}' to '{target}' is not compatible with data category of target port '{target_port}'.\n"
                message += f"Source port '{source_port}' data categories: {source_categories}\n"
                message += f"Target port '{target_port}' data categories: {target_categories}\n"
                message += "Please ensure that the data categories are compatible"
                raise ValueError(message)
            if target_categories.issubset(
                source_categories
            ) and not source_categories.issubset(target_categories):
                _log.warning(
                    "Data category superset (source broader than target)",
                    source=source,
                    target=target,
                    source_port=source_port,
                    target_port=target_port,
                )
            # In other cases the edge is valid no questions asked

    # --- Internal methods for compilation ---
    def _uncompile(self) -> None:
        """Signals the Pipeline is no longer fit for running."""
        self._compiled = False

    def _compile(self) -> None:
        """Perform the full compilation sequence.

        Steps executed in order:

        1. Clear previous node instances.
        2. Prune unconnected nodes (if ``autoprune`` is enabled).
        3. Validate the graph layout.
        4. Instantiate all node objects.

        """
        log = _log.bind(pipeline_name=self.name)
        log.log_phase("compilation", "start")

        try:
            # Clear the node objects
            self._node_objects = {}
            # If needed, prune the graph (remove Nodes without outgoing or incoming edges)
            if self.settings.autoprune:
                self._prune()

            # Validate the layout
            self._validate()

            # Now that everything is validated, instantiate the nodes
            self._instantiate_node_objects()
        except Exception as exc:
            log.exception("Compilation failed", exc)  # noqa: PLE1205, TRY401 - nodeml Logger API
            raise

        # Finish compilation
        self._compiled = True
        log.log_phase(
            "compilation",
            "end",
            params={
                "node_count": len(self._node_objects),
                "edge_count": len(self._config.edges),
            },
        )

    # --- Rendering utils ---
    def render(
        self,
        title: str | None = None,
        backend: str = "plotly",
        figsize: tuple[int, int] | None = None,
    ) -> None:
        """Render the pipeline graph and display it interactively.

        Args:
            title: Figure title.  Defaults to ``"Pipeline: <name>"``.
            backend: Rendering backend.  Currently only ``"plotly"`` is
                supported.
            figsize: ``(width, height)`` of the figure in inches.  ``None``
                (the default) scales the figure with the layout shape.

        Raises:
            ValueError: If *backend* is not supported.

        """
        if title is None:
            title = f"Pipeline: {self.name}"
        if backend == "plotly":
            fig = render_pipeline_graph_plotly(
                pipeline=self, title=title, figsize=figsize
            )
            fig.show()
        else:
            message = f"Backend '{backend}' is not supported for rendering. Supported backends are: ['plotly']."
            raise ValueError(message)

    def render_to_html(
        self,
        title: str | None = None,
        backend: str = "plotly",
        figsize: tuple[int, int] | None = None,
        *,
        full_html: bool = True,
    ) -> str:
        """Render the pipeline graph and return it as an HTML string.

        Args:
            title: Figure title.  Defaults to ``"Pipeline: <name>"``.
            backend: Rendering backend.  Currently only ``"plotly"`` is
                supported.
            figsize: ``(width, height)`` of the figure in inches.  ``None``
                (the default) scales the figure with the layout shape.
            full_html: When ``True``, return a self-contained HTML document;
                otherwise return only the ``<div>`` fragment.

        Returns:
            HTML string of the rendered graph.

        Raises:
            ValueError: If *backend* is not supported.

        """
        if title is None:
            title = f"Pipeline: {self.name}"
        if backend == "plotly":
            fig = render_pipeline_graph_plotly(
                pipeline=self, title=title, figsize=figsize
            )
            return fig.to_html(full_html=full_html)
        message = f"Backend '{backend}' is not supported for rendering. Supported backends are: ['plotly']."
        raise ValueError(message)
