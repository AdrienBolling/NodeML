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
from typing import Any, Literal

import networkx as nx
from pydantic import BaseModel, SerializeAsAny, field_validator
from pydantic_settings import BaseSettings

from nodeml.core.common.data.data import (
    DATA_CATEGORY_MAPPING,
    DATA_STRUCTURE_MAPPING,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.common.exceptions import (
    PipelineCompilationError,
    PipelineError,
    PipelineGraphError,
    PipelineValidationError,
)
from nodeml.core.common.logging import Logger
from nodeml.core.common.typechecking.typeguards import has_params
from nodeml.core.common.version import Version
from nodeml.core.nodes.node import Node, NodeConfig, NodeType, Port
from nodeml.core.nodes.registry.node_registry import NODE_REGISTRY
from nodeml.core.pipeline.render import render_pipeline_graph_plotly

_log = Logger("nodeml.pipeline")


def decompile[**P, T](method: Callable[P, T]) -> Callable[P, T]:
    """Mark a Pipeline method so that calling it resets the compiled state.

    Any method decorated with ``@decompile`` will automatically set the
    pipeline back to *uncompiled* after execution, forcing a re-compilation
    before the next run.  If the method raises, the pipeline does not
    change and keeps its compiled state.

    Args:
        method: The Pipeline method to wrap.

    Returns:
        A wrapper that calls *method* and then marks the pipeline as
        uncompiled.

    """

    @wraps(method)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        self = args[0]
        if not isinstance(self, Pipeline):
            message = f"Expected 'self' to be an instance of Pipeline, got {type(self)} instead."
            raise TypeError(message)
        result = method(*args, **kwargs)
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
            the target.  When the target is the Sink node, ``target_port`` is
            the name of a Sink port: :meth:`Pipeline.compile` creates that
            port from the definition of ``source_port``.

    """

    source: str
    target: str
    ports_map: list[tuple[str, str]]


def _merge_ports_maps(
    existing: list[tuple[str, str]], new: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Append the pairs of *new* that are not in *existing*, keeping the order."""
    return existing + [pair for pair in new if pair not in existing]


def _rebuild_node_configs(nodes: Any) -> Any:
    """Validate serialized node configs with their registered config class.

    A config that was dumped to JSON loses its concrete class.  Each entry
    names its registered node class, so the matching config class can
    rebuild every field (hyperparameters, running config, ports).
    Entries that already hold a :class:`NodeConfig` instance, or ``None``,
    are returned unchanged.
    """
    if not isinstance(nodes, Mapping):
        return nodes
    rebuilt: dict[str, Any] = {}
    for node_name, entry in nodes.items():
        if (
            isinstance(entry, list | tuple)
            and len(entry) == 2  # noqa: PLR2004 - (class name, config)
            and isinstance(entry[1], Mapping)
        ):
            class_name, raw_config = entry
            config_class = NODE_REGISTRY.get_node_config_class(class_name)
            rebuilt[node_name] = (class_name, config_class.model_validate(raw_config))
        else:
            rebuilt[node_name] = entry
    return rebuilt


class PipelineConfig(BaseModel):
    """Define the global, user-facing configuration for a NodeML Pipeline.

    The config serializes every field of the concrete node configs, and
    :meth:`model_validate` / :meth:`model_validate_json` rebuild the concrete
    config classes through the node registry.  So a config saved with
    :meth:`Pipeline.save_config_to_dir` loads back without loss.

    Attributes:
        nodes: Mapping of node names to ``(node_class_name, NodeConfig | None)``
            tuples.  A ``None`` config causes the default config to be used.
        edges: List of edges describing connections between nodes.
        name: Human-readable pipeline name.
        version: Semantic version of the pipeline.
        settings: Pipeline-level settings (e.g. autoprune).

    """

    nodes: Mapping[str, tuple[str, SerializeAsAny[NodeConfig] | None]] = {}
    edges: list[Edge] = []
    name: str = "My Pipeline"
    version: Version = Version(major=0, minor=1, patch=0)
    settings: PipelineSettings = PipelineSettings()

    @field_validator("nodes", mode="before")
    @classmethod
    def _rebuild_nodes(cls, nodes: Any) -> Any:
        """Rebuild serialized node configs with their concrete classes."""
        return _rebuild_node_configs(nodes)


class _InternalPipelineConfig(BaseModel):
    """Internal configuration for a NodeML Pipeline.

    Stores fully-instantiated node configs and edge data keyed by
    ``(source, target)`` for fast lookup.  This is not part of the
    user-facing API.

    Attributes:
        nodes: Mapping of node names to ``(node_class_name, NodeConfig)``
            tuples.  All configs are guaranteed to be fully instantiated.
        edges: Edges keyed by ``(source_name, target_name)`` for O(1)
            lookup.  The dict keeps the order in which edges were added.
        name: Human-readable pipeline name.
        version: Semantic version of the pipeline.
        settings: Pipeline-level settings.

    """

    nodes: dict[str, tuple[str, SerializeAsAny[NodeConfig]]] = {}
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

    Every method that changes the layout (nodes or edges) marks the pipeline
    as uncompiled.  Call :meth:`compile` again before the next run.
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

        # Validate the edges
        for edge in self._config.edges.values():
            self._validate_edge(edge)
        self._check_sink_port_names()

        # Initialize the graph representation according to the provided config
        self._init_graph_repr()

    # --- Alternative constructors ---
    @classmethod
    def from_config_file(cls, path: str | Path) -> "Pipeline":
        """Build an uncompiled pipeline from a JSON config file.

        Args:
            path: A file written by :meth:`save_config_to_dir`.

        Returns:
            A new, uncompiled :class:`Pipeline`.

        """
        config = PipelineConfig.model_validate_json(Path(path).read_text())
        return cls(config=config)

    @classmethod
    def load_from_dir(
        cls,
        dir_path: str | Path,
        file_basename: str | None = None,
        *,
        load_params: bool = True,
    ) -> "Pipeline":
        """Rebuild a pipeline saved with :meth:`save_config_to_dir`.

        The pipeline is compiled.  If *load_params* is ``True`` and the
        directory has the matching params file (see
        :meth:`save_params_to_dir`), the trained parameters are loaded too.

        Args:
            dir_path: Directory that holds the saved files.
            file_basename: The :attr:`file_basename` of the saved pipeline.
                When ``None``, the directory must hold exactly one
                ``*_config.json`` file.
            load_params: Load the trained parameters when the file exists.

        Returns:
            A compiled :class:`Pipeline`.

        Raises:
            PipelineError: If no config file, or more than one, matches.

        """
        directory = Path(dir_path)
        if file_basename is not None:
            config_path = directory / f"{file_basename}_config.json"
            if not config_path.is_file():
                message = f"Config file not found: {config_path}"
                raise PipelineError(message)
        else:
            candidates = sorted(directory.glob("*_config.json"))
            if len(candidates) != 1:
                message = (
                    f"Expected exactly one '*_config.json' file in {directory}, "
                    f"found {[c.name for c in candidates]}. Pass file_basename."
                )
                raise PipelineError(message)
            config_path = candidates[0]

        pipeline = cls.from_config_file(config_path)
        pipeline.compile()
        params_file = directory / f"{pipeline.file_basename}_params.pkl"
        if load_params and params_file.is_file():
            pipeline.load_params_from_dir(directory)
        return pipeline

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
    def resolved_nodes(self) -> dict[str, tuple[str, NodeConfig]]:
        """Get the nodes with the Sink config that :meth:`compile` uses.

        The stored Sink config does not change; the Sink config in this
        mapping has the ports that compilation creates from the Sink edges.
        """
        nodes = dict(self._config.nodes)
        if self.sink_node_name is not None:
            class_name, sink_config = nodes[self.sink_node_name]
            nodes[self.sink_node_name] = (
                class_name,
                self._build_sink_config(sink_config),
            )
        return nodes

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
        """Get the edges connected to the sink node, in the order they were added."""
        if self.sink_node_name is None:
            return {}
        return {
            key: edge
            for key, edge in self._config.edges.items()
            if edge.target == self.sink_node_name
        }

    @property
    def graph_wo_metrics(self) -> nx.DiGraph:
        """Return a filtered graph view excluding metric nodes and their edges."""

        def is_metric(node_name: str) -> bool:
            _, node_config = self._config.nodes[node_name]
            return node_config.node_type == NodeType.METRIC

        def filter_nodes(node_name: str) -> bool:
            return not is_metric(node_name)

        def filter_edges(source: str, target: str) -> bool:
            return not (is_metric(source) or is_metric(target))

        return nx.subgraph_view(
            self._graph, filter_node=filter_nodes, filter_edge=filter_edges
        )

    #### Public API ####
    # --- Public API for pipeline management ---

    @decompile
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
            PipelineValidationError: If *node_name* already exists or a
                second Sink node is added.

        """
        if node_name in self._config.nodes:
            message = f"Node '{node_name}' already exists in the pipeline."
            raise PipelineValidationError(message)
        if node_config is None:
            node_config = self._get_default_node_config(node_type)
        if node_config.node_type == NodeType.SINK and self.sink_node_name is not None:
            message = f"Sink node '{self.sink_node_name}' already exists in the pipeline. Cannot add another sink node '{node_name}'."
            raise PipelineValidationError(message)
        # Add the node to the pipeline configuration
        self._config.nodes[node_name] = (node_type, node_config)
        # Track the sink node name so compile() and related checks see it.
        if node_config.node_type == NodeType.SINK:
            self._sink_node_name = node_name
        # Add the node to the graph structure
        self._add_node_to_nx(node_name)

    @decompile
    def remove_node(self, node_name: str) -> None:
        """Remove a node and all its connected edges from the pipeline.

        Args:
            node_name: Name of the node to remove.

        Raises:
            PipelineValidationError: If *node_name* does not exist.

        """
        if node_name not in self._config.nodes:
            message = f"Node '{node_name}' does not exist in the pipeline."
            raise PipelineValidationError(message)
        # Remove every edge that starts or ends at this node.
        for key in [k for k in self._config.edges if node_name in k]:
            del self._config.edges[key]
        if node_name == self.sink_node_name:
            self._sink_node_name = None
        # Remove the node from the pipeline configuration and node objects
        del self._config.nodes[node_name]
        self._node_objects.pop(node_name, None)
        # Remove the node (and its edges) from the graph structure
        self._remove_node_from_nx(node_name)

    @decompile
    def update_node(self, node_name: str, node_config: NodeConfig) -> None:
        """Update a node's configuration in the pipeline.

        The edges of the node are validated against the new configuration.

        Args:
            node_name: Name of the existing node to update.
            node_config: New configuration to assign to the node.

        Raises:
            PipelineValidationError: If *node_name* does not exist, if the
                new config has another node type, or if an edge of the node
                is not valid with the new config.

        """
        if node_name not in self._config.nodes:
            message = f"Node '{node_name}' does not exist in the pipeline."
            raise PipelineValidationError(message)
        node_type, old_config = self._config.nodes[node_name]
        if node_config.node_type != old_config.node_type:
            message = (
                f"Cannot change the type of node '{node_name}' from "
                f"'{old_config.node_type}' to '{node_config.node_type}'. "
                "Remove the node and add a new one instead."
            )
            raise PipelineValidationError(message)
        self._config.nodes[node_name] = (node_type, node_config)
        try:
            for edge in self.edges:
                if node_name in (edge.source, edge.target):
                    self._validate_edge(edge)
        except PipelineValidationError:
            self._config.nodes[node_name] = (node_type, old_config)
            raise

    @decompile
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the pipeline.

        If an edge between the same source and target already exists, the
        port mappings are merged (duplicates removed, order kept).

        Args:
            edge: The edge to add.

        Raises:
            PipelineValidationError: If the edge originates from the Sink
                node or fails validation.

        """
        # Validate early; compile() validates again.
        self._validate_edge(edge)
        key = (edge.source, edge.target)
        if key in self._config.edges:
            existing = self._config.edges[key]
            merged = existing.model_copy(
                update={
                    "ports_map": _merge_ports_maps(existing.ports_map, edge.ports_map)
                }
            )
        else:
            merged = edge.model_copy(deep=True)
        previous = self._config.edges.get(key)
        self._config.edges[key] = merged
        try:
            self._check_sink_port_names()
        except PipelineValidationError:
            if previous is None:
                del self._config.edges[key]
            else:
                self._config.edges[key] = previous
            raise
        if previous is None:
            self._add_edge_to_nx(merged)
        else:
            self._update_edge_in_nx(merged)

    @decompile
    def remove_edge(self, source: str, target: str) -> None:
        """Remove an edge from the pipeline.

        Args:
            source: Name of the source node.
            target: Name of the target node.

        Raises:
            PipelineValidationError: If the edge does not exist.

        """
        if (source, target) not in self._config.edges:
            message = (
                f"Edge from '{source}' to '{target}' does not exist in the pipeline."
            )
            raise PipelineValidationError(message)
        # Remove the edge from the pipeline configuration
        del self._config.edges[(source, target)]
        # Remove the edge from the graph structure
        self._remove_edge_from_nx(Edge(source=source, target=target, ports_map=[]))

    @decompile
    def update_edge(self, edge: Edge) -> None:
        """Replace the ports_map of an existing edge.

        Args:
            edge: Edge with the updated ``ports_map``.  The ``source`` and
                ``target`` fields identify the existing edge to update.

        Raises:
            PipelineValidationError: If the edge does not exist or fails
                validation.

        """
        key = (edge.source, edge.target)
        if key not in self._config.edges:
            message = f"Edge from '{edge.source}' to '{edge.target}' does not exist in the pipeline."
            raise PipelineValidationError(message)
        self._validate_edge(edge)
        previous = self._config.edges[key]
        self._config.edges[key] = edge.model_copy(deep=True)
        try:
            self._check_sink_port_names()
        except PipelineValidationError:
            self._config.edges[key] = previous
            raise
        self._update_edge_in_nx(self._config.edges[key])

    def compile(self) -> None:
        """Compile the pipeline: validate the graph and instantiate the nodes.

        When the pipeline was compiled before and then edited, a node whose
        class and config object did not change keeps its node object, so it
        also keeps its fitted state.  The Sink ports are created from the
        edges that end at the Sink, in the order in which the edges were
        added.

        Raises:
            PipelineCompilationError: If the pipeline is already compiled.
            PipelineValidationError: If validation fails (e.g. missing Sink
                node, cycles, incompatible ports).

        """
        if self.compiled:
            message = "Pipeline is already compiled. Change the pipeline before you compile it again."
            raise PipelineCompilationError(message)
        self._compile()

    ## --- Public API for getters ---
    def get_node_config(self, node_name: str) -> NodeConfig:
        """Get a node's configuration from the pipeline.

        Args:
            node_name: Name of the node.

        Returns:
            The :class:`NodeConfig` associated with *node_name*.

        Raises:
            PipelineValidationError: If *node_name* does not exist.

        """
        if node_name not in self._config.nodes:
            message = f"Node '{node_name}' does not exist in the pipeline."
            raise PipelineValidationError(message)
        return self._config.nodes[node_name][1]

    def get_validated_sink_node_name(self) -> str:
        """Get the name of the sink node, raising if none is configured.

        Returns:
            The sink node name.

        Raises:
            PipelineValidationError: If no Sink node exists in the pipeline.

        """
        if self.sink_node_name is None:
            message = "No Sink node found in the pipeline. A pipeline must have one Sink node to be valid."
            raise PipelineValidationError(message)
        return self.sink_node_name

    def get_edge(self, source: str, target: str) -> Edge:
        """Get an edge's configuration from the pipeline.

        Args:
            source: Name of the source node.
            target: Name of the target node.

        Returns:
            The :class:`Edge` between *source* and *target*.

        Raises:
            PipelineValidationError: If the edge does not exist.

        """
        if (source, target) not in self._config.edges:
            message = (
                f"Edge from '{source}' to '{target}' does not exist in the pipeline."
            )
            raise PipelineValidationError(message)
        return self._config.edges[(source, target)]

    def get_source_node_names(self) -> list[str]:
        """Get the names of all source nodes in the pipeline.

        Returns:
            List of node names whose type is :attr:`NodeType.SOURCE`.

        """
        return [
            node_name
            for node_name, (_, node_config) in self._config.nodes.items()
            if node_config.node_type == NodeType.SOURCE
        ]

    def get_metric_node_names(self) -> list[str]:
        """Get the names of all metric nodes in the pipeline.

        Returns:
            List of node names whose type is :attr:`NodeType.METRIC`.

        """
        return [
            node_name
            for node_name, (_, node_config) in self._config.nodes.items()
            if node_config.node_type == NodeType.METRIC
        ]

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
            PipelineValidationError: If a node does not exist or does not
                support parameters.

        """
        for node_name, node_params in params.items():
            if node_name not in self.node_objects:
                message = f"Node '{node_name}' does not exist in the compiled pipeline. Cannot set parameters for a missing node."
                raise PipelineValidationError(message)
            node_object = self.node_objects[node_name]
            if not has_params(node_object):
                message = f"Node '{node_name}' does not have parameters. Cannot set parameters for node that does not have parameters."
                raise PipelineValidationError(message)
            node_object.set_params(node_params)

    def save_params_to_dir(self, dir_path: str | Path) -> None:
        """Save all node parameters to a pickle file in the given directory.

        Pickle is used so that arbitrary Python objects inside the aggregated
        params dict (numpy arrays, sklearn internals, torch tensors, etc.) are
        handled transparently.  The directory is created if needed.

        Args:
            dir_path: Directory in which the pickle file will be written.

        """
        params = self.get_params()
        path = self._params_path(dir_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(params, f)

    def load_params_from_dir(self, dir_path: str | Path) -> None:
        """Load node parameters from a previously saved pickle file.

        Only load files that you trust: unpickling can run arbitrary code.

        Args:
            dir_path: Directory containing the parameter pickle file.

        """
        with self._params_path(dir_path).open("rb") as f:
            params = pickle.load(f)  # noqa: S301 - only load files you trust
        self.set_params(params)

    def save_config_to_dir(self, dir_path: str | Path) -> None:
        """Save the pipeline configuration as a JSON file.

        The file holds every field of every node config.  Load it with
        :meth:`from_config_file` or :meth:`load_from_dir`.  The directory is
        created if needed.

        Args:
            dir_path: Directory in which the JSON file will be written.

        """
        path = Path(dir_path) / f"{self.file_basename}_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.config.model_dump_json(indent=2))

    def save_html_to_dir(self, dir_path: str | Path, **render_kwargs: Any) -> None:
        """Render the pipeline graph and save it as an HTML file.

        Args:
            dir_path: Directory in which the HTML file will be written.
            **render_kwargs: Forwarded to :meth:`render_to_html`
                (``title``, ``backend``, ``figsize``, ``full_html``,
                ``include_plotlyjs``).

        """
        html = self.render_to_html(**render_kwargs)
        path = Path(dir_path) / f"{self.file_basename}_graph.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html)

    ## --- Public API for identity ---
    def hash(self) -> str:
        """Return a stable hex digest identifying this pipeline's layout and config.

        The digest is a SHA-256 hash derived from the user-facing
        :class:`PipelineConfig`: node names, node types, every field of the
        node configs (hyperparameters and running configs included), edges
        (with port mappings), plus pipeline name, version, and settings.
        Nodes, edges, and port mappings are sorted canonically so
        construction order does not perturb the result.

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
    def _params_path(self, dir_path: str | Path) -> Path:
        """Return the path of the params pickle file in *dir_path*."""
        return Path(dir_path) / f"{self.file_basename}_params.pkl"

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
        edges_dict: dict[tuple[str, str], Edge] = {}
        for edge in config.edges:
            key = (edge.source, edge.target)
            if key in edges_dict:
                # Merge the ports_map of two edges between the same nodes.
                existing = edges_dict[key]
                edges_dict[key] = existing.model_copy(
                    update={
                        "ports_map": _merge_ports_maps(
                            existing.ports_map, edge.ports_map
                        )
                    }
                )
            else:
                edges_dict[key] = edge.model_copy(deep=True)
        return _InternalPipelineConfig(
            nodes=fully_instantiated_nodes,
            edges=edges_dict,
            name=config.name,
            version=config.version,
            settings=config.settings,
        )

    def _instantiate_node_objects(self, previous: dict[str, Node]) -> None:
        """Instantiate the node objects based on the current pipeline configuration.

        Args:
            previous: Node objects of the last compilation.  A node whose
                class and config object did not change is reused.

        """
        for node_name, (node_type, config) in self.resolved_nodes.items():
            node_class = NODE_REGISTRY.get_node_class(node_type)
            old = previous.get(node_name)
            if old is not None and type(old) is node_class and old.config is config:
                self._node_objects[node_name] = old
                _log.debug("Node reused", node_name=node_name)
                continue
            self._node_objects[node_name] = node_class(config=config)
            _log.debug(
                "Node instantiated",
                node_name=node_name,
                node_type=config.node_type,
            )

    def _build_sink_config(self, sink_config: NodeConfig) -> NodeConfig:
        """Return a copy of *sink_config* with ports created from the Sink edges.

        Each ``(source_port, sink_port)`` pair of an edge to the Sink creates
        a Sink input port and a Sink output port named ``sink_port``, with
        the definition (array type, category, shape, modes) of the source
        output port.  Ports follow the order of the edges, then the order of
        each ``ports_map``.

        Args:
            sink_config: The Sink config stored in the pipeline.

        Returns:
            A new config; the stored config does not change.

        """
        in_ports: dict[str, Port] = {}
        for edge in self.sink_edges_dict.values():
            source_ports = self.get_node_config(edge.source).out_ports
            for source_port, sink_port in edge.ports_map:
                port = source_ports[source_port].model_copy(deep=True)
                port.desc = f"Output '{source_port}' of node '{edge.source}'."
                in_ports[sink_port] = port
        out_ports = {
            name: port.model_copy(deep=True) for name, port in in_ports.items()
        }
        return sink_config.model_copy(
            update={"in_ports": in_ports, "out_ports": out_ports}
        )

    def _prune(self) -> None:
        """Prune the pipeline by removing any nodes that are not connected to any Node."""
        connected_nodes = set()
        for edge in self.edges:
            connected_nodes.add(edge.source)
            connected_nodes.add(edge.target)

        for node_name in set(self.nodes.keys()) - connected_nodes:
            _log.debug("Unconnected node pruned", node_name=node_name)
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
                    raise PipelineValidationError(message)
                sink_node_name = node_name
        return sink_node_name

    # --- Internal methods for NX graph management ---
    def _init_graph_repr(self) -> None:
        """Initialize the graph structure of the pipeline based on the current configuration."""
        for node_name in self._config.nodes.keys():
            self._add_node_to_nx(node_name)
        for edge in self._config.edges.values():
            self._add_edge_to_nx(edge)

    def _add_edge_to_nx(self, edge: Edge) -> None:
        """Add an edge to the graph structure of the pipeline."""
        self._graph.add_edge(edge.source, edge.target)
        self._graph.edges[edge.source, edge.target]["ports_map"] = edge.ports_map

    def _add_node_to_nx(self, node_name: str) -> None:
        """Add a node to the graph structure of the pipeline."""
        self._graph.add_node(node_name)

    def _update_edge_in_nx(self, edge: Edge) -> None:
        """Update an edge in the graph structure of the pipeline."""
        if not self._graph.has_edge(edge.source, edge.target):
            message = f"Edge from '{edge.source}' to '{edge.target}' does not exist in the graph. Cannot update non-existing edge."
            raise PipelineGraphError(message)
        self._graph.edges[edge.source, edge.target]["ports_map"] = edge.ports_map

    def _remove_edge_from_nx(self, edge: Edge) -> None:
        """Remove an edge from the graph structure of the pipeline."""
        if not self._graph.has_edge(edge.source, edge.target):
            message = f"Edge from '{edge.source}' to '{edge.target}' does not exist in the graph. Cannot remove non-existing edge."
            raise PipelineGraphError(message)
        self._graph.remove_edge(edge.source, edge.target)

    def _remove_node_from_nx(self, node_name: str) -> None:
        """Remove a node from the graph structure of the pipeline."""
        if not self._graph.has_node(node_name):
            message = f"Node '{node_name}' does not exist in the graph. Cannot remove non-existing node."
            raise PipelineGraphError(message)
        self._graph.remove_node(node_name)

    # --- Internal methods for validation ---
    def _validate(self) -> None:
        """Validate the pipeline layout and configuration.

        This method checks for issues such as:
        - Presence of a Sink node
        - Validity of edges (source and target nodes should exist, ports_map should be valid, etc.)
        - Unique Sink port names
        - Absence of cycles in the graph

        Raises:
            PipelineValidationError: If any validation check fails.
            PipelineGraphError: If the graph contains a cycle.

        """
        if self.sink_node_name is None:
            message = "Pipeline must contain a Sink node, but no Sink node was found."
            raise PipelineValidationError(message)

        for edge in self.edges:
            self._validate_edge(edge)
        self._check_sink_port_names()

        if not nx.is_directed_acyclic_graph(self._graph):
            message = "Pipeline graph contains cycles, but cycles are not allowed."
            raise PipelineGraphError(message)

    def _check_sink_port_names(self) -> None:
        """Check that no two Sink edges write to the same Sink port.

        Raises:
            PipelineValidationError: If a Sink port name is used twice.

        """
        seen: dict[str, str] = {}
        for edge in self.sink_edges_dict.values():
            for source_port, sink_port in edge.ports_map:
                origin = f"{edge.source}.{source_port}"
                if sink_port in seen:
                    message = (
                        f"Sink port '{sink_port}' receives both '{seen[sink_port]}' "
                        f"and '{origin}'. Give each output its own Sink port name."
                    )
                    raise PipelineValidationError(message)
                seen[sink_port] = origin

    def _validate_edge(self, edge: Edge) -> None:
        """Validate an edge's configuration and port compatibility.

        Checks performed:
        - Source and target nodes exist in the pipeline.
        - No edge originates from the Sink node.
        - All ports in ``ports_map`` exist on their respective nodes.  A
          Sink port does not need to exist: compilation creates it.
        - Data structures, categories and execution modes of connected
          ports are compatible (not checked for Sink ports, which copy the
          source port).

        Args:
            edge: The edge to validate.

        Raises:
            PipelineValidationError: If any validation check fails.

        """
        source_node_conf = self.get_node_config(edge.source)
        target_node_conf = self.get_node_config(edge.target)

        if source_node_conf.node_type == NodeType.SINK:
            message = f"Edge from '{edge.source}' to '{edge.target}' is not valid because it originates from a Sink node, which cannot have outgoing edges."
            raise PipelineValidationError(message)

        source_out_ports = source_node_conf.out_ports
        target_in_ports = target_node_conf.in_ports
        to_sink = target_node_conf.node_type == NodeType.SINK

        for source_port, target_port in edge.ports_map:
            if source_port not in source_out_ports:
                message = f"Source port '{source_port}' in edge from '{edge.source}' to '{edge.target}' does not exist in the source node's out_ports."
                message += f"\nAvailable source ports: {list(source_out_ports.keys())}"
                raise PipelineValidationError(message)
            if not to_sink and target_port not in target_in_ports:
                message = f"Target port '{target_port}' in edge from '{edge.source}' to '{edge.target}' does not exist in the target node's in_ports."
                message += f"\nAvailable target ports: {list(target_in_ports.keys())}"
                raise PipelineValidationError(message)

        if to_sink:
            # Sink ports are copies of the source ports, so they always match.
            return

        self._is_compatible_structures(
            source_out_ports, target_in_ports, edge.ports_map, edge.source, edge.target
        )
        self._is_compatible_categories(
            source_out_ports, target_in_ports, edge.ports_map, edge.source, edge.target
        )
        # For example, a port that runs only in "training" cannot feed a port
        # that runs only in "inference".
        self._are_execution_modes_compatible(
            source_out_ports, target_in_ports, edge.ports_map, edge.source, edge.target
        )

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
                raise PipelineValidationError(message)

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
                raise PipelineValidationError(message)
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
                raise PipelineValidationError(message)
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

        1. Keep the previous node instances for reuse.
        2. Prune unconnected nodes (if ``autoprune`` is enabled).
        3. Validate the graph layout.
        4. Instantiate the node objects (the Sink gets its generated ports).

        """
        log = _log.bind(pipeline_name=self.name)
        log.log_phase("compilation", "start")

        previous = self._node_objects
        try:
            self._node_objects = {}
            if self.settings.autoprune:
                self._prune()
            self._validate()
            self._instantiate_node_objects(previous)
        except Exception as exc:
            log.exception("Compilation failed", exc)  # noqa: PLE1205, TRY401 - nodeml Logger API
            raise

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
        include_plotlyjs: bool | Literal["cdn"] = "cdn",
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
            include_plotlyjs: ``"cdn"`` (the default) loads plotly.js from
                a CDN, which keeps the file small.  ``True`` embeds
                plotly.js (about 5 MB) so the file also works offline.

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
            return fig.to_html(full_html=full_html, include_plotlyjs=include_plotlyjs)
        message = f"Backend '{backend}' is not supported for rendering. Supported backends are: ['plotly']."
        raise ValueError(message)
