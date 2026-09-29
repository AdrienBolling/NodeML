"""SmartRunner implementation module."""

import time
from collections.abc import Iterable, Mapping
from typing import Any

from jaxtyping import jaxtyped
from tqdm.auto import tqdm

from nodeml.core.common.data.data import (
    DATA_CATEGORY_MAPPING,
    ArrayLike,
    ArrayLikeEnum,
    Data,
    DataContext,
    TabularData,
    TabularDataContext,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.common.exceptions import (
    DataContextError,
    DataTypeError,
    NodeError,
    NodeInputError,
)
from nodeml.core.common.logging import Logger
from nodeml.core.common.typechecking.typeguards import accepts_inputs_source_node
from nodeml.core.nodes.node import Node, NodeType, Port
from nodeml.core.pipeline.pipeline import Edge, Pipeline
from nodeml.core.pipeline.runners.pipeline_runner import PipelineRunner, RunnerConfig

# Node outputs are (array, context) pairs.
_ARRAY_CONTEXT_LEN = 2


class SmartRunnerConfig(RunnerConfig):
    """Configuration for the SmartRunner."""


class SmartRunner(PipelineRunner):
    """SmartRunner implementation for the NodeML Framework.

    The SmartRunner walks the graph backwards from the Sink (``train``,
    ``infer``) or from the metric nodes (``evaluate``).  It runs a
    predecessor only when the edge feeds an input port that is active in
    the current execution mode, and runs each node at most once per call.

    Between nodes, data travels as :class:`TabularData`.  The runner checks:

    * that each output context describes its data (same columns, same
      order), so contexts cannot drift from the data;
    * that each node returns only the ports that it declares;
    * that each input matches the category and shape of its port, with
      dimension names shared across the input ports of a node.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        config: SmartRunnerConfig | None = None,
    ) -> None:
        """Initialize the SmartRunner with a pipeline.

        Args:
            pipeline: The Pipeline object containing the DAG structure and nodes.
            config: Optional configuration for the runner.

        """
        if not pipeline.compiled:
            msg = "Pipeline must be compiled before being passed to the SmartRunner."
            raise ValueError(msg)  # No logger yet — __init__ hasn't finished
        self._pipeline = pipeline
        self._config = config if config is not None else SmartRunnerConfig()
        self._mode = NodeExecutionMode.DEFAULT
        self._node_outputs: dict[
            str, dict[str, Data]
        ] = {}  # For memoization of node outputs during execution
        self._metric_node_outputs: dict[
            str, dict[str, Any]
        ] = {}  # To store the outputs of metric nodes during evaluation
        self._log = Logger(
            "nodeml.runner.smart",
            pipeline_name=pipeline.name,
            pipeline_version=pipeline.version,
        )
        self._input_data: (
            Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]] | None
        ) = None  # To store the input data passed to the runner, which can be used by the nodes during execution if needed.
        self._pbar: tqdm | None = None  # Per-phase progress bar, if verbose.

    # --- PipelineRunner API implementation ---

    def train(
        self,
        input_data: Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]]
        | None = None,
    ) -> None:
        """Train the pipeline by executing the graph up to the sink node.

        Args:
            input_data: Optional external data keyed by
                ``{node_name: {port_name: (array, context)}}``.

        """
        self._reset_caches()
        self._validate_input_data(input_data)
        self._input_data = input_data
        self._mode = NodeExecutionMode.TRAINING
        self._log.log_phase("training", "start")
        t0 = time.perf_counter()
        try:
            last_node = self.pipeline.get_validated_sink_node_name()
            self._start_progress([last_node], desc="Training")
            self._call_node(last_node)
        except Exception as exc:
            self._log.exception("Training failed", exc)
            raise
        finally:
            self._close_progress()
        self._log.log_phase(
            "training", "end", duration_ms=(time.perf_counter() - t0) * 1000
        )

    def evaluate(
        self,
        input_data: Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]]
        | None = None,
    ) -> Mapping[str, tuple[ArrayLike, DataContext]]:
        """Evaluate the pipeline by executing all metric nodes.

        Args:
            input_data: Optional external data keyed by
                ``{node_name: {port_name: (array, context)}}``.

        Returns:
            Flat mapping of metric outputs to ``(array, context)`` tuples.
            Single-port metric nodes are keyed by the node name; multi-port
            metric nodes are keyed by ``f"{node_name}.{port_name}"``.

        """
        self._reset_caches()
        self._validate_input_data(input_data)
        self._input_data = input_data
        self._mode = NodeExecutionMode.EVALUATION
        self._log.log_phase("evaluation", "start")
        t0 = time.perf_counter()
        try:
            metric_node_names = self.get_metric_node_names()
            self._start_progress(metric_node_names, desc="Evaluation")
            for metric_node_name in metric_node_names:
                self._call_node(metric_node_name)
        except Exception as exc:
            self._log.exception("Evaluation failed", exc)
            raise
        finally:
            self._close_progress()
        self._log.log_phase(
            "evaluation", "end", duration_ms=(time.perf_counter() - t0) * 1000
        )
        return self._flatten_metric_outputs()

    def infer(
        self,
        input_data: Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]]
        | None = None,
    ) -> Mapping[str, tuple[ArrayLike, DataContext]]:
        """Run inference and return the sink node outputs.

        Args:
            input_data: Optional external data keyed by
                ``{node_name: {port_name: (array, context)}}``.

        Returns:
            Mapping of sink output port names to ``(array, context)`` tuples.

        """
        self._reset_caches()
        self._validate_input_data(input_data)
        self._input_data = input_data
        self._mode = NodeExecutionMode.INFERENCE
        self._log.log_phase("inference", "start")
        t0 = time.perf_counter()
        try:
            last_node = self.pipeline.get_validated_sink_node_name()
            self._start_progress([last_node], desc="Inference")
            result = self._call_node(last_node)
        except Exception as exc:
            self._log.exception("Inference failed", exc)
            raise
        finally:
            self._close_progress()
        self._log.log_phase(
            "inference", "end", duration_ms=(time.perf_counter() - t0) * 1000
        )
        return {
            port_name: self._convert_data_to_tuple(data)
            for port_name, data in result.items()
        }

    def _reset_caches(self) -> None:
        """Clear memoized outputs so each phase recomputes from scratch."""
        self._node_outputs = {}
        self._metric_node_outputs = {}
        self._input_data = None

    # --- Progress bar helpers ---

    def _start_progress(self, leaves: Iterable[str], *, desc: str) -> None:
        """Open a tqdm progress bar sized to the nodes that will execute.

        Mirrors :meth:`_call_node`'s mode-aware walk: a predecessor is
        only counted when the incoming edge feeds at least one input
        port active in the current execution mode.  This keeps the bar
        total equal to the number of nodes the runner will actually
        invoke (e.g. target-side nodes drop out during inference).

        Args:
            leaves: Entry-point nodes for the current phase (the sink
                for train/infer, every metric node for evaluate).
            desc: Human-readable label shown on the left of the bar.

        """
        if not self._config.verbose:
            return
        covered = self._collect_execution_set(leaves)
        self._pbar = tqdm(
            total=len(covered),
            desc=desc,
            unit="nodes",
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} {unit} "
                "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            ),
        )

    def _collect_execution_set(self, leaves: Iterable[str]) -> set[str]:
        """Walk the graph the same way :meth:`_call_node` will execute it.

        A predecessor is only followed when the connecting edge maps to
        at least one input port active in the current execution mode.
        Nodes whose output only feeds inactive ports are therefore
        pruned from the set.

        Args:
            leaves: Entry-point nodes for the current phase.

        Returns:
            Set of node names that will execute in the current mode.

        """
        visited: set[str] = set()

        def walk(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            for edge in self._active_incoming_edges(name):
                walk(edge.source)

        for leaf in leaves:
            walk(leaf)
        return visited

    def _port_active_in_mode(self, port: Port) -> bool:
        """Return whether ``port`` is active in the current execution mode."""
        return self._mode in port.mode or NodeExecutionMode.ALL in port.mode

    def _active_incoming_edges(self, node_name: str) -> list[Edge]:
        """Edges feeding ``node_name`` that carry at least one port active now.

        Filters out edges whose every ``(source_port, target_port)`` pair
        lands on an input port inactive in the current execution mode —
        those are irrelevant to the work about to happen and should not
        drag their producers into the run.
        """
        node = self.node_objects[node_name]
        active: list[Edge] = []
        for pred in self.pipeline.graph.predecessors(node_name):
            edge = self.pipeline.get_edge(pred, node_name)
            if any(
                self._port_active_in_mode(node.in_ports[tgt])
                for _src, tgt in edge.ports_map
            ):
                active.append(edge)
        return active

    def _advance_progress(self, node_name: str) -> None:
        """Advance the active progress bar by one, postfixed with *node_name*."""
        if self._pbar is None:
            return
        self._pbar.set_postfix_str(node_name, refresh=False)
        self._pbar.update(1)

    def _close_progress(self) -> None:
        """Close and discard the active progress bar, if any."""
        if self._pbar is None:
            return
        self._pbar.close()
        self._pbar = None

    # --- Internal methods for node execution ---

    def _convert_from_node_output(
        self,
        output: tuple[ArrayLike, DataContext],
        *,
        node_name: str,
        port_name: str,
    ) -> Data:
        """Convert one ``(array, context)`` output to the common data type.

        Args:
            output: The ``(array, context)`` tuple produced for one port.
            node_name: Name of the node that produced the output (for errors).
            port_name: Name of the output port (for errors).

        Returns:
            The output as :class:`TabularData`.

        Raises:
            DataContextError: If the context does not describe the data (for
                example, other columns or another column order).
            NodeError: If the output is not an ``(array, context)`` tuple
                with a :class:`TabularDataContext`.

        """
        if not isinstance(output, tuple) or len(output) != _ARRAY_CONTEXT_LEN:
            msg = (
                f"Node '{node_name}' returned {type(output).__name__} on port "
                f"'{port_name}'; expected an (array, context) tuple."
            )
            self._log.error(msg, node_name=node_name, port=port_name)
            raise NodeError(msg)
        data, context = output
        if not isinstance(context, TabularDataContext):
            msg = (
                f"Node '{node_name}' returned a {type(context).__name__} context on "
                f"port '{port_name}'. The SmartRunner supports only TabularDataContext."
            )
            self._log.error(msg, node_name=node_name, port=port_name)
            raise NodeError(msg)
        try:
            return TabularData(
                data=data,
                columns=context.columns,
                dtypes=context.dtypes,
                categories=context.categories,
            )
        except (DataContextError, ValueError) as exc:
            msg = f"Output '{port_name}' of node '{node_name}' is not consistent with its context: {exc}"
            self._log.error(msg, node_name=node_name, port=port_name)
            raise DataContextError(msg) from exc

    def _convert_to_node_input(
        self, data: Data, arr_type: ArrayLikeEnum
    ) -> tuple[ArrayLike, DataContext]:
        """Convert the common data type to the ``arr_type`` that a port expects.

        Args:
            data: The data in the common data type.
            arr_type: The array type of the receiving port.

        Returns:
            An ``(array, context)`` tuple.

        Raises:
            ValueError: If the data type or the array type is not supported.

        """
        if not isinstance(data, TabularData):
            msg = f"Unsupported data type: {type(data)}. Currently only TabularData is supported as input data type for the SmartRunner."
            self._log.error(msg)
            raise ValueError(msg)  # noqa: TRY004 - kept for compatibility
        match arr_type:
            case ArrayLikeEnum.PANDAS:
                return data.to_pandas()
            case ArrayLikeEnum.NUMPY:
                return data.to_numpy()
            case ArrayLikeEnum.TORCH:
                return data.to_tensor()
            case _:
                msg = f"Unsupported array type: {arr_type}. Supported types are: {ArrayLikeEnum.PANDAS}, {ArrayLikeEnum.NUMPY}, {ArrayLikeEnum.TORCH}."
                self._log.error(msg)
                raise ValueError(msg)

    def _convert_data_to_tuple(self, data: Data) -> tuple[ArrayLike, DataContext]:
        """Convert an internal :class:`Data` into the public ``(array, context)`` tuple.

        Used when returning values across the runner's public surface
        (``evaluate`` / ``infer``) to match the abstract
        :class:`PipelineRunner` contract.
        """
        if isinstance(data, TabularData):
            return data.to_pandas()
        msg = f"Unsupported data type: {type(data)}. Currently only TabularData is supported as output data type for the SmartRunner."
        self._log.error(msg)
        raise ValueError(msg)

    def _flatten_metric_outputs(
        self,
    ) -> dict[str, tuple[ArrayLike, DataContext]]:
        """Flatten the nested ``_metric_node_outputs`` into the public return shape.

        Single-port metric nodes are keyed by the node name; multi-port
        metric nodes are keyed by ``f"{node_name}.{port_name}"`` to avoid
        collisions.
        """
        flat: dict[str, tuple[ArrayLike, DataContext]] = {}
        for node_name, ports in self._metric_node_outputs.items():
            if len(ports) == 1:
                data = next(iter(ports.values()))
                flat[node_name] = self._convert_data_to_tuple(data)
            else:
                for port_name, data in ports.items():
                    flat[f"{node_name}.{port_name}"] = self._convert_data_to_tuple(data)
        return flat

    def _validate_input_data(
        self,
        input_data: Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]] | None,
    ) -> None:
        """Check that every key of *input_data* names a node that accepts inputs.

        Raises:
            NodeInputError: If a key names an unknown node, or a node that
                does not accept external inputs.

        """
        if input_data is None:
            return
        unknown = [name for name in input_data if name not in self.node_objects]
        if unknown:
            msg = (
                f"input_data has keys {unknown} that are not nodes of the pipeline. "
                f"Nodes: {sorted(self.node_objects)}."
            )
            raise NodeInputError(msg)
        refused = [
            name
            for name in input_data
            if not accepts_inputs_source_node(self.node_objects[name])
        ]
        if refused:
            msg = (
                f"Nodes {refused} do not accept external inputs. Only source "
                "nodes with accepts_inputs=True (for example InputsPassthrough) do."
            )
            raise NodeInputError(msg)

    def _call_node(self, node_name: str) -> dict[str, Data]:
        """Recursively execute a node and all its predecessors.

        Results are memoized in ``_node_outputs`` so each node is executed
        at most once per pipeline run.

        Args:
            node_name: Name of the node to execute.

        Returns:
            Dict mapping output port names to :class:`Data` objects in the
            common data type.

        """
        self._log.log_node_call(node_name, self._mode)
        # Only recurse through predecessors whose edge feeds a port that
        # is active in the current mode — skipping lets modes prune whole
        # subgraphs (e.g. target loaders during inference).
        edges = self._active_incoming_edges(node_name)
        for edge in edges:
            if edge.source not in self._node_outputs:
                self._node_outputs[edge.source] = self._call_node(edge.source)

        t0 = time.perf_counter()
        value = self._execute_node(node_name, edges)
        duration_ms = (time.perf_counter() - t0) * 1000

        node_type = self.node_objects[node_name].config.node_type
        self._log.log_node_execution(
            node_name, self._mode, duration_ms=duration_ms, node_type=node_type
        )
        self._advance_progress(node_name)
        # Metric outputs are collected for evaluate(); other outputs are
        # memoized by the caller.
        if node_type == NodeType.METRIC:
            self._metric_node_outputs[node_name] = value
        return value

    def _execute_node(
        self, node_name: str, incoming_edges: list[Edge]
    ) -> dict[str, Data]:
        """Execute a single node after gathering, checking and converting its inputs.

        Args:
            node_name: Name of the node to execute.
            incoming_edges: Edges feeding into this node.

        Returns:
            Dict mapping output port names to :class:`Data` objects.

        """
        node = self.node_objects[node_name]
        # Nodes can adjust their behaviour to the mode (for example, a
        # source node picks the file of the current phase).
        node.set_execution_mode(self._mode)

        # Gather the inputs of the active ports, and remember where each
        # input comes from for the error messages.
        inputs: dict[str, Data] = {}
        origins: dict[str, tuple[str, str]] = {}
        for edge in incoming_edges:
            pred_output = self._node_outputs[edge.source]
            for source_key, target_key in edge.ports_map:
                # Skip the pairs whose target port is inactive in this mode.
                if not self._port_active_in_mode(node.in_ports[target_key]):
                    continue
                inputs[target_key] = pred_output[source_key]
                origins[target_key] = (edge.source, source_key)
                self._log.log_data_flow(
                    edge.source,
                    node_name,
                    data_shape=str(inputs[target_key].shape),
                )
        self._typecheck_inputs(node_name, node, inputs, origins)

        # Convert each input to the array type of its port, then run the node.
        node_inputs = {
            key: self._convert_to_node_input(value, node.in_ports[key].arr_type)
            for key, value in inputs.items()
        }
        return self._get_node_outputs(node_name, node_inputs)

    def _typecheck_inputs(
        self,
        node_name: str,
        node: Node,
        inputs: dict[str, Data],
        origins: dict[str, tuple[str, str]],
    ) -> None:
        """Check every input against the category and shape of its port.

        The checks of one node share one jaxtyping memo, so a dimension name
        (for example ``batch``) must have the same size on every input port.
        Thus X and y of a model must have the same number of rows.

        Args:
            node_name: Name of the receiving node.
            node: The receiving node.
            inputs: Inputs keyed by input port name.
            origins: ``(source node, source port)`` of each input.

        Raises:
            DataTypeError: If an input does not match its port.

        """
        ordered = [key for key in node.in_ports if key in inputs]
        expected = {
            key: self._get_jaxtyping_type_from_port(node.in_ports[key])
            for key in ordered
        }

        @jaxtyped(typechecker=None)
        def _first_mismatch() -> str | None:
            for key in ordered:
                if not isinstance(inputs[key], expected[key]):  # pyright: ignore[reportArgumentType]
                    return key
            return None

        failed = _first_mismatch()
        if failed is None:
            return
        data = inputs[failed]
        pred_node_name, source_key = origins[failed]
        msg = f"Data type mismatch between source node '{pred_node_name}' and target node '{node_name}' on edge with source port '{source_key}' and target port '{failed}'."
        msg += f"\nExpected type: {expected[failed]}. Data: shape - {data.shape}, dtype - {data.dtype}."
        if len(ordered) > 1:
            shapes = {key: inputs[key].shape for key in ordered}
            msg += (
                f"\nA dimension name in data_shape must have the same size on every "
                f"input port of the node. Input shapes: {shapes}."
            )
        self._log.error(
            msg,
            node_name=node_name,
            source_node=pred_node_name,
            source_port=source_key,
            target_port=failed,
        )
        raise DataTypeError(msg)

    def _get_jaxtyping_type_from_port(self, port: Port) -> type:
        """Build the jaxtyping type that checks the category and shape of a port."""
        # Only tabular data exists today; other data structures (for example
        # time series) would select another class here.
        data_category = DATA_CATEGORY_MAPPING[port.data_category]
        return data_category[TabularData, port.data_shape]  # pyright: ignore[reportReturnType]

    def _external_inputs(
        self, node_name: str
    ) -> dict[str, tuple[ArrayLike, DataContext]] | None:
        """Return the external inputs of *node_name*, or ``None`` if it has none.

        External inputs come from the ``input_data`` argument of the public
        methods.  They go through the common data type, so that a bad
        context fails here with a clear message.
        """
        node = self.node_objects[node_name]
        if self._input_data is None or not accepts_inputs_source_node(node):
            return None
        node_input_data = self._input_data.get(node_name)
        if node_input_data is None:
            return None
        return {
            key: self._convert_to_node_input(
                self._convert_from_node_output(
                    value, node_name=node_name, port_name=key
                ),
                ArrayLikeEnum.PANDAS,
            )
            for key, value in node_input_data.items()
        }

    def _get_node_outputs(
        self, node_name: str, inputs: dict[str, tuple[ArrayLike, DataContext]]
    ) -> dict[str, Data]:
        """Execute a node and convert its raw outputs to the common data type.

        Args:
            node_name: Name of the node to execute.
            inputs: Pre-converted inputs keyed by port name.

        Returns:
            Dict mapping output port names to :class:`Data` objects.

        Raises:
            ValueError: If the current execution mode is unsupported.
            NodeError: If the node returns a port that it does not declare.

        """
        self._check_inputs_completeness(node_name, inputs)
        node = self.node_objects[node_name]
        external = self._external_inputs(node_name)
        if external is not None:
            inputs = external
        match self.mode:
            case NodeExecutionMode.TRAINING:
                output = node.node_fit_transform(inputs)
            case NodeExecutionMode.INFERENCE | NodeExecutionMode.EVALUATION:
                output = node.node_transform(inputs)
            case _:
                message = f"Unsupported execution mode: {self.mode}"
                self._log.error(message, node_name=node_name)
                raise ValueError(message)
        undeclared = [key for key in output if key not in node.out_ports]
        if undeclared:
            message = (
                f"Node '{node_name}' returned ports {undeclared} that it does not "
                f"declare. Declared output ports: {list(node.out_ports)}."
            )
            self._log.error(message, node_name=node_name)
            raise NodeError(message)
        return {
            key: self._convert_from_node_output(
                value, node_name=node_name, port_name=key
            )
            for key, value in output.items()
        }

    def _check_inputs_completeness(
        self, node_name: str, inputs: dict[str, tuple[ArrayLike, DataContext]]
    ) -> None:
        """Check that all required inputs for a node are present.

        A missing input is tolerated when the port is optional, or when its
        ``mode`` list does not include the current execution mode (and does
        not include ``"all"``).

        Args:
            node_name: Name of the node being checked.
            inputs: Currently available inputs keyed by port name.

        Raises:
            NodeInputError: If a required input is missing for the current mode.

        """
        node = self.node_objects[node_name]
        for missing_input in set(node.in_ports) - set(inputs):
            port = node.in_ports[missing_input]
            if port.optional or not self._port_active_in_mode(port):
                continue
            message = f"Node '{node_name}' ({type(node).__name__}) is missing required input: '{missing_input}'"
            self._log.error(message, node_name=node_name, missing_input=missing_input)
            raise NodeInputError(message)
