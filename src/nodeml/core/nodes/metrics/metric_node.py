"""Module for the Metric node wrapper."""

from abc import abstractmethod

from pydantic import BaseModel, model_validator

from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.nodes.node import Node, NodeConfig, NodeMetadata, NodeType, Port


class MetricNodeRunningConfig(BaseModel):
    """Running configuration for a Metric Node in a NodeML Pipeline."""


class MetricNodeConfig[R: MetricNodeRunningConfig](NodeConfig):
    """Configuration for a Metric Node in a NodeML Pipeline.

    Generic over ``R``, which must be a :class:`MetricNodeRunningConfig`
    subclass carrying metric-specific runtime parameters.

    Attributes:
        node_type: Always ``NodeType.METRIC``.
        in_ports: Input port definitions for the metric node.
        out_ports: Output port definitions for the metric node.
        running_config: Optional runtime parameters for metric execution.

    """

    node_type: NodeType = NodeType.METRIC
    in_ports: dict[str, Port] = {}
    out_ports: dict[str, Port] = {}
    running_config: R | None = None

    @model_validator(mode="after")
    def _default_ports_to_eval_only(self) -> "MetricNodeConfig[R]":
        """Make the ports without an explicit mode evaluation-only.

        Metric nodes score a run after the forward pass.  Thus, by default,
        the runner walks their ports only during evaluation.  A port that
        sets ``mode`` explicitly (also ``["all"]``) keeps its mode.

        The validator replaces a port with a copy.  It does not change the
        :class:`Port` object of the caller, which can belong to other
        configs too.

        Returns:
            The validated config.

        """
        eval_only = [NodeExecutionMode.EVALUATION]
        for ports in (self.in_ports, self.out_ports):
            for name, port in ports.items():
                if "mode" not in port.model_fields_set:
                    ports[name] = port.model_copy(update={"mode": list(eval_only)})
        return self


class MetricNodeMetadata(NodeMetadata):
    """Metadata for a Metric Node in a NodeML Pipeline."""


class MetricNode[D_I, D_C_I, D_O, D_C_O](Node[D_I, D_C_I, D_O, D_C_O]):
    """Node wrapper for an accumulator-style metric in a NodeML Pipeline.

    Metrics follow a **reset/update/compute** pattern:

    * :meth:`reset` clears the accumulated state.
    * :meth:`update` adds one batch of data to the state.
    * :meth:`compute` returns the result for the state.  It does not
      clear the state.

    :meth:`node_transform` scores one batch: it calls :meth:`reset`,
    :meth:`update` and :meth:`compute`, then :meth:`reset` again, also when
    an error occurs.  Thus each evaluation starts from a clean state.  A
    batched evaluation can call :meth:`reset` once, :meth:`update` for
    each batch, and :meth:`compute` at the end.
    """

    metadata = MetricNodeMetadata()

    def __init__(self, *, config: MetricNodeConfig) -> None:
        """Initialise the Metric Node with the given configuration.

        Args:
            config: Metric configuration with ports and running config.

        """
        self._config = config

    # --- Methods to implement for the metric node ---

    @abstractmethod
    def update(self, data: dict[str, tuple[D_I, D_C_I]]) -> None:
        """Accumulate metric state from the given batch of data.

        Args:
            data: Mapping of input port name to ``(data, context)`` tuples.

        """

    @abstractmethod
    def compute(self) -> dict[str, tuple[D_O, D_C_O]]:
        """Compute and return the final metric value from accumulated state.

        Returns:
            Mapping of output port name to ``(data, context)`` tuples.

        """

    def reset(self) -> None:
        """Clear the state that :meth:`update` accumulates.

        The default implementation does nothing, so a stateless metric does
        not need to implement it.  Override it when :meth:`update` keeps
        state.
        """

    # --- API convenience ---

    @property
    def running_config(self) -> MetricNodeRunningConfig | None:
        """Return the running configuration of the metric."""
        return self._config.running_config

    @property
    def config(self) -> MetricNodeConfig:
        """Return the full configuration of the metric."""
        return self._config

    # --- Node API override don't touch that ---

    def node_fit(self, data: dict[str, tuple[D_I, D_C_I]]) -> None:
        """Fit the Metric Node. This method will be called during the training phase of the pipeline."""

    def node_transform(
        self, data: dict[str, tuple[D_I, D_C_I]]
    ) -> dict[str, tuple[D_O, D_C_O]]:
        """Score one batch of data from a clean state.

        Args:
            data: Mapping of input port name to ``(data, context)`` tuples.

        Returns:
            Mapping of output port name to ``(data, context)`` tuples.

        """
        # Reset before the update: drop the state of earlier calls.
        self.reset()
        try:
            self.update(data)
            return self.compute()
        finally:
            # Reset after the result or an error: no state leaks to the
            # next evaluation.
            self.reset()
