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
        """Narrow port modes to ``evaluation`` unless the metric opts out.

        Metric nodes exist to score a run after the forward pass, so by
        default their ports should only be walked during evaluation.
        We only rewrite ports that still carry the :class:`Port` default
        (``["all"]``) — explicit overrides by the metric author or user
        are preserved.
        """
        port_default = [NodeExecutionMode.ALL]
        eval_only = [NodeExecutionMode.EVALUATION]
        for ports in (self.in_ports, self.out_ports):
            for port in ports.values():
                if port.mode == port_default:
                    port.mode = list(eval_only)
        return self


class MetricNodeMetadata(NodeMetadata):
    """Metadata for a Metric Node in a NodeML Pipeline."""


class MetricNode[D_I, D_C_I, D_O, D_C_O](Node[D_I, D_C_I, D_O, D_C_O]):
    """Node wrapper for an accumulator-style metric in a NodeML Pipeline.

    Metrics follow an **update/compute** pattern: :meth:`update` accumulates
    state from incoming batches and :meth:`compute` produces the final result.
    :meth:`node_transform` calls both in sequence.
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
        """Transform with the Metric Node. This method will be called during all phase of the pipeline."""
        # Metrics typically do not produce output data in the same way as other nodes, so this can be left empty or return an empty dict.
        self.update(data)
        return self.compute()
