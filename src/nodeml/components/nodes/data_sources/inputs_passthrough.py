"""InputsPassthrough data-source node for the NodeML Framework.

A source node that turns externally supplied inputs into pipeline outputs
without any transformation. Declare one ``out_port`` per stream you want
to expose downstream, then feed the pipeline at run time with
``input_data={node_name: {port_name: (array, context), ...}}`` — the
passthrough forwards that payload unchanged on the matching output ports.

Typical uses
------------
* Wiring a pipeline to an external input source (tests, notebooks,
  serving endpoints) without writing a dedicated loader.
* Splitting a single external payload into several typed streams (e.g.
  ``features`` / ``targets``) that downstream nodes consume separately.

Operating contract
------------------
* ``fit`` / ``setup_source`` do nothing — there is no state to learn or
  resource to open.
* ``transform`` receives ``data: dict[str, tuple[ArrayLike, DataContext]]``
  keyed by port name and returns it as-is. Before returning, it validates:

  - every key in *data* corresponds to a declared ``out_port``;
  - every ``out_port`` that is active in the current ``execution_mode``
    and not marked ``optional`` has a matching entry in *data*.

A port is considered active in the current mode when its ``mode`` list
contains the mode or ``"all"``.
"""

from typing import Any

from pydantic import Field

from nodeml.core.common.data.data import ArrayLike, DataContext
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.nodes.data_source.data_source import (
    DataSourceConfig,
    DataSourceMetadata,
    DataSourceNode,
    DataSourceRunningConfig,
)
from nodeml.core.nodes.node import Port


class InputsPassthroughMetadata(DataSourceMetadata):
    """Metadata for the InputsPassthrough node."""

    node_name: str = "InputsPassthrough"
    description: str = (
        "Forward externally supplied inputs to the pipeline unchanged. "
        "One entry is produced per declared out_port."
    )


class InputsPassthroughRunningConfig(DataSourceRunningConfig):
    """No run-time knobs; output ports are declared on the config directly."""


class InputsPassthroughConfig(
    DataSourceConfig[InputsPassthroughRunningConfig],
):
    """Full configuration for the InputsPassthrough node.

    The set of output streams is entirely defined by ``out_ports``: every
    port declared here becomes a key the caller must supply in the
    external ``input_data`` payload (subject to the port's ``mode`` list
    and ``optional`` flag).
    """

    running_config: InputsPassthroughRunningConfig = Field(
        default_factory=InputsPassthroughRunningConfig,
        description="No run-time knobs.",
    )
    in_ports: dict[str, Port] = Field(
        default={},
        description="Pure source node — no input ports.",
    )
    out_ports: dict[str, Port] = Field(
        default={},
        description=(
            "User-defined output ports. Each key becomes an expected entry "
            "in the external input payload at transform time."
        ),
    )


class InputsPassthrough(
    DataSourceNode[ArrayLike, DataContext, ArrayLike, DataContext],
):
    """Forward externally supplied ``(array, context)`` tuples unchanged.

    The runner routes the external ``input_data[node_name]`` payload into
    :meth:`fetch_data`; this node validates the payload against its
    declared ``out_ports`` and returns it as-is.

    Setting :attr:`accepts_inputs` to ``True`` signals to the
    :class:`~nodeml.core.pipeline.runners.smart_runner.SmartRunner` that
    this source wants the external inputs passed through rather than
    producing data on its own.

    Example:
    -------
    ::

        cfg = InputsPassthroughConfig(
            out_ports={
                "features": Port(
                    arr_type=ArrayLikeEnum.PANDAS,
                    data_structure=DataStructureEnum.TABULAR,
                    data_category=DataCategoryEnum.MIXED,
                    data_shape="batch feature",
                    desc="Feature matrix forwarded from the caller.",
                ),
                "targets": Port(
                    arr_type=ArrayLikeEnum.PANDAS,
                    data_structure=DataStructureEnum.TABULAR,
                    data_category=DataCategoryEnum.NUMERICAL,
                    data_shape="batch 1",
                    desc="Regression targets, only required in training.",
                    mode=[str(NodeExecutionMode.TRAINING)],
                ),
            },
        )
        node = InputsPassthrough(config=cfg)
        # Then at pipeline run time:
        runner.train(input_data={"source": {
            "features": (X_train, X_ctx),
            "targets":  (y_train, y_ctx),
        }})

    """

    metadata = InputsPassthroughMetadata()
    accepts_inputs: bool = True

    def __init__(self, *, config: InputsPassthroughConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config

    # --- DataSourceNode interface ----------------------------------------

    def setup_source(self) -> None:
        """No-op: nothing to set up for a passthrough node."""
        return

    def fetch_data(
        self,
        data: dict[str, tuple[ArrayLike, DataContext]] | None = None,
    ) -> dict[str, tuple[ArrayLike, DataContext]]:
        """Validate and return the externally supplied inputs unchanged.

        Parameters
        ----------
        data:
            Mapping of port name to ``(array, context)`` tuples provided
            by the runner from its external ``input_data`` payload.

        Returns
        -------
        dict
            The same mapping, unchanged. A shallow copy is made so the
            runner's internal structures are not aliased downstream.

        Raises
        ------
        ValueError
            If *data* contains keys that do not match any declared
            ``out_port``, or if a required port (active in the current
            ``execution_mode`` and not ``optional``) has no entry.

        """
        payload: dict[str, tuple[ArrayLike, DataContext]] = dict(data or {})
        self._validate_inputs(payload)
        return payload

    # --- Private helpers --------------------------------------------------

    def _validate_inputs(
        self,
        data: dict[str, tuple[ArrayLike, DataContext]],
    ) -> None:
        """Check that *data* matches the declared out_ports for the current mode."""
        declared = self._config.out_ports
        unknown = [key for key in data if key not in declared]
        if unknown:
            msg = (
                f"InputsPassthrough: received inputs for unknown ports "
                f"{unknown}. Declared out_ports: {sorted(declared)}."
            )
            raise ValueError(msg)

        mode = self.execution_mode  # type: ignore[attr-defined]
        missing = [
            name
            for name, port in declared.items()
            if self._port_is_required(port, mode) and name not in data
        ]
        if missing:
            msg = (
                f"InputsPassthrough: missing required inputs for ports "
                f"{missing} in execution mode '{mode}'."
            )
            raise ValueError(msg)

    @staticmethod
    def _port_is_required(port: Port, mode: Any) -> bool:
        """Return whether *port* must receive an input in *mode*."""
        if port.optional:
            return False
        return mode in port.mode or NodeExecutionMode.ALL in port.mode

    # --- Convenience ------------------------------------------------------

    @property
    def running_config(self) -> InputsPassthroughRunningConfig:
        """Access the running configuration."""
        return self._config.running_config
