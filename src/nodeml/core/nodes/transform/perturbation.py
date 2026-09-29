"""The PerturbationNode: an entry point for perturbed rows in a pipeline.

A :class:`PerturbationNode` sits on an edge of a trained pipeline, for
example right after the data source.  It has two behaviours:

* **Pass-through** (the default): the node returns its input.  The method
  :meth:`PerturbationNode.perturb` is the hook for perturbations that change
  the input (for example noise or shifts); the base class does not change
  the data.
* **Injection**: when ``running_config.inject_in_inference`` is ``True``,
  the input port is not active in inference, so the runner does not run the
  nodes before the PerturbationNode.  In inference, the node returns the
  rows given with :meth:`PerturbationNode.set_rows`.  The
  :class:`~nodeml.core.pipeline.counterfactuals.CounterfactualEvaluator`
  uses this mode: each prediction that a counterfactual method asks for
  runs only the part of the pipeline after the PerturbationNode.

In training and evaluation, the node is always a pass-through.
"""

from typing import Self

import pandas as pd
from pydantic import Field, model_validator

from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    TabularDataContext,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.common.exceptions import NodeInputError
from nodeml.core.nodes.node import Port
from nodeml.core.nodes.transform.transform import (
    TransformConfig,
    TransformHyperParameters,
    TransformMetadata,
    TransformNode,
    TransformRunningConfig,
)

PERTURBATION_INPUT_PORT = "input"
"""Name of the input port of a PerturbationNode."""

PERTURBATION_OUTPUT_PORT = "output"
"""Name of the output port of a PerturbationNode."""

type _Table = tuple[pd.DataFrame, TabularDataContext]


class PerturbationMetadata(TransformMetadata):
    """Metadata for the PerturbationNode."""

    node_name: str = "PerturbationNode"
    description: str = (
        "Entry point for perturbed rows. Passes its input through, or returns "
        "injected rows in inference."
    )
    trainable: bool = False


class PerturbationHyperParameters(TransformHyperParameters):
    """The PerturbationNode has no hyperparameters."""


class PerturbationRunningConfig(TransformRunningConfig):
    """Runtime options of the PerturbationNode."""

    inject_in_inference: bool = Field(
        default=False,
        description=(
            "If True, the input port is not active in inference, and the node "
            "returns the rows given with set_rows(). The nodes before the "
            "PerturbationNode then do not run in inference."
        ),
    )


def _default_port(desc: str) -> Port:
    """Return the default port: mixed tabular data of any width."""
    return Port(
        arr_type=ArrayLikeEnum.PANDAS,
        data_structure=DataStructureEnum.TABULAR,
        data_category=DataCategoryEnum.MIXED,
        data_shape="batch feature",
        desc=desc,
    )


class PerturbationNodeConfig(
    TransformConfig[PerturbationHyperParameters, PerturbationRunningConfig]
):
    """Configuration of the PerturbationNode.

    The ``input`` and ``output`` ports accept mixed tabular data by default.
    To insert the node on an edge, copy the definition of the source port
    into both ports, so that the port checks of the pipeline stay the same.
    """

    hyperparameters: PerturbationHyperParameters = Field(
        default_factory=PerturbationHyperParameters
    )
    running_config: PerturbationRunningConfig = Field(
        default_factory=PerturbationRunningConfig
    )
    in_ports: dict[str, Port] = Field(
        default_factory=lambda: {
            PERTURBATION_INPUT_PORT: _default_port("Rows to pass through.")
        }
    )
    out_ports: dict[str, Port] = Field(
        default_factory=lambda: {
            PERTURBATION_OUTPUT_PORT: _default_port(
                "The input rows, or the injected rows in inference."
            )
        }
    )

    @model_validator(mode="after")
    def _check_ports(self) -> Self:
        """Check the port names and set the input port mode for injection.

        With injection, the input port is optional and active only in
        training and evaluation.  The validator replaces the port with a
        copy, so a Port object shared with another config does not change.
        """
        if set(self.in_ports) != {PERTURBATION_INPUT_PORT} or set(self.out_ports) != {
            PERTURBATION_OUTPUT_PORT
        }:
            msg = (
                "A PerturbationNode has exactly one input port "
                f"'{PERTURBATION_INPUT_PORT}' and one output port "
                f"'{PERTURBATION_OUTPUT_PORT}'. Got in_ports "
                f"{list(self.in_ports)} and out_ports {list(self.out_ports)}."
            )
            raise ValueError(msg)
        if self.running_config.inject_in_inference:
            port = self.in_ports[PERTURBATION_INPUT_PORT]
            self.in_ports[PERTURBATION_INPUT_PORT] = port.model_copy(
                update={
                    "optional": True,
                    "mode": [
                        NodeExecutionMode.TRAINING,
                        NodeExecutionMode.EVALUATION,
                    ],
                }
            )
        return self


class PerturbationNode(
    TransformNode[
        pd.DataFrame, TabularDataContext, pd.DataFrame, TabularDataContext, None
    ]
):
    """Pass rows through, or return injected rows in inference.

    The node has no learned state.  It keeps a copy of the last input that
    it passed through in :attr:`last_input`, so that a caller can read the
    rows at this point of the pipeline.

    Subclasses can override :meth:`perturb` to change the rows that pass
    through, for example to add noise.
    """

    metadata = PerturbationMetadata()

    def __init__(self, *, config: PerturbationNodeConfig) -> None:
        """Initialise the node with its configuration."""
        self._config: PerturbationNodeConfig = config
        # Stateless: the node can transform without a fit, also after a reload.
        self._fitted = True
        self._rows: _Table | None = None
        self._last_input: _Table | None = None

    @property
    def last_input(self) -> _Table | None:
        """The last ``(DataFrame, context)`` input that passed through, or ``None``."""
        return self._last_input

    def set_rows(self, df: pd.DataFrame, context: TabularDataContext) -> None:
        """Set the rows that the node returns in inference, with injection.

        Args:
            df: The rows to inject.
            context: The context of *df*.  It must name the columns of *df*
                in the same order.

        Raises:
            DataContextError: If *context* does not describe *df*.

        """
        context.check_columns(df.columns)
        self._rows = (df, context)

    def clear_rows(self) -> None:
        """Forget the injected rows."""
        self._rows = None

    def perturb(self, df: pd.DataFrame, context: TabularDataContext) -> _Table:
        """Return the rows to pass on.  The base class does not change them.

        Args:
            df: The input rows.
            context: The context of *df*.

        Returns:
            The rows to pass on and their context.

        """
        return df, context

    def fit(self, data: dict[str, _Table]) -> None:
        """Do nothing: the node has no learned state."""
        _ = data

    def transform(self, data: dict[str, _Table]) -> dict[str, _Table]:
        """Pass the input through, or return the injected rows in inference.

        Args:
            data: The ``input`` port, when it is active.

        Returns:
            ``{"output": (DataFrame, context)}``.

        Raises:
            NodeInputError: If injection is on in inference and no rows are
                set, or if the input is missing in another mode.

        """
        injecting = self._config.running_config.inject_in_inference
        if injecting and self.execution_mode == NodeExecutionMode.INFERENCE:
            if self._rows is None:
                msg = (
                    "The PerturbationNode injects rows in inference, but no rows "
                    "are set. Call set_rows() before inference."
                )
                raise NodeInputError(msg)
            return {PERTURBATION_OUTPUT_PORT: self._rows}
        if PERTURBATION_INPUT_PORT not in data:
            msg = f"The PerturbationNode needs its '{PERTURBATION_INPUT_PORT}' port in this mode."
            raise NodeInputError(msg)
        df, context = data[PERTURBATION_INPUT_PORT]
        self._last_input = (df, context)
        return {PERTURBATION_OUTPUT_PORT: self.perturb(df, context)}

    def get_params(self) -> None:
        """Return ``None``: the node has no learned parameters."""
        return

    def set_params(self, params: None) -> None:
        """Do nothing: the node has no learned parameters."""
        _ = params
