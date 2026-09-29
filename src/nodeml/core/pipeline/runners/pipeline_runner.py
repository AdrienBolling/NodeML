"""Define the base class for a NodeML PipelineRunner.

This class serves as a blueprint for implementing various pipeline execution strategies.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from nodeml.core.common.data.data import ArrayLike, DataContext
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.nodes.node import Node
from nodeml.core.pipeline.pipeline import Pipeline


class RunnerConfig(BaseModel):
    """Define the configuration schema for a PipelineRunner.

    Attributes:
        verbose: If ``True``, the runner shows a per-phase :mod:`tqdm`
            progress bar tracking node execution.  Disable when running
            many short pipelines (e.g. hyperparameter tuning trials) or
            when stdout is captured.

    """

    verbose: bool = True


class PipelineRunner(ABC):
    """Abstract base class for executing a compiled NodeML Pipeline.

    A ``PipelineRunner`` encapsulates the logic for training, evaluating, and
    running inference on a pipeline.  Concrete subclasses implement the
    actual execution strategy (e.g. :class:`SmartRunner`).
    """

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        config: RunnerConfig,
    ) -> None:
        """Initialize the PipelineRunner with a pipeline and configuration.

        Args:
            pipeline: A compiled :class:`Pipeline` to execute.
            config: Runner-specific configuration.

        """
        self._pipeline = pipeline
        self._config = config
        self._mode = NodeExecutionMode.DEFAULT

    # --- Convenience API

    @property
    def pipeline(self) -> Pipeline:
        """Get the pipeline associated with this runner."""
        return self._pipeline

    @property
    def config(self) -> RunnerConfig:
        """Get the configuration of this runner."""
        return self._config

    @property
    def mode(self) -> NodeExecutionMode:
        """Get the current execution mode of the runner. The mode is supposed to be set automatically according to the function used."""
        return self._mode

    @property
    def node_objects(self) -> dict[str, Node]:
        """Get the mapping of node names to their instantiated objects in the pipeline."""
        return self._pipeline.node_objects

    def get_params(self) -> dict[str, dict[str, Any]]:
        """Get the parameters of all nodes in the pipeline."""
        return self._pipeline.get_params()

    def set_params(self, params: dict[str, dict[str, Any]]) -> None:
        """Set the parameters of all nodes in the pipeline."""
        self._pipeline.set_params(params=params)

    def save_params_to_dir(self, dir_path: str | Path) -> None:
        """Save the parameters of all nodes in the pipeline to a directory."""
        self._pipeline.save_params_to_dir(dir_path=dir_path)

    def load_params_from_dir(self, dir_path: str | Path) -> None:
        """Load the parameters of all nodes in the pipeline from a directory."""
        self._pipeline.load_params_from_dir(dir_path=dir_path)

    def get_metric_node_names(self) -> list[str]:
        """Get the list of metric node names in the pipeline."""
        return self._pipeline.get_metric_node_names()

    # --- API to implement for any PipelineRunner implementation ---
    # INFO : Note the absence of a 'tune' method, as I think it is best suited to be part of a Wrapper for PipelineRunners, rather than the PipelineRunner itself.
    # We will see where it belongs on the long run.

    @abstractmethod
    def train(
        self,
        input_data: Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]]
        | None = None,
    ) -> None:
        """Train the pipeline end-to-end.

        Args:
            input_data: Optional external data keyed by
                ``{node_name: {port_name: (array, context)}}``.  Source
                nodes with ``accepts_inputs=True`` use this data.

        """
        ...

    @abstractmethod
    def evaluate(
        self,
        input_data: Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]]
        | None = None,
    ) -> Mapping[str, tuple[ArrayLike, DataContext]]:
        """Evaluate the pipeline and return computed metrics.

        Args:
            input_data: Optional external data (same structure as
                :meth:`train`).

        Returns:
            Mapping of metric names to ``(array, context)`` tuples.

        """
        ...

    @abstractmethod
    def infer(
        self,
        input_data: Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]]
        | None = None,
    ) -> Mapping[str, tuple[ArrayLike, DataContext]]:
        """Run inference and return the sink node outputs.

        Args:
            input_data: Optional external data (same structure as
                :meth:`train`).

        Returns:
            Mapping of Sink port names to ``(array, context)`` tuples.

        """
        ...
