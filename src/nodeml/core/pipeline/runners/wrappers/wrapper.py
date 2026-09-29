"""Base wrapper class for PipelineRunners."""

from abc import ABC
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nodeml.core.common.data.data import ArrayLike, DataContext
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.nodes.node import Node
from nodeml.core.pipeline.pipeline import Pipeline
from nodeml.core.pipeline.runners.pipeline_runner import PipelineRunner


class PipelineRunnerWrapper[D_O, M, C](PipelineRunner, ABC):
    """Abstract wrapper that adds functionality around a :class:`PipelineRunner`.

    Subclasses can layer cross-cutting concerns (tuning, logging, caching,
    etc.) on top of any runner without modifying its implementation.  Every
    :class:`PipelineRunner` method delegates to the wrapped runner, so
    wrappers can be stacked.

    Type Parameters:
        D_O: Data output type of the wrapped runner.
        M: Metrics output type of the wrapped runner.
        C: Configuration model for the wrapper itself.
    """

    def __init__(
        self,
        pipeline_runner: PipelineRunner,
        *,
        config: C,
    ) -> None:
        """Initialize the PipelineRunnerWrapper.

        Args:
            pipeline_runner: The runner instance to wrap.
            config: Wrapper-specific configuration.

        """
        self._pipeline_runner = pipeline_runner
        self._config = config
        # PipelineRunner methods that are not overridden below read these.
        self._pipeline = pipeline_runner.pipeline
        self._mode = pipeline_runner.mode

    # --- Convenience API

    @property
    def pipeline_runner(self) -> PipelineRunner:
        """Get the pipeline runner associated with this wrapper."""
        return self._pipeline_runner

    @property
    def pipeline(self) -> Pipeline:
        """Get the pipeline object from the underlying runner."""
        return self._pipeline_runner.pipeline

    @property
    def unwrapped(self) -> PipelineRunner:
        """Get the original, unwrapped pipeline runner."""
        return (
            self._pipeline_runner.unwrapped
            if hasattr(self._pipeline_runner, "unwrapped")
            else self._pipeline_runner
        )

    @property
    def config(self) -> C:
        """Get the configuration of this wrapper."""
        return self._config

    @property
    def mode(self) -> NodeExecutionMode:
        """Get the current execution mode of the underlying runner. The mode is supposed to be set automatically according to the function used."""
        return self._pipeline_runner.mode

    @property
    def node_objects(self) -> dict[str, Node]:
        """Get the mapping of node names to their instantiated objects in the underlying pipeline."""
        return self._pipeline_runner.node_objects

    def get_params(self) -> dict[str, dict[str, Any]]:
        """Get the parameters of all nodes in the underlying pipeline."""
        return self._pipeline_runner.get_params()

    def set_params(self, params: dict[str, dict[str, Any]]) -> None:
        """Set the parameters of all nodes in the underlying pipeline."""
        self._pipeline_runner.set_params(params=params)

    def save_params_to_dir(self, dir_path: str | Path) -> None:
        """Save the parameters of the underlying pipeline to a directory."""
        self._pipeline_runner.save_params_to_dir(dir_path=dir_path)

    def load_params_from_dir(self, dir_path: str | Path) -> None:
        """Load the parameters of the underlying pipeline from a directory."""
        self._pipeline_runner.load_params_from_dir(dir_path=dir_path)

    def get_metric_node_names(self) -> list[str]:
        """Get the metric node names of the underlying pipeline."""
        return self._pipeline_runner.get_metric_node_names()

    # --- API to implement for any PipelineRunnerWrapper implementation ---

    def train(
        self,
        input_data: Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]]
        | None = None,
    ) -> None:
        """Train the underlying pipeline runner.

        Args:
            input_data: Optional external data forwarded to the wrapped
                runner, keyed by ``{node_name: {port_name: (array, context)}}``.

        """
        return self._pipeline_runner.train(input_data=input_data)

    def infer(
        self,
        input_data: Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]]
        | None = None,
    ) -> Mapping[str, tuple[ArrayLike, DataContext]]:
        """Infer with the underlying pipeline runner.

        Args:
            input_data: Optional external data forwarded to the wrapped runner.

        Returns:
            Sink-node outputs as ``(array, context)`` tuples.

        """
        return self._pipeline_runner.infer(input_data=input_data)

    def evaluate(
        self,
        input_data: Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]]
        | None = None,
    ) -> Mapping[str, tuple[ArrayLike, DataContext]]:
        """Evaluate with the underlying pipeline runner.

        Args:
            input_data: Optional external data forwarded to the wrapped runner.

        Returns:
            Metric outputs as ``(array, context)`` tuples.

        """
        return self._pipeline_runner.evaluate(input_data=input_data)
