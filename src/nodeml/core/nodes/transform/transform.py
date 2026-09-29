"""Define the base Transform class for the NodeML Framework."""

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from nodeml.core.nodes.node import Node, NodeConfig, NodeMetadata, NodeType


class TransformMetadata(NodeMetadata):
    """Metadata for a TransformNode in a NodeML Pipeline."""

    _node_type: NodeType = NodeType.TRANSFORM
    trainable: bool = True  # Transforms are trainable by default, but this can be overridden for specific transforms that are not trainable (e.g., a transform that is just a wrapper around a pre-trained model that cannot be further trained).


class TransformRunningConfig(BaseModel):
    """Running configuration for a TransformNode in the NodeML Framework.

    This will usually be used for execution parameters that are not relevant for the definition of the transform itself, but rather for how to run it.
    For example, in some transforms, this could be very specific parameters such as the backend to use for computations etc.
    """


class TransformHyperParameters(BaseModel):
    """Hyperparameters for a TransformNode in the NodeML Framework.

    This will usually be used for parameters that are relevant for the definition of the transform itself, and that are relevant to be tuned during hyperparameter tuning.
    For example, in some transforms, this could be the window size for a rolling window transform, etc.
    """


R = TypeVar("R", bound=TransformRunningConfig)
H = TypeVar("H", bound=TransformHyperParameters)


class TransformConfig[H, R](NodeConfig):
    """Configuration for all Transform nodes in the NodeML Framework.

    Generic over two type parameters:

    * ``H`` -- a :class:`TransformHyperParameters` subclass holding tuneable
      parameters (e.g. window size, normalisation mode).
    * ``R`` -- a :class:`TransformRunningConfig` subclass holding runtime
      execution parameters.

    Attributes:
        node_type: Always ``NodeType.TRANSFORM``.
        hyperparameters: Tuneable hyperparameters for this transform.
        running_config: Runtime execution parameters.

    """

    node_type: NodeType = NodeType.TRANSFORM
    hyperparameters: H
    running_config: R


class TransformNode[D_I, D_C_I, D_O, D_C_O, P](Node[D_I, D_C_I, D_O, D_C_O], ABC):
    """Base class for all transform nodes in the NodeML Framework.

    Tracks a ``_fitted`` flag that is set to ``True`` after :meth:`node_fit`
    completes. Calling :meth:`node_transform` before fitting raises a
    ``ValueError``.
    """

    metadata = TransformMetadata()

    def __init__(self, *, config: TransformConfig) -> None:
        """Initialise the TransformNode with the given configuration.

        Args:
            config: Transform configuration including hyperparameters and
                runtime settings.

        """
        self._config = config
        self._fitted = False

    # --- Abstract Methods to reimplement ---

    @abstractmethod
    def fit(self, data: dict[str, tuple[D_I, D_C_I]]) -> None:
        """Fit the transform's internal state on the provided data.

        Args:
            data: Mapping of input port name to ``(data, context)`` tuples.

        """
        ...

    @abstractmethod
    def transform(
        self, data: dict[str, tuple[D_I, D_C_I]]
    ) -> dict[str, tuple[D_O, D_C_O]]:
        """Apply the fitted transform to the given data.

        Args:
            data: Mapping of input port name to ``(data, context)`` tuples.

        Returns:
            Mapping of output port name to ``(data, context)`` tuples.

        """
        ...

    # --- API convenience ---

    @property
    def running_config(self) -> TransformRunningConfig | None:
        """Convenience property to access the running configuration of the transform."""
        return self._config.running_config

    @property
    def hyperparameters(self) -> TransformHyperParameters | None:
        """Convenience property to access the hyperparameters of the transform."""
        return self._config.hyperparameters

    @property
    def config(self) -> TransformConfig:
        """Convenience property to access the full configuration of the transform."""
        return self._config

    @property
    def params(self) -> P:
        """Convenience property to access the current parameters of the transform."""
        return self.get_params()

    # --- Implementations for Node interface, don't touch without a very good reason ---

    def node_fit(self, data: dict[str, tuple[D_I, D_C_I]]) -> None:
        """Delegate to :meth:`fit` and mark the transform as fitted.

        Args:
            data: Mapping of input port name to ``(data, context)`` tuples.

        """
        self.fit(data=data)
        self._fitted = True

    def node_transform(
        self, data: dict[str, tuple[D_I, D_C_I]]
    ) -> dict[str, tuple[D_O, D_C_O]]:
        """Delegate to :meth:`transform` after verifying the node has been fitted.

        Args:
            data: Mapping of input port name to ``(data, context)`` tuples.

        Returns:
            Mapping of output port name to ``(data, context)`` tuples.

        Raises:
            ValueError: If the transform has not been fitted yet.

        """
        if hasattr(self, "_fitted") and not self._fitted:
            msg = "TransformNode must be fitted before calling transform."
            raise ValueError(msg)
        return self.transform(data=data)
