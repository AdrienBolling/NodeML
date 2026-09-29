"""Define the base Model class for the NodeML Framework."""

from abc import ABC, abstractmethod

from pydantic.main import BaseModel

from nodeml.core.nodes.node import (
    Node,
    NodeConfig,
    NodeMetadata,
    NodeType,
)


class ModelMetadata(NodeMetadata):
    """Metadata for a Model in a NodeML Pipeline."""

    _node_type: NodeType = NodeType.MODEL
    trainable: bool = True  # Models are trainable by default, but this can be overridden for specific models that are not trainable (e.g., a model that is just a wrapper around a pre-trained model that cannot be further trained).


class ModelRunningConfig(BaseModel):
    """Running configuration for a Model in the NodeML Framework.

    This will usually be used for execution parameters that are not relevant for the definition of the model itself, but rather for how to run it.
    For example, in most ML models, this could be very specific training parameters such as whether to enable bootstrapping, the precise backend to use for computations etc.
    """


class ModelHyperParameters(BaseModel):
    """Hyperparameters for a Model in the NodeML Framework.

    This will usually be used for parameters that are relevant for the definition of the model itself, and that are relevant to be tuned during hyperparameter tuning.
    For example, in most ML models, this could be the learning rate, the number of layers, etc.
    """


class ModelConfig[H: ModelHyperParameters, R: ModelRunningConfig](NodeConfig):
    """Base configuration for all Models in the NodeML Framework.

    Generic over two type parameters:

    * ``H`` -- a :class:`ModelHyperParameters` subclass holding tuneable
      hyperparameters (e.g. learning rate, layer count).
    * ``R`` -- a :class:`ModelRunningConfig` subclass holding runtime
      execution parameters (e.g. backend, bootstrapping flags).

    Attributes:
        node_type: Always ``NodeType.MODEL``.
        hyperparameters: Tuneable hyperparameters for this model.
        running_config: Runtime execution parameters.

    """

    node_type: NodeType = NodeType.MODEL
    hyperparameters: H
    running_config: R


class Model[D_I, D_C_I, D_O, D_C_O, P](
    Node[D_I, D_C_I, D_O, D_C_O], ABC
):  # Model is already implicitely an ABC via Node but explicit is better.
    """Base class for all models in the NodeML Framework.

    Bridges the Node interface to a conventional ML fit/predict API:
    :meth:`node_fit` delegates to :meth:`fit` and :meth:`node_transform`
    delegates to :meth:`predict`.

    The additional generic parameter ``P`` represents the type of the model's
    learned parameters, accessible via :meth:`get_params` / :meth:`set_params`.
    """

    metadata = ModelMetadata()

    def __init__(self, *, config: ModelConfig) -> None:
        """Minimal constructor for Model class."""
        self._config = config

    # --- Abstract Methods to reimplement ---

    @abstractmethod
    def fit(self, data: dict[str, tuple[D_I, D_C_I]]) -> None:
        """Fit the model on the provided data.

        Args:
            data: Mapping of input port name to ``(data, context)`` tuples.

        """
        ...

    @abstractmethod
    def predict(
        self, data: dict[str, tuple[D_I, D_C_I]]
    ) -> dict[str, tuple[D_O, D_C_O]]:
        """Generate predictions from the fitted model.

        Args:
            data: Mapping of input port name to ``(data, context)`` tuples.

        Returns:
            Mapping of output port name to ``(data, context)`` tuples.

        """
        ...

    @abstractmethod
    def get_params(self) -> P:
        """Return the model's current learned parameters.

        Returns:
            The learned parameters object of type ``P``.

        """
        ...

    @abstractmethod
    def set_params(self, params: P) -> None:
        """Replace the model's learned parameters.

        Args:
            params: New parameters to set on the model.

        """
        ...

    # --- API convenience ---

    @property
    def running_config(self) -> ModelRunningConfig | None:
        """Property to get the model running configuration."""
        return self._config.running_config

    @property
    def hyperparameters(self) -> ModelHyperParameters | None:
        """Property to get the model hyperparameters."""
        return self._config.hyperparameters

    @property
    def parameters(self) -> P:
        """Property to get the model parameters."""
        return self.get_params()

    @property
    def config(self) -> ModelConfig:
        """Property to get the full model configuration."""
        return self._config

    # --- Implementations for Node interface, don't touch without a very good reason ---

    def node_fit(self, data: dict[str, tuple[D_I, D_C_I]]) -> None:
        """Delegate to :meth:`fit`.

        Args:
            data: Mapping of input port name to ``(data, context)`` tuples.

        """
        return self.fit(data=data)

    def node_transform(
        self, data: dict[str, tuple[D_I, D_C_I]]
    ) -> dict[str, tuple[D_O, D_C_O]]:
        """Delegate to :meth:`predict`.

        Args:
            data: Mapping of input port name to ``(data, context)`` tuples.

        Returns:
            Mapping of output port name to ``(data, context)`` tuples.

        """
        return self.predict(data=data)
