"""MLP (Multi-Layer Perceptron) model node for the NodeML Framework.

Implements a simple feed-forward regression network with a configurable
number of hidden layers and units per layer using PyTorch.  The node
expects two input ports:

* ``X`` -- feature matrix ``(batch, features)`` as a torch Tensor
* ``y`` -- target matrix ``(batch, targets)`` as a torch Tensor (training / evaluation only)

and emits one output port:

* ``pred`` -- float64 predictions ``(batch, targets)`` as a CPU torch Tensor

The node does regression only: it trains with the mean squared error loss
and the output layer has no activation.
"""

from typing import Any, Literal

from pydantic import Field
from ray import tune
from torch import nn

from nodeml.components.nodes.models._torch_base import (
    ACTIVATIONS,
    TorchHyperParameters,
    TorchModelConfig,
    TorchModelNode,
    TorchRunningConfig,
)
from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.nodes.models.model import ModelMetadata
from nodeml.core.nodes.node import Port


class MLPMetadata(ModelMetadata):
    """Metadata for the MLP model node."""

    node_name: str = "MLP"
    description: str = (
        "Multi-Layer Perceptron built with PyTorch, for regression. "
        "Supports configurable hidden layers, activation functions "
        "and dropout. Trains with the mean squared error loss."
    )


class MLPHyperParameters(TorchHyperParameters):
    """Tuneable hyperparameters for the MLP.

    These are the parameters that make sense to explore during
    hyperparameter search.  All other knobs live in
    :class:`MLPRunningConfig`.
    """

    hidden_layers: int = Field(
        default=2,
        ge=1,
        description=(
            "Number of hidden layers in the network. "
            "More layers increase capacity but may overfit on small datasets."
        ),
    )
    hidden_units: int = Field(
        default=64,
        ge=1,
        description=(
            "Number of units in each hidden layer. "
            "All hidden layers share the same width."
        ),
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Dropout probability applied after each hidden layer. "
            "``0.0`` disables dropout."
        ),
    )
    activation: Literal["relu", "tanh", "sigmoid"] = Field(
        default="relu",
        description="Activation function applied after each hidden layer.",
    )


class MLPRunningConfig(TorchRunningConfig):
    """Execution-time options for the MLP.

    These affect training behaviour but are usually held fixed during
    hyperparameter search.
    """


hyperparameter_space: dict[str, Any] = {
    "learning_rate": tune.loguniform(1e-4, 1e-2),
    "hidden_layers": tune.choice([1, 2, 3, 4]),
    "hidden_units": tune.choice([32, 64, 128, 256]),
    "dropout": tune.uniform(0.0, 0.5),
    "activation": tune.choice(["relu", "tanh", "sigmoid"]),
}


def _build_mlp(
    in_features: int,
    out_features: int,
    hidden_layers: int,
    hidden_units: int,
    dropout: float,
    activation: str,
) -> nn.Sequential:
    """Build a feed-forward network from config values.

    Args:
        in_features: Number of feature columns.
        out_features: Number of target columns.
        hidden_layers: Number of hidden layers.
        hidden_units: Width of each hidden layer.
        dropout: Dropout probability after each hidden layer.
        activation: Name of the activation after each hidden layer.

    Returns:
        A network that maps ``(batch, in_features)`` to
        ``(batch, out_features)``.

    """
    act_cls = ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    prev = in_features
    for _ in range(hidden_layers):
        layers.append(nn.Linear(prev, hidden_units))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = hidden_units
    layers.append(nn.Linear(prev, out_features))
    return nn.Sequential(*layers)


class MLPConfig(
    TorchModelConfig[
        MLPHyperParameters,
        MLPRunningConfig,
    ]
):
    """Full configuration for the MLP node."""

    hyperparameters: MLPHyperParameters = Field(
        default_factory=MLPHyperParameters,
        description="Tuneable hyperparameters (learning_rate, hidden_layers, hidden_units, dropout, activation).",
    )
    running_config: MLPRunningConfig = Field(
        default_factory=MLPRunningConfig,
        description="Execution-time options (epochs, batch_size, random_state, device).",
    )
    in_ports: dict[str, Port] = Field(
        default={
            "X": Port(
                arr_type=ArrayLikeEnum.TORCH,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch features",
                desc="Input feature matrix (batch, features).",
            ),
            "y": Port(
                arr_type=ArrayLikeEnum.TORCH,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch targets",
                mode=[NodeExecutionMode.TRAINING, NodeExecutionMode.EVALUATION],
                desc="Target values (batch, targets). Required during training and evaluation only.",
            ),
        },
        description="Input ports: 'X' (features, all modes) and 'y' (targets, training/evaluation).",
    )
    out_ports: dict[str, Port] = Field(
        default={
            "pred": Port(
                arr_type=ArrayLikeEnum.TORCH,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch targets",
                desc="Predicted values (float64, on the CPU), one column for each target.",
            ),
        },
        description="Output ports: 'pred' (predicted targets).",
    )


class MLPNode(TorchModelNode):
    """Multi-Layer Perceptron regression node.

    :class:`TorchModelNode` builds and trains the network in :meth:`fit`,
    once the input and output sizes are known.  :meth:`set_params` rebuilds
    the network on a new node.
    """

    metadata = MLPMetadata()
    hyperparameter_space = hyperparameter_space

    def _build_network(self, in_features: int, out_features: int) -> nn.Module:
        """Return a new MLP for the node config.

        Args:
            in_features: Number of feature columns.
            out_features: Number of target columns.

        Returns:
            The network.

        """
        hp = self._config.hyperparameters
        return _build_mlp(
            in_features=in_features,
            out_features=out_features,
            hidden_layers=hp.hidden_layers,
            hidden_units=hp.hidden_units,
            dropout=hp.dropout,
            activation=hp.activation,
        )
