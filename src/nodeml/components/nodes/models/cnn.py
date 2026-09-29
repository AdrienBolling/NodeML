"""CNN (1-D Convolutional Neural Network) model node for the NodeML Framework.

Implements a 1-D CNN regressor suited for time-series / sequential tabular
data.  The architecture stacks a configurable number of ``Conv1d`` blocks
(each optionally followed by pooling and dropout), then flattens and passes
through fully-connected aggregation layers whose count and width are also
tuneable.

The network receives a 2-D feature tensor ``(batch, features)`` and
reshapes it to ``(batch, 1, features)`` (single-channel sequence) before
it applies the convolutions.

Ports:

* ``X`` -- feature tensor ``(batch, features)``
* ``y`` -- target tensor ``(batch, targets)`` (training / evaluation only)
* ``pred`` -- float64 predictions ``(batch, targets)`` as a CPU tensor

The node does regression only: it trains with the mean squared error loss
and the output layer has no activation.
"""

from typing import Any, Literal

import torch
from pydantic import Field
from ray import tune
from torch import nn

from nodeml.components.nodes.models._torch_base import (
    ACTIVATIONS,
    TorchModelNode,
    TorchRunningConfig,
)
from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.nodes.models.model import (
    ModelConfig,
    ModelHyperParameters,
    ModelMetadata,
)
from nodeml.core.nodes.node import Port


class CNNMetadata(ModelMetadata):
    """Metadata for the CNN model node."""

    node_name: str = "CNN"
    description: str = (
        "1-D Convolutional Neural Network built with PyTorch, for regression. "
        "Supports tuneable convolutional blocks (filters, kernel size, "
        "pooling) and fully-connected aggregation layers. "
        "Trains with the mean squared error loss."
    )


class CNNHyperParameters(ModelHyperParameters):
    """Tuneable hyperparameters for the CNN.

    Controls the convolutional backbone and the aggregation head.
    """

    # --- Convolutional backbone ---
    num_conv_layers: int = Field(
        default=2,
        ge=1,
        description="Number of Conv1d blocks stacked sequentially.",
    )
    num_filters: int = Field(
        default=32,
        ge=1,
        description=(
            "Number of output filters (channels) in each Conv1d layer. "
            "All convolutional layers share the same width."
        ),
    )
    kernel_size: int = Field(
        default=3,
        ge=1,
        description="Kernel size for every Conv1d layer.",
    )
    pooling: Literal["max", "avg", "none"] = Field(
        default="max",
        description=(
            "Pooling strategy applied after each convolutional block. "
            "``'none'`` disables pooling."
        ),
    )
    pool_size: int = Field(
        default=2,
        ge=1,
        description="Kernel size for the pooling layer (ignored when pooling is ``'none'``).",
    )

    # --- Aggregation head ---
    num_fc_layers: int = Field(
        default=1,
        ge=1,
        description="Number of fully-connected layers after the convolutional backbone.",
    )
    fc_units: int = Field(
        default=64,
        ge=1,
        description="Number of units in each fully-connected aggregation layer.",
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Dropout probability applied after each conv and FC layer.",
    )
    activation: Literal["relu", "tanh", "sigmoid"] = Field(
        default="relu",
        description="Activation function used throughout the network.",
    )


class CNNRunningConfig(TorchRunningConfig):
    """Execution-time options for the CNN."""


hyperparameter_space: dict[str, Any] = {
    "num_conv_layers": tune.choice([1, 2, 3]),
    "num_filters": tune.choice([16, 32, 64, 128]),
    "kernel_size": tune.choice([3, 5, 7]),
    "pooling": tune.choice(["max", "avg", "none"]),
    "pool_size": tune.choice([2, 3]),
    "num_fc_layers": tune.choice([1, 2, 3]),
    "fc_units": tune.choice([32, 64, 128, 256]),
    "dropout": tune.uniform(0.0, 0.5),
    "activation": tune.choice(["relu", "tanh", "sigmoid"]),
}


class _CNNNetwork(nn.Module):
    """Convolutional backbone followed by a fully-connected head."""

    def __init__(self, conv: nn.Module, fc: nn.Module) -> None:
        """Store the two parts of the network.

        Args:
            conv: Backbone that maps ``(batch, 1, features)`` to
                ``(batch, filters, length)``.
            fc: Head that maps ``(batch, filters * length)`` to
                ``(batch, targets)``.

        """
        super().__init__()
        self.conv = conv
        self.fc = fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, features)`` to ``(batch, targets)``."""
        x = self.conv(x.unsqueeze(1))
        return self.fc(x.flatten(1))


def _build_cnn(  # noqa: PLR0913 - one argument per architecture setting
    in_features: int,
    out_features: int,
    num_conv_layers: int,
    num_filters: int,
    kernel_size: int,
    pooling: str,
    pool_size: int,
    num_fc_layers: int,
    fc_units: int,
    dropout: float,
    activation: str,
) -> nn.Module:
    """Build a 1-D CNN from config values.

    Args:
        in_features: Number of feature columns (the sequence length).
        out_features: Number of target columns.
        num_conv_layers: Number of ``Conv1d`` blocks.
        num_filters: Number of filters in each ``Conv1d`` layer.
        kernel_size: Kernel size of each ``Conv1d`` layer.
        pooling: ``"max"``, ``"avg"`` or ``"none"``.
        pool_size: Kernel size of the pooling layers.
        num_fc_layers: Number of hidden fully-connected layers.
        fc_units: Width of each hidden fully-connected layer.
        dropout: Dropout probability after each block and layer.
        activation: Name of the activation function.

    Returns:
        A network that maps ``(batch, in_features)`` to
        ``(batch, out_features)``.

    """
    act_cls = ACTIVATIONS[activation]

    # --- Convolutional backbone ---
    conv_layers: list[nn.Module] = []
    in_channels = 1
    seq_len = in_features
    for _ in range(num_conv_layers):
        # Pad to preserve length before pooling shrinks it.
        padding = kernel_size // 2
        conv_layers.append(
            nn.Conv1d(in_channels, num_filters, kernel_size, padding=padding)
        )
        conv_layers.append(act_cls())
        if dropout > 0:
            conv_layers.append(nn.Dropout(dropout))
        if pooling != "none" and seq_len >= pool_size:
            if pooling == "max":
                conv_layers.append(nn.MaxPool1d(pool_size))
            else:
                conv_layers.append(nn.AvgPool1d(pool_size))
            seq_len = seq_len // pool_size
        in_channels = num_filters

    conv_backbone = nn.Sequential(*conv_layers)
    flat_size = num_filters * seq_len

    # --- Fully-connected aggregation head ---
    fc_layers: list[nn.Module] = []
    prev = flat_size
    for _ in range(num_fc_layers):
        fc_layers.append(nn.Linear(prev, fc_units))
        fc_layers.append(act_cls())
        if dropout > 0:
            fc_layers.append(nn.Dropout(dropout))
        prev = fc_units
    fc_layers.append(nn.Linear(prev, out_features))
    return _CNNNetwork(conv_backbone, nn.Sequential(*fc_layers))


class CNNConfig(
    ModelConfig[
        CNNHyperParameters,
        CNNRunningConfig,
    ]
):
    """Full configuration for the CNN node."""

    hyperparameters: CNNHyperParameters = Field(
        default_factory=CNNHyperParameters,
        description="Tuneable hyperparameters (conv layers, filters, kernel, pooling, FC head, dropout, activation).",
    )
    running_config: CNNRunningConfig = Field(
        default_factory=CNNRunningConfig,
        description="Execution-time options (learning_rate, epochs, batch_size, random_state, device).",
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


class CNNNode(TorchModelNode):
    """1-D Convolutional Neural Network regression node.

    :class:`TorchModelNode` builds and trains the network in :meth:`fit`,
    once the input and output sizes are known.  :meth:`set_params` rebuilds
    the network on a new node.
    """

    metadata = CNNMetadata()
    hyperparameter_space = hyperparameter_space

    def _build_network(self, in_features: int, out_features: int) -> nn.Module:
        """Return a new CNN for the node config.

        Args:
            in_features: Number of feature columns.
            out_features: Number of target columns.

        Returns:
            The network.

        """
        hp = self._config.hyperparameters
        return _build_cnn(
            in_features=in_features,
            out_features=out_features,
            num_conv_layers=hp.num_conv_layers,
            num_filters=hp.num_filters,
            kernel_size=hp.kernel_size,
            pooling=hp.pooling,
            pool_size=hp.pool_size,
            num_fc_layers=hp.num_fc_layers,
            fc_units=hp.fc_units,
            dropout=hp.dropout,
            activation=hp.activation,
        )
