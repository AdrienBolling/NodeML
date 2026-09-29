"""CNN (1-D Convolutional Neural Network) model node for the NodeML Framework.

Implements a 1-D CNN suited for time-series / sequential tabular data.  The
architecture stacks a configurable number of ``Conv1d`` blocks (each
optionally followed by pooling and dropout), then flattens and passes
through fully-connected aggregation layers whose count and width are also
tuneable.

Input conventions
-----------------
The node receives a 2-D feature tensor ``(batch, features)`` and internally
reshapes it to ``(batch, 1, features)`` (single-channel sequence) before
applying the convolutions.

Ports
-----
* ``X`` -- feature tensor ``(batch, features)``
* ``y`` -- target tensor ``(batch, targets)`` (training / evaluation only)
* ``pred`` -- predicted values ``(batch, targets)``
"""

from typing import Any, Literal

import torch
from pydantic import Field
from ray import tune
from torch import nn

from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    TabularDataContext,
    tabular_context_from_dict_dump,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.nodes.models.model import (
    Model,
    ModelConfig,
    ModelHyperParameters,
    ModelMetadata,
    ModelRunningConfig,
)
from nodeml.core.nodes.node import Port


class CNNMetadata(ModelMetadata):
    """Metadata for the CNN model node."""

    node_name: str = "CNN"
    description: str = (
        "1-D Convolutional Neural Network built with PyTorch. "
        "Supports tuneable convolutional blocks (filters, kernel size, "
        "pooling) and fully-connected aggregation layers."
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


class CNNRunningConfig(ModelRunningConfig):
    """Execution-time options for the CNN."""

    learning_rate: float = Field(
        default=1e-3,
        gt=0,
        description="Learning rate for the Adam optimiser.",
    )
    epochs: int = Field(
        default=100,
        ge=1,
        description="Number of training epochs.",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Mini-batch size for training.",
    )
    random_state: int | None = Field(
        default=None,
        ge=0,
        description="Seed for PyTorch RNG. ``None`` uses non-deterministic behaviour.",
    )


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

type _CNNParams = dict[str, Any]

_ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


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

    Returns an ``nn.Module`` whose forward pass expects input shaped
    ``(batch, 1, features)`` and produces ``(batch, out_features)``.
    """
    act_cls = _ACTIVATION_MAP[activation]

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
    fc_head = nn.Sequential(*fc_layers)

    class _CNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = conv_backbone
            self.fc = fc_head

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv(x)
            x = x.flatten(1)
            return self.fc(x)

    return _CNN()


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
        description="Execution-time options (learning_rate, epochs, batch_size, random_state).",
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
                desc="Predicted values (batch, targets).",
            ),
        },
        description="Output ports: 'pred' (predicted targets).",
    )


class CNNNode(
    Model[
        torch.Tensor,
        TabularDataContext,
        torch.Tensor,
        TabularDataContext,
        _CNNParams,
    ]
):
    """1-D Convolutional Neural Network model node.

    The PyTorch network is lazily built on the first :meth:`fit` call
    (once input/output dimensions are known).  Input tensors
    ``(batch, features)`` are reshaped to ``(batch, 1, features)``
    internally.
    """

    metadata = CNNMetadata()
    hyperparameter_space = hyperparameter_space

    def __init__(self, *, config: CNNConfig) -> None:
        """Store *config*; the network is built lazily in :meth:`fit`."""
        self._config = config
        self._model: nn.Module | None = None
        self._target_context_dump: dict[str, list[str]] = {}

        if config.running_config.random_state is not None:
            torch.manual_seed(config.running_config.random_state)

    # --- Model interface --------------------------------------------------

    def fit(self, data: dict[str, tuple[torch.Tensor, TabularDataContext]]) -> None:
        """Train the CNN on the provided *(X, y)* pair.

        Parameters
        ----------
        data:
            Must contain keys ``"X"`` (features) and ``"y"`` (targets).

        """
        X, _ = data["X"]
        y, y_ctx = data["y"]

        X = X.float()
        y = y.float()
        if y.ndim == 1:
            y = y.unsqueeze(1)

        in_features = X.shape[1]
        out_features = y.shape[1]
        hp = self._config.hyperparameters

        self._model = _build_cnn(
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

        # Reshape X to (batch, 1, features) for Conv1d.
        X_conv = X.unsqueeze(1)

        optimiser = torch.optim.Adam(
            self._model.parameters(),
            lr=self._config.running_config.learning_rate,
        )
        loss_fn = nn.MSELoss()
        batch_size = self._config.running_config.batch_size
        n = X.shape[0]

        self._model.train()
        for _ in range(self._config.running_config.epochs):
            perm = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                X_batch = X_conv[idx]
                y_batch = y[idx]
                optimiser.zero_grad()
                pred = self._model(X_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimiser.step()

        self._target_context_dump = y_ctx.dump_dict

    def predict(
        self, data: dict[str, tuple[torch.Tensor, TabularDataContext]]
    ) -> dict[str, tuple[torch.Tensor, TabularDataContext]]:
        """Predict using the fitted CNN.

        Parameters
        ----------
        data:
            Must contain key ``"X"`` (features).

        Returns
        -------
        dict
            ``{"pred": (predictions, context)}`` where predictions is a
            2-D tensor ``(batch, targets)``.

        """
        X, _ = data["X"]
        X = X.float().unsqueeze(1)  # (batch, 1, features)

        self._model.eval()
        with torch.no_grad():
            pred = self._model(X)

        pred_ctx = tabular_context_from_dict_dump(self._target_context_dump)
        return {"pred": (pred, pred_ctx)}

    def get_params(self) -> _CNNParams:
        """Return the serialisable state of this node."""
        return {
            "model_state_dict": self._model.state_dict() if self._model else {},
            "target_context": self._target_context_dump,
        }

    def set_params(self, params: _CNNParams) -> None:
        """Restore node state from a previously serialised param dict."""
        if self._model is not None and params["model_state_dict"]:
            self._model.load_state_dict(params["model_state_dict"])
        self._target_context_dump = params["target_context"]
