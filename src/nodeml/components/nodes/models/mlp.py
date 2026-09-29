"""MLP (Multi-Layer Perceptron) model node for the NodeML Framework.

Implements a simple feed-forward neural network with a configurable number
of hidden layers and units per layer using PyTorch.  The node expects two
input ports:

* ``X`` -- feature matrix ``(batch, features)`` as a torch Tensor
* ``y`` -- target matrix ``(batch, targets)`` as a torch Tensor (training / evaluation only)

and emits one output port:

* ``pred`` -- predicted values ``(batch, targets)`` as a torch Tensor
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


class MLPMetadata(ModelMetadata):
    """Metadata for the MLP model node."""

    node_name: str = "MLP"
    description: str = (
        "Multi-Layer Perceptron built with PyTorch. "
        "Supports configurable hidden layers, activation functions, "
        "and dropout for both regression and classification tasks."
    )


class MLPHyperParameters(ModelHyperParameters):
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


class MLPRunningConfig(ModelRunningConfig):
    """Execution-time options for the MLP.

    These affect training behaviour but are usually held fixed during
    hyperparameter search.
    """

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
        description=(
            "Seed for PyTorch random-number generators. "
            "``None`` uses non-deterministic behaviour."
        ),
    )


hyperparameter_space: dict[str, Any] = {
    "hidden_layers": tune.choice([1, 2, 3, 4]),
    "hidden_units": tune.choice([32, 64, 128, 256]),
    "dropout": tune.uniform(0.0, 0.5),
    "activation": tune.choice(["relu", "tanh", "sigmoid"]),
}

type _MLPParams = dict[str, Any]

_ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def _build_mlp(
    in_features: int,
    out_features: int,
    hidden_layers: int,
    hidden_units: int,
    dropout: float,
    activation: str,
) -> nn.Sequential:
    """Build a simple feed-forward network from config values."""
    act_cls = _ACTIVATION_MAP[activation]
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
    ModelConfig[
        MLPHyperParameters,
        MLPRunningConfig,
    ]
):
    """Full configuration for the MLP node."""

    hyperparameters: MLPHyperParameters = Field(
        default_factory=MLPHyperParameters,
        description="Tuneable hyperparameters (hidden_layers, hidden_units, dropout, activation).",
    )
    running_config: MLPRunningConfig = Field(
        default_factory=MLPRunningConfig,
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


class MLPNode(
    Model[
        torch.Tensor,
        TabularDataContext,
        torch.Tensor,
        TabularDataContext,
        _MLPParams,
    ]
):
    """Multi-Layer Perceptron model node.

    The PyTorch network is lazily built on the first :meth:`fit` call
    (once input/output dimensions are known).  The target context is
    captured during :meth:`fit` and replayed on every :meth:`predict`
    call.
    """

    metadata = MLPMetadata()
    hyperparameter_space = hyperparameter_space

    def __init__(self, *, config: MLPConfig) -> None:
        """Store *config*; the network is built lazily in :meth:`fit`."""
        self._config = config
        self._model: nn.Sequential | None = None
        self._target_context_dump: dict[str, list[str]] = {}

        if config.running_config.random_state is not None:
            torch.manual_seed(config.running_config.random_state)

    # --- Model interface --------------------------------------------------

    def fit(self, data: dict[str, tuple[torch.Tensor, TabularDataContext]]) -> None:
        """Train the MLP on the provided *(X, y)* pair.

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

        self._model = _build_mlp(
            in_features=in_features,
            out_features=out_features,
            hidden_layers=hp.hidden_layers,
            hidden_units=hp.hidden_units,
            dropout=hp.dropout,
            activation=hp.activation,
        )

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
                X_batch, y_batch = X[idx], y[idx]
                optimiser.zero_grad()
                pred = self._model(X_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimiser.step()

        self._target_context_dump = y_ctx.dump_dict

    def predict(
        self, data: dict[str, tuple[torch.Tensor, TabularDataContext]]
    ) -> dict[str, tuple[torch.Tensor, TabularDataContext]]:
        """Predict using the fitted MLP.

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
        X = X.float()

        self._model.eval()
        with torch.no_grad():
            pred = self._model(X)

        pred_ctx = tabular_context_from_dict_dump(self._target_context_dump)
        return {"pred": (pred, pred_ctx)}

    def get_params(self) -> _MLPParams:
        """Return the serialisable state of this node."""
        return {
            "model_state_dict": self._model.state_dict() if self._model else {},
            "target_context": self._target_context_dump,
        }

    def set_params(self, params: _MLPParams) -> None:
        """Restore node state from a previously serialised param dict."""
        if self._model is not None and params["model_state_dict"]:
            self._model.load_state_dict(params["model_state_dict"])
        self._target_context_dump = params["target_context"]
