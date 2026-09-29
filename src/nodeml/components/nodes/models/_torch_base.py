"""Shared base classes for the PyTorch regression model nodes."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

import torch
from pydantic import Field
from torch import nn

from nodeml.components.nodes.models._common import (
    not_fitted_error,
    prediction_context,
)
from nodeml.core.common.data.data import TabularDataContext
from nodeml.core.common.exceptions import NodeConfigError, NodeInputError
from nodeml.core.nodes.models.model import Model, ModelConfig, ModelRunningConfig

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}

type TorchParams = dict[str, Any]


class TorchRunningConfig(ModelRunningConfig):
    """Training options that the PyTorch model nodes share.

    These options change how the node trains, but not the network.  They
    usually stay fixed during a hyperparameter search.
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
            "Seed for the weight initialisation, the shuffling and the dropout. "
            "The node does not change the global PyTorch RNG when a seed is set. "
            "``None`` uses the global PyTorch RNG, so the results change."
        ),
    )
    device: Literal["cpu", "cuda", "auto"] = Field(
        default="cpu",
        description=(
            "Device for training and prediction. "
            "``'auto'`` uses CUDA if it is available, else the CPU. "
            "Predictions always come back on the CPU."
        ),
    )


class TorchModelNode(
    Model[
        torch.Tensor,
        TabularDataContext,
        torch.Tensor,
        TabularDataContext,
        TorchParams,
    ],
    ABC,
):
    """Base class for a regression node that trains a PyTorch network.

    A subclass implements :meth:`_build_network`.  This class does the rest:

    * :meth:`fit` builds a new network and trains it with Adam and the mean
      squared error loss.  If ``random_state`` is set, the training is
      repeatable and does not change the global PyTorch RNG.
    * :meth:`predict` returns float64 predictions on the CPU.  The output
      columns take the names of the training targets.
    * :meth:`get_params` returns the weights and the network size, so that
      :meth:`set_params` can rebuild the network on a new node.

    The nodes do regression only: the loss is the mean squared error and
    the output has no activation.
    """

    def __init__(self, *, config: ModelConfig) -> None:
        """Store the config.  The network is built in :meth:`fit`.

        Args:
            config: The node configuration.

        """
        super().__init__(config=config)
        self._network: nn.Module | None = None
        self._in_features: int | None = None
        self._out_features: int | None = None
        # The output columns take the target column names.
        self._target_context_dump: dict[str, list[str]] = {}

    @abstractmethod
    def _build_network(self, in_features: int, out_features: int) -> nn.Module:
        """Return a new network for the node config.

        Args:
            in_features: Number of feature columns.
            out_features: Number of target columns.

        Returns:
            A module that maps a ``(batch, in_features)`` float32 tensor to
            a ``(batch, out_features)`` tensor.

        """

    # --- Helpers ----------------------------------------------------------

    def _device(self) -> torch.device:
        """Return the device that the running config selects.

        Raises:
            NodeConfigError: If the config asks for CUDA and CUDA is not
                available.

        """
        requested = self._config.running_config.device
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            msg = (
                f"{type(self).__name__} has device='cuda', but CUDA is not "
                "available. Set device to 'cpu' or 'auto'."
            )
            raise NodeConfigError(msg)
        return torch.device(requested)

    @contextmanager
    def _seeded_rng(self, device: torch.device) -> Iterator[None]:
        """Seed the RNG for the code in the block, if the config has a seed.

        The block runs on a copy of the RNG state.  At the end of the block
        the global RNG state comes back, so other code sees no change.
        Without a seed, the block uses the global RNG as usual.

        Args:
            device: The training device.  Its CUDA RNG is seeded too.

        Yields:
            Nothing.

        """
        seed = self._config.running_config.random_state
        if seed is None:
            yield
            return
        cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            # Seed only the forked generators: torch.manual_seed would also
            # seed the CUDA generators of the other devices.
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed(seed)
            yield

    def _train(self, network: nn.Module, X: torch.Tensor, y: torch.Tensor) -> None:
        """Train *network* on *X* and *y* with Adam and the MSE loss.

        Args:
            network: The network to train, on the device of *X*.
            X: Float32 features ``(batch, features)``.
            y: Float32 targets ``(batch, targets)``.

        """
        running = self._config.running_config
        optimiser = torch.optim.Adam(network.parameters(), lr=running.learning_rate)
        loss_fn = nn.MSELoss()
        n_rows = X.shape[0]
        network.train()
        for _ in range(running.epochs):
            # Shuffle on the CPU, so that the order is the same on all devices.
            order = torch.randperm(n_rows).to(X.device)
            for start in range(0, n_rows, running.batch_size):
                batch = order[start : start + running.batch_size]
                optimiser.zero_grad()
                loss = loss_fn(network(X[batch]), y[batch])
                loss.backward()
                optimiser.step()
        network.eval()

    # --- Model interface --------------------------------------------------

    def fit(self, data: dict[str, tuple[torch.Tensor, TabularDataContext]]) -> None:
        """Build a new network and train it on the ``X`` and ``y`` ports.

        Args:
            data: Must contain the keys ``"X"`` (features) and ``"y"``
                (targets).

        Raises:
            NodeConfigError: If the config asks for CUDA and CUDA is not
                available.

        """
        X, _ = data["X"]
        y, y_ctx = data["y"]
        device = self._device()
        X = X.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        if y.ndim == 1:
            y = y.unsqueeze(1)
        with self._seeded_rng(device):
            # Build on the CPU, so that the initial weights do not depend
            # on the device.
            network = self._build_network(X.shape[1], y.shape[1]).to(device)
            self._train(network, X, y)
        self._network = network
        self._in_features = X.shape[1]
        self._out_features = y.shape[1]
        self._target_context_dump = y_ctx.dump_dict

    def predict(
        self, data: dict[str, tuple[torch.Tensor, TabularDataContext]]
    ) -> dict[str, tuple[torch.Tensor, TabularDataContext]]:
        """Predict on the ``X`` port.

        Args:
            data: Must contain the key ``"X"`` (features).  The node ignores
                ``"y"``.

        Returns:
            ``{"pred": (predictions, context)}``.  The predictions are a
            float64 CPU tensor of shape ``(batch, targets)``.

        Raises:
            NodeNotFittedError: If the node is not fitted.
            NodeInputError: If ``X`` does not have the number of features
                that the node was fitted on.

        """
        if self._network is None:
            raise not_fitted_error(self)
        X, _ = data["X"]
        if X.shape[1] != self._in_features:
            msg = (
                f"{type(self).__name__} was fitted on {self._in_features} features, "
                f"but X has {X.shape[1]} columns."
            )
            raise NodeInputError(msg)
        device = next(self._network.parameters()).device
        self._network.eval()
        with torch.no_grad():
            pred = self._network(X.to(device=device, dtype=torch.float32))
        pred = pred.to(device="cpu", dtype=torch.float64)
        return {
            "pred": (pred, prediction_context(self._target_context_dump["columns"]))
        }

    def get_params(self) -> TorchParams:
        """Return the fitted state of the node.

        Returns:
            A picklable dict with these keys:

            * ``"model_state_dict"``: the network weights as CPU tensors, or
              an empty dict if the node is not fitted.
            * ``"in_features"`` and ``"out_features"``: the network size.
            * ``"target_context"``: the dumped context of the training
              targets.

        """
        state: dict[str, torch.Tensor] = {}
        if self._network is not None:
            state = {
                key: value.detach().to("cpu", copy=True)
                for key, value in self._network.state_dict().items()
            }
        return {
            "model_state_dict": state,
            "in_features": self._in_features,
            "out_features": self._out_features,
            "target_context": self._target_context_dump,
        }

    def set_params(self, params: TorchParams) -> None:
        """Rebuild the network from :meth:`get_params` output.

        The node needs no :meth:`fit` call after this method.

        Args:
            params: The output of :meth:`get_params`.

        Raises:
            NodeInputError: If the params have weights but no network size.
                NodeML 0.1 did not save the network size.
            NodeConfigError: If the config asks for CUDA and CUDA is not
                available.

        """
        state = params["model_state_dict"]
        if not state:
            self._network = None
            self._in_features = self._out_features = None
            self._target_context_dump = dict(params.get("target_context", {}))
            return
        in_features = params.get("in_features")
        out_features = params.get("out_features")
        if in_features is None or out_features is None:
            msg = (
                f"The params of {type(self).__name__} have no network size "
                "(in_features, out_features). NodeML 0.1 did not save it. "
                "Train the node again."
            )
            raise NodeInputError(msg)
        # The weights are loaded next, so the initial values do not matter.
        # A forked RNG keeps the global RNG unchanged.
        with torch.random.fork_rng(devices=[]):
            network = self._build_network(in_features, out_features)
        network.load_state_dict(state)
        self._network = network.to(self._device()).eval()
        self._in_features = in_features
        self._out_features = out_features
        self._target_context_dump = dict(params["target_context"])
