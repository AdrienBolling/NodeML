"""Numerical imputation transform node for the NodeML Framework.

Fills missing values in numerical columns using one of three strategies:

* ``"mean"``     - replace NaNs with the per-column mean (learned at fit).
* ``"median"``   - replace NaNs with the per-column median (learned at fit).
* ``"constant"`` - replace NaNs with a user-supplied constant value.

The output keeps the dtype of each column when the fill value fits it.
When it does not fit (for example, a mean of ``2.5`` in an ``Int64``
column), the column becomes ``float64``.

Per-column fill values are persisted via :meth:`get_params` /
:meth:`set_params` for checkpointing.
"""

from typing import Literal

import pandas as pd
from pydantic import Field

from nodeml.components.utils.dataframe import check_fitted_columns, fill_missing
from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    TabularDataContext,
)
from nodeml.core.nodes.node import Port
from nodeml.core.nodes.transform.transform import (
    TransformConfig,
    TransformHyperParameters,
    TransformMetadata,
    TransformNode,
    TransformRunningConfig,
)

# Serialisable params: column name -> fill value.
type _NumericalImputationParams = dict[str, float]


class NumericalImputationMetadata(TransformMetadata):
    """Metadata for the NumericalImputation node."""

    node_name: str = "NumericalImputation"
    description: str = (
        "Fill missing values in numerical columns using mean, median, or a constant."
    )
    trainable: bool = True


class NumericalImputationRunningConfig(TransformRunningConfig):
    """No run-time knobs for this node."""


class NumericalImputationHyperParameters(TransformHyperParameters):
    """Tuneable hyperparameters for the NumericalImputation node."""

    strategy: Literal["mean", "median", "constant"] = Field(
        default="mean",
        description=(
            "Imputation strategy. "
            "``'mean'`` and ``'median'`` are learned per column at fit time. "
            "``'constant'`` uses the ``value`` field."
        ),
    )
    value: float = Field(
        default=0.0,
        description="Fill value used when strategy is ``'constant'``.",
    )


class NumericalImputationConfig(
    TransformConfig[
        NumericalImputationHyperParameters,
        NumericalImputationRunningConfig,
    ],
):
    """Full configuration for the NumericalImputation node."""

    hyperparameters: NumericalImputationHyperParameters = Field(
        default_factory=NumericalImputationHyperParameters,
        description="Imputation strategy and constant value.",
    )
    running_config: NumericalImputationRunningConfig = Field(
        default_factory=NumericalImputationRunningConfig,
        description="No run-time knobs for this node.",
    )
    in_ports: dict[str, Port] = Field(
        default={
            "input": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch feature",
                desc="Numerical DataFrame with potential missing values.",
            ),
        },
        description="Input port: 'input' (numerical DataFrame).",
    )
    out_ports: dict[str, Port] = Field(
        default={
            "output": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch feature",
                desc="Numerical DataFrame with missing values filled.",
            ),
        },
        description="Output port: 'output' (imputed numerical DataFrame).",
    )


class NumericalImputation(
    TransformNode[
        pd.DataFrame,
        TabularDataContext,
        pd.DataFrame,
        TabularDataContext,
        _NumericalImputationParams,
    ],
):
    """Fill missing values in numerical columns.

    Example:
        >>> cfg = NumericalImputationConfig(
        ...     hyperparameters=NumericalImputationHyperParameters(strategy="median"),
        ... )
        >>> node = NumericalImputation(config=cfg)
        >>> out = node.node_fit_transform({"input": (df, ctx)})

    """

    metadata = NumericalImputationMetadata()

    def __init__(self, *, config: NumericalImputationConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config
        self._params: _NumericalImputationParams = {}
        self._fitted = False

    # --- TransformNode interface ------------------------------------------

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Learn the fill value of each column of ``data["input"]``.

        Args:
            data: Must contain the key ``"input"``.

        """
        df, _ = data["input"]
        hp = self._config.hyperparameters

        match hp.strategy:
            case "mean":
                self._params = {col: float(df[col].mean()) for col in df.columns}
            case "median":
                self._params = {col: float(df[col].median()) for col in df.columns}
            case "constant":
                self._params = dict.fromkeys(df.columns, hp.value)

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Fill the missing values with the values learned during fit.

        Args:
            data: Must contain the key ``"input"``.

        Returns:
            ``{"output": (df, ctx)}``.  The context describes the output
            dtypes.

        Raises:
            NodeInputError: If the input columns are not the fitted columns.

        """
        df, ctx = data["input"]
        check_fitted_columns(self._params, df.columns, node_name="NumericalImputation")
        result = df.copy()
        for col in df.columns:
            result[col] = fill_missing(
                df[col], self._params[col], fallback_dtype="float64"
            )
        return {"output": (result, ctx.aligned_to(result))}

    def get_params(self) -> _NumericalImputationParams:
        """Return the fill value of each column.

        Returns:
            A mapping of column name to fill value.

        """
        return self._params

    def set_params(self, params: _NumericalImputationParams) -> None:
        """Restore the fill values returned by :meth:`get_params`.

        Args:
            params: The params of a fitted NumericalImputation node.

        """
        self._params = params
        self._fitted = True
