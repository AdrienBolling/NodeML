"""Minimal Node subclasses used as test doubles.

These shims exercise the Node/DataSource/Transform/Metric base classes
without pulling in heavy third-party dependencies (sklearn, torch) so
that the core machinery can be tested in isolation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field

from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    NumericalData,
    TabularDataContext,
)
from nodeml.core.nodes.data_source.data_source import (
    DataSourceConfig,
    DataSourceNode,
    DataSourceRunningConfig,
)
from nodeml.core.nodes.metrics.metric_node import (
    MetricNode,
    MetricNodeConfig,
    MetricNodeRunningConfig,
)
from nodeml.core.nodes.models.model import (
    Model,
    ModelConfig,
    ModelHyperParameters,
    ModelRunningConfig,
)
from nodeml.core.nodes.node import Port
from nodeml.core.nodes.transform.transform import (
    TransformConfig,
    TransformHyperParameters,
    TransformNode,
    TransformRunningConfig,
)

# ---------------------------------------------------------------------------
# DataSource shim
# ---------------------------------------------------------------------------


class _ConstantSourceRunningConfig(DataSourceRunningConfig):
    pass


class ConstantSourceConfig(DataSourceConfig[_ConstantSourceRunningConfig]):
    """Config for a source that returns a fixed DataFrame on the ``output`` port."""

    running_config: _ConstantSourceRunningConfig = Field(
        default_factory=_ConstantSourceRunningConfig,
    )
    in_ports: dict[str, Port] = Field(default={})
    out_ports: dict[str, Port] = Field(
        default={
            "output": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch feature",
                desc="Constant numerical data produced by the shim.",
            ),
        },
    )


class ConstantSource(
    DataSourceNode[None, None, pd.DataFrame, TabularDataContext],
):
    """DataSource shim that returns a user-supplied DataFrame unchanged."""

    def __init__(self, *, config: ConstantSourceConfig) -> None:
        self._config = config
        self._payload: tuple[pd.DataFrame, TabularDataContext] | None = None
        self.setup_calls = 0

    def setup_source(self) -> None:
        self.setup_calls += 1

    def fetch_data(
        self, data: dict[str, tuple[None, None]] | None = None
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        _ = data
        if self._payload is None:
            msg = "ConstantSource was instantiated without a payload."
            raise RuntimeError(msg)
        return {"output": self._payload}

    def set_payload(self, payload: tuple[pd.DataFrame, TabularDataContext]) -> None:
        """Change the DataFrame this source emits on its output port."""
        self._payload = payload


# ---------------------------------------------------------------------------
# Transform shim
# ---------------------------------------------------------------------------


class _IdentityTransformRunningConfig(TransformRunningConfig):
    pass


class _IdentityTransformHyperParameters(TransformHyperParameters):
    pass


type _IdentityParams = dict[str, Any]


class IdentityTransformConfig(
    TransformConfig[
        _IdentityTransformHyperParameters,
        _IdentityTransformRunningConfig,
    ]
):
    """Config for an identity transform that simply passes data through."""

    hyperparameters: _IdentityTransformHyperParameters = Field(
        default_factory=_IdentityTransformHyperParameters,
    )
    running_config: _IdentityTransformRunningConfig = Field(
        default_factory=_IdentityTransformRunningConfig,
    )
    in_ports: dict[str, Port] = Field(
        default={
            "input": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch feature",
                desc="Input DataFrame.",
            ),
        }
    )
    out_ports: dict[str, Port] = Field(
        default={
            "output": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch feature",
                desc="Output DataFrame.",
            ),
        }
    )


class IdentityTransform(
    TransformNode[
        pd.DataFrame,
        TabularDataContext,
        pd.DataFrame,
        TabularDataContext,
        _IdentityParams,
    ]
):
    """Transform shim that records fit/transform calls and returns data unchanged."""

    def __init__(self, *, config: IdentityTransformConfig) -> None:
        self._config = config
        self._fitted = False
        self.fit_calls = 0
        self.transform_calls = 0

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        self.fit_calls += 1

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        self.transform_calls += 1
        return {"output": data["input"]}

    def get_params(self) -> _IdentityParams:
        return {"fitted": self._fitted}

    def set_params(self, params: _IdentityParams) -> None:
        self._fitted = bool(params.get("fitted", False))


# ---------------------------------------------------------------------------
# Model shim
# ---------------------------------------------------------------------------


class _MeanModelHyperParameters(ModelHyperParameters):
    pass


class _MeanModelRunningConfig(ModelRunningConfig):
    pass


type _MeanModelParams = dict[str, Any]


class MeanModelConfig(
    ModelConfig[_MeanModelHyperParameters, _MeanModelRunningConfig],
):
    """A trivial model that predicts the training-time mean of ``y`` for every input."""

    hyperparameters: _MeanModelHyperParameters = Field(
        default_factory=_MeanModelHyperParameters,
    )
    running_config: _MeanModelRunningConfig = Field(
        default_factory=_MeanModelRunningConfig,
    )
    in_ports: dict[str, Port] = Field(
        default={
            "X": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch feature",
                desc="Features.",
            ),
            "y": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch 1",
                desc="Targets (only required at training time).",
                mode=["training", "evaluation"],
            ),
        }
    )
    out_ports: dict[str, Port] = Field(
        default={
            "pred": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch 1",
                desc="Predicted mean target value.",
            ),
        }
    )


class MeanModel(
    Model[
        np.ndarray,
        TabularDataContext,
        np.ndarray,
        TabularDataContext,
        _MeanModelParams,
    ]
):
    """Model shim that predicts the training mean regardless of X."""

    def __init__(self, *, config: MeanModelConfig) -> None:
        self._config = config
        self._mean = 0.0
        self._target_ctx: TabularDataContext | None = None

    def fit(self, data: dict[str, tuple[np.ndarray, TabularDataContext]]) -> None:
        _, _ = data["X"]
        y, y_ctx = data["y"]
        self._mean = float(np.mean(y))
        self._target_ctx = y_ctx

    def predict(
        self, data: dict[str, tuple[np.ndarray, TabularDataContext]]
    ) -> dict[str, tuple[np.ndarray, TabularDataContext]]:
        X, _ = data["X"]
        pred = np.full(shape=(X.shape[0], 1), fill_value=self._mean, dtype=np.float64)
        ctx = self._target_ctx or TabularDataContext(
            columns=["pred"],
            dtypes=[np.dtype("float64")],
            categories=[NumericalData],
        )
        return {"pred": (pred, ctx)}

    def get_params(self) -> _MeanModelParams:
        return {"mean": self._mean}

    def set_params(self, params: _MeanModelParams) -> None:
        self._mean = float(params["mean"])


# ---------------------------------------------------------------------------
# Metric shim
# ---------------------------------------------------------------------------


class _SumCountMetricRunningConfig(MetricNodeRunningConfig):
    pass


class SumCountMetricConfig(MetricNodeConfig[_SumCountMetricRunningConfig]):
    """Metric shim that returns the count of rows fed through update()."""

    running_config: _SumCountMetricRunningConfig = Field(
        default_factory=_SumCountMetricRunningConfig,
    )
    in_ports: dict[str, Port] = Field(
        default={
            "pred": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch 1",
                desc="Predictions.",
            ),
            "target": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch 1",
                desc="Targets.",
            ),
        }
    )
    out_ports: dict[str, Port] = Field(
        default={
            "score": Port(
                arr_type=ArrayLikeEnum.NUMPY,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="1 1",
                desc="Row-count scalar.",
            ),
        }
    )


class SumCountMetric(
    MetricNode[
        np.ndarray,
        TabularDataContext,
        np.ndarray,
        TabularDataContext,
    ]
):
    """Metric shim: counts the total number of rows seen in update()."""

    def __init__(self, *, config: SumCountMetricConfig) -> None:
        self._config = config
        self._count = 0

    def update(self, data: dict[str, tuple[np.ndarray, TabularDataContext]]) -> None:
        pred, _ = data["pred"]
        self._count += int(pred.shape[0])

    def reset(self) -> None:
        self._count = 0

    def compute(self) -> dict[str, tuple[np.ndarray, TabularDataContext]]:
        return {
            "score": (
                np.array([[self._count]], dtype=np.float64),
                TabularDataContext(
                    columns=["count"],
                    dtypes=[np.dtype("float64")],
                    categories=[NumericalData],
                ),
            ),
        }
