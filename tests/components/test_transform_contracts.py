"""Contract tests that every registered transform node must pass.

* The registry knows the running config and hyperparameter classes.
* ``TransformConfig`` gets its type arguments in the order ``[H, R]``.
* Each output port is declared, and each output context describes its
  DataFrame (the checks of the SmartRunner).
* Save -> load: a new node with ``set_params(get_params())`` and no fit
  gives the same output as the fitted node.
* A full pipeline gives the same inference output after
  ``Pipeline.load_from_dir``.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nodeml.components.nodes.data_sources.inputs_passthrough import (
    InputsPassthroughConfig,
)
from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    CategoricalData,
    DataCategoryEnum,
    DataStructureEnum,
    NumericalData,
    TabularData,
    TabularDataContext,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.nodes.data_sink.sink import SinkConfig
from nodeml.core.nodes.node import NodeConfig, Port
from nodeml.core.nodes.registry.node_registry import NODE_REGISTRY
from nodeml.core.nodes.transform.transform import (
    TransformConfig,
    TransformHyperParameters,
    TransformNode,
    TransformRunningConfig,
)
from nodeml.core.pipeline.pipeline import Edge, Pipeline, PipelineConfig
from nodeml.core.pipeline.runners.smart_runner import SmartRunner

type Inputs = dict[str, tuple[pd.DataFrame, TabularDataContext]]


def _ctx(df: pd.DataFrame) -> TabularDataContext:
    """Return a context with the dtype-inferred categories of *df*."""
    return TabularDataContext(columns=[], dtypes=[], categories=[]).aligned_to(df)


def _numerical() -> tuple[pd.DataFrame, TabularDataContext]:
    rng = np.random.default_rng(0)
    base = rng.standard_normal(12)
    df = pd.DataFrame(
        {
            "n0": base,
            "n1": rng.standard_normal(12),
            "n_const": np.full(12, 0.5),
            "n_copy": base * 2.0 + 1.0,
            "n_int": np.arange(12),
        }
    )
    df.loc[3, "n1"] = np.nan
    df.loc[11, "n0"] = 40.0  # An outlier.
    return df, _ctx(df)


def _categorical() -> tuple[pd.DataFrame, TabularDataContext]:
    df = pd.DataFrame(
        {
            "c_str": ["a", "b", None, "a", "c", "b"] * 2,
            "c_code": pd.Series([1.0, 2.0, 2.0, np.nan, 10.0, 1.0] * 2),
        }
    )
    ctx = TabularDataContext(
        columns=list(df.columns),
        dtypes=list(df.dtypes),
        categories=[CategoricalData, CategoricalData],
    )
    return df, ctx


def _mixed() -> tuple[pd.DataFrame, TabularDataContext]:
    num, num_ctx = _numerical()
    cat, cat_ctx = _categorical()
    df = pd.concat([cat, num], axis=1).assign(mostly_missing=[np.nan] * 10 + [1.0, 2.0])
    ctx = TabularDataContext(
        columns=list(df.columns),
        dtypes=list(df.dtypes),
        categories=[*cat_ctx.categories, *num_ctx.categories, NumericalData],
    )
    return df, ctx


def _target() -> tuple[pd.DataFrame, TabularDataContext]:
    df = pd.DataFrame({"y": np.linspace(0.0, 1.0, 12)}, index=range(100, 112))
    return df, _ctx(df)


INPUTS: dict[str, Callable[[], Inputs]] = {
    "IQROutlierFilter": lambda: {
        "input": _numerical(),
        "target": _target(),
        "sliced": _categorical(),
    },
    "ZScoreOutlierFilter": lambda: {
        "input": _numerical(),
        "target": _target(),
        "sliced": _categorical(),
    },
    "CorrelationFilter": lambda: {"input": _numerical()},
    "MissingRateFilter": lambda: {"input": _mixed()},
    "VarianceFilter": lambda: {"input": _numerical()},
    "DataCategoryFilter": lambda: {"input": _mixed()},
    "LabelEncoding": lambda: {"input": _categorical()},
    "OneHotEncoding": lambda: {"input": _categorical()},
    "CategoricalImputation": lambda: {"input": _categorical()},
    "NumericalImputation": lambda: {"input": _numerical()},
    "MinMaxScaler": lambda: {"input": _numerical()},
    "RobustScaler": lambda: {"input": _numerical()},
    "StandardScaler": lambda: {"input": _numerical()},
    "ColumnOrder": lambda: {"input": _mixed()},
    "FeatureConcatenate": lambda: {"input_1": _numerical(), "input_2": _categorical()},
    "RowConcatenate": lambda: {"input_1": _numerical(), "input_2": _numerical()},
}

TRANSFORMS = sorted(
    name
    for name in NODE_REGISTRY.keys()
    if NODE_REGISTRY.get(name)["node_class"].__module__.startswith(
        "nodeml.components.nodes.transforms"
    )
)


def _make(name: str, config: NodeConfig | None = None) -> TransformNode:
    config_class = NODE_REGISTRY.get_node_config_class(name)
    return NODE_REGISTRY.get_node_class(name)(config=config or config_class())


def _active(node: TransformNode, inputs: Inputs, mode: NodeExecutionMode) -> Inputs:
    """Keep the inputs whose port is active in *mode*."""
    return {
        port: value
        for port, value in inputs.items()
        if mode in node.in_ports[port].mode
        or NodeExecutionMode.ALL in node.in_ports[port].mode
    }


def _check_outputs(node: TransformNode, outputs: Inputs) -> None:
    """Apply the output checks of the SmartRunner, and check the dtypes."""
    undeclared = set(outputs) - set(node.out_ports)
    assert not undeclared, f"undeclared output ports {undeclared}"
    for df, ctx in outputs.values():
        TabularData(
            data=df, columns=ctx.columns, dtypes=ctx.dtypes, categories=ctx.categories
        )
        assert ctx.dtypes == list(df.dtypes)


def test_every_transform_has_inputs() -> None:
    assert sorted(INPUTS) == TRANSFORMS


@pytest.mark.parametrize("name", TRANSFORMS)
class TestTransformContract:
    def test_registry_has_the_config_classes(self, name) -> None:
        entry = NODE_REGISTRY.get(name)
        fields = entry["node_config_class"].model_fields
        assert entry["running_config"] is fields["running_config"].annotation
        assert entry["hyperparameters"] is fields["hyperparameters"].annotation

    def test_transform_config_type_arguments_are_hyper_then_running(self, name) -> None:
        config_class = NODE_REGISTRY.get_node_config_class(name)
        generic_bases = [
            base
            for base in config_class.__mro__
            if getattr(base, "__pydantic_generic_metadata__", {}).get("origin")
            is TransformConfig
        ]
        assert generic_bases
        hyper, running = generic_bases[0].__pydantic_generic_metadata__["args"]
        assert issubclass(hyper, TransformHyperParameters)
        assert issubclass(running, TransformRunningConfig)

    def test_new_node_with_loaded_params_gives_the_same_output(self, name) -> None:
        inputs = INPUTS[name]()
        fitted = _make(name)
        fitted.set_execution_mode(NodeExecutionMode.TRAINING)
        _check_outputs(fitted, fitted.node_fit_transform(inputs))

        # Save -> load: JSON config, pickled params, and no fit.
        config = type(fitted.config).model_validate_json(
            fitted.config.model_dump_json()
        )
        reborn = _make(name, config)
        reborn.set_params(pickle.loads(pickle.dumps(fitted.get_params())))  # noqa: S301

        for mode in (NodeExecutionMode.INFERENCE, NodeExecutionMode.EVALUATION):
            fitted.set_execution_mode(mode)
            reborn.set_execution_mode(mode)
            active = _active(fitted, inputs, mode)
            expected = fitted.node_transform(active)
            actual = reborn.node_transform(active)
            _check_outputs(reborn, actual)
            assert set(actual) == set(expected)
            for port, (df, ctx) in actual.items():
                pd.testing.assert_frame_equal(df, expected[port][0])
                assert ctx == expected[port][1]


# --- Pipeline save -> load -> infer ------------------------------------------


def _port(category: DataCategoryEnum, shape: str, **kwargs: object) -> Port:
    return Port(
        arr_type=ArrayLikeEnum.PANDAS,
        data_structure=DataStructureEnum.TABULAR,
        data_category=category,
        data_shape=shape,
        desc="test port",
        **kwargs,
    )


def _build_pipeline() -> Pipeline:
    """Build source -> preprocessing -> sink with most transforms."""
    source = InputsPassthroughConfig(
        out_ports={
            "X": _port(DataCategoryEnum.MIXED, "batch feature"),
            "y": _port(
                DataCategoryEnum.NUMERICAL,
                "batch 1",
                mode=[NodeExecutionMode.TRAINING, NodeExecutionMode.EVALUATION],
            ),
        }
    )

    def default(name: str) -> tuple[str, NodeConfig]:
        return name, NODE_REGISTRY.get_node_config_class(name)()

    nodes: dict[str, tuple[str, NodeConfig]] = {
        "source": ("InputsPassthrough", source),
        "missing": default("MissingRateFilter"),
        "split": default("DataCategoryFilter"),
        "iqr": default("IQROutlierFilter"),
        "num_impute": default("NumericalImputation"),
        "variance": default("VarianceFilter"),
        "correlation": default("CorrelationFilter"),
        "scaler": default("StandardScaler"),
        "cat_impute": default("CategoricalImputation"),
        "one_hot": default("OneHotEncoding"),
        "concat": default("FeatureConcatenate"),
        "order": default("ColumnOrder"),
        "sink": ("Sink", SinkConfig()),
    }
    edges = [
        Edge(source="source", target="missing", ports_map=[("X", "input")]),
        Edge(source="missing", target="split", ports_map=[("output", "input")]),
        Edge(
            source="split",
            target="iqr",
            ports_map=[("numerical", "input"), ("categorical", "sliced")],
        ),
        Edge(source="source", target="iqr", ports_map=[("y", "target")]),
        Edge(source="iqr", target="num_impute", ports_map=[("output", "input")]),
        Edge(source="num_impute", target="variance", ports_map=[("output", "input")]),
        Edge(source="variance", target="correlation", ports_map=[("output", "input")]),
        Edge(source="correlation", target="scaler", ports_map=[("output", "input")]),
        Edge(source="iqr", target="cat_impute", ports_map=[("sliced", "input")]),
        Edge(source="cat_impute", target="one_hot", ports_map=[("output", "input")]),
        Edge(source="scaler", target="concat", ports_map=[("output", "input_1")]),
        Edge(source="one_hot", target="concat", ports_map=[("output", "input_2")]),
        Edge(source="concat", target="order", ports_map=[("output", "input")]),
        Edge(source="order", target="sink", ports_map=[("output", "features")]),
    ]
    pipeline = Pipeline(
        config=PipelineConfig(nodes=nodes, edges=edges, name="transforms")
    )
    pipeline.compile()
    return pipeline


def test_pipeline_infers_the_same_after_load_from_dir(tmp_path: Path) -> None:
    X, X_ctx = _mixed()
    y = pd.DataFrame({"y": np.linspace(0.0, 1.0, len(X))})
    pipeline = _build_pipeline()
    runner = SmartRunner(pipeline)
    runner.train(input_data={"source": {"X": (X, X_ctx), "y": (y, _ctx(y))}})
    expected_df, expected_ctx = runner.infer(input_data={"source": {"X": (X, X_ctx)}})[
        "features"
    ]
    # The outlier filter removes rows only in training.
    assert len(expected_df) == len(X)
    assert "n_const" not in expected_df.columns
    assert "n_copy" not in expected_df.columns
    assert "mostly_missing" not in expected_df.columns
    assert not expected_df.isna().to_numpy().any()

    pipeline.save_config_to_dir(tmp_path)
    pipeline.save_params_to_dir(tmp_path)
    loaded = Pipeline.load_from_dir(tmp_path)
    actual_df, actual_ctx = SmartRunner(loaded).infer(
        input_data={"source": {"X": (X, X_ctx)}}
    )["features"]
    pd.testing.assert_frame_equal(actual_df, expected_df)
    assert actual_ctx == expected_ctx
