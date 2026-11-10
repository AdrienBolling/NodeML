# Write a custom node

A custom node is a Python class that derives from the base class of a node type.
You define its config (ports, hyperparameters, running config), implement a few methods, and register it in {py:data}`~nodeml.core.nodes.registry.node_registry.NODE_REGISTRY`.
After registration, a pipeline uses the node by its registry name, as it uses a built-in node.
See {doc}`concepts` for nodes and ports, and {doc}`/examples/custom_node` for a notebook that builds a custom transform.

## Choose a base class

Each node type has a base class and a config base class:

| Node type   | Base class | Config base class | Methods to implement |
|-------------|------------|-------------------|----------------------|
| Data source | {py:class}`~nodeml.core.nodes.data_source.data_source.DataSourceNode` | {py:class}`~nodeml.core.nodes.data_source.data_source.DataSourceConfig` | `setup_source()`, `fetch_data(data)` |
| Transform   | {py:class}`~nodeml.core.nodes.transform.transform.TransformNode` | {py:class}`~nodeml.core.nodes.transform.transform.TransformConfig` | `fit(data)`, `transform(data)`, `get_params()`, `set_params(params)` |
| Model       | {py:class}`~nodeml.core.nodes.models.model.Model` | {py:class}`~nodeml.core.nodes.models.model.ModelConfig` | `fit(data)`, `predict(data)`, `get_params()`, `set_params(params)` |
| Metric      | {py:class}`~nodeml.core.nodes.metrics.metric_node.MetricNode` | {py:class}`~nodeml.core.nodes.metrics.metric_node.MetricNodeConfig` | `update(data)`, `compute()`, and `reset()` when `update()` keeps state |
| Sink        | {py:class}`~nodeml.core.nodes.data_sink.sink.Sink` | {py:class}`~nodeml.core.nodes.data_sink.sink.SinkConfig` | None. Use the built-in `Sink`. |

The runner calls {py:meth}`~nodeml.core.nodes.node.Node.node_fit_transform` in training and {py:meth}`~nodeml.core.nodes.node.Node.node_transform` in the other modes (see {doc}`running`).
Each base class implements `node_fit()` and `node_transform()` with the methods of the table. Do not override them.
The base classes behave as follows:

- **Data source.** `setup_source()` runs once in each runner call that runs the source, in every mode. Read `self.execution_mode` to select the data of the current mode. `fetch_data()` returns the data. A source with the class attribute `accepts_inputs = True` receives the `input_data` entry of the node in `fetch_data(data)`.
- **Transform.** `node_fit()` calls `fit()` and sets `self._fitted = True`. `node_transform()` raises `ValueError` when `self._fitted` is `False`.
- **Model.** `node_fit()` calls `fit()`, and `node_transform()` calls `predict()`. The base class does not check that the model is fitted. Raise {py:exc}`~nodeml.core.common.exceptions.NodeNotFittedError` in `predict()` when the model is not fitted.
- **Metric.** `node_fit()` does nothing. `node_transform()` calls `reset()`, `update()`, `compute()`, then `reset()` again. A metric port without an explicit `mode` is active only in `evaluation`.
- **Sink.** A pipeline has one Sink. Compilation creates its ports from the edges that end at the Sink.

## Define the config

Derive the config class from the config base class of the node type.
Give each field a default value. Then a `None` config in a {py:class}`~nodeml.core.pipeline.pipeline.PipelineConfig` gives the default config.

- `hyperparameters` holds the values that a tuner can change. Transforms and models have this field.
- `running_config` holds the execution values, for example file paths. All node types except the Sink have this field.
- `in_ports` and `out_ports` map port names to {py:class}`~nodeml.core.nodes.node.Port` objects. Edges and data dicts use these names.

The fields of a `Port`:

| Field            | Default | Use |
|------------------|---------|-----|
| `arr_type`       | —       | The array type that the node receives or returns: `pd.DataFrame`, `np.ndarray` or `torch.Tensor`. |
| `data_structure` | `Data`  | Use `DataStructureEnum.TABULAR`, as the built-in nodes do. Compilation logs a warning when two connected ports have different structures. |
| `data_category`  | —       | `numerical_data`, `categorical_data` or `mixed_data`. |
| `data_shape`     | —       | The shape for the input checks, for example `"batch features"`. |
| `optional`       | `False` | When `True`, the port can stay empty. |
| `desc`           | —       | A description of the port. |
| `mode`           | `["all"]` | The execution modes in which the port is active. |

## Write the node class

1. Give the type parameters of the base class: input data, input context, output data and output context. Transforms and models have a fifth parameter, the type of the learned parameters.
2. Write `__init__(self, *, config)`. The `config` argument must be a keyword argument, else NodeML raises `TypeError`. Store it in `self._config`.
3. Set the class attribute `metadata` to an instance of a {py:class}`~nodeml.core.nodes.node.NodeMetadata` subclass. The registry stores the fields of this instance. The built-in nodes add the fields `node_name` and `description`.
4. Implement the methods of the table. Each method that receives data gets a dict of `(array, context)` tuples, keyed by input port name.
5. Return a dict of `(array, context)` tuples, keyed by output port name. Return only the ports that the config declares.

You can also set the class attribute `hyperparameter_space` to a dict of Ray Tune search spaces. The tuner reads it to build its default search space.

## Build output contexts

The runner checks that each output context describes its data (see {doc}`running`).
When a node changes the columns, it must also change the context.
Build the output context by column name:

- `ctx.select(columns)` ({py:meth}`~nodeml.core.common.data.data.TabularDataContext.select`) returns a context with only these columns, in the given order. Use it when you keep or reorder columns. An unknown name raises {py:exc}`~nodeml.core.common.exceptions.DataContextError`.
- `ctx.aligned_to(df)` ({py:meth}`~nodeml.core.common.data.data.TabularDataContext.aligned_to`) returns a context that describes `df` exactly. A known column keeps its category. A new column gets a category from its dtype. Use it when you change values or dtypes, or add columns.
- `TabularDataContext(columns, dtypes, categories)` builds a context for new data, for example predictions. The three lists must have the same length.

## Example: a transform

This transform learns two quantiles of each column during the fit. It clips each column to these quantiles.

```python
import numpy as np
import pandas as pd
from pydantic import Field

from nodeml import NODE_REGISTRY
from nodeml.core.common.data.data import (
    ArrayLikeEnum, DataCategoryEnum, DataStructureEnum, NumericalData, TabularDataContext,
)
from nodeml.core.nodes.node import Port
from nodeml.core.nodes.transform.transform import (
    TransformConfig, TransformHyperParameters, TransformMetadata, TransformNode,
    TransformRunningConfig,
)


def numerical_port(desc: str) -> Port:
    return Port(
        arr_type=ArrayLikeEnum.PANDAS, data_structure=DataStructureEnum.TABULAR,
        data_category=DataCategoryEnum.NUMERICAL, data_shape="batch features", desc=desc,
    )


class ClipperMetadata(TransformMetadata):
    node_name: str = "QuantileClipper"
    description: str = "Clip each column to the quantiles learned during fit."


class ClipperHyperParameters(TransformHyperParameters):
    quantile: float = Field(default=0.05, gt=0.0, lt=0.5)


class ClipperRunningConfig(TransformRunningConfig):
    pass


class ClipperConfig(TransformConfig[ClipperHyperParameters, ClipperRunningConfig]):
    hyperparameters: ClipperHyperParameters = Field(default_factory=ClipperHyperParameters)
    running_config: ClipperRunningConfig = Field(default_factory=ClipperRunningConfig)
    in_ports: dict[str, Port] = Field(default={"input": numerical_port("Data.")})
    out_ports: dict[str, Port] = Field(default={"output": numerical_port("Clipped data.")})


type ClipperParams = dict[str, dict[str, float]]
type TabularInputs = dict[str, tuple[pd.DataFrame, TabularDataContext]]


class QuantileClipper(
    TransformNode[pd.DataFrame, TabularDataContext, pd.DataFrame, TabularDataContext, ClipperParams]
):
    metadata = ClipperMetadata()

    def __init__(self, *, config: ClipperConfig) -> None:
        self._config = config
        self._params: ClipperParams = {"lower": {}, "upper": {}}
        self._fitted = False  # node_transform() refuses to run before a fit.

    def fit(self, data: TabularInputs) -> None:
        df, _ctx = data["input"]
        q = self._config.hyperparameters.quantile
        self._params = {"lower": df.quantile(q).to_dict(), "upper": df.quantile(1 - q).to_dict()}

    def transform(self, data: TabularInputs) -> TabularInputs:
        df, ctx = data["input"]
        lower = pd.Series(self._params["lower"])[df.columns]
        upper = pd.Series(self._params["upper"])[df.columns]
        clipped = df.clip(lower=lower, upper=upper, axis=1)
        return {"output": (clipped, ctx.aligned_to(clipped))}

    def get_params(self) -> ClipperParams:
        return self._params

    def set_params(self, params: ClipperParams) -> None:
        self._params = params
        self._fitted = True  # A node with loaded parameters can transform.


NODE_REGISTRY.register(
    name="QuantileClipper",
    node_class=QuantileClipper,
    node_config_class=ClipperConfig,
    running_config_class=ClipperRunningConfig,
    hyperparameters_class=ClipperHyperParameters,
)
```

Use the node by its registry name:

```python
from nodeml.core.pipeline.pipeline import Edge, Pipeline, PipelineConfig
from nodeml.core.pipeline.runners.smart_runner import SmartRunner, SmartRunnerConfig

rng = np.random.default_rng(0)
df = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
ctx = TabularDataContext(["a", "b"], list(df.dtypes), [NumericalData] * 2)

source_config = NODE_REGISTRY.get_node_config_class("InputsPassthrough")(
    out_ports={"X": numerical_port("Features.")}
)
pipe = Pipeline(config=PipelineConfig(
    nodes={
        "source": ("InputsPassthrough", source_config),
        "clip": ("QuantileClipper", None),  # None gives the default ClipperConfig.
        "sink": ("Sink", None),
    },
    edges=[
        Edge(source="source", target="clip", ports_map=[("X", "input")]),
        Edge(source="clip", target="sink", ports_map=[("output", "clipped")]),
    ],
))
pipe.compile()
runner = SmartRunner(pipe, config=SmartRunnerConfig(verbose=False))
runner.train(input_data={"source": {"X": (df, ctx)}})
clipped, clipped_ctx = runner.infer(input_data={"source": {"X": (df, ctx)}})["clipped"]
print(clipped.max().round(2).to_dict())  # {'a': 1.5, 'b': 1.72}
```

## Mixins

A mixin is a class that adds the same behaviour to more than one node class.
NodeML identifies a mixin by the class attribute `_is_mixin = True`.

1. Put the mixin after the node base class in the list of base classes. `Node.__init_subclass__` raises `TypeError` at class creation when a mixin comes before the node base class.
2. Give the mixin an `__init__` that accepts the keyword argument `config`. NodeML calls it after the `__init__` of the node, with the node config. A mixin can also define `__post_init__(self, *, config)`, which NodeML calls after the mixin `__init__`.
3. To configure a mixin, subclass {py:class}`~nodeml.core.common.mixins.mixin.MixinSettings`. Put an instance in the `mixins_config` dict of the node config, keyed by a mixin name. `MixinSettings` is a `pydantic-settings` class, so its fields can also come from environment variables.

This example continues the example above:

```python
from nodeml.core.common.mixins.mixin import MixinSettings
from nodeml.core.nodes.node import NodeConfig


class CounterSettings(MixinSettings):
    label: str = "calls"


class CallCounterMixin:
    _is_mixin = True

    def __init__(self, *, config: NodeConfig | None = None) -> None:
        settings = config.mixins_config.get("counter") if config else None
        self.counter_label = settings.label if settings else "calls"
        self.calls = 0


class CountedClipper(QuantileClipper, CallCounterMixin):  # The node base class comes first.
    def transform(self, data: TabularInputs) -> TabularInputs:
        self.calls += 1
        return super().transform(data)


node = CountedClipper(
    config=ClipperConfig(mixins_config={"counter": CounterSettings(label="clips")})
)
node.node_fit_transform({"input": (df, ctx)})
print(node.counter_label, node.calls)  # clips 1
```

## Register the node

{py:meth}`~nodeml.core.nodes.registry.node_registry.NodeRegistry.register` takes these arguments:

- `name`: the registry name. A pipeline config refers to the node by this name.
- `node_class` and `node_config_class`: the node class and its config class.
- `running_config_class` and `hyperparameters_class` (optional): the classes of the two config fields. The registry stores them in the entry, for example `NODE_REGISTRY["QuantileClipper"]["running_config"]`.

The registry reads the node type from the default value of the `node_type` field of the config class.
The config base classes set this default.
A name that is already registered raises {py:exc}`~nodeml.core.common.exceptions.RegistryError`.
To register a name again, for example when you run a notebook cell again, remove the old entry first:

```python
if "QuantileClipper" in NODE_REGISTRY:
    NODE_REGISTRY.unregister("QuantileClipper")
```

Register the node before you build a pipeline that uses it.
Also register it before you load a saved pipeline config that uses it: the pipeline finds the config class of each node in the registry.

## Add a node to the NodeML package

The built-in nodes are in the subpackages of `src/nodeml/components/`.
Each subpackage that defines nodes has a `_register.py` module with a `register_nodes()` function:

```python
from nodeml.core.nodes.registry.node_registry import NODE_REGISTRY

from .quantile_clipper import ClipperConfig, QuantileClipper


def register_nodes() -> None:
    NODE_REGISTRY.register(
        name="QuantileClipper", node_class=QuantileClipper, node_config_class=ClipperConfig
    )
```

`import nodeml` runs the auto-discovery.
It walks the packages under `nodeml.components`, imports each `_register` module, and calls its `register_nodes()`.
Each directory must have an `__init__.py` file, else the walk does not find it.
A `_register` module without `register_nodes()` causes a warning in the log, and NodeML skips it.
See {doc}`built_in_nodes` for the nodes that NodeML supplies.
