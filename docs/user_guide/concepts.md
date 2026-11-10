# Core concepts

NodeML builds a machine learning pipeline as a directed acyclic graph (DAG) of nodes.
This page describes the parts of the graph: nodes, node configs, ports, the data between nodes, and the node registry.
{doc}`pipelines` shows how to connect nodes into a pipeline, and {doc}`running` shows how a runner executes it.

## Overview

Node
: A processing unit, for example a data source, a scaler, a model, or a metric.
  A node receives data on its input ports and gives data on its output ports.

Node config
: A pydantic model that holds the ports and the parameters of one node.
  The pipeline creates each node object from its node config.

Port
: A named input or output of a node.
  A port declares the array type, the data category, the shape, and the execution modes of its data.

Edge
: A connection from the output ports of one node to the input ports of another node. See {doc}`pipelines`.

Node registry
: The catalog of the node classes.
  It gives the node class and the config class for a registry name, for example `"StandardScaler"`.

## Nodes and node types

All node classes are subclasses of {py:class}`~nodeml.core.nodes.node.Node`.
The base class has two abstract methods:

- {py:meth}`~nodeml.core.nodes.node.Node.node_fit` learns the state of the node from the data.
- {py:meth}`~nodeml.core.nodes.node.Node.node_transform` computes the outputs from the data.

{py:meth}`~nodeml.core.nodes.node.Node.node_fit_transform` calls the two methods in sequence.
Each method receives a dict that maps an input port name to an `(array, context)` tuple.
`node_transform` returns a dict that maps an output port name to an `(array, context)` tuple.

The `node_type` field of the node config is a {py:class}`~nodeml.core.nodes.node.NodeType` member.
Each node type has a base class and a config class.
The base class implements `node_fit` and `node_transform`, and calls the methods of the last column.

| `NodeType` | Base class | Config class | Methods of a subclass |
|---|---|---|---|
| `SOURCE` | {py:class}`~nodeml.core.nodes.data_source.data_source.DataSourceNode` | {py:class}`~nodeml.core.nodes.data_source.data_source.DataSourceConfig` | `setup_source`, `fetch_data` |
| `TRANSFORM` | {py:class}`~nodeml.core.nodes.transform.transform.TransformNode` | {py:class}`~nodeml.core.nodes.transform.transform.TransformConfig` | `fit`, `transform`, `get_params`, `set_params` |
| `MODEL` | {py:class}`~nodeml.core.nodes.models.model.Model` | {py:class}`~nodeml.core.nodes.models.model.ModelConfig` | `fit`, `predict`, `get_params`, `set_params` |
| `METRIC` | {py:class}`~nodeml.core.nodes.metrics.metric_node.MetricNode` | {py:class}`~nodeml.core.nodes.metrics.metric_node.MetricNodeConfig` | `update`, `compute`, `reset` (optional) |
| `SINK` | {py:class}`~nodeml.core.nodes.data_sink.sink.Sink` | {py:class}`~nodeml.core.nodes.data_sink.sink.SinkConfig` | None. Use the built-in `Sink`. |
| `BASE` | {py:class}`~nodeml.core.nodes.node.Node` | {py:class}`~nodeml.core.nodes.node.NodeConfig` | `node_fit`, `node_transform` |

When you compile a pipeline, it creates each node object with `node_class(config=node_config)`.
A positional config raises `TypeError`.
Before the runner calls a node, it sets the `execution_mode` of the node object.
For the built-in nodes, see {doc}`built_in_nodes`. To write a node class, see {doc}`custom_nodes`.

## Node configs

A node config is an instance of {py:class}`~nodeml.core.nodes.node.NodeConfig`, or of a subclass.
`NodeConfig` is a pydantic model with these fields:

- `node_type`: the {py:class}`~nodeml.core.nodes.node.NodeType` of the node.
- `in_ports` and `out_ports`: dicts that map a port name to a {py:class}`~nodeml.core.nodes.node.Port`.
- `mixins_config`: the settings of the mixins of the node. See [Mixins](#mixins).

The config classes of the node types add one or two fields:

- `hyperparameters` holds the parameters that define the node, for example a threshold.
  Transform and model configs have this field.
  {py:class}`~nodeml.core.pipeline.tuners.ray_tuner.RayPipelineTuner` searches these parameters.
- `running_config` holds the parameters that control how the node runs, for example a file path.
  Source, transform, model, and metric configs have this field.

Each built-in node has its own config class, with default values for all fields.
Get the config class from the registry, create a config, and change the fields that you need:

```python
from nodeml import NODE_REGISTRY

csv_config = NODE_REGISTRY.get_node_config_class("TabularCSVFetcher")()
csv_config.running_config.csv_path = "data/fake_batch.csv"
csv_config.running_config.context_path = "data/fake_batch_context.json"

filter_config = NODE_REGISTRY.get_node_config_class("MissingRateFilter")()
filter_config.hyperparameters.threshold = 0.0

print(csv_config.node_type)  # source
print(list(csv_config.out_ports))  # ['output']
```

Each config object gets a UUID when you create it. Thus two configs with equal values make two different nodes.
The UUID is a private attribute, so a saved config and the pipeline hash do not include it.

Each node class also has a `metadata` class attribute, a {py:class}`~nodeml.core.nodes.node.NodeMetadata` model.
The metadata describes the node class, not one node object.
{py:class}`~nodeml.core.nodes.transform.transform.TransformMetadata` and {py:class}`~nodeml.core.nodes.models.model.ModelMetadata` add the `trainable` field.
All the built-in nodes except `Sink` also add `node_name` and `description`.

## Ports

A {py:class}`~nodeml.core.nodes.node.Port` is the data contract of one input or output of a node:

```python
from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.nodes.node import Port

target_port = Port(
    arr_type=ArrayLikeEnum.PANDAS,
    data_structure=DataStructureEnum.TABULAR,
    data_category=DataCategoryEnum.NUMERICAL,
    data_shape="batch 1",
    desc="Regression target, not needed for inference.",
    mode=[NodeExecutionMode.TRAINING, NodeExecutionMode.EVALUATION],
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `arr_type` | {py:class}`~nodeml.core.common.data.data.ArrayLikeEnum` | required | `PANDAS` (`pd.DataFrame`), `NUMPY` (`np.ndarray`), or `TORCH` (`torch.Tensor`). |
| `data_structure` | {py:class}`~nodeml.core.common.data.data.DataStructureEnum` | `DATA` | `DATA` or `TABULAR`. |
| `data_category` | {py:class}`~nodeml.core.common.data.data.DataCategoryEnum` | required | `NUMERICAL`, `CATEGORICAL`, or `MIXED`. |
| `data_shape` | `str` | required | The shape of the data, as a jaxtyping shape string. |
| `optional` | `bool` | `False` | If `True`, the port can stay without data. |
| `desc` | `str` | required | A description of the port. |
| `mode` | list of {py:class}`~nodeml.core.common.enums.NodeExecutionMode` | `["all"]` | The execution modes in which the port is active. |

The pipeline compares the two ports of each edge. {doc}`pipelines` lists all the edge checks.

**Array type.**
For an input port, `arr_type` is the array type that the node receives.
The runner converts the data to this type before it calls the node.
Thus `StandardScaler`, which gives a `pd.DataFrame`, can feed `LinearRegression`, which receives a `np.ndarray`.

**Data structure.**
`TABULAR` is a subclass of `DATA`.
An edge between a `DATA` port and a `TABULAR` port is valid, but the pipeline logs a warning.
The built-in nodes use `TABULAR`.

**Data category.**
`MIXED` includes the numerical, the categorical, and the mixed data.
An edge is valid when the category of one port includes the category of the other port:

| Source port | Target port | Result |
|---|---|---|
| `NUMERICAL` | `NUMERICAL` or `MIXED` | Valid. |
| `MIXED` | `NUMERICAL` | Valid. The pipeline logs a warning. |
| `NUMERICAL` | `CATEGORICAL` | `PipelineValidationError`. |

At run time, the runner checks the categories of the data against the input port.
Thus a `NUMERICAL` port rejects data that has a categorical column.

**Data shape.**
`data_shape` is a [jaxtyping shape string](https://docs.kidger.site/jaxtyping/api/array/#shape).
Tabular data has two dimensions: the rows (the batch) and the columns (the features).
Separate the dimension names with a space, for example `"batch feature"`.
A number sets a fixed size: `"batch 1"` accepts one column only. An underscore accepts any size.
In one node, a dimension name has the same size on all the input ports.
For example, the `X` port (`"batch features"`) and the `y` port (`"batch targets"`) of `LinearRegression` must have the same number of rows.

**Execution modes.**
{py:class}`~nodeml.core.common.enums.NodeExecutionMode` has the members `TRAINING`, `INFERENCE`, `EVALUATION`, `ALL`, and `DEFAULT`.
A port is active in the modes of its `mode` list. The default `["all"]` makes it active in all modes.
Do not use `DEFAULT` in a port: it is the mode of a node object before a runner sets a mode.
`Port` validates `mode` also when you assign a value, so an unknown value raises `pydantic.ValidationError`.
A metric config changes each port without an explicit `mode` to `["evaluation"]`.
The two ports of an edge must share a mode, or one of them must contain `ALL`.
{doc}`running` explains how the runner uses the modes.

**Optional ports.**
The runner does not raise an error when an optional input port receives no data.
Thus the node must accept a missing key in its input dict.
For an `InputsPassthrough` source, you do not have to give data for an optional output port.

## Data between nodes

Nodes exchange data as `(array, context)` tuples.
The array is a `pd.DataFrame`, a `np.ndarray`, or a `torch.Tensor`. On an input, it has the `arr_type` of the input port.
The context is a {py:class}`~nodeml.core.common.data.data.TabularDataContext`, which describes the columns of the array.

A context has three lists, aligned by position: `columns`, `dtypes`, and `categories`.
Entry `i` of each list describes column `i` of the array.
A category is one of the classes {py:class}`~nodeml.core.common.data.data.NumericalData`, {py:class}`~nodeml.core.common.data.data.CategoricalData`, and {py:class}`~nodeml.core.common.data.data.MixedData`.
{py:func}`~nodeml.core.common.data.data.infer_category` gives the category of a dtype.
`DATA_CATEGORY_MAPPING` in {py:mod}`nodeml.core.common.data.data` gives the class of a `DataCategoryEnum` member.
If the three lists have different lengths, the context raises {py:class}`~nodeml.core.common.exceptions.DataContextError`.

A node that changes the columns must return a context with the same columns, in the same order.
If not, the runner raises `DataContextError`.
These methods build a context by column name, so the context cannot drift from the data:

- {py:meth}`~nodeml.core.common.data.data.TabularDataContext.select` keeps the given columns, in the given order.
- {py:meth}`~nodeml.core.common.data.data.TabularDataContext.aligned_to` describes a DataFrame.
  It keeps the known categories and infers the categories of the new columns.
- {py:meth}`~nodeml.core.common.data.data.TabularDataContext.check_columns` raises `DataContextError` when the names or their order differ.

This example builds a context and calls a node object directly, for example to test the node:

```python
import pandas as pd

from nodeml import NODE_REGISTRY
from nodeml.core.common.data.data import TabularDataContext, infer_category

df = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [10.0, 20.0, 30.0]})
context = TabularDataContext(
    columns=list(df.columns),
    dtypes=list(df.dtypes),
    categories=[infer_category(dtype) for dtype in df.dtypes],
)
print(context.dump_dict["categories"])  # ['numerical_data', 'numerical_data']
print(context.select(["x2"]).columns)  # ['x2']

scaler_class = NODE_REGISTRY.get_node_class("StandardScaler")
scaler = scaler_class(config=NODE_REGISTRY.get_node_config_class("StandardScaler")())

outputs = scaler.node_fit_transform({"input": (df, context)})
scaled_df, scaled_context = outputs["output"]
print(scaled_df["x1"].tolist())  # [-1.0, 0.0, 1.0]
print(scaled_context.columns)  # ['x1', 'x2']
```

`dump_dict` gives the JSON form of a context, and {py:func}`~nodeml.core.common.data.data.tabular_context_from_dict_dump` reads it back.
`TabularCSVFetcher` reads its context file in this form.

Between two nodes, the runner keeps the data as {py:class}`~nodeml.core.common.data.data.TabularData`.
`TabularData` holds a DataFrame together with its columns, dtypes, and categories.
Its methods `to_pandas`, `to_numpy`, and `to_tensor` give an `(array, context)` tuple.
The runner creates the `TabularData` objects, so you do not need to create them.

## The node registry

{py:data}`~nodeml.core.nodes.registry.node_registry.NODE_REGISTRY` is the global {py:class}`~nodeml.core.nodes.registry.node_registry.NodeRegistry`.
`import nodeml` registers all the built-in nodes in it.
The registry maps each registry name to an entry, a dict with these keys:

- `node_class` and `node_config_class`: the node class and its config class.
- `node_type`: the node type, as a string, for example `"transform"`.
- `running_config` and `hyperparameters`: the classes of the two config fields, or `None`.
- `metadata`: the fields of the `metadata` of the node class, as a dict.

```python
from nodeml import NODE_REGISTRY

print(NODE_REGISTRY.list(columns=["name", "node_type", "node_class"]))

entry = NODE_REGISTRY["LinearRegression"]
print(entry["node_type"])  # model
print(entry["node_class"].__name__)  # LinearRegressionNode
print(entry["metadata"]["trainable"])  # True
```

{py:meth}`~nodeml.core.common.registry_display.RegistryDisplayMixin.list` shows the entries.
In a Jupyter notebook, it shows an interactive table with a search field, a sort control, and a source code viewer.
Outside a notebook, or without `ipywidgets`, it returns a plain-text table.
This table has many columns, so use the `columns` argument to select some.

{py:meth}`~nodeml.core.nodes.registry.node_registry.NodeRegistry.get_node_config_class` and {py:meth}`~nodeml.core.nodes.registry.node_registry.NodeRegistry.get_node_class` give the two classes of an entry.
`NODE_REGISTRY[name]` and {py:meth}`~nodeml.core.common.registry.Registry.get` give the full entry.
`in`, `len`, `keys()`, and `items()` work as for a dict.
An unknown name raises {py:class}`~nodeml.core.common.exceptions.RegistryError`.
A second registration with the same name also raises `RegistryError`.

The registry name can be different from the class name.
For example, the registry name `"LinearRegression"` gives the class `LinearRegressionNode`.
A pipeline config uses the registry name. To register your own node class, see {doc}`custom_nodes`.

## Mixins

A mixin adds behaviour to a node class.
Mark the mixin class with the class attribute `_is_mixin = True`, and put it after the node base class in the bases.
For example, write `class CountedScaler(StandardScaler, CallCounterMixin)`.
If the mixin comes first, the class definition raises `TypeError`.
After the `__init__` of the node, NodeML calls the `__init__` of the mixin with the `config` keyword argument.
Then it calls the `__post_init__` of the mixin, if the mixin has one.

The `mixins_config` field of a node config maps a mixin name to a {py:class}`~nodeml.core.common.mixins.mixin.MixinSettings` object.
The built-in nodes do not use mixins.

:::{note}
The pipeline config does not serialize the fields of a `MixinSettings` subclass.
Thus `save_config_to_dir` does not save them, and the pipeline hash ignores them.
:::
