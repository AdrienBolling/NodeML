# Pipelines

A pipeline connects nodes with edges into a directed acyclic graph (DAG).
This page shows how to build, check, edit, view, save, and load a pipeline.
For nodes, node configs, and ports, see {doc}`concepts`. For the execution of a pipeline, see {doc}`running`.
The examples on this page continue each other. Run them in sequence.

## Build a pipeline

A {py:class}`~nodeml.core.pipeline.pipeline.PipelineConfig` describes the full pipeline.
Give it to {py:class}`~nodeml.core.pipeline.pipeline.Pipeline`:

```python
from nodeml import NODE_REGISTRY
from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.nodes.node import Port
from nodeml.core.pipeline.pipeline import Edge, Pipeline, PipelineConfig

source_config = NODE_REGISTRY.get_node_config_class("InputsPassthrough")(
    out_ports={
        "X": Port(
            arr_type=ArrayLikeEnum.PANDAS,
            data_structure=DataStructureEnum.TABULAR,
            data_category=DataCategoryEnum.NUMERICAL,
            data_shape="batch feature",
            desc="Feature matrix.",
        ),
        "y": Port(
            arr_type=ArrayLikeEnum.PANDAS,
            data_structure=DataStructureEnum.TABULAR,
            data_category=DataCategoryEnum.NUMERICAL,
            data_shape="batch 1",
            desc="Regression target.",
            mode=[NodeExecutionMode.TRAINING, NodeExecutionMode.EVALUATION],
        ),
    },
)

config = PipelineConfig(
    name="Regression Demo",
    nodes={
        "source": ("InputsPassthrough", source_config),
        "scaler": ("StandardScaler", None),
        "model": ("LinearRegression", None),
        "r2": ("R2Score", None),
        "sink": ("Sink", None),
    },
    edges=[
        Edge(source="source", target="scaler", ports_map=[("X", "input")]),
        Edge(source="scaler", target="model", ports_map=[("output", "X")]),
        Edge(source="source", target="model", ports_map=[("y", "y")]),
        Edge(source="model", target="r2", ports_map=[("pred", "pred")]),
        Edge(source="source", target="r2", ports_map=[("y", "target")]),
        Edge(source="model", target="sink", ports_map=[("pred", "prediction")]),
    ],
)
pipe = Pipeline(config=config)
```

`PipelineConfig` has these fields:

- `nodes`: a mapping from a node name to a `(registry name, node config)` tuple.
  A `None` config gives the default config of the registered config class.
- `edges`: a list of {py:class}`~nodeml.core.pipeline.pipeline.Edge` objects.
- `name` and `version`: the name (default `"My Pipeline"`) and the {py:class}`~nodeml.core.common.version.Version` (default `0.1.0`).
- `settings`: the {py:class}`~nodeml.core.pipeline.pipeline.PipelineSettings`. See [Pipeline settings](#pipeline-settings).

You can also start from an empty pipeline, `Pipeline()`, and add each node and each edge.
For example, `pipe.add_node("scaler", "StandardScaler")` adds a node with the default config.
See [Edit a pipeline](#edit-a-pipeline).

## Edges and ports maps

An `Edge` connects a `source` node to a `target` node.
Its `ports_map` is a list of `(source_port, target_port)` pairs.
Each pair sends the data of an output port of the source to an input port of the target.
A pipeline has at most one edge for each `(source, target)` pair.
When you add a second edge between the same nodes, the pipeline merges the two ports maps.
The merge removes the duplicate pairs and keeps the order.

The pipeline checks the edges when you create the pipeline, change a node or an edge, and compile:

1. The source node and the target node exist.
2. The source node is not the Sink.
3. Each source port is an output port of the source node.
4. Each target port is an input port of the target node.
5. The data structures, the data categories, and the execution modes of the two ports are compatible.
   {doc}`concepts` gives the rules.

Checks 4 and 5 do not apply to an edge to the Sink.
A failed check raises {py:class}`~nodeml.core.common.exceptions.PipelineValidationError`, a subclass of `ValueError`:

```python
from nodeml.core.common.exceptions import PipelineValidationError

try:
    pipe.add_edge(Edge(source="source", target="model", ports_map=[("X", "features")]))
except PipelineValidationError as error:
    print(error)
# Target port 'features' in edge from 'source' to 'model' does not exist in the
# target node's in_ports.
# Available target ports: ['X', 'y']
```

:::{note}
The pipeline does not reject two edges that feed the same input port.
Connect each input port to one output port only.
:::

## The Sink

A pipeline has one Sink node, with the registry name `"Sink"`.
A second Sink raises `PipelineValidationError`. An edge cannot start at the Sink.

Send each output that you need to the Sink with an edge.
In the ports map of this edge, the target port is a name that you choose for the Sink port.
When you compile, each `(source_port, sink_port)` pair gives one input port and one output port on the Sink.
Each Sink port is a copy of the source port, with the same array type, category, shape, and modes.
The Sink ports follow the order of the edges, then the order of each ports map.
Two pairs that use the same Sink port name raise `PipelineValidationError`.

Compile replaces the ports of the Sink config, so do not declare ports on it.
The stored Sink config does not change, thus the pipeline hash does not change.
{py:attr}`~nodeml.core.pipeline.pipeline.Pipeline.resolved_nodes` gives the node configs with the generated Sink ports.

```python
pipe.compile()
sink = pipe.node_objects["sink"]
print(list(sink.in_ports))  # ['prediction']
print(sink.in_ports["prediction"].desc)  # Output 'pred' of node 'model'.
print(pipe.get_node_config("sink").in_ports)  # {}
```

`runner.infer()` returns the Sink outputs, keyed by Sink port name. See {doc}`running`.

## Compile

Compile the pipeline before you run it.
{py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.compile` does these steps:

1. If `settings.autoprune` is `True`, it removes each node that has no edge.
2. It validates the pipeline: a Sink exists, each edge is valid, each Sink port name is unique, and the graph has no cycle.
3. It creates the node objects from the registry. The Sink object gets the generated ports.

When compile succeeds, {py:attr}`~nodeml.core.pipeline.pipeline.Pipeline.compiled` becomes `True`.
{py:attr}`~nodeml.core.pipeline.pipeline.Pipeline.node_objects` then maps each node name to its node object.
A node whose class and config object did not change since the last compile keeps its node object.
Thus it also keeps its fitted state.

| Error | Cause |
|---|---|
| {py:class}`~nodeml.core.common.exceptions.PipelineValidationError` | There is no Sink, an edge is not valid, or a Sink port name is used two times. |
| {py:class}`~nodeml.core.common.exceptions.PipelineGraphError` | The graph has a cycle. This error is a subclass of `PipelineValidationError`. |
| {py:class}`~nodeml.core.common.exceptions.PipelineCompilationError` | The pipeline is already compiled. Edit it before you compile it again. |
| {py:class}`~nodeml.core.common.exceptions.RegistryError` | A registry name is not in the node registry. |

:::{note}
Autoprune removes the nodes from the pipeline config.
Thus {py:attr}`~nodeml.core.pipeline.pipeline.Pipeline.config` and {py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.hash` change after the compile.
:::

## Edit a pipeline

| Method | Action |
|---|---|
| {py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.add_node` | Adds a node from a registry name. Without a config, it uses the default config. |
| {py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.remove_node` | Removes a node and all its edges. |
| {py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.update_node` | Replaces the config of a node. The node type must stay the same. |
| {py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.add_edge` | Adds an edge, or merges its ports map into the existing edge. |
| {py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.remove_edge` | Removes the edge from a source node to a target node. |
| {py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.update_edge` | Replaces the ports map of an existing edge. |

Each method checks the change.
If the change is not valid, the method raises an error and the pipeline does not change.
A successful change marks the pipeline as uncompiled. Compile it again before the next run:

```python
pipe.add_edge(Edge(source="source", target="sink", ports_map=[("y", "target")]))
print(pipe.compiled)  # False
pipe.compile()
print(list(pipe.node_objects["sink"].in_ports))  # ['prediction', 'target']
```

The pipeline keeps the config objects that you give to it. It does not copy them.
A change to a config object does not mark the pipeline as uncompiled.
To change a node config, give a new config object to `update_node`.
The next compile then creates a new node object for this node:

```python
model_config_class = NODE_REGISTRY.get_node_config_class("LinearRegression")
running_config_class = NODE_REGISTRY["LinearRegression"]["running_config"]
pipe.update_node(
    "model",
    model_config_class(running_config=running_config_class(fit_intercept=False)),
)
pipe.compile()
```

To read the pipeline, use `get_node_config`, `get_edge`, `nodes`, `edges`, `sink_node_name`, `get_source_node_names`, and `get_metric_node_names`.
{py:attr}`~nodeml.core.pipeline.pipeline.Pipeline.graph` gives the graph as a `networkx.DiGraph`.

## Pipeline settings

`PipelineSettings` has one field, `autoprune`, with the default `True`.
With `autoprune=False`, compile keeps the nodes that have no edge, and creates their node objects:

```python
from nodeml.core.common.version import Version
from nodeml.core.pipeline.pipeline import PipelineSettings

other = Pipeline(
    config=PipelineConfig(
        name="Regression Demo",
        version=Version(major=1, minor=0, patch=0),
        settings=PipelineSettings(autoprune=False),
    )
)
print(other.file_basename)  # Regression_Demo_v1.0.0
```

## View the graph

The pipeline renders its graph as an interactive Plotly figure:

```python
pipe.render()  # Shows the figure, for example in a notebook.

html = pipe.render_to_html(title="Regression demo")
pipe.save_html_to_dir("saved_pipeline")  # Writes Regression_Demo_v0.1.0_graph.html.
```

{py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.render_to_html` returns an HTML string.
By default, the HTML loads plotly.js from a CDN.
With `include_plotlyjs=True`, the HTML contains plotly.js (about 5 MB) and works offline.
{py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.save_html_to_dir` writes this HTML to a file.

In the figure, the nodes are in columns from left to right.
The source nodes are in the first column, and the metric nodes are in a column to the right of the Sink.
Each other node is in the column of its longest path to the Sink.
The colour and the marker of a node show its node type.
A menu selects the execution mode. In each mode, the figure shows only the edges and the ports that are active in that mode.
The hover text of a node shows its config. The hover text of an edge shows its ports map.

Each node must be a source node, a metric node, or have a path to the Sink.
If not, the render raises an error.

To change the figure before you show it, call {py:func}`~nodeml.core.pipeline.render.render_pipeline_graph_plotly`.
It returns the `plotly.graph_objects.Figure`, for example `render_pipeline_graph_plotly(pipe, execution_mode="inference")`.

## Save and load

All the files of a pipeline start with {py:attr}`~nodeml.core.pipeline.pipeline.Pipeline.file_basename`, `{name}_v{version}`.
In the name, each sequence of characters other than letters, digits, `-`, and `_` becomes one `_`.
Then the `_` characters at the start and at the end are removed.
Each save method creates the directory if necessary.

| Method | File | Content |
|---|---|---|
| {py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.save_config_to_dir` | `{file_basename}_config.json` | The pipeline config, with all the fields of the node configs. |
| {py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.save_params_to_dir` | `{file_basename}_params.pkl` | The output of `get_params`, as a pickle. |
| {py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.save_html_to_dir` | `{file_basename}_graph.html` | The rendered graph. |

{py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.get_params` maps each node name to the learned parameters of the node object.
It includes the node objects that have the `get_params` and `set_params` methods, for example transforms and models.
It reads the node objects, so compile and train the pipeline before you save the parameters.
This example trains the pipeline with a {py:class}`~nodeml.core.pipeline.runners.smart_runner.SmartRunner`, then saves it:

```python
import numpy as np
import pandas as pd

from nodeml.core.common.data.data import TabularDataContext, infer_category
from nodeml.core.pipeline.runners.smart_runner import SmartRunner


def context_of(df):
    return TabularDataContext(
        columns=list(df.columns),
        dtypes=list(df.dtypes),
        categories=[infer_category(dtype) for dtype in df.dtypes],
    )


rng = np.random.default_rng(0)
X = pd.DataFrame({"x1": rng.standard_normal(100), "x2": rng.standard_normal(100)})
y = pd.DataFrame({"target": 3.0 * X["x1"] - 2.0 * X["x2"]})

runner = SmartRunner(pipe)
runner.train(input_data={"source": {"X": (X, context_of(X)), "y": (y, context_of(y))}})
print(list(pipe.get_params()))  # ['scaler', 'model']

pipe.save_config_to_dir("saved_pipeline")
pipe.save_params_to_dir("saved_pipeline")
```

{py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.load_from_dir` rebuilds a saved pipeline:

1. It finds the config file. Without `file_basename`, the directory must contain exactly one `*_config.json` file.
   If it finds no file or more than one file, it raises {py:class}`~nodeml.core.common.exceptions.PipelineError`.
2. It creates the pipeline from the config. The registry gives the config class of each node.
3. It compiles the pipeline.
4. If `load_params=True` (the default) and the params file exists, it loads the parameters.

```python
loaded = Pipeline.load_from_dir("saved_pipeline")
print(loaded.compiled)  # True
print(loaded.hash() == pipe.hash())  # True

inputs = {"source": {"X": (X, context_of(X))}}
before = runner.infer(input_data=inputs)["prediction"][0]
after = SmartRunner(loaded).infer(input_data=inputs)["prediction"][0]
print(before.equals(after))  # True
```

{py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.from_config_file` gives an uncompiled pipeline from a config file.
Compile it, then load the parameters with {py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.load_params_from_dir`.
The registry must contain the registry names of the saved nodes.
For a custom node, import the module that registers it before you load the pipeline.

:::{warning}
Load only params files that you trust. Unpickling can run arbitrary code.
:::

{py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.hash` returns a SHA-256 hex digest of the pipeline config.
The digest includes the node names, the registry names, all the fields of the node configs, the edges, the name, the version, and the settings.
It sorts the nodes, the edges, and the ports maps, so the build order does not change it.
It does not include the learned parameters. Use it to check that two pipelines have the same config.
The version is part of `file_basename`, so two versions of a pipeline can use the same directory.
In this case, give the `file_basename` argument to `load_from_dir`.

For full examples, see {doc}`/examples/pipeline`, {doc}`/examples/inputs_passthrough`, and {doc}`/examples/save_and_load_pipeline`.
