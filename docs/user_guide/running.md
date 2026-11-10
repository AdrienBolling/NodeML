# Run a pipeline

A runner executes a compiled pipeline.
{py:class}`~nodeml.core.pipeline.runners.smart_runner.SmartRunner` is the runner that NodeML supplies.
It has three methods: {py:meth}`~nodeml.core.pipeline.runners.smart_runner.SmartRunner.train`, {py:meth}`~nodeml.core.pipeline.runners.smart_runner.SmartRunner.infer` and {py:meth}`~nodeml.core.pipeline.runners.smart_runner.SmartRunner.evaluate`.
This page explains what the runner does in each call, which checks it does, and which errors you can see.
See {doc}`pipelines` to build and compile a pipeline.

## Create a runner

Compile the pipeline, then give it to the runner:

```python
pipe.compile()
runner = SmartRunner(pipe, config=SmartRunnerConfig(verbose=False))
```

- The runner raises `ValueError` if the pipeline is not compiled.
- {py:class}`~nodeml.core.pipeline.runners.smart_runner.SmartRunnerConfig` has one field, `verbose`. When `verbose` is `True` (the default), each call shows a `tqdm` progress bar with one step for each node that runs.
- Every method that edits the pipeline marks it as uncompiled. The runner checks the compiled state only when you create it. Thus, compile the pipeline again after each edit, before the next runner call.

## A complete example

This pipeline scales two features, fits a linear regression, and scores it with R². An {py:class}`~nodeml.components.nodes.data_sources.inputs_passthrough.InputsPassthrough` source receives the data from the caller.

```python
import numpy as np
import pandas as pd

from nodeml import NODE_REGISTRY
from nodeml.core.common.data.data import (
    ArrayLikeEnum, DataCategoryEnum, DataStructureEnum, NumericalData, TabularDataContext,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.nodes.node import Port
from nodeml.core.pipeline.pipeline import Edge, Pipeline, PipelineConfig
from nodeml.core.pipeline.runners.smart_runner import SmartRunner, SmartRunnerConfig


def numerical_pair(df: pd.DataFrame) -> tuple[pd.DataFrame, TabularDataContext]:
    """Return the DataFrame and a context that describes it."""
    categories = [NumericalData] * df.shape[1]
    return df, TabularDataContext(list(df.columns), list(df.dtypes), categories)


def port(shape: str, modes: list[NodeExecutionMode] | None = None) -> Port:
    return Port(
        arr_type=ArrayLikeEnum.PANDAS, data_structure=DataStructureEnum.TABULAR,
        data_category=DataCategoryEnum.NUMERICAL, data_shape=shape, desc="Numerical data.",
        mode=modes or [NodeExecutionMode.ALL],
    )


rng = np.random.default_rng(0)
X = pd.DataFrame({"x1": rng.normal(size=100), "x2": rng.normal(size=100)})
y = pd.DataFrame({"target": 3 * X["x1"] - 2 * X["x2"]})

source_config = NODE_REGISTRY.get_node_config_class("InputsPassthrough")(
    out_ports={
        "X": port("batch features"),
        # The targets are not necessary for inference.
        "y": port("batch targets", [NodeExecutionMode.TRAINING, NodeExecutionMode.EVALUATION]),
    }
)
pipe = Pipeline(config=PipelineConfig(
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
        Edge(source="model", target="sink", ports_map=[("pred", "pred")]),
    ],
))
pipe.compile()
runner = SmartRunner(pipe, config=SmartRunnerConfig(verbose=False))

runner.train(input_data={"source": {"X": numerical_pair(X), "y": numerical_pair(y)}})
predictions = runner.infer(input_data={"source": {"X": numerical_pair(X)}})
metrics = runner.evaluate(
    input_data={"source": {"X": numerical_pair(X), "y": numerical_pair(y)}}
)

pred_df, pred_ctx = predictions["pred"]
print(pred_df.shape)                                       # (100, 1)
print({name: float(df.iloc[0, 0]) for name, (df, _) in metrics.items()})  # {'r2': 1.0}
```

## Execution modes

Each runner method sets one execution mode:

| Runner method | Mode         | Start nodes      | Node method that the runner calls |
|---------------|--------------|------------------|-----------------------------------|
| `train()`     | `training`   | the Sink         | `node_fit_transform()`            |
| `infer()`     | `inference`  | the Sink         | `node_transform()`                |
| `evaluate()`  | `evaluation` | all metric nodes | `node_transform()`                |

Each port has a `mode` list. A port is active when the list contains the current mode or `all`.
Before a node runs, the runner sets the mode on the node.
A node reads it in `self.execution_mode`.

## Graph walk and pruning

The runner walks the graph backwards from the start nodes:

1. For a node, find the incoming edges that feed at least one active input port.
2. Run the source node of each of these edges first (recursively).
3. Run the node itself.

Thus a branch that feeds only inactive ports does not run.
For example, the `y` port of `LinearRegression` is active only in `training` and `evaluation`.
If the targets come from a separate source node, that node does not run during inference.
`train()` and `infer()` do not run the metric nodes: the walk starts at the Sink, and metric ports are active only in `evaluation` by default.

The runner stores the outputs of each node for the current call.
Thus a node that feeds more than one node runs only once.
Each call to `train()`, `infer()` or `evaluate()` starts with no stored outputs.
`evaluate()` on a pipeline without metric nodes runs no node and returns an empty dict.

## Data between nodes

Between nodes, data travels as {py:class}`~nodeml.core.common.data.data.TabularData`.
For each input, the runner converts the data to the array type of the receiving port (`pd.DataFrame`, `np.ndarray` or `torch.Tensor`).
The node receives a dict of `(array, context)` tuples, keyed by input port name.

A {py:class}`~nodeml.core.common.data.data.TabularDataContext` lists the columns, the dtypes and the data categories, aligned by position.
The runner checks each node output, so that the context cannot drift from the data:

- The output must be an `(array, context)` tuple with a `TabularDataContext`. If not, the runner raises {py:exc}`~nodeml.core.common.exceptions.NodeError`.
- The context must name the same columns as a DataFrame, in the same order. For a NumPy array or a tensor, it must have one entry for each column. If not, the runner raises {py:exc}`~nodeml.core.common.exceptions.DataContextError`, with the node name and the port name.
- For a DataFrame, the runner uses the dtypes of the DataFrame, not the dtypes of the context. For a NumPy array or a tensor, the runner casts the columns to the dtypes of the context.
- A node must return only the output ports that it declares. If not, the runner raises `NodeError`.

Build the output contexts with {py:meth}`~nodeml.core.common.data.data.TabularDataContext.select` or {py:meth}`~nodeml.core.common.data.data.TabularDataContext.aligned_to`. See {doc}`custom_nodes`.

## Input checks

Before a node runs, the runner checks each active input against the category and the shape of its port:

- The category comes from the context. A `numerical_data` port accepts only numerical columns. A `categorical_data` port accepts only categorical columns. A `mixed_data` port accepts all columns.
- The shape comes from the `data_shape` string, for example `"batch features"`. A name is a dimension, a number is a fixed size, and `_` is a dimension of any size.
- The checks of one node share the dimension names. For example, `X` (`"batch features"`) and `y` (`"batch targets"`) of a model must have the same number of rows.

A mismatch raises {py:exc}`~nodeml.core.common.exceptions.DataTypeError`.
A required input port that is active in the current mode and has no input raises {py:exc}`~nodeml.core.common.exceptions.NodeInputError`.
This occurs, for example, when no edge feeds the port.
An optional port (`optional=True`) can stay empty.

## External inputs

`train()`, `infer()` and `evaluate()` accept `input_data`, keyed by `{node_name: {port_name: (array, context)}}`.
Only source nodes with the class attribute `accepts_inputs = True` accept external inputs.
{py:class}`~nodeml.components.nodes.data_sources.inputs_passthrough.InputsPassthrough` is such a node.

1. At the start of the call, the runner checks the keys. A key that names no node, or a node that does not accept inputs, raises `NodeInputError`.
2. When the source node runs, the runner converts each value to `TabularData`. The checks of node outputs apply, so a bad value raises `NodeError` or `DataContextError`.
3. The source node receives each value as a `pd.DataFrame` with its context.

`InputsPassthrough` returns the values on the output ports with the same names.
It raises `ValueError` for a port name that it does not declare.
It also raises `ValueError` when a required output port that is active in the current mode has no value.
See {doc}`/examples/inputs_passthrough`.

## Data sources

A data source runs `setup_source()` once in each runner call that runs the source, in every mode. Thus:

- A pipeline with loaded parameters can run `infer()` without `train()` first.
- A source can read other data in each mode. For example, {py:class}`~nodeml.components.nodes.data_sources.tabular_csv_fetcher.TabularCSVFetcher` reads `inference_csv_path` during inference when you set it, and `csv_path` when you do not. `evaluation_csv_path` works the same way during evaluation.

## Results

- `infer()` returns the Sink outputs as `{sink_port: (DataFrame, context)}`, in Sink port order.
- `evaluate()` returns the metric outputs as `(DataFrame, context)` tuples. A metric node with one output port is keyed by the node name. A metric node with more ports is keyed by `"node_name.port_name"` for each port.
- `train()` returns `None`. The fitted state stays in the node objects. Save it with `runner.save_params_to_dir(path)`, which calls {py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.save_params_to_dir`.

If a node raises an error, the runner logs the error and raises it again.

## Train and validation splits

{py:class}`~nodeml.core.pipeline.validation.train_val_split.TabularTrainValSplit` splits an `input_data` dict into two dicts with the same structure.
Give one dict to `train()` and the other to `evaluate()`.
This example continues the complete example:

```python
from nodeml.core.pipeline.validation.train_val_split import (
    TabularTrainValSplit, TabularTrainValSplitConfig,
)

splitter = TabularTrainValSplit(config=TabularTrainValSplitConfig(
    data_nodes=[("source", ["X"])],
    target_nodes=[("source", ["y"])],
    val_proportion=0.2,
    randomize=True,
    seed=0,
))
train_data, val_data = splitter.split(
    {"source": {"X": numerical_pair(X), "y": numerical_pair(y)}}
)
runner.train(input_data=train_data)
val_metrics = runner.evaluate(input_data=val_data)
```

The splitter computes one set of row indices and applies it to all referenced ports.
Thus the rows of `X` and `y` stay aligned.
All referenced ports must have the same number of rows, else `split()` raises `ValueError`.
The fields of {py:class}`~nodeml.core.pipeline.validation.train_val_split.TabularTrainValSplitConfig`:

| Field            | Default | Use |
|------------------|---------|-----|
| `data_nodes`     | —       | `(node_name, [port_name, ...])` tuples to split. |
| `target_nodes`   | `None`  | Tuples to split with the same rows. `best_kldiv` does not use them. |
| `val_proportion` | `0.2`   | Fraction of the rows for validation, between 0 and 1. |
| `randomize`      | `False` | Shuffle the rows. When `False`, the last rows are the validation rows. |
| `best_kldiv`     | `False` | Try `kldiv_trials` random splits. Keep the split with the lowest KL divergence between the validation and training distributions of `data_nodes`. This option always shuffles. |
| `kldiv_trials`   | `100`   | Number of random splits for `best_kldiv`. |
| `seed`           | `None`  | Random seed. |

A split value is a `(DataFrame, context)` tuple with a new index that starts at 0.
The splitter copies the entries that the config does not reference into both dicts.

## Runner wrappers

A wrapper adds behaviour to a runner, and does not change the runner.
{py:class}`~nodeml.core.pipeline.runners.wrappers.wrapper.PipelineRunnerWrapper` takes a runner and a config.
Each method of the wrapper calls the same method of the wrapped runner.
Override the methods that you need, and call `super()` in them:

```python
import time
from collections.abc import Mapping

from pydantic import BaseModel

from nodeml.core.pipeline.runners.wrappers.wrapper import PipelineRunnerWrapper


class TimerConfig(BaseModel):
    label: str = "train"


class TimerWrapper(PipelineRunnerWrapper[object, object, TimerConfig]):
    """Print the duration of each training."""

    def train(self, input_data: Mapping | None = None) -> None:
        start = time.perf_counter()
        super().train(input_data=input_data)
        print(f"{self.config.label}: {time.perf_counter() - start:.3f} s")


timed = TimerWrapper(runner, config=TimerConfig())
timed.train(input_data=train_data)
metrics = timed.evaluate(input_data=val_data)
assert timed.unwrapped is runner
```

A wrapper is also a runner, so you can wrap a wrapper.
{py:attr}`~nodeml.core.pipeline.runners.wrappers.wrapper.PipelineRunnerWrapper.unwrapped` returns the runner at the bottom of the stack.
To log runs to MLflow, see {doc}`tracking`.

## Write another runner

{py:class}`~nodeml.core.pipeline.runners.pipeline_runner.PipelineRunner` is the abstract base class of all runners.
A subclass implements `train()`, `infer()` and `evaluate()`, with the `input_data` argument.
The base class supplies `pipeline`, `config`, `mode`, `node_objects`, `get_params()`, `set_params()`, `save_params_to_dir()`, `load_params_from_dir()` and `get_metric_node_names()`.
The config is a {py:class}`~nodeml.core.pipeline.runners.pipeline_runner.RunnerConfig` or a subclass of it.

## Errors

All NodeML errors derive from {py:exc}`~nodeml.core.common.exceptions.NodeMLError`.
Most also derive from a built-in exception, as the table shows.
For example, an `except ValueError` clause also catches `NodeInputError`.

| Exception | Also a | When you see it |
|---|---|---|
| {py:exc}`~nodeml.core.common.exceptions.RegistryError` | `ValueError` | A node name is registered twice, or a name is not in the registry. |
| {py:exc}`~nodeml.core.common.exceptions.PipelineError` | — | `Pipeline.load_from_dir()` finds no config file, or more than one. |
| {py:exc}`~nodeml.core.common.exceptions.PipelineValidationError` | `ValueError` | An edit or `compile()` finds an invalid layout: no Sink, two Sinks, an unknown node or port, incompatible ports. |
| {py:exc}`~nodeml.core.common.exceptions.PipelineGraphError` | `ValueError` | `compile()` finds a cycle. |
| {py:exc}`~nodeml.core.common.exceptions.PipelineCompilationError` | `ValueError` | You call `compile()` on a compiled pipeline. |
| {py:exc}`~nodeml.core.common.exceptions.NodeError` | — | A node returns a port that it does not declare. A node output or an `input_data` value is not an `(array, TabularDataContext)` tuple. |
| {py:exc}`~nodeml.core.common.exceptions.NodeConfigError` | `ValueError` | A node config has an invalid value. |
| {py:exc}`~nodeml.core.common.exceptions.NodeInputError` | `ValueError` | An `input_data` key is not valid, a required input is missing, or a built-in node receives columns that it cannot use. |
| {py:exc}`~nodeml.core.common.exceptions.NodeNotFittedError` | `ValueError` | A built-in model predicts before it is fitted. |
| {py:exc}`~nodeml.core.common.exceptions.DataTypeError` | `TypeError` | An input does not match the category or the shape of its port. |
| {py:exc}`~nodeml.core.common.exceptions.DataContextError` | `ValueError` | A context does not match its data. |

{py:exc}`~nodeml.core.common.exceptions.PipelineExecutionError` (also a `RuntimeError`) exists, but the current code does not raise it.

Some errors are built-in exceptions only:

- `SmartRunner()` with an uncompiled pipeline raises `ValueError`.
- A transform that is not fitted raises `ValueError` in `node_transform()`.
- `InputsPassthrough` raises `ValueError` for an unknown or a missing port.
- `TabularCSVFetcher` raises `FileNotFoundError` when a file does not exist.
