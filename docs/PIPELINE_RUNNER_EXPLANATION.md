# How the SmartRunner executes a pipeline

This document describes how `SmartRunner` runs a compiled `Pipeline`. The code is in `src/nodeml/core/pipeline/runners/smart_runner.py`.

## Pipeline and compilation

A pipeline is a directed acyclic graph (DAG):

- A **node** is a processing unit: a data source, a transform, a model, a metric, or the Sink.
- An **edge** connects output ports of one node to input ports of another node. Its `ports_map` lists `(source_port, target_port)` pairs.
- Each **port** declares an array type (`pd.DataFrame`, `np.ndarray` or `torch.Tensor`), a data category (numerical, categorical or mixed), a shape (for example `"batch features"`), and the execution modes in which it is active.

Call `Pipeline.compile()` before you create a runner. Compilation does these steps:

1. Remove the nodes that have no edge (when `settings.autoprune` is `True`).
2. Validate the graph: one Sink, valid edges, compatible ports, no cycle.
3. Create the Sink ports from the edges that end at the Sink, in edge order. Each Sink port is a copy of its source port.
4. Instantiate the node objects. After an edit, a node whose class and config did not change keeps its node object and its fitted state.

Every method that edits the pipeline marks it as uncompiled. Compile it again before the next run.

## Execution modes

| Runner method | Mode         | Start nodes      | Node method called      |
|---------------|--------------|------------------|-------------------------|
| `train()`     | `training`   | the Sink         | `node_fit_transform()`  |
| `infer()`     | `inference`  | the Sink         | `node_transform()`      |
| `evaluate()`  | `evaluation` | all metric nodes | `node_transform()`      |

Each port has a `mode` list. A port is active when the list contains the current mode or `all`.

## Graph walk

The runner walks the graph backwards from the start nodes:

1. For a node, find the incoming edges that feed at least one active input port.
2. Run the source node of each of these edges first (recursively).
3. Run the node itself.

Thus a branch that feeds only inactive ports does not run. For example, the target loader does not run during inference, because the `y` port of a model is active only in `training` and `evaluation`.

The runner stores the outputs of each node for the current call. A node runs at most once per call. Each call to `train()`, `infer()` or `evaluate()` starts with an empty store.

## Data between nodes

Between nodes, data travels as `TabularData`. For each input, the runner converts the data to the array type of the receiving port, and passes an `(array, TabularDataContext)` tuple to the node.

The `TabularDataContext` lists the columns, the dtypes and the data categories, aligned by position. The runner makes sure that the context does not drift from the data:

- An output context must name the same columns as the data, in the same order. If not, the runner raises `DataContextError` and names the node and the port.
- For a DataFrame, the dtypes come from the DataFrame, so stale context dtypes do not propagate.
- A node must return only the output ports that it declares. If not, the runner raises `NodeError`.

Nodes build their output contexts with `TabularDataContext.select()` or `TabularDataContext.aligned_to()`, which work by column name.

## Input checks

Before a node runs, the runner checks every active input against the category and the shape of its port. The checks of one node share the dimension names of the `data_shape` strings. For example, `X` (`"batch features"`) and `y` (`"batch targets"`) of a model must have the same number of rows. A mismatch raises `DataTypeError`.

A required input that is missing in the current mode raises `NodeInputError`. Optional ports can stay empty.

## External inputs

`train()`, `infer()` and `evaluate()` accept `input_data`, keyed by `{node_name: {port_name: (array, context)}}`. Only source nodes with `accepts_inputs = True`, for example `InputsPassthrough`, accept external inputs. A key that names another node, or no node, raises `NodeInputError`.

`TabularTrainValSplit.split()` returns two dicts in this format, for `train()` and `evaluate()`.

## Data sources

A data source runs `setup_source()` once for each runner call, in every mode. Thus:

- A reloaded pipeline can run `infer()` without `train()` first.
- A source can read other data in each mode. For example, `TabularCSVFetcher` reads `inference_csv_path` during inference when you set it, and `csv_path` otherwise.

## Results

- `infer()` returns the Sink outputs as `{sink_port: (DataFrame, context)}`, in Sink port order.
- `evaluate()` returns the metric outputs. A metric node with one output port is keyed by the node name. A metric node with more ports uses `"node_name.port_name"`.
- `train()` returns nothing. The fitted state stays in the node objects. Save it with `Pipeline.save_params_to_dir()`.
