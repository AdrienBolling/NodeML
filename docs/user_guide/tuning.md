# Hyperparameter tuning

{py:class}`~nodeml.core.pipeline.tuners.ray_tuner.RayPipelineTuner` searches the hyperparameters of a pipeline with [Ray Tune](https://docs.ray.io/en/latest/tune/index.html).
Each trial builds a new pipeline, trains it, evaluates it and reports the metrics and one objective to Ray Tune.
This page uses a small in-memory pipeline.
The notebook {doc}`/examples/hyperparameter_tuning` tunes a larger pipeline that reads CSV files.

## How a trial works

When you create the tuner, it copies the config of the pipeline.
Later changes to the pipeline object do not change the trials.
Each trial does these steps:

1. Ray Tune samples one value for each key of the search space.
2. The trial applies the sampled values to a copy of the pipeline config. Pydantic validates each value.
3. The trial compiles a new pipeline and creates a {py:class}`~nodeml.core.pipeline.runners.smart_runner.SmartRunner`.
4. The trial calls `train` and then `evaluate` on the runner, with the same `input_data`.
5. The trial converts the metrics to float values and calculates the objective.
6. The trial reports the metrics, the objective `optimization_metric` and a checkpoint to Ray Tune.

## Declare a search space in a node

A node is tunable when its class has a `hyperparameter_space` attribute that is a `dict`.
Each key is a field of the hyperparameters model of the node.
Each value is a Ray Tune search-space object, for example `tune.choice` or `tune.uniform`.
The `MissingRateFilter` node declares this space:

```python
hyperparameter_space = {"threshold": tune.uniform(0.0, 1.0)}
```

These built-in nodes have a search space:

| Node | Hyperparameters in the search space |
|---|---|
| `RandomForestRegressor`, `RandomForestClassifier` | `n_estimators`, `max_depth` |
| `GradientBoostingRegressor`, `GradientBoostingClassifier` | `n_estimators`, `max_depth`, `learning_rate` |
| `MLP` | `learning_rate`, `hidden_layers`, `hidden_units`, `dropout`, `activation` |
| `CNN` | `learning_rate`, `num_conv_layers`, `num_filters`, `kernel_size`, `pooling`, `pool_size`, `num_fc_layers`, `fc_units`, `dropout`, `activation` |
| `MissingRateFilter`, `VarianceFilter` | `threshold` |
| `CorrelationFilter` | `threshold`, `method` |
| `ZScoreOutlierFilter` | `zscore_cutoff`, `threshold`, `strategy` |
| `IQROutlierFilter` | `iqr_multiplier`, `threshold`, `strategy` |

To add a search space to your own node, see {doc}`custom_nodes`.

## Run the tuner

The examples on this page use the pipeline below.
It has an `InputsPassthrough` source, a `RandomForestRegressor` node named `forest`, and two metric nodes, `mse` and `r2`.

:::{dropdown} Code of the example pipeline
```{include} ../_includes/example_pipeline.md
```
:::

Do these steps to run a search:

1. Create the tuner with the compiled pipeline.
2. Get the default search space with {py:meth}`~nodeml.core.pipeline.tuners.ray_tuner.RayPipelineTuner.default_hyperparameter_space`. Its keys have the form `"node_name/hp_name"`.
3. Change, add or remove keys. A key that is not in the search space keeps the value of the pipeline config. A plain value, not a search-space object, gives the same value to all trials.
4. Call {py:meth}`~nodeml.core.pipeline.tuners.ray_tuner.RayPipelineTuner.tune`. Give the direction of the objective in `TuneConfig(mode=...)`.

```python
import ray
from ray import tune
from ray.tune import RunConfig, TuneConfig

from nodeml.core.pipeline.tuners.ray_tuner import (
    RayPipelineTuner,
    RayPipelineTunerConfig,
)

tuner = RayPipelineTuner(pipe, config=RayPipelineTunerConfig())

param_space = tuner.default_hyperparameter_space()
param_space["forest/n_estimators"] = tune.choice([10, 20])
param_space["forest/max_depth"] = tune.choice([2, 4, None])

ray.init(num_cpus=2)
tuner.tune(
    param_space,
    optimization_metric="mse",
    input_data=input_data,
    tune_config=TuneConfig(num_samples=3, mode="min"),
    run_config=RunConfig(name="forest_search"),
)
```

The tuner gives `input_data` to Ray with `tune.with_parameters`.
Ray puts the data in its object store one time, and all trials use it.
A pipeline that reads its data in a source node, for example `TabularCSVFetcher`, needs no `input_data`.
`RunConfig` goes to Ray Tune without changes. Ray Tune writes the results and the checkpoints in its storage path, `~/ray_results` by default.

## Metric names

The trial converts the output of `evaluate` to float values.
These names are the keys of the trial metrics, and the values that `optimization_metric` accepts:

| Metric output | Key |
|---|---|
| Metric node `mse` with one output port and one value | `mse` |
| Metric node `scores` with more than one output port, port `r2` | `scores.r2` |
| Metric node `mse` with one row of values in the columns `score_0` and `score_1` | `mse.score_0`, `mse.score_1` |

The column names come from the DataFrame columns.
For a numpy array or a torch tensor, they come from the `TabularDataContext` when its length matches, else from the column index (`0`, `1`, ...).
A metric with more than one row, or with a value that is not a number, raises a `ValueError` in the trial.
{py:func}`~nodeml.core.pipeline.tuners.metrics.metrics_to_scalars` applies this rule.

Each trial also reports the objective with the key {py:data}`~nodeml.core.pipeline.tuners.ray_tuner.OPTIMIZATION_METRIC` (`"optimization_metric"`).
Ray Tune ranks the trials by this key.
If you set `TuneConfig(metric=...)`, the tuner ignores it and logs a warning.

## Read the results

{py:meth}`~nodeml.core.pipeline.tuners.ray_tuner.RayPipelineTuner.get_best` returns the best Ray Tune `Result`.
By default, it ranks by `optimization_metric` with the `mode` of the last `tune` call.
To rank by another metric key, give the metric and the mode.
{py:attr}`~nodeml.core.pipeline.tuners.ray_tuner.RayPipelineTuner.results` is the Ray Tune `ResultGrid` of the last run.

```python
best = tuner.get_best()
print(best.config)  # For example: {'forest/n_estimators': 10, 'forest/max_depth': None}
print(best.metrics["mse"], best.metrics["r2"])

best_r2 = tuner.get_best(metric="r2", mode="max")
print(best_r2.config)

table = tuner.results.get_dataframe()
print(table[["config/forest/n_estimators", "config/forest/max_depth", "mse"]])
```

The checkpoint of each trial holds the config and the trained parameters of the trial pipeline.
{py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.load_from_dir` rebuilds the trained pipeline:

```python
from nodeml.core.pipeline.runners.smart_runner import SmartRunner

with best.checkpoint.as_directory() as checkpoint_dir:
    best_pipe = Pipeline.load_from_dir(checkpoint_dir)
predictions = SmartRunner(best_pipe).infer(input_data={"source": {"X": pair(X)}})
```

## Use a custom objective

Give `metric_aggregator` to calculate the objective from more than one metric.
It receives the dict of the metric keys and their float values, and returns one float.
When you give `metric_aggregator`, the tuner does not use `optimization_metric`.

```python
def objective(metrics: dict[str, float]) -> float:
    """Return a value to minimize: the MSE plus a penalty for a low R2."""
    return metrics["mse"] + (1.0 - metrics["r2"])


tuner.tune(
    param_space,
    metric_aggregator=objective,
    input_data=input_data,
    tune_config=TuneConfig(num_samples=3, mode="min"),
    run_config=RunConfig(name="forest_search_custom"),
)
print(tuner.get_best().metrics["optimization_metric"])
ray.shutdown()
```

## Errors

| Cause | Result |
|---|---|
| No `tune_config`, or `mode` is not `"min"` or `"max"` | `tune` raises `ValueError` before Ray Tune starts. |
| No `optimization_metric` and no `metric_aggregator` | `tune` raises `ValueError`. |
| A key is not `"node_name/hp_name"`, names an unknown node or hyperparameter, or a value is not valid | The trial raises {py:exc}`~nodeml.core.common.exceptions.NodeConfigError`. |
| `optimization_metric` is not a metric key | The trial raises `KeyError` with the list of the available keys. |
| `get_best` before `tune`, or `get_best(metric=...)` without `mode` | `get_best` raises `ValueError`. |

When a trial raises an error, Ray Tune marks the trial as failed.
At the end of `tune`, the tuner logs the number of trials and the number of failed trials (see {doc}`logging`).

## Ray settings

- When you start Python with `uv run`, Ray tries to start its workers with `uv run` too. Set `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` to make the workers use the Python environment of the driver.
- The workers can run in a different working directory. Use absolute paths in the configs of source nodes, for example the CSV paths of `TabularCSVFetcher`.
- `RayPipelineTunerConfig.runner_config` is the {py:class}`~nodeml.core.pipeline.runners.smart_runner.SmartRunnerConfig` of the runner in each trial. The tuner sets `verbose=False`, so the trials show no progress bar.
