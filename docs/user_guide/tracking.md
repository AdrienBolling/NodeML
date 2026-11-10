# Experiment tracking

{py:class}`~nodeml.components.pipe_runner_wrappers.mlflow_logger.MLFlowLoggerWrapper` logs the training and the evaluation of a pipeline to [MLflow](https://mlflow.org/docs/latest/).
It wraps a runner and logs the pipeline config, the graph, the trained parameters and the metrics.
The notebook {doc}`/examples/mlflow_logging` shows the same steps with a larger pipeline.

## Wrap a runner

`MLFlowLoggerWrapper` takes a runner and has the same `train`, `evaluate` and `infer` methods.
It also forwards `get_params`, `set_params`, `save_params_to_dir` and `load_params_from_dir` to the runner.
It does not inherit from {py:class}`~nodeml.core.pipeline.runners.wrappers.wrapper.PipelineRunnerWrapper` (see {doc}`running`).
It can wrap any {py:class}`~nodeml.core.pipeline.runners.pipeline_runner.PipelineRunner`, also a `PipelineRunnerWrapper`.
A `PipelineRunnerWrapper` cannot wrap an `MLFlowLoggerWrapper`, because `MLFlowLoggerWrapper` has no `pipeline` attribute.
Its `runner` property gives the wrapped runner.

The examples on this page use the pipeline of {doc}`tuning`: an `InputsPassthrough` source named `source`, a `RandomForestRegressor`, and the metric nodes `mse` and `r2`.

:::{dropdown} Code of the example pipeline
```{include} ../_includes/example_pipeline.md
```
:::

```python
from nodeml.components.pipe_runner_wrappers.mlflow_logger import (
    MLFlowLoggerWrapper,
    MLFlowLoggerWrapperConfig,
)
from nodeml.core.pipeline.runners.smart_runner import SmartRunner

config = MLFlowLoggerWrapperConfig(
    tracking_uri="sqlite:///mlflow.db",
    experiment_name="nodeml_demo",
    run_name="forest_baseline",
    log_train_time=True,
    tags={"dataset": "synthetic"},
)
mlflow_runner = MLFlowLoggerWrapper(SmartRunner(pipe), config=config)

mlflow_runner.train(input_data=input_data)
mlflow_runner.evaluate(input_data, prefix="val")
mlflow_runner.evaluate(input_data, prefix="test")
```

## What the wrapper logs

| Item | When | Name in MLflow | Config field |
|---|---|---|---|
| Tags `pipeline_name`, `pipeline_version`, `pipeline_nodes`, and the user tags | When a run starts | Run tags | `tags` |
| The full {py:class}`~nodeml.core.pipeline.pipeline.PipelineConfig` as JSON | `train`, before the training | `pipeline/{basename}_config.json` | `log_pipeline_config` |
| An interactive HTML render of the graph | `train`, before the training | `pipeline/{basename}_graph.html` | `log_pipeline_graph` |
| The trained parameters of all nodes, as a pickle file | `train`, after a successful training | `pipeline/{basename}_params.pkl` | `log_pipeline_params` |
| The training time in seconds | `train`, after a successful training | `train_duration_s` | `log_train_time` |
| The metrics and the evaluation time in seconds | Each `evaluate` call | The metric keys and `evaluate_duration_s` | Always |

`{basename}` is the `file_basename` of the pipeline, for example `demo_v0.1.0`.
The artifact directory is always {py:data}`~nodeml.components.pipe_runner_wrappers.mlflow_logger.PIPELINE_ARTIFACT_DIR` (`pipeline/`).
MLflow adds its own `mlflow.*` tags, for example the user and the source file.
`infer` logs nothing.

The metric keys follow the rule of the tuner (see the metric names in {doc}`tuning`): `mse`, `scores.r2`, `mse.score_0`.
With `prefix="val"`, all the keys of the call get the prefix `val/`, also `evaluate_duration_s`.
The wrapper does not log a metric that it cannot convert to a float.
Instead, it logs a warning.

## Run lifecycle

1. `train` creates a new run, logs the items in the table above, trains the pipeline and ends the run with the status `FINISHED`.
2. If the training raises an error, the run ends with the status `FAILED` (`KILLED` for `KeyboardInterrupt`). The wrapper raises the error again.
3. `evaluate` logs into the run of the last successful `train`. It does not change the status of that run.
4. If there is no such run, `evaluate` creates a new run and ends it.
5. Each `evaluate` call in a run gets the next MLflow `step`: 0, 1, 2, and so on.

In the example above, `val/mse` has the step 0 and `test/mse` has the step 1.
If the wrapper cannot log an artifact or a metric, it logs a warning.
The result of the training or of the evaluation does not change.

## Configuration

{py:class}`~nodeml.components.pipe_runner_wrappers.mlflow_logger.MLFlowLoggerWrapperConfig` is a pydantic-settings class.
It reads environment variables with the prefix `MLFLOW_`.

| Field | Default | Environment variable |
|---|---|---|
| `tracking_uri` | `None` | `MLFLOW_TRACKING_URI` |
| `experiment_name` | `"nodeml"` | `MLFLOW_EXPERIMENT_NAME` |
| `run_name` | `None` | `MLFLOW_RUN_NAME` |
| `nested` | `False` | `MLFLOW_NESTED` |
| `log_pipeline_params` | `True` | `MLFLOW_LOG_PIPELINE_PARAMS` |
| `log_pipeline_config` | `True` | `MLFLOW_LOG_PIPELINE_CONFIG` |
| `log_pipeline_graph` | `True` | `MLFLOW_LOG_PIPELINE_GRAPH` |
| `log_train_time` | `False` | `MLFLOW_LOG_TRAIN_TIME` |
| `tags` | `{}` | `MLFLOW_TAGS`, as a JSON object |

A value that you give to the constructor has priority over an environment variable.
An environment variable has priority over a value in an env file.
To read an env file, give its path: `MLFlowLoggerWrapperConfig(env_file=".env")`.
The env file can hold other `MLFLOW_*` variables, for example `MLFLOW_TRACKING_USERNAME`.
The config ignores them and does not export them.
An unknown keyword argument raises `TypeError`.
When you create the wrapper, it logs the value and the source of each field (see {doc}`logging`).

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export MLFLOW_EXPERIMENT_NAME="production_pipelines"
```

With these variables, `MLFlowLoggerWrapper(runner)` needs no config argument.

## Tracking URI

The wrapper has its own `MlflowClient`, with the URI of `tracking_uri`.
It does not change the global MLflow state: the tracking URI, the active experiment and the active run stay the same.
When `tracking_uri` is `None`, the client uses `mlflow.get_tracking_uri()`, so a URI that you set with `mlflow.set_tracking_uri` applies.
`mlflow_runner.client.tracking_uri` gives the URI that the wrapper uses.

The wrapper creates the experiment if it does not exist.
If the experiment exists but is deleted, the wrapper raises `ValueError`.
MLflow selects the location of the artifacts.
With a local SQLite URI, MLflow writes them to `mlruns/` in the working directory.

## Read a run and load the pipeline

`mlflow_runner.client` is an `MlflowClient`, and `mlflow_runner.run_id` is the ID of the last run.
{py:func}`~nodeml.components.pipe_runner_wrappers.mlflow_logger.load_pipeline_from_run` downloads the `pipeline/` artifacts of a run and returns the compiled pipeline with its trained parameters.

```python
from nodeml.components.pipe_runner_wrappers.mlflow_logger import load_pipeline_from_run

run = mlflow_runner.client.get_run(mlflow_runner.run_id)
print(run.info.status, run.data.metrics)

pipeline = load_pipeline_from_run(
    mlflow_runner.run_id, tracking_uri=mlflow_runner.client.tracking_uri
)
predictions = SmartRunner(pipeline).infer(input_data={"source": {"X": pair(X)}})
```

:::{warning}
The parameters are a pickle file. Load only the runs that you trust.
:::

To see the runs in the MLflow UI, start it with the same tracking URI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Nested runs

Set `nested=True` to make each new run a child of the active MLflow run.
If `mlflow.active_run()` returns a run, the wrapper adds the tag `mlflow.parentRunId` with the ID of that run.
Start the parent run with `mlflow.start_run()` on the same tracking server as the wrapper.
