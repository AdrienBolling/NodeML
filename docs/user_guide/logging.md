# Logging

NodeML writes structured log records with the standard Python {py:mod}`logging` module.
Each record has an event, a level, a logger name and a `context` dict with fields such as `pipeline_name` or `node_name`.
The `nodeml` logger has only a {py:class}`logging.NullHandler`, so the library shows no logs until you configure logging.
Call {py:func}`~nodeml.core.common.logging.logger.configure` to show the logs.

## Show the logs

```python
import sys

from nodeml.core.common.logging import configure

configure(level="INFO", stream=sys.stdout, fmt="text")
```

`configure` adds a handler to the `nodeml` logger, sets the format and sets the level.
With no arguments, it writes JSON lines to `sys.stderr` at the level `INFO`.

| Argument | Default | Use |
|---|---|---|
| `level` | `logging.INFO` | A level number or a level name, for example `"DEBUG"`. |
| `filepath` | `None` | A file for the logs. The handler appends to the file. |
| `stream` | `None` | A stream for the logs, when `filepath` is `None`. `None` means `sys.stderr`. |
| `fmt` | `"json"` | `"json"` for {py:class}`~nodeml.core.common.logging.logger.JSONFormatter`, `"text"` for {py:class}`~nodeml.core.common.logging.logger.TextFormatter`. |
| `ensure_ascii` | `False` | For JSON only: escape the characters that are not ASCII. |
| `show_timestamp` | `True` | For text only: start each line with the time. |
| `use_color` | `True` | For text only: show the level in color with ANSI codes. Set it to `False` for a file. |
| `logger_name` | `"nodeml"` | The logger to configure. |

## Change the level

`configure` sets the level on the logger, not on the handler.
Each call sets the level for all the handlers of the logger.
To change only the level, use the standard {py:mod}`logging` API:

```python
import logging

logging.getLogger("nodeml").setLevel(logging.WARNING)
logging.getLogger("nodeml.runner.smart").setLevel(logging.DEBUG)
```

At `DEBUG`, the runner also logs each node call and each data transfer between two nodes.

## Choose the format

The text format gives one line for each record: the time, the level, the logger name, the event and the context fields.
The time is in UTC.

```text
13:50:15 INFO     [nodeml.my_project] Data loaded  experiment=demo  num_rows=200
```

The JSON format gives one JSON object on one line for each record:

```json
{"timestamp": "2026-11-10T13:50:15.373868+00:00", "level": "INFO", "event": "Data loaded", "logger": "nodeml.my_project", "context": {"experiment": "demo", "num_rows": 200}}
```

| Key | Value |
|---|---|
| `timestamp` | The time of the record, in ISO 8601 format, in UTC. |
| `level` | The level name, for example `"INFO"`. |
| `event` | The message of the record. |
| `logger` | The logger name. |
| `context` | The bound fields and the fields of the call. A value that is not JSON becomes a string. |

You can also use the formatters with your own handlers, for example `handler.setFormatter(JSONFormatter())`.

## Write to a file

`configure` keeps one handler for each format.
A second call with the same `fmt` replaces the handler of that format.
So you can write text to the terminal and JSON to a file at the same time:

```python
configure(level="INFO", stream=sys.stdout, fmt="text")
configure(level="INFO", filepath="pipeline.log", fmt="json")
```

To write JSON to the terminal and to a file, attach a second handler with `JSONFormatter` yourself.

## The `nodeml` logger hierarchy

All NodeML loggers are children of the `nodeml` logger.
So the level and the handlers of `nodeml` apply to all of them.

| Logger | Records |
|---|---|
| `nodeml.pipeline` | Compilation of a pipeline, pruned nodes, port compatibility warnings. |
| `nodeml.runner.smart` | The phases of {py:class}`~nodeml.core.pipeline.runners.smart_runner.SmartRunner`, the node calls and the errors. |
| `nodeml.pipeline.tuner.ray` | The tuning run and each trial (see {doc}`tuning`). |
| `nodeml.serving` | The start of a serving replica (see {doc}`serving`). |
| `nodeml.mlflow` | The MLflow config, the runs, the artifacts and the warnings (see {doc}`tracking`). |
| `nodeml.components` | The registration of the built-in nodes. |
| `nodeml.components.metrics` | Warnings of the metric nodes, for example the rows with NaN that they drop. `nodeml.components.metrics.mape` warns about zero targets. |
| `nodeml.counterfactuals` | The counterfactual evaluator. |

The `nodeml` logger sends its records to the Python root logger too.
If you also configure the root logger, for example with {py:func}`logging.basicConfig`, each record shows two times.
To stop this, set `logging.getLogger("nodeml").propagate = False`.

## Log from your own code

{py:class}`~nodeml.core.common.logging.logger.Logger` wraps a standard logger and adds a context to each record.
Give it a name below `nodeml`, so that `configure` applies to it.

```python
import sys

from nodeml.core.common.logging import Logger, configure

configure(level="INFO", stream=sys.stdout, fmt="text", use_color=False)

log = Logger("nodeml.my_project", experiment="demo")
log.info("Data loaded", num_rows=200)

trial_log = log.bind(trial=3)
with trial_log.timed("Training", model="forest") as scoped:
    scoped.info("Model fitted")
```

{py:meth}`~nodeml.core.common.logging.logger.Logger.bind` returns a new logger with more context fields.
{py:meth}`~nodeml.core.common.logging.logger.Logger.timed` logs `Training start` and `Training end` with `duration_ms`.
If the block raises an error, it logs `Training error` with the traceback, and raises the error again.
The fields of a call have priority over the bound fields with the same name.

## Logs in Ray workers

`configure` applies to the current Python process only.
The trials of {doc}`tuning` and the replicas of {doc}`serving` run in Ray worker processes, so their records do not show by default.
To configure the workers, give a setup function to Ray:

```python
import sys

import ray

from nodeml.core.common.logging import configure


def setup_logging():
    configure(level="INFO", stream=sys.stdout, fmt="json")


setup_logging()
ray.init(runtime_env={"worker_process_setup_hook": setup_logging})
```

Ray sends the output of each worker to the driver, with the worker name at the start of each line.

## Library settings

NodeML reads no environment variable for logging.
Three NodeML classes are pydantic-settings classes.
A pydantic-settings class reads its fields from environment variables:

| Class | Environment variables |
|---|---|
| {py:class}`~nodeml.components.pipe_runner_wrappers.mlflow_logger.MLFlowLoggerWrapperConfig` | `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME` and the other fields with the prefix `MLFLOW_`. See {doc}`tracking`. |
| {py:class}`~nodeml.core.pipeline.pipeline.PipelineSettings` | `AUTOPRUNE` sets the `autoprune` field, with no prefix. See {doc}`pipelines`. |
| {py:class}`~nodeml.core.common.mixins.mixin.MixinSettings` | None. It is the base class of the mixin settings in `NodeConfig.mixins_config`, and it has no fields. |

:::{note}
NodeML creates the default `settings` of {py:class}`~nodeml.core.pipeline.pipeline.PipelineConfig` when Python imports the module.
So `AUTOPRUNE=false` changes the default of all pipelines only when you set it before you import NodeML.
:::
