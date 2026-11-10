# Serving

The module {py:mod}`nodeml.core.pipeline.serving.ray_serve` serves a trained pipeline with [Ray Serve](https://docs.ray.io/en/latest/serve/index.html).
The deployment has two interfaces:

- A Ray interface for Python clients: `handle.predict.remote(...)` and `handle.evaluate.remote(...)`. Ray sends the Python objects with pickle, with no JSON conversion.
- An HTTP interface with FastAPI: `POST /infer`, `POST /evaluate`, `GET /health` and `GET /info`, with JSON bodies.

## Save a trained pipeline

The examples on this page use the pipeline of {doc}`tuning`: an `InputsPassthrough` source named `source`, a `RandomForestRegressor`, and the metric nodes `mse` and `r2`.

:::{dropdown} Code of the example pipeline
```{include} ../_includes/example_pipeline.md
```
:::

Train the pipeline, then save its config and its trained parameters in one directory:

```python
from pathlib import Path

from nodeml.core.pipeline.runners.smart_runner import SmartRunner

SmartRunner(pipe).train(input_data=input_data)

save_dir = Path("saved_pipeline").resolve()
pipe.save_config_to_dir(save_dir)
pipe.save_params_to_dir(save_dir)
```

The directory now holds `demo_v0.1.0_config.json` and `demo_v0.1.0_params.pkl`.
See {doc}`pipelines` for more about saving and loading.

## Deploy the pipeline

{py:func}`~nodeml.core.pipeline.serving.ray_serve.build_pipeline_app` creates the Ray Serve application.
Give the pipeline in one of two forms:

- `pipeline_dir`: a directory with the saved config and parameters. If the directory holds more than one pipeline, also give `file_basename`.
- `pipeline_config` and `params_dir`: a {py:class}`~nodeml.core.pipeline.pipeline.PipelineConfig` and the directory of the trained parameters.

Give exactly one of `pipeline_dir` and `pipeline_config`, else the function raises `ValueError`.
Each replica reads the directories when it starts.
Use absolute paths that exist on the nodes of the replicas.

```python
import ray
from ray import serve

from nodeml.core.pipeline.serving.ray_serve import (
    PipelineServingConfig,
    build_pipeline_app,
)

ray.init(num_cpus=2)
app = build_pipeline_app(
    pipeline_dir=save_dir,
    config=PipelineServingConfig(num_replicas=1, ray_actor_options={"num_cpus": 1}),
)
handle = serve.run(app, name="demo")
```

`serve.run` returns a Ray Serve `DeploymentHandle`.
By default, Ray Serve listens for HTTP requests on `http://127.0.0.1:8000/`.
{py:class}`~nodeml.core.pipeline.serving.ray_serve.PipelineServingConfig` has these fields:

| Field | Default | Use |
|---|---|---|
| `runner_config` | `SmartRunnerConfig(verbose=False)` | The config of the {py:class}`~nodeml.core.pipeline.runners.smart_runner.SmartRunner` in each replica. |
| `num_replicas` | `1` | The number of replicas of the deployment. |
| `ray_actor_options` | `None` | The Ray actor options of each replica, for example `{"num_cpus": 1, "num_gpus": 0.5}`. `None` keeps the Ray Serve defaults. |

To run the application as a service, put `app` in a module, for example `my_service.py`.
Then start the module with the Ray Serve command line.
Press Ctrl+C to stop it.

```bash
serve run my_service:app
```

## Call the deployment from Python

The Ray interface takes and returns the same objects as the runner: `{node_name: {port_name: (array, context)}}` in, and `{name: (array, context)}` out.
`predict` runs `infer` and returns the outputs of the Sink.
`evaluate` returns the metrics.
In the example pipeline, the Sink port is `pred`.

```python
predictions = handle.predict.remote({"source": {"X": pair(X.head(3))}}).result()
frame, context = predictions["pred"]

metrics = handle.evaluate.remote(input_data).result()
print(metrics["mse"][0])
```

## Send HTTP requests

| Route | Body | Response |
|---|---|---|
| `POST /infer` | {py:class}`~nodeml.core.pipeline.serving.ray_serve.InferRequest` | The outputs of the Sink, by port name. |
| `POST /evaluate` | {py:class}`~nodeml.core.pipeline.serving.ray_serve.EvaluateRequest` | The metrics, by metric name. |
| `GET /health` | None | `{"status": "ok", "pipeline": <name>, "version": <version>}` |
| `GET /info` | None | The name, the version, the node names and the number of edges. |

A request body has the key `input_data`, with the form `{node_name: {port_name: port_payload}}`.
Each port payload is a {py:class}`~nodeml.core.pipeline.serving.ray_serve.PortPayload` with two keys:

`data`
: A list of records. Each record is one row, and its keys must be the columns of the context.

`context`
: The context of the data, with the lists `columns`, `dtypes` and `categories`. {py:attr}`~nodeml.core.common.data.data.TabularDataContext.dump_dict` gives this dict.

```python
import requests

X_frame, X_context = pair(X.head(2))
body = {
    "input_data": {
        "source": {
            "X": {
                "data": X_frame.to_dict(orient="records"),
                "context": X_context.dump_dict,
            }
        }
    }
}
response = requests.post("http://127.0.0.1:8000/infer", json=body, timeout=30)
print(response.json())
```

The server makes a DataFrame from the records, with the column order and the dtypes of the context.
The response has one {py:class}`~nodeml.core.pipeline.serving.ray_serve.PortResponse` for each output, with the same two keys:

```json
{
  "pred": {
    "data": [{"target": 0.5179}, {"target": 0.7078}],
    "context": {"columns": ["target"], "dtypes": ["float64"], "categories": ["numerical_data"]}
  }
}
```

Each value in `data` has its JSON type: integers stay integers and strings stay strings.
NaN, infinite and missing values become `null`.
Dates and times become ISO 8601 strings.
For an output that is not a DataFrame, the column names come from the context when its length matches.
Else, the column names are `col_0`, `col_1`, and so on.

The server returns HTTP 422 in two cases: the record keys are not the context columns, or a value does not convert to its dtype.
The `detail` key of the response gives the cause.

## Stop the deployment

Remove the application, then stop Ray Serve and Ray:

```python
serve.delete("demo")  # Remove the application "demo".
serve.shutdown()  # Stop Ray Serve.
ray.shutdown()
```

## Use the pipeline without Ray

{py:class}`~nodeml.core.pipeline.serving.ray_serve.ServedPipeline` holds the logic of the deployment, without Ray.
It takes the same arguments as `build_pipeline_app`.
Use it to test a pipeline and a request body in a normal Python process:

```python
from nodeml.core.pipeline.serving.ray_serve import InferRequest, ServedPipeline

served = ServedPipeline(pipeline_dir=save_dir)
response = served.infer_json(InferRequest.model_validate(body))
print(response["pred"].data)
```

`ServedPipeline` has the methods `predict`, `evaluate`, `infer_json`, `evaluate_json`, `health_info` and `pipeline_info`.
The Ray Serve deployment `PipelineServing` adds the HTTP routes to this class.
