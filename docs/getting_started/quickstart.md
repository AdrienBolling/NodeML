---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Quickstart

This page builds a small regression pipeline, trains it, runs inference, evaluates it and saves it.
The pipeline reads in-memory data, scales the features, fits a linear regression and computes the R² score.
Each section adds one step.
Together, the code blocks make one script that runs from top to bottom.
The documentation build runs this script, so the outputs on this page come from the current code.

Install NodeML before you start (see {doc}`installation`).

```{code-cell} python
:tags: [remove-cell]

# Write the saved files to a temporary folder, not into the docs.
import os
import tempfile

os.chdir(tempfile.mkdtemp())
```

## 1. Make the data

```{code-cell} python
import numpy as np
import pandas as pd

from nodeml.core.common.data.data import NumericalData, TabularDataContext

rng = np.random.default_rng(0)
X = pd.DataFrame({"x1": rng.normal(size=200), "x2": rng.normal(size=200)})
y = pd.DataFrame({"target": 3.0 * X["x1"] + 2.0 * X["x2"] + rng.normal(0.0, 0.1, 200)})
X_train, X_test = X.iloc[:150], X.iloc[150:]
y_train, y_test = y.iloc[:150], y.iloc[150:]

X_context = TabularDataContext(
    columns=["x1", "x2"], dtypes=list(X.dtypes), categories=[NumericalData] * 2
)
y_context = TabularDataContext(
    columns=["target"], dtypes=list(y.dtypes), categories=[NumericalData]
)
```

Tabular data goes into a pipeline as a pair: a pandas DataFrame and a {py:class}`~nodeml.core.common.data.data.TabularDataContext`.
The context gives the name, the dtype and the data category of each column.

## 2. Declare the data source

```{code-cell} python
from nodeml.components.nodes.data_sources.inputs_passthrough import InputsPassthroughConfig
from nodeml.core.common.data.data import ArrayLikeEnum, DataCategoryEnum, DataStructureEnum
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.nodes.node import Port

source_config = InputsPassthroughConfig(
    out_ports={
        "X": Port(
            arr_type=ArrayLikeEnum.PANDAS,
            data_structure=DataStructureEnum.TABULAR,
            data_category=DataCategoryEnum.NUMERICAL,
            data_shape="batch feature",
            desc="The features.",
        ),
        "y": Port(
            arr_type=ArrayLikeEnum.PANDAS,
            data_structure=DataStructureEnum.TABULAR,
            data_category=DataCategoryEnum.NUMERICAL,
            data_shape="batch 1",
            desc="The target.",
            mode=[NodeExecutionMode.TRAINING, NodeExecutionMode.EVALUATION],
        ),
    }
)
```

The {py:class}`~nodeml.components.nodes.data_sources.inputs_passthrough.InputsPassthrough` source node forwards the data that you give to the runner.
Declare one output {py:class}`~nodeml.core.nodes.node.Port` for each input.
The `y` port is active only in training and evaluation, so inference needs no target.

## 3. Build the pipeline

```{code-cell} python
from nodeml.core.pipeline.pipeline import Edge, Pipeline, PipelineConfig

pipe = Pipeline(
    config=PipelineConfig(
        name="quickstart",
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
    )
)
```

Each node has a name, a registered node type and a config.
The config `None` selects the default config of the node type.
Each {py:class}`~nodeml.core.pipeline.pipeline.Edge` connects output ports of its source node to input ports of its target node.
The Sink collects the pipeline outputs: the edge `("pred", "pred")` creates the Sink port `pred`.

## 4. Compile and train

```{code-cell} python
from nodeml.core.pipeline.runners.smart_runner import SmartRunner

pipe.compile()
runner = SmartRunner(pipe)
runner.train(
    input_data={"source": {"X": (X_train, X_context), "y": (y_train, y_context)}}
)
```

{py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.compile` validates the graph and creates the node objects.
The {py:class}`~nodeml.core.pipeline.runners.smart_runner.SmartRunner` runs the compiled pipeline.
`input_data` maps each source node name to its ports, and each port to a `(DataFrame, context)` pair.
The runner shows a progress bar for each phase.
To remove it, give `config=SmartRunnerConfig(verbose=False)` to the runner (see {py:class}`~nodeml.core.pipeline.runners.smart_runner.SmartRunnerConfig`).

## 5. Run inference

```{code-cell} python
outputs = runner.infer(input_data={"source": {"X": (X_test, X_context)}})
predictions, _ = outputs["pred"]
print(predictions.head())
```

{py:meth}`~nodeml.core.pipeline.runners.smart_runner.SmartRunner.infer` returns one `(DataFrame, context)` pair for each Sink port.
The runner does not run the `y` port in inference, so `input_data` holds only `X`.

## 6. Evaluate

```{code-cell} python
metrics = runner.evaluate(
    input_data={"source": {"X": (X_test, X_context), "y": (y_test, y_context)}}
)
r2, _ = metrics["r2"]
print(f"R2 on the test rows: {r2.iloc[0, 0]:.4f}")
```

{py:meth}`~nodeml.core.pipeline.runners.smart_runner.SmartRunner.evaluate` runs the metric nodes and returns their outputs.
For a metric node with one output port, the key is the name of the node.

## 7. Save and load

```{code-cell} python
pipe.save_config_to_dir("saved_pipeline")
pipe.save_params_to_dir("saved_pipeline")

loaded = Pipeline.load_from_dir("saved_pipeline")
loaded_runner = SmartRunner(loaded)
loaded_outputs = loaded_runner.infer(input_data={"source": {"X": (X_test, X_context)}})
assert np.allclose(loaded_outputs["pred"][0], predictions)
```

{py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.save_config_to_dir` writes the pipeline config as JSON.
{py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.save_params_to_dir` writes the trained parameters as a pickle file.
The file names start with the pipeline name and version, for example `quickstart_v0.1.0_config.json`.
{py:meth}`~nodeml.core.pipeline.pipeline.Pipeline.load_from_dir` rebuilds the pipeline, compiles it and loads the parameters.
The loaded pipeline gives the same predictions.

```{warning}
Load only parameter files that you trust. Unpickling can run any code.
```

## Next steps

- {doc}`/user_guide/concepts`: nodes, ports, data and the registry.
- {doc}`/user_guide/pipelines`: build, edit, save and load a pipeline.
- {doc}`/user_guide/running`: the execution modes of the runner.
- {doc}`/user_guide/built_in_nodes`: the data sources, transforms, models and metrics that NodeML supplies.
- {doc}`/user_guide/custom_nodes`: write and register your own node.
- {doc}`/examples/inputs_passthrough`: more about in-memory data.
- {doc}`/examples`: all the example notebooks.
- {doc}`/api/index`: the API reference.
