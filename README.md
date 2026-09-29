# NodeML

A Python framework for building, training, and deploying **ML pipelines as directed acyclic graphs (DAGs)**. Each node in the graph — data source, transform, model, or metric — communicates through typed ports, and the runner handles execution order, mode-aware pruning, and progress tracking automatically.

## Key ideas

- **Pipeline = DAG.** Nodes declare typed input/output ports; edges wire them together. The framework resolves execution order via topological sort.
- **Mode-aware execution.** Ports carry an execution mode (`training`, `inference`, `evaluation`, or `all`). During inference the runner automatically skips target-only branches; during evaluation it walks metric nodes.
- **Registry-driven.** All built-in nodes self-register at import time. Discover them with `NODE_REGISTRY.list()`, retrieve configs with `NODE_REGISTRY.get_node_config_class(...)`.
- **Save / load.** A trained pipeline is fully captured by a JSON config and a pickle of fitted parameters — `save_config_to_dir`, `save_params_to_dir`, then `Pipeline.load_from_dir` to rebuild, compile and load it.
- **One Sink.** Send every output you need to the Sink node with an edge. Compilation creates the Sink ports from these edges, in edge order.

## Built-in nodes

| Category | Nodes |
|---|---|
| **Data sources** | `TabularCSVFetcher`, `InputsPassthrough` |
| **Transforms** | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `OneHotEncoding`, `LabelEncoding`, `IQROutlierFilter`, `ZScoreOutlierFilter`, `NumericalImputation`, `CategoricalImputation`, `CorrelationFilter`, `VarianceFilter`, `MissingRateFilter`, `DataCategoryFilter`, `ColumnOrder`, `FeatureConcatenate`, `RowConcatenate` |
| **Models** | `LinearRegression`, `RandomForestRegressor`, `RandomForestClassifier`, `GradientBoostingRegressor`, `GradientBoostingClassifier`, `MLP`, `CNN` |
| **Metrics** | `R2Score`, `MSE`, `MAE`, `MAPE`, `Accuracy`, `AUROC`, `F1Score`, `Precision`, `Recall` |
| **Sink** | `Sink` |

## Quick start

```python
from nodeml import NODE_REGISTRY
from nodeml.core.pipeline.pipeline import Edge, Pipeline, PipelineConfig
from nodeml.core.pipeline.runners.smart_runner import SmartRunner

# Configure nodes
source_cfg = NODE_REGISTRY.get_node_config_class("InputsPassthrough")(...)
model_cfg  = NODE_REGISTRY.get_node_config_class("LinearRegression")()

# Build the DAG
pipe = Pipeline(config=PipelineConfig(
    nodes={
        "source": ("InputsPassthrough", source_cfg),
        "model":  ("LinearRegression",  model_cfg),
        "sink":   ("Sink", NODE_REGISTRY.get_node_config_class("Sink")()),
    },
    edges=[
        Edge(source="source", target="model", ports_map=[("X", "X"), ("y", "y")]),
        Edge(source="model",  target="sink",  ports_map=[("pred", "dump")]),
    ],
))

# Compile, train, infer
pipe.compile()
runner = SmartRunner(pipe)
runner.train(input_data={"source": {"X": X_pair, "y": y_pair}})
preds = runner.infer(input_data={"source": {"X": X_pair}})
```

## Examples

| Notebook | Description |
|---|---|
| [`pipeline.ipynb`](examples/pipeline.ipynb) | End-to-end pipeline with CSV sources, transforms, model, and metrics |
| [`inputs_passthrough.ipynb`](examples/inputs_passthrough.ipynb) | Feeding in-memory data for training, inference, and evaluation |
| [`save_and_load_pipeline.ipynb`](examples/save_and_load_pipeline.ipynb) | Persisting and reloading a trained pipeline |
| [`custom_node.ipynb`](examples/custom_node.ipynb) | Creating and registering a custom node |
| [`hyperparameter_tuning.ipynb`](examples/hyperparameter_tuning.ipynb) | Hyperparameter search with Ray Tune |
| [`mlflow_logging.ipynb`](examples/mlflow_logging.ipynb) | Logging pipeline runs to MLflow |
| [`counterfactuals.ipynb`](examples/counterfactuals.ipynb) | Counterfactual explanations with CELIA |
| [`dice_regression.ipynb`](examples/dice_regression.ipynb) | DiCE regression with CELIA: the CELIA API, then the evaluator |

## Counterfactual explanations

`CounterfactualEvaluator` asks: *what must change in this input, so that the prediction moves into a target range?* It uses the [CELIA](https://github.com/serval-uni-lu/celia) library, which is an optional dependency:

```bash
uv sync --extra counterfactuals        # or: pip install "nodeml[counterfactuals]"
```

The evaluator rebuilds a trained pipeline, inserts a `PerturbationNode` (by default right after the data source), and gives the rest of the pipeline to CELIA as a black-box model. Each scenario (a CELIA method and a target range) runs on each sample as a Ray task. The report holds the baseline prediction of each sample, and the validity, distance, sparsity and constraint violations of each counterfactual. The original pipeline does not change.

```python
from nodeml.core.pipeline.counterfactuals import (
    CounterfactualEvaluator, CounterfactualEvaluatorConfig, CounterfactualScenario,
    FeatureConstraints, TargetRange,
)

evaluator = CounterfactualEvaluator.from_pipeline(
    trained_pipeline,
    config=CounterfactualEvaluatorConfig(
        constraints=FeatureConstraints(immutable_columns=["age"]),
    ),
)
report = evaluator.evaluate(
    [CounterfactualScenario(
        name="dice",
        method="dice",
        target=TargetRange(kind="relative", low=-0.4, high=-0.2),
        generate_kwargs={"total_CFs": 3},
    )],
    samples=X.iloc[:5],
    reference_input={"source": {"X": (X, X_context)}},
)
report.summary_frame()
```

Supported now: regression, with the DiCE and NNCE methods of CELIA.

## Installation

Requires **Python 3.13+**.

```bash
git clone git@github.com:AdrienBolling/NodeML.git
cd NodeML
uv sync
```

## Contributing

```bash
uv sync                      # installs the package and the dev group
uv run pre-commit install    # ruff, ty, uv-lock and nbstripout hooks
```

The `nbstripout` hook removes notebook outputs before each commit.

Run the tests:

```bash
uv run pytest
```

Build the docs:

```bash
uv run --group docs sphinx-build docs docs/_build
```
