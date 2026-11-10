# Examples

The example notebooks are in the [`examples/`](https://github.com/AdrienBolling/NodeML/tree/main/examples) folder of the repository.
The documentation build runs each notebook, so the outputs on these pages come from the current code.
A notebook that fails also fails the build.

| Notebook | Description |
|---|---|
| {doc}`examples/pipeline` | Build an end-to-end pipeline with CSV sources, transforms, a model and metrics. |
| {doc}`examples/inputs_passthrough` | Give in-memory data to a pipeline for training, inference and evaluation. |
| {doc}`examples/save_and_load_pipeline` | Save a trained pipeline and load it again. |
| {doc}`examples/custom_node` | Write and register a custom node. |
| {doc}`examples/hyperparameter_tuning` | Search hyperparameters with Ray Tune. |
| {doc}`examples/mlflow_logging` | Log pipeline runs to MLflow. |
| {doc}`examples/counterfactuals` | Find counterfactual explanations with CELIA. |
| {doc}`examples/dice_regression` | Use DiCE regression with CELIA: first the CELIA API, then the evaluator. |

To run a notebook on your computer:

```bash
uv sync --extra counterfactuals
uv run jupyter lab examples/
```

```{toctree}
:hidden:
:glob:

examples/*
```
