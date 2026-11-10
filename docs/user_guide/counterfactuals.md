# Counterfactual explanations

A counterfactual explanation answers this question: *what must change in this input, so that the prediction moves into a target range?*
The {py:class}`~nodeml.core.pipeline.counterfactuals.evaluator.CounterfactualEvaluator` finds and scores counterfactuals for a trained pipeline.
It uses the [CELIA](https://github.com/serval-uni-lu/celia) library, and it can run in parallel with Ray.
The original pipeline and its trained parameters do not change.

The examples on this page use the trained pipeline `pipe` and the data of the {doc}`/getting_started/quickstart`.
For complete examples with a categorical column, see {doc}`/examples/counterfactuals` and {doc}`/examples/dice_regression`.

## Install CELIA

CELIA is an optional dependency. Install the `counterfactuals` extra:

```bash
uv sync --extra counterfactuals
```

For other installation methods, see {doc}`/getting_started/installation`.
The configuration and report classes work without CELIA.
The evaluation needs CELIA.

## How the evaluator works

The evaluator keeps a copy of the pipeline config and of the trained parameters.
For each call to {py:meth}`~nodeml.core.pipeline.counterfactuals.evaluator.CounterfactualEvaluator.evaluate`, it does these steps:

1. It rebuilds the pipeline and inserts a {py:class}`~nodeml.core.nodes.transform.perturbation.PerturbationNode` at the insertion point. Every edge that leaves the insertion port then goes through the PerturbationNode. An extra edge sends the prediction to the Sink.
2. It runs the pipeline in inference on the reference input. It reads the rows that reach the PerturbationNode (the reference rows) and their predictions. CELIA uses these rows as the data distribution, and the predictions as the targets. Thus the counterfactuals explain the model, not the data.
3. It predicts each sample: this is the baseline prediction.
4. It runs one task for each (scenario, sample) pair. Each task rebuilds the pipeline with a PerturbationNode in injection mode, and gives it to the CELIA method as a black-box model. In injection mode, the nodes before the PerturbationNode do not run in inference. Each prediction that the method asks for runs only the nodes after the PerturbationNode.
5. It predicts each counterfactual again with the pipeline, and scores it.

## Create the evaluator

```python
from nodeml.core.pipeline.counterfactuals import (
    CounterfactualEvaluator,
    CounterfactualEvaluatorConfig,
    CounterfactualReport,
    CounterfactualScenario,
    FeatureConstraints,
    TargetRange,
)

evaluator = CounterfactualEvaluator.from_pipeline(
    pipe,
    config=CounterfactualEvaluatorConfig(
        constraints=FeatureConstraints(immutable_columns=["x2"]),
    ),
)
print(evaluator.points)
```

{py:meth}`~nodeml.core.pipeline.counterfactuals.evaluator.CounterfactualEvaluator.from_pipeline` needs a compiled, trained pipeline.
For a pipeline saved with `save_config_to_dir` and `save_params_to_dir`, use {py:meth}`~nodeml.core.pipeline.counterfactuals.evaluator.CounterfactualEvaluator.from_dir`.
The {py:class}`~nodeml.core.pipeline.counterfactuals.config.CounterfactualEvaluatorConfig` sets two points of the pipeline:

{py:class}`~nodeml.core.pipeline.counterfactuals.config.InsertionPoint`
: The output port that receives the PerturbationNode. The default is the source port that the model needs in inference: here `source.X`. There must be exactly one such source port. The counterfactuals, the samples and the constraints use the columns at this point. To insert the node after preprocessing, set `InsertionPoint(node=..., port=...)`. All the inference inputs of the model must go through this point.

{py:class}`~nodeml.core.pipeline.counterfactuals.config.OutputPoint`
: The prediction that the counterfactuals must change. The default is the only node output that feeds the Sink. If this output has more than one column, set `OutputPoint(column=...)`.

{py:attr}`~nodeml.core.pipeline.counterfactuals.evaluator.CounterfactualEvaluator.points` shows the resolved points.
If a point is ambiguous or does not exist, the creation of the evaluator raises {py:exc}`~nodeml.core.common.exceptions.PipelineValidationError`.

## Define the scenarios

A {py:class}`~nodeml.core.pipeline.counterfactuals.config.CounterfactualScenario` is one CELIA method with its settings and a target range.
Each scenario has a unique `name`.
`explainer_kwargs` go to the constructor of the CELIA explainer, and `generate_kwargs` go to its `generate_counterfactuals` method.
`constraints` replaces the constraints of the evaluator config for this scenario.

```python
scenarios = [
    CounterfactualScenario(
        name="dice",
        method="dice",
        target=TargetRange(kind="delta", low=1.0, high=2.0),
        generate_kwargs={"total_CFs": 2, "random_seed": 0},
    ),
    CounterfactualScenario(
        name="nnce",
        method="nnce",
        target=TargetRange(kind="absolute", low=4.0, high=6.0),
        generate_kwargs={"n_counterfactuals": 2},
        constraints=FeatureConstraints(),  # NNCE can change x1 and x2
    ),
]
```

The `nnce` scenario has its own empty constraints.
NNCE returns rows of the reference data, and no reference row has the same `x2` value as a sample.
Thus, with the immutable column `x2`, NNCE finds no counterfactual.

A {py:class}`~nodeml.core.pipeline.counterfactuals.config.TargetRange` gives the range for a sample from its baseline prediction `p`.
`low` must be less than `high`.
For a negative baseline, `"relative"` swaps the two bounds, so that the low bound stays below the high bound.

| `kind` | Range |
|---|---|
| `"absolute"` | `[low, high]` |
| `"delta"` (default) | `[p + low, p + high]` |
| `"relative"` | `[p * (1 + low), p * (1 + high)]`, for example `low=0.1, high=0.2` asks for 10 % to 20 % more than a positive baseline |

The supported methods:

| `method` | CELIA explainer | Settings |
|---|---|---|
| `"dice"` (default) | `DiceRegressorExplainer` | `explainer_kwargs={"method": ...}` selects `"random"` (default), `"genetic"` or `"kdtree"`. `generate_kwargs={"total_CFs": n}` sets the number of counterfactuals. |
| `"nnce"` | `NNCERegressorExplainer` | `generate_kwargs={"n_counterfactuals": n}` sets the number of counterfactuals. |

DiCE `"random"` and NNCE convert every value to float.
For these methods, the evaluator gives integer codes to CELIA instead of string categories, and decodes the results.
NNCE then measures distances between codes, which is only an approximation.
Do not give `features_to_vary` or `permitted_range` to DiCE: CELIA sets them from the constraints.

## Set the constraints

{py:class}`~nodeml.core.pipeline.counterfactuals.config.FeatureConstraints` holds the real-world constraints on the columns at the insertion point.
CELIA uses them, and each method supports a subset of them.

| Field | Meaning |
|---|---|
| `immutable_columns` | Columns that a counterfactual must not change. |
| `feasible_ranges` | The allowed `(min, max)` range of a numerical column. |
| `feasible_categories` | The allowed values of a categorical column. |
| `monotonic_increasing_columns` | Columns that can only increase. |
| `correlated_features` | `(cause, effect, delta)` triples: when `cause` increases, `effect` increases by `delta`. |
| `frozen_columns` | Columns that CELIA does not see. Every counterfactual keeps the value of the sample. Use them for identifiers and for columns with many categories. |

A method can break a constraint.
Thus the evaluator checks the immutable columns and the feasible values of each counterfactual, and reports each violation.

CELIA does not accept missing values.
With `missing_values="freeze"` (the default), a column that is missing in the sample is frozen.
CELIA gets the reference rows with their missing values filled: the median for a continuous column, the most frequent value for another column.
The pipeline always gets the real rows.
With `missing_values="error"`, a missing value gives the status `"error"`.

## Run the evaluation

```python
report = evaluator.evaluate(
    scenarios,
    samples=X_test.iloc[:3],
    reference_input={"source": {"X": (X_train, X_context)}},
)
print(report.summary_frame())
```

`samples` are the rows to explain, with the columns at the insertion point.
The index labels of `samples` become the sample identifiers.
Instead of `samples`, you can give `sample_indices`: positions in the reference rows, which also become the identifiers.
`reference_input` is the runner input for the reference rows.
For a source that loads its own data, for example a {py:class}`~nodeml.components.nodes.data_sources.tabular_csv_fetcher.TabularCSVFetcher`, keep the default `None`.

Before you set an absolute target range, look at the predictions of the reference rows.
{py:meth}`~nodeml.core.pipeline.counterfactuals.evaluator.CounterfactualEvaluator.reference_data` returns them in a {py:class}`~nodeml.core.pipeline.counterfactuals.evaluator.ReferenceData`.
If the model cannot reach the target range, each method gives the status `"not_found"`.

### Ray tasks

With `use_ray=True` (the default), each (scenario, sample) pair is one Ray task, and the tasks run in parallel.
Each task reserves `num_cpus_per_task` CPUs (default 1).
If Ray is not initialized, the evaluator calls `ray.init()`.
To use a specific Ray cluster, call `ray.init()` before the evaluation.
With `use_ray=False`, the tasks run one after the other in the current process.

### Failure isolation

A failure in one task does not stop the evaluation.
The failed task gives a result with the status `"error"` and a message, and the other tasks keep their results.
This rule applies to these failures:

- A method raises an exception.
- The pipeline cannot predict or score the counterfactuals.
- A task raises an exception in Ray or in the current process.

A sample can also have no baseline prediction: the pipeline cannot predict it, or the prediction is not a finite number.
Then the sample gets the status `"error"` in each scenario.
The evaluator logs a warning when a method or a task fails.

The status of each result:

| Status | Meaning |
|---|---|
| `"found"` | The method returned counterfactuals. Each counterfactual can be valid or not valid. |
| `"not_found"` | The method found no counterfactual. |
| `"already_in_range"` | The baseline prediction is already in the target range. |
| `"error"` | The task failed. The `message` gives the cause. |

Some errors stop the evaluation before the tasks start.
For example, duplicate scenario names or samples without the columns at the insertion point raise {py:exc}`~nodeml.core.common.exceptions.NodeConfigError`.

## Read the report

{py:meth}`~nodeml.core.pipeline.counterfactuals.evaluator.CounterfactualEvaluator.evaluate` returns a {py:class}`~nodeml.core.pipeline.counterfactuals.report.CounterfactualReport`.
The report holds the pipeline name, version and hash, the resolved points, the columns, the scenarios and one {py:class}`~nodeml.core.pipeline.counterfactuals.report.SampleResult` for each (scenario, sample) pair.
Each result holds the original row, the baseline prediction, the target range, the status, and a {py:class}`~nodeml.core.pipeline.counterfactuals.report.CounterfactualRecord` for each counterfactual.

Each counterfactual has these metrics:

- **Validity** (`valid`): the pipeline prediction for the counterfactual is in the target range.
- **Proximity** (`l1_distance`, `l2_distance`): the distances to the original row over the continuous columns. Each difference is divided by the range of the column in the reference rows.
- **Sparsity** (`n_changed`): the number of changed columns.
- **Violations** (`violations`): a changed immutable column (`"immutable:<column>"`), or a value outside the feasible values (`"feasible:<column>"`).

The report gives these pandas views:

| Method | Rows |
|---|---|
| {py:meth}`~nodeml.core.pipeline.counterfactuals.report.CounterfactualReport.summary_frame` | One row for each scenario: the number of samples for each status, the number of counterfactuals, the validity rate, the mean metrics, the number of violations and the run time. |
| {py:meth}`~nodeml.core.pipeline.counterfactuals.report.CounterfactualReport.samples_frame` | One row for each (scenario, sample) pair: the baseline prediction, the target range, the status and the message. |
| {py:meth}`~nodeml.core.pipeline.counterfactuals.report.CounterfactualReport.counterfactuals_frame` | One row for each counterfactual: its metrics and all its columns. |
| {py:meth}`~nodeml.core.pipeline.counterfactuals.report.CounterfactualReport.changes_frame` | The original row of one sample, then its counterfactuals. `"-"` marks a value that did not change. |

In the summary, the mean distances and the mean number of changes use only the valid counterfactuals.

```python
print(report.samples_frame()[["scenario", "sample_id", "status", "n_valid"]])
print(report.changes_frame("dice", "150"))
```

The sample identifier `"150"` is the index label of the first row of `X_test`.

## Save and load a report

The report is a pydantic model.
{py:meth}`~nodeml.core.pipeline.counterfactuals.report.CounterfactualReport.save_json` writes it as JSON, and creates the parent directory if necessary.
{py:meth}`~nodeml.core.pipeline.counterfactuals.report.CounterfactualReport.load_json` reads it back.

```python
path = report.save_json("reports/counterfactuals.json")
assert CounterfactualReport.load_json(path) == report
```

## What is supported now

- **Task**: regression only. Classification is not supported yet.
- **Methods**: DiCE (`"random"`, `"genetic"` and `"kdtree"`) and NNCE.
- **BugDoc**: not supported. With CELIA 0.1.0, the BugDoc regressor always fails.

For the structure of a pipeline, see {doc}`pipelines`.
For the execution modes of the runner, see {doc}`running`.
For the nodes that you can use before the model, see {doc}`built_in_nodes`.
