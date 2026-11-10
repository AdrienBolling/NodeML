# API reference

This reference comes from the docstrings in the source code.
Each module has one page. A package page lists its submodules.
An object is on the page of the module that defines it.
For example, {py:class}`~nodeml.core.pipeline.pipeline.Pipeline` is on the page of {py:mod}`nodeml.core.pipeline.pipeline`.

The main entry points:

| Object | Use |
|---|---|
| {py:data}`~nodeml.core.nodes.registry.node_registry.NODE_REGISTRY` | Find the nodes and their config classes. |
| {py:class}`~nodeml.core.pipeline.pipeline.Pipeline` | Build, compile, save and load a pipeline. |
| {py:class}`~nodeml.core.pipeline.runners.smart_runner.SmartRunner` | Train, infer and evaluate a compiled pipeline. |
| {py:class}`~nodeml.core.pipeline.validation.train_val_split.TabularTrainValSplit` | Split tabular data into training and validation inputs. |
| {py:class}`~nodeml.core.pipeline.counterfactuals.evaluator.CounterfactualEvaluator` | Find counterfactual explanations for a trained pipeline. |

% autosummary finds only reStructuredText directives, so keep this block in eval-rst.
```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   nodeml
```
