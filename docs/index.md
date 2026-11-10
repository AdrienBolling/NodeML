# NodeML

NodeML is a Python framework to build, train, tune and serve machine learning pipelines.
A pipeline is a directed acyclic graph (DAG) of nodes.
Each node is a data source, a transform, a model, a metric or the Sink.
Nodes connect through typed ports.
The runner finds the execution order, runs only the branches that the current mode needs, and checks the data between nodes.

```python
from nodeml import NODE_REGISTRY

NODE_REGISTRY.keys()  # the names of all built-in nodes
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: getting_started/index
:link-type: doc

Install NodeML and train your first pipeline.
:::

:::{grid-item-card} User guide
:link: user_guide/index
:link-type: doc

Learn the concepts: nodes, ports, pipelines, runners, tuning, serving and counterfactual explanations.
:::

:::{grid-item-card} Examples
:link: examples
:link-type: doc

Run the example notebooks. The documentation build runs each notebook, so the outputs are always current.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Read the reference of each module, class and function.
:::
::::

## Key ideas

Pipeline = DAG
: Nodes declare typed input and output ports. Edges connect output ports to input ports. Compilation validates the graph and sorts it.

Mode-aware execution
: Each port has the execution modes in which it is active: `training`, `inference`, `evaluation` or `all`. In inference, the runner does not run the branches that only feed targets. In evaluation, it runs the metric nodes.

Registry
: The built-in nodes register themselves in {py:data}`~nodeml.core.nodes.registry.node_registry.NODE_REGISTRY` when you import `nodeml`. You can register your own nodes in the same registry.

Save and load
: A JSON config and a pickle of the fitted parameters capture a trained pipeline completely.

One Sink
: Connect each output that you need to the Sink node. Compilation creates the Sink ports from these edges.

```{toctree}
:hidden:
:maxdepth: 2

getting_started/index
user_guide/index
examples
api/index
development
```
