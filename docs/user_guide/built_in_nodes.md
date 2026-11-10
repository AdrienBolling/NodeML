# Built-in nodes

NodeML supplies data sources, transforms, models, metrics and the Sink.
Each node registers in {py:data}`~nodeml.core.nodes.registry.node_registry.NODE_REGISTRY` under a registry name.
Use the registry name in a {py:class}`~nodeml.core.pipeline.pipeline.PipelineConfig`, and to get the config class of the node:

```python
from nodeml import NODE_REGISTRY

config_class = NODE_REGISTRY.get_node_config_class("StandardScaler")
config = config_class()
```

The tables below come from the registry when the documentation builds.
The class page in the API reference gives the ports, the config fields and the hyperparameters of each node.

% The directive writes reStructuredText, so it must run in an eval-rst block.
```{eval-rst}
.. node-catalog::
```

## List the nodes in Python

`NODE_REGISTRY.keys()` returns the registry names.
`NODE_REGISTRY.list()` returns the registry as a plain-text table.
In a notebook, it shows an interactive table instead.
See {doc}`concepts` for the registry, and {doc}`custom_nodes` to add a node.
