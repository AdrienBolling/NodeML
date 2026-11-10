```python
import numpy as np
import pandas as pd

from nodeml import NODE_REGISTRY
from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    NumericalData,
    TabularDataContext,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.nodes.node import Port
from nodeml.core.pipeline.pipeline import Edge, Pipeline, PipelineConfig


def pair(frame):
    """Return the (DataFrame, context) tuple of a numerical DataFrame."""
    context = TabularDataContext(
        columns=list(frame.columns),
        dtypes=list(frame.dtypes),
        categories=[NumericalData] * frame.shape[1],
    )
    return frame, context


def numerical_port(shape, **kwargs):
    """Return a port for a numerical DataFrame."""
    return Port(
        arr_type=ArrayLikeEnum.PANDAS,
        data_structure=DataStructureEnum.TABULAR,
        data_category=DataCategoryEnum.NUMERICAL,
        data_shape=shape,
        desc="Numerical data.",
        **kwargs,
    )


def default_config(node_type):
    """Return the default config of a registered node."""
    return NODE_REGISTRY.get_node_config_class(node_type)()


rng = np.random.default_rng(0)
X = pd.DataFrame(rng.standard_normal((200, 3)), columns=["x0", "x1", "x2"])
y = pd.DataFrame({"target": 3.0 * X["x0"] - X["x1"] + rng.normal(0, 0.1, 200)})
input_data = {"source": {"X": pair(X), "y": pair(y)}}

source_config = default_config("InputsPassthrough")
source_config.out_ports = {
    "X": numerical_port("batch feature"),
    "y": numerical_port(
        "batch 1", mode=[NodeExecutionMode.TRAINING, NodeExecutionMode.EVALUATION]
    ),
}
nodes = {
    "source": ("InputsPassthrough", source_config),
    "forest": ("RandomForestRegressor", default_config("RandomForestRegressor")),
    "mse": ("MSE", default_config("MSE")),
    "r2": ("R2Score", default_config("R2Score")),
    "sink": ("Sink", default_config("Sink")),
}
edges = [
    Edge(source="source", target="forest", ports_map=[("X", "X"), ("y", "y")]),
    Edge(source="forest", target="sink", ports_map=[("pred", "pred")]),
]
for metric in ("mse", "r2"):
    edges.append(Edge(source="forest", target=metric, ports_map=[("pred", "pred")]))
    edges.append(Edge(source="source", target=metric, ports_map=[("y", "target")]))

pipe = Pipeline(config=PipelineConfig(nodes=nodes, edges=edges, name="demo"))
pipe.compile()
```
