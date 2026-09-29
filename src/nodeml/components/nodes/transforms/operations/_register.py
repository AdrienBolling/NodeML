"""Register operation transform nodes from this package into the NODE_REGISTRY."""

from nodeml.core.nodes.registry.node_registry import NODE_REGISTRY

from .column_order import (
    ColumnOrder,
    ColumnOrderConfig,
    ColumnOrderHyperParameters,
    ColumnOrderRunningConfig,
)
from .feature_concatenate import (
    FeatureConcatenate,
    FeatureConcatenateConfig,
    FeatureConcatenateHyperParameters,
    FeatureConcatenateRunningConfig,
)
from .row_concatenate import (
    RowConcatenate,
    RowConcatenateConfig,
    RowConcatenateHyperParameters,
    RowConcatenateRunningConfig,
)


def register_nodes() -> None:
    """Register all operation transform nodes defined in this package."""
    NODE_REGISTRY.register(
        name="RowConcatenate",
        node_class=RowConcatenate,
        node_config_class=RowConcatenateConfig,
        running_config_class=RowConcatenateRunningConfig,
        hyperparameters_class=RowConcatenateHyperParameters,
    )
    NODE_REGISTRY.register(
        name="FeatureConcatenate",
        node_class=FeatureConcatenate,
        node_config_class=FeatureConcatenateConfig,
        running_config_class=FeatureConcatenateRunningConfig,
        hyperparameters_class=FeatureConcatenateHyperParameters,
    )
    NODE_REGISTRY.register(
        name="ColumnOrder",
        node_class=ColumnOrder,
        node_config_class=ColumnOrderConfig,
        running_config_class=ColumnOrderRunningConfig,
        hyperparameters_class=ColumnOrderHyperParameters,
    )
