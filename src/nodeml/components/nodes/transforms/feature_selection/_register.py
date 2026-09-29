"""Register feature-selection transform nodes from this package into the NODE_REGISTRY."""

from nodeml.core.nodes.registry.node_registry import NODE_REGISTRY

from .correlation_filter import (
    CorrelationFilter,
    CorrelationFilterConfig,
    CorrelationFilterHyperParameters,
    CorrelationFilterRunningConfig,
)
from .data_category_filter import (
    DataCategoryFilter,
    DataCategoryFilterConfig,
    DataCategoryFilterHyperParameters,
    DataCategoryFilterRunningConfig,
)
from .missing_rate_filter import (
    MissingRateFilter,
    MissingRateFilterConfig,
    MissingRateFilterHyperParameters,
    MissingRateFilterRunningConfig,
)
from .variance_filter import (
    VarianceFilter,
    VarianceFilterConfig,
    VarianceFilterHyperParameters,
    VarianceFilterRunningConfig,
)


def register_nodes() -> None:
    """Register all feature-selection nodes defined in this package."""
    NODE_REGISTRY.register(
        name="DataCategoryFilter",
        node_class=DataCategoryFilter,
        node_config_class=DataCategoryFilterConfig,
        running_config_class=DataCategoryFilterRunningConfig,
        hyperparameters_class=DataCategoryFilterHyperParameters,
    )
    NODE_REGISTRY.register(
        name="MissingRateFilter",
        node_class=MissingRateFilter,
        node_config_class=MissingRateFilterConfig,
        running_config_class=MissingRateFilterRunningConfig,
        hyperparameters_class=MissingRateFilterHyperParameters,
    )
    NODE_REGISTRY.register(
        name="VarianceFilter",
        node_class=VarianceFilter,
        node_config_class=VarianceFilterConfig,
        running_config_class=VarianceFilterRunningConfig,
        hyperparameters_class=VarianceFilterHyperParameters,
    )
    NODE_REGISTRY.register(
        name="CorrelationFilter",
        node_class=CorrelationFilter,
        node_config_class=CorrelationFilterConfig,
        running_config_class=CorrelationFilterRunningConfig,
        hyperparameters_class=CorrelationFilterHyperParameters,
    )
