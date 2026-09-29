"""Built-in component nodes for the NodeML framework.

This package provides ready-to-use implementations of data sources, models,
transforms, metrics, and sinks. All components are automatically discovered
and registered in the global :data:`~nodeml.core.nodes.registry.node_registry.NODE_REGISTRY`
when the ``nodeml`` package is imported.

To add a custom component, create a ``_register.py`` module in the appropriate
subdirectory with a ``register_nodes()`` function.
"""

import importlib
import pkgutil

from nodeml.core.common.logging import Logger

_log = Logger("nodeml.components")


def _auto_discovery() -> None:
    """Walk all subpackages and import every ``_register`` module.

    Each ``_register`` module is expected to define a ``register_nodes()``
    callable that registers its nodes with the global NODE_REGISTRY.
    """
    for module_info in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
        if not module_info.name.endswith("._register"):
            continue
        mod = importlib.import_module(module_info.name)
        register_nodes = getattr(mod, "register_nodes", None)
        if callable(register_nodes):
            register_nodes()
            _log.debug("Registered nodes", module=module_info.name)
        else:
            _log.warning(
                "Module has no register_nodes() function, skipped",
                module=module_info.name,
            )
