"""Base class for registries in the NodeML framework.

CRUD operations live here.  All rendering (plain-text tables, notebook
widgets) is provided by :class:`~nodeml.core.common.registry_display.RegistryDisplayMixin`.
"""

from collections.abc import ItemsView, Iterator
from typing import Any

from nodeml.core.common.registry_display import RegistryDisplayMixin


class Registry(RegistryDisplayMixin):
    """Base Registry.

    It is deliberately schema-flexible:
    any extra field you add to an entry will automatically appear in the overview.
    """

    _column_order: list[str] = []

    def __init__(self, entity: str) -> None:
        """Initialise an empty registry for *entity* entries."""
        self._entity = entity
        self._registry: dict[str, dict[str, Any]] = {}

    # --------------------------------------------------------
    # Core CRUD
    # --------------------------------------------------------

    def _register(
        self,
        name: str,
        class_callable: type,
        config_class_callable: type,
        **extra_fields: Any,
    ) -> None:
        """Register an entity with its class and config class.

        This is the internal registration method used by subclasses to add
        entries with a predefined schema (entity class + config class).

        Args:
            name: Unique name for the registry entry.
            class_callable: The entity class to register.
            config_class_callable: The associated configuration class.
            **extra_fields: Arbitrary extra metadata stored alongside the entry.

        Raises:
            ValueError: If *name* is already registered.

        Example:
            registry._register(
                "resize",
                ResizeMetric,
                ResizeConfig,
                family="vision",
                tags=["image", "preprocess"],
                stable=True,
            )

        """
        if name in self._registry:
            message = (
                f"{self._entity.capitalize()} '{name}' is already registered. Please choose a different "
                f"name or unregister the existing entry first."
            )
            raise ValueError(message)

        self._registry[name] = {
            f"{self._entity}_class": class_callable,
            f"{self._entity}_config_class": config_class_callable,
            **extra_fields,
        }

    def register_entry(self, name: str, **fields: Any) -> None:
        """Fully generic registration.

        Example:
            registry.register_entry(
                "resize",
                metric_class=ResizeMetric,
                config_class=ResizeConfig,
                family="vision",
                tags=["image", "preprocess"],
            )

        """
        if name in self._registry:
            message = (
                f"{self._entity.capitalize()} '{name}' is already registered. Please choose a different "
                f"name or unregister the existing {self._entity} first."
            )
            raise ValueError(message)

        self._registry[name] = dict(fields)

    def unregister(self, name: str) -> None:
        """Remove a previously registered entry.

        Args:
            name: The name of the entry to remove.

        Raises:
            ValueError: If *name* is not currently registered.

        """
        if name not in self._registry:
            message = f"{self._entity.capitalize()} '{name}' is not registered. Cannot unregister a non-existent {self._entity}."
            raise ValueError(message)
        del self._registry[name]

    def get(self, name: str) -> dict[str, Any]:
        """Retrieve the metadata dict for a registered entry.

        Args:
            name: The name of the entry to look up.

        Returns:
            A dictionary of all fields stored for the entry.

        Raises:
            ValueError: If *name* is not registered.

        """
        if name not in self._registry:
            message = f"{self._entity.capitalize()} '{name}' is not registered. Please register the {self._entity} before trying to retrieve it."
            raise ValueError(message)
        return self._registry[name]

    def __getitem__(self, name: str) -> dict[str, Any]:
        """Return the entry for *name*, delegating to :meth:`get`."""
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        """Check whether *name* is registered."""
        return name in self._registry

    def __len__(self) -> int:
        """Return the number of registered entries."""
        return len(self._registry)

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered entry names."""
        return iter(self._registry)

    def __str__(self) -> str:
        """Return a one-line summary of the registered names."""
        return f"{self._entity.capitalize()}Registry with {len(self._registry)} registered {self._entity}s: {list(self._registry.keys())}"

    def __repr__(self) -> str:
        """Return the full registry content."""
        return f"{self._entity.capitalize()}Registry(registry={self._registry})"

    def keys(self) -> list[str]:
        """Return a list of all registered entry names."""
        return list(self._registry.keys())

    def items(self) -> ItemsView[str, dict[str, Any]]:
        """Return all ``(name, fields)`` pairs in the registry."""
        return self._registry.items()
