"""Sphinx directive that lists the built-in nodes from ``NODE_REGISTRY``.

The catalog comes from the registry at build time, so it cannot go out of
date. The directive writes reStructuredText, so in a MyST page put it in an
``eval-rst`` block::

    ```{eval-rst}
    .. node-catalog::
    ```
"""

from typing import Any, ClassVar

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective

from nodeml import NODE_REGISTRY

# Section order and titles. Node types that are not here come last.
SECTIONS = {
    "source": "Data sources",
    "transform": "Transforms",
    "model": "Models",
    "metric": "Metrics",
    "sink": "Sink",
}


def _summary(obj: type) -> str:
    """Return the first line of the docstring of *obj*."""
    doc = (obj.__doc__ or "").strip()
    return doc.splitlines()[0] if doc else ""


def _class_ref(obj: type) -> str:
    """Return a cross-reference to *obj*, or plain code for a private module."""
    path = f"{obj.__module__}.{obj.__qualname__}"
    if any(part.startswith("_") for part in obj.__module__.split(".")):
        return f"``{obj.__qualname__}``"
    return f":py:class:`~{path}`"


class NodeCatalog(SphinxDirective):
    """Render one table per node type, with the registry name of each node."""

    has_content = False
    option_spec: ClassVar[dict[str, Any]] = {}

    def run(self) -> list[nodes.Node]:
        """Build the tables from the registry."""
        groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for name, entry in NODE_REGISTRY.items():
            groups.setdefault(str(entry["node_type"]), []).append((name, entry))
        order = [t for t in SECTIONS if t in groups] + sorted(
            t for t in groups if t not in SECTIONS
        )

        lines: list[str] = []
        for node_type in order:
            title = SECTIONS.get(node_type, node_type.capitalize())
            lines += [
                f".. rubric:: {title}",
                "",
                ".. list-table::",
                "   :header-rows: 1",
                "   :widths: 25 30 45",
                "",
                "   * - Registry name",
                "     - Node class",
                "     - Description",
            ]
            for name, entry in sorted(groups[node_type]):
                node_class = entry["node_class"]
                summary = _summary(node_class).replace("|", r"\|")
                lines += [
                    f"   * - ``{name}``",
                    f"     - {_class_ref(node_class)}",
                    f"     - {summary}",
                ]
            lines.append("")

        container = nodes.container()
        self.state.nested_parse(
            StringList(lines, source="node-catalog"), self.content_offset, container
        )
        return list(container.children)


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the directive."""
    app.add_directive("node-catalog", NodeCatalog)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
