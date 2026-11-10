"""Sphinx configuration for the NodeML documentation.

Build the docs from the repository root:

    uv run --group docs --extra counterfactuals sphinx-build -W --keep-going docs docs/_build/html
"""

import os
import sys
from importlib.metadata import version as package_version
from pathlib import Path
from typing import cast

from docutils.nodes import Element, TextElement
from pydantic import BaseModel
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.domains.python import PythonDomain
from sphinx.environment import BuildEnvironment
from sphinx.ext import intersphinx

DOCS_DIR = Path(__file__).parent
sys.path.insert(0, str(DOCS_DIR / "_ext"))

# Ray finds the "uv run" parent process and tries to ship the project to its
# workers. The notebooks run inside the project, so turn this hook off.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
# Make Plotly figures in the notebooks HTML that loads plotly.js from a CDN.
# The default output type is not shown by myst-nb.
os.environ.setdefault("PLOTLY_RENDERER", "notebook_connected")
# -- Project information -----------------------------------------------------

project = "NodeML"
author = "Adrien Bolling"
copyright = "2025, Adrien Bolling"  # noqa: A001
release = package_version("nodeml")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
    "node_catalog",
]

templates_path = ["_templates"]
# Files in _includes are parts of other pages.
exclude_patterns = ["_build", "_includes", "jupyter_execute", "**.ipynb_checkpoints"]

# Fail on every broken cross-reference. Use with "sphinx-build -W".
nitpicky = True
nitpick_ignore_regex = [
    # Packages without an intersphinx inventory.
    (
        r"py:.*",
        r"(jaxtyping|beartype|ray|mlflow|fastapi|starlette|celia|torchmetrics|plotly|tqdm)(\..*)?",
    ),
    # Names from these packages in type hints.
    (
        r"py:.*",
        r"(HTTPException|MlflowException|TuneConfig|RunConfig|ResultGrid|go\.Figure)",
    ),
    # A Ray Serve deployment is not a class, so autodoc does not document it.
    (r"py:.*", r"(.+\.)?PipelineServing"),
    # The type of jaxtyping.AbstractDtype.dtypes contains a sentinel object.
    (r"py:.*", r"typing\.Literal\[<object object at .*"),
    # Private modules and objects do not have a page.
    (r"py:.*", r"(.+\.)?_[^_].*"),
    # Type parameters, for example T or D_I.
    (r"py:.*", r"(.+\.)?([A-Z]|D_[A-Z_]+)"),
]

# -- MyST and notebooks ------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# Notebook outputs are not in git (nbstripout), so run every notebook at
# build time. A failing cell fails the build. The cache skips notebooks that
# did not change since the last local build.
nb_execution_mode = "cache"
nb_execution_timeout = 600
nb_execution_raise_on_error = True
nb_execution_show_tb = True
nb_merge_streams = True

# -- API reference -----------------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    # Document each object once, in the module that defines it. Re-exports
    # in __init__.py (listed in __all__) would make duplicates.
    "ignore-module-all": True,
}
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_preserve_defaults = True

napoleon_google_docstring = True
napoleon_numpy_docstring = False
# Render "Attributes:" sections as fields, not as separate objects. Separate
# objects collide with the attributes that autodoc documents.
napoleon_use_ivar = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "NodeML"
html_static_path = ["_static"]
html_show_sourcelink = False
html_theme_options = {
    "github_url": "https://github.com/AdrienBolling/NodeML",
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "secondary_sidebar_items": ["page-toc"],
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Link check --------------------------------------------------------------

# These sites make their anchors with JavaScript, so linkcheck cannot see them.
linkcheck_anchors_ignore_for_url = [
    r"https://github\.com/.*",
    r"https://docs\.pytorch\.org/.*",
]

# -- Hooks -------------------------------------------------------------------

# Pydantic members that each model has. They add no information to the pages.
PYDANTIC_MEMBERS = {
    "model_config",
    "model_post_init",
    "model_fields",
    "model_computed_fields",
}

# Targets that autodoc writes from the runtime type, and their inventory names.
REFERENCE_ALIASES = {
    "pandas.core.frame.DataFrame": "pandas.DataFrame",
    "pandas.core.series.Series": "pandas.Series",
    "BaseModel": "pydantic.BaseModel",
    "Path": "pathlib.Path",
}


def _skip_member(
    _app: Sphinx,
    _what: str,
    name: str,
    _obj: object,
    _skip: bool,  # noqa: FBT001
    _options: object,
) -> bool | None:
    """Skip members that would make duplicates or noise.

    ``class C(TransformConfig[H, R])`` puts ``TransformConfig[H, R]`` in the
    namespace of the module. Autodoc would document the base class again.
    """
    if "[" in name or name in PYDANTIC_MEMBERS:
        return True
    return None


def _drop_pydantic_init_doc(
    _app: Sphinx, what: str, _name: str, obj: object, _options: object, lines: list[str]
) -> None:
    """Remove the ``BaseModel.__init__`` docstring from the model pages.

    ``autoclass_content = "both"`` adds the docstring of ``__init__`` to the
    class docstring. A model that does not define ``__init__`` gets the
    generic text of Pydantic.
    """
    if (
        what == "class"
        and isinstance(obj, type)
        and issubclass(obj, BaseModel)
        and lines
        and lines[0].startswith("Create a new model by parsing and validating")
    ):
        lines.clear()


def _hide_pydantic_signature(
    _app: Sphinx,
    what: str,
    _name: str,
    obj: object,
    _options: object,
    signature: str | None,
    return_annotation: str | None,
) -> tuple[str | None, str | None] | None:
    """Remove the signature of Pydantic models.

    The signature lists every field with its default value. The field
    descriptions below the class give the same information in a clear form.
    """
    if what == "class" and isinstance(obj, type) and issubclass(obj, BaseModel):
        return "", return_annotation
    return signature, return_annotation


def _context_classes(node: pending_xref) -> list[type]:
    """Return the classes in which a short name can be a member.

    These are the current class and its bases, then the classes of the
    current module and their bases.
    """
    module_name = node.get("py:module")
    module = sys.modules.get(module_name) if module_name else None
    if module is None:
        return []
    classes: list[type] = []
    class_name = node.get("py:class")
    if class_name:
        current: object = module
        for part in class_name.split("."):
            current = getattr(current, part, None)
        if isinstance(current, type):
            classes += current.__mro__
    for value in vars(module).values():
        if isinstance(value, type) and value.__module__ == module_name:
            classes += value.__mro__
    return classes


def _external_reference(
    app: Sphinx,
    env: BuildEnvironment,
    node: pending_xref,
    contnode: TextElement,
    names: list[str],
) -> Element | None:
    """Resolve the first of *names* that an intersphinx inventory has."""
    original = node["reftarget"]
    for name in names:
        node["reftarget"] = name
        resolved = intersphinx.missing_reference(app, env, node, contnode)
        if resolved is not None:
            return resolved
    node["reftarget"] = original
    return None


def _local_match(
    domain: PythonDomain, target: str, candidates: list[str]
) -> str | None:
    """Return the documented object for *target*, or ``None``."""
    for name in [*candidates, target]:
        if name in domain.objects:
            return name
    matches = [name for name in domain.objects if name.endswith("." + target)]
    return matches[0] if len(matches) == 1 else None


def _resolve_reference(
    app: Sphinx, env: BuildEnvironment, node: pending_xref, contnode: TextElement
) -> Element | None:
    """Resolve a Python reference that the normal search does not find.

    1. Change a runtime path to the name in an intersphinx inventory.
    2. Change a parametrized generic, for example ``ModelConfig[H, R]``, to
       its base class.
    3. Resolve a short name, for example ``:meth:`fit``` in a docstring, as a
       member of the current class, of its bases, or of the module classes.
    4. Resolve a short name, for example ``Pipeline`` in a type hint, when
       exactly one documented object has this name.
    5. Resolve a member of a base class from another package, for example
       ``model_dump`` of ``pydantic.BaseModel``, with intersphinx.
    6. Show the name of a base class from a private module as plain text.
    """
    if node.get("refdomain") != "py":
        return None
    target = node["reftarget"].lstrip(".~")
    if target in REFERENCE_ALIASES:
        return _external_reference(
            app, env, node, contnode, [REFERENCE_ALIASES[target]]
        )
    target = target.split("[", 1)[0]
    domain = cast("PythonDomain", env.get_domain("py"))
    classes = _context_classes(node)
    candidates = [f"{c.__module__}.{c.__qualname__}.{target}" for c in classes]
    match = _local_match(domain, target, candidates)
    if match is not None:
        return domain.resolve_xref(
            env, node["refdoc"], app.builder, node["reftype"], match, node, contnode
        )
    external = [name for name in candidates if not name.startswith("nodeml.")]
    resolved = _external_reference(app, env, node, contnode, external)
    if resolved is not None:
        return resolved
    if any(c.__name__ == target and "._" in f".{c.__module__}" for c in classes):
        return contnode
    return None


def setup(app: Sphinx) -> None:
    """Connect the hooks."""
    app.connect("autodoc-skip-member", _skip_member)
    app.connect("autodoc-process-docstring", _drop_pydantic_init_doc)
    app.connect("autodoc-process-signature", _hide_pydantic_signature)
    app.connect("missing-reference", _resolve_reference)
