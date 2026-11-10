# Development

This page tells you how to set up the repository, run the checks, and maintain this documentation.

## Set up the repository

```bash
git clone https://github.com/AdrienBolling/NodeML.git
cd NodeML
uv sync                      # the package and the dev group
uv run pre-commit install    # ruff, ty, uv-lock and nbstripout hooks
```

The `nbstripout` hook removes the notebook outputs before each commit.
Thus the notebooks in git have no outputs.

## Run the checks

| Check | Command |
|---|---|
| Tests | `uv run pytest` |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format --check .` |
| Types | `uv run ty check src` |

The `Checks` workflow runs the same commands on each pull request and on each push to `main`.
The `ty` job only reports problems, because `ty` is an alpha release.

## Build the documentation

```bash
uv run --group docs --extra counterfactuals sphinx-build -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser.
The build runs all the example notebooks, so it needs the `counterfactuals` extra.
The first build takes some minutes. Later builds use the notebook cache in `docs/_build/.jupyter_cache` and run only the notebooks that changed.

`-W` changes each warning into an error, and `--keep-going` shows all the errors before the build stops.
The configuration sets `nitpicky = True`, so a broken cross-reference is an error.

Check the external links:

```bash
uv run --group docs --extra counterfactuals sphinx-build -b linkcheck docs docs/_build/linkcheck
```

## How the documentation is organised

| Folder or file | Content | Maintenance |
|---|---|---|
| `docs/index.md` | Home page | Manual |
| `docs/getting_started/` | Installation and quickstart | Manual |
| `docs/user_guide/` | One page for each topic | Manual |
| `docs/user_guide/built_in_nodes.md` | Catalog of the built-in nodes | Automatic: the `node-catalog` directive in `docs/_ext/node_catalog.py` reads `NODE_REGISTRY` |
| `docs/examples/` | Link to the `examples/` folder | Automatic: the build runs each notebook |
| `docs/api/` | API reference | Automatic: `autosummary` makes one page for each module in `docs/api/generated/` |
| `docs/conf.py` | Sphinx configuration | Manual |

The pages are [MyST Markdown](https://myst-parser.readthedocs.io/).
The `docs/api/generated/` folder is not in git. The build makes it again each time.

## Update the documentation

New module
: Do nothing. `autosummary` finds it. A module whose name starts with `_` is private and does not get a page.

New node
: Register it in `NODE_REGISTRY`. The catalog shows it at the next build. Write the first line of the class docstring as a one-sentence description, because the catalog shows that line.

New notebook
: Add it to `examples/` and add a row to the table in `docs/examples.md`. The notebook must have one `#` title, and must run from start to end in a few minutes.

New guide page
: Add the file to `docs/user_guide/` and add it to the toctree and the list in `docs/user_guide/index.md`.

Docstrings
: Use the Google style (`Args:`, `Returns:`, `Raises:`). Refer to other objects with a role, for example ``:class:`~nodeml.core.pipeline.pipeline.Pipeline` ``. Use the full path of the module that defines the object.

## Publish the documentation

The `Documentation` workflow (`.github/workflows/docs.yml`) does these steps:

1. On each pull request, it builds the documentation with `-W`. A warning fails the pull request.
2. On each push to `main`, it builds the documentation and publishes it to GitHub Pages.
3. Each Monday, it checks the external links. You can also start this check by hand from the Actions tab.
