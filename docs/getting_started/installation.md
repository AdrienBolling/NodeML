# Installation

NodeML needs Python 3.13 or later.
NodeML is not on PyPI. Install it from the Git repository.

## Install with uv

Use [uv](https://docs.astral.sh/uv/) to work on the repository.
uv creates the virtual environment from the lock file `uv.lock`.

```bash
git clone https://github.com/AdrienBolling/NodeML.git
cd NodeML
uv sync
```

`uv sync` installs NodeML and the `dev` dependency group.
Run your code with `uv run`, for example `uv run python my_script.py`.

## Install in another project

Add NodeML as a Git dependency of your project:

```bash
uv add "nodeml @ git+https://github.com/AdrienBolling/NodeML.git"
```

With pip:

```bash
pip install "nodeml @ git+https://github.com/AdrienBolling/NodeML.git"
```

## Optional dependencies

The `counterfactuals` extra installs [CELIA](https://github.com/serval-uni-lu/celia).
You need it for {doc}`/user_guide/counterfactuals`.

```bash
uv sync --extra counterfactuals
```

In another project, add the extra to the requirement: `nodeml[counterfactuals] @ git+https://github.com/AdrienBolling/NodeML.git`.

```{note}
CELIA and some of its dependencies install from Git only.
Thus PyPI does not accept them, and they stay in an extra.
```

## Check the installation

```bash
uv run python -c "from nodeml import NODE_REGISTRY; print(len(NODE_REGISTRY), 'nodes')"
```

The command prints the number of built-in nodes.
Continue with the {doc}`quickstart`.
