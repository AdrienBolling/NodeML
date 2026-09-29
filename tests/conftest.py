"""Shared fixtures for the NodeML test suite.

All heavy side-effects (importing ``nodeml`` to trigger the auto-discovery
of registered nodes, writing CSV/JSON files to a tmp dir) live here so
individual test modules can stay lean and declarative.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Triggers component auto-registration — must happen before any test that
# looks up nodes in ``NODE_REGISTRY`` by name.
import nodeml  # noqa: F401
from nodeml.core.common.data.data import (
    NumericalData,
    TabularData,
    TabularDataContext,
)
from tests.shims import tabular as tabular_shim

# ---------------------------------------------------------------------------
# Deterministic tabular fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def numerical_dataframe() -> pd.DataFrame:
    """Return a 20x3 deterministic numerical DataFrame."""
    return tabular_shim.numerical_df()


@pytest.fixture
def numerical_context(numerical_dataframe: pd.DataFrame) -> TabularDataContext:
    """Context that marks every column of ``numerical_dataframe`` as numerical."""
    return tabular_shim.numerical_context(numerical_dataframe)


@pytest.fixture
def numerical_pair(
    numerical_dataframe: pd.DataFrame,
    numerical_context: TabularDataContext,
) -> tuple[pd.DataFrame, TabularDataContext]:
    """``(DataFrame, TabularDataContext)`` tuple ready to drop onto a port."""
    return numerical_dataframe, numerical_context


@pytest.fixture
def tabular_data(numerical_dataframe: pd.DataFrame) -> TabularData:
    """``TabularData`` wrapping the numerical DataFrame fixture."""
    return TabularData(
        data=numerical_dataframe,
        columns=list(numerical_dataframe.columns),
        dtypes=list(numerical_dataframe.dtypes),
        categories=[NumericalData for _ in numerical_dataframe.columns],
    )


@pytest.fixture
def regression_dataset():
    """Deterministic (X, y) dataset with y ≈ X @ coefs.

    Returns:
        ``((X_df, X_ctx), (y_df, y_ctx), true_coefs)`` — see
        :func:`tests.shims.tabular.linear_regression_dataset`.

    """
    return tabular_shim.linear_regression_dataset()


# ---------------------------------------------------------------------------
# Temporary CSV + JSON context fixture for TabularCSVFetcher
# ---------------------------------------------------------------------------


@pytest.fixture
def csv_and_context_files(
    tmp_path: Path,
    numerical_dataframe: pd.DataFrame,
    numerical_context: TabularDataContext,
) -> tuple[Path, Path]:
    """Write the numerical DataFrame + matching context JSON to *tmp_path*.

    Yields a ``(csv_path, ctx_path)`` tuple that can be fed directly to
    ``TabularCSVFetcherRunningConfig``.
    """
    csv_path = tmp_path / "data.csv"
    ctx_path = tmp_path / "data.json"
    numerical_dataframe.to_csv(csv_path, index=False)
    ctx_path.write_text(json.dumps(numerical_context.dump_dict))
    return csv_path, ctx_path


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    """Shared, seeded numpy RNG for tests that need fresh random numbers."""
    return np.random.default_rng(42)
