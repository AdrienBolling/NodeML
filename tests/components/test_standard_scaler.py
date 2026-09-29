"""Tests for :class:`nodeml.components.nodes.transforms.scalers.standard_scaler.StandardScaler`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nodeml.components.nodes.transforms.scalers.standard_scaler import (
    StandardScaler,
    StandardScalerConfig,
)
from tests.shims.tabular import numerical_context, numerical_pair


class TestStandardScaler:
    def test_scaled_columns_are_zero_mean_unit_std(self) -> None:
        df, ctx = numerical_pair(n_rows=200, n_cols=3, seed=7)
        node = StandardScaler(config=StandardScalerConfig())

        out = node.node_fit_transform({"input": (df, ctx)})
        scaled, _ = out["output"]

        # Zero mean / unit std (allow 1e-8 numerical noise).
        means = scaled.mean().to_numpy()
        stds = scaled.std().to_numpy()
        np.testing.assert_allclose(means, np.zeros_like(means), atol=1e-8)
        np.testing.assert_allclose(stds, np.ones_like(stds), atol=1e-6)

    def test_params_round_trip(self) -> None:
        df, ctx = numerical_pair()
        node = StandardScaler(config=StandardScalerConfig())
        node.node_fit({"input": (df, ctx)})

        reborn = StandardScaler(config=StandardScalerConfig())
        reborn.set_params(node.get_params())
        assert reborn.get_params() == node.get_params()

    def test_zero_variance_column_is_only_centred(self) -> None:
        df = pd.DataFrame(
            {
                "const": np.ones(5, dtype=np.float64),
                "var": np.arange(5, dtype=np.float64),
            }
        )
        ctx = numerical_context(df)

        node = StandardScaler(config=StandardScalerConfig())
        out = node.node_fit_transform({"input": (df, ctx)})
        scaled, _ = out["output"]
        # Constant column collapses to zero; variance column is standardised.
        np.testing.assert_allclose(scaled["const"].to_numpy(), np.zeros(5))
