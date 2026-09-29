"""Train / validation split utility for NodeML pipelines.

Operates on the ``input_data`` structure used by
:class:`~nodeml.core.pipeline.runners.smart_runner.SmartRunner`
(``{node_name: {port_name: (array, context)}}``) and produces two copies
(train and validation) that can be passed directly to ``runner.train()``
and ``runner.evaluate()``.  Values can also be :class:`TabularData`
objects.  Entries that the config does not reference are copied unchanged
into both outputs.

Example::

    splitter = TabularTrainValSplit(config=TabularTrainValSplitConfig(
        data_nodes=[("features_source", ["output"])],
        target_nodes=[("target_source", ["output"])],
        val_proportion=0.2,
        randomize=True,
        best_kldiv=True,
    ))
    train_data, val_data = splitter.split(input_data)
    runner.train(input_data=train_data)
    metrics = runner.evaluate(input_data=val_data)
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from nodeml.core.common.data.data import (
    ArrayLike,
    DataContext,
    TabularData,
    TabularDataContext,
)

type _PortValue = tuple[ArrayLike, DataContext] | TabularData
type _InputData = Mapping[str, Mapping[str, _PortValue]]
type _SplitData = dict[str, dict[str, _PortValue]]


def _as_tabular(value: _PortValue, where: str) -> TabularData:
    """Return *value* as :class:`TabularData`.

    Raises:
        TypeError: If *value* is neither TabularData nor an
            ``(array, TabularDataContext)`` tuple.

    """
    if isinstance(value, TabularData):
        return value
    if (
        isinstance(value, tuple)
        and len(value) == 2  # noqa: PLR2004 - (array, context)
        and isinstance(value[1], TabularDataContext)
    ):
        data, ctx = value
        return TabularData(data, ctx.columns, ctx.dtypes, ctx.categories)
    msg = (
        "TabularTrainValSplit supports TabularData or (array, "
        f"TabularDataContext) tuples, got {type(value)} at {where}."
    )
    raise TypeError(msg)


class TabularTrainValSplitConfig(BaseModel):
    """Configuration for the train / validation splitter."""

    data_nodes: list[tuple[str, list[str]]] = Field(
        description=(
            "Source nodes whose data is split.  Each tuple is "
            "``(node_name, [port_name, ...])``.  All referenced "
            "TabularData objects must have the same number of rows."
        ),
    )
    target_nodes: list[tuple[str, list[str]]] | None = Field(
        default=None,
        description=(
            "Target nodes to split in sync with data_nodes.  "
            "``None`` if no targets are needed."
        ),
    )
    randomize: bool = Field(
        default=False,
        description="Shuffle row order before splitting.",
    )
    val_proportion: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="Fraction of rows reserved for validation.",
    )
    best_kldiv: bool = Field(
        default=False,
        description=(
            "When ``True``, try multiple random splits and keep the one "
            "whose validation distribution best matches the training "
            "distribution (lowest KL divergence).  Implies randomize."
        ),
    )
    kldiv_trials: int = Field(
        default=100,
        ge=1,
        description="Number of random splits to evaluate when best_kldiv is True.",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility.",
    )


class TabularTrainValSplit:
    """Split ``input_data`` into train and validation subsets.

    The split indices are computed once from the data referenced by
    ``data_nodes`` (and optionally ``target_nodes``) and applied to every
    referenced ``TabularData`` object, preserving the row correspondence
    across all source and target nodes.
    """

    def __init__(self, config: TabularTrainValSplitConfig) -> None:
        """Initialize the splitter.

        Args:
            config: Split configuration controlling proportions, randomization,
                and KL-divergence optimization.

        """
        self._config = config

    def split(self, input_data: _InputData) -> tuple[_SplitData, _SplitData]:
        """Split *input_data* into ``(train_data, val_data)``.

        Both returned dicts have the ``node_name -> port_name -> value``
        structure of the runner ``input_data``, so they can be passed
        directly to the runner.  A split value is a
        ``(pd.DataFrame, TabularDataContext)`` tuple; an entry that the
        config does not reference is copied unchanged.

        Args:
            input_data: Full dataset keyed by
                ``{node_name: {port_name: (array, context)}}``.  Values can
                also be :class:`TabularData`.

        Returns:
            A ``(train_data, val_data)`` tuple.

        """
        all_refs = self._all_refs()
        tabular = {
            (node, port): _as_tabular(input_data[node][port], f"{node}:{port}")
            for node, port in all_refs
        }
        n_rows = self._verify_row_count(tabular)

        train_idx, val_idx = self._compute_indices(tabular, n_rows)

        train_data = self._apply_split(input_data, tabular, train_idx)
        val_data = self._apply_split(input_data, tabular, val_idx)
        return train_data, val_data

    # ------------------------------------------------------------------
    # Reference helpers
    # ------------------------------------------------------------------

    def _all_refs(self) -> list[tuple[str, str]]:
        """Flatten config into ``(node_name, port_name)`` pairs."""
        refs = self._data_refs()
        if self._config.target_nodes is not None:
            refs.extend(
                (node_name, port)
                for node_name, ports in self._config.target_nodes
                for port in ports
            )
        return refs

    def _data_refs(self) -> list[tuple[str, str]]:
        """Only the data (non-target) references."""
        return [
            (node_name, port)
            for node_name, ports in self._config.data_nodes
            for port in ports
        ]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _verify_row_count(tabular: dict[tuple[str, str], TabularData]) -> int:
        """Check that every referenced value has the same number of rows."""
        n_rows: int | None = None
        for (node_name, port_name), data in tabular.items():
            rows = data.shape[0]
            if n_rows is None:
                n_rows = rows
            elif rows != n_rows:
                msg = (
                    f"Row count mismatch: {node_name}:{port_name} has "
                    f"{rows} rows but expected {n_rows}."
                )
                raise ValueError(msg)
        if n_rows is None:
            msg = "No data references provided."
            raise ValueError(msg)
        return n_rows

    # ------------------------------------------------------------------
    # Index computation
    # ------------------------------------------------------------------

    def _compute_indices(
        self,
        tabular: dict[tuple[str, str], TabularData],
        n_rows: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(train_indices, val_indices)``."""
        rng = np.random.default_rng(self._config.seed)
        split_point = int(n_rows * (1 - self._config.val_proportion))

        if self._config.best_kldiv:
            return self._best_kldiv_split(tabular, n_rows, split_point, rng)

        indices = np.arange(n_rows)
        if self._config.randomize:
            rng.shuffle(indices)

        return indices[:split_point], indices[split_point:]

    def _best_kldiv_split(
        self,
        tabular: dict[tuple[str, str], TabularData],
        n_rows: int,
        split_point: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Try multiple random splits and keep the one with lowest KL divergence."""
        features = self._gather_features(tabular)

        best_kl = float("inf")
        best_train: np.ndarray | None = None
        best_val: np.ndarray | None = None

        for _ in range(self._config.kldiv_trials):
            indices = rng.permutation(n_rows)
            train_idx = indices[:split_point]
            val_idx = indices[split_point:]

            kl = _kl_divergence_score(features, train_idx, val_idx)
            if kl < best_kl:
                best_kl = kl
                best_train = train_idx
                best_val = val_idx

        if best_train is None or best_val is None:
            msg = "No split was evaluated; kldiv_trials must be at least 1."
            raise ValueError(msg)
        return best_train, best_val

    def _gather_features(
        self,
        tabular: dict[tuple[str, str], TabularData],
    ) -> pd.DataFrame:
        """Concatenate all data_nodes columns into a single DataFrame for KL computation."""
        # Positional index: the frames of different ports are aligned by row.
        frames = [
            tabular[ref].to_pandas()[0].reset_index(drop=True)
            for ref in self._data_refs()
        ]
        return pd.concat(frames, axis=1)

    # ------------------------------------------------------------------
    # Split application
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_split(
        input_data: _InputData,
        tabular: dict[tuple[str, str], TabularData],
        indices: np.ndarray,
    ) -> _SplitData:
        """Slice every referenced value and copy the other entries unchanged."""
        result: _SplitData = {
            node_name: dict(ports) for node_name, ports in input_data.items()
        }
        for (node_name, port_name), data in tabular.items():
            df, ctx = data.to_pandas()
            result[node_name][port_name] = (
                df.iloc[indices].reset_index(drop=True),
                ctx.copy(),
            )
        return result


# ----------------------------------------------------------------------
# KL divergence helpers (no scipy dependency)
# ----------------------------------------------------------------------

_N_BINS = 20
_SMOOTHING = 1e-10


def _kl_divergence_score(
    features: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> float:
    """Sum of per-column KL(val || train) with Laplace smoothing."""
    total_kl = 0.0
    for col in features.columns:
        series = features[col]
        train_col = series.iloc[train_idx]
        val_col = series.iloc[val_idx]

        if pd.api.types.is_numeric_dtype(series):
            lo = series.min()
            hi = series.max()
            if lo == hi:
                continue
            bins = np.linspace(lo, hi, _N_BINS + 1)
            p = np.histogram(train_col, bins=bins)[0].astype(float)
            q = np.histogram(val_col, bins=bins)[0].astype(float)
        else:
            categories = series.unique()
            p = np.array([(train_col == c).sum() for c in categories], dtype=float)
            q = np.array([(val_col == c).sum() for c in categories], dtype=float)

        # Smoothing and normalisation
        p += _SMOOTHING
        q += _SMOOTHING
        p /= p.sum()
        q /= q.sum()

        # KL(q || p): how well does val match train
        total_kl += float(np.sum(q * np.log(q / p)))

    return total_kl
