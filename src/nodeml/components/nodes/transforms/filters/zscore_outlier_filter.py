"""Z-Score outlier filter transform node for the NodeML Framework.

A value is an outlier when its absolute z-score is more than a
configurable cutoff, that is, when it is outside
``[mean - cutoff·std, mean + cutoff·std]``.  Two strategies are available:

* ``"remove"`` - remove the rows with outliers (default), in the training
  mode only.
* ``"cap"``    - clip each outlier value to ``mean ± cutoff·std``, in all
  modes.

The filter learns the **mean** and **std** of each column during
:meth:`fit` and uses them again at :meth:`transform` time.  The filter
skips a column with a zero or undefined std (for example, a fit on one
row): such a column never has outliers.  See :mod:`._outlier_filter` for
the shared behaviour.
"""

from typing import Any

import pandas as pd
from pydantic import Field
from ray import tune

from nodeml.core.nodes.node import Port
from nodeml.core.nodes.transform.transform import TransformConfig, TransformMetadata

from ._outlier_filter import (
    OutlierFilter,
    OutlierFilterHyperParameters,
    OutlierFilterParams,
    OutlierFilterRunningConfig,
    outlier_filter_in_ports,
    outlier_filter_out_ports,
)


class ZScoreOutlierFilterMetadata(TransformMetadata):
    """Metadata for the ZScoreOutlierFilter node."""

    node_name: str = "ZScoreOutlierFilter"
    description: str = (
        "Detect and handle outliers using the Z-score method. "
        "A value is an outlier if |z| > zscore_cutoff."
    )


class ZScoreOutlierFilterRunningConfig(OutlierFilterRunningConfig):
    """Run-time knobs that do not affect the learned parameters."""


class ZScoreOutlierFilterHyperParameters(OutlierFilterHyperParameters):
    """Tuneable hyperparameters for the ZScoreOutlierFilter."""

    zscore_cutoff: float = Field(
        default=3.0,
        gt=0.0,
        description=(
            "Absolute Z-score above which a value is considered an outlier. "
            "The standard choice is 3.0 (≈99.7 % of a Gaussian is within ±3 sigma). "
            "Lower values are more aggressive."
        ),
    )


# Exposed at module level so external tuners can discover the search space.
hyperparameter_space: dict[str, Any] = {
    "zscore_cutoff": tune.uniform(0.5, 10.0),
    "threshold": tune.uniform(0.0, 1.0),
    "strategy": tune.choice(["remove", "cap"]),
}


class ZScoreOutlierFilterConfig(
    TransformConfig[
        ZScoreOutlierFilterHyperParameters,
        ZScoreOutlierFilterRunningConfig,
    ]
):
    """Full configuration for the ZScoreOutlierFilter node."""

    hyperparameters: ZScoreOutlierFilterHyperParameters = Field(
        default_factory=ZScoreOutlierFilterHyperParameters,
        description="Tuneable hyperparameters (zscore_cutoff, threshold, strategy).",
    )
    running_config: ZScoreOutlierFilterRunningConfig = Field(
        default_factory=ZScoreOutlierFilterRunningConfig,
        description="Run-time options (filtering_columns).",
    )
    in_ports: dict[str, Port] = Field(
        default_factory=outlier_filter_in_ports,
        description="Input ports: 'input', 'target' (training/evaluation), 'sliced' (any DataFrame to keep in sync).",
    )
    out_ports: dict[str, Port] = Field(
        default_factory=outlier_filter_out_ports,
        description="Output ports: 'output', 'target' (training/evaluation), 'sliced' (synced auxiliary).",
    )


class ZScoreOutlierFilter(OutlierFilter):
    """Z-Score outlier filter.

    Learns the mean and std of each column on the training split.  Then
    uses these statistics to detect and handle outliers at transform time.
    Row removal happens only in the training mode.

    Example:
        >>> cfg = ZScoreOutlierFilterConfig(
        ...     hyperparameters=ZScoreOutlierFilterHyperParameters(zscore_cutoff=2.5),
        ... )
        >>> node = ZScoreOutlierFilter(config=cfg)

    """

    metadata = ZScoreOutlierFilterMetadata()
    hyperparameter_space = hyperparameter_space

    def _fit_statistics(self, candidates: pd.DataFrame) -> OutlierFilterParams:
        """Return the mean and std of each column of *candidates*.

        The std is NaN for a fit on one row, and zero for a constant
        column.  :meth:`_raw_bounds` skips these columns.
        """
        means = candidates.mean()
        stds = candidates.std()
        return {
            "mean": {col: float(means[col]) for col in candidates.columns},
            "std": {col: float(stds[col]) for col in candidates.columns},
        }

    def _raw_bounds(self) -> tuple[pd.Series, pd.Series]:
        """Return ``mean ± zscore_cutoff·std`` for each fitted column.

        A column with a zero or NaN std gets NaN bounds, so it is skipped.
        """
        k = self._config.hyperparameters.zscore_cutoff
        mean = pd.Series(self._params["mean"], dtype="float64")
        std = pd.Series(self._params["std"], dtype="float64")
        std = std.where(std > 0)  # NaN for a zero or NaN std.
        return mean - k * std, mean + k * std
