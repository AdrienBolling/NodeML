"""IQR (Inter-Quartile Range) outlier filter transform node for the NodeML Framework.

A value is an outlier when it is outside the Tukey fences
``[Q1 - k·IQR, Q3 + k·IQR]``, where *k* is a configurable multiplier.
Two strategies are available:

* ``"remove"`` - remove the rows with outliers (default), in the training
  mode only.
* ``"cap"``    - clip each outlier value to the fence, in all modes.

The filter learns the **Q1**, **Q3** and **IQR** of each column during
:meth:`fit` and uses them again at :meth:`transform` time.  See
:mod:`._outlier_filter` for the shared behaviour.

The standard Tukey fence uses ``k=1.5`` (mild outliers) or ``k=3.0``
(extreme outliers).
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


class IQROutlierFilterMetadata(TransformMetadata):
    """Metadata for the IQROutlierFilter node."""

    node_name: str = "IQROutlierFilter"
    description: str = (
        "Detect and handle outliers using the IQR (Tukey fence) method. "
        "A value is an outlier if it falls outside [Q1 - k·IQR, Q3 + k·IQR]."
    )


class IQROutlierFilterRunningConfig(OutlierFilterRunningConfig):
    """Run-time knobs that do not affect the learned parameters."""


class IQROutlierFilterHyperParameters(OutlierFilterHyperParameters):
    """Tuneable hyperparameters for the IQROutlierFilter."""

    iqr_multiplier: float = Field(
        default=1.5,
        gt=0.0,
        description=(
            "IQR multiplier *k*. "
            "``k=1.5`` flags mild outliers (Tukey's default); "
            "``k=3.0`` flags only extreme outliers."
        ),
    )


# Exposed at module level so external tuners can discover the search space.
hyperparameter_space: dict[str, Any] = {
    "iqr_multiplier": tune.uniform(0.5, 5.0),
    "threshold": tune.uniform(0.0, 1.0),
    "strategy": tune.choice(["remove", "cap"]),
}


class IQROutlierFilterConfig(
    TransformConfig[
        IQROutlierFilterHyperParameters,
        IQROutlierFilterRunningConfig,
    ]
):
    """Full configuration for the IQROutlierFilter node."""

    hyperparameters: IQROutlierFilterHyperParameters = Field(
        default_factory=IQROutlierFilterHyperParameters,
        description="Tuneable hyperparameters (iqr_multiplier, threshold, strategy).",
    )
    running_config: IQROutlierFilterRunningConfig = Field(
        default_factory=IQROutlierFilterRunningConfig,
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


class IQROutlierFilter(OutlierFilter):
    """IQR (Tukey fence) outlier filter.

    Learns the Q1, Q3 and IQR of each column on the training split.  Then
    uses these statistics to detect and handle outliers at transform time.
    Row removal happens only in the training mode.

    Example:
        >>> cfg = IQROutlierFilterConfig(
        ...     hyperparameters=IQROutlierFilterHyperParameters(
        ...         iqr_multiplier=3.0, strategy="cap"
        ...     ),
        ... )
        >>> node = IQROutlierFilter(config=cfg)

    """

    metadata = IQROutlierFilterMetadata()
    hyperparameter_space = hyperparameter_space

    def _fit_statistics(self, candidates: pd.DataFrame) -> OutlierFilterParams:
        """Return the Q1, Q3 and IQR of each column of *candidates*."""
        q1 = candidates.quantile(0.25)
        q3 = candidates.quantile(0.75)
        iqr = q3 - q1
        return {
            "q1": {col: float(q1[col]) for col in candidates.columns},
            "q3": {col: float(q3[col]) for col in candidates.columns},
            "iqr": {col: float(iqr[col]) for col in candidates.columns},
        }

    def _raw_bounds(self) -> tuple[pd.Series, pd.Series]:
        """Return the Tukey fences of each fitted column."""
        k = self._config.hyperparameters.iqr_multiplier
        q1 = pd.Series(self._params["q1"], dtype="float64")
        q3 = pd.Series(self._params["q3"], dtype="float64")
        iqr = pd.Series(self._params["iqr"], dtype="float64")
        return q1 - k * iqr, q3 + k * iqr
