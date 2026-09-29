"""TabularCSVFetcher data-source node for the NodeML Framework.

Loads a CSV file and its companion context JSON file, validates their
consistency, and exposes the result as a single ``"output"`` port carrying
a ``pd.DataFrame`` + ``TabularDataContext``.

The files are read again at the start of each runner call, from the paths
of the current execution mode (see :class:`TabularCSVFetcherRunningConfig`).

The context JSON is expected to follow the ``TabularDataContext.dump_dict``
schema::

    {
        "columns":    ["col_a", "col_b", ...],
        "dtypes":     ["float64", "object", ...],
        "categories": ["numerical_data", "categorical_data", ...]
    }
"""

import json
from pathlib import Path

import pandas as pd
from pydantic import Field

from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    TabularDataContext,
    tabular_context_from_dict_dump,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.nodes.data_source.data_source import (
    DataSourceConfig,
    DataSourceMetadata,
    DataSourceNode,
    DataSourceRunningConfig,
)
from nodeml.core.nodes.node import Port


class TabularCSVFetcherMetadata(DataSourceMetadata):
    """Metadata for the TabularCSVFetcher node."""

    node_name: str = "TabularCSVFetcher"
    description: str = (
        "Load tabular data from a CSV file with its companion context JSON."
    )


class TabularCSVFetcherRunningConfig(DataSourceRunningConfig):
    """Run-time configuration for the TabularCSVFetcher.

    ``csv_path`` and ``context_path`` are used in every execution mode.  Set
    the ``inference_*`` or ``evaluation_*`` paths to read other files in
    that mode; a path left to ``None`` falls back to the default path.

    Attributes:
        csv_path: Path to the CSV file to load.
        context_path: Path to the JSON file containing the
            ``TabularDataContext`` metadata (columns, dtypes, categories).
        inference_csv_path: CSV file for the inference mode.
        inference_context_path: Context file for the inference mode.
        evaluation_csv_path: CSV file for the evaluation mode.
        evaluation_context_path: Context file for the evaluation mode.

    """

    csv_path: str = Field(
        default="",
        description="Path to the CSV file to load.",
    )
    context_path: str = Field(
        default="",
        description="Path to the JSON context file (columns, dtypes, categories).",
    )
    inference_csv_path: str | None = Field(
        default=None,
        description="CSV file for the inference mode. None uses csv_path.",
    )
    inference_context_path: str | None = Field(
        default=None,
        description="Context file for the inference mode. None uses context_path.",
    )
    evaluation_csv_path: str | None = Field(
        default=None,
        description="CSV file for the evaluation mode. None uses csv_path.",
    )
    evaluation_context_path: str | None = Field(
        default=None,
        description="Context file for the evaluation mode. None uses context_path.",
    )

    def paths_for(self, mode: NodeExecutionMode) -> tuple[str, str]:
        """Return the ``(csv_path, context_path)`` pair for *mode*."""
        if mode == NodeExecutionMode.INFERENCE:
            return (
                self.inference_csv_path or self.csv_path,
                self.inference_context_path or self.context_path,
            )
        if mode == NodeExecutionMode.EVALUATION:
            return (
                self.evaluation_csv_path or self.csv_path,
                self.evaluation_context_path or self.context_path,
            )
        return self.csv_path, self.context_path


class TabularCSVFetcherConfig(
    DataSourceConfig[TabularCSVFetcherRunningConfig],
):
    """Full configuration for the TabularCSVFetcher node."""

    running_config: TabularCSVFetcherRunningConfig = Field(
        default_factory=TabularCSVFetcherRunningConfig,
        description="Paths to the CSV and context files.",
    )
    in_ports: dict[str, Port] = Field(
        default={},
        description="No input ports — this is a pure data source.",
    )
    out_ports: dict[str, Port] = Field(
        default={
            "output": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.MIXED,
                data_shape="batch feature",
                desc="Tabular data loaded from the CSV file.",
            ),
        },
        description="Output ports: 'output' (tabular DataFrame).",
    )


class TabularCSVFetcher(
    DataSourceNode[None, None, pd.DataFrame, TabularDataContext],
):
    """Load tabular data from a CSV file paired with a context JSON.

    ``setup_source`` validates that both files exist and that the context
    is consistent with the CSV columns.  ``fetch_data`` reads them and
    returns the data on the ``"output"`` port.

    Example::

        >>> cfg = TabularCSVFetcherConfig(
    ...     running_config=TabularCSVFetcherRunningConfig(
    ...         csv_path="data/train.csv",
    ...         context_path="data/train_context.json",
    ...     ),
    ... )
    >>> node = TabularCSVFetcher(config=cfg)

    """

    metadata = TabularCSVFetcherMetadata()

    def __init__(self, *, config: TabularCSVFetcherConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config
        self._df: pd.DataFrame | None = None
        self._context: TabularDataContext | None = None

    # --- DataSourceNode interface --------------------------------------------

    def setup_source(self) -> None:
        """Load the CSV and context files of the current execution mode.

        Raises:
            FileNotFoundError: If a file does not exist.
            ValueError: If the context does not match the CSV columns.

        """
        csv_str, ctx_str = self._config.running_config.paths_for(self.execution_mode)
        csv_path = Path(csv_str)
        ctx_path = Path(ctx_str)

        if not csv_path.is_file():
            msg = f"CSV file not found: {csv_path}"
            raise FileNotFoundError(msg)
        if not ctx_path.is_file():
            msg = f"Context JSON file not found: {ctx_path}"
            raise FileNotFoundError(msg)

        # Load both files
        self._df = pd.read_csv(csv_path)
        with Path(ctx_path).open() as f:
            raw_context: dict[str, list[str]] = json.load(f)

        self._context = tabular_context_from_dict_dump(raw_context)
        self._validate_context_matches_data(self._df, self._context)

    def fetch_data(
        self, data: dict[str, tuple[None, None]] | None = None
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Return the loaded CSV data on the ``"output"`` port."""
        _ = data  # Unused for this data source
        if self._df is None or self._context is None:
            msg = "Data not loaded. Call setup_source() (via node_fit) first."
            raise RuntimeError(msg)
        return {"output": (self._df, self._context)}

    # --- Validation ----------------------------------------------------------

    @staticmethod
    def _validate_context_matches_data(
        df: pd.DataFrame,
        ctx: TabularDataContext,
    ) -> None:
        """Ensure the context JSON is consistent with the CSV data.

        Checks that the three context lists (columns, dtypes, categories) all
        have the same length, and that the context names the CSV columns in
        the same order.
        """
        n_data_cols = df.shape[1]
        n_ctx_columns = len(ctx.columns)
        n_ctx_dtypes = len(ctx.dtypes)
        n_ctx_categories = len(ctx.categories)

        # Internal consistency of the context file
        if not (n_ctx_columns == n_ctx_dtypes == n_ctx_categories):
            msg = (
                f"Context file is internally inconsistent: "
                f"columns ({n_ctx_columns}), dtypes ({n_ctx_dtypes}), "
                f"and categories ({n_ctx_categories}) must all have the same length."
            )
            raise ValueError(msg)

        # Context vs. data consistency
        if n_ctx_columns != n_data_cols:
            msg = (
                f"Context file does not match the CSV data: "
                f"context defines {n_ctx_columns} columns but the CSV has {n_data_cols}."
            )
            raise ValueError(msg)
        # Same count: the names must also match, in the same order.
        ctx.check_columns(df.columns)

    # --- Convenience ---------------------------------------------------------

    @property
    def running_config(self) -> TabularCSVFetcherRunningConfig:
        """Access the running configuration."""
        return self._config.running_config
