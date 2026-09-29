"""ColumnOrder transform node for the NodeML Framework.

Enforces a deterministic column order on every DataFrame that flows
through this node. This is useful when downstream nodes (or an external
model) are sensitive to the physical ordering of columns — most
scikit-learn estimators and PyTorch models index by position rather than
by name, so an upstream reordering of features silently corrupts
predictions. Placing a ``ColumnOrder`` node right before such a consumer
pins the schema to a known order.

Two operating modes
-------------------
The mode is selected by ``running_config.column_order``:

1. **Explicit mode** — ``column_order`` is a list of strings:
   the list is treated as the source of truth. ``fit`` performs no
   learning (it only mirrors the configured list into the params so
   ``get_params`` returns the order actually applied), and every
   ``transform`` call reorders the input to match.

2. **Learned mode** — ``column_order`` is ``None`` (default):
   ``fit`` captures the column order of the input DataFrame and stores
   it in the node params. ``transform`` then replays that stored order.
   This is the natural choice for a training pipeline that should pin
   the schema observed at training time.

After a node is fitted in learned mode, switching the running config to
an explicit ``column_order`` overrides the learned order — the running
config always wins. This keeps the "enforce what the user asked for"
semantics unambiguous.

Enforcement semantics
---------------------
Let *order* be the list returned by mode resolution above.

* **Missing columns** (in *order* but not in the input DataFrame) always
  raise ``ValueError``, regardless of the ``strict`` flag.
* **Extra columns** (in the input DataFrame but not in *order*) are
  handled according to ``running_config.strict``:

  - ``strict=False`` (default): dropped from both the output DataFrame
    and the output :class:`TabularDataContext`.
  - ``strict=True``: trigger ``ValueError`` — the input must contain
    exactly the columns of *order*.

* **Column order of the output** is exactly *order*, in that order —
  this is the whole point of the node.
* **Context consistency** — the output ``TabularDataContext`` has any
  dropped columns removed so that ``columns``, ``dtypes``, and
  ``categories`` remain aligned with the output DataFrame.

Serialisation
-------------
``get_params`` / ``set_params`` expose ``{"column_order": [...]}``. In
learned mode the stored list is what ``fit`` captured; in explicit mode
it mirrors the running-config list. The node is safe to checkpoint and
restore via the standard pipeline param save/load path.
"""

from copy import deepcopy
from typing import Any, cast

import pandas as pd
from pydantic import Field

from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    TabularDataContext,
)
from nodeml.core.nodes.node import Port
from nodeml.core.nodes.transform.transform import (
    TransformConfig,
    TransformHyperParameters,
    TransformMetadata,
    TransformNode,
    TransformRunningConfig,
)

# Serialisable params: the captured column order (used when not supplied
# via running_config).
type _ColumnOrderParams = dict[str, list[str]]


class ColumnOrderMetadata(TransformMetadata):
    """Metadata for the ColumnOrder node."""

    node_name: str = "ColumnOrder"
    description: str = (
        "Enforce a deterministic column order on the input DataFrame. "
        "Either uses a user-supplied order from the running config, or "
        "captures the input order at fit time and replays it at transform."
    )
    trainable: bool = True


class ColumnOrderRunningConfig(TransformRunningConfig):
    """Run-time knobs for the ColumnOrder node."""

    column_order: list[str] | None = Field(
        default=None,
        description=(
            "Explicit column order to enforce. When set, this list is "
            "used as-is on every transform call and fit does not learn "
            "anything. When ``None`` (default), fit captures the input's "
            "column order and transform replays it."
        ),
    )
    strict: bool = Field(
        default=False,
        description=(
            "When ``True``, the input DataFrame must contain exactly the "
            "columns of the enforced order — extras trigger a ``ValueError``. "
            "When ``False`` (default), extra columns are silently dropped "
            "from the output. Missing columns always raise regardless of "
            "this flag."
        ),
    )


class ColumnOrderHyperParameters(TransformHyperParameters):
    """No learnable hyperparameters."""


class ColumnOrderConfig(
    TransformConfig[
        ColumnOrderRunningConfig,
        ColumnOrderHyperParameters,
    ]
):
    """Full configuration for the ColumnOrder node."""

    hyperparameters: ColumnOrderHyperParameters = Field(
        default_factory=ColumnOrderHyperParameters,
        description="No tuneable hyperparameters for this node.",
    )
    running_config: ColumnOrderRunningConfig = Field(
        default_factory=ColumnOrderRunningConfig,
        description="Run-time options (column_order, strict).",
    )
    in_ports: dict[str, Port] = Field(
        default={
            "input": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.MIXED,
                data_shape="batch feature",
                desc="DataFrame whose columns should be ordered.",
            ),
        },
        description="Input ports: 'input' (DataFrame to reorder).",
    )
    out_ports: dict[str, Port] = Field(
        default={
            "output": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.MIXED,
                data_shape="batch _",
                desc=(
                    "DataFrame whose columns match the enforced order. "
                    "Feature dimension may shrink if the input had extra "
                    "columns not in the target order."
                ),
            ),
        },
        description="Output ports: 'output' (ordered DataFrame).",
    )


class ColumnOrder(
    TransformNode[
        pd.DataFrame,
        TabularDataContext,
        pd.DataFrame,
        TabularDataContext,
        _ColumnOrderParams,
    ]
):
    """Enforce a deterministic column order on the input DataFrame.

    See the module docstring for a full description of the behaviour.

    Summary of modes
    ----------------
    * **Explicit** — ``running_config.column_order`` is a list: that list
      is enforced on every transform; fit only mirrors it into params.
    * **Learned** — ``running_config.column_order`` is ``None``: fit
      captures the input's column order; transform replays it.

    The running config always wins over learned params when both are set.

    Summary of enforcement
    ----------------------
    * Missing columns (in the target order but not in the input):
      always raise ``ValueError``.
    * Extra columns (in the input but not in the target order):

      - ``strict=False`` (default): dropped from the output DataFrame
        and context.
      - ``strict=True``: raise ``ValueError``.

    The output DataFrame always has exactly the columns of the target
    order, in that order, and the output context is kept consistent.

    Examples
    --------
    Explicit mode — enforce a fixed schema regardless of input order::

        cfg = ColumnOrderConfig(
            running_config=ColumnOrderRunningConfig(
                column_order=["a", "b", "c"],
            ),
        )
        node = ColumnOrder(config=cfg)

    Learned mode — pin the training-time schema and replay at inference::

        cfg = ColumnOrderConfig()  # column_order=None, strict=False
        node = ColumnOrder(config=cfg)
        node.fit({"input": (train_df, train_ctx)})
        # subsequent transform() calls return DataFrames whose columns
        # match train_df.columns in order.

    Strict mode — fail fast on unexpected columns::

        cfg = ColumnOrderConfig(
            running_config=ColumnOrderRunningConfig(
                column_order=["a", "b", "c"],
                strict=True,
            ),
        )

    """

    metadata = ColumnOrderMetadata()

    def __init__(self, *, config: ColumnOrderConfig) -> None:
        """Initialise the node with its configuration."""
        self._config = config
        self._params: _ColumnOrderParams = {"column_order": []}
        self._fitted = False

    # --- TransformNode interface ------------------------------------------

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Record the column order to enforce at transform time.

        Behaviour depends on the running config:

        * If ``column_order`` is set, the params mirror that list —
          ``fit`` does not inspect the input's own column order. This
          keeps ``get_params`` returning the order that is actually
          applied regardless of the input seen at fit time.
        * If ``column_order`` is ``None``, the params capture the input
          DataFrame's column order (``df.columns``) as the order to
          replay on future transforms.

        Parameters
        ----------
        data:
            Must contain key ``"input"`` (the DataFrame whose column
            order should be captured in learned mode). Only the
            DataFrame's columns attribute is read in learned mode; the
            data payload and context are otherwise ignored.

        """
        df, _ = data["input"]
        configured = self._config.running_config.column_order
        if configured is not None:
            self._params = {"column_order": list(configured)}
        else:
            self._params = {"column_order": list(df.columns)}

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Reorder the DataFrame columns to match the enforced order.

        Resolution of the target order: the running-config
        ``column_order`` wins if set, else the params captured at fit
        time are used. Then:

        1. Any column in the target order that is missing from the
           input raises ``ValueError``.
        2. Any extra column in the input is checked against the
           ``strict`` flag — ``strict=True`` raises, ``strict=False``
           drops it silently.
        3. The output DataFrame is ``df[target_order]`` (exactly the
           target order, in that order).
        4. The output context is a deep copy of the input context with
           the dropped columns removed so that its ``columns``,
           ``dtypes`` and ``categories`` stay aligned with the output
           DataFrame.

        Parameters
        ----------
        data:
            Must contain key ``"input"``.

        Returns
        -------
        dict
            ``{"output": (reordered_df, reordered_ctx)}``.

        Raises
        ------
        ValueError
            If the input is missing any column from the target order,
            or (when ``strict=True``) contains any column not in the
            target order.

        """
        df, ctx = data["input"]
        order = self._resolve_order()
        order_set = set(order)

        missing = [c for c in order if c not in df.columns]
        if missing:
            msg = (
                f"ColumnOrder: input DataFrame is missing required columns "
                f"{missing}. Present columns: {list(df.columns)}."
            )
            raise ValueError(msg)

        extras = [c for c in df.columns if c not in order_set]
        if extras and self._config.running_config.strict:
            msg = (
                f"ColumnOrder(strict=True): input DataFrame has unexpected "
                f"columns {extras}. Enforced order: {order}."
            )
            raise ValueError(msg)

        out_ctx = deepcopy(ctx)
        out_ctx.remove_columns(extras)
        return {"output": (cast("pd.DataFrame", df[order]), out_ctx)}

    def get_params(self) -> _ColumnOrderParams:
        """Return the enforced column order."""
        return self._params

    def set_params(self, params: _ColumnOrderParams) -> None:
        """Restore a previously captured column order (checkpointing)."""
        self._params = params
        self._fitted = True

    # --- Private helpers --------------------------------------------------

    def _resolve_order(self) -> list[str]:
        """Return the column order to enforce at transform time.

        The running-config value always wins when set, even after a fit
        that captured a different order — this keeps the "enforce what
        the user asked for" semantics unambiguous.
        """
        configured = self._config.running_config.column_order
        if configured is not None:
            return list(configured)
        return list(self._params["column_order"])


# No hyperparameters to tune; exposed as an empty dict for consistency.
hyperparameter_space: dict[str, Any] = {}
