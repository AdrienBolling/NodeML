"""Structured JSON logger for the NodeML Framework.

Provides :class:`Logger`, a thin wrapper around :mod:`logging` that produces
JSON-serializable log records suitable for database export.  The wrapper
does **not** inherit from :class:`logging.Logger` -- it delegates to one.

Design principles
-----------------
* **Silent by default** - the ``nodeml`` root logger ships with a
  :class:`logging.NullHandler`.  Users opt in by calling :func:`configure`
  or by attaching their own handler.
* **Structured context** - every log record carries a JSON-serializable
  ``context`` dict assembled from *bound* fields (via :meth:`Logger.bind`)
  and per-call *kwargs*.
* **Library helpers** - convenience methods for recurring pipeline patterns
  (node execution, phase transitions, data flow, fit completion).

Standard kwargs
---------------
All kwargs are optional on every call.  They are merged into the record's
``context`` dict (per-call kwargs override bound context on collision).

Pipeline context (typically bound once per scope):

* ``node_name``       - human name of the node, e.g. ``"RandomForestRegressor"``
* ``node_type``       - :class:`~nodeml.core.nodes.node.NodeType` value
* ``pipeline_phase``  - :class:`~nodeml.core.common.enums.NodeExecutionMode` value
* ``port_name``       - port involved in the current operation

Data context (useful between nodes, not enforced inside them):

* ``data_structure``  - :class:`~nodeml.core.common.data.data.DataStructureEnum` value
  (e.g. ``"TabularData"``)
* ``data_category``   - :class:`~nodeml.core.common.data.data.DataCategoryEnum` value
  (e.g. ``"numerical_data"``, ``"mixed_data"``)
* ``data_shape``      - tuple or string describing the array shape

Operational:

* ``duration_ms``     - wall-clock time in milliseconds
* ``params``          - arbitrary JSON-serializable dict for extra data
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Package-level NullHandler -- keeps the library silent until the user
# explicitly configures logging.
# ---------------------------------------------------------------------------

logging.getLogger("nodeml").addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class Logger:
    """Structured JSON logger for the NodeML framework.

    Wraps a standard :class:`logging.Logger` and enriches every record with
    a JSON-serializable ``context`` dict.

    Args:
        name: Dot-separated logger name (forwarded to :func:`logging.getLogger`).
            Convention: ``"nodeml.<module>"`` so that the ``nodeml`` root logger
            controls the entire library.
        **context: Initial bound context merged into every record produced by
            this instance.  See *Standard kwargs* above.

    Example:
        >>> log = Logger("nodeml.pipeline.runner", pipeline_phase="training")
        >>> log.info("Pipeline started")
        >>> node_log = log.bind(node_name="RandomForest", node_type="model")
        >>> node_log.info("Fit complete", duration_ms=142.3)

    """

    __slots__ = ("_context", "_logger")

    def __init__(self, name: str, **context: Any) -> None:
        """Create a new structured logger.

        Args:
            name: Dot-separated logger name forwarded to
                :func:`logging.getLogger`.
            **context: Initial bound context merged into every record.

        """
        self._logger = logging.getLogger(name)
        self._context: dict[str, Any] = context

    # ------------------------------------------------------------------
    # Context binding
    # ------------------------------------------------------------------

    def bind(self, **kwargs: Any) -> Logger:
        """Return a **new** logger with *kwargs* merged into the bound context.

        The parent logger and its underlying :class:`logging.Logger` are
        shared -- only the context dict is copied and extended.
        """
        # logging.getLogger returns the same underlying logger for a name.
        return Logger(self._logger.name, **{**self._context, **kwargs})

    def unbind(self, *keys: str) -> Logger:
        """Return a new logger with the given keys removed from context."""
        reduced = {k: v for k, v in self._context.items() if k not in keys}
        return Logger(self._logger.name, **reduced)

    # ------------------------------------------------------------------
    # Standard logging API
    # ------------------------------------------------------------------

    def debug(self, event: str, **kwargs: Any) -> None:
        """Log at DEBUG level."""
        self._log(logging.DEBUG, event, kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        """Log at INFO level."""
        self._log(logging.INFO, event, kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        """Log at WARNING level."""
        self._log(logging.WARNING, event, kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        """Log at ERROR level."""
        self._log(logging.ERROR, event, kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        """Log at CRITICAL level."""
        self._log(logging.CRITICAL, event, kwargs)

    def exception(self, event: str, exc: BaseException, **kwargs: Any) -> None:
        """Log at ERROR level with the full exception traceback attached.

        The traceback is captured via :mod:`traceback` and stored in
        ``context["error_trace"]`` as a string.  ``context["error_type"]``
        is set to the exception class name.
        """
        kwargs["error_type"] = type(exc).__qualname__
        kwargs["error_trace"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        self._log(logging.ERROR, event, kwargs)

    # ------------------------------------------------------------------
    # Library-specific helpers
    # ------------------------------------------------------------------

    def log_node_execution(
        self,
        node_name: str,
        phase: str,
        *,
        duration_ms: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Log that a node executed within a pipeline phase.

        Args:
            node_name: Human name of the node.
            phase: Execution phase (``"training"``, ``"inference"``, ``"evaluation"``).
            duration_ms: Wall-clock time in milliseconds. Defaults to ``None``.
            **kwargs: Additional context merged into the log record.

        """
        extra: dict[str, Any] = {
            "node_name": node_name,
            "pipeline_phase": phase,
        }
        if duration_ms is not None:
            extra["duration_ms"] = duration_ms
        extra.update(kwargs)
        self.info(f"Node executed: {node_name}", **extra)

    def log_phase(
        self,
        phase: str,
        status: str,
        **kwargs: Any,
    ) -> None:
        """Log a pipeline phase transition (start / end / error).

        Args:
            phase: Execution phase.
            status: ``"start"``, ``"end"``, or ``"error"``.
            **kwargs: Additional context merged into the log record.

        """
        self.info(
            f"Phase {phase} {status}",
            pipeline_phase=phase,
            phase_status=status,
            **kwargs,
        )

    def log_data_flow(
        self,
        source: str,
        target: str,
        *,
        data_structure: str | None = None,
        data_category: str | None = None,
        data_shape: tuple[int, ...] | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log data moving between two nodes.

        Args:
            source: Name of the source node.
            target: Name of the target node.
            data_structure: E.g. ``"TabularData"``. Defaults to ``None``.
            data_category: E.g. ``"numerical_data"``, ``"mixed_data"``. Defaults to ``None``.
            data_shape: Shape of the data being transferred. Defaults to ``None``.
            **kwargs: Additional context merged into the log record.

        """
        extra: dict[str, Any] = {"source": source, "target": target}
        if data_structure is not None:
            extra["data_structure"] = data_structure
        if data_category is not None:
            extra["data_category"] = data_category
        if data_shape is not None:
            extra["data_shape"] = (
                str(data_shape) if not isinstance(data_shape, str) else data_shape
            )
        extra.update(kwargs)
        self.debug(f"Data flow: {source} -> {target}", **extra)

    def log_fit(
        self,
        node_name: str,
        *,
        duration_ms: float | None = None,
        params_summary: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log that a node completed fitting.

        Args:
            node_name: Human name of the node.
            duration_ms: Wall-clock time in milliseconds. Defaults to ``None``.
            params_summary: Optional summary of fitted parameters (must be
                JSON-serializable). Defaults to ``None``.
            **kwargs: Additional context merged into the log record.

        """
        extra: dict[str, Any] = {"node_name": node_name}
        if duration_ms is not None:
            extra["duration_ms"] = duration_ms
        if params_summary is not None:
            extra["params"] = params_summary
        extra.update(kwargs)
        self.info(f"Fit complete: {node_name}", **extra)

    def log_node_call(
        self,
        node_name: str,
        phase: str,
        **kwargs: Any,
    ) -> None:
        """Log that a node is being called within a pipeline phase.

        Args:
            node_name: Human name of the node.
            phase: Execution phase (``"training"``, ``"inference"``, ``"evaluation"``).
            **kwargs: Additional context merged into the log record.

        """
        extra: dict[str, Any] = {
            "node_name": node_name,
            "pipeline_phase": phase,
        }
        extra.update(kwargs)
        self.debug(f"Node called: {node_name}", **extra)

    def log_trial(
        self,
        trial_id: str | int,
        status: Literal["start", "end", "error"],
        *,
        trial_config: dict[str, Any] | None = None,
        metric_name: str | None = None,
        metric_value: float | None = None,
        duration_ms: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Log a hyperparameter-tuning trial transition.

        Args:
            trial_id: Identifier of the trial (index, Ray trial id, ...).
            status: ``"start"``, ``"end"``, or ``"error"``.
            trial_config: Sampled hyperparameters for this trial.
            metric_name: Name of the reported metric (only on ``"end"``).
            metric_value: Scalar value of the reported metric.
            duration_ms: Wall-clock trial duration.
            **kwargs: Additional context merged into the log record.

        """
        extra: dict[str, Any] = {
            "trial_id": trial_id,
            "trial_status": status,
        }
        if trial_config is not None:
            extra["trial_config"] = trial_config
        if metric_name is not None:
            extra["metric_name"] = metric_name
        if metric_value is not None:
            extra["metric_value"] = metric_value
        if duration_ms is not None:
            extra["duration_ms"] = duration_ms
        extra.update(kwargs)
        level_fn = self.error if status == "error" else self.info
        level_fn(f"Trial {trial_id} {status}", **extra)

    # ------------------------------------------------------------------
    # Timing helper
    # ------------------------------------------------------------------

    @contextmanager
    def timed(
        self,
        event: str,
        *,
        level: int = logging.INFO,
        **context: Any,
    ) -> Iterator[Logger]:
        """Context manager that emits ``{event} start`` / ``end`` / ``error`` logs.

        On enter, logs ``{event} start`` at *level*.  On normal exit, logs
        ``{event} end`` at *level* with ``duration_ms`` attached.  On
        exception, logs ``{event} error`` at ``ERROR`` with the traceback
        and ``duration_ms``, then re-raises.

        The context manager yields a child :class:`Logger` with *context*
        bound, so callers can attach additional fields inside the block
        without duplicating kwargs on every line.

        Args:
            event: Human-readable name of the scope (e.g. ``"Tuning"``).
            level: Level for the ``start`` / ``end`` records.  Defaults to
                ``INFO``.
            **context: Fields bound to both the start/end log records and
                the yielded child logger.

        Yields:
            A child logger with *context* bound.

        Example:
            >>> with log.timed("Tuning", num_samples=8) as scoped:
            ...     scoped.debug("Sampled config", config=cfg)
            ...     run_tuning()

        """
        scoped = self.bind(**context) if context else self
        t0 = time.perf_counter()
        scoped._log(level, f"{event} start", {"phase_status": "start"})  # noqa: SLF001 - same class
        try:
            yield scoped
        except BaseException as exc:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            scoped.exception(
                f"{event} error",
                exc,
                phase_status="error",
                duration_ms=duration_ms,
            )
            raise
        duration_ms = (time.perf_counter() - t0) * 1000.0
        scoped._log(  # noqa: SLF001 - same class
            level,
            f"{event} end",
            {"phase_status": "end", "duration_ms": duration_ms},
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log(self, level: int, event: str, extra_kwargs: dict[str, Any]) -> None:
        """Build the merged context and emit a log record."""
        if not self._logger.isEnabledFor(level):
            return

        context = {**self._context, **extra_kwargs}
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=level,
            fn="",
            lno=0,
            msg=event,
            args=(),
            exc_info=None,
        )
        # Attach context so formatters (especially JSONFormatter) can read it.
        record.context = context  # type: ignore[attr-defined]
        record.event = event  # type: ignore[attr-defined]
        self._logger.handle(record)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return the logger name and its bound context."""
        return f"Logger(name={self._logger.name!r}, context={self._context!r})"


# ---------------------------------------------------------------------------
# JSONFormatter
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON objects.

    Each record is serialised as a JSON object with the following top-level
    keys:

    * ``timestamp`` -ISO-8601 UTC string.
    * ``level``     -log level name (``"DEBUG"``, ``"INFO"``, …).
    * ``event``     - the human-readable message passed to the logging call.
    * ``logger``    -dot-separated logger name.
    * ``context``   - the merged context dict attached by :class:`Logger`.

    Non-serialisable values inside *context* are coerced to their ``repr()``.

    Args:
        ensure_ascii: Forwarded to :func:`json.dumps`.  ``False`` (default)
            allows UTF-8 output; set to ``True`` if the downstream consumer
            expects pure ASCII.

    """

    def __init__(self, *, ensure_ascii: bool = False) -> None:
        """Initialize the JSON formatter.

        Args:
            ensure_ascii: Forwarded to :func:`json.dumps`. Defaults to ``False``.

        """
        super().__init__()
        self._ensure_ascii = ensure_ascii

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        context: dict[str, Any] = getattr(record, "context", {})
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", record.getMessage()),
            "logger": record.name,
            "context": self._safe_serialise(context),
        }
        return json.dumps(payload, ensure_ascii=self._ensure_ascii, default=str)

    # ------------------------------------------------------------------

    @staticmethod
    def _safe_serialise(obj: Any) -> Any:
        """Recursively ensure *obj* is JSON-serialisable.

        Dicts and lists are traversed; everything else is returned as-is
        (``json.dumps`` will use the ``default=str`` fallback for
        non-serialisable leaves).
        """
        if isinstance(obj, dict):
            return {k: JSONFormatter._safe_serialise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [JSONFormatter._safe_serialise(v) for v in obj]
        return obj


# ---------------------------------------------------------------------------
# TextFormatter
# ---------------------------------------------------------------------------


class TextFormatter(logging.Formatter):
    """Human-readable single-line formatter for terminal / notebook output.

    Produces lines like::

        12:34:56 INFO     [nodeml.runner.smart] Phase training start  pipeline_name=My Pipeline phase_status=start

    Args:
        show_timestamp: Include an ``HH:MM:SS`` timestamp prefix.
            Defaults to ``True``.
        use_color: Colorise the log level with ANSI codes.  Defaults to
            ``True``; disable when piping to a file or on Windows terminals
            that do not support ANSI.

    """

    _COLORS: dict[int, str] = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[35m",  # magenta
    }
    _RESET = "\033[0m"

    def __init__(
        self,
        *,
        show_timestamp: bool = True,
        use_color: bool = True,
    ) -> None:
        """Initialize the text formatter.

        Args:
            show_timestamp: Include a timestamp prefix. Defaults to ``True``.
            use_color: Use ANSI colour codes. Defaults to ``True``.

        """
        super().__init__()
        self._show_timestamp = show_timestamp
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        parts: list[str] = []

        # Timestamp
        if self._show_timestamp:
            ts = datetime.fromtimestamp(record.created, tz=UTC)
            parts.append(ts.strftime("%H:%M:%S"))

        # Level
        level = record.levelname.ljust(8)
        if self._use_color:
            color = self._COLORS.get(record.levelno, "")
            level = f"{color}{level}{self._RESET}"
        parts.append(level)

        # Logger name
        parts.append(f"[{record.name}]")

        # Event
        event = getattr(record, "event", record.getMessage())
        parts.append(event)

        # Context as key=value pairs
        context: dict[str, Any] = getattr(record, "context", {})
        if context:
            kv = "  ".join(f"{k}={v}" for k, v in context.items())
            parts.append(f" {kv}")

        return " ".join(parts)


# ---------------------------------------------------------------------------
# configure() convenience function
# ---------------------------------------------------------------------------

# Formatter types that configure() manages -- used for idempotent cleanup.
_MANAGED_FORMATTERS = (JSONFormatter, TextFormatter)


def configure(
    *,
    level: int | str = logging.INFO,
    filepath: str | Path | None = None,
    stream: Any | None = None,
    fmt: Literal["json", "text"] = "json",
    ensure_ascii: bool = False,
    show_timestamp: bool = True,
    use_color: bool = True,
    logger_name: str = "nodeml",
) -> logging.Logger:
    """One-call setup for the ``nodeml`` logging hierarchy.

    Attaches a formatter to the chosen handler and sets the log level on
    the ``nodeml`` root logger.  Can be called multiple times -- each call
    only replaces the handler whose formatter matches *fmt*, so a text
    handler (stdout) and a JSON handler (file) can coexist.

    Args:
        level: Logging level (``logging.DEBUG``, ``"INFO"``, etc.).
        filepath: If given, logs are written to this file (append mode).
        stream: Writable stream (e.g. ``sys.stdout``).  Used when *filepath*
            is ``None``.  Defaults to ``sys.stderr`` when both are ``None``.
        fmt: ``"json"`` for :class:`JSONFormatter` (default, suitable for
            database export) or ``"text"`` for :class:`TextFormatter`
            (human-readable, suitable for terminals and notebooks).
        ensure_ascii: Forwarded to :class:`JSONFormatter` (ignored when *fmt*
            is ``"text"``).
        show_timestamp: Forwarded to :class:`TextFormatter` (ignored when
            *fmt* is ``"json"``).
        use_color: Forwarded to :class:`TextFormatter` (ignored when *fmt* is
            ``"json"``).
        logger_name: Name of the logger to configure.  Defaults to ``"nodeml"``
            (the library root).

    Returns:
        The configured standard-library ``logging.Logger``.

    Example:
        **Notebook / terminal** (human-readable to stdout)::

            from nodeml.core.common.logging import configure
            configure(level="DEBUG", stream=sys.stdout, fmt="text")

        **Log file** (JSON lines for database export)::

            configure(level="DEBUG", filepath="pipeline.log")

        **Both at the same time** (call twice -- they don't conflict)::

            configure(level="DEBUG", stream=sys.stdout, fmt="text")
            configure(level="DEBUG", filepath="pipeline.log", fmt="json")

    """
    root = logging.getLogger(logger_name)

    # Determine which formatter class this call manages.
    formatter_cls: type[logging.Formatter] = (
        TextFormatter if fmt == "text" else JSONFormatter
    )

    # Remove handlers previously added by configure() for the *same* fmt
    # (idempotency per formatter type).
    for h in list(root.handlers):
        if isinstance(h.formatter, formatter_cls):
            root.removeHandler(h)
            h.close()

    # Build the handler.
    if filepath is not None:
        handler: logging.Handler = logging.FileHandler(
            str(filepath), mode="a", encoding="utf-8"
        )
    else:
        handler = logging.StreamHandler(stream or sys.stderr)

    # Build the formatter.
    if fmt == "text":
        handler.setFormatter(
            TextFormatter(show_timestamp=show_timestamp, use_color=use_color)
        )
    else:
        handler.setFormatter(JSONFormatter(ensure_ascii=ensure_ascii))

    root.addHandler(handler)
    root.setLevel(level)
    return root
