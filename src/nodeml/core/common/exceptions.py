"""Custom exception classes for the NodeML framework.

Every class derives from :class:`NodeMLError`, so callers can catch all
framework errors with one ``except`` clause.  Most classes also derive from
the built-in exception that the framework raised before these classes
existed (``ValueError``, ``TypeError`` or ``RuntimeError``), so existing
``except ValueError`` clauses keep working.
"""


class NodeMLError(Exception):
    """Base class for all exceptions raised by the NodeML framework."""


# --- Registry exceptions ---
class RegistryError(NodeMLError, ValueError):
    """A registry lookup or registration failed (unknown or duplicate name)."""


# --- Pipeline exceptions ---
class PipelineError(NodeMLError):
    """Base class for all exceptions raised by the Pipeline."""


class PipelineValidationError(PipelineError, ValueError):
    """The pipeline layout or configuration is invalid."""


class PipelineGraphError(PipelineValidationError):
    """The pipeline graph is invalid (for example, it contains a cycle)."""


class PipelineCompilationError(PipelineError, ValueError):
    """The pipeline cannot be compiled in its current state."""


class PipelineExecutionError(PipelineError, RuntimeError):
    """A runner could not execute the pipeline."""


# --- Node exceptions ---
class NodeError(NodeMLError):
    """Base class for all exceptions raised by Nodes."""


class NodeConfigError(NodeError, ValueError):
    """A node configuration is invalid."""


class NodeInputError(NodeError, ValueError):
    """A node received missing, unknown or malformed inputs."""


class NodeNotFittedError(NodeError, ValueError):
    """A node was used for inference before it was fitted."""


class DataTypeError(NodeError, TypeError):
    """Data on a port does not match the type that the port declares."""


class DataContextError(NodeError, ValueError):
    """A data context does not match the data that it describes."""
