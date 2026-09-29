"""Define the atomic component of a NodeML Pipeline: the Node."""

import uuid
from abc import ABC, ABCMeta, abstractmethod
from enum import StrEnum
from functools import wraps
from typing import Any, override

from pydantic import BaseModel, PrivateAttr

from nodeml.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
)
from nodeml.core.common.enums import NodeExecutionMode
from nodeml.core.common.mixins.mixin import MixinSettings


class NodeMetadata(BaseModel):
    """Base metadata container for all node types in a NodeML Pipeline.

    Subclasses (e.g., ModelMetadata, TransformMetadata) extend this to carry
    node-type-specific metadata such as trainability flags.
    """


class NodeType(StrEnum):
    """Enumeration of node types available in a NodeML Pipeline.

    Attributes:
        BASE: Generic base node with no specialised behaviour.
        SOURCE: Data source node that produces data for the pipeline.
        SINK: Terminal node that consumes data without producing output.
        TRANSFORM: Stateless or stateful data transformation node.
        MODEL: Trainable model node (fit/predict pattern).
        METRIC: Evaluation metric node (update/compute accumulator pattern).

    """

    BASE = "base"
    SOURCE = "source"
    SINK = "sink"
    TRANSFORM = "transform"
    MODEL = "model"
    METRIC = "metric"


class Port(BaseModel):
    """Describes an input or output port on a NodeML Node.

    Ports define the data contract between connected nodes in a pipeline,
    specifying the expected array type, structure, category, and shape.

    Attributes:
        arr_type: Array backend this port accepts or produces (e.g. pandas DataFrame,
            numpy ndarray).
        data_structure: Logical data structure (e.g. tabular, time-series).
            Defaults to ``DataStructureEnum.DATA``.
        data_category: Semantic category of the data (e.g. numerical, categorical,
            mixed).
        data_shape: Shape descriptor for jaxtyping validation. See
            https://docs.kidger.site/jaxtyping/api/array/#shape for the format.
        optional: If ``True``, the port may be left unconnected in the pipeline
            and the node must handle that gracefully. Defaults to ``False``.
        desc: Human-readable description of this port's purpose.
        mode: Pipeline execution modes in which this port is active.
            Defaults to ``["all"]``.

    """

    arr_type: ArrayLikeEnum
    data_structure: DataStructureEnum = DataStructureEnum.DATA
    data_category: DataCategoryEnum
    data_shape: str
    optional: bool = False
    desc: str
    mode: list[NodeExecutionMode] = [NodeExecutionMode.ALL]


class NodeConfig(BaseModel):
    """Configuration for a Node in a NodeML Pipeline.

    Each ``NodeConfig`` instance is assigned a unique UUID on creation so that
    two nodes sharing identical configuration values are still considered
    distinct entities inside a pipeline.

    Attributes:
        _id: Auto-generated UUID that uniquely identifies this config instance.
        node_type: The type of node this config belongs to. Defaults to
            ``NodeType.BASE``.
        in_ports: Mapping of port names to ``Port`` definitions for inputs.
        out_ports: Mapping of port names to ``Port`` definitions for outputs.
        mixins_config: Per-mixin settings keyed by mixin name.

    """

    _id: uuid.UUID = PrivateAttr(default_factory=uuid.uuid4)
    node_type: NodeType = NodeType.BASE
    in_ports: dict[str, Port] = {}
    out_ports: dict[str, Port] = {}
    mixins_config: dict[str, MixinSettings] = {}

    @override
    def __hash__(self) -> int:
        """Hash the NodeConfig based on its UUID. Assume that two Nodes with the same config can be considered different."""
        return hash(self._id)

    @property
    def id(self) -> uuid.UUID:
        """Property to get the unique identifier of the NodeConfig."""
        return self._id


class MetaPostInitHook(ABCMeta):
    """Metaclass that automatically invokes ``__post_init__`` after ``__init__``.

    This mirrors the dataclass ``__post_init__`` pattern for regular classes.
    Any class using this metaclass can define a ``__post_init__`` method that
    will be called with the same arguments right after ``__init__`` completes.
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Create an instance and call its ``__post_init__`` hook if defined.

        Args:
            *args: Positional arguments forwarded to ``__init__``.
            **kwargs: Keyword arguments forwarded to ``__init__``.

        Returns:
            The fully initialised instance.

        """
        # Before we do __new__ and __init__
        instance: object = super().__call__(*args, **kwargs)
        # After we do __post_init__
        post_hook = getattr(instance, "__post_init__", None)
        if callable(post_hook):
            _ = post_hook(*args, **kwargs)
        return instance


class Node[D_I, D_C_I, D_O, D_C_O](ABC, metaclass=MetaPostInitHook):
    """Base class for all nodes in a NodeML Pipeline.

    A node is the atomic processing unit of a pipeline. It receives data on
    its input ports, optionally fits internal state, and produces output on
    its output ports.

    The class is generic over four type parameters that define the data
    contract:

    * ``D_I`` -- data input type (e.g. ``pd.DataFrame``).
    * ``D_C_I`` -- data context input type carrying metadata alongside input.
    * ``D_O`` -- data output type.
    * ``D_C_O`` -- data context output type.

    Subclasses must implement :meth:`node_fit` and :meth:`node_transform`.
    """

    _is_node: bool = True
    metadata = NodeMetadata()
    # Class-level default: subclasses often do not call Node.__init__.
    _execution_mode: NodeExecutionMode = NodeExecutionMode.DEFAULT

    def __init__(self, *, config: NodeConfig) -> None:
        """Initialise the Node with the given configuration.

        Args:
            config: Node configuration defining ports, type, and mixins.

        """
        self._config = config
        self._execution_mode = NodeExecutionMode.DEFAULT

    @property
    def execution_mode(self) -> NodeExecutionMode:
        """Get the current execution mode of the node."""
        return self._execution_mode

    @property
    def in_ports(self) -> dict[str, Port]:
        """Get the input ports of the Node."""
        return self._config.in_ports

    @property
    def out_ports(self) -> dict[str, Port]:
        """Get the output ports of the Node."""
        return self._config.out_ports

    @property
    def node_type(self) -> NodeType:
        """Get the type of the Node."""
        return self._config.node_type

    @property
    def config(self) -> NodeConfig:
        """Get the configuration of the Node."""
        return self._config

    def set_execution_mode(self, mode: NodeExecutionMode) -> None:
        """Set the execution mode of the node.

        Args:
            mode: The execution mode to set for the node.

        """
        self._execution_mode = mode

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate subclass definition and wrap its ``__init__``.

        Wraps ``__init__`` to extract the ``config`` kwarg early and validates
        that Mixin classes appear *after* the Node base in the MRO to ensure
        proper initialisation order.

        Raises:
            TypeError: If ``config`` is not passed as a keyword argument, or if
                a Mixin class appears before the Node base in the inheritance list.

        """
        super().__init_subclass__(**kwargs)

        # Wrap only an __init__ that this class defines.  An inherited
        # __init__ is already wrapped by the parent class.
        if "__init__" in cls.__dict__:
            orig_init = cls.__init__

            @wraps(orig_init)
            def wrapped_init(self: "Node", *args: Any, **kwargs: Any) -> None:
                if "config" not in kwargs:
                    msg = "config must be passed as a keyword argument"
                    raise TypeError(msg)
                self._config = kwargs["config"]
                orig_init(self, *args, **kwargs)

            cls.__init__ = wrapped_init

        bases = cls.__bases__
        base_idx = None
        for idx, base in enumerate(bases):
            if getattr(base, "_is_node", False):
                base_idx = idx
        if base_idx is None:
            return

        for idx, base in enumerate(bases):
            if getattr(base, "_is_mixin", False) and idx < base_idx:
                message = f"Mixin class {base.__name__} must be placed after the base class in the inheritance order.\n"
                message += f"Use 'class {cls.__name__}({bases[base_idx].__name__}, {base.__name__}):' instead of 'class {cls.__name__}({base.__name__}, {bases[base_idx].__name__}):'."
                raise TypeError(message)

    def __post_init__(
        self, *, config: NodeConfig | None = None, **_kwargs: Any
    ) -> None:
        """Run common post-initialisation logic for all Nodes.

        Propagates initialisation up the MRO so that Mixin ``__init__`` and
        ``__post_init__`` methods run after the node is initialised.

        Args:
            config: Optional node configuration forwarded along the MRO chain.
            **_kwargs: Other ``__init__`` arguments of the subclass.  They are
                not forwarded to mixins.

        """
        # Call the next __init__ in the MRO (a mixin, if any).  object.__init__
        # takes no arguments, so it gets none.  Errors raised by a mixin are
        # not hidden.
        if self._next_init_owner() is object:
            super().__init__()
        else:
            super().__init__(config=config)  # type: ignore[call-arg]
        # Mixins can define their own __post_init__.
        if getattr(super(), "__post_init__", None):
            super().__post_init__(config=config)  # type: ignore[misc]

    def _next_init_owner(self) -> type:
        """Return the first class after :class:`Node` in the MRO that defines ``__init__``."""
        mro = type(self).__mro__
        for klass in mro[mro.index(Node) + 1 :]:
            if "__init__" in klass.__dict__:
                return klass
        return object

    @property
    def id(self) -> uuid.UUID:
        """Get the unique identifier of the Node based on its configuration."""
        return self._config.id

    @abstractmethod
    def node_fit(self, data: dict[str, tuple[D_I, D_C_I]]) -> None:
        """Fit the node's internal state using the provided data.

        May also be called with empty data to perform setup logic (e.g. for
        data-source nodes that establish connections).

        Args:
            data: Mapping of port name to ``(data, context)`` tuples.

        """
        raise NotImplementedError

    @abstractmethod
    def node_transform(
        self, data: dict[str, tuple[D_I, D_C_I]]
    ) -> dict[str, tuple[D_O, D_C_O]]:
        """Transform data through the node and produce output.

        Args:
            data: Mapping of port name to ``(data, context)`` tuples.

        Returns:
            Mapping of output port name to ``(data, context)`` tuples.

        """
        raise NotImplementedError

    def node_fit_transform(
        self, data: dict[str, tuple[D_I, D_C_I]]
    ) -> dict[str, tuple[D_O, D_C_O]]:
        """Fit the node then transform data in a single call.

        Convenience method that calls :meth:`node_fit` followed by
        :meth:`node_transform`.

        Args:
            data: Mapping of port name to ``(data, context)`` tuples.

        Returns:
            Mapping of output port name to ``(data, context)`` tuples.

        """
        self.node_fit(data)
        return self.node_transform(data)
