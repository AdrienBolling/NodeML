"""Module for the base types of data in the NodeML Framework."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import jaxtyping
import numpy as np
import pandas as pd
import torch
from pandas.api.types import (
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
    pandas_dtype,
)

from nodeml.core.common.exceptions import DataContextError


class Data:
    """Base class for all data types in the NodeML Framework."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the data."""
        msg = "Subclasses must implement the shape property."
        raise NotImplementedError(msg)

    @property
    def dtype(self) -> str:
        """Return the dtype of the data."""
        msg = "Subclasses must implement the dtype property."
        raise NotImplementedError(msg)


@dataclass
class DataContext:
    """Base class for all data contexts in the NodeML Framework."""


type ArrayLike = pd.DataFrame | np.ndarray | torch.Tensor


class ArrayLikeEnum(StrEnum):
    """Enum for array-like data types. This is used for the arr_type field in the Port model to specify the type of data array that a port accepts or outputs."""

    PANDAS = "pd.DataFrame"
    NUMPY = "np.ndarray"
    TORCH = "torch.Tensor"


ARRAY_MAPPING = {
    ArrayLikeEnum.PANDAS: pd.DataFrame,
    ArrayLikeEnum.NUMPY: np.ndarray,
    ArrayLikeEnum.TORCH: torch.Tensor,
}


# --- Jaxtyping things
class CategoricalData(jaxtyping.AbstractDtype):
    """Generic dtype for jaxtyping. This is used to bypass the dtype check in jaxtyping, since we want to allow any dtype for our data types, and we will handle dtype validation ourselves in the data classes.

    We will be using jaxtyping only for shape checking.
    """

    dtypes = ["categorical_data"]

    def __str__(self) -> str:
        """Represent the CategoricalData dtype as a string."""
        return "categorical_data"


class NumericalData(jaxtyping.AbstractDtype):
    """Generic dtype for jaxtyping. This is used to bypass the dtype check in jaxtyping, since we want to allow any dtype for our data types, and we will handle dtype validation ourselves in the data classes.

    We will be using jaxtyping only for shape checking.
    """

    dtypes = ["numerical_data"]

    def __str__(self) -> str:
        """Represent the NumericalData dtype as a string."""
        return "numerical_data"


class MixedData(jaxtyping.AbstractDtype):
    """Used to check if the category of data are compatible between nodes through jaxtyping."""

    dtypes = ["categorical_data", "numerical_data", "mixed_data"]

    def __str__(self) -> str:
        """Represent the MixedData dtype as a string."""
        return "mixed_data"


DataCategory = CategoricalData | NumericalData | MixedData


class DataCategoryEnum(StrEnum):
    """Enum for data categories. (used for pydantic model dumping and loading from json)."""

    CATEGORICAL = "categorical_data"
    NUMERICAL = "numerical_data"
    MIXED = "mixed_data"


DATA_CATEGORY_MAPPING: dict[DataCategoryEnum, type[DataCategory]] = {
    DataCategoryEnum.CATEGORICAL: CategoricalData,
    DataCategoryEnum.NUMERICAL: NumericalData,
    DataCategoryEnum.MIXED: MixedData,
}
INVERSE_DATA_CATEGORY_MAPPING: dict[type[DataCategory], DataCategoryEnum] = {
    v: k for k, v in DATA_CATEGORY_MAPPING.items()
}


def infer_category(dtype: object) -> type[DataCategory]:
    """Infer the data category of a column from its dtype.

    Numeric and boolean dtypes are numerical.  Categorical, object and
    string dtypes are categorical.  Any other dtype (for example a datetime)
    is mixed, which is the most permissive category.

    Args:
        dtype: A numpy or pandas dtype.

    Returns:
        The inferred category type.

    """
    if is_numeric_dtype(dtype):
        return NumericalData
    if (
        isinstance(dtype, pd.CategoricalDtype)
        or is_object_dtype(dtype)
        or is_string_dtype(dtype)
    ):
        return CategoricalData
    return MixedData


def _check_column_names(expected: Iterable[object], actual: Iterable[object]) -> None:
    """Raise if the context names *expected* differ from the data names *actual*."""
    expected_names = [str(col) for col in expected]
    actual_names = [str(col) for col in actual]
    if expected_names != actual_names:
        msg = (
            "The context does not match the data columns. "
            f"Context columns: {expected_names}. Data columns: {actual_names}. "
            "Build the output context with TabularDataContext.select() "
            "or TabularDataContext.aligned_to()."
        )
        raise DataContextError(msg)


@dataclass
class TabularDataContext(DataContext):
    """Column names, dtypes and data categories of a tabular batch.

    The three lists are aligned by position: entry ``i`` of each list
    describes column ``i`` of the data.  A node that changes the columns of
    its data must return a context with the same columns in the same order.
    Use :meth:`select` or :meth:`aligned_to` to build such a context by
    column name, so that the lists cannot drift from the data.

    Raises:
        DataContextError: If the three lists do not have the same length.

    """

    columns: list[str]
    dtypes: list[np.dtype]
    categories: list[type[DataCategory]]

    def __post_init__(self) -> None:
        """Check that the three lists describe the same number of columns."""
        if not (len(self.columns) == len(self.dtypes) == len(self.categories)):
            msg = (
                "TabularDataContext lists must have the same length, but got "
                f"{len(self.columns)} columns, {len(self.dtypes)} dtypes and "
                f"{len(self.categories)} categories."
            )
            raise DataContextError(msg)

    @property
    def dump_dict(self) -> dict[str, list[str]]:
        """Dump the TabularDataContext to a dictionary for serialization."""
        return {
            "columns": self.columns,
            "dtypes": [
                str(dtype) for dtype in self.dtypes
            ],  # We can only serialize the name of the dtype, since it's a type.
            "categories": [
                str(INVERSE_DATA_CATEGORY_MAPPING[cat]) for cat in self.categories
            ],  # We can only serialize the name of the category, since it's a type.
        }

    @property
    def dump_tuple(self) -> tuple[list[str], list[np.dtype], list[type[DataCategory]]]:
        """Dump the TabularDataContext to a tuple for serialization."""
        return (self.columns, self.dtypes, self.categories)

    def copy(self) -> "TabularDataContext":
        """Return a copy of the context with new lists."""
        return TabularDataContext(
            columns=list(self.columns),
            dtypes=list(self.dtypes),
            categories=list(self.categories),
        )

    def select(self, columns: Sequence[str]) -> "TabularDataContext":
        """Return a new context with only *columns*, in the given order.

        Args:
            columns: Names of the columns to keep.  The order of this
                sequence is the order of the new context.

        Returns:
            A new :class:`TabularDataContext`.

        Raises:
            DataContextError: If a name is not in the context.

        """
        positions = {name: i for i, name in enumerate(self.columns)}
        missing = [name for name in columns if name not in positions]
        if missing:
            msg = f"Columns {missing} are not in the context. Known columns: {self.columns}."
            raise DataContextError(msg)
        index = [positions[name] for name in columns]
        return TabularDataContext(
            columns=[self.columns[i] for i in index],
            dtypes=[self.dtypes[i] for i in index],
            categories=[self.categories[i] for i in index],
        )

    def aligned_to(self, df: pd.DataFrame) -> "TabularDataContext":
        """Return a context that describes *df* exactly.

        The new context has the columns of *df* in the order of *df*, and
        the dtypes of *df*.  The category of each column comes from this
        context when the column name is known, and is inferred from the
        dtype for a new column.

        Args:
            df: The DataFrame that the new context must describe.

        Returns:
            A new :class:`TabularDataContext`.

        """
        known = dict(zip(self.columns, self.categories, strict=True))
        columns = [str(col) for col in df.columns]
        dtypes = list(df.dtypes)
        categories = [
            known.get(col, infer_category(dtype))
            for col, dtype in zip(columns, dtypes, strict=True)
        ]
        return TabularDataContext(columns=columns, dtypes=dtypes, categories=categories)

    def check_columns(self, columns: Iterable[object]) -> None:
        """Check that the context lists exactly *columns*, in the same order.

        Args:
            columns: The column labels of the data that the context describes.

        Raises:
            DataContextError: If the names or their order differ.

        """
        _check_column_names(self.columns, columns)

    def remove_columns(self, columns_to_remove: list[str]) -> None:
        """Remove columns (and their dtypes/categories) from the context.

        Args:
            columns_to_remove: Names of the columns to drop. Columns not
                present in the context are silently ignored.

        """
        # Find the indexes of the columns to remove
        indexes_to_remove = [
            self.columns.index(col) for col in columns_to_remove if col in self.columns
        ]
        # Remove the columns, dtypes, and categories at the corresponding indexes
        self.columns = [
            col for i, col in enumerate(self.columns) if i not in indexes_to_remove
        ]
        self.dtypes = [
            dtype for i, dtype in enumerate(self.dtypes) if i not in indexes_to_remove
        ]
        self.categories = [
            cat for i, cat in enumerate(self.categories) if i not in indexes_to_remove
        ]


def tabular_context_from_dict_dump(
    dump_dict: dict[str, list[str]],
) -> TabularDataContext:
    """Reconstruct a :class:`TabularDataContext` from its serialized dict form.

    Args:
        dump_dict: Dictionary with ``"columns"``, ``"dtypes"``, and
            ``"categories"`` keys, as produced by
            :attr:`TabularDataContext.dump_dict`.

    Returns:
        A new :class:`TabularDataContext` instance.

    """
    return TabularDataContext(
        columns=dump_dict["columns"],
        # pandas_dtype also accepts pandas extension dtypes such as "category".
        dtypes=[pandas_dtype(dtype_str) for dtype_str in dump_dict["dtypes"]],
        categories=[
            DATA_CATEGORY_MAPPING[DataCategoryEnum(cat_str)]
            for cat_str in dump_dict["categories"]
        ],
    )


# INFO : For now there isn't any specifics that need to be defined across all data types. It is intentional, this type is only used to ensure all inputs/outputs come from the NodeML library.


class TabularData(Data):
    """Common data type for Tabular Data in the NodeML framework. This type is only supposed to be used by the Pipeline, it is not expected that the user interacts with it directly.

    Basically a wrapper around some pandas data, can be easily cast from and to several other data types.
    Schemas are enforced for the dimensions order using jaxtyping and beartypes.

    In particular, for tabular data, the order is (B, F) -> (Batch, Feature)

    For example to access the j-th feature of the i-th sample of a batch, we can do data[i, j].

    Any data is considered a batch, at worst a batch of 1 or 0 samples. This will make it easier to reshape and convert in the long run.

    Since in the background the data is a pandas dataframe, there is also the matter of colum names and dtypes. In cases where the target conversion type does not natively handle these, a tuple will be returned as follows :
    (data_array, column_names, dtypes). It is up to the user to preserve the order of columns, and pass them along as needed (the ColumnOrder transform node is the node that changes the column order).
    Finally, we explictely attach the category of each feature, this is gonna be useful because pandas tends to implicitely convert data types as it sees fit, and some model are not supposed to ingest categorical or numerical data.
    """

    _TABULAR_NDIM = 2  # (batch, feature)

    def __init__(
        self,
        data: ArrayLike,
        columns: list[str] | None,
        dtypes: list[np.dtype] | None,
        categories: list[type[DataCategory]] | None = None,
        *,
        infer_categories: bool = True,
    ) -> None:
        """Initialize the TabularData from an array-like source.

        Args:
            data: The raw data as a pandas DataFrame, NumPy array, or torch Tensor.
            columns: Column names corresponding to the feature axis.  For a
                DataFrame, the names must equal the DataFrame columns, in the
                same order; ``None`` skips this check.
            dtypes: Expected dtypes for each column.  A DataFrame keeps its
                own dtypes, so this argument is ignored for a DataFrame.
            categories: Per-column data category types. Required for NumPy/Tensor
                inputs.  For a DataFrame, ``None`` infers them from the dtypes
                when *infer_categories* is ``True``.
            infer_categories: If ``True`` and *categories* is ``None``, infer
                categories from the dtypes when *data* is a DataFrame.
                Defaults to ``True``.

        Raises:
            ValueError: If *data* is an unsupported type or if *categories* is
                missing for non-DataFrame inputs.
            DataContextError: If *columns* does not match the DataFrame columns.

        """
        self._infer_categories = infer_categories
        if isinstance(data, pd.DataFrame):
            if columns is not None:
                _check_column_names(columns, data.columns)
            # from_pandas may infer categories, so validate *after* conversion.
            self.from_pandas(data, categories)
        elif categories is None or columns is None or dtypes is None:
            msg = "Categories must be provided for numpy and torch data, together with columns and dtypes."
            raise ValueError(msg)
        elif isinstance(data, np.ndarray):
            self.from_numpy(data, columns, dtypes, categories)
        elif isinstance(data, torch.Tensor):
            self.from_tensor(data, columns, dtypes, categories)
        else:
            msg = f"Unsupported data type: {type(data)}"
            raise ValueError(msg)

    # --- Convenience API ---
    @property
    def data(
        self,
    ) -> tuple[
        pd.DataFrame | None,
        list[str] | None,
        list[np.dtype] | None,
        list[type[DataCategory]] | None,
    ]:
        """Return the underlying data."""
        return self._data, self._columns, self._dtypes, self._categories

    @property
    def is_initialized(self) -> bool:
        """Return whether the data is initialized."""
        return (
            self._data is not None
            and self._columns is not None
            and self._dtypes is not None
            and self._categories is not None
        )

    # --- Pipeline compatibility check methods ---

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the data."""
        return self._data.shape

    @property
    def dtype(self) -> str:
        """Return "mixed_data" if the data has mixed categories, otherwise return the category of the data."""
        if self._categories is None:
            msg = "Data is not initialized yet."
            raise ValueError(msg)
        if all(cat == self._categories[0] for cat in self._categories):
            return str(INVERSE_DATA_CATEGORY_MAPPING[self._categories[0]])
        return str(DataCategoryEnum.MIXED)

    @property
    def dtypes(self) -> list[np.dtype]:
        """Return the dtypes of the data."""
        return self._dtypes

    @property
    def columns(self) -> list[str]:
        """Return the column names of the data."""
        return self._columns

    @property
    def categories(self) -> list[type[DataCategory]]:
        """Return the categories of the data."""
        if self._categories is None:
            msg = "Data is not initialized yet."
            raise ValueError(msg)
        return self._categories

    @property
    def category(self) -> type[DataCategory]:
        """Return the category of the data.

        This is an aggregate, if all features are of the same category, return that category, otherwise return mixed.
        """
        if self._categories is None:
            msg = "Data is not initialized yet."
            raise ValueError(msg)
        if all(cat == self._categories[0] for cat in self._categories):
            return self._categories[0]
        return MixedData

    # --- Validation methods ---

    def _validate_data(
        self,
        data: ArrayLike | None,
        columns: list[str] | None,
        dtypes: list[np.dtype] | None,
        categories: list[type[DataCategory]] | None,
    ) -> bool:
        """Validate dimensional consistency of the data and its metadata.

        Args:
            data: The array-like data to validate (may be ``None``).
            columns: Column name list.
            dtypes: Dtype list.
            categories: Category type list.

        Returns:
            ``True`` if validation passes.

        Raises:
            ValueError: If the data is not 2-D, or if the lengths of
                *columns*, *dtypes*, or *categories* do not match the feature
                dimension of *data*.

        """
        # Check that the data is 2D, has column names, and that the dtypes can be inferred.
        if data is not None and data.ndim != self._TABULAR_NDIM:
            msg = f"Data must be {self._TABULAR_NDIM}D, but got {data.ndim}D."
            raise ValueError(msg)
        if columns is None:
            msg = "Columns must be provided."
            raise ValueError(msg)
        if dtypes is None:
            msg = "Dtypes must be provided."
            raise ValueError(msg)
        if categories is None:
            msg = "Categories must be provided."
            raise ValueError(msg)
        if data is not None:
            if len(columns) != data.shape[1]:
                msg = f"Number of columns must match data shape, but got {len(columns)} columns and data with shape {data.shape}."
                raise ValueError(msg)
            if len(dtypes) != data.shape[1]:
                msg = f"Number of dtypes must match data shape, but got {len(dtypes)} dtypes and data with shape {data.shape}."
                raise ValueError(msg)
            if len(categories) != data.shape[1]:
                msg = f"Number of categories must match data shape, but got {len(categories)} categories and data with shape {data.shape}."
                raise ValueError(msg)
        return True

    # --- Conversion FROM methods ---
    def from_pandas(
        self, data: pd.DataFrame, categories: list[type[DataCategory]] | None = None
    ) -> None:
        """Initialize the TabularData from a pandas DataFrame.

        Column names, dtypes, and (optionally) categories are extracted
        directly from *data*.  If *categories* is ``None`` and
        ``infer_categories`` was set at construction, categories are inferred
        from the column dtypes.

        Args:
            data: Source DataFrame with shape ``(batch, features)``.
            categories: Explicit per-column category types. If ``None``,
                categories are inferred when ``infer_categories`` is enabled.

        Raises:
            ValueError: If *categories* length does not match the number of columns.

        """
        columns = data.columns.tolist()
        dtypes = data.dtypes.tolist()

        if categories is None and self._infer_categories:
            # Explicit categories are more precise; inference is the fallback.
            categories = [infer_category(dtype) for dtype in dtypes]
        if categories is not None and len(categories) != len(columns):
            msg = (
                f"Length of categories ({len(categories)}) must match number "
                f"of columns ({len(columns)}): {columns}."
            )
            raise DataContextError(msg)

        self._validate_data(data, columns, dtypes, categories)

        self._data = data
        self._columns = columns
        self._dtypes = dtypes
        self._categories = categories

    def from_numpy(
        self,
        data: np.ndarray,
        columns: list[str],
        dtypes: list[np.dtype],
        categories: list[type[DataCategory]],
    ) -> None:
        """Initialize the TabularData from a NumPy array.

        The array is converted to a pandas DataFrame internally.

        Args:
            data: 2-D NumPy array with shape ``(batch, features)``.
            columns: Column names for each feature.
            dtypes: Target NumPy dtype per column.
            categories: Per-column data category types.

        Raises:
            ValueError: If dimensions or lengths are inconsistent.

        """
        self._validate_data(data, columns, dtypes, categories)
        self._data = pd.DataFrame(
            data,
            columns=columns,
        ).astype(
            dict(zip(columns, dtypes, strict=True))
        )  # Convert to pandas DataFrame and set column names and dtypes
        self._columns = columns
        self._dtypes = dtypes
        self._categories = categories

    def from_tensor(
        self,
        data: torch.Tensor,
        columns: list[str],
        dtypes: list[np.dtype],
        categories: list[type[DataCategory]],
    ) -> None:
        """Initialize the TabularData from a PyTorch tensor.

        The tensor is moved to CPU, converted to NumPy, then delegated to
        :meth:`from_numpy`.

        Args:
            data: 2-D tensor with shape ``(batch, features)``.
            columns: Column names for each feature.
            dtypes: Target NumPy dtype per column.
            categories: Per-column data category types.

        Raises:
            ValueError: If dimensions or lengths are inconsistent.

        """
        self.from_numpy(data.cpu().numpy(), columns, dtypes, categories)

    # --- Conversion TO methods ---
    def to_pandas(self) -> tuple[pd.DataFrame, TabularDataContext]:
        """Convert the TabularData to a pandas DataFrame.

        Returns:
            A tuple of ``(DataFrame, TabularDataContext)``.

        Raises:
            ValueError: If the data has not been initialized.

        """
        if not self.is_initialized:
            msg = "Data is not initialized yet."
            raise ValueError(msg)
        if (
            self._categories is None
        ):  # Typechecker stuff, could be removed without issues (tbi, do we really lose performance)
            msg = "Data is not initialized yet."
            raise ValueError(msg)
        return (
            self._data,
            TabularDataContext(self._columns, self._dtypes, self._categories),
        )

    def to_numpy(self) -> tuple[np.ndarray, TabularDataContext]:
        """Convert the TabularData to a NumPy array.

        Returns:
            A tuple of ``(ndarray, TabularDataContext)`` preserving column
            metadata alongside the raw array.

        Raises:
            ValueError: If the data has not been initialized.

        """
        if not self.is_initialized:
            msg = "Data is not initialized yet."
            raise ValueError(msg)
        if (
            self._categories is None
        ):  # Typechecker stuff, could be removed without issues
            msg = "Data is not initialized yet."
            raise ValueError(msg)
        return (
            self._data.to_numpy(),
            TabularDataContext(self._columns, self._dtypes, self._categories),
        )

    def to_tensor(self) -> tuple[torch.Tensor, TabularDataContext]:
        """Convert the TabularData to a PyTorch tensor.

        Returns:
            A tuple of ``(Tensor, TabularDataContext)`` preserving column
            metadata alongside the raw tensor.

        Raises:
            ValueError: If the data has not been initialized.

        """
        if not self.is_initialized:
            msg = "Data is not initialized yet."
            raise ValueError(msg)
        if (
            self._categories is None
        ):  # Typechecker stuff, could be removed without issues
            msg = "Data is not initialized yet."
            raise ValueError(msg)
        return (
            torch.from_numpy(self._data.to_numpy()),
            TabularDataContext(self._columns, self._dtypes, self._categories),
        )

    # --- Overloading basic functions for convenience ---
    def __str__(self) -> str:
        """Represent the TabularData as a pandas array with multi-index columns for column names, dtypes, and categories."""
        col_names = [str(col) for col in self._columns]
        dtype_names = [str(dt) for dt in self._dtypes]
        category_names = (
            [str(INVERSE_DATA_CATEGORY_MAPPING[cat]) for cat in self._categories]
            if self._categories is not None
            else ["unknown" for _ in col_names]
        )

        # Create a DataFrame only for display purposes
        df = pd.DataFrame(
            self._data.to_numpy(),
            columns=pd.MultiIndex.from_arrays(
                [col_names, dtype_names, category_names],
                names=["column", "dtype", "category"],
            ),
        )  # Display the DataFrame as HTML for better formatting in Jupyter notebooks
        return f"{self.__class__.__name__} (Shape: {self.shape})\n{df}"

    def _repr_html_(self) -> str:
        """Represent the TabularData as a pandas array with multi-index columns for column names, dtypes, and categories, in HTML format for better display in Jupyter notebooks."""
        col_names = [str(col) for col in self._columns]
        dtype_names = [str(dt) for dt in self._dtypes]
        category_names = (
            [str(INVERSE_DATA_CATEGORY_MAPPING[cat]) for cat in self._categories]
            if self._categories is not None
            else ["unknown" for _ in col_names]
        )

        # Create a DataFrame only for display purposes
        df = pd.DataFrame(
            self._data.to_numpy(),
            columns=pd.MultiIndex.from_arrays(
                [col_names, dtype_names, category_names],
                names=["column", "dtype", "category"],
            ),
        )
        html = f"<h3>{self.__class__.__name__} (Shape: {self.shape})</h3>"
        html += df._repr_html_()  # type: ignore[operator]
        return html


class DataStructureEnum(StrEnum):
    """Enum for data structures. This is used for the data_structure field in the Port model to specify the type of data structure that a port accepts or outputs."""

    DATA = "Data"
    TABULAR = "TabularData"


DATA_STRUCTURE_MAPPING = {
    DataStructureEnum.DATA: Data,
    DataStructureEnum.TABULAR: TabularData,
}

INVERSE_DATA_STRUCTURE_MAPPING = {v: k for k, v in DATA_STRUCTURE_MAPPING.items()}
