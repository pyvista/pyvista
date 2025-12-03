"""Wrappers for :vtk:`vtkDataObject`.

The data objects does not have any sort of spatial reference.

"""

from __future__ import annotations

import numpy as np

import pyvista as pv

from . import _vtk_core as _vtk
from .dataobject import DataObject
from .datasetattributes import DataSetAttributes
from .utilities.arrays import FieldAssociation
from .utilities.arrays import FieldLiteral
from .utilities.arrays import RowLiteral
from .utilities.arrays import get_array
from .utilities.arrays import row_array


class Table(DataObject, _vtk.vtkTable):
    """Wrapper for the :vtk:`vtkTable` class.

    Create by passing a 2D NumPy array of shape (``n_rows`` by ``n_columns``)
    or from a dictionary containing NumPy arrays.

    Examples
    --------
    >>> import pyvista as pv
    >>> import numpy as np
    >>> arrays = np.random.default_rng().random((100, 3))
    >>> table = pv.Table(arrays)

    """

    def __init__(self, *args, deep: bool = True, **kwargs):  # noqa: ARG002
        """Initialize the table."""
        super().__init__()
        if len(args) == 1:
            if isinstance(args[0], _vtk.vtkTable):
                if deep:
                    self.deep_copy(args[0])
                else:
                    self.shallow_copy(args[0])
            elif isinstance(args[0], (np.ndarray, list)):
                self._from_arrays(args[0])
            elif isinstance(args[0], dict):
                self._from_dict(args[0])
            elif 'pandas.core.frame.DataFrame' in str(type(args[0])):
                self._from_pandas(args[0])
            else:
                msg = f'Table unable to be made from ({type(args[0])})'
                raise TypeError(msg)

    @staticmethod
    def _prepare_arrays(arrays):
        arrays = np.asarray(arrays)
        if arrays.ndim == 1:
            return np.reshape(arrays, (1, -1))
        elif arrays.ndim == 2:
            return arrays.T
        else:
            msg = 'Only 1D or 2D arrays are supported by Tables.'
            raise ValueError(msg)

    def _from_arrays(self, arrays) -> None:
        np_table = self._prepare_arrays(arrays)
        for i, array in enumerate(np_table):
            self.row_arrays[f'Array {i}'] = array

    def _from_dict(self, array_dict):
        for array in array_dict.values():
            if not isinstance(array, np.ndarray) and array.ndim < 3:
                msg = 'Dictionary must contain only NumPy arrays with maximum of 2D.'
                raise ValueError(msg)
        for name, array in array_dict.items():
            self.row_arrays[name] = array

    def _from_pandas(self, data_frame) -> None:
        for name in data_frame.keys():
            self.row_arrays[name] = data_frame[name].values

    @property
    def n_rows(self):
        """Return the number of rows.

        Returns
        -------
        int
            The number of rows.

        """
        return self.GetNumberOfRows()

    @n_rows.setter
    def n_rows(self, n) -> None:
        """Set the number of rows.

        Parameters
        ----------
        n : int
            The number of rows.

        """
        self.SetNumberOfRows(n)

    @property
    def n_columns(self):
        """Return the number of columns.

        Returns
        -------
        int
            The number of columns.

        """
        return self.GetNumberOfColumns()

    @property
    def n_arrays(self):
        """Return the number of columns.

        Alias for: ``n_columns``.

        Returns
        -------
        int
            The number of columns.

        """
        return self.n_columns

    def _row_array(self, name=None):
        """Return row scalars of a vtk object.

        Parameters
        ----------
        name : str
            Name of row scalars to retrieve.

        Returns
        -------
        numpy.ndarray
            Numpy array of the row.

        """
        return self.row_arrays.get_array(name)

    @property
    def row_arrays(self):
        """Return the all row arrays.

        Returns
        -------
        int
            The all row arrays.

        """
        return DataSetAttributes(
            vtkobject=self.GetRowData(),
            dataset=self,  # type: ignore[arg-type]
            association=FieldAssociation.ROW,
        )

    def keys(self):
        """Return the table keys.

        Returns
        -------
        list
            List of the array names of this table.

        """
        return self.row_arrays.keys()

    def items(self):
        """Return the table items.

        Returns
        -------
        list
            List containing tuples pairs of the name and array of the table arrays.

        """
        return self.row_arrays.items()

    def values(self):
        """Return the table values.

        Returns
        -------
        list
            List of the table arrays.

        """
        return self.row_arrays.values()

    def update(self, data) -> None:
        """Set the table data using a dict-like update.

        Parameters
        ----------
        data : DataSetAttributes
            Other dataset attributes to update from.

        """
        if isinstance(data, (np.ndarray, list)):
            # Allow table updates using array data
            data = self._prepare_arrays(data)
            data = {f'Array {i}': array for i, array in enumerate(data)}
        self.row_arrays.update(data)
        self.Modified()

    def pop(self, name):
        """Pop off an array by the specified name.

        Parameters
        ----------
        name : int or str
            Index or name of the row array.

        Returns
        -------
        pyvista.pyvista_ndarray
            PyVista array.

        """
        return self.row_arrays.pop(name)

    def __getitem__(self, index):
        """Search row data for an array."""
        return self._row_array(name=index)

    def _ipython_key_completions_(self):
        return self.keys()

    def get(self, index):
        """Get an array by its name.

        Parameters
        ----------
        index : int or str
            Index or name of the row.

        Returns
        -------
        pyvista.pyvista_ndarray
            PyVista array.

        """
        return self[index]

    def __setitem__(self, name, scalars) -> None:
        """Add/set an array in the row_arrays."""
        self.row_arrays[name] = scalars

    def _remove_array(self, _, key) -> None:
        """Remove a single array by name from each field (internal helper)."""
        self.row_arrays.remove(key)

    def __delitem__(self, name) -> None:
        """Remove an array by the specified name."""
        del self.row_arrays[name]

    def __iter__(self):
        """Return the iterator across all arrays."""
        for array_name in self.row_arrays:
            yield self.row_arrays[array_name]

    def _get_attrs(self):
        """Return the representation methods."""
        attrs = []
        attrs.append(('N Rows', self.n_rows, '{}'))
        return attrs

    def _repr_html_(self):
        """Return a pretty representation for Jupyter notebooks.

        It includes header details and information about all arrays.

        """
        fmt = ''
        if self.n_arrays > 0:
            fmt += "<table style='width: 100%;'>"
            fmt += '<tr><th>Header</th><th>Data Arrays</th></tr>'
            fmt += '<tr><td>'
        # Get the header info
        fmt += self.head(display=False, html=True)
        # Fill out scalars arrays
        if self.n_arrays > 0:
            fmt += '</td><td>'
            fmt += '\n'
            fmt += "<table style='width: 100%;'>\n"
            titles = ['Name', 'Type', 'N Comp', 'Min', 'Max']
            fmt += '<tr>' + ''.join([f'<th>{t}</th>' for t in titles]) + '</tr>\n'
            row = '<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>\n'
            row = '<tr>' + ''.join(['<td>{}</td>' for i in range(len(titles))]) + '</tr>\n'

            def format_array(key):
                """Format array information for printing (internal helper)."""
                arr = row_array(self, key)
                dl, dh = self.get_data_range(key)
                dl = pv.FLOAT_FORMAT.format(dl)  # type: ignore[assignment]
                dh = pv.FLOAT_FORMAT.format(dh)  # type: ignore[assignment]
                ncomp = 0 if arr is None else arr.shape[1] if arr.ndim > 1 else 1
                dtype = None if arr is None else arr.dtype
                return row.format(key, dtype, ncomp, dl, dh)

            for i in range(self.n_arrays):
                key = self.GetRowData().GetArrayName(i)
                fmt += format_array(key)

            fmt += '</table>\n'
            fmt += '\n'
            fmt += '</td></tr> </table>'
        return fmt

    def __repr__(self):
        """Return the object representation."""
        return self.head(display=False, html=False)

    def __str__(self):
        """Return the object string representation."""
        return self.head(display=False, html=False)

    def to_pandas(self):
        """Create a Pandas DataFrame from this Table.

        Returns
        -------
        pandas.DataFrame
            This table represented as a pandas dataframe.

        """
        try:
            import pandas as pd  # noqa: PLC0415
        except ImportError:  # pragma: no cover
            msg = 'Install ``pandas`` to use this feature.'
            raise ImportError(msg)
        data_frame = pd.DataFrame()
        for name, array in self.items():
            data_frame[name] = array
        return data_frame

    def save(self, *args, **kwargs):  # pragma: no cover
        """Save the table."""
        msg = "Please use the `to_pandas` method and harness Pandas' wonderful file IO methods."
        raise NotImplementedError(msg)

    def get_data_range(  # type: ignore[override]
        self,
        arr: str | None = None,
        preference: FieldLiteral | RowLiteral = 'row',
    ) -> tuple[float, float]:
        """Get the min and max of a named array.

        Parameters
        ----------
        arr : str, numpy.ndarray, optional
            The name of the array to get the range. If ``None``, the active scalar
            is used.

        preference : str, optional
            When scalars is specified, this is the preferred array type
            to search for in the dataset.  Must be either ``'row'`` or
            ``'field'``.

        Returns
        -------
        tuple
            ``(min, max)`` of the array.

        """
        if arr is None:
            # use the first array in the row data
            arr = self.GetRowData().GetArrayName(0)
        if isinstance(arr, str):
            arr = get_array(self, arr, preference=preference)  # type: ignore[assignment]
        # If array has no tuples return a NaN range
        if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):  # type: ignore[attr-defined]
            return (np.nan, np.nan)
        # Use the array range
        return np.nanmin(arr), np.nanmax(arr)

    @property
    def is_empty(self) -> bool:  # numpydoc ignore=RT01
        """Return ``True`` if the table has no rows and no columns.

        .. versionadded:: 0.45

        Examples
        --------
        >>> import pyvista as pv
        >>> import numpy as np
        >>> table = pv.Table()
        >>> table.is_empty
        True

        >>> arrays = np.random.default_rng().random((100, 3))
        >>> table = pv.Table(arrays)
        >>> table.is_empty
        False

        """
        return self.n_rows == 0 and self.n_columns == 0
