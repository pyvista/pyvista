"""This module provides wrappers for vtkDataObjects: data onjects without any
sort of spatial reference.
"""
import collections
import vtk
import numpy as np


import pyvista
from pyvista.utilities import convert_array, row_scalar, raise_not_matching, ROW_DATA_FIELD, parse_field_choice


from .common import DataObject, _ScalarsDict


class Table(vtk.vtkTable, DataObject):
    """Wrapper for the ``vtkTable`` class"""
    def __init__(self, *args, **kwargs):
        self._row_bool_array_names = []


    @property
    def n_rows(self):
        return self.GetNumberOfRows()

    @n_rows.setter
    def n_rows(self, n):
        print('hmmmm')
        self.SetNumberOfRows(n)
        print(self.GetNumberOfRows())

    @property
    def n_columns(self):
        return self.GetNumberOfColumns()


    @property
    def n_arrays(self):
        return self.n_columns


    @property
    def row_arrays(self):
        """ Returns the all row arrays """
        pdata = self.GetRowData()
        narr = pdata.GetNumberOfArrays()

        # Update data if necessary
        if hasattr(self, '_row_arrays'):
            keys = list(self._row_arrays.keys())
            if narr == len(keys):
                if keys:
                    if self._row_arrays[keys[0]].shape[0] == self.n_rows:
                        return self._row_arrays
                else:
                    return self._row_arrays

        # dictionary with callbacks
        self._row_arrays = RowScalarsDict(self)

        for i in range(narr):
            name = pdata.GetArrayName(i)
            self._row_arrays[name] = self._row_scalar(name)

        self._row_arrays.enable_callback()
        return self._row_arrays


    def keys(self):
        return self.row_arrays.keys()


    def items(self):
        return self.row_arrays.items()

    def update(self, data):
        self.row_arrays.update(data)


    def pop(self, name):
        """Pops off an array by the specified name"""
        return self.row_arrays.pop(name)


    def _add_row_scalar(self, scalars, name, deep=True):
        """
        Adds scalars to the vtk object.

        Parameters
        ----------
        scalars : numpy.ndarray
            Numpy array of scalars.  Must match number of points.

        name : str
            Name of point scalars to add.

        deep : bool, optional
            Does not copy scalars when False.  A reference to the scalars
            must be kept to avoid a segfault.

        """
        if scalars is None:
            raise TypeError('Empty array unable to be added')

        if not isinstance(scalars, np.ndarray):
            scalars = np.array(scalars)

        if self.n_rows == 0 or self.n_columns == 0:
            self.n_rows = scalars.shape[0]
        elif scalars.shape[0] != self.n_rows:
            raise Exception('Number of scalars must match the number of rows (%d)'
                            % self.n_rows)

        if not scalars.flags.c_contiguous:
            raise AssertionError('Array must be contigious')
        if scalars.dtype == np.bool:
            scalars = scalars.view(np.uint8)
            self._row_bool_array_names.append(name)

        vtkarr = convert_array(scalars, deep=deep)
        vtkarr.SetName(name)
        self.AddColumn(vtkarr)



    def __getitem__(self, index):
        """ Searches row data for an array """
        if isinstance(index, str):
            name = index
        elif isinstance(index, int):
            name = self.GetRowData().GetArrayName(index)
        else:
            raise KeyError('Index ({}) not understood. Index must be a string name or a tuple of string name and string preference.'.format(index))
        return row_scalar(self, name)


    def get(self, index):
        """Get an array by its name"""
        return self[index]


    def __setitem__(self, name, scalars):
        """Add/set an array in the row_arrays"""
        if scalars is None:
            raise TypeError('Empty array unable to be added')
        if not isinstance(scalars, np.ndarray):
            scalars = np.array(scalars)
        self.row_arrays[name] = scalars


    def _remove_array(self, field, key):
        """internal helper to remove a single array by name from each field"""
        field = parse_field_choice(field)
        if field == ROW_DATA_FIELD:
            self.GetRowData().RemoveArray(key)
        else:
            raise NotImplementedError('Not able to remove arrays from the ({}) data fiedl'.format(field))
        return


    def __delitem__(self, name):
        """Removes an array by the specified name"""
        del self.row_arrays[name]


    def __iter__(self):
        """The iterator across all arrays"""
        self._iter_n = 0
        return self


    def next(self):
        """Get the next block from the iterator"""
        if self._iter_n < self.n_arrays:
            result = self[self._iter_n]
            self._iter_n += 1
            return result
        else:
            raise StopIteration


    __next__ = next


    def _get_attrs(self):
        """An internal helper for the representation methods"""
        attrs = []
        attrs.append(("N Rows", self.n_rows, "{}"))
        return attrs


    def _repr_html_(self):
        """A pretty representation for Jupyter notebooks that includes header
        details and information about all scalar arrays"""
        fmt = ""
        if self.n_arrays > 0:
            fmt += "<table>"
            fmt += "<tr><th>Header</th><th>Data Arrays</th></tr>"
            fmt += "<tr><td>"
        # Get the header info
        fmt += self.head(display=False, html=True)
        # Fill out scalar arrays
        if self.n_arrays > 0:
            fmt += "</td><td>"
            fmt += "\n"
            fmt += "<table>\n"
            titles = ["Name", "Type", "N Comp", "Min", "Max"]
            fmt += "<tr>" + "".join(["<th>{}</th>".format(t) for t in titles]) + "</tr>\n"
            row = "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>\n"
            row = "<tr>" + "".join(["<td>{}</td>" for i in range(len(titles))]) + "</tr>\n"

            def format_array(key):
                """internal helper to foramt array information for printing"""
                arr = row_scalar(self, key)
                dl, dh = self.get_data_range(key)
                dl = pyvista.FLOAT_FORMAT.format(dl)
                dh = pyvista.FLOAT_FORMAT.format(dh)
                if arr.ndim > 1:
                    ncomp = arr.shape[1]
                else:
                    ncomp = 1
                return row.format(key, arr.dtype, ncomp, dl, dh)

            for i in range(self.n_arrays):
                key = self.GetRowData().GetArrayName(i)
                fmt += format_array(key)

            fmt += "</table>\n"
            fmt += "\n"
            fmt += "</td></tr> </table>"
        return fmt


    def __repr__(self):
        """Object representation"""
        return self.head(display=False, html=False)

    def __str__(self):
        """Object string representation"""
        return self.head(display=False, html=False)





class RowScalarsDict(_ScalarsDict):
    """
    Updates internal row data when an array is added or removed from
    the dictionary.
    """

    def __init__(self, data):
        _ScalarsDict.__init__(self, data)
        self.remover = lambda key: self.data._remove_array(ROW_DATA_FIELD, key)
        self.modifier = lambda *args: self.data.GetRowData().Modified()

    def adder(self, scalars, name, set_active=False, deep=True):
        self.data._add_row_scalar(scalars, name, deep=deep)
