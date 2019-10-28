"""This module provides wrappers for vtkDataObjects: data onjects without any
sort of spatial reference.
"""
import numpy as np
import vtk

import pyvista
from pyvista.utilities import (ROW_DATA_FIELD, convert_array, get_array,
                               parse_field_choice, row_array, vtk_bit_array_to_char)

from .common import DataObject, _ScalarsDict

try:
    import pandas as pd
except ImportError:
    pd = None






class Table(vtk.vtkTable, DataObject):
    """Wrapper for the ``vtkTable`` class. Create by passing a 2D NumPy array
    of shape (``n_rows`` by ``n_columns``) or from a dictionary containing
    NumPy arrays.

    Example
    -------
    >>> import pyvista as pv
    >>> import numpy as np
    >>> arrays = np.random.rand(100, 3)
    >>> table = pv.Table(arrays)

    """
    def __init__(self, *args, **kwargs):

        if len(args) == 1:
            if isinstance(args[0], vtk.vtkTable):
                deep = kwargs.get('deep', True)
                if deep:
                    self.deep_copy(args[0])
                else:
                    self.shallow_copy(args[0])
            elif isinstance(args[0], np.ndarray):
                self._from_arrays(args[0])
            elif isinstance(args[0], dict):
                self._from_dict(args[0])
            elif pd is not None and isinstance(args[0], pd.DataFrame):
                self._from_pandas(args[0])
            else:
                raise TypeError('Table unable to be made from ({})'.format(type(args[0])))


        self._row_bool_array_names = []


    def _from_arrays(self, arrays):
        if not arrays.ndim == 2:
            raise AssertionError('Only 2D arrays are supported by Tables.')
        np_table = arrays.T
        for i, array in enumerate(np_table):
            self.row_arrays['Array {}'.format(i)] = array
        return


    def _from_dict(self, array_dict):
        for array in array_dict.values():
            if not isinstance(array, (np.ndarray)) and array.ndim < 3:
                raise RuntimeError('Dictionaty must contain only NumPy arrays with maximum of 2D.')
        for name, array in array_dict.items():
            self.row_arrays[name] = array
        return


    def _from_pandas(self, data_frame):
        for name in data_frame.keys():
            self.row_arrays[name] = data_frame[name]
        return


    @property
    def n_rows(self):
        return self.GetNumberOfRows()


    @n_rows.setter
    def n_rows(self, n):
        self.SetNumberOfRows(n)


    @property
    def n_columns(self):
        return self.GetNumberOfColumns()


    @property
    def n_arrays(self):
        return self.n_columns


    def _row_array(self, name=None):
        """
        Returns row scalars of a vtk object

        Parameters
        ----------
        name : str
            Name of row scalars to retrieve.

        Returns
        -------
        scalars : np.ndarray
            Numpy array of scalars

        """
        if name is None:
            # use first array
            name = self.GetRowData().GetArrayName(0)
            if name is None:
                raise RuntimeError('No arrays present to fetch.')
        vtkarr = self.GetRowData().GetAbstractArray(name)
        if vtkarr is None:
            raise AssertionError('({}) is not a row scalar'.format(name))

        # numpy does not support bit array data types
        if isinstance(vtkarr, vtk.vtkBitArray):
            vtkarr = vtk_bit_array_to_char(vtkarr)
            if name not in self._row_bool_array_names:
                self._row_bool_array_names.append(name)

        array = convert_array(vtkarr)
        if array.dtype == np.uint8 and name in self._row_bool_array_names:
            array = array.view(np.bool)
        return array


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
            self._row_arrays[name] = self._row_array(name)

        self._row_arrays.enable_callback()
        return self._row_arrays


    def keys(self):
        return list(self.row_arrays.keys())


    def items(self):
        return self.row_arrays.items()


    def values(self):
        return self.row_arrays.values()


    def update(self, data):
        self.row_arrays.update(data)


    def pop(self, name):
        """Pops off an array by the specified name"""
        return self.row_arrays.pop(name)


    def _add_row_array(self, scalars, name, deep=True):
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
            scalars = np.ascontiguousarray(scalars)
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
        return row_array(self, name)


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
                arr = row_array(self, key)
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


    def to_pandas(self):
        """Create a Pandas DataFrame from this Table"""
        if pd is None:
            raise ImportError('You must have Pandas installed.')
        data_frame = pd.DataFrame()
        for name, array in self.items():
            data_frame[name] = array
        return data_frame


    def save(self, *args, **kwargs):
        raise NotImplementedError("Please use the `to_pandas` method and "
                                  "harness Pandas' wonderful file IO methods.")


    def get_data_range(self, arr=None, preference='row'):
        """Get the non-NaN min and max of a named scalar array

        Parameters
        ----------
        arr : str, np.ndarray, optional
            The name of the array to get the range. If None, the active scalar
            is used

        preference : str, optional
            When scalars is specified, this is the perfered scalar type to
            search for in the dataset.  Must be either ``'row'`` or
            ``'field'``.

        """
        if arr is None:
            # use the first array in the row data
            self.GetRowData().GetArrayName(0)
        if isinstance(arr, str):
            arr = get_array(self, arr, preference=preference)
        # If array has no tuples return a NaN range
        if arr is None or arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
            return (np.nan, np.nan)
        # Use the array range
        return np.nanmin(arr), np.nanmax(arr)



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
        self.data._add_row_array(scalars, name, deep=deep)



class Texture(vtk.vtkTexture):
    """A helper class for vtkTextures"""
    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            if isinstance(args[0], vtk.vtkTexture):
                self._from_texture(args[0])
            elif isinstance(args[0], np.ndarray):
                self._from_array(args[0])
            elif isinstance(args[0], vtk.vtkImageData):
                self._from_image_data(args[0])
            elif isinstance(args[0], str):
                self._from_texture(pyvista.read_texture(args[0]))
            else:
                raise TypeError('Table unable to be made from ({})'.format(type(args[0])))

    def _from_texture(self, texture):
        image = texture.GetInput()
        self._from_image_data(image)

    def _from_image_data(self, image):
        if not isinstance(image, pyvista.UniformGrid):
            image = pyvista.UniformGrid(image)
        self.SetInputDataObject(image)
        return self.Update()


    def _from_array(self, image):
        if image.ndim != 3 or image.shape[2] != 3:
            raise AssertionError('Input image must be nn by nm by RGB')
        grid = pyvista.UniformGrid((image.shape[1], image.shape[0], 1))
        grid.point_arrays['Image'] = np.flip(image.swapaxes(0,1), axis=1).reshape((-1, 3), order='F')
        grid.set_active_scalar('Image')
        return self._from_image_data(grid)


    def flip(self, axis):
        """Flip this texture inplace along the specifed axis. 0 for X and
        1 for Y."""
        if axis < 0 or axis > 1:
            raise RuntimeError("Axis {} out of bounds".format(axis))
        ax = [1, 0]
        array = self.to_array()
        array = np.flip(array, axis=ax[axis])
        return self._from_array(array)


    def to_image(self):
        return self.GetInput()


    def to_array(self):
        image = self.to_image()
        shape = (image.dimensions[0], image.dimensions[1], 3)
        return np.flip(image.active_scalar.reshape(shape, order='F'), axis=1).swapaxes(1,0)


    def plot(self, *args, **kwargs):
        """Plot the texture as image data by itself"""
        return self.to_image().plot(*args, **kwargs)


    @property
    def repeat(self):
        return self.GetRepeat()

    @repeat.setter
    def repeat(self, flag):
        self.SetRepeat(flag)
