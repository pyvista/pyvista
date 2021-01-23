"""This module provides wrappers for vtkDataObjects.

The data objects does not have any sort of spatial reference.

"""
import imageio
import numpy as np
import vtk

import pyvista
from pyvista.utilities import (FieldAssociation, assert_empty_kwargs, get_array,
                               row_array)
from .common import DataObject
from .datasetattributes import DataSetAttributes


class Table(vtk.vtkTable, DataObject):
    """Wrapper for the ``vtkTable`` class.

    Create by passing a 2D NumPy array of shape (``n_rows`` by ``n_columns``)
    or from a dictionary containing NumPy arrays.

    Example
    -------
    >>> import pyvista as pv
    >>> import numpy as np
    >>> arrays = np.random.rand(100, 3)
    >>> table = pv.Table(arrays)

    """

    def __init__(self, *args, **kwargs):
        """Initialize the table."""
        super().__init__(*args, **kwargs)
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
            elif 'pandas.core.frame.DataFrame' in str(type(args[0])):
                self._from_pandas(args[0])
            else:
                raise TypeError(f'Table unable to be made from ({type(args[0])})')

    def _from_arrays(self, arrays):
        if not arrays.ndim == 2:
            raise ValueError('Only 2D arrays are supported by Tables.')
        np_table = arrays.T
        for i, array in enumerate(np_table):
            self.row_arrays[f'Array {i}'] = array
        return

    def _from_dict(self, array_dict):
        for array in array_dict.values():
            if not isinstance(array, np.ndarray) and array.ndim < 3:
                raise ValueError('Dictionary must contain only NumPy arrays with maximum of 2D.')
        for name, array in array_dict.items():
            self.row_arrays[name] = array
        return

    def _from_pandas(self, data_frame):
        for name in data_frame.keys():
            self.row_arrays[name] = data_frame[name].values

    @property
    def n_rows(self):
        """Return the number of rows."""
        return self.GetNumberOfRows()

    @n_rows.setter
    def n_rows(self, n):
        """Set the number of rows."""
        self.SetNumberOfRows(n)

    @property
    def n_columns(self):
        """Return the number of columns."""
        return self.GetNumberOfColumns()

    @property
    def n_arrays(self):
        """Return the number of columns.

        Alias for: ``n_columns``.

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
        scalars : np.ndarray
            Numpy array of scalars

        """
        return self.row_arrays[name]

    @property
    def row_arrays(self):
        """Return the all row arrays."""
        return DataSetAttributes(vtkobject=self.GetRowData(), dataset=self, association=FieldAssociation.ROW)

    def keys(self):
        """Return the table keys."""
        return self.row_arrays.keys()

    def items(self):
        """Return the table items."""
        return self.row_arrays.items()

    def values(self):
        """Return the table values."""
        return self.row_arrays.values()

    def update(self, data):
        """Set the table data."""
        self.row_arrays.update(data)

    def pop(self, name):
        """Pops off an array by the specified name."""
        return self.row_arrays.pop(name)

    def _add_row_array(self, scalars, name, deep=True):
        """Add scalars to the vtk object.

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
        self.row_arrays[name] = scalars

    def __getitem__(self, index):
        """Search row data for an array."""
        return self._row_array(name=index)

    def _ipython_key_completions_(self):
        return self.keys()

    def get(self, index):
        """Get an array by its name."""
        return self[index]

    def __setitem__(self, name, scalars):
        """Add/set an array in the row_arrays."""
        self.row_arrays[name] = scalars

    def _remove_array(self, field, key):
        """Remove a single array by name from each field (internal helper)."""
        self.row_arrays.remove(key)

    def __delitem__(self, name):
        """Remove an array by the specified name."""
        del self.row_arrays[name]

    def __iter__(self):
        """Return the iterator across all arrays."""
        for array_name in self.row_arrays:
            yield self.row_arrays[array_name]

    def _get_attrs(self):
        """Return the representation methods."""
        attrs = []
        attrs.append(("N Rows", self.n_rows, "{}"))
        return attrs

    def _repr_html_(self):
        """Return a pretty representation for Jupyter notebooks.

        It includes header details and information about all arrays.

        """
        fmt = ""
        if self.n_arrays > 0:
            fmt += "<table>"
            fmt += "<tr><th>Header</th><th>Data Arrays</th></tr>"
            fmt += "<tr><td>"
        # Get the header info
        fmt += self.head(display=False, html=True)
        # Fill out scalars arrays
        if self.n_arrays > 0:
            fmt += "</td><td>"
            fmt += "\n"
            fmt += "<table>\n"
            titles = ["Name", "Type", "N Comp", "Min", "Max"]
            fmt += "<tr>" + "".join([f"<th>{t}</th>" for t in titles]) + "</tr>\n"
            row = "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>\n"
            row = "<tr>" + "".join(["<td>{}</td>" for i in range(len(titles))]) + "</tr>\n"

            def format_array(key):
                """Format array information for printing (internal helper)."""
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
        """Return the object representation."""
        return self.head(display=False, html=False)

    def __str__(self):
        """Return the object string representation."""
        return self.head(display=False, html=False)

    def to_pandas(self):
        """Create a Pandas DataFrame from this Table."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError('Install ``pandas`` to use this feature.')
        data_frame = pd.DataFrame()
        for name, array in self.items():
            data_frame[name] = array
        return data_frame

    def save(self, *args, **kwargs):
        """Save the table."""
        raise NotImplementedError("Please use the `to_pandas` method and "
                                  "harness Pandas' wonderful file IO methods.")

    def get_data_range(self, arr=None, preference='row'):
        """Get the non-NaN min and max of a named array.

        Parameters
        ----------
        arr : str, np.ndarray, optional
            The name of the array to get the range. If None, the active scalar
            is used

        preference : str, optional
            When scalars is specified, this is the preferred array type
            to search for in the dataset.  Must be either ``'row'`` or
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


class Texture(vtk.vtkTexture, DataObject):
    """A helper class for vtkTextures."""

    _READERS = {'.bmp': vtk.vtkBMPReader, '.dem': vtk.vtkDEMReader, '.dcm': vtk.vtkDICOMImageReader,
                '.img': vtk.vtkDICOMImageReader, '.jpeg': vtk.vtkJPEGReader, '.jpg': vtk.vtkJPEGReader,
                '.mhd': vtk.vtkMetaImageReader, '.nrrd': vtk.vtkNrrdReader, '.nhdr': vtk.vtkNrrdReader,
                '.png': vtk.vtkPNGReader, '.pnm': vtk.vtkPNMReader, '.slc': vtk.vtkSLCReader,
                '.tiff': vtk.vtkTIFFReader, '.tif': vtk.vtkTIFFReader}

    def __init__(self, *args, **kwargs):
        """Initialize the texture."""
        super().__init__(*args, **kwargs)
        assert_empty_kwargs(**kwargs)

        if len(args) == 1:
            if isinstance(args[0], vtk.vtkTexture):
                self._from_texture(args[0])
            elif isinstance(args[0], np.ndarray):
                self._from_array(args[0])
            elif isinstance(args[0], vtk.vtkImageData):
                self._from_image_data(args[0])
            elif isinstance(args[0], str):
                self._from_file(filename=args[0])
            else:
                raise TypeError(f'Table unable to be made from ({type(args[0])})')

    def _from_file(self, filename):
        try:
            image = self._load_file(filename)
            if image.GetNumberOfPoints() < 2:
                raise ValueError("Problem reading the image with VTK.")
            self._from_image_data(image)
        except (KeyError, ValueError):
            self._from_array(imageio.imread(filename))

    def _from_texture(self, texture):
        image = texture.GetInput()
        self._from_image_data(image)

    def _from_image_data(self, image):
        if not isinstance(image, pyvista.UniformGrid):
            image = pyvista.UniformGrid(image)
        self.SetInputDataObject(image)
        return self.Update()

    def _from_array(self, image):
        """Create a texture from a np.ndarray."""
        if not 2 <= image.ndim <= 3:
            # we support 2 [single component image] or 3 [e.g. rgb or rgba] dims
            raise ValueError('Input image must be nn by nm by RGB[A]')

        if image.ndim == 3:
            if not 3 <= image.shape[2] <= 4:
                raise ValueError('Third dimension of the array must be of size 3 (RGB) or 4 (RGBA)')
            n_components = image.shape[2]
        elif image.ndim == 2:
            n_components = 1

        grid = pyvista.UniformGrid((image.shape[1], image.shape[0], 1))
        grid.point_arrays['Image'] = np.flip(image.swapaxes(0, 1), axis=1).reshape((-1, n_components), order='F')
        grid.set_active_scalars('Image')
        return self._from_image_data(grid)

    @property
    def repeat(self):
        """Repeat the texture."""
        return self.GetRepeat()

    @repeat.setter
    def repeat(self, flag):
        self.SetRepeat(flag)

    @property
    def n_components(self):
        """Components in the image (e.g. 3 [or 4] for RGB[A])."""
        image = self.to_image()
        return image.active_scalars.shape[1]

    def flip(self, axis):
        """Flip this texture inplace along the specified axis. 0 for X and 1 for Y."""
        if not 0 <= axis <= 1:
            raise ValueError(f"Axis {axis} out of bounds")
        array = self.to_array()
        array = np.flip(array, axis=1 - axis)
        return self._from_array(array)

    def to_image(self):
        """Return the texture as an image."""
        return self.GetInput()

    def to_array(self):
        """Return the texture as a np.ndarray."""
        image = self.to_image()

        if image.active_scalars.ndim > 1:
            shape = (image.dimensions[1], image.dimensions[0], self.n_components)
        else:
            shape = (image.dimensions[1], image.dimensions[0])

        return np.flip(image.active_scalars.reshape(shape, order='F'), axis=1).swapaxes(1,0)

    def plot(self, *args, **kwargs):
        """Plot the texture as image data by itself."""
        return self.to_image().plot(*args, **kwargs)

    def copy(self):
        """Make a copy of this texture."""
        return Texture(self.to_image().copy())
