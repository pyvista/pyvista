"""This module provides wrappers for vtkDataObjects.

The data objects does not have any sort of spatial reference.

"""
import warnings

import numpy as np

import pyvista
from pyvista import _vtk
from pyvista.utilities import FieldAssociation, PyvistaDeprecationWarning, get_array, row_array

from .dataset import DataObject
from .datasetattributes import DataSetAttributes


class Table(_vtk.vtkTable, DataObject):
    """Wrapper for the ``vtkTable`` class.

    Create by passing a 2D NumPy array of shape (``n_rows`` by ``n_columns``)
    or from a dictionary containing NumPy arrays.

    Examples
    --------
    >>> import pyvista as pv
    >>> import numpy as np
    >>> arrays = np.random.rand(100, 3)
    >>> table = pv.Table(arrays)

    """

    def __init__(self, *args, **kwargs):
        """Initialize the table."""
        super().__init__(*args, **kwargs)
        if len(args) == 1:
            if isinstance(args[0], _vtk.vtkTable):
                deep = kwargs.get('deep', True)
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
                raise TypeError(f'Table unable to be made from ({type(args[0])})')

    @staticmethod
    def _prepare_arrays(arrays):
        arrays = np.asarray(arrays)
        if arrays.ndim == 1:
            return np.reshape(arrays, (1, -1))
        elif arrays.ndim == 2:
            return arrays.T
        else:
            raise ValueError('Only 1D or 2D arrays are supported by Tables.')

    def _from_arrays(self, arrays):
        np_table = self._prepare_arrays(arrays)
        for i, array in enumerate(np_table):
            self.row_arrays[f'Array {i}'] = array

    def _from_dict(self, array_dict):
        for array in array_dict.values():
            if not isinstance(array, np.ndarray) and array.ndim < 3:
                raise ValueError('Dictionary must contain only NumPy arrays with maximum of 2D.')
        for name, array in array_dict.items():
            self.row_arrays[name] = array

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
        numpy.ndarray
            Numpy array of the row.

        """
        return self.row_arrays.get_array(name)

    @property
    def row_arrays(self):
        """Return the all row arrays."""
        return DataSetAttributes(
            vtkobject=self.GetRowData(), dataset=self, association=FieldAssociation.ROW
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

    def update(self, data):
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
        """Create a Pandas DataFrame from this Table.

        Returns
        -------
        pandas.DataFrame
            This table represented as a pandas dataframe.

        """
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            raise ImportError('Install ``pandas`` to use this feature.')
        data_frame = pd.DataFrame()
        for name, array in self.items():
            data_frame[name] = array
        return data_frame

    def save(self, *args, **kwargs):  # pragma: no cover
        """Save the table."""
        raise NotImplementedError(
            "Please use the `to_pandas` method and harness Pandas' wonderful file IO methods."
        )

    def get_data_range(self, arr=None, preference='row'):
        """Get the non-NaN min and max of a named array.

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
            self.GetRowData().GetArrayName(0)
        if isinstance(arr, str):
            arr = get_array(self, arr, preference=preference)
        # If array has no tuples return a NaN range
        if arr is None or arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
            return (np.nan, np.nan)
        # Use the array range
        return np.nanmin(arr), np.nanmax(arr)


class Texture(_vtk.vtkTexture, DataObject):
    """A helper class for vtkTextures.

    Parameters
    ----------
    uinput : str, vtkImageData, vtkTexture, list, optional
        Filename, ``vtkImagedata``, ``vtkTexture``, or a list of images to
        create a cubemap. If a list of images, must be of the same size and in
        the following order:

        * +X
        * -X
        * +Y
        * -Y
        * +Z
        * -Z

    **kwargs : dict, optional
        Optional arguments when reading from a file.

    Examples
    --------
    Load a texture from file. Files should be "image" or "image-like" files.

    >>> import os
    >>> import pyvista
    >>> from pyvista import examples
    >>> path = examples.download_masonry_texture(load=False)
    >>> os.path.basename(path)
    'masonry.bmp'
    >>> texture = pyvista.Texture(path)
    >>> texture  # doctest:+SKIP
    Texture (0x7f3131807fa0)
      Components:      3
      Cube Map:        False
      Dimensions:      256, 256

    Create a texture from an RGB array. Note how this is colored per "point"
    rather than per "pixel".

    >>> import numpy as np
    >>> arr = np.array(
    ...     [
    ...         [255, 255, 255],
    ...         [255, 0, 0],
    ...         [0, 255, 0],
    ...         [0, 0, 255],
    ...     ], dtype=np.uint8
    ... )
    >>> arr = arr.reshape((2, 2, 3))
    >>> texture = pyvista.Texture(arr)
    >>> texture.plot()

    Create a cubemap from 6 textures.

    >>> px = examples.download_sky(direction='px')  # doctest:+SKIP
    >>> nx = examples.download_sky(direction='nx')  # doctest:+SKIP
    >>> py = examples.download_sky(direction='py')  # doctest:+SKIP
    >>> ny = examples.download_sky(direction='ny')  # doctest:+SKIP
    >>> pz = examples.download_sky(direction='pz')  # doctest:+SKIP
    >>> nz = examples.download_sky(direction='nz')  # doctest:+SKIP
    >>> texture = pyvista.Texture([px, nx, py, ny, pz, nz])  # doctest:+SKIP
    >>> texture.cube_map  # doctest:+SKIP
    True

    """

    def __init__(self, uinput=None, **kwargs):
        """Initialize the texture."""
        super().__init__(uinput, **kwargs)

        if isinstance(uinput, _vtk.vtkTexture):
            self._from_texture(uinput)
        elif isinstance(uinput, np.ndarray):
            self._from_array(uinput)
        elif isinstance(uinput, _vtk.vtkImageData):
            self._from_image_data(uinput)
        elif isinstance(uinput, str):
            self._from_file(filename=uinput, **kwargs)
        elif isinstance(uinput, list):
            self.SetMipmap(True)
            self.SetInterpolate(True)
            self.cube_map = True  # Must be set prior to setting images

            # add each image to the cubemap
            for i, image in enumerate(uinput):
                flip = _vtk.vtkImageFlip()
                flip.SetInputDataObject(image)
                flip.SetFilteredAxis(1)  # flip y axis
                flip.Update()
                self.SetInputDataObject(i, flip.GetOutput())
        elif uinput is None:
            pass
        else:
            raise TypeError(f'Cannot create a pyvista.Texture from ({type(uinput)})')

    def __repr__(self):
        """Return the object representation."""
        return pyvista.DataSet.__repr__(self)

    def _get_attrs(self):
        """Return the representation methods (internal helper)."""
        attrs = []
        attrs.append(("Components", self.n_components, "{:d}"))
        attrs.append(("Cube Map", self.cube_map, "{:}"))
        attrs.append(("Dimensions", self.dimensions, "{:d}, {:d}"))
        return attrs

    @property
    def dimensions(self) -> tuple:
        """Dimensions of the texture.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> texture.dimensions
        (256, 256)

        """
        input_data = self.GetInput()
        if input_data is None:
            return (0, 0)
        return input_data.GetDimensions()[:2]

    def _from_file(self, filename, **kwargs):
        try:
            image = pyvista.read(filename, **kwargs)
            if image.GetNumberOfPoints() < 2:  # pragma: no cover
                raise ValueError("Problem reading the image with VTK.")
            self._from_image_data(image)
        except (KeyError, ValueError):
            from imageio import imread

            self._from_array(imread(filename))

    def _from_texture(self, texture):
        image = texture.GetInput()
        self._from_image_data(image)

    def _from_image_data(self, image):
        if not isinstance(image, pyvista.UniformGrid):
            image = pyvista.UniformGrid(image)
        self.SetInputDataObject(image)
        self.Update()

    def _from_array(self, image):
        """Create a texture from a np.ndarray."""
        if not 2 <= image.ndim <= 3:
            # we support 2 [single component image] or 3 [e.g. rgb or rgba] dims
            raise ValueError('Input image must be nn by nm by RGB[A]')

        if image.ndim == 3:
            if image.shape[2] not in [1, 3, 4]:
                raise ValueError(
                    'Third dimension of the array must be of size 1 (greyscale), 3 (RGB), or 4 (RGBA)'
                )
            n_components = image.shape[2]
        elif image.ndim == 2:
            n_components = 1

        grid = pyvista.UniformGrid(dims=(image.shape[1], image.shape[0], 1))
        grid.point_data['Image'] = np.flip(image.swapaxes(0, 1), axis=1).reshape(
            (-1, n_components), order='F'
        )
        grid.set_active_scalars('Image')
        self._from_image_data(grid)

    @property
    def repeat(self) -> bool:
        """Enable or disable repeating the texture."""
        return bool(self.GetRepeat())

    @repeat.setter
    def repeat(self, flag: bool):
        self.SetRepeat(flag)

    @property
    def n_components(self) -> int:
        """Components in the image.

        Single component textures are greyscale, while 3 or 4 component are
        used for representing RGB and RGBA images.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> texture.n_components
        3

        """
        input_data = self.GetInput()
        if input_data is None:
            return 0
        return input_data.GetPointData().GetScalars().GetNumberOfComponents()

    def flip(self, axis):
        """Flip this texture inplace along the specified axis. 0 for X and 1 for Y."""
        if axis not in [0, 1]:
            raise ValueError(f"axis must be 0 or 1. Got {axis}")
        array = np.flip(self.image_data, axis=1 - axis)
        self._from_array(array)

    def to_image(self):
        """Return the texture as an image.

        Returns
        -------
        pyvista.UniformGrid
            Texture represented as a uniform grid.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> texture.to_image()  # doctest:+SKIP
        UniformGrid (0x7f313126afa0)
          N Cells:      65025
          N Points:     65536
          X Bounds:     0.000e+00, 2.550e+02
          Y Bounds:     0.000e+00, 2.550e+02
          Z Bounds:     0.000e+00, 0.000e+00
          Dimensions:   256, 256, 1
          Spacing:      1.000e+00, 1.000e+00, 1.000e+00
          N Arrays:     1

        """
        input_dataset = self.GetInput()
        if isinstance(input_dataset, pyvista.UniformGrid):
            return input_dataset

        # this results in a copy
        return pyvista.wrap(input_dataset)

    def to_array(self):
        """Return the texture as an array.

        Returns
        -------
        pyvista_ndarray
            Texture data as a numpy array.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> arr = texture.to_array()
        >>> arr.shape
        (256, 256, 3)
        >>> arr.dtype
        dtype('uint8')

        """
        # Deprecated on v0.35.0, estimated removal on v0.37.0
        warnings.warn(  # pragma: no cover
            '`Texture.to_array` is deprecated. Use `image_data` instead', PyvistaDeprecationWarning
        )
        return self.image_data

    def plot(self, *args, **kwargs):
        """Plot the texture an image.

        If the texture is a cubemap, it will be displayed as a skybox.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> texture.plot()

        """
        if self.cube_map:
            return self._plot_skybox(*args, **kwargs)
        kwargs.setdefault('cpos', 'xy')
        kwargs.setdefault('rgba', self.n_components > 1)
        kwargs.setdefault('show_axes', False)
        kwargs.setdefault('show_scalar_bar', False)
        if self.n_components == 1:
            kwargs.setdefault('cmap', 'gray')
        return self.to_image().plot(*args, **kwargs)

    def _plot_skybox(self, *args, **kwargs):
        """Plot this texture as a skybox."""
        cpos = kwargs.pop('cpos', 'xy')
        pl = pyvista.Plotter(*args, **kwargs)
        pl.add_actor(self.to_skybox())
        pl.camera_position = cpos
        pl.show()

    @property
    def cube_map(self):
        """Return ``True`` if cube mapping is enabled and ``False`` otherwise.

        Is this texture a cube map, if so it needs 6 inputs, one for
        each side of the cube. You must set this before connecting the
        inputs.  The inputs must all have the same size, data type,
        and depth.
        """
        return self.GetCubeMap()

    @cube_map.setter
    def cube_map(self, flag):
        """Enable cube mapping if ``flag`` is True, disable it otherwise."""
        self.SetCubeMap(flag)

    def copy(self, deep=True):
        """Make a copy of this texture.

        Parameters
        ----------
        deep : bool, optional
            Perform a deep copy when ``True``. Shallow copy when ``False``.

        Returns
        -------
        pyvista.Texture
            Copied texture.
        """
        return Texture(self.to_image().copy(deep=deep))

    def to_skybox(self):
        """Return the texture as a ``vtkSkybox`` if cube mapping is enabled.

        Returns
        -------
        vtk.vtkSkybox
            Skybox if cube mapping is enabled.  Otherwise, raises an exception.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_sky_box_cube_map()  # doctest:+SKIP
        >>> texture.to_skybox()  # doctest:+SKIP
        <vtkmodules.vtkRenderingOpenGL2.vtkOpenGLSkybox(0x464dbb0) at 0x7f3130fab1c0>

        """
        if not self.cube_map:
            raise ValueError('This texture is not a cube map.')
        skybox = _vtk.vtkSkybox()
        skybox.SetTexture(self)
        return skybox

    @property
    def image_data(self):
        """Return the underlying image data associated with this texture.

        Returns
        -------
        pyvista.pyvista_ndarray
            Array of the image data.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> texture.image_data.shape
        (256, 256, 3)
        >>> texture.image_data.shape
        dtype('uint8')

        """
        data = pyvista.wrap(self.GetInput().GetPointData().GetScalars())
        if data.ndim > 1:
            shape = (self.dimensions[1], self.dimensions[0], data.shape[1])
        else:
            shape = (self.dimensions[1], self.dimensions[0])

        return np.flip(data.reshape(shape, order='F'), axis=1).swapaxes(1, 0)

    @image_data.setter
    def image_data(self, image_data):
        """Set the image data."""
        self._from_array(image_data)

    def to_greyscale(self):
        """Convert this texture as a single component (greyscale) texture.

        Returns
        -------
        pyvista.Texture
            Texture converted to greyscale. If already black and white,

        Notes
        -----
        The transparency channel (if available) will be dropped.

        Uses NTSC/PAL implementation.

        Examples
        --------
        >>> from pyvista import examples
        >>> texture = examples.download_masonry_texture()
        >>> bw_texture = texture.to_bw()
        >>> bw_texture.plot()

        """
        if self.n_components == 1:
            return self

        new_texture = self.copy()
        r, g, b = np.asarray(new_texture.image_data).T[:3]
        greyscale = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)
        greyscale = greyscale.reshape(self.dimensions, order='F').swapaxes(1, 0)
        new_texture.image_data = greyscale
        return new_texture
