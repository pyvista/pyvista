"""Sub-classes for vtk.vtkRectilinearGrid and vtk.vtkImageData."""
from functools import wraps
import logging
import pathlib
from typing import Sequence, Tuple, Union
import warnings

import numpy as np

import pyvista
from pyvista import _vtk
from pyvista.core.dataset import DataSet
from pyvista.core.filters import RectilinearGridFilters, UniformGridFilters, _get_output
from pyvista.utilities import abstract_class, assert_empty_kwargs
import pyvista.utilities.helpers as helpers
from pyvista.utilities.misc import PyVistaDeprecationWarning, raise_has_duplicates

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')


@abstract_class
class Grid(DataSet):
    """A class full of common methods for non-pointset grids."""

    def __init__(self, *args, **kwargs):
        """Initialize the grid."""
        super().__init__()

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        """Return the grid's dimensions.

        These are effectively the number of points along each of the
        three dataset axes.

        Examples
        --------
        Create a uniform grid with dimensions ``(1, 2, 3)``.

        >>> import pyvista
        >>> grid = pyvista.UniformGrid(dimensions=(2, 3, 4))
        >>> grid.dimensions
        (2, 3, 4)
        >>> grid.plot(show_edges=True)

        Set the dimensions to ``(3, 4, 5)``

        >>> grid.dimensions = (3, 4, 5)
        >>> grid.plot(show_edges=True)

        """
        return self.GetDimensions()

    @dimensions.setter
    def dimensions(self, dims: Sequence[int]):
        """Set the dataset dimensions."""
        self.SetDimensions(*dims)
        self.Modified()

    def _get_attrs(self):
        """Return the representation methods (internal helper)."""
        attrs = DataSet._get_attrs(self)
        attrs.append(("Dimensions", self.dimensions, "{:d}, {:d}, {:d}"))
        return attrs


class RectilinearGrid(_vtk.vtkRectilinearGrid, Grid, RectilinearGridFilters):
    """Dataset with variable spacing in the three coordinate directions.

    Can be initialized in several ways:

    * Create empty grid
    * Initialize from a ``vtk.vtkRectilinearGrid`` object
    * Initialize directly from the point arrays

    Parameters
    ----------
    uinput : str, pathlib.Path, vtk.vtkRectilinearGrid, numpy.ndarray, optional
        Filename, dataset, or array to initialize the rectilinear grid from. If a
        filename is passed, pyvista will attempt to load it as a
        :class:`RectilinearGrid`. If passed a ``vtk.vtkRectilinearGrid``, it
        will be wrapped. If a :class:`numpy.ndarray` is passed, this will be
        loaded as the x range.

    y : numpy.ndarray, optional
        Coordinates of the points in y direction. If this is passed, ``uinput``
        must be a :class:`numpy.ndarray`.

    z : numpy.ndarray, optional
        Coordinates of the points in z direction. If this is passed, ``uinput``
        and ``y`` must be a :class:`numpy.ndarray`.

    check_duplicates : bool, optional
        Check for duplications in any arrays that are passed. Defaults to
        ``False``. If ``True``, an error is raised if there are any duplicate
        values in any of the array-valued input arguments.

    deep : bool, optional
        Whether to deep copy a ``vtk.vtkRectilinearGrid`` object.
        Default is ``False``.  Keyword only.

    Examples
    --------
    >>> import pyvista
    >>> import vtk
    >>> import numpy as np

    Create an empty grid.

    >>> grid = pyvista.RectilinearGrid()

    Initialize from a vtk.vtkRectilinearGrid object

    >>> vtkgrid = vtk.vtkRectilinearGrid()
    >>> grid = pyvista.RectilinearGrid(vtkgrid)

    Create from NumPy arrays.

    >>> xrng = np.arange(-10, 10, 2)
    >>> yrng = np.arange(-10, 10, 5)
    >>> zrng = np.arange(-10, 10, 1)
    >>> grid = pyvista.RectilinearGrid(xrng, yrng, zrng)
    >>> grid.plot(show_edges=True)

    """

    _WRITERS = {'.vtk': _vtk.vtkRectilinearGridWriter, '.vtr': _vtk.vtkXMLRectilinearGridWriter}

    def __init__(self, *args, check_duplicates=False, deep=False, **kwargs):
        """Initialize the rectilinear grid."""
        super().__init__()

        if len(args) == 1:
            if isinstance(args[0], _vtk.vtkRectilinearGrid):
                if deep:
                    self.deep_copy(args[0])
                else:
                    self.shallow_copy(args[0])
            elif isinstance(args[0], (str, pathlib.Path)):
                self._from_file(args[0], **kwargs)
            elif isinstance(args[0], np.ndarray):
                self._from_arrays(args[0], None, None, check_duplicates)
            else:
                raise TypeError(f'Type ({type(args[0])}) not understood by `RectilinearGrid`')

        elif len(args) == 3 or len(args) == 2:
            arg0_is_arr = isinstance(args[0], np.ndarray)
            arg1_is_arr = isinstance(args[1], np.ndarray)
            if len(args) == 3:
                arg2_is_arr = isinstance(args[2], np.ndarray)
            else:
                arg2_is_arr = False

            if all([arg0_is_arr, arg1_is_arr, arg2_is_arr]):
                self._from_arrays(args[0], args[1], args[2], check_duplicates)
            elif all([arg0_is_arr, arg1_is_arr]):
                self._from_arrays(args[0], args[1], None, check_duplicates)
            else:
                raise TypeError("Arguments not understood by `RectilinearGrid`.")

    def __repr__(self):
        """Return the default representation."""
        return DataSet.__repr__(self)

    def __str__(self):
        """Return the str representation."""
        return DataSet.__str__(self)

    def _update_dimensions(self):
        """Update the dimensions if coordinates have changed."""
        return self.SetDimensions(len(self.x), len(self.y), len(self.z))

    def _from_arrays(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray, check_duplicates: bool = False
    ):
        """Create VTK rectilinear grid directly from numpy arrays.

        Each array gives the uniques coordinates of the mesh along each axial
        direction. To help ensure you are using this correctly, we take the unique
        values of each argument.

        Parameters
        ----------
        x : numpy.ndarray
            Coordinates of the points in x direction.

        y : numpy.ndarray
            Coordinates of the points in y direction.

        z : numpy.ndarray
            Coordinates of the points in z direction.

        check_duplicates : bool, optional
            Check for duplications in any arrays that are passed.

        """
        # Set the coordinates along each axial direction
        # Must at least be an x array
        if check_duplicates:
            raise_has_duplicates(x)

        # edges are shown as triangles if x is not floating point
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(float)
        self.SetXCoordinates(helpers.convert_array(x.ravel()))
        if y is not None:
            if check_duplicates:
                raise_has_duplicates(y)
            if not np.issubdtype(y.dtype, np.floating):
                y = y.astype(float)
            self.SetYCoordinates(helpers.convert_array(y.ravel()))
        if z is not None:
            if check_duplicates:
                raise_has_duplicates(z)
            if not np.issubdtype(z.dtype, np.floating):
                z = z.astype(float)
            self.SetZCoordinates(helpers.convert_array(z.ravel()))
        # Ensure dimensions are properly set
        self._update_dimensions()

    @property
    def meshgrid(self) -> list:
        """Return a meshgrid of numpy arrays for this mesh.

        This simply returns a :func:`numpy.meshgrid` of the
        coordinates for this mesh in ``ij`` indexing. These are a copy
        of the points of this mesh.

        """
        return np.meshgrid(self.x, self.y, self.z, indexing='ij')

    @property  # type: ignore
    def points(self) -> np.ndarray:  # type: ignore
        """Return a copy of the points as an n by 3 numpy array.

        Notes
        -----
        Points of a :class:`pyvista.RectilinearGrid` cannot be
        set. Set point coordinates with :attr:`RectilinearGrid.x`,
        :attr:`RectilinearGrid.y`, or :attr:`RectilinearGrid.z`.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista
        >>> xrng = np.arange(-10, 10, 10, dtype=float)
        >>> yrng = np.arange(-10, 10, 10, dtype=float)
        >>> zrng = np.arange(-10, 10, 10, dtype=float)
        >>> grid = pyvista.RectilinearGrid(xrng, yrng, zrng)
        >>> grid.points
        array([[-10., -10., -10.],
               [  0., -10., -10.],
               [-10.,   0., -10.],
               [  0.,   0., -10.],
               [-10., -10.,   0.],
               [  0., -10.,   0.],
               [-10.,   0.,   0.],
               [  0.,   0.,   0.]])

        """
        xx, yy, zz = self.meshgrid
        return np.c_[xx.ravel(order='F'), yy.ravel(order='F'), zz.ravel(order='F')]

    @points.setter
    def points(self, points):
        """Raise an AttributeError.

        This setter overrides the base class's setter to ensure a user
        does not attempt to set them.
        """
        raise AttributeError(
            "The points cannot be set. The points of "
            "`RectilinearGrid` are defined in each axial direction. Please "
            "use the `x`, `y`, and `z` setters individually."
        )

    @property
    def x(self) -> np.ndarray:
        """Return or set the coordinates along the X-direction.

        Examples
        --------
        Return the x coordinates of a RectilinearGrid.

        >>> import numpy as np
        >>> import pyvista
        >>> xrng = np.arange(-10, 10, 10, dtype=float)
        >>> yrng = np.arange(-10, 10, 10, dtype=float)
        >>> zrng = np.arange(-10, 10, 10, dtype=float)
        >>> grid = pyvista.RectilinearGrid(xrng, yrng, zrng)
        >>> grid.x
        array([-10.,   0.])

        Set the x coordinates of a RectilinearGrid.

        >>> grid.x = [-10.0, 0.0, 10.0]
        >>> grid.x
        array([-10.,   0.,  10.])

        """
        return helpers.convert_array(self.GetXCoordinates())

    @x.setter
    def x(self, coords: Sequence):
        """Set the coordinates along the X-direction."""
        self.SetXCoordinates(helpers.convert_array(coords))
        self._update_dimensions()
        self.Modified()

    @property
    def y(self) -> np.ndarray:
        """Return or set the coordinates along the Y-direction.

        Examples
        --------
        Return the y coordinates of a RectilinearGrid.

        >>> import numpy as np
        >>> import pyvista
        >>> xrng = np.arange(-10, 10, 10, dtype=float)
        >>> yrng = np.arange(-10, 10, 10, dtype=float)
        >>> zrng = np.arange(-10, 10, 10, dtype=float)
        >>> grid = pyvista.RectilinearGrid(xrng, yrng, zrng)
        >>> grid.y
        array([-10.,   0.])

        Set the y coordinates of a RectilinearGrid.

        >>> grid.y = [-10.0, 0.0, 10.0]
        >>> grid.y
        array([-10.,   0.,  10.])

        """
        return helpers.convert_array(self.GetYCoordinates())

    @y.setter
    def y(self, coords: Sequence):
        """Set the coordinates along the Y-direction."""
        self.SetYCoordinates(helpers.convert_array(coords))
        self._update_dimensions()
        self.Modified()

    @property
    def z(self) -> np.ndarray:
        """Return or set the coordinates along the Z-direction.

        Examples
        --------
        Return the z coordinates of a RectilinearGrid.

        >>> import numpy as np
        >>> import pyvista
        >>> xrng = np.arange(-10, 10, 10, dtype=float)
        >>> yrng = np.arange(-10, 10, 10, dtype=float)
        >>> zrng = np.arange(-10, 10, 10, dtype=float)
        >>> grid = pyvista.RectilinearGrid(xrng, yrng, zrng)
        >>> grid.z
        array([-10.,   0.])

        Set the z coordinates of a RectilinearGrid.

        >>> grid.z = [-10.0, 0.0, 10.0]
        >>> grid.z
        array([-10.,   0.,  10.])

        """
        return helpers.convert_array(self.GetZCoordinates())

    @z.setter
    def z(self, coords: Sequence):
        """Set the coordinates along the Z-direction."""
        self.SetZCoordinates(helpers.convert_array(coords))
        self._update_dimensions()
        self.Modified()

    @Grid.dimensions.setter  # type: ignore
    def dimensions(self, dims):
        """Do not let the dimensions of the RectilinearGrid be set."""
        raise AttributeError(
            "The dimensions of a `RectilinearGrid` are implicitly "
            "defined and thus cannot be set."
        )

    def cast_to_structured_grid(self) -> 'pyvista.StructuredGrid':
        """Cast this rectilinear grid to a structured grid.

        Returns
        -------
        pyvista.StructuredGrid
            This grid as a structured grid.

        """
        alg = _vtk.vtkRectilinearGridToPointSet()
        alg.SetInputData(self)
        alg.Update()
        return _get_output(alg)


class UniformGrid(_vtk.vtkImageData, Grid, UniformGridFilters):
    """Models datasets with uniform spacing in the three coordinate directions.

    Can be initialized in one of several ways:

    - Create empty grid
    - Initialize from a vtk.vtkImageData object
    - Initialize based on dimensions, cell spacing, and origin.

    .. versionchanged:: 0.33.0
        First argument must now be either a path or
        ``vtk.vtkImageData``. Use keyword arguments to specify the
        dimensions, spacing, and origin of the uniform grid.

    .. versionchanged:: 0.37.0
        The ``dims`` parameter has been renamed to ``dimensions``.

    Parameters
    ----------
    uinput : str, vtk.vtkImageData, pyvista.UniformGrid, optional
        Filename or dataset to initialize the uniform grid from.  If
        set, remainder of arguments are ignored.

    dimensions : iterable, optional
        Dimensions of the uniform grid.

    spacing : iterable, optional
        Spacing of the uniform in each dimension.  Defaults to
        ``(1.0, 1.0, 1.0)``. Must be positive.

    origin : iterable, optional
        Origin of the uniform grid.  Defaults to ``(0.0, 0.0, 0.0)``.

    deep : bool, optional
        Whether to deep copy a ``vtk.vtkImageData`` object.
        Default is ``False``.  Keyword only.

    Examples
    --------
    Create an empty UniformGrid.

    >>> import pyvista
    >>> grid = pyvista.UniformGrid()

    Initialize from a ``vtk.vtkImageData`` object.

    >>> import vtk
    >>> vtkgrid = vtk.vtkImageData()
    >>> grid = pyvista.UniformGrid(vtkgrid)

    Initialize using using just the grid dimensions and default
    spacing and origin. These must be keyword arguments.

    >>> grid = pyvista.UniformGrid(dimensions=(10, 10, 10))

    Initialize using dimensions and spacing.

    >>> grid = pyvista.UniformGrid(
    ...     dimensions=(10, 10, 10),
    ...     spacing=(2, 1, 5),
    ... )

    Initialize using dimensions, spacing, and an origin.

    >>> grid = pyvista.UniformGrid(
    ...     dimensions=(10, 10, 10),
    ...     spacing=(2, 1, 5),
    ...     origin=(10, 35, 50),
    ... )

    Initialize from another UniformGrid.

    >>> grid = pyvista.UniformGrid(
    ...     dimensions=(10, 10, 10),
    ...     spacing=(2, 1, 5),
    ...     origin=(10, 35, 50),
    ... )
    >>> grid_from_grid = pyvista.UniformGrid(grid)
    >>> grid_from_grid == grid
    True

    """

    _WRITERS = {'.vtk': _vtk.vtkDataSetWriter, '.vti': _vtk.vtkXMLImageDataWriter}

    def __init__(
        self,
        uinput=None,
        *args,
        dimensions=None,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        deep=False,
        **kwargs,
    ):
        """Initialize the uniform grid."""
        super().__init__()

        # permit old behavior
        if isinstance(uinput, Sequence) and not isinstance(uinput, str):
            warnings.warn(
                "Behavior of pyvista.UniformGrid has changed. First argument must be "
                "either a ``vtk.vtkImageData`` or path.",
                PyVistaDeprecationWarning,
            )
            dimensions = uinput
            uinput = None

        if dimensions is None and 'dims' in kwargs:
            dimensions = kwargs.pop('dims')
            warnings.warn(
                '`dims` argument is deprecated. Please use `dimensions`.', PyVistaDeprecationWarning
            )
        assert_empty_kwargs(**kwargs)

        if args:
            warnings.warn(
                "Behavior of pyvista.UniformGrid has changed. Use keyword arguments "
                "to specify dimensions, spacing, and origin. For example:\n\n"
                "    >>> grid = pyvista.UniformGrid(\n"
                "    ...     dimensions=(10, 10, 10),\n"
                "    ...     spacing=(2, 1, 5),\n"
                "    ...     origin=(10, 35, 50),\n"
                "    ... )\n",
                PyVistaDeprecationWarning,
            )
            origin = args[0]
            if len(args) > 1:
                spacing = args[1]
            if len(args) > 2:
                raise ValueError(
                    "Too many additional arguments specified for UniformGrid. "
                    f"Accepts at most 2, and {len(args)} have been input."
                )

        # first argument must be either vtkImageData or a path
        if uinput is not None:
            if isinstance(uinput, _vtk.vtkImageData):
                if deep:
                    self.deep_copy(uinput)
                else:
                    self.shallow_copy(uinput)
            elif isinstance(uinput, (str, pathlib.Path)):
                self._from_file(uinput)
            else:
                raise TypeError(
                    "First argument, ``uinput`` must be either ``vtk.vtkImageData`` "
                    f"or a path, not {type(uinput)}.  Use keyword arguments to "
                    "specify dimensions, spacing, and origin. For example:\n\n"
                    "    >>> grid = pyvista.UniformGrid(\n"
                    "    ...     dimensions=(10, 10, 10),\n"
                    "    ...     spacing=(2, 1, 5),\n"
                    "    ...     origin=(10, 35, 50),\n"
                    "    ... )\n"
                )
        elif dimensions is not None:
            self._from_specs(dimensions, spacing, origin)

    def __repr__(self):
        """Return the default representation."""
        return DataSet.__repr__(self)

    def __str__(self):
        """Return the default str representation."""
        return DataSet.__str__(self)

    def _from_specs(self, dims: Sequence[int], spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
        """Create VTK image data directly from numpy arrays.

        A uniform grid is defined by the point spacings for each axis
        (uniform along each individual axis) and the number of points on each axis.
        These are relative to a specified origin (default is ``(0.0, 0.0, 0.0)``).

        Parameters
        ----------
        dims : tuple(int)
            Length 3 tuple of ints specifying how many points along each axis.

        spacing : tuple(float)
            Length 3 tuple of floats/ints specifying the point spacings
            for each axis. Must be positive.

        origin : tuple(float)
            Length 3 tuple of floats/ints specifying minimum value for each axis.

        """
        xn, yn, zn = dims[0], dims[1], dims[2]
        xo, yo, zo = origin[0], origin[1], origin[2]
        self.SetDimensions(xn, yn, zn)
        self.SetOrigin(xo, yo, zo)
        self.spacing = (spacing[0], spacing[1], spacing[2])

    @property  # type: ignore
    def points(self) -> np.ndarray:  # type: ignore
        """Build a copy of the implicitly defined points as a numpy array.

        Notes
        -----
        The ``points`` for a :class:`pyvista.UniformGrid` cannot be set.

        Examples
        --------
        >>> import pyvista
        >>> grid = pyvista.UniformGrid(dimensions=(2, 2, 2))
        >>> grid.points
        array([[0., 0., 0.],
               [1., 0., 0.],
               [0., 1., 0.],
               [1., 1., 0.],
               [0., 0., 1.],
               [1., 0., 1.],
               [0., 1., 1.],
               [1., 1., 1.]])

        """
        # Get grid dimensions
        nx, ny, nz = self.dimensions
        nx -= 1
        ny -= 1
        nz -= 1
        # get the points and convert to spacings
        dx, dy, dz = self.spacing
        # Now make the cell arrays
        ox, oy, oz = np.array(self.origin) + np.array(self.extent[::2])  # type: ignore
        x = np.insert(np.cumsum(np.full(nx, dx)), 0, 0.0) + ox
        y = np.insert(np.cumsum(np.full(ny, dy)), 0, 0.0) + oy
        z = np.insert(np.cumsum(np.full(nz, dz)), 0, 0.0) + oz
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        return np.c_[xx.ravel(order='F'), yy.ravel(order='F'), zz.ravel(order='F')]

    @points.setter
    def points(self, points):
        """Points cannot be set.

        This setter overrides the base class's setter to ensure a user does not
        attempt to set them. See https://github.com/pyvista/pyvista/issues/713.

        """
        raise AttributeError(
            "The points cannot be set. The points of "
            "`UniformGrid`/`vtkImageData` are implicitly defined by the "
            "`origin`, `spacing`, and `dimensions` of the grid."
        )

    @property
    def x(self) -> np.ndarray:
        """Return all the X points.

        Examples
        --------
        >>> import pyvista
        >>> grid = pyvista.UniformGrid(dimensions=(2, 2, 2))
        >>> grid.x
        array([0., 1., 0., 1., 0., 1., 0., 1.])

        """
        return self.points[:, 0]

    @property
    def y(self) -> np.ndarray:
        """Return all the Y points.

        Examples
        --------
        >>> import pyvista
        >>> grid = pyvista.UniformGrid(dimensions=(2, 2, 2))
        >>> grid.y
        array([0., 0., 1., 1., 0., 0., 1., 1.])

        """
        return self.points[:, 1]

    @property
    def z(self) -> np.ndarray:
        """Return all the Z points.

        Examples
        --------
        >>> import pyvista
        >>> grid = pyvista.UniformGrid(dimensions=(2, 2, 2))
        >>> grid.z
        array([0., 0., 0., 0., 1., 1., 1., 1.])

        """
        return self.points[:, 2]

    @property
    def origin(self) -> Tuple[float]:
        """Return the origin of the grid (bottom southwest corner).

        Examples
        --------
        >>> import pyvista
        >>> grid = pyvista.UniformGrid(dimensions=(5, 5, 5))
        >>> grid.origin
        (0.0, 0.0, 0.0)

        Show how the origin is in the bottom "southwest" corner of the
        UniformGrid.

        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(grid, show_edges=True)
        >>> _ = pl.add_axes_at_origin(ylabel=None)
        >>> pl.camera_position = 'xz'
        >>> pl.show()

        Set the origin to ``(1, 1, 1)`` and show how this shifts the
        UniformGrid.

        >>> grid.origin = (1, 1, 1)
        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(grid, show_edges=True)
        >>> _ = pl.add_axes_at_origin(ylabel=None)
        >>> pl.camera_position = 'xz'
        >>> pl.show()

        """
        return self.GetOrigin()

    @origin.setter
    def origin(self, origin: Sequence[Union[float, int]]):
        """Set the origin."""
        self.SetOrigin(origin[0], origin[1], origin[2])
        self.Modified()

    @property
    def spacing(self) -> Tuple[float, float, float]:
        """Return or set the spacing for each axial direction.

        Notes
        -----
        Spacing must be non-negative. While VTK accepts negative
        spacing, this results in unexpected behavior. See:
        https://github.com/pyvista/pyvista/issues/1967

        Examples
        --------
        Create a 5 x 5 x 5 uniform grid.

        >>> import pyvista
        >>> grid = pyvista.UniformGrid(dimensions=(5, 5, 5))
        >>> grid.spacing
        (1.0, 1.0, 1.0)
        >>> grid.plot(show_edges=True)

        Modify the spacing to ``(1, 2, 3)``

        >>> grid.spacing = (1, 2, 3)
        >>> grid.plot(show_edges=True)

        """
        return self.GetSpacing()

    @spacing.setter
    def spacing(self, spacing: Sequence[Union[float, int]]):
        """Set spacing."""
        if min(spacing) < 0:
            raise ValueError(f"Spacing must be non-negative, got {spacing}")
        self.SetSpacing(*spacing)
        self.Modified()

    def _get_attrs(self):
        """Return the representation methods (internal helper)."""
        attrs = Grid._get_attrs(self)
        fmt = "{}, {}, {}".format(*[pyvista.FLOAT_FORMAT] * 3)
        attrs.append(("Spacing", self.spacing, fmt))
        return attrs

    def cast_to_structured_grid(self) -> 'pyvista.StructuredGrid':
        """Cast this uniform grid to a structured grid.

        Returns
        -------
        pyvista.StructuredGrid
            This grid as a structured grid.

        """
        alg = _vtk.vtkImageToStructuredGrid()
        alg.SetInputData(self)
        alg.Update()
        return _get_output(alg)

    def cast_to_rectilinear_grid(self) -> 'RectilinearGrid':
        """Cast this uniform grid to a rectilinear grid.

        Returns
        -------
        pyvista.RectilinearGrid
            This uniform grid as a rectilinear grid.

        """

        def gen_coords(i):
            coords = (
                np.cumsum(np.insert(np.full(self.dimensions[i] - 1, self.spacing[i]), 0, 0))
                + self.origin[i]
            )
            return coords

        xcoords = gen_coords(0)
        ycoords = gen_coords(1)
        zcoords = gen_coords(2)
        grid = pyvista.RectilinearGrid(xcoords, ycoords, zcoords)
        grid.point_data.update(self.point_data)
        grid.cell_data.update(self.cell_data)
        grid.field_data.update(self.field_data)
        grid.copy_meta_from(self, deep=True)
        return grid

    @property
    def extent(self) -> tuple:
        """Return or set the extent of the UniformGrid.

        The extent is simply the first and last indices for each of the three axes.

        Examples
        --------
        Create a ``UniformGrid`` and show its extent.

        >>> import pyvista
        >>> grid = pyvista.UniformGrid(dimensions=(10, 10, 10))
        >>> grid.extent
        (0, 9, 0, 9, 0, 9)

        >>> grid.extent = (2, 5, 2, 5, 2, 5)
        >>> grid.extent
        (2, 5, 2, 5, 2, 5)

        Note how this also modifies the grid bounds and dimensions. Since we
        use default spacing of 1 here, the bounds match the extent exactly.

        >>> grid.bounds
        (2.0, 5.0, 2.0, 5.0, 2.0, 5.0)
        >>> grid.dimensions
        (4, 4, 4)

        """
        return self.GetExtent()

    @extent.setter
    def extent(self, new_extent: Sequence[int]):
        """Set the extent of the UniformGrid."""
        if len(new_extent) != 6:
            raise ValueError('Extent must be a vector of 6 values.')
        self.SetExtent(new_extent)

    @wraps(RectilinearGridFilters.to_tetrahedra)
    def to_tetrahedra(self, *args, **kwargs):
        """Cast to a rectangular grid and then convert to tetrahedra."""
        return self.cast_to_rectilinear_grid().to_tetrahedra(*args, **kwargs)
