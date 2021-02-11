"""Sub-classes for vtk.vtkRectilinearGrid and vtk.vtkImageData."""
import pathlib
import logging

import numpy as np
import vtk
from vtk import vtkImageData, vtkRectilinearGrid
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import pyvista
from pyvista.utilities import abstract_class
from .common import Common
from .filters import _get_output, UniformGridFilters

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')


@abstract_class
class Grid(Common):
    """A class full of common methods for non-pointset grids."""

    def __init__(self, *args, **kwargs):
        """Initialize the grid."""
        super().__init__()

    @property
    def dimensions(self):
        """Return a length 3 tuple of the grid's dimensions.

        These are effectively the number of nodes along each of the three dataset axes.

        """
        return list(self.GetDimensions())

    @dimensions.setter
    def dimensions(self, dims):
        """Set the dataset dimensions. Pass a length three tuple of integers."""
        nx, ny, nz = dims[0], dims[1], dims[2]
        self.SetDimensions(nx, ny, nz)
        self.Modified()

    def _get_attrs(self):
        """Return the representation methods (internal helper)."""
        attrs = Common._get_attrs(self)
        attrs.append(("Dimensions", self.dimensions, "{:d}, {:d}, {:d}"))
        return attrs


class RectilinearGrid(vtkRectilinearGrid, Grid):
    """Extend the functionality of a vtk.vtkRectilinearGrid object.

    Can be initialized in several ways:

    - Create empty grid
    - Initialize from a vtk.vtkRectilinearGrid object
    - Initialize directly from the point arrays

    See _from_arrays in the documentation for more details on initializing
    from point arrays

    Examples
    --------
    >>> import pyvista
    >>> import vtk
    >>> import numpy as np

    >>> # Create empty grid
    >>> grid = pyvista.RectilinearGrid()

    >>> # Initialize from a vtk.vtkRectilinearGrid object
    >>> vtkgrid = vtk.vtkRectilinearGrid()
    >>> grid = pyvista.RectilinearGrid(vtkgrid)

    >>> # Create from NumPy arrays
    >>> xrng = np.arange(-10, 10, 2)
    >>> yrng = np.arange(-10, 10, 5)
    >>> zrng = np.arange(-10, 10, 1)
    >>> grid = pyvista.RectilinearGrid(xrng, yrng, zrng)


    """

    _READERS = {'.vtk': vtk.vtkRectilinearGridReader, '.vtr': vtk.vtkXMLRectilinearGridReader}
    _WRITERS = {'.vtk': vtk.vtkRectilinearGridWriter, '.vtr': vtk.vtkXMLRectilinearGridWriter}

    def __init__(self, *args, **kwargs):
        """Initialize the rectilinear grid."""
        super().__init__()

        if len(args) == 1:
            if isinstance(args[0], vtk.vtkRectilinearGrid):
                self.deep_copy(args[0])
            elif isinstance(args[0], (str, pathlib.Path)):
                self._from_file(args[0])
            elif isinstance(args[0], np.ndarray):
                self._from_arrays(args[0], None, None)
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
                self._from_arrays(args[0], args[1], args[2])
            elif all([arg0_is_arr, arg1_is_arr]):
                self._from_arrays(args[0], args[1], None)
            else:
                raise TypeError("Arguments not understood by `RectilinearGrid`.")

    def __repr__(self):
        """Return the default representation."""
        return Common.__repr__(self)

    def __str__(self):
        """Return the str representation."""
        return Common.__str__(self)

    def _update_dimensions(self):
        """Update the dimensions if coordinates have changed."""
        return self.SetDimensions(len(self.x), len(self.y), len(self.z))

    def _from_arrays(self, x, y, z):
        """Create VTK rectilinear grid directly from numpy arrays.

        Each array gives the uniques coordinates of the mesh along each axial
        direction. To help ensure you are using this correctly, we take the unique
        values of each argument.

        Parameters
        ----------
        x : np.ndarray
            Coordinates of the nodes in x direction.

        y : np.ndarray
            Coordinates of the nodes in y direction.

        z : np.ndarray
            Coordinates of the nodes in z direction.

        """
        # Set the coordinates along each axial direction
        # Must at least be an x array
        x = np.unique(x.ravel())
        self.SetXCoordinates(numpy_to_vtk(x))
        if y is not None:
            y = np.unique(y.ravel())
            self.SetYCoordinates(numpy_to_vtk(y))
        if z is not None:
            z = np.unique(z.ravel())
            self.SetZCoordinates(numpy_to_vtk(z))
        # Ensure dimensions are properly set
        self._update_dimensions()

    @property
    def meshgrid(self):
        """Return a meshgrid of numpy arrays for this mesh.

        This simply returns a ``numpy.meshgrid`` of the coordinates for this
        mesh in ``ij`` indexing. These are a copy of the points of this mesh.

        """
        return np.meshgrid(self.x, self.y, self.z, indexing='ij')

    @property
    def points(self):
        """Return a copy of the points as an n by 3 numpy array."""
        xx, yy, zz = self.meshgrid
        return np.c_[xx.ravel(order='F'), yy.ravel(order='F'), zz.ravel(order='F')]

    @points.setter
    def points(self, points):
        """Points must be set along each axial direction.

        Please set the point coordinates with the ``x``, ``y``, and ``z``
        setters.

        This setter overrides the base class's setter to ensure a user does not
        attempt to set them.

        """
        raise AttributeError("The points cannot be set. The points of "
            "`RectilinearGrid` are defined in each axial direction. Please "
            "use the `x`, `y`, and `z` setters individually."
            )

    @property
    def x(self):
        """Get the coordinates along the X-direction."""
        return vtk_to_numpy(self.GetXCoordinates())

    @x.setter
    def x(self, coords):
        """Set the coordinates along the X-direction."""
        self.SetXCoordinates(numpy_to_vtk(coords))
        self._update_dimensions()
        self.Modified()

    @property
    def y(self):
        """Get the coordinates along the Y-direction."""
        return vtk_to_numpy(self.GetYCoordinates())

    @y.setter
    def y(self, coords):
        """Set the coordinates along the Y-direction."""
        self.SetYCoordinates(numpy_to_vtk(coords))
        self._update_dimensions()
        self.Modified()

    @property
    def z(self):
        """Get the coordinates along the Z-direction."""
        return vtk_to_numpy(self.GetZCoordinates())

    @z.setter
    def z(self, coords):
        """Set the coordinates along the Z-direction."""
        self.SetZCoordinates(numpy_to_vtk(coords))
        self._update_dimensions()
        self.Modified()

    @Grid.dimensions.setter  # type: ignore
    def dimensions(self, dims):
        """Do not let the dimensions of the RectilinearGrid be set."""
        raise AttributeError("The dimensions of a `RectilinearGrid` are implicitly defined and thus cannot be set.")

    def cast_to_structured_grid(self):
        """Cast this rectilinear grid to a :class:`pyvista.StructuredGrid`."""
        alg = vtk.vtkRectilinearGridToPointSet()
        alg.SetInputData(self)
        alg.Update()
        return _get_output(alg)


class UniformGrid(vtkImageData, Grid, UniformGridFilters):
    """Extend the functionality of a vtk.vtkImageData object.

    Can be initialized in several ways:

    - Create empty grid
    - Initialize from a vtk.vtkImageData object
    - Initialize directly from the point arrays

    See ``_from_specs`` in the documentation for more details on initializing
    from point arrays

    Examples
    --------
    >>> import pyvista
    >>> import vtk
    >>> import numpy as np

    >>> # Create empty grid
    >>> grid = pyvista.UniformGrid()

    >>> # Initialize from a vtk.vtkImageData object
    >>> vtkgrid = vtk.vtkImageData()
    >>> grid = pyvista.UniformGrid(vtkgrid)

    >>> # Using just the grid dimensions
    >>> dims = (10, 10, 10)
    >>> grid = pyvista.UniformGrid(dims)

    >>> # Using dimensions and spacing
    >>> spacing = (2, 1, 5)
    >>> grid = pyvista.UniformGrid(dims, spacing)

    >>> # Using dimensions, spacing, and an origin
    >>> origin = (10, 35, 50)
    >>> grid = pyvista.UniformGrid(dims, spacing, origin)

    """

    _READERS = {'.vtk': vtk.vtkDataSetReader, '.vti': vtk.vtkXMLImageDataReader}
    _WRITERS = {'.vtk': vtk.vtkDataSetWriter, '.vti': vtk.vtkXMLImageDataWriter}

    def __init__(self, *args, **kwargs):
        """Initialize the uniform grid."""
        super().__init__()

        if len(args) == 1:
            if isinstance(args[0], vtk.vtkImageData):
                self.deep_copy(args[0])
            elif isinstance(args[0], (str, pathlib.Path)):
                self._from_file(args[0])
            else:
                arg0_is_valid = len(args[0]) == 3
                self._from_specs(args[0])

        elif len(args) > 1 and len(args) < 4:
            arg0_is_valid = len(args[0]) == 3
            arg1_is_valid = False
            if len(args) > 1:
                arg1_is_valid = len(args[1]) == 3
            arg2_is_valid = False
            if len(args) > 2:
                arg2_is_valid = len(args[2]) == 3

            if all([arg0_is_valid, arg1_is_valid, arg2_is_valid]):
                self._from_specs(args[0], args[1], args[2])
            elif all([arg0_is_valid, arg1_is_valid]):
                self._from_specs(args[0], args[1])

    def __repr__(self):
        """Return the default representation."""
        return Common.__repr__(self)

    def __str__(self):
        """Return the default str representation."""
        return Common.__str__(self)

    def _from_specs(self, dims, spacing=(1.0,1.0,1.0), origin=(0.0, 0.0, 0.0)):
        """Create VTK image data directly from numpy arrays.

        A uniform grid is defined by the node spacings for each axis
        (uniform along each individual axis) and the number of nodes on each axis.
        These are relative to a specified origin (default is ``(0.0, 0.0, 0.0)``).

        Parameters
        ----------
        dims : tuple(int)
            Length 3 tuple of ints specifying how many nodes along each axis

        spacing : tuple(float)
            Length 3 tuple of floats/ints specifying the node spacings for each axis

        origin : tuple(float)
            Length 3 tuple of floats/ints specifying minimum value for each axis

        """
        xn, yn, zn = dims[0], dims[1], dims[2]
        xs, ys, zs = spacing[0], spacing[1], spacing[2]
        xo, yo, zo = origin[0], origin[1], origin[2]
        self.SetDimensions(xn, yn, zn)
        self.SetOrigin(xo, yo, zo)
        self.SetSpacing(xs, ys, zs)

    @property
    def points(self):
        """Build a copy of the implicitly defined points as a numpy array."""
        # Get grid dimensions
        nx, ny, nz = self.dimensions
        nx -= 1
        ny -= 1
        nz -= 1
        # get the points and convert to spacings
        dx, dy, dz = self.spacing
        # Now make the cell arrays
        ox, oy, oz = np.array(self.origin) + np.array(self.extent[::2])
        x = np.insert(np.cumsum(np.full(nx, dx)), 0, 0.0) + ox
        y = np.insert(np.cumsum(np.full(ny, dy)), 0, 0.0) + oy
        z = np.insert(np.cumsum(np.full(nz, dz)), 0, 0.0) + oz
        xx, yy, zz = np.meshgrid(x,y,z, indexing='ij')
        return np.c_[xx.ravel(order='F'), yy.ravel(order='F'), zz.ravel(order='F')]

    @points.setter
    def points(self, points):
        """Points cannot be set.

        This setter overrides the base class's setter to ensure a user does not
        attempt to set them. See https://github.com/pyvista/pyvista/issues/713.

        """
        raise AttributeError("The points cannot be set. The points of "
            "`UniformGrid`/`vtkImageData` are implicitly defined by the "
            "`origin`, `spacing`, and `dimensions` of the grid."
            )

    @property
    def x(self):
        """Return all the X points."""
        return self.points[:, 0]

    @property
    def y(self):
        """Return all the Y points."""
        return self.points[:, 1]

    @property
    def z(self):
        """Return all the Z points."""
        return self.points[:, 2]

    @property
    def origin(self):
        """Return the origin of the grid (bottom southwest corner)."""
        return list(self.GetOrigin())

    @origin.setter
    def origin(self, origin):
        """Set the origin. Pass a length three tuple of floats."""
        ox, oy, oz = origin[0], origin[1], origin[2]
        self.SetOrigin(ox, oy, oz)
        self.Modified()

    @property
    def spacing(self):
        """Get the spacing for each axial direction."""
        return list(self.GetSpacing())

    @spacing.setter
    def spacing(self, spacing):
        """Set the spacing in each axial direction.

        Pass a length three tuple of floats.

        """
        dx, dy, dz = spacing[0], spacing[1], spacing[2]
        self.SetSpacing(dx, dy, dz)
        self.Modified()

    def _get_attrs(self):
        """Return the representation methods (internal helper)."""
        attrs = Grid._get_attrs(self)
        fmt = "{}, {}, {}".format(*[pyvista.FLOAT_FORMAT]*3)
        attrs.append(("Spacing", self.spacing, fmt))
        return attrs

    def cast_to_structured_grid(self):
        """Cast this uniform grid to a :class:`pyvista.StructuredGrid`."""
        alg = vtk.vtkImageToStructuredGrid()
        alg.SetInputData(self)
        alg.Update()
        return _get_output(alg)

    def cast_to_rectilinear_grid(self):
        """Cast this uniform grid to a :class:`pyvista.RectilinearGrid`."""
        def gen_coords(i):
            coords = np.cumsum(np.insert(np.full(self.dimensions[i] - 1,
                                                 self.spacing[i]), 0, 0)
                               ) + self.origin[i]
            return coords
        xcoords = gen_coords(0)
        ycoords = gen_coords(1)
        zcoords = gen_coords(2)
        grid = pyvista.RectilinearGrid(xcoords, ycoords, zcoords)
        grid.point_arrays.update(self.point_arrays)
        grid.cell_arrays.update(self.cell_arrays)
        grid.field_arrays.update(self.field_arrays)
        grid.copy_meta_from(self)
        return grid
