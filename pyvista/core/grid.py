"""Sub-classes for vtk.vtkRectilinearGrid and vtk.vtkImageData."""

import logging
import os

import numpy as np
import vtk
from vtk import vtkImageData, vtkRectilinearGrid
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import pyvista

from .common import Common
from .filters import _get_output, UniformGridFilters

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')


class Grid(Common):
    """A class full of common methods for non-pointset grids."""

    def __new__(cls, *args, **kwargs):
        """Allocate a grid."""
        if cls is Grid:
            raise TypeError("pyvista.Grid is an abstract class and may not be instantiated.")
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        """Initialize the grid."""
        super(Grid, self).__init__()

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

    def __init__(self, *args, **kwargs):
        """Initialize the rectilinear grid."""
        super(RectilinearGrid, self).__init__()

        if len(args) == 1:
            if isinstance(args[0], vtk.vtkRectilinearGrid):
                self.deep_copy(args[0])
            elif isinstance(args[0], str):
                self._load_file(args[0])
            elif isinstance(args[0], np.ndarray):
                self._from_arrays(args[0], None, None)
            else:
                raise TypeError("Type ({}) not understood by `RectilinearGrid`.".format(type(args[0])))

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
        """Return the a meshgrid of numpy arrays for this mesh.

        This simply returns a ``numpy.meshgrid`` of the coordinates for this
        mesh in ``ij`` indexing.

        """
        return np.meshgrid(self.x, self.y, self.z, indexing='ij')


    @property
    def points(self):
        """Return all of the points as an n by 3 numpy array."""
        xx, yy, zz = self.meshgrid
        return np.c_[xx.ravel(order='F'), yy.ravel(order='F'), zz.ravel(order='F')]


    @points.setter
    def points(self, points):
        """Set points without copying."""
        if not isinstance(points, np.ndarray):
            raise TypeError('Points must be a numpy array')
        # get the unique coordinates along each axial direction
        x = np.unique(points[:,0])
        y = np.unique(points[:,1])
        z = np.unique(points[:,2])
        # Set the vtk coordinates
        self._from_arrays(x, y, z)
        #self._point_ref = points
        self.Modified()


    def _load_file(self, filename):
        """
        Load a rectilinear grid from a file.

        The file extension will select the type of reader to use.  A .vtk
        extension will use the legacy reader, while .vtr will select the VTK
        XML reader.

        Parameters
        ----------
        filename : str
            Filename of grid to be loaded.

        """
        filename = os.path.abspath(os.path.expanduser(filename))
        # check file exists
        if not os.path.isfile(filename):
            raise Exception('{} does not exist'.format(filename))

        # Check file extension
        if '.vtr' in filename:
            legacy_writer = False
        elif '.vtk' in filename:
            legacy_writer = True
        else:
            raise Exception(
                'Extension should be either ".vtr" (xml) or ".vtk" (legacy)')

        # Create reader
        if legacy_writer:
            reader = vtk.vtkRectilinearGridReader()
        else:
            reader = vtk.vtkXMLRectilinearGridReader()

        # load file to self
        reader.SetFileName(filename)
        reader.Update()
        grid = reader.GetOutput()
        self.shallow_copy(grid)

    def save(self, filename, binary=True):
        """Write a rectilinear grid to disk.

        Parameters
        ----------
        filename : str
            Filename of grid to be written.  The file extension will select the
            type of writer to use.  ".vtk" will use the legacy writer, while
            ".vtr" will select the VTK XML writer.

        binary : bool, optional
            Writes as a binary file by default.  Set to False to write ASCII.


        Notes
        -----
        Binary files write much faster than ASCII, but binary files written on
        one system may not be readable on other systems.  Binary can be used
        only with the legacy writer.

        """
        filename = os.path.abspath(os.path.expanduser(filename))
        # Use legacy writer if vtk is in filename
        if '.vtk' in filename:
            writer = vtk.vtkRectilinearGridWriter()
            legacy = True
        elif '.vtr' in filename:
            writer = vtk.vtkXMLRectilinearGridWriter()
            legacy = False
        else:
            raise Exception('Extension should be either ".vtr" (xml) or'
                            '".vtk" (legacy)')
        # Write
        writer.SetFileName(filename)
        writer.SetInputData(self)
        if binary and legacy:
            writer.SetFileTypeToBinary()
        writer.Write()

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


    @Grid.dimensions.setter
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

    def __init__(self, *args, **kwargs):
        """Initialize the uniform grid."""
        super(UniformGrid, self).__init__()

        if len(args) == 1:
            if isinstance(args[0], vtk.vtkImageData):
                self.deep_copy(args[0])
            elif isinstance(args[0], str):
                self._load_file(args[0])
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
        """Return a pointer to the points as a numpy object."""
        # Get grid dimensions
        nx, ny, nz = self.dimensions
        nx -= 1
        ny -= 1
        nz -= 1
        # get the points and convert to spacings
        dx, dy, dz = self.spacing
        # Now make the cell arrays
        ox, oy, oz = self.origin
        x = np.insert(np.cumsum(np.full(nx, dx)), 0, 0.0) + ox
        y = np.insert(np.cumsum(np.full(ny, dy)), 0, 0.0) + oy
        z = np.insert(np.cumsum(np.full(nz, dz)), 0, 0.0) + oz
        xx, yy, zz = np.meshgrid(x,y,z, indexing='ij')
        return np.c_[xx.ravel(order='F'), yy.ravel(order='F'), zz.ravel(order='F')]

    @points.setter
    def points(self, points):
        """Set points without copying."""
        if not isinstance(points, np.ndarray):
            raise TypeError('Points must be a numpy array')
        # get the unique coordinates along each axial direction
        x = np.unique(points[:,0])
        y = np.unique(points[:,1])
        z = np.unique(points[:,2])
        nx, ny, nz = len(x), len(y), len(z)
        # diff returns an empty array if the input is constant
        dx, dy, dz = [np.diff(d) if len(np.diff(d)) > 0 else d for d in (x, y, z)]
        # TODO: this needs to be tested (unique might return a tuple)
        dx, dy, dz = np.unique(dx), np.unique(dy), np.unique(dz)
        ox, oy, oz = np.min(x), np.min(y), np.min(z)
        # Build the vtk object
        self._from_specs((nx,ny,nz), (dx,dy,dz), (ox,oy,oz))
        #self._point_ref = points
        self.Modified()


    def _load_file(self, filename):
        """
        Load image data from a file.

        The file extension will select the type of reader to use.  A ``.vtk``
        extension will use the legacy reader, while ``.vti`` will select the VTK
        XML reader.

        Parameters
        ----------
        filename : str
            Filename of grid to be loaded.

        """
        filename = os.path.abspath(os.path.expanduser(filename))
        # check file exists
        if not os.path.isfile(filename):
            raise Exception('{} does not exist'.format(filename))

        # Check file extension
        if '.vti' in filename:
            legacy_writer = False
        elif '.vtk' in filename:
            legacy_writer = True
        else:
            raise Exception(
                'Extension should be either ".vti" (xml) or ".vtk" (legacy)')

        # Create reader
        if legacy_writer:
            reader = vtk.vtkDataSetReader()
        else:
            reader = vtk.vtkXMLImageDataReader()

        # load file to self
        reader.SetFileName(filename)
        reader.Update()
        grid = reader.GetOutput()
        self.shallow_copy(grid)

    def save(self, filename, binary=True):
        """Write image data grid to disk.

        Parameters
        ----------
        filename : str
            Filename of grid to be written.  The file extension will select the
            type of writer to use.  ".vtk" will use the legacy writer, while
            ".vti" will select the VTK XML writer.

        binary : bool, optional
            Writes as a binary file by default.  Set to False to write ASCII.


        Notes
        -----
        Binary files write much faster than ASCII, but binary files written on
        one system may not be readable on other systems.  Binary can be used
        only with the legacy writer.

        """
        filename = os.path.abspath(os.path.expanduser(filename))
        # Use legacy writer if vtk is in filename
        if '.vtk' in filename:
            writer = vtk.vtkDataSetWriter()
            legacy = True
        elif '.vti' in filename:
            writer = vtk.vtkXMLImageDataWriter()
            legacy = False
        else:
            raise Exception('Extension should be either ".vti" (xml) or'
                            '".vtk" (legacy)')
        # Write
        writer.SetFileName(filename)
        writer.SetInputData(self)
        if binary and legacy:
            writer.SetFileTypeToBinary()
        writer.Write()

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
