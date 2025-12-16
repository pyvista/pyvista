"""Sub-classes for :vtk:`vtkRectilinearGrid` and :vtk:`vtkImageData`."""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Literal
from typing import cast

import numpy as np

import pyvista as pv
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista._warn_external import warn_external
from pyvista.core import _validation
from pyvista.core.utilities.writer import BaseWriter
from pyvista.core.utilities.writer import BMPWriter
from pyvista.core.utilities.writer import DataSetWriter
from pyvista.core.utilities.writer import JPEGWriter
from pyvista.core.utilities.writer import NIFTIImageWriter
from pyvista.core.utilities.writer import PNGWriter
from pyvista.core.utilities.writer import PNMWriter
from pyvista.core.utilities.writer import RectilinearGridWriter
from pyvista.core.utilities.writer import TIFFWriter
from pyvista.core.utilities.writer import XMLImageDataWriter
from pyvista.core.utilities.writer import XMLRectilinearGridWriter

if TYPE_CHECKING:
    from typing_extensions import Self

    from pyvista import StructuredGrid
    from pyvista import UnstructuredGrid
    from pyvista import pyvista_ndarray
    from pyvista.core._typing_core import MatrixLike
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core import RotationLike
    from pyvista.core._typing_core import TransformLike
    from pyvista.core._typing_core import VectorLike


from . import _vtk_core as _vtk
from .dataset import DataSet
from .filters import ImageDataFilters
from .filters import RectilinearGridFilters
from .filters import _get_output
from .utilities.arrays import array_from_vtkmatrix
from .utilities.arrays import convert_array
from .utilities.arrays import raise_has_duplicates
from .utilities.arrays import vtkmatrix_from_array
from .utilities.misc import abstract_class


@abstract_class
class Grid(DataSet):
    """A class full of common methods for non-pointset grids."""

    @property
    def dimensions(self: Self) -> tuple[int, int, int]:
        """Return the grid's dimensions.

        These are effectively the number of points along each of the
        three dataset axes.

        Returns
        -------
        tuple[int]
            Dimensions of the grid.

        Examples
        --------
        Create a uniform grid with dimensions ``(1, 2, 3)``.

        >>> import pyvista as pv
        >>> grid = pv.ImageData(dimensions=(2, 3, 4))
        >>> grid.dimensions
        (2, 3, 4)
        >>> grid.plot(show_edges=True)

        Set the dimensions to ``(3, 4, 5)``

        >>> grid.dimensions = (3, 4, 5)
        >>> grid.plot(show_edges=True)

        """
        return self.GetDimensions()

    @dimensions.setter
    def dimensions(self: Self, dims: VectorLike[int]) -> None:
        self.SetDimensions(*dims)
        self.Modified()

    def _get_attrs(self: Self) -> list[tuple[str, Any, str]]:
        """Return the representation methods (internal helper)."""
        attrs = DataSet._get_attrs(self)
        attrs.append(('Dimensions', self.dimensions, '{:d}, {:d}, {:d}'))
        return attrs


class RectilinearGrid(Grid, RectilinearGridFilters, _vtk.vtkRectilinearGrid):
    """Dataset with variable spacing in the three coordinate directions.

    Can be initialized in several ways:

    * Create empty grid
    * Initialize from a :vtk:`vtkRectilinearGrid` object
    * Initialize directly from the point arrays

    Parameters
    ----------
    uinput : str, pathlib.Path, :vtk:`vtkRectilinearGrid`, numpy.ndarray, optional
        Filename, dataset, or array to initialize the rectilinear grid from. If a
        filename is passed, pyvista will attempt to load it as a
        :class:`RectilinearGrid`. If passed a :vtk:`vtkRectilinearGrid`, it
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
        Whether to deep copy a :vtk:`vtkRectilinearGrid` object.
        Default is ``False``.  Keyword only.

    Examples
    --------
    >>> import pyvista as pv
    >>> import vtk
    >>> import numpy as np

    Create an empty grid.

    >>> grid = pv.RectilinearGrid()

    Initialize from a :vtk:`vtkRectilinearGrid` object

    >>> vtkgrid = vtk.vtkRectilinearGrid()
    >>> grid = pv.RectilinearGrid(vtkgrid)

    Create from NumPy arrays.

    >>> xrng = np.arange(-10, 10, 2)
    >>> yrng = np.arange(-10, 10, 5)
    >>> zrng = np.arange(-10, 10, 1)
    >>> grid = pv.RectilinearGrid(xrng, yrng, zrng)
    >>> grid.plot(show_edges=True)

    """

    _WRITERS: ClassVar[dict[str, type[BaseWriter]]] = {
        '.vtk': RectilinearGridWriter,
        '.vtr': XMLRectilinearGridWriter,
    }

    def __init__(
        self: Self,
        *args,
        check_duplicates: bool = False,
        deep: bool = False,
        **kwargs,
    ) -> None:  # numpydoc ignore=PR01,RT01
        """Initialize the rectilinear grid."""
        super().__init__(**kwargs)

        if len(args) == 1:
            if isinstance(args[0], _vtk.vtkRectilinearGrid):
                if deep:
                    self.deep_copy(args[0])
                else:
                    self.shallow_copy(args[0])
            elif isinstance(args[0], (str, Path)):
                self._from_file(args[0], **kwargs)
            elif isinstance(args[0], (np.ndarray, Sequence)):
                self._from_arrays(
                    x=np.asanyarray(args[0]),
                    y=None,  # type: ignore[arg-type]
                    z=None,  # type: ignore[arg-type]
                    check_duplicates=check_duplicates,
                )
            else:
                msg = f'Type ({type(args[0])}) not understood by `RectilinearGrid`'
                raise TypeError(msg)

        elif len(args) == 3 or len(args) == 2:
            arg0_is_arr = isinstance(args[0], (np.ndarray, Sequence))
            arg1_is_arr = isinstance(args[1], (np.ndarray, Sequence))
            arg2_is_arr = isinstance(args[2], (np.ndarray, Sequence)) if len(args) == 3 else False

            if all([arg0_is_arr, arg1_is_arr, arg2_is_arr]):
                self._from_arrays(
                    x=np.asanyarray(args[0]),
                    y=np.asanyarray(args[1]),
                    z=np.asanyarray(args[2]),  # type: ignore[misc]
                    check_duplicates=check_duplicates,
                )
            elif all([arg0_is_arr, arg1_is_arr]):
                self._from_arrays(
                    x=np.asanyarray(args[0]),
                    y=np.asanyarray(args[1]),
                    z=None,  # type: ignore[arg-type]
                    check_duplicates=check_duplicates,
                )
            else:
                msg = 'Arguments not understood by `RectilinearGrid`.'
                raise TypeError(msg)

    def __repr__(self: Self) -> str:
        """Return the default representation."""
        return DataSet.__repr__(self)

    def __str__(self: Self) -> str:
        """Return the str representation."""
        return DataSet.__str__(self)

    def _update_dimensions(self: Self) -> None:
        """Update the dimensions if coordinates have changed."""
        self.SetDimensions(len(self.x), len(self.y), len(self.z))

    def _from_arrays(
        self: Self,
        *,
        x: NumpyArray[float],
        y: NumpyArray[float],
        z: NumpyArray[float],
        check_duplicates: bool = False,
    ) -> None:
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
        self.SetXCoordinates(convert_array(x.ravel()))
        if y is not None:
            if check_duplicates:
                raise_has_duplicates(y)
            if not np.issubdtype(y.dtype, np.floating):
                y = y.astype(float)
            self.SetYCoordinates(convert_array(y.ravel()))
        if z is not None:
            if check_duplicates:
                raise_has_duplicates(z)
            if not np.issubdtype(z.dtype, np.floating):
                z = z.astype(float)
            self.SetZCoordinates(convert_array(z.ravel()))
        # Ensure dimensions are properly set
        self._update_dimensions()

    @property
    def meshgrid(
        self: Self,
    ) -> tuple[NumpyArray[float], NumpyArray[float], NumpyArray[float]]:
        """Return a meshgrid of numpy arrays for this mesh.

        This simply returns a :func:`numpy.meshgrid` of the
        coordinates for this mesh in ``ij`` indexing. These are a copy
        of the points of this mesh.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Tuple of numpy arrays representing the points of this mesh.

        """
        # Converting to tuple needed to be consistent type across numpy version
        # Remove when support is dropped for numpy 1.x
        # We also know this is 3-length so make it so in typing
        out = tuple(np.meshgrid(self.x, self.y, self.z, indexing='ij'))
        # Python 3.8 does not allow subscripting tuple, but only used for type checking
        if TYPE_CHECKING:
            out = cast('tuple[NumpyArray[float], NumpyArray[float], NumpyArray[float]]', out)
        return out

    @property  # type: ignore[override]
    def points(self: Self) -> NumpyArray[float]:
        """Return a copy of the points as an ``(n, 3)`` numpy array.

        Returns
        -------
        numpy.ndarray
            Array of points.

        Notes
        -----
        Points of a :class:`pyvista.RectilinearGrid` cannot be
        set. Set point coordinates with :attr:`RectilinearGrid.x`,
        :attr:`RectilinearGrid.y`, or :attr:`RectilinearGrid.z`.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> xrng = np.arange(-10, 10, 10, dtype=float)
        >>> yrng = np.arange(-10, 10, 10, dtype=float)
        >>> zrng = np.arange(-10, 10, 10, dtype=float)
        >>> grid = pv.RectilinearGrid(xrng, yrng, zrng)
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
        if pv.vtk_version_info >= (9, 4, 0):
            return convert_array(self.GetPoints().GetData())

        xx, yy, zz = self.meshgrid
        return np.c_[xx.ravel(order='F'), yy.ravel(order='F'), zz.ravel(order='F')]

    @points.setter
    def points(
        self: Self,
        points: MatrixLike[float] | _vtk.vtkPoints,  # noqa: ARG002
    ) -> None:  # numpydoc ignore=PR01
        """Raise an AttributeError.

        This setter overrides the base class's setter to ensure a user
        does not attempt to set them.
        """
        msg = (
            'The points cannot be set. The points of '
            '`RectilinearGrid` are defined in each axial direction. Please '
            'use the `x`, `y`, and `z` setters individually.'
        )
        raise AttributeError(msg)

    @property
    def x(self: Self) -> NumpyArray[float]:
        """Return or set the coordinates along the X-direction.

        Returns
        -------
        numpy.ndarray
            Array of points along the X-direction.

        Examples
        --------
        Return the x coordinates of a RectilinearGrid.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> xrng = np.arange(-10, 10, 10, dtype=float)
        >>> yrng = np.arange(-10, 10, 10, dtype=float)
        >>> zrng = np.arange(-10, 10, 10, dtype=float)
        >>> grid = pv.RectilinearGrid(xrng, yrng, zrng)
        >>> grid.x
        array([-10.,   0.])

        Set the x coordinates of a RectilinearGrid.

        >>> grid.x = [-10.0, 0.0, 10.0]
        >>> grid.x
        array([-10.,   0.,  10.])

        """
        return convert_array(self.GetXCoordinates())

    @x.setter
    def x(self: Self, coords: VectorLike[float]) -> None:
        self.SetXCoordinates(convert_array(coords))
        self._update_dimensions()
        self.Modified()

    @property
    def y(self: Self) -> NumpyArray[float]:
        """Return or set the coordinates along the Y-direction.

        Returns
        -------
        numpy.ndarray
            Array of points along the Y-direction.

        Examples
        --------
        Return the y coordinates of a RectilinearGrid.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> xrng = np.arange(-10, 10, 10, dtype=float)
        >>> yrng = np.arange(-10, 10, 10, dtype=float)
        >>> zrng = np.arange(-10, 10, 10, dtype=float)
        >>> grid = pv.RectilinearGrid(xrng, yrng, zrng)
        >>> grid.y
        array([-10.,   0.])

        Set the y coordinates of a RectilinearGrid.

        >>> grid.y = [-10.0, 0.0, 10.0]
        >>> grid.y
        array([-10.,   0.,  10.])

        """
        return convert_array(self.GetYCoordinates())

    @y.setter
    def y(self: Self, coords: VectorLike[float]) -> None:
        self.SetYCoordinates(convert_array(coords))
        self._update_dimensions()
        self.Modified()

    @property
    def z(self: Self) -> NumpyArray[float]:
        """Return or set the coordinates along the Z-direction.

        Returns
        -------
        numpy.ndarray
            Array of points along the Z-direction.

        Examples
        --------
        Return the z coordinates of a RectilinearGrid.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> xrng = np.arange(-10, 10, 10, dtype=float)
        >>> yrng = np.arange(-10, 10, 10, dtype=float)
        >>> zrng = np.arange(-10, 10, 10, dtype=float)
        >>> grid = pv.RectilinearGrid(xrng, yrng, zrng)
        >>> grid.z
        array([-10.,   0.])

        Set the z coordinates of a RectilinearGrid.

        >>> grid.z = [-10.0, 0.0, 10.0]
        >>> grid.z
        array([-10.,   0.,  10.])

        """
        return convert_array(self.GetZCoordinates())

    @z.setter
    def z(self: Self, coords: VectorLike[float]) -> None:
        self.SetZCoordinates(convert_array(coords))
        self._update_dimensions()
        self.Modified()

    @Grid.dimensions.setter  # type: ignore[attr-defined]
    def dimensions(self: Self, _dims: VectorLike[int]) -> None:
        """Set Dimensions.

        Parameters
        ----------
        _dims : sequence
            Ignored dimensions.

        """
        msg = (
            'The dimensions of a `RectilinearGrid` are implicitly defined and thus cannot be set.'
        )
        raise AttributeError(msg)

    def cast_to_structured_grid(self: Self) -> StructuredGrid:
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


class ImageData(Grid, ImageDataFilters, _vtk.vtkImageData):
    """Models datasets with uniform spacing in the three coordinate directions.

    Can be initialized in one of several ways:

    - Create empty grid
    - Initialize from a :vtk:`vtkImageData` object
    - Initialize based on dimensions, cell spacing, and origin.

    .. versionchanged:: 0.33.0
        First argument must now be either a path or
        :vtk:`vtkImageData`. Use keyword arguments to specify the
        dimensions, spacing, and origin of the uniform grid.

    .. versionchanged:: 0.37.0
        The ``dims`` parameter has been renamed to ``dimensions``.

    Parameters
    ----------
    uinput : str | :vtk:`vtkImageData` | ImageData, optional
        Filename or dataset to initialize the uniform grid from.  If
        set, remainder of arguments are ignored.

    dimensions : sequence[int], optional
        :attr:`dimensions` of the uniform grid.

    spacing : sequence[float], default: (1.0, 1.0, 1.0)
        :attr:`spacing` of the uniform grid in each dimension. Must be positive.

    origin : sequence[float], default: (0.0, 0.0, 0.0)
        :attr:`origin` of the uniform grid.

    deep : bool, default: False
        Whether to deep copy a :vtk:`vtkImageData` object. Keyword only.

    direction_matrix : RotationLike, optional
        The :attr:`direction_matrix` is a 3x3 matrix which controls the orientation of
        the image data.

        .. versionadded:: 0.45

    offset : int | VectorLike[int], default: (0, 0, 0)
        The offset defines the minimum :attr:`extent` of the image. Offset values
        can be positive or negative. In physical space, the offset is relative
        to the image's :attr:`origin`.

        .. versionadded:: 0.45

    See Also
    --------
    :ref:`create_uniform_grid_example`

    Examples
    --------
    Create an empty ImageData.

    >>> import pyvista as pv
    >>> grid = pv.ImageData()

    Initialize from a :vtk:`vtkImageData` object.

    >>> import vtk
    >>> vtkgrid = vtk.vtkImageData()
    >>> grid = pv.ImageData(vtkgrid)

    Initialize using just the grid dimensions and default
    spacing and origin. These must be keyword arguments.

    >>> grid = pv.ImageData(dimensions=(10, 10, 10))

    Initialize using dimensions and spacing.

    >>> grid = pv.ImageData(
    ...     dimensions=(10, 10, 10),
    ...     spacing=(2, 1, 5),
    ... )

    Initialize using dimensions, spacing, and an origin.

    >>> grid = pv.ImageData(
    ...     dimensions=(10, 10, 10),
    ...     spacing=(2, 1, 5),
    ...     origin=(10, 35, 50),
    ... )

    Initialize from another ImageData.

    >>> grid = pv.ImageData(
    ...     dimensions=(10, 10, 10),
    ...     spacing=(2, 1, 5),
    ...     origin=(10, 35, 50),
    ... )
    >>> grid_from_grid = pv.ImageData(grid)
    >>> grid_from_grid == grid
    True

    """

    _WRITERS: ClassVar[dict[str, type[BaseWriter]]] = {
        '.bmp': BMPWriter,
        '.jpeg': JPEGWriter,
        '.jpg': JPEGWriter,
        '.nii': NIFTIImageWriter,
        '.nii.gz': NIFTIImageWriter,
        '.png': PNGWriter,
        '.pnm': PNMWriter,
        '.tif': TIFFWriter,
        '.tiff': TIFFWriter,
        '.vtk': DataSetWriter,
        '.vti': XMLImageDataWriter,
    }

    @_deprecate_positional_args(allowed=['uinput'])
    def __init__(  # noqa: PLR0917
        self: Self,
        uinput: ImageData | str | Path | None = None,
        dimensions: VectorLike[int] | None = None,
        spacing: VectorLike[float] = (1.0, 1.0, 1.0),
        origin: VectorLike[float] = (0.0, 0.0, 0.0),
        deep: bool = False,  # noqa: FBT001, FBT002
        direction_matrix: RotationLike | None = None,
        offset: int | VectorLike[int] | None = None,
    ) -> None:
        """Initialize the uniform grid."""
        super().__init__()

        # first argument must be either vtkImageData or a path
        if uinput is not None:
            if isinstance(uinput, _vtk.vtkImageData):
                if deep:
                    self.deep_copy(uinput)
                else:
                    self.shallow_copy(uinput)
            elif isinstance(uinput, (str, Path)):
                self._from_file(uinput)
            else:
                msg = (  # type: ignore[unreachable]
                    'First argument, ``uinput`` must be either ``vtkImageData`` '
                    f'or a path, not {type(uinput)}.  Use keyword arguments to '
                    'specify dimensions, spacing, and origin. For example:\n\n'
                    '    >>> grid = pv.ImageData(\n'
                    '    ...     dimensions=(10, 10, 10),\n'
                    '    ...     spacing=(2, 1, 5),\n'
                    '    ...     origin=(10, 35, 50),\n'
                    '    ... )\n'
                )
                raise TypeError(msg)
        else:
            if dimensions is not None:
                self.dimensions = dimensions
            self.origin = origin
            self.spacing = spacing
            if direction_matrix is not None:
                self.direction_matrix = direction_matrix
            if offset is not None:
                self.offset = offset

    def __repr__(self: Self) -> str:
        """Return the default representation."""
        return DataSet.__repr__(self)

    def __str__(self: Self) -> str:
        """Return the default str representation."""
        return DataSet.__str__(self)

    def __getitem__(  # type: ignore[override]
        self, key: tuple[str, Literal['cell', 'point', 'field']] | str | tuple[int, int, int]
    ) -> ImageData | pyvista_ndarray:
        """Search for a data array or slice with IJK indexing."""
        # Return point, cell, or field data
        if isinstance(key, str) or (
            isinstance(key, tuple) and len(key) > 0 and isinstance(key[0], str)  # type: ignore[redundant-expr]
        ):
            return super().__getitem__(key)
        return self.extract_subset(self._compute_voi_from_index(key), rebase_coordinates=False)

    def _compute_voi_from_index(
        self,
        indices: tuple[
            int | slice | tuple[int, int],
            int | slice | tuple[int, int],
            int | slice | tuple[int, int],
        ],
        *,
        index_mode: Literal['extent', 'dimensions'] = 'dimensions',
        strict_index: bool = False,
    ) -> NumpyArray[int]:
        """Compute VOI extents from indexing values."""
        _validation.check_contains(
            ['extent', 'dimensions'], must_contain=index_mode, name='index_mode'
        )
        if not (isinstance(indices, tuple) and len(indices) == 3):  # type: ignore[redundant-expr]
            msg = 'Exactly 3 slices must be specified, one for each IJK-coordinate axis.'  # type: ignore[unreachable]
            raise IndexError(msg)

        dims = self.dimensions
        extent = self.extent
        voi = list(extent)

        for axis, slicer in enumerate(indices):
            _validation.check_instance(slicer, (int, tuple, list, slice), name='index')

            offset = extent[axis * 2]
            index_offset = 0 if index_mode == 'extent' else offset

            if isinstance(slicer, (list, tuple)):
                rng = _validation.validate_array(
                    slicer, must_have_dtype=int, must_have_length=2, to_list=True
                )
                slicer = slice(*rng)  # noqa: PLW2901

            if isinstance(slicer, slice):
                start = slicer.start if slicer.start is not None else 0
                stop = slicer.stop if slicer.stop is not None else dims[axis]
                step = slicer.step
                if step not in (None, 1):
                    msg = 'Only contiguous slices with step=1 are supported.'
                    raise ValueError(msg)

                # Handle negative indices
                if start < 0:
                    start += dims[axis]
                if stop < 0:
                    stop += dims[axis]

            else:  # isinstance(slicer, int)
                min_allowed = offset - dims[axis] - index_offset
                max_allowed = min_allowed + dims[axis] * 2 - 1
                if slicer < min_allowed or slicer > max_allowed:
                    msg = (
                        f'index {slicer} is out of bounds for axis {axis} with size {dims[axis]}.'
                        f'\nValid range of valid index values (inclusive) is '
                        f'[{min_allowed}, {max_allowed}].'
                    )
                    raise IndexError(msg)
                if slicer < 0:
                    slicer += dims[axis]  # noqa: PLW2901
                start = slicer
                stop = start + 1

            voi[axis * 2] = index_offset + start
            voi[axis * 2 + 1] = index_offset + stop - 1

        clipped = pv.ImageDataFilters._clip_extent(voi, clip_to=self.extent)
        if strict_index and (
            any(min_ < clp for min_, clp in zip(voi[::2], clipped[::2], strict=True))
            or any(max_ > clp for max_, clp in zip(voi[1::2], clipped[1::2], strict=True))
        ):
            msg = (
                f'The requested volume of interest {tuple(voi)} '
                f"is outside the input's extent {extent}."
            )
            raise IndexError(msg)
        return clipped

    @property  # type: ignore[override]
    def points(self: Self) -> NumpyArray[float]:
        """Build a copy of the implicitly defined points as a numpy array.

        Returns
        -------
        numpy.ndarray
            Array of points representing the image data.

        Notes
        -----
        The ``points`` for a :class:`pyvista.ImageData` cannot be set.

        Examples
        --------
        >>> import pyvista as pv
        >>> grid = pv.ImageData(dimensions=(2, 2, 2))
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
        if pv.vtk_version_info >= (9, 4, 0):
            return convert_array(self.GetPoints().GetData())

        # Handle empty case
        if not all(self.dimensions):
            return np.zeros((0, 3))

        # Get grid dimensions
        nx, ny, nz = self.dimensions
        nx -= 1
        ny -= 1
        nz -= 1
        # get the points and convert to spacings
        dx, dy, dz = self.spacing
        # Now make the cell arrays
        ox, oy, oz = np.array(self.origin) + self.extent[::2] * np.array([dx, dy, dz])
        x = np.insert(np.cumsum(np.full(nx, dx)), 0, 0.0) + ox
        y = np.insert(np.cumsum(np.full(ny, dy)), 0, 0.0) + oy
        z = np.insert(np.cumsum(np.full(nz, dz)), 0, 0.0) + oz
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        points = np.c_[xx.ravel(order='F'), yy.ravel(order='F'), zz.ravel(order='F')]

        direction = self.direction_matrix
        if not np.array_equal(direction, np.eye(3)):
            return pv.Transform().rotate(direction, point=self.origin).apply(points, copy=False)
        return points

    @points.setter
    def points(
        self: Self,
        points: MatrixLike[float] | _vtk.vtkPoints,  # noqa: ARG002
    ) -> None:  # numpydoc ignore=PR01
        """Points cannot be set.

        This setter overrides the base class's setter to ensure a user does not
        attempt to set them. See https://github.com/pyvista/pyvista/issues/713.

        """
        msg = (
            'The points cannot be set. The points of '
            '`ImageData`/`vtkImageData` are implicitly defined by the '
            '`origin`, `spacing`, and `dimensions` of the grid.'
        )
        raise AttributeError(msg)

    @property
    def x(self: Self) -> NumpyArray[float]:  # numpydoc ignore=RT01
        """Return all the X points.

        Examples
        --------
        >>> import pyvista as pv
        >>> grid = pv.ImageData(dimensions=(2, 2, 2))
        >>> grid.x
        array([0., 1., 0., 1., 0., 1., 0., 1.])

        """
        return self.points[:, 0]

    @property
    def y(self: Self) -> NumpyArray[float]:  # numpydoc ignore=RT01
        """Return all the Y points.

        Examples
        --------
        >>> import pyvista as pv
        >>> grid = pv.ImageData(dimensions=(2, 2, 2))
        >>> grid.y
        array([0., 0., 1., 1., 0., 0., 1., 1.])

        """
        return self.points[:, 1]

    @property
    def z(self: Self) -> NumpyArray[float]:  # numpydoc ignore=RT01
        """Return all the Z points.

        Examples
        --------
        >>> import pyvista as pv
        >>> grid = pv.ImageData(dimensions=(2, 2, 2))
        >>> grid.z
        array([0., 0., 0., 0., 1., 1., 1., 1.])

        """
        return self.points[:, 2]

    @property
    def origin(self: Self) -> tuple[float]:  # numpydoc ignore=RT01
        """Return the origin of the grid (bottom southwest corner).

        Examples
        --------
        >>> import pyvista as pv
        >>> grid = pv.ImageData(dimensions=(5, 5, 5))
        >>> grid.origin
        (0.0, 0.0, 0.0)

        Show how the origin is in the bottom "southwest" corner of the
        ImageData.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(grid, show_edges=True)
        >>> _ = pl.add_axes_at_origin(ylabel=None)
        >>> pl.camera_position = 'xz'
        >>> pl.show()

        Set the origin to ``(1, 1, 1)`` and show how this shifts the
        ImageData.

        >>> grid.origin = (1, 1, 1)
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(grid, show_edges=True)
        >>> _ = pl.add_axes_at_origin(ylabel=None)
        >>> pl.camera_position = 'xz'
        >>> pl.show()

        """
        return self.GetOrigin()  # type: ignore[return-value]

    @origin.setter
    def origin(self: Self, origin: VectorLike[float]) -> None:
        self.SetOrigin(*origin)
        self.Modified()

    @property
    def spacing(self: Self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the spacing for each axial direction.

        Notes
        -----
        Spacing must be non-negative. While VTK accepts negative
        spacing, this results in unexpected behavior. See:
        https://github.com/pyvista/pyvista/issues/1967

        Examples
        --------
        Create a 5 x 5 x 5 uniform grid.

        >>> import pyvista as pv
        >>> grid = pv.ImageData(dimensions=(5, 5, 5))
        >>> grid.spacing
        (1.0, 1.0, 1.0)
        >>> grid.plot(show_edges=True)

        Modify the spacing to ``(1, 2, 3)``

        >>> grid.spacing = (1, 2, 3)
        >>> grid.plot(show_edges=True)

        """
        return self.GetSpacing()

    @spacing.setter
    def spacing(self: Self, spacing: VectorLike[float]) -> None:
        spacing_ = _validation.validate_array3(
            spacing, must_be_in_range=[0, float('inf')], name='spacing'
        )
        self.SetSpacing(*spacing_)
        self.Modified()

    def _get_attrs(self: Self) -> list[tuple[str, Any, str]]:
        """Return the representation methods (internal helper)."""
        attrs = Grid._get_attrs(self)
        fmt = '{}, {}, {}'.format(*[pv.FLOAT_FORMAT] * 3)
        attrs.append(('Spacing', self.spacing, fmt))
        return attrs

    def cast_to_structured_grid(self: Self) -> StructuredGrid:
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

    def cast_to_rectilinear_grid(self: Self) -> RectilinearGrid:
        """Cast this uniform grid to a rectilinear grid.

        Returns
        -------
        pyvista.RectilinearGrid
            This uniform grid as a rectilinear grid.

        """
        rectilinear_coords = self._generate_rectilinear_coords()
        grid = pv.RectilinearGrid(*rectilinear_coords)
        grid.point_data.update(self.point_data)
        grid.cell_data.update(self.cell_data)
        grid.field_data.update(self.field_data)
        grid.copy_meta_from(self, deep=True)
        return grid

    def _generate_rectilinear_coords(
        self: Self,
    ) -> list[NumpyArray[float]]:
        """Generate rectilinear coordinates (internal helper).

        Returns
        -------
        list[NumpyArray[float]]
            Rectilinear coordinates over the three dimensions.

        """
        dims = self.dimensions
        spacing = self.spacing
        origin = self.origin
        offset = self.offset
        direction = self.direction_matrix

        # Off-axis rotation is not supported by RectilinearGrid
        if np.allclose(np.abs(direction), np.eye(3)):
            sign = np.diagonal(direction)
        else:
            sign = np.array((1.0, 1.0, 1.0))
            msg = (
                'The direction matrix is not a diagonal matrix and cannot be used when casting to '
                'RectilinearGrid.\nThe direction is ignored. Consider casting to StructuredGrid '
                'instead.'
            )
            warn_external(msg, RuntimeWarning)

        # Use linspace to avoid rounding error accumulation
        ijk = [np.linspace(offset[i], offset[i] + dims[i] - 1, dims[i]) for i in range(3)]
        return [ijk[axis] * spacing[axis] * sign[axis] + origin[axis] for axis in range(3)]

    @property
    def extent(
        self: Self,
    ) -> tuple[int, int, int, int, int, int]:  # numpydoc ignore=RT01
        """Return or set the extent of the ImageData.

        The extent is simply the first and last indices for each of the three axes.
        It encodes information about the image's :attr:`offset` and :attr:`dimensions`.

        Examples
        --------
        Create a ``ImageData`` and show its extent.

        >>> import pyvista as pv
        >>> grid = pv.ImageData(dimensions=(10, 10, 10))
        >>> grid.extent
        (0, 9, 0, 9, 0, 9)

        >>> grid.extent = (2, 5, 2, 5, 2, 5)
        >>> grid.extent
        (2, 5, 2, 5, 2, 5)

        Note how this also modifies the grid's :attr:`offset`, :attr:`dimensions`,
        and :attr:`bounds`. Since we use default spacing of 1 here, the bounds
        match the extent exactly.

        >>> grid.offset
        (2, 2, 2)

        >>> grid.dimensions
        (4, 4, 4)

        >>> grid.bounds
        BoundsTuple(x_min = 2.0,
                    x_max = 5.0,
                    y_min = 2.0,
                    y_max = 5.0,
                    z_min = 2.0,
                    z_max = 5.0)

        """
        return self.GetExtent()

    @extent.setter
    def extent(self: Self, new_extent: VectorLike[int]) -> None:
        new_extent_ = _validation.validate_arrayN(
            new_extent,
            must_be_integer=True,
            must_have_length=6,
            to_list=True,
            dtype_out=int,
        )
        self.SetExtent(new_extent_)

    @property
    def offset(self: Self) -> tuple[int, int, int]:  # numpydoc ignore=RT01
        """Return or set the index offset of the ImageData.

        The offset is simply the first indices for each of the three axes
        and defines the minimum :attr:`extent` of the image. Offset values
        can be positive or negative. In physical space, the offset is relative
        to the image's :attr:`origin`.

        .. versionadded:: 0.45

        Examples
        --------
        Create a ``ImageData`` and show that the offset is zeros by default.

        >>> import pyvista as pv
        >>> grid = pv.ImageData(dimensions=(10, 10, 10))
        >>> grid.offset
        (0, 0, 0)

        The offset defines the minimum extent.

        >>> grid.extent
        (0, 9, 0, 9, 0, 9)

        Set the offset to a new value for all axes.

        >>> grid.offset = 2
        >>> grid.offset
        (2, 2, 2)

        Show the extent again. Note how all values have increased by the offset value.

        >>> grid.extent
        (2, 11, 2, 11, 2, 11)

        Set the offset for each axis separately and show the extent again.

        >>> grid.offset = (-1, -2, -3)
        >>> grid.extent
        (-1, 8, -2, 7, -3, 6)

        """
        return self.extent[::2]

    @offset.setter
    def offset(self: Self, offset: int | VectorLike[int]) -> None:
        offset_ = _validation.validate_array3(
            offset, broadcast=True, must_be_integer=True, dtype_out=int
        )
        dims = self.dimensions
        self.extent = (
            offset_[0],
            offset_[0] + dims[0] - 1,
            offset_[1],
            offset_[1] + dims[1] - 1,
            offset_[2],
            offset_[2] + dims[2] - 1,
        )

    @wraps(RectilinearGridFilters.to_tetrahedra)
    def to_tetrahedra(
        self: Self, *args, **kwargs
    ) -> UnstructuredGrid:  # numpydoc ignore=PR01,RT01
        """Cast to a rectangular grid and then convert to tetrahedra."""
        return self.cast_to_rectilinear_grid().to_tetrahedra(*args, **kwargs)

    @property
    def direction_matrix(self: Self) -> NumpyArray[float]:
        """Set or get the direction matrix.

        The direction matrix is a 3x3 matrix which controls the orientation of the
        image data.

        .. versionadded:: 0.45

        Returns
        -------
        np.ndarray
            Direction matrix as a 3x3 NumPy array.

        """
        return array_from_vtkmatrix(self.GetDirectionMatrix())

    @direction_matrix.setter
    def direction_matrix(self: Self, matrix: RotationLike) -> None:
        self.SetDirectionMatrix(vtkmatrix_from_array(_validation.validate_transform3x3(matrix)))

    @property
    def index_to_physical_matrix(self: Self) -> NumpyArray[float]:
        """Return or set 4x4 matrix to transform index space (ijk) to physical space (xyz).

        .. note::
            Setting this property modifies the object's :class:`~pyvista.ImageData.origin`,
            :class:`~pyvista.ImageData.spacing`, and :class:`~pyvista.ImageData.direction_matrix`
            properties.

        .. versionadded:: 0.45

        Returns
        -------
        np.ndarray
            4x4 transformation matrix.

        """
        return array_from_vtkmatrix(self.GetIndexToPhysicalMatrix())

    @index_to_physical_matrix.setter
    def index_to_physical_matrix(
        self: Self, matrix: TransformLike
    ) -> None:  # numpydoc ignore=GL08
        T, R, N, S, K = pv.Transform(matrix).decompose()
        if not np.allclose(K, np.eye(3)):
            msg = (
                'The transformation has a shear component which is not supported by ImageData.\n'
                'Cast to StructuredGrid first to fully support shear transformations, or use\n'
                '`Transform.decompose()` to remove this component.'
            )
            raise ValueError(msg)

        self.origin = T
        self.direction_matrix = R * N
        self.spacing = S

    @property
    def physical_to_index_matrix(self: Self) -> NumpyArray[float]:
        """Return or set 4x4 matrix to transform from physical space (xyz) to index space (ijk).

        .. note::
            Setting this property modifies the object's :class:`~pyvista.ImageData.origin`,
            :class:`~pyvista.ImageData.spacing`, and :class:`~pyvista.ImageData.direction_matrix`
            properties.

        .. versionadded:: 0.45

        Returns
        -------
        np.ndarray
            4x4 transformation matrix.

        """
        return array_from_vtkmatrix(self.GetPhysicalToIndexMatrix())

    @physical_to_index_matrix.setter
    def physical_to_index_matrix(
        self: Self, matrix: TransformLike
    ) -> None:  # numpydoc ignore=GL08
        self.index_to_physical_matrix = pv.Transform(matrix).inverse_matrix
