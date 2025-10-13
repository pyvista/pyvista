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
from typing import get_args
import warnings

import numpy as np

import pyvista
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core import _validation

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

_AxisOptions = Literal[0, 1, 2, 'x', 'y', 'z']
_StackModeOptions = Literal[
    'strict', 'resample', 'crop-dimensions', 'crop-extents', 'preserve-extents'
]
_StackDTypePolicyOptions = Literal['strict', 'promote', 'match']
_StackComponentPolicyOptions = Literal['strict', 'promote']


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

    @property
    def dimensionality(self: Self) -> int:
        """Return the dimensionality of the grid.

        Returns
        -------
        int
            The grid dimensionality.

        Examples
        --------
        Get the dimensionality of a 2D uniform grid.

        >>> import pyvista as pv
        >>> grid = pv.ImageData(dimensions=(1, 2, 3))
        >>> grid.dimensionality
        2

        Get the dimensionality of a 3D uniform grid.

        >>> grid = pv.ImageData(dimensions=(2, 3, 4))
        >>> grid.dimensionality
        3

        """
        dims = np.asarray(self.dimensions)
        return int(3 - (dims == 1).sum())


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

    _WRITERS: ClassVar[
        dict[
            str,
            type[_vtk.vtkRectilinearGridWriter | _vtk.vtkXMLRectilinearGridWriter],
        ]
    ] = {  # type: ignore[assignment]
        '.vtk': _vtk.vtkRectilinearGridWriter,
        '.vtr': _vtk.vtkXMLRectilinearGridWriter,
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
        if pyvista.vtk_version_info >= (9, 4, 0):
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

    _WRITERS: ClassVar[dict[str, type[_vtk.vtkDataSetWriter | _vtk.vtkXMLImageDataWriter]]] = {  # type: ignore[assignment]
        '.vtk': _vtk.vtkDataSetWriter,
        '.vti': _vtk.vtkXMLImageDataWriter,
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

        clipped = pyvista.ImageDataFilters._clip_extent(voi, clip_to=self.extent)
        if strict_index and (
            any(min_ < clp for min_, clp in zip(voi[::2], clipped[::2]))
            or any(max_ > clp for max_, clp in zip(voi[1::2], clipped[1::2]))
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
        if pyvista.vtk_version_info >= (9, 4, 0):
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
            return (
                pyvista.Transform().rotate(direction, point=self.origin).apply(points, copy=False)
            )
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
        fmt = '{}, {}, {}'.format(*[pyvista.FLOAT_FORMAT] * 3)
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
        grid = pyvista.RectilinearGrid(*rectilinear_coords)
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
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

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

    @wraps(RectilinearGridFilters.to_tetrahedra)  # type:ignore[has-type]
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
        T, R, N, S, K = pyvista.Transform(matrix).decompose()
        if not np.allclose(K, np.eye(3)):
            warnings.warn(
                'The transformation matrix has a shear component which has been removed. \n'
                'Shear is not supported when setting `ImageData` `index_to_physical_matrix`.',
                stacklevel=2,
            )

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
        self.index_to_physical_matrix = pyvista.Transform(matrix).inverse_matrix

    def stack(  # type: ignore[misc]
        self: ImageData,
        images: ImageData | Sequence[ImageData],
        axis: _AxisOptions | None = None,
        *,
        mode: _StackModeOptions | None = None,
        resample_kwargs: dict[str, Any] | None = None,
        dtype_policy: _StackDTypePolicyOptions | None = None,
        component_policy: _StackComponentPolicyOptions | None = None,
    ) -> ImageData:
        """Stack :class:`~pyvista.ImageData` along an axis.

        Parameters
        ----------
        images : ImageData | Sequence[ImageData]
            The input image(s) to stack. The default active scalars are used for all images.
            By default, all images must have:

            #. identical dimensions except along the stacking axis,
            #. the same scalar dtype, and
            #. the same number of scalar components.

            Use ``mode`` to allow stacking images with mismatched dimensions,
            ``dtype_policy`` to allow stacking images with different dtypes, and/or
            ``component_policy`` to allow stacking images with differing number of components.

        axis : int | str, default: 'x'
            Axis along which the images are stacked:

            - ``0`` or ``'x'``: x-axis
            - ``1`` or ``'y'``: y-axis
            - ``2`` or ``'z'``: z-axis

        mode : str, default: 'strict'
            Stacking mode to use. This determines how images are placed in the output. All modes
            operate along the specified ``axis`` except for ``'preserve-extents'``. Specify one of:

            - ``'strict'``: all images must have identical dimensions except along the stacking
              axis.
            - ``'resample'``: :meth:`resample` any images being stacked such that their dimensions
              match the input. Off-axis dimensions are resampled, and the on-axis dimension is not.
              If 2D images are stacked with a 2D input, the aspect ratios of the stacked images are
              preserved.
            - ``'crop-dimensions'``: :meth:`crop` images being stacked such that their dimensions
              match the input dimensions exactly. The images are center-cropped.
            - ``'crop-extents'``: :meth:`crop` images being stacked using the extent of the input
              to crop each image such that all images have the same extent before stacking.
            - ``'preserve-extents'``: the extent of all images are preserved and used to place the
              images in the output. The whole extent of the output is the union of the input whole
              extents. The origin and spacing is taken from the first input.

            .. note::
                For the ``crop`` and ``preserve-extents`` modes, any portion of the output not
                covered by the inputs is set to zero.

        dtype_policy : 'strict' | 'promote' | 'match', default: 'strict'
            - ``'strict'``: Do not cast any scalar array dtypes. All images being stacked must
              have the same dtype, else a ``TypeError`` is raised.
            - ``'promote'``: Use :func:`numpy.result_type` to compute the dtype of the output
              image scalars. This option safely casts all input arrays to a common dtype before
              stacking.
            - ``'match'``: Cast all array dtypes to match the input's dtype. This casting is
              unsafe as it may downcast values and lose precision.

        component_policy : 'strict' | 'promote', default: 'strict'
            - ``'strict'``: Do not modify the number of components of any scalars. All images being
              stacked must have the number of components, else a ``ValueError`` is raised.
            - ``'promote'``: Increase the number of components if necessary. Grayscale scalars
              with one component may be promoted to RGB or RGBA scalars by duplicating values,
              and RGB scalars may be promoted to RGBA scalars by including an opacity component.

        resample_kwargs : dict, optional
            Keyword arguments passed to :meth:`resample` when using ``'resample'`` mode. Specify
            ``interpolation``, ``border_mode``, ``anti_aliasing`` options.

        Returns
        -------
        ImageData
            The stacked image.

        Examples
        --------
        Load a 2D image: :func:`~pyvista.examples.downloads.download_beach`.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> beach = examples.download_beach()

        Use :meth:`select_values` to make a second version with white values converted to black
        to distinguish it from the original.

        >>> white = [255, 255, 255]
        >>> black = [0, 0, 0]
        >>> beach_black = beach.select_values(white, fill_value=black, invert=True)

        Stack them along the x-axis.

        >>> stacked = beach.stack(beach_black, axis='x')
        >>> plot_kwargs = dict(
        ...     rgb=True,
        ...     lighting=False,
        ...     cpos='xy',
        ...     zoom='tight',
        ...     show_axes=False,
        ...     show_scalar_bar=False,
        ... )
        >>> stacked.plot(**plot_kwargs)

        Stack them along the y-axis.

        >>> stacked = beach.stack(beach_black, axis='y')
        >>> stacked.plot(**plot_kwargs)

        By default, stacking requires that all off-axis dimensions match the input. Use the
        ``mode`` keyword to enable stacking images with mismatched dimensions.

        Load a second 2D image with different dimensions:
        :func:`~pyvista.examples.downloads.download_bird`.

        >>> bird = examples.download_bird()
        >>> bird.dimensions
        (458, 342, 1)
        >>> beach.dimensions
        (100, 100, 1)

        Stack the images using the ``resample`` mode to automatically resample the image. Linear
        interpolation with antialiasing is used to avoid sampling artifacts.

        >>> resample_kwargs = {'interpolation': 'linear', 'anti_aliasing': True}
        >>> stacked = beach.stack(
        ...     bird, mode='resample', resample_kwargs=resample_kwargs
        ... )
        >>> stacked.dimensions
        (233, 100, 1)
        >>> stacked.plot(**plot_kwargs)

        Use the ``'preserve-extents'`` mode. Using this mode naively may not produce the desired
        result, e.g. if we stack ``beach`` with ``bird``, the ``beach`` image is completely
        overwritten since their :attr:`~pyvista.ImageData.extent`s fully overlap.

        >>> beach.extent
        (0, 99, 0, 99, 0, 0)
        >>> bird.extent
        (0, 457, 0, 341, 0, 0)

        >>> stacked = beach.stack(bird, mode='preserve-extents')
        >>> stacked.extent
        (0, 457, 0, 341, 0, 0)
        >>> stacked.plot(**plot_kwargs)

        Set the ``beach`` :attr:`~pyvista.ImageData.offset` so that there is only partial overlap
        instead.

        >>> beach.offset = (-50, -50, 0)
        >>> beach.extent
        (-50, 49, -50, 49, 0, 0)

        >>> stacked = beach.stack(bird, mode='preserve-extents')
        >>> stacked.extent
        (-50, 457, -50, 341, 0, 0)
        >>> stacked.plot(**plot_kwargs)

        Reverse the stacking order.

        >>> stacked = bird.stack(beach, mode='preserve-extents')
        >>> stacked.plot(**plot_kwargs)

        Use ``'crop-dimensions'`` to center-crop the images to match the input's
        dimensions.

        >>> stacked = beach.stack(bird, mode='crop-dimensions')
        >>> stacked.plot(**plot_kwargs)

        Reverse the stacking order.

        >>> stacked = bird.stack(beach, mode='crop-dimensions')
        >>> stacked.plot(**plot_kwargs)

        Reset the offset and use ``'crop-extents'`` mode to automatically :meth:`crop` each image
        using the input's extent. This crops the lower-left portion of ``bird``, since the `
        `beach`` extent corresponds to the bottom lower left portion of the ``bird`` extent.

        >>> beach.offset = (0, 0, 0)
        >>> stacked = beach.stack(bird, mode='crop-extents')
        >>> stacked.plot(**plot_kwargs)

        Load a binary image: :func:`~pyvista.examples.downloads.download_yinyang()`.

        >>> yinyang = examples.download_yinyang()

        Use ``component_policy`` to stack grayscale images with RGB(A) images.

        >>> stacked = yinyang.stack(
        ...     [bird, beach], mode='resample', component_policy='promote'
        ... )
        >>> stacked.plot(**plot_kwargs)

        """
        from pyvista.core.filters import _get_output  # noqa: PLC0415
        from pyvista.core.filters import _update_alg  # noqa: PLC0415

        def _compute_resample_kwargs(
            ref_image: ImageData, img: ImageData, axis: int
        ) -> dict[str, NumpyArray[float]]:
            """Compute resampling keywords for stacking img with ref_image along an axis."""
            ref_dims = np.array(ref_image.dimensions, dtype=int)
            img_dims = np.array(img.dimensions, dtype=int)

            is_2d_ref = ref_image.dimensionality == 2
            is_2d_img = img.dimensionality == 2

            if is_2d_ref and is_2d_img:
                # Try to preserve image aspect ratio when images are 2D along the same dimensions
                off_axis = np.arange(3) != axis
                not_singleton = ref_dims != 1
                fixed_axis = off_axis & not_singleton
                has_one_fixed_axis = np.count_nonzero(fixed_axis) == 1
                if has_one_fixed_axis:
                    # Resample the image proportionally to match the single fixed axis
                    sample_rate = ref_dims[fixed_axis] / img_dims[fixed_axis]
                    return {'sample_rate': sample_rate}

            # Image must match the reference's non-stacking axes exactly,
            # but we leave the image's stacking axis unchanged
            new_dims = ref_dims.copy()
            new_dims[axis] = img_dims[axis]
            return {'dimensions': new_dims}

        # Validate mode
        if mode is not None:
            options = get_args(_StackModeOptions)
            _validation.check_contains(options, must_contain=mode, name='mode')
        else:
            mode = 'strict'

        # Validate axis
        if axis is not None:
            if mode == 'preserve-extents':
                msg = "The axis keyword cannot be used with 'preserve-extents' mode."
                raise ValueError(msg)
            options = get_args(_AxisOptions)
            _validation.check_contains(options, must_contain=axis, name='axis')
            mapping = {'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}
            axis_num = mapping[axis]
        else:
            axis_num = 0

        # Validate dtype policy
        if dtype_policy is not None:
            options = get_args(_StackDTypePolicyOptions)
            _validation.check_contains(options, must_contain=dtype_policy, name='dtype_policy')
        else:
            dtype_policy = 'strict'

        # Validate component policy
        if component_policy is not None:
            options = get_args(_StackComponentPolicyOptions)
            _validation.check_contains(
                options, must_contain=component_policy, name='component_policy'
            )
        else:
            component_policy = 'strict'

        all_images = [self, *images] if isinstance(images, Sequence) else [self, images]

        self_dimensions = self.dimensions
        self_extent = self.extent
        all_dtypes: list[np.dtype] = []
        all_n_components: list[int] = []
        all_scalars: list[str] = []
        for i, img in enumerate(all_images):
            if i > 0:
                _validation.check_instance(img, pyvista.ImageData)

            # Create shallow copies so we can safely modify if needed
            img_shallow_copy = img.copy(deep=False)
            _, scalars = img_shallow_copy._validate_point_scalars()
            all_scalars.append(scalars)
            array = img.point_data[scalars]
            all_dtypes.append(array.dtype)
            n_components = array.shape[1] if array.ndim == 2 else array.ndim
            all_n_components.append(n_components)

            if i == 0 and mode in ['resample', 'crop-dimensions']:
                # These modes should not be affected by offset, so we zero it
                img_shallow_copy.offset = (0, 0, 0)
            if i > 0 and mode != 'preserve-extents':
                if (dims := img.dimensions) != self_dimensions:
                    # Need to deal with the dimensions mismatch
                    if mode == 'strict':
                        # Allow mismatch only along stacking axis
                        for ax in range(3):
                            if ax != axis and dims[ax] != self_dimensions[ax]:
                                msg = (
                                    f'Image {i - 1} dimensions {img.dimensions} must match the '
                                    f'input dimensions {self.dimensions} along axis {axis}.\n'
                                    f'Use the `mode` keyword to allow stacking with mismatched '
                                    f'dimensions.'
                                )
                                raise ValueError(msg)
                    elif mode == 'resample':
                        kwargs = {}
                        if resample_kwargs:
                            _validation.check_instance(resample_kwargs, dict)
                            allowed_kwargs = ('anti_aliasing', 'interpolation', 'border_mode')
                            for kwarg in resample_kwargs.keys():
                                _validation.check_contains(
                                    allowed_kwargs, must_contain=kwarg, name='resample_kwargs'
                                )
                            kwargs = resample_kwargs

                        computed_kwargs = _compute_resample_kwargs(self, img, axis=axis_num)
                        kwargs.update(computed_kwargs)
                        img_shallow_copy = img_shallow_copy.resample(**kwargs)
                        img_shallow_copy.offset = (0, 0, 0)

                    elif mode == 'crop-extents':
                        img_shallow_copy = img_shallow_copy.crop(extent=self_extent)
                    elif mode == 'crop-dimensions':
                        img_shallow_copy = img_shallow_copy.crop(dimensions=self_dimensions)
                        img_shallow_copy.offset = (0, 0, 0)

            # Replace input with shallow copy
            all_images[i] = img_shallow_copy

        if len(set(all_dtypes)) > 1:
            # Need to cast all scalars to the same dtype
            if dtype_policy == 'strict':
                msg = (
                    f'The dtypes of the scalar arrays do not match. Got multiple '
                    f"dtypes: {set(all_dtypes)}.\nSet the dtype policy to 'promote' or "
                    f"'match' to cast the inputs to a single dtype."
                )
                raise TypeError(msg)
            elif dtype_policy == 'promote':
                dtype_out = np.result_type(*all_dtypes)
            else:  # dtype_policy == 'match'
                dtype_out = all_dtypes[0]

            for img, scalars in zip(all_images, all_scalars):
                array = img.point_data[scalars]
                img.point_data[scalars] = array.astype(dtype_out, copy=False)
        else:
            dtype_out = all_images[0].point_data[all_scalars[0]].dtype

        if len(set(all_n_components)) > 1:
            # Need to ensure all scalars have the same number of components
            if component_policy == 'strict':
                msg = (
                    f'The number of components in the scalar arrays do not match. Got n '
                    f'components: {set(all_n_components)}.\nSet the component policy to '
                    f"'promote' to automatically increase the number of components as needed."
                )
                raise ValueError(msg)
            else:  # component_policy == 'promote'
                if not set(all_n_components) < {1, 3, 4}:
                    msg = (
                        'Unable to promote scalars. Only promotion for grayscale (1 component), '
                        'RGB (3 components),\n and RGBA (4 components) is supported. Got'
                        f'{all_n_components}'
                    )
                    raise ValueError(msg)
                target_n_components = max(all_n_components)
                for img, n_components, scalars in zip(all_images, all_n_components, all_scalars):
                    if n_components < target_n_components:
                        array = img.point_data[scalars]
                        if n_components < 3:
                            array = np.vstack((array, array, array)).T  # type: ignore[assignment]
                        if target_n_components == 4:
                            fill_value = (
                                np.iinfo(dtype_out).max
                                if np.issubdtype(dtype_out, np.integer)
                                else 1.0
                            )
                            new_array = np.full((len(array), 4), fill_value, dtype=dtype_out)
                            new_array[:, :3] = array
                            array = new_array  # type: ignore[assignment]

                        img.point_data[scalars] = array

        alg = _vtk.vtkImageAppend()
        alg.SetAppendAxis(axis_num)
        alg.SetPreserveExtents(mode == 'preserve-extents')

        for img in all_images:
            alg.AddInputData(img)

        _update_alg(alg)
        output = _get_output(alg)
        output.offset = self.offset
        return output
