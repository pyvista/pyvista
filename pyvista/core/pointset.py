"""Sub-classes and wrappers for :vtk:`vtkPointSet`."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
import contextlib
from functools import cached_property
from functools import wraps
import numbers
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Literal
from typing import cast

import numpy as np

import pyvista as pv
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista._warn_external import warn_external

from . import _vtk_core as _vtk
from .cell import CellArray
from .cell import _get_connectivity_array
from .cell import _get_irregular_cells
from .cell import _get_offset_array
from .cell import _get_regular_cells
from .celltype import CellType
from .dataset import DataSet
from .errors import CellSizeError
from .errors import PointSetCellOperationError
from .errors import PointSetDimensionReductionError
from .errors import PointSetNotSupported
from .filters import DataObjectFilters
from .filters import PolyDataFilters
from .filters import StructuredGridFilters
from .filters import UnstructuredGridFilters
from .filters import _get_output
from .filters.data_object import _MeshValidationReport
from .filters.data_object import _MeshValidator
from .utilities.arrays import convert_array
from .utilities.cells import create_mixed_cells
from .utilities.cells import get_mixed_cells
from .utilities.cells import numpy_to_idarr
from .utilities.fileio import _CompressionOptions
from .utilities.fileio import get_ext
from .utilities.misc import abstract_class
from .utilities.points import vtk_points
from .utilities.writer import BaseWriter
from .utilities.writer import HDFWriter
from .utilities.writer import HoudiniPolyDataWriter
from .utilities.writer import IVWriter
from .utilities.writer import OBJWriter
from .utilities.writer import PLYWriter
from .utilities.writer import PolyDataWriter
from .utilities.writer import SimplePointsWriter
from .utilities.writer import STLWriter
from .utilities.writer import StructuredGridWriter
from .utilities.writer import UnstructuredGridWriter
from .utilities.writer import XMLPolyDataWriter
from .utilities.writer import XMLStructuredGridWriter
from .utilities.writer import XMLUnstructuredGridWriter

if TYPE_CHECKING:
    from typing_extensions import Self

    from ._typing_core import ArrayLike
    from ._typing_core import BoundsTuple
    from ._typing_core import CellArrayLike
    from ._typing_core import MatrixLike
    from ._typing_core import NumpyArray
    from ._typing_core import VectorLike


DEFAULT_INPLACE_WARNING = (
    'You did not specify a value for `inplace` and the default value will '
    'be changing to `False` in future versions for point-based meshes (e.g., '
    '`PolyData`). Please make sure you are not assuming this to be an inplace '
    'operation.'
)


@abstract_class
class _PointSet(DataSet):
    """PyVista's equivalent of :vtk:`vtkPointSet`.

    This holds methods common to PolyData and UnstructuredGrid.
    """

    _WRITERS: ClassVar[dict[str, type[BaseWriter]]] = {
        '.xyz': SimplePointsWriter,
    }

    @_deprecate_positional_args
    def center_of_mass(self, scalars_weight: bool = False) -> NumpyArray[float]:  # noqa: FBT001, FBT002
        """Return the coordinates for the center of mass of the mesh.

        Parameters
        ----------
        scalars_weight : bool, default: False
            Flag for using the mesh scalars as weights.

        Returns
        -------
        numpy.ndarray
            Coordinates for the center of mass.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Sphere(center=(1, 1, 1))
        >>> mesh.center_of_mass()
        array([1., 1., 1.])

        """
        alg = _vtk.vtkCenterOfMass()
        alg.SetInputDataObject(self)
        alg.SetUseScalarsAsWeights(scalars_weight)
        alg.Update()
        return np.array(alg.GetCenter())

    def shallow_copy(self, to_copy: DataSet) -> None:  # type: ignore[override]
        """Create a shallow copy from a different dataset into this one.

        This method mutates this dataset and returns ``None``.

        Parameters
        ----------
        to_copy : pyvista.DataSet
            Data object to perform the shallow copy from.

        """
        # Set default points if needed
        if not to_copy.GetPoints():
            to_copy.SetPoints(_vtk.vtkPoints())
        DataSet.shallow_copy(self, cast('_vtk.vtkDataObject', to_copy))

    @_deprecate_positional_args(allowed=['ind'])
    def remove_cells(
        self,
        ind: VectorLike[bool] | VectorLike[int],
        inplace: bool = False,  # noqa: FBT001, FBT002
    ) -> _PointSet:
        """Remove cells.

        Parameters
        ----------
        ind : VectorLike[int] | VectorLike[bool]
            Cell indices to be removed.  The array can also be a
            boolean array of the same size as the number of cells.

        inplace : bool, default: False
            Whether to update the mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Same type as the input, but with the specified cells
            removed.

        See Also
        --------
        :ref:`ghost_cells_example`

        Examples
        --------
        Remove 20 cells from an unstructured grid.

        >>> from pyvista import examples
        >>> import pyvista as pv
        >>> hex_mesh = pv.read(examples.hexbeamfile)
        >>> removed = hex_mesh.remove_cells(range(10, 20))
        >>> removed.plot(color='lightblue', show_edges=True, line_width=3)

        """
        if isinstance(ind, np.ndarray):
            if ind.dtype == np.bool_ and ind.size != self.n_cells:
                msg = f'Boolean array size must match the number of cells ({self.n_cells})'
                raise ValueError(msg)
        ghost_cells = np.zeros(self.n_cells, np.uint8)
        ghost_cells[ind] = _vtk.vtkDataSetAttributes.DUPLICATECELL

        target = self if inplace else self.copy()

        target.cell_data[_vtk.vtkDataSetAttributes.GhostArrayName()] = ghost_cells
        target.RemoveGhostCells()
        return target

    def points_to_double(self) -> Self:
        """Convert the points datatype to double precision.

        Returns
        -------
        pyvista.PointSet
            Pointset with points in double precision.

        Notes
        -----
        This operates in place.

        Examples
        --------
        Create a mesh that has points of the type ``float32`` and
        convert the points to ``float64``.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh.points.dtype
        dtype('float32')
        >>> _ = mesh.points_to_double()
        >>> mesh.points.dtype
        dtype('float64')

        """
        if self.points.dtype != np.double:
            self.points = self.points.astype(np.double)
        return self

    # todo: `transform_all_input_vectors` is not handled when modifying inplace
    @_deprecate_positional_args(allowed=['xyz'])
    def translate(
        self: Self,
        xyz: VectorLike[float],
        transform_all_input_vectors: bool = False,  # noqa: FBT001, FBT002
        inplace: bool = False,  # noqa: FBT001, FBT002
    ):
        """Translate the mesh.

        Parameters
        ----------
        xyz : VectorLike[float]
            A vector of three floats of cartesian values to translate the mesh with.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are transformed. Otherwise, only
            the points, normals and active vectors are transformed. This is
            only valid when not updating in place.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        pyvista.PointSet
            Translated pointset.

        Examples
        --------
        Create a sphere and translate it by ``(2, 1, 2)``.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh.center
        (0.0, 0.0, 0.0)
        >>> trans = mesh.translate((2, 1, 2), inplace=True)
        >>> trans.center
        (2.0, 1.0, 2.0)

        """
        if inplace:
            self.points += np.asarray(xyz)
            return self
        return pv.DataObjectFilters.translate(
            self,
            xyz,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )


class PointSet(_PointSet, _vtk.vtkPointSet):
    """Concrete class for storing a set of points.

    This is a concrete class representing a set of points that specifies the
    interface for datasets that explicitly use "point" arrays to represent
    geometry. This class is useful for improving the performance of filters on
    point clouds, but not plotting.

    For further details see :vtk:`vtkPointSet`.

    Parameters
    ----------
    var_inp : :vtk:`vtkPointSet`, MatrixLike[float], optional
        Flexible input type.  Can be a :vtk:`vtkPointSet`, in which case
        this PointSet object will be copied if ``deep=True`` and will
        be a shallow copy if ``deep=False``.

        List, numpy array, or sequence containing point locations. Must be an
        ``(N, 3)`` array of points.

    deep : bool, default: False
        Whether to copy the input ``points``, or to create a PointSet from them
        without copying them.  Setting ``deep=True`` ensures that the original
        arrays can be modified outside the mesh without affecting the
        mesh.

    force_float : bool, default: True
        Casts the datatype to ``float32`` if points datatype is non-float.  Set
        this to ``False`` to allow non-float types, though this may lead to
        truncation of intermediate floats when transforming datasets.

    Examples
    --------
    Create a simple point cloud of 10 points from a numpy array.

    >>> import numpy as np
    >>> import pyvista as pv
    >>> rng = np.random.default_rng(seed=0)
    >>> points = rng.random((10, 3))
    >>> pset = pv.PointSet(points)

    Plot the pointset. Note: this casts to a :class:`pyvista.PolyData`
    internally when plotting.

    >>> pset.plot(point_size=10)

    """

    @_deprecate_positional_args(allowed=['var_inp'])
    def __init__(self, var_inp=None, deep: bool = False, force_float: bool = True) -> None:  # noqa: FBT001, FBT002
        """Initialize the pointset."""
        super().__init__()

        if var_inp is None:
            return
        elif isinstance(var_inp, _vtk.vtkPointSet):
            if deep:
                self.deep_copy(var_inp)
            else:
                self.shallow_copy(var_inp)  # type: ignore[arg-type]
        else:
            self.SetPoints(vtk_points(var_inp, deep=deep, force_float=force_float))

    def __repr__(self):
        """Return the standard representation."""
        return DataSet.__repr__(self)

    def __str__(self):
        """Return the standard str representation."""
        return DataSet.__str__(self)

    @_deprecate_positional_args
    def cast_to_polydata(self, deep: bool = True):  # noqa: FBT001, FBT002
        """Cast this dataset to polydata.

        Parameters
        ----------
        deep : bool, deep: True
            Whether to copy the pointset points, or to create a PolyData
            without copying them.  Setting ``deep=True`` ensures that the
            original arrays can be modified outside the PolyData without
            affecting the PolyData.

        Returns
        -------
        pyvista.PolyData
            PointSet cast to a ``pyvista.PolyData``.

        """
        pdata = PolyData(self.points, deep=deep)
        if deep:
            pdata.point_data.update(self.point_data)  # update performs deep copy
        else:
            for key, value in self.point_data.items():
                pdata.point_data[key] = value
        return pdata

    def cast_to_unstructured_grid(self) -> pv.UnstructuredGrid:
        """Cast this dataset to :class:`pyvista.UnstructuredGrid`.

        A deep copy of the points and point data is made.

        Returns
        -------
        pyvista.UnstructuredGrid
            Dataset cast to a :class:`pyvista.UnstructuredGrid`.

        Examples
        --------
        Cast a :class:`pyvista.PointSet` to a
        :class:`pyvista.UnstructuredGrid`.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> mesh = examples.download_cloud_dark_matter()
        >>> type(mesh)
        <class 'pyvista.core.pointset.PointSet'>
        >>> grid = mesh.cast_to_unstructured_grid()
        >>> type(grid)
        <class 'pyvista.core.pointset.UnstructuredGrid'>

        """
        return self.cast_to_polydata(deep=False).cast_to_unstructured_grid()

    @wraps(DataSet.plot)
    def plot(self, *args, **kwargs):  # type: ignore[override]  # numpydoc ignore=RT01
        """Cast to PolyData and plot."""
        pdata = self.cast_to_polydata(deep=False)
        kwargs.setdefault('style', 'points')
        return pdata.plot(*args, **kwargs)

    @wraps(PolyDataFilters.threshold)
    def threshold(self, *args, **kwargs):  # type: ignore[override]  # numpydoc ignore=RT01
        """Cast to PolyData and threshold.

        Need this because cell-wise operations fail for PointSets.
        """
        return self.cast_to_polydata(deep=False).threshold(*args, **kwargs).cast_to_pointset()

    @wraps(PolyDataFilters.threshold_percent)
    def threshold_percent(self, *args, **kwargs):  # type: ignore[override]  # numpydoc ignore=RT01
        """Cast to PolyData and threshold.

        Need this because cell-wise operations fail for PointSets.
        """
        return (
            self.cast_to_polydata(deep=False).threshold_percent(*args, **kwargs).cast_to_pointset()
        )

    @wraps(PolyDataFilters.explode)
    def explode(self, *args, **kwargs):  # type: ignore[override]  # numpydoc ignore=RT01
        """Cast to PolyData and explode.

        The explode filter relies on cells.

        """
        return self.cast_to_polydata(deep=False).explode(*args, **kwargs).cast_to_pointset()

    @wraps(PolyDataFilters.delaunay_3d)
    def delaunay_3d(self, *args, **kwargs):  # type: ignore[override]  # numpydoc ignore=RT01
        """Cast to PolyData and run delaunay_3d."""
        return self.cast_to_polydata(deep=False).delaunay_3d(*args, **kwargs)

    @property
    def area(self) -> float:  # numpydoc ignore=RT01
        """Return 0.0 since a PointSet has no area."""
        return 0.0

    @property
    def volume(self) -> float:  # numpydoc ignore=RT01
        """Return 0.0 since a PointSet has no volume."""
        return 0.0

    def contour(self, *args, **kwargs):  # noqa: ARG002
        """Raise dimension reducing operations are not supported."""
        msg = 'Contour and other dimension reducing filters are not supported on PointSets'
        raise PointSetNotSupported(msg)

    def cell_data_to_point_data(self, *args, **kwargs):  # noqa: ARG002
        """Raise PointSets do not have cells."""
        msg = 'PointSets contain no cells or cell data.'
        raise PointSetNotSupported(msg)

    def point_data_to_cell_data(self, *args, **kwargs):  # noqa: ARG002
        """Raise PointSets do not have cells."""
        msg = 'PointSets contain no cells or cell data.'
        raise PointSetNotSupported(msg)

    def triangulate(self, *args, **kwargs):  # noqa: ARG002
        """Raise cell operations are not supported."""
        raise PointSetCellOperationError

    def decimate_boundary(self, *args, **kwargs):  # noqa: ARG002
        """Raise cell operations are not supported."""
        raise PointSetCellOperationError

    def find_cells_along_line(self, *args, **kwargs):  # noqa: ARG002
        """Raise cell operations are not supported."""
        raise PointSetCellOperationError

    def tessellate(self, *args, **kwargs):  # noqa: ARG002
        """Raise cell operations are not supported."""
        raise PointSetCellOperationError

    def slice(self, *args, **kwargs):  # noqa: ARG002
        """Raise dimension reducing operations are not supported."""
        raise PointSetDimensionReductionError

    def slice_along_axis(self, *args, **kwargs):  # noqa: ARG002
        """Raise dimension reducing operations are not supported."""
        raise PointSetDimensionReductionError

    def slice_along_line(self, *args, **kwargs):  # noqa: ARG002
        """Raise dimension reducing operations are not supported."""
        raise PointSetDimensionReductionError

    def slice_implicit(self, *args, **kwargs):  # noqa: ARG002
        """Raise dimension reducing operations are not supported."""
        raise PointSetDimensionReductionError

    def slice_orthogonal(self, *args, **kwargs):  # noqa: ARG002
        """Raise dimension reducing operations are not supported."""
        raise PointSetDimensionReductionError

    def shrink(self, *args, **kwargs):  # noqa: ARG002
        """Raise cell operations are not supported."""
        raise PointSetCellOperationError

    def separate_cells(self, *args, **kwargs):  # noqa: ARG002
        """Raise cell operations are not supported."""
        raise PointSetCellOperationError

    def remove_cells(self, *args, **kwargs):  # noqa: ARG002
        """Raise cell operations are not supported."""
        raise PointSetCellOperationError

    def point_is_inside_cell(self, *args, **kwargs):  # noqa: ARG002
        """Raise cell operations are not supported."""
        raise PointSetCellOperationError

    def extract_surface(self, *args, **kwargs):  # noqa: ARG002
        """Raise extract surface are not supported."""
        raise PointSetCellOperationError

    def extract_geometry(self, *args, **kwargs):  # noqa: ARG002
        """Raise extract geometry are not supported."""
        raise PointSetCellOperationError

    def cell_validator(self, *args, **kwargs):  # noqa: ARG002
        """Raise cell operations are not supported."""
        raise PointSetCellOperationError

    @wraps(DataObjectFilters.validate_mesh)
    def validate_mesh(  # type: ignore[override]  # numpydoc ignore=RT01
        self: Self,
        validation_fields: _MeshValidator._AllValidationOptions
        | Sequence[_MeshValidator._AllValidationOptions]
        | None = None,
        *args,
        **kwargs,
    ) -> _MeshValidationReport[Self]:
        """Wrap validate_mesh with cell-related fields removed."""
        if validation_fields is None:
            fields: list[_MeshValidator._AllValidationOptions] = [
                *_MeshValidator._allowed_data_fields,
                *_MeshValidator._allowed_point_fields,
            ]
            fields.remove('unused_points')
            return DataSet.validate_mesh(self, fields, *args, **kwargs)
        return DataSet.validate_mesh(self, validation_fields, *args, **kwargs)


class PolyData(_PointSet, PolyDataFilters, _vtk.vtkPolyData):
    """Dataset consisting of surface geometry (e.g. vertices, lines, and polygons).

    Can be initialized in several ways:

    - Create an empty mesh
    - Initialize from a :vtk:`vtkPolyData`
    - Using vertices
    - Using vertices and faces
    - From a file

    .. deprecated:: 0.44.0
       The parameters ``n_faces``, ``n_lines``, ``n_strips``, and
       ``n_verts`` are deprecated and no longer used. They were
       previously used to speed up the construction of the corresponding
       cell arrays but no longer provide any benefit.

    Parameters
    ----------
    var_inp : :vtk:`vtkPolyData`, str, sequence, optional
        Flexible input type.  Can be a :vtk:`vtkPolyData`, in which case
        this PolyData object will be copied if ``deep=True`` and will
        be a shallow copy if ``deep=False``.

        Also accepts a path, which may be local path as in
        ``'my_mesh.stl'`` or global path like ``'/tmp/my_mesh.ply'``
        or ``'C:/Users/user/my_mesh.ply'``.

        Otherwise, this must be a points array or list containing one
        or more points.  Each point must have 3 dimensions.  If
        ``faces``, ``lines``, ``strips``, and ``verts`` are all
        ``None``, then the ``PolyData`` object will be created with
        vertex cells with ``n_verts`` equal to the number of ``points``.

    faces : sequence[int], :vtk:`vtkCellArray`, CellArray, optional
        Polygonal faces of the mesh. Can be either a padded connectivity
        array or an explicit cell array object.

        In the padded array format, faces must contain padding
        indicating the number of points in the face.  For example, the
        two faces ``[10, 11, 12]`` and ``[20, 21, 22, 23]`` will be
        represented as ``[3, 10, 11, 12, 4, 20, 21, 22, 23]``.  This
        lets you have an arbitrary number of points per face.

        When not including the face connectivity array, each point
        will be assigned to a single vertex.  This is used for point
        clouds that have no connectivity.

    n_faces : int, optional
        Deprecated. Not used.

    lines : sequence[int], :vtk:`vtkCellArray`, CellArray, optional
        Line connectivity. Like ``faces``, this can be either a padded
        connectivity array or an explicit cell array object. The padded
        array format requires padding indicating the number of points in
        a line segment.  For example, the two line segments ``[0, 1]``
        and ``[1, 2, 3, 4]`` will be represented as
        ``[2, 0, 1, 4, 1, 2, 3, 4]``.

    n_lines : int, optional
        Deprecated. Not used.

    strips : sequence[int], :vtk:`vtkCellArray`, CellArray, optional
        Triangle strips connectivity.  Triangle strips require an
        initial triangle, and the following points of the strip. Each
        triangle is built with the new point and the two previous
        points.

        Just as in ``lines`` and ``faces``, this connectivity can be
        specified as either a padded array or an explicit cell array
        object. The padded array requires a padding indicating the
        number of points. For example, a single triangle strip of the 10
        point indices ``[0, 1, 2, 3, 6, 7, 4, 5, 0, 1]`` requires
        padding of ``10`` and should be input as
        ``[10, 0, 1, 2, 3, 6, 7, 4, 5, 0, 1]``.

    n_strips : int, optional
        Deprecated. Not used.

    deep : bool, optional
        Whether to copy the inputs, or to create a mesh from them
        without copying them.  Setting ``deep=True`` ensures that the
        original arrays can be modified outside the mesh without
        affecting the mesh. Default is ``False``.

    force_ext : str, optional
        If initializing from a file, force the reader to treat the
        file as if it had this extension as opposed to the one in the
        file.

    force_float : bool, optional
        Casts the datatype to ``float32`` if points datatype is
        non-float.  Default ``True``. Set this to ``False`` to allow
        non-float types, though this may lead to truncation of
        intermediate floats when transforming datasets.

    verts : sequence[int], :vtk:`vtkCellArray`, CellArray, optional
        The verts connectivity.  Like ``faces``, ``lines``, and
        ``strips`` this can be supplied as either a padded array or an
        explicit cell array object. In the padded array format,
        the padding indicates the number of vertices in each cell.  For
        example, ``[1, 0, 1, 1, 1, 2]`` indicates three vertex cells
        each with one point, and ``[2, 0, 1, 2, 2, 3]`` indicates two
        polyvertex cells each with two points.

    n_verts : int, optional
        Deprecated. Not used.

    See Also
    --------
    pyvista.PolyData.from_regular_faces
    pyvista.PolyData.from_irregular_faces

    Examples
    --------
    >>> import vtk
    >>> import numpy as np
    >>> from pyvista import examples
    >>> import pyvista as pv

    Seed random number generator for reproducible plots

    >>> rng = np.random.default_rng(seed=0)

    Create an empty mesh.

    >>> mesh = pv.PolyData()

    Initialize from a :vtk:`vtkPolyData` object.

    >>> vtkobj = vtk.vtkPolyData()
    >>> mesh = pv.PolyData(vtkobj)

    Initialize from just points, creating vertices

    >>> points = np.array([[0, 0, 0], [1, 0, 0], [1, 0.5, 0], [0, 0.5, 0]])
    >>> mesh = pv.PolyData(points)

    Initialize from points and faces, creating polygonal faces.

    >>> faces = np.hstack([[3, 0, 1, 2], [3, 0, 3, 2]])
    >>> mesh = pv.PolyData(points, faces)

    Initialize from points and lines.

    >>> lines = np.hstack([[2, 0, 1], [2, 1, 2]])
    >>> mesh = pv.PolyData(points, lines=lines)

    Initialize from points and triangle strips.

    >>> strips = np.hstack([[4, 0, 1, 3, 2]])
    >>> mesh = pv.PolyData(points, strips=strips)

    It is also possible to create with multiple cell types.

    >>> verts = [1, 0]
    >>> lines = [2, 1, 2]
    >>> mesh = pv.PolyData(points, verts=verts, lines=lines)

    Initialize from a filename.

    >>> mesh = pv.PolyData(examples.antfile)

    Construct a set of random line segments using a ``pv.CellArray``.
    Because every line in this example has the same size, in this case
    two points, we can use ``pv.CellArray.from_regular_cells`` to
    construct the ``lines`` cell array. This is the most efficient
    method to construct a cell array.

    >>> n_points = 20
    >>> n_lines = n_points // 2
    >>> points = rng.random((n_points, 3))
    >>> lines = rng.integers(low=0, high=n_points, size=(n_lines, 2))
    >>> mesh = pv.PolyData(points, lines=pv.CellArray.from_regular_cells(lines))
    >>> mesh.cell_data['line_idx'] = np.arange(n_lines)
    >>> mesh.plot(scalars='line_idx')

    Construct a set of random triangle strips using a ``pv.CellArray``.
    Because each strip in this example can have a different number
    of points, we use ``pv.CellArray.from_irregular_cells`` to construct
    the ``strips`` cell array.

    >>> n_strips = 4
    >>> n_verts_per_strip = rng.integers(low=3, high=7, size=n_strips)
    >>> n_points = 10 * sum(n_verts_per_strip)
    >>> points = rng.random((n_points, 3))
    >>> strips = [
    ...     rng.integers(low=0, high=n_points, size=nv) for nv in n_verts_per_strip
    ... ]
    >>> mesh = pv.PolyData(
    ...     points, strips=pv.CellArray.from_irregular_cells(strips)
    ... )
    >>> mesh.cell_data['strip_idx'] = np.arange(n_strips)
    >>> mesh.plot(show_edges=True, scalars='strip_idx')

    Construct a mesh reusing the ``faces`` ``pv.CellArray`` from another
    mesh. The VTK methods ``GetPolys``, ``GetLines``, ``GetStrips``, and
    ``GetVerts`` return the underlying ``CellArray``s for the ``faces``,
    ``lines``, ``strips``, and ``verts`` properties respectively.
    Reusing cell arrays like this can be a performance optimization for
    large meshes because it avoids allocating new arrays.

    >>> small_sphere = pv.Sphere().compute_normals()
    >>> inflated_points = (
    ...     small_sphere.points + 0.1 * small_sphere.point_data['Normals']
    ... )
    >>> larger_sphere = pv.PolyData(inflated_points, faces=small_sphere.GetPolys())
    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(small_sphere, color='red', show_edges=True)
    >>> _ = pl.add_mesh(larger_sphere, color='blue', opacity=0.3, show_edges=True)
    >>> pl.show()

    See :ref:`create_poly_example` for more examples.

    """

    _USE_STRICT_N_FACES = False

    _WRITERS: ClassVar[dict[str, type[BaseWriter]]] = {
        '.ply': PLYWriter,
        '.vtp': XMLPolyDataWriter,
        '.stl': STLWriter,
        '.vtk': PolyDataWriter,
        '.geo': HoudiniPolyDataWriter,
        '.obj': OBJWriter,
        '.iv': IVWriter,
    }
    if _vtk.vtk_version_info >= (9, 4):
        _WRITERS.update({'.vtkhdf': HDFWriter})

    @_deprecate_positional_args(allowed=['var_inp', 'faces'])
    def __init__(  # noqa: PLR0917
        self,
        var_inp: _vtk.vtkPolyData | str | Path | MatrixLike[float] | None = None,
        faces: CellArrayLike | None = None,
        n_faces: int | None = None,
        lines: CellArrayLike | None = None,
        n_lines: int | None = None,
        strips: CellArrayLike | None = None,
        n_strips: int | None = None,
        deep: bool = False,  # noqa: FBT001, FBT002
        force_ext: str | None = None,
        force_float: bool = True,  # noqa: FBT001, FBT002
        verts: CellArrayLike | None = None,
        n_verts: int | None = None,
    ) -> None:
        """Initialize the polydata."""
        local_parms = locals()
        super().__init__()

        # allow empty input
        if var_inp is None:
            return

        # filename
        opt_kwarg = ['faces', 'n_faces', 'lines', 'n_lines']
        if isinstance(var_inp, (str, Path)):
            for kwarg in opt_kwarg:
                if local_parms[kwarg]:
                    msg = 'No other arguments should be set when first parameter is a string'
                    raise ValueError(msg)
            self._from_file(var_inp, force_ext=force_ext)  # is filename

            return

        # PolyData-like
        if isinstance(var_inp, _vtk.vtkPolyData):
            for kwarg in opt_kwarg:
                if local_parms[kwarg]:
                    msg = 'No other arguments should be set when first parameter is a PolyData'
                    raise ValueError(msg)
            if deep:
                self.deep_copy(var_inp)
            else:
                self.shallow_copy(var_inp)  # type: ignore[arg-type]
            # Validate connectivity
            self._raise_invalid_point_references(as_warning=True)
            return

        # First parameter is points
        if isinstance(var_inp, (np.ndarray, list, _vtk.vtkDataArray)):
            self.SetPoints(vtk_points(var_inp, deep=deep, force_float=force_float))

        else:
            msg = f"""
                Invalid Input type:

                Expected first argument to be either a:
                - vtkPolyData
                - pyvista.PolyData
                - numeric numpy.ndarray (1 or 2 dimensions)
                - List (flat or nested with 3 points per vertex)
                - vtkDataArray

                Instead got: {type(var_inp)}"""
            raise TypeError(dedent(msg.strip('\n')))

        # At this point, points have been setup, add faces and/or lines
        if faces is lines is strips is verts is None:
            # one cell per point (point cloud case)
            verts = self._make_vertex_cells(self.n_points)

        for k, v in (('verts', verts), ('strips', strips), ('faces', faces), ('lines', lines)):
            if v is None:
                continue

            # These properties can be supplied as either arrays or pre-constructed `CellArray`s
            if not isinstance(v, _vtk.vtkCellArray):
                try:
                    v = CellArray(v)  # noqa: PLW2901
                except CellSizeError as err:
                    # Raise an additional error so user knows which property triggered the error
                    msg = f'`{k}` cell array size is invalid.'
                    raise CellSizeError(msg) from err

            setattr(self, k, v)

        # deprecated 0.44.0, convert to error in 0.47.0, remove 0.48.0
        for k, v in (  # type: ignore[assignment]
            ('n_verts', n_verts),
            ('n_strips', n_strips),
            ('n_faces', n_faces),
            ('n_lines', n_lines),
        ):
            if v is not None:
                msg = f'PolyData constructor parameter `{k}` is deprecated and no longer used.'
                raise TypeError(msg)

    def _post_file_load_processing(self) -> None:
        """Execute after loading a PolyData from file."""
        # When loading files with just point arrays, create and
        # set the polydata vertices
        if self.n_points > 0 and self.n_cells == 0:
            self.verts = self._make_vertex_cells(self.n_points)
        else:
            # Validate connectivity
            self._raise_invalid_point_references(as_warning=True)

    def _check_invalid_point_references(
        self, attr: Literal['verts', 'lines', 'faces', 'strips']
    ) -> str | None:
        """Verify that a connectivity array does not reference invalid points."""
        vtkattr = {'verts': 'Verts', 'lines': 'Lines', 'faces': 'Polys', 'strips': 'Strips'}[attr]
        if getattr(self, f'GetNumberOf{vtkattr}')():
            n_points = self.n_points
            vtk_connectivity = getattr(self, f'Get{vtkattr}')().GetConnectivityArray()
            conn_min, conn_max = vtk_connectivity.GetRange()
            if conn_min < 0 or conn_max >= n_points:
                return (
                    f'The connectivity of `{type(self).__name__}.{attr}` includes references to\n'
                    f'point ids that do not exist. The point ids must be non-negative and strictly'
                    f' less than the\nnumber of points ({n_points}).'
                )
        return None

    def _raise_invalid_point_references(
        self,
        attr: Literal['verts', 'lines', 'faces', 'strips'] | None = None,
        *,
        as_warning: bool = False,
    ):
        """Raise error if the specified connectivity array has invalid point references.

        If attr is None, all connectivity arrays are checked.

        """

        def _raise_invalid_point_references(attr_):
            msg = self._check_invalid_point_references(attr_)
            if msg is not None:
                if as_warning:
                    warn_external(msg, pv.InvalidMeshWarning)
                else:
                    raise pv.InvalidMeshError(msg)

        if attr is None:
            # Check all connectivity arrays
            _raise_invalid_point_references('verts')
            _raise_invalid_point_references('lines')
            _raise_invalid_point_references('faces')
            _raise_invalid_point_references('strips')
            return

        _raise_invalid_point_references(attr)

    def __repr__(self) -> str:
        """Return the standard representation."""
        return DataSet.__repr__(self)

    def __str__(self) -> str:
        """Return the standard str representation."""
        return DataSet.__str__(self)

    @staticmethod
    def _make_vertex_cells(npoints: int) -> NumpyArray[int]:
        cells = np.empty((npoints, 2), dtype=pv.ID_TYPE)
        cells[:, 0] = 1
        cells[:, 1] = np.arange(npoints, dtype=pv.ID_TYPE)
        return cells

    @property
    def verts(self) -> NumpyArray[int]:  # numpydoc ignore=RT01
        """Get the vertex cells.

        Returns
        -------
        numpy.ndarray
            Array of vertex cell indices.

        Examples
        --------
        Create a point cloud polydata and return the vertex cells.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> rng = np.random.default_rng(seed=0)
        >>> points = rng.random((5, 3))
        >>> pdata = pv.PolyData(points)
        >>> pdata.verts
        array([1, 0, 1, 1, 1, 2, 1, 3, 1, 4])

        Set vertex cells.  Note how the mesh plots both the surface
        mesh and the additional vertices in a single plot.

        >>> mesh = pv.Plane(i_resolution=3, j_resolution=3)
        >>> mesh.verts = np.vstack(
        ...     (
        ...         np.ones(mesh.n_points, dtype=np.int64),
        ...         np.arange(mesh.n_points),
        ...     )
        ... ).T
        >>> mesh.plot(
        ...     color='lightblue',
        ...     render_points_as_spheres=True,
        ...     point_size=60,
        ... )

        Vertex cells can also be set to a ``CellArray``. The following
        ``verts`` assignment is equivalent to the one above.

        >>> mesh.verts = pv.CellArray.from_regular_cells(
        ...     np.arange(mesh.n_points).reshape((-1, 1))
        ... )

        """
        self.GetVerts().ExportLegacyFormat(arr := _vtk.vtkIdTypeArray())
        return _vtk.vtk_to_numpy(arr)

    @verts.setter
    def verts(self, verts: CellArrayLike) -> None:
        if isinstance(verts, _vtk.vtkCellArray):
            self.SetVerts(verts)
        else:
            self.SetVerts(CellArray(verts))
        self._raise_invalid_point_references('verts')

    @property
    def lines(self) -> NumpyArray[int]:  # numpydoc ignore=RT01
        """Return the connectivity array of the lines of this PolyData.

        Lines can also be set by assigning a :class:`~pyvista.CellArray`.

        Examples
        --------
        Return the lines from a spline.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> points = np.random.default_rng().random((3, 3))
        >>> spline = pv.Spline(points, 10)
        >>> spline.lines
        array([10,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9])

        """
        self.GetLines().ExportLegacyFormat(arr := _vtk.vtkIdTypeArray())
        return _vtk.vtk_to_numpy(arr).ravel()

    @lines.setter
    def lines(self, lines: CellArrayLike) -> None:
        if isinstance(lines, _vtk.vtkCellArray):
            self.SetLines(lines)
        else:
            self.SetLines(CellArray(lines))
        self._raise_invalid_point_references('lines')

    @property
    def faces(self) -> NumpyArray[int]:  # numpydoc ignore=RT01
        """Return the connectivity array of the faces of this PolyData.

        The faces array is organized as::

           [n0, p0_0, p0_1, ..., p0_n, n1, p1_0, p1_1, ..., p1_n, ...]

        where ``n0`` is the number of points in face 0, and ``pX_Y`` is the
        Y'th point in face X.

        For example, a triangle and a quadrilateral might be represented as::

           [3, 0, 1, 2, 4, 0, 1, 3, 4]

        Where the two individual faces would be ``[3, 0, 1, 2]`` and ``[4, 0, 1, 3, 4]``.

        Faces can also be set by assigning a :class:`~pyvista.CellArray` object
        instead of an array.

        Returns
        -------
        numpy.ndarray
            Array of face connectivity.

        See Also
        --------
        pyvista.PolyData.regular_faces
        pyvista.PolyData.irregular_faces

        Notes
        -----
        The array returned cannot be modified in place and will raise a
        ``ValueError`` if attempted.

        You can, however, set the faces directly. See the example.

        Examples
        --------
        >>> import pyvista as pv
        >>> plane = pv.Plane(i_resolution=2, j_resolution=2)
        >>> plane.faces
        array([4, 0, 1, 4, 3, 4, 1, 2, 5, 4, 4, 3, 4, 7, 6, 4, 4, 5, 8, 7])

        Note how the faces contain a "padding" indicating the number
        of points per face:

        >>> plane.faces.reshape(-1, 5)
        array([[4, 0, 1, 4, 3],
               [4, 1, 2, 5, 4],
               [4, 3, 4, 7, 6],
               [4, 4, 5, 8, 7]])

        Set the faces directly. The following example creates a simple plane
        with a single square faces and modifies it to have two triangles
        instead.

        >>> mesh = pv.Plane(i_resolution=1, j_resolution=1)
        >>> mesh.faces = [3, 0, 1, 2, 3, 3, 2, 1]
        >>> mesh.faces
        array([3, 0, 1, 2, 3, 3, 2, 1])

        """
        self.GetPolys().ExportLegacyFormat(arr := _vtk.vtkIdTypeArray())
        array = _vtk.vtk_to_numpy(arr)
        # Flag this array as read only to ensure users do not attempt to write to it.
        array.flags['WRITEABLE'] = False
        return array

    @faces.setter
    def faces(self, faces: CellArrayLike) -> None:
        if isinstance(faces, _vtk.vtkCellArray):
            self.SetPolys(faces)
        else:
            # TODO: faster to mutate in-place if array is same size?
            self.SetPolys(CellArray(faces))
        self._raise_invalid_point_references('faces')

    @property
    def regular_faces(self) -> NumpyArray[int]:  # numpydoc ignore=RT01
        """Return a face array of point indices when all faces have the same size.

        Returns
        -------
        numpy.ndarray
            Array of face indices with shape (n_faces, face_size).

        See Also
        --------
        pyvista.PolyData.faces

        Notes
        -----
        This property does not validate that the mesh's faces are all
        actually the same size. If they're not, this property may either
        raise a `ValueError` or silently return an incorrect array.

        Examples
        --------
        Get the regular face array of a plane with 2x2 arrangement of cells
        as a 4x4 array.

        >>> import pyvista as pv
        >>> plane = pv.Plane(i_resolution=2, j_resolution=2)
        >>> plane.regular_faces
        array([[0, 1, 4, 3],
               [1, 2, 5, 4],
               [3, 4, 7, 6],
               [4, 5, 8, 7]])

        """
        return _get_regular_cells(self.GetPolys())

    @regular_faces.setter
    def regular_faces(self, faces: MatrixLike[int]) -> None:  # numpydoc ignore=PR01
        """Set the face cells from an (n_faces, face_size) array."""
        self.faces = CellArray.from_regular_cells(faces)

    @classmethod
    @_deprecate_positional_args(allowed=['points', 'faces'])
    def from_regular_faces(
        cls,
        points: MatrixLike[float],
        faces: MatrixLike[int],
        deep: bool = False,  # noqa: FBT001, FBT002
    ):
        """Alternate `pyvista.PolyData` convenience constructor from point and regular face arrays.

        Parameters
        ----------
        points : MatrixLike[float]
            A (n_points, 3) array of points.

        faces : MatrixLike[int]
            A (n_faces, face_size) array of face indices. For a triangle mesh, ``face_size = 3``.

        deep : bool, default: False
            Whether to deep copy the faces array into :vtk:`vtkCellArray` connectivity data.

        Returns
        -------
        pyvista.PolyData
            The newly constructed mesh.

        See Also
        --------
        pyvista.PolyData.from_irregular_faces

        Examples
        --------
        Construct a tetrahedron from four triangles

        >>> import pyvista as pv
        >>> points = [[1.0, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1]]
        >>> faces = [[0, 1, 2], [1, 3, 2], [0, 2, 3], [0, 3, 1]]
        >>> tetra = pv.PolyData.from_regular_faces(points, faces)
        >>> tetra.plot()

        """
        return cls(points, faces=CellArray.from_regular_cells(faces, deep=deep))

    @property
    def irregular_faces(self) -> tuple[NumpyArray[int], ...]:  # numpydoc ignore=RT01
        """Return a tuple of face arrays.

        Returns
        -------
        tuple[numpy.ndarray]
            Tuple of length n_faces where each element is an array of point
            indices for points in that face.

        See Also
        --------
        pyvista.PolyData.faces
        pyvista.PolyData.regular_faces

        Examples
        --------
        Get the face arrays of the five faces of a pyramid.

        >>> import pyvista as pv
        >>> pyramid = pv.Pyramid().extract_surface()
        >>> pyramid.irregular_faces  # doctest: +NORMALIZE_WHITESPACE
        (array([0, 1, 2, 3]),
         array([0, 3, 4]),
         array([0, 4, 1]),
         array([3, 2, 4]),
         array([2, 1, 4]))

        """
        return _get_irregular_cells(self.GetPolys())

    @irregular_faces.setter
    def irregular_faces(self, faces: Sequence[VectorLike[int]]) -> None:  # numpydoc ignore=PR01
        """Set the faces from a sequence of face arrays."""
        self.faces = CellArray.from_irregular_cells(faces)  # type: ignore[arg-type]

    @classmethod
    def from_irregular_faces(cls, points: MatrixLike[float], faces: Sequence[VectorLike[int]]):
        """Alternate `pyvista.PolyData` convenience constructor from point and ragged face arrays.

        Parameters
        ----------
        points : MatrixLike[float]
            A (n_points, 3) array of points.

        faces : Sequence[VectorLike[int]]
            A sequence of face vectors containing point indices.

        Returns
        -------
        pyvista.PolyData
            The newly constructed mesh.

        See Also
        --------
        pyvista.PolyData.from_regular_faces

        Examples
        --------
        Construct a pyramid from five points and five faces

        >>> import pyvista as pv
        >>> points = [
        ...     (1, 1, 0),
        ...     (-1, 1, 0),
        ...     (-1, -1, 0),
        ...     (1, -1, 0),
        ...     (0, 0, 1.61),
        ... ]
        >>> faces = [
        ...     (0, 1, 2, 3),
        ...     (0, 3, 4),
        ...     (0, 4, 1),
        ...     (3, 2, 4),
        ...     (2, 1, 4),
        ... ]
        >>> pyramid = pv.PolyData.from_irregular_faces(points, faces)
        >>> pyramid.plot()

        """
        return cls(points, faces=CellArray.from_irregular_cells(faces))  # type: ignore[arg-type]

    @property
    def strips(self) -> NumpyArray[int]:  # numpydoc ignore=RT01
        """Return a pointer to the strips as a numpy array.

        Returns
        -------
        numpy.ndarray
            Array of strip indices.

        Examples
        --------
        >>> import pyvista as pv
        >>> polygon = pv.Rectangle()
        >>> extruded = polygon.extrude((0, 0, 1), capping=False)
        >>> extruded.strips
        array([4, 0, 1, 4, 5, 4, 1, 2, 5, 6, 4, 2, 3, 6, 7, 4, 3, 0, 7, 4])

        """
        self.GetStrips().ExportLegacyFormat(arr := _vtk.vtkIdTypeArray())
        return _vtk.vtk_to_numpy(arr)

    @strips.setter
    def strips(self, strips: CellArrayLike) -> None:
        if isinstance(strips, _vtk.vtkCellArray):
            self.SetStrips(strips)
        else:
            self.SetStrips(CellArray(strips))
        self._raise_invalid_point_references('strips')

    @property
    def is_all_triangles(self) -> bool:  # numpydoc ignore=RT01
        """Return if all the faces of the :class:`pyvista.PolyData` are triangles.

        Returns
        -------
        bool
            ``True`` if all the faces of the :class:`pyvista.PolyData`
            are triangles and does not contain any vertices or lines.

        Examples
        --------
        Show a mesh from :func:`pyvista.Plane` is not composed of all
        triangles.

        >>> import pyvista as pv
        >>> plane = pv.Plane()
        >>> plane.is_all_triangles
        False

        Show that the mesh from :func:`pyvista.Sphere` contains only
        triangles.

        >>> sphere = pv.Sphere()
        >>> sphere.is_all_triangles
        True

        """
        # Need to make sure there are only face cells and no lines/verts
        if not self.n_faces_strict or self.n_lines or self.n_verts:
            return False

        # early return if not all triangular
        if self._connectivity_array.size % 3:
            return False

        # next, check if there are three points per face
        return bool((np.diff(self._offset_array) == 3).all())

    def __sub__(self, cutting_mesh):
        """Compute boolean difference of two meshes."""
        return self.boolean_difference(cutting_mesh)

    def __isub__(self, cutting_mesh):
        """Compute boolean difference of two meshes and update this mesh."""
        return self.boolean_difference(cutting_mesh)

    def __and__(self, other_mesh):
        """Compute boolean intersection of two meshes."""
        return self.boolean_intersection(other_mesh)

    def __or__(self, other_mesh):
        """Compute boolean union of two meshes."""
        return self.boolean_union(other_mesh)

    @property
    def _offset_array(self) -> NumpyArray[int]:
        """Return the array used to store cell offsets."""
        return _get_offset_array(self.GetPolys())

    @property
    def _connectivity_array(self) -> NumpyArray[int]:
        """Return the array with the point ids that define the cells' connectivity."""
        return _get_connectivity_array(self.GetPolys())

    @property
    def n_lines(self) -> int:  # numpydoc ignore=RT01
        """Return the number of lines.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Line()
        >>> mesh.n_lines
        1

        """
        return self.GetNumberOfLines()

    @property
    def n_verts(self) -> int:  # numpydoc ignore=RT01
        """Return the number of vertices.

        A vertex is a 0D cell, which is usually a cell that references one point,
        a :vtk:`vtkVertex`. It can also be a :vtk:`vtkPolyVertex`.
        See `pyvista.PolyData.n_points` for the more common measure.

        Examples
        --------
        Create a simple mesh containing just two points and return the
        number of vertices. By default, when constructing a PolyData with points but no cells,
        vertices are automatically created, one per point.

        >>> import pyvista as pv
        >>> mesh = pv.PolyData([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        >>> mesh.n_points, mesh.n_verts
        (2, 2)

        If any other cells are specified, these vertices are not created.

        >>> import pyvista as pv
        >>> mesh = pv.PolyData([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]], lines=[2, 0, 1])
        >>> mesh.n_points, mesh.n_verts
        (2, 0)

        """
        return self.GetNumberOfVerts()

    @property
    def n_strips(self) -> int:  # numpydoc ignore=RT01
        """Return the number of strips.

        Examples
        --------
        Create a simple mesh with one triangle strip and return the
        number of triangles.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> vertices = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        >>> strip = np.array([3, 0, 1, 2])
        >>> mesh = pv.PolyData(vertices, strips=strip)
        >>> mesh.n_strips
        1

        """
        return self.GetNumberOfStrips()

    @staticmethod
    def use_strict_n_faces(mode: bool) -> None:  # noqa: FBT001
        """Global opt-in to strict n_faces.

        Parameters
        ----------
        mode : bool
            If true, all future calls to :attr:`n_faces <pyvista.PolyData.n_faces>`
            will return the same thing as :attr:`n_faces_strict <pyvista.PolyData.n_faces_strict>`.

        """
        PolyData._USE_STRICT_N_FACES = mode

    @property
    def n_faces(self) -> int:  # numpydoc ignore=RT01
        """Return the number of cells.

        .. deprecated:: 0.43.0
            The current (deprecated) behavior of this property is to
            return the total number of cells, i.e. the sum of the number of
            vertices, lines, triangle strips, and polygonal faces.
            In the future, this will change to return only the number of
            polygonal faces, i.e. those cells represented in the
            `pv.PolyData.faces` array. If you want the total number of cells,
            use `pv.PolyData.n_cells`. If you want only the number of polygonal faces,
            use `pv.PolyData.n_faces_strict`. Alternatively, you can opt into the
            future behavior globally by calling `pv.PolyData.use_strict_n_faces(True)`,
            in which case `pv.PolyData.n_faces` will return the same thing as
            `pv.PolyData.n_faces_strict`.

        """
        if PolyData._USE_STRICT_N_FACES:
            return self.n_faces_strict

        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.49.0
        msg = (
            'The non-strict behavior of `pv.PolyData.n_faces` has been removed. '
            'Use `pv.PolyData.n_cells` or `pv.PolyData.n_faces_strict` instead. '
            'See the documentation in `pv.PolyData.n_faces` for more information.'
        )
        raise AttributeError(msg)

    @property
    def n_faces_strict(self) -> int:  # numpydoc ignore=RT01
        """Return the number of polygonal faces.

        Returns
        -------
        int :
             Number of faces represented in the :attr:`n_faces <pyvista.PolyData.n_faces>` array.

        Examples
        --------
        Create a mesh with one face and one line

        >>> import pyvista as pv
        >>> mesh = pv.PolyData(
        ...     [(0.0, 0, 0), (1, 0, 0), (0, 1, 0)],
        ...     faces=[3, 0, 1, 2],
        ...     lines=[2, 0, 1],
        ... )
        >>> mesh.n_cells, mesh.n_faces_strict
        (2, 1)

        """
        return self.GetNumberOfPolys()

    @_deprecate_positional_args(allowed=['filename'])
    def save(  # type: ignore[override]  # noqa: PLR0917
        self,
        filename: Path | str,
        binary: bool = True,  # noqa: FBT001, FBT002
        texture: NumpyArray[np.uint8] | str | None = None,
        recompute_normals: bool = True,  # noqa: FBT001, FBT002
        compression: _CompressionOptions = 'zlib',
    ) -> None:
        """Write a surface mesh to disk.

        Written file may be an ASCII or binary ply, stl, or vtk mesh
        file.

        Parameters
        ----------
        filename : str, Path
            Filename of mesh to be written.  File type is inferred from
            the extension of the filename unless overridden with
            ftype.  Can be one of many of the supported  the following
            types (``'.ply'``, ``'.vtp'``, ``'.stl'``, ``'.vtk``, ``'.geo'``,
            ``'.obj'``, ``'.iv'``).

        binary : bool, default: True
            Writes the file as binary when ``True`` and ASCII when ``False``.

        texture : str, numpy.ndarray, optional
            Write a single texture array to file when using a PLY
            file.  Texture array must be a 3 or 4 component array with
            the datatype ``np.uint8``.  Array may be a cell array or a
            point array, and may also be a string if the array already
            exists in the PolyData.

            If a string is provided, the texture array will be saved
            to disk as that name.  If an array is provided, the
            texture array will be saved as ``'RGBA'`` if the array
            contains an alpha channel (i.e. 4 component array), or
            as ``'RGB'`` if the array is just a 3 component array.

            .. note::
               This feature is only available when saving PLY files.

        recompute_normals : bool, default: True
            When ``True``, if ply or stl format is chosen, the face normals
            are computed in place to ensure the mesh is properly saved.
            Set this to ``False`` to save instead the already existing normal
            array in the PolyData.

        compression : str or None, default: 'zlib'
            The compression type to use when ``binary`` is ``True``
            and VTK writer is of type :vtk:`vtkXMLWriter`. This
            argument has no effect otherwise. Acceptable values are
            ``'zlib'``, ``'lz4'``, ``'lzma'``, and ``None``. ``None``
            indicates no compression.

            .. versionadded:: 0.47

        Notes
        -----
        Binary files write much faster than ASCII and have a smaller
        file size.

        Examples
        --------
        Save a mesh as a STL.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> sphere.save('my_mesh.stl')  # doctest:+SKIP

        Save a mesh as a PLY.

        >>> sphere = pv.Sphere()
        >>> sphere.save('my_mesh.ply')  # doctest:+SKIP

        Save a mesh as a PLY with a texture array.  Here we also
        create a simple RGB array representing the texture.

        >>> import numpy as np
        >>> sphere = pv.Sphere()
        >>> texture = np.zeros((sphere.n_points, 3), np.uint8)
        >>> # Just the green channel is set as a repeatedly
        >>> # decreasing value
        >>> texture[:, 1] = np.arange(sphere.n_points)[::-1]
        >>> sphere.point_data['my_texture'] = texture
        >>> sphere.save('my_mesh.ply', texture='my_texture')  # doctest:+SKIP

        Alternatively, provide just the texture array.  This will be
        written to the file as ``'RGB'`` since it does not contain an
        alpha channel.

        >>> sphere.save('my_mesh.ply', texture=texture)  # doctest:+SKIP

        Save a mesh as a VTK file.

        >>> sphere = pv.Sphere()
        >>> sphere.save('my_mesh.vtk')  # doctest:+SKIP

        """
        filename = Path(filename).expanduser().resolve()
        ftype = get_ext(filename)
        # Recompute normals prior to save.  Corrects a bug were some
        # triangular meshes are not saved correctly
        if ftype in ['.stl', '.ply'] and recompute_normals:
            with contextlib.suppress(TypeError):
                self.compute_normals(inplace=True)
        super().save(filename, binary=binary, texture=texture, compression=compression)

    @property
    def volume(self) -> float:  # numpydoc ignore=RT01
        """Return the approximate volume of the dataset.

        This will throw a VTK error/warning if not a closed surface.

        Returns
        -------
        float
            Total volume of the mesh.

        Examples
        --------
        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> sphere.volume
        0.5183

        """
        mprop = _vtk.vtkMassProperties()
        mprop.SetInputData(self.triangulate())
        return mprop.GetVolume()

    @property
    def point_normals(self) -> pv.pyvista_ndarray:  # numpydoc ignore=RT01
        """Return the point normals.

        The active point normals are returned if they exist. Otherwise, they
        are computed with :func:`~pyvista.PolyDataFilters.compute_normals`
        using the default options.

        Returns
        -------
        pyvista.pyvista_ndarray
            Array of point normals.

        Examples
        --------
        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> sphere.point_normals
        pyvista_ndarray([[ 0.        ,  0.        ,  1.        ],
                         [ 0.        ,  0.        , -1.        ],
                         [ 0.10811902,  0.        ,  0.99413794],
                         ...,
                         [ 0.31232402, -0.06638652, -0.9476532 ],
                         [ 0.21027282, -0.04469487, -0.97662055],
                         [ 0.10575636, -0.02247921, -0.99413794]],
                        shape=(842, 3), dtype=float32)

        """
        if self.point_data.active_normals is not None:
            normals = self.point_data.active_normals
        else:
            normals = self.compute_normals(cell_normals=False, inplace=False).point_data['Normals']
        return normals

    @property
    def cell_normals(self) -> pv.pyvista_ndarray:  # numpydoc ignore=RT01
        """Return the cell normals.

        The active cell normals are returned if they exist. Otherwise, they
        are computed with :func:`~pyvista.PolyDataFilters.compute_normals`
        using the default options.

        Returns
        -------
        pyvista.pyvista_ndarray
            Array of cell normals.

        Examples
        --------
        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> sphere.cell_normals
        pyvista_ndarray([[ 0.05413816,  0.00569015,  0.9985172 ],
                         [ 0.05177207,  0.01682176,  0.9985172 ],
                         [ 0.04714328,  0.02721819,  0.9985172 ],
                         ...,
                         [ 0.26742265, -0.02810723, -0.96316934],
                         [ 0.1617585 , -0.01700151, -0.9866839 ],
                         [ 0.1617585 , -0.01700151, -0.9866839 ]],
                        shape=(1680, 3), dtype=float32)

        """
        if self.cell_data.active_normals is not None:
            normals = self.cell_data.active_normals
        else:
            normals = self.compute_normals(point_normals=False, inplace=False).cell_data['Normals']
        return normals

    @property
    def face_normals(self) -> pv.pyvista_ndarray:  # numpydoc ignore=RT01
        """Return the cell normals.

        Alias to :func:`PolyData.cell_normals`.

        Returns
        -------
        pyvista.pyvista_ndarray
            Array of face normals.

        Examples
        --------
        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> sphere.face_normals
        pyvista_ndarray([[ 0.05413816,  0.00569015,  0.9985172 ],
                         [ 0.05177207,  0.01682176,  0.9985172 ],
                         [ 0.04714328,  0.02721819,  0.9985172 ],
                         ...,
                         [ 0.26742265, -0.02810723, -0.96316934],
                         [ 0.1617585 , -0.01700151, -0.9866839 ],
                         [ 0.1617585 , -0.01700151, -0.9866839 ]],
                        shape=(1680, 3), dtype=float32)

        """
        return self.cell_normals

    @cached_property
    def obbTree(self) -> _vtk.vtkOBBTree:  # noqa: N802  # numpydoc ignore=RT01
        """Return the obbTree of the polydata.

        An obbTree is an object to generate oriented bounding box (OBB)
        trees. An oriented bounding box is a bounding box that does not
        necessarily line up along coordinate axes. The OBB tree is a
        hierarchical tree structure of such boxes, where deeper levels of OBB
        confine smaller regions of space.

        .. warning::

            This property is expensive to compute and is therefore cached. If the mesh's
            geometry is modified, the obb tree will no longer be valid.

        """
        obb_tree = _vtk.vtkOBBTree()
        obb_tree.SetDataSet(self)
        obb_tree.BuildLocator()
        return obb_tree

    @property
    def n_open_edges(self) -> int:  # numpydoc ignore=RT01
        """Return the number of open edges on this mesh.

        Examples
        --------
        Return the number of open edges on a sphere.

        >>> import pyvista as pv
        >>> sphere = pv.Sphere()
        >>> sphere.n_open_edges
        0

        Return the number of open edges on a plane.

        >>> plane = pv.Plane(i_resolution=1, j_resolution=1)
        >>> plane.n_open_edges
        4

        """
        alg = _vtk.vtkFeatureEdges()
        alg.FeatureEdgesOff()
        alg.BoundaryEdgesOn()
        alg.NonManifoldEdgesOn()
        alg.SetInputDataObject(self)
        alg.Update()
        return alg.GetOutput().GetNumberOfCells()

    @property
    def is_manifold(self) -> bool:  # numpydoc ignore=RT01
        """Return if the mesh is manifold (no open edges).

        Examples
        --------
        Show a sphere is manifold.

        >>> import pyvista as pv
        >>> pv.Sphere().is_manifold
        True

        Show a plane is not manifold.

        >>> pv.Plane().is_manifold
        False

        """
        return self.n_open_edges == 0

    def __del__(self) -> None:
        """Delete the object."""
        # avoid a reference cycle that can't be resolved with vtkPolyData
        self._glyph_geom = None
        self.obbTree = None  # type: ignore[assignment]


@abstract_class
class PointGrid(_PointSet):
    """Class in common with structured and unstructured grids."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ARG002
        """Initialize the point grid."""
        super().__init__()

    def plot_curvature(self: Self, curv_type='mean', **kwargs):
        """Plot the curvature of the external surface of the grid.

        Parameters
        ----------
        curv_type : str, default: "mean"
            One of the following strings indicating curvature types.
            - ``'mean'``
            - ``'gaussian'``
            - ``'maximum'``
            - ``'minimum'``

        **kwargs : dict, optional
            Optional keyword arguments.  See :func:`pyvista.plot`.

        Returns
        -------
        list
            Camera position, focal point, and view up.  Returned when
            ``return_cpos`` is ``True``.

        """
        trisurf = self.extract_surface().triangulate()
        return trisurf.plot_curvature(curv_type, **kwargs)


class UnstructuredGrid(PointGrid, UnstructuredGridFilters, _vtk.vtkUnstructuredGrid):
    """Dataset used for arbitrary combinations of all possible cell types.

    Can be initialized by the following:

    - Creating an empty grid
    - From a :vtk:`vtkPolyData` or :vtk:`vtkStructuredGrid` object
    - From cell, cell types, and point arrays
    - From a file

    Parameters
    ----------
    args : str, :vtk:`vtkUnstructuredGrid`, iterable
        See examples below.
    deep : bool, default: False
        Whether to deep copy a :vtk:`vtkUnstructuredGrid` object.
        Default is ``False``.  Keyword only.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> import vtk

    Create an empty grid

    >>> grid = pv.UnstructuredGrid()

    Copy a :vtk:`vtkUnstructuredGrid`

    >>> vtkgrid = vtk.vtkUnstructuredGrid()
    >>> grid = pv.UnstructuredGrid(vtkgrid)

    From a filename.

    >>> grid = pv.UnstructuredGrid(examples.hexbeamfile)
    >>> grid.plot(show_edges=True)

    From arrays. Here we create a single tetrahedron.

    >>> cells = [4, 0, 1, 2, 3]
    >>> celltypes = [pv.CellType.TETRA]
    >>> points = [
    ...     [1.0, 1.0, 1.0],
    ...     [1.0, -1.0, -1.0],
    ...     [-1.0, 1.0, -1.0],
    ...     [-1.0, -1.0, 1.0],
    ... ]
    >>> grid = pv.UnstructuredGrid(cells, celltypes, points)
    >>> grid.plot(show_edges=True)

    See the :ref:`create_unstructured_surface_example` example for more details
    on creating unstructured grids within PyVista.

    """

    _WRITERS: ClassVar[dict[str, type[BaseWriter]]] = {
        '.vtu': XMLUnstructuredGridWriter,
        '.vtk': UnstructuredGridWriter,
    }
    if _vtk.vtk_version_info >= (9, 4):
        _WRITERS['.vtkhdf'] = HDFWriter

    def __init__(self, *args, deep: bool = False, **kwargs) -> None:
        """Initialize the unstructured grid."""
        super().__init__()

        if not args:
            return
        if len(args) == 1:
            if isinstance(args[0], _vtk.vtkUnstructuredGrid):
                if deep:
                    self.deep_copy(args[0])
                else:
                    self.shallow_copy(args[0])  # type: ignore[arg-type]

            elif isinstance(args[0], (str, Path)):
                self._from_file(args[0], **kwargs)

            elif isinstance(args[0], (_vtk.vtkStructuredGrid, _vtk.vtkPolyData)):
                vtkappend = _vtk.vtkAppendFilter()
                vtkappend.AddInputData(args[0])
                vtkappend.Update()
                self.shallow_copy(vtkappend.GetOutput())

            else:
                itype = type(args[0])
                msg = f'Cannot work with input type {itype}'
                raise TypeError(msg)

        # Cell dictionary creation
        elif len(args) == 2 and isinstance(args[0], dict) and isinstance(args[1], np.ndarray):
            self._from_cells_dict(args[0], args[1], deep=deep)
            self._check_for_consistency()

        elif len(args) == 3:
            arg0_is_seq = isinstance(args[0], (np.ndarray, Sequence))
            arg1_is_seq = isinstance(args[1], (np.ndarray, Sequence))
            arg2_is_seq = isinstance(args[2], (np.ndarray, Sequence))

            if all([arg0_is_seq, arg1_is_seq, arg2_is_seq]):
                self._from_arrays(args[0], args[1], args[2], deep=deep, **kwargs)
                self._check_for_consistency()
            else:
                msg = 'All input types must be sequences.'
                raise TypeError(msg)
        else:
            msg = (
                'Invalid parameters.  Initialization with arrays requires the '
                'following arrays:\n`cells`, `cell_type`, `points`'
            )
            raise TypeError(msg)

    def __repr__(self):
        """Return the standard representation."""
        return DataSet.__repr__(self)

    def __str__(self):
        """Return the standard str representation."""
        return DataSet.__str__(self)

    def _from_cells_dict(self, cells_dict, points, *, deep: bool = True):
        if points.ndim != 2 or points.shape[-1] != 3:
            msg = 'Points array must be a [M, 3] array'
            raise ValueError(msg)

        nr_points = points.shape[0]
        cell_types, cells = create_mixed_cells(cells_dict, nr_points)
        self._from_arrays(cells, cell_types, points, deep=deep)

    def _from_arrays(
        self,
        cells,
        cell_type,
        points,
        *,
        deep: bool = True,
        force_float: bool = True,
    ) -> None:
        """Create VTK unstructured grid from numpy arrays.

        Parameters
        ----------
        cells : sequence[int]
            Array of cells.  Each cell contains the number of points in the
            cell and the node numbers of the cell.

        cell_type : sequence[int]
            Cell types of each cell.  Each cell type numbers can be found from
            vtk documentation.  More efficient if using ``np.uint8``. See
            example below.

        points : sequence[float]
            Numpy array containing point locations.

        deep : bool, default: True
            When ``True``, makes a copy of the points array.  Default
            ``False``.  Cells and cell types are always copied.

        force_float : bool, default: True
            Casts the datatype to ``float32`` if points datatype is
            non-float.  Set this to ``False`` to allow non-float types,
            though this may lead to truncation of intermediate floats when
            transforming datasets.

        Examples
        --------
        >>> import numpy as np
        >>> from pyvista import CellType
        >>> import pyvista as pv
        >>> cell0_ids = [8, 0, 1, 2, 3, 4, 5, 6, 7]
        >>> cell1_ids = [8, 8, 9, 10, 11, 12, 13, 14, 15]
        >>> cells = np.hstack((cell0_ids, cell1_ids))
        >>> cell_type = np.array([CellType.HEXAHEDRON, CellType.HEXAHEDRON], np.int8)

        >>> cell1 = np.array(
        ...     [
        ...         [0, 0, 0],
        ...         [1, 0, 0],
        ...         [1, 1, 0],
        ...         [0, 1, 0],
        ...         [0, 0, 1],
        ...         [1, 0, 1],
        ...         [1, 1, 1],
        ...         [0, 1, 1],
        ...     ],
        ...     dtype=np.float32,
        ... )

        >>> cell2 = np.array(
        ...     [
        ...         [0, 0, 2],
        ...         [1, 0, 2],
        ...         [1, 1, 2],
        ...         [0, 1, 2],
        ...         [0, 0, 3],
        ...         [1, 0, 3],
        ...         [1, 1, 3],
        ...         [0, 1, 3],
        ...     ],
        ...     dtype=np.float32,
        ... )

        >>> points = np.vstack((cell1, cell2))

        >>> grid = pv.UnstructuredGrid(cells, cell_type, points)

        """
        # convert to arrays upfront
        cells = np.asarray(cells)
        cell_type = np.asarray(cell_type)
        points = np.asarray(points)

        # Convert to vtk arrays
        vtkcells = CellArray(cells)
        if cell_type.dtype != np.uint8:
            cell_type = cell_type.astype(np.uint8)
        cell_type = _vtk.numpy_to_vtk(cell_type, deep=deep)

        points = vtk_points(points, deep=deep, force_float=force_float)
        self.SetPoints(points)

        self.SetCells(cell_type, vtkcells)

    def _check_for_consistency(self):
        """Check if size of offsets and celltypes match the number of cells.

        Checks if the number of offsets and celltypes correspond to
        the number of cells.  Called after initialization of the self
        from arrays.
        """
        if self.n_cells != self.celltypes.size:
            msg = (
                f'Number of cell types ({self.celltypes.size}) '
                f'must match the number of cells {self.n_cells})'
            )
            raise ValueError(msg)

        if self.n_cells != self.offset.size - 1:  # pragma: no cover
            msg = (
                f'Size of the offset ({self.offset.size}) '
                f'must be one greater than the number of cells ({self.n_cells})'
            )
            raise ValueError(msg)

    @property
    def cells(self) -> NumpyArray[int]:  # numpydoc ignore=RT01
        """Return the cell data as a numpy object.

        This is the old style VTK data layout::

           [n0, p0_0, p0_1, ..., p0_n, n1, p1_0, p1_1, ..., p1_n, ...]

        where ``n0`` is the number of points in cell 0, and ``pX_Y`` is the
        Y'th point in cell X.

        For example, a triangle and a line might be represented as::

           [3, 0, 1, 2, 2, 0, 1]

        Where the two individual cells would be ``[3, 0, 1, 2]`` and ``[2, 0, 1]``.

        See Also
        --------
        pyvista.DataSet.get_cell
        pyvista.UnstructuredGrid.cell_connectivity
        pyvista.UnstructuredGrid.offset

        Notes
        -----
        The array returned cannot be modified in place and will raise a
        ``ValueError`` if attempted.

        You can, however, set the cells directly. See the example.

        Examples
        --------
        Return the indices of the first two cells from the example hex
        beam.  Note how the cells have "padding" indicating the number
        of points per cell.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> grid = examples.load_hexbeam()
        >>> grid.cells[:18]
        array([ 8,  0,  2,  8,  7, 27, 36, 90, 81,  8,  2,  1,  4,  8, 36, 18, 54,
               90])

        While you cannot change the array inplace, you can overwrite it. For example:

        >>> grid.cells = [8, 0, 1, 2, 3, 4, 5, 6, 7]

        """
        # Flag this array as read only to ensure users do not attempt to write to it.
        self._get_cells().ExportLegacyFormat(arr := _vtk.vtkIdTypeArray())
        array = _vtk.vtk_to_numpy(arr)
        array.flags['WRITEABLE'] = False
        return array

    @cells.setter
    def cells(self, cells) -> None:
        vtk_idarr = numpy_to_idarr(cells, deep=False, return_ind=False)
        self._get_cells().ImportLegacyFormat(vtk_idarr)

    def _get_cells(self):
        cells = self.GetCells()
        return _vtk.vtkCellArray() if cells is None else cells  # type: ignore[redundant-expr]

    @property
    def faces(self) -> NumpyArray[int]:
        """Return the polyhedron faces.

        .. deprecated:: 0.45.0
            This property is deprecated and will be removed in a future release.
            VTK has deprecated `GetFaces` and `GetFaceLocations` in VTK 9.4 and
            may be removed in a future release of VTK. Please use
            `polyhedral_faces` instead.

        Returns
        -------
        numpy.ndarray
            Array of faces.

        """
        return convert_array(self.GetFaces())

    @property
    def polyhedron_faces(self) -> NumpyArray[int]:
        """Return the polyhedron faces.

        Returns
        -------
        numpy.ndarray
            Array of faces.

        """
        if pv.vtk_version_info < (9, 4):
            polyhedron_faces = pv.convert_array(self.GetFaces())

            if polyhedron_faces is None:
                return np.array([], dtype=int)  # type: ignore[unreachable]

            cell_faces = []
            i = 0

            while i < len(polyhedron_faces):
                faces_: list[VectorLike[int]] = []
                n_faces = polyhedron_faces[i]
                i += 1

                while len(faces_) < n_faces:
                    n_vertices = polyhedron_faces[i]
                    faces_.append([n_vertices, *polyhedron_faces[i + 1 : i + 1 + n_vertices]])
                    i += n_vertices + 1

                cell_faces.append(np.concatenate(faces_))

            return np.concatenate(cell_faces)

        else:
            faces = self.GetPolyhedronFaces()  # vtkCellArray
            if faces is None:
                return np.array([], dtype=int)  # type: ignore[unreachable]

            faces.ExportLegacyFormat(arr := _vtk.vtkIdTypeArray())
            return convert_array(arr)

    @property
    def face_locations(self) -> NumpyArray[int]:
        """Return polyhedron face locations.

        .. deprecated:: 0.45.0
            This property is deprecated and will be removed in a future release.
            VTK has deprecated `GetFaces` and `GetFaceLocations` in VTK 9.4 and
            may be removed in a future release of VTK. Please use
            `polyhedral_face_locations` instead.

        Returns
        -------
        numpy.ndarray
            Array of face locations.

        """
        return convert_array(self.GetFaceLocations())

    @property
    def polyhedron_face_locations(self) -> NumpyArray[int]:
        """Return the polyhedron face locations.

        Returns
        -------
        numpy.ndarray
            Array of faces.

        """
        if pv.vtk_version_info < (9, 4):
            polyhedron_faces = pv.convert_array(self.GetFaces())

            if polyhedron_faces is None:
                return np.array([], dtype=int)  # type: ignore[unreachable]

            i, face_counts = 0, []

            while i < len(polyhedron_faces):
                n_faces = polyhedron_faces[i]
                face_counts.append(n_faces)
                face_count = 0
                i += 1

                while face_count < n_faces:
                    i += polyhedron_faces[i] + 1
                    face_count += 1

            locations = [[0]] * self.n_cells
            face_count = 0

            for i, n_faces in zip(
                np.flatnonzero(self.celltypes == pv.CellType.POLYHEDRON),
                face_counts,
                strict=True,
            ):
                locations[i] = [n_faces, *(np.arange(n_faces) + face_count)]
                face_count += n_faces

            return np.concatenate(locations)

        else:
            faces = self.GetPolyhedronFaceLocations()  # vtkCellArray
            if faces is None:
                return np.array([], dtype=int)  # type: ignore[unreachable]

            faces.ExportLegacyFormat(arr := _vtk.vtkIdTypeArray())
            return convert_array(arr)

    @property
    def cells_dict(self) -> dict[np.uint8, NumpyArray[int]]:  # numpydoc ignore=RT01
        """Return a dictionary that contains all cells mapped from cell types.

        This function returns a :class:`numpy.ndarray` for each cell
        type in an ordered fashion.  Note that this function only
        works with element types of fixed sizes.

        .. versionchanged:: 0.46

            An empty dict ``{}`` is returned instead of ``None`` if
            the input is empty.

        Returns
        -------
        dict
            A dictionary mapping containing all cells of this unstructured grid.
            Structure: vtk_enum_type (int) -> cells (:class:`numpy.ndarray`).

        See Also
        --------
        pyvista.DataSet.get_cell

        Examples
        --------
        Return the cells dictionary of the sample hex beam.  Note how
        there is only one key/value pair as the hex beam example is
        composed of only all hexahedral cells, which is
        ``CellType.HEXAHEDRON``, which evaluates to 12.

        Also note how there is no padding for the cell array.  This
        approach may be more helpful than the ``cells`` property when
        extracting cells.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> hex_beam = pv.read(examples.hexbeamfile)
        >>> hex_beam.cells_dict  # doctest:+SKIP
        {12: array([[ 0,  2,  8,  7, 27, 36, 90, 81],
                [ 2,  1,  4,  8, 36, 18, 54, 90],
                [ 7,  8,  6,  5, 81, 90, 72, 63],
                ...
                [44, 26, 62, 98, 11, 10, 13, 17],
                [89, 98, 80, 71, 16, 17, 15, 14],
                [98, 62, 53, 80, 17, 13, 12, 15]])}

        """
        return get_mixed_cells(self)

    @property
    def cell_connectivity(self) -> NumpyArray[int]:  # numpydoc ignore=RT01
        """Return the cell connectivity as a numpy array.

        This is effectively :attr:`UnstructuredGrid.cells` without the
        padding.

        Returns
        -------
        numpy.ndarray
            Connectivity array.

        See Also
        --------
        pyvista.DataSet.get_cell

        Examples
        --------
        Return the cell connectivity for the first two cells.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> hex_beam = pv.read(examples.hexbeamfile)
        >>> hex_beam.cell_connectivity[:16]
        array([ 0,  2,  8,  7, 27, 36, 90, 81,  2,  1,  4,  8, 36, 18, 54, 90])

        """
        carr = self._get_cells()
        return _vtk.vtk_to_numpy(carr.GetConnectivityArray())

    @_deprecate_positional_args
    def linear_copy(self, deep: bool = False):  # noqa: FBT001, FBT002
        """Return a copy of the unstructured grid containing only linear cells.

        Converts the following cell types to their linear equivalents.

        - :attr:`~pyvista.CellType.QUADRATIC_TRIANGLE`   --> :attr:`~pyvista.CellType.TRIANGLE`
        - :attr:`~pyvista.CellType.QUADRATIC_QUAD`       --> :attr:`~pyvista.CellType.QUAD`
        - :attr:`~pyvista.CellType.QUADRATIC_TETRA`      --> :attr:`~pyvista.CellType.TETRA`
        - :attr:`~pyvista.CellType.QUADRATIC_PYRAMID`    --> :attr:`~pyvista.CellType.PYRAMID`
        - :attr:`~pyvista.CellType.QUADRATIC_WEDGE`      --> :attr:`~pyvista.CellType.WEDGE`
        - :attr:`~pyvista.CellType.QUADRATIC_HEXAHEDRON` --> :attr:`~pyvista.CellType.HEXAHEDRON`

        Parameters
        ----------
        deep : bool, default: False
            When ``True``, makes a copy of the points array.
            Cells and cell types are always copied.

        Returns
        -------
        pyvista.UnstructuredGrid
            UnstructuredGrid containing only linear cells when
            ``deep=False``.

        """
        lgrid = self.copy(deep=deep)

        # grab the vtk object
        vtk_cell_type = _vtk.numpy_to_vtk(self._get_cell_types_array(), deep=True)
        celltype = _vtk.vtk_to_numpy(vtk_cell_type)
        celltype[celltype == CellType.QUADRATIC_TETRA] = CellType.TETRA
        celltype[celltype == CellType.QUADRATIC_PYRAMID] = CellType.PYRAMID
        celltype[celltype == CellType.QUADRATIC_WEDGE] = CellType.WEDGE
        celltype[celltype == CellType.QUADRATIC_HEXAHEDRON] = CellType.HEXAHEDRON

        # track quad mask for later
        quad_quad_mask = celltype == CellType.QUADRATIC_QUAD
        celltype[quad_quad_mask] = CellType.QUAD

        quad_tri_mask = celltype == CellType.QUADRATIC_TRIANGLE
        celltype[quad_tri_mask] = CellType.TRIANGLE

        cells = _vtk.vtkCellArray()
        cells.DeepCopy(self._get_cells())
        if pv.vtk_version_info >= (9, 5):
            face_locations = self.GetPolyhedronFaceLocations()
            faces = self.GetPolyhedronFaces()
            lgrid.SetPolyhedralCells(vtk_cell_type, cells, face_locations, faces)
        else:
            vtk_offset = self.GetCellLocationsArray()
            lgrid.SetCells(vtk_cell_type, vtk_offset, cells)

        # fixing bug with display of quad cells
        if np.any(quad_quad_mask):
            quad_offset = lgrid.offset[:-1][quad_quad_mask]
            base_point = lgrid.cell_connectivity[quad_offset]
            lgrid.cell_connectivity[quad_offset + 4] = base_point
            lgrid.cell_connectivity[quad_offset + 5] = base_point
            lgrid.cell_connectivity[quad_offset + 6] = base_point
            lgrid.cell_connectivity[quad_offset + 7] = base_point

        if np.any(quad_tri_mask):
            tri_offset = lgrid.offset[:-1][quad_tri_mask]
            base_point = lgrid.cell_connectivity[tri_offset]
            lgrid.cell_connectivity[tri_offset + 3] = base_point
            lgrid.cell_connectivity[tri_offset + 4] = base_point
            lgrid.cell_connectivity[tri_offset + 5] = base_point

        return lgrid

    @property
    def celltypes(self) -> NumpyArray[np.uint8]:  # numpydoc ignore=RT01
        """Return the cell types array.

        The array contains integer values corresponding to the :attr:`pyvista.Cell.type`
        of each cell in the dataset. See the :class:`pyvista.CellType` enum for more
        information about cell type.

        Returns
        -------
        numpy.ndarray
            Array of cell types.

        Examples
        --------
        This mesh contains only linear hexahedral cells, type
        :attr:`pyvista.CellType.HEXAHEDRON`, which evaluates to 12.

        >>> from pyvista import examples
        >>> hex_beam = examples.load_hexbeam()
        >>> hex_beam.celltypes
        array([12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
               12, 12, 12, 12, 12, 12], dtype=uint8)

        Compare this to :attr:`distinct_cell_types`.

        >>> hex_beam.distinct_cell_types
        {<CellType.HEXAHEDRON: 12>}

        """
        return _vtk.vtk_to_numpy(self._get_cell_types_array())

    def _get_cell_types_array(self):
        array = (
            self.GetCellTypes()  # type: ignore[call-arg,func-returns-value]
            if pv.vtk_version_info > (9, 5, 99)
            else self.GetCellTypesArray()
        )

        if array is None:
            array = _vtk.vtkUnsignedCharArray()
        return array

    @property
    def distinct_cell_types(self) -> set[CellType]:
        """Return the set of distinct cell types in this dataset.

        The set contains :class:`~pyvista.CellType` values corresponding to the
        :attr:`pyvista.Cell.type` of each distinct cell in the dataset.

        .. versionadded:: 0.47

        Returns
        -------
        set[CellType]
            Set of :class:`~pyvista.CellType` values.

        Examples
        --------
        Load a mesh with linear :attr:`pyvista.CellType.HEXAHEDRON` cells.

        >>> from pyvista import examples
        >>> hex_beam = examples.load_hexbeam()
        >>> hex_beam.distinct_cell_types
        {<CellType.HEXAHEDRON: 12>}

        Compare this to :attr:`celltypes`.

        >>> hex_beam.celltypes
        array([12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
               12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
               12, 12, 12, 12, 12, 12], dtype=uint8)

        """
        cell_types = (
            convert_array(self.GetDistinctCellTypesArray())
            if pv.vtk_version_info >= (9, 5, 0)
            else np.unique(self.celltypes)
        )
        return {pv.CellType(cell_num) for cell_num in cell_types}

    @property
    def offset(self) -> NumpyArray[float]:  # numpydoc ignore=RT01
        """Return the cell locations array.

        This is the location of the start of each cell in
        :attr:`cell_connectivity`.

        Returns
        -------
        numpy.ndarray
            Array of cell offsets indicating the start of each cell.

        Notes
        -----
        The array returned is immutable and cannot be written to. If you
        need to modify this array, create a copy of it using
        :func:`numpy.copy`.

        Examples
        --------
        Return the cell offset array.  Since this mesh is composed of
        all hexahedral cells, note how each cell starts at 8 greater
        than the prior cell.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> hex_beam = pv.read(examples.hexbeamfile)
        >>> hex_beam.offset
        array([  0,   8,  16,  24,  32,  40,  48,  56,  64,  72,  80,  88,  96,
               104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200,
               208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304,
               312, 320])

        """
        carr = self._get_cells()
        # This will be the number of cells + 1.
        array = _vtk.vtk_to_numpy(carr.GetOffsetsArray())
        array.flags['WRITEABLE'] = False
        return array

    def cast_to_explicit_structured_grid(self):
        """Cast to an explicit structured grid.

        Returns
        -------
        pyvista.ExplicitStructuredGrid
            An explicit structured grid.

        Raises
        ------
        TypeError
            If the unstructured grid doesn't have the ``'BLOCK_I'``,
            ``'BLOCK_J'`` and ``'BLOCK_K'`` cells arrays.

        See Also
        --------
        pyvista.ExplicitStructuredGrid.cast_to_unstructured_grid

        Examples
        --------
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)

        >>> grid = grid.hide_cells(range(80, 120))
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)

        >>> grid = grid.cast_to_unstructured_grid()
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)

        >>> grid = grid.cast_to_explicit_structured_grid()
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)

        """
        s1 = {'BLOCK_I', 'BLOCK_J', 'BLOCK_K'}
        s2 = self.cell_data.keys()
        if not s1.issubset(s2):
            msg = "'BLOCK_I', 'BLOCK_J' and 'BLOCK_K' cell arrays are required"
            raise TypeError(msg)
        alg = _vtk.vtkUnstructuredGridToExplicitStructuredGrid()
        alg.SetInputData(self)
        alg.SetInputArrayToProcess(0, 0, 0, 1, 'BLOCK_I')
        alg.SetInputArrayToProcess(1, 0, 0, 1, 'BLOCK_J')
        alg.SetInputArrayToProcess(2, 0, 0, 1, 'BLOCK_K')
        alg.Update()
        grid = _get_output(alg)
        grid.cell_data.remove('ConnectivityFlags')  # unrequired
        return grid


class StructuredGrid(PointGrid, StructuredGridFilters, _vtk.vtkStructuredGrid):
    """Dataset used for topologically regular arrays of data.

    Can be initialized in one of the following several ways:

    * Create empty grid.
    * Initialize from a filename.
    * Initialize from a :vtk:`vtkStructuredGrid` object.
    * Initialize directly from one or more :class:`numpy.ndarray`. See the
      example or the documentation of ``uinput``.

    Parameters
    ----------
    uinput : str, Path, :vtk:`vtkStructuredGrid`, numpy.ndarray, optional
        Filename, dataset, or array to initialize the structured grid from. If
        a filename is passed, pyvista will attempt to load it as a
        :class:`StructuredGrid`. If passed a :vtk:`vtkStructuredGrid`, it will
        be wrapped as a deep copy.

        If a :class:`numpy.ndarray` is provided and ``y`` and ``z`` are empty,
        this array will define the points of this :class:`StructuredGrid`.
        Set the dimensions with :attr:`StructuredGrid.dimensions`.

        Otherwise, this parameter will be loaded as the ``x`` points, and ``y``
        and ``z`` points must be set. The shape of this array defines the shape
        of the structured data and the shape should be ``(dimx, dimy,
        dimz)``. Missing trailing dimensions are assumed to be ``1``.

    y : numpy.ndarray, optional
        Coordinates of the points in y direction. If this is passed, ``uinput``
        must be a :class:`numpy.ndarray` and match the shape of ``y``.

    z : numpy.ndarray, optional
        Coordinates of the points in z direction. If this is passed, ``uinput``
        and ``y`` must be a :class:`numpy.ndarray` and match the shape of ``z``.

    deep : optional
        Whether to deep copy a StructuredGrid object.
        Default is ``False``.  Keyword only.

    **kwargs : dict, optional
        Additional keyword arguments passed when reading from a file or loading
        from arrays.

    See Also
    --------
    :ref:`create_structured_surface_example`

    Examples
    --------
    >>> import pyvista as pv
    >>> import vtk
    >>> import numpy as np

    Create an empty structured grid.

    >>> grid = pv.StructuredGrid()

    Initialize from a :vtk:`vtkStructuredGrid` object

    >>> vtkgrid = vtk.vtkStructuredGrid()
    >>> grid = pv.StructuredGrid(vtkgrid)

    Create from NumPy arrays using :func:`numpy.meshgrid`.

    >>> xrng = np.linspace(-5, 5, 10)
    >>> yrng = np.linspace(-8, 8, 4)
    >>> zrng = np.linspace(-7, 4, 20)
    >>> x, y, z = np.meshgrid(xrng, yrng, zrng, indexing='ij')
    >>> grid = pv.StructuredGrid(x, y, z)
    >>> grid
    StructuredGrid (...)
      N Cells:      513
      N Points:     800
      X Bounds:     -5.000e+00, 5.000e+00
      Y Bounds:     -8.000e+00, 8.000e+00
      Z Bounds:     -7.000e+00, 4.000e+00
      Dimensions:   10, 4, 20
      N Arrays:     0

    Note how the grid dimensions match the shape of the input arrays.

    >>> (xrng.size, yrng.size, zrng.size)
    (10, 4, 20)

    """

    _WRITERS: ClassVar[dict[str, type[StructuredGridWriter | XMLStructuredGridWriter]]] = {
        '.vtk': StructuredGridWriter,
        '.vts': XMLStructuredGridWriter,
    }  # type: ignore[assignment]

    def __init__(self, uinput=None, y=None, z=None, *args, deep: bool = False, **kwargs) -> None:
        """Initialize the structured grid."""
        super().__init__()

        if args:
            msg = 'Too many args to create StructuredGrid.'
            raise ValueError(msg)

        if isinstance(uinput, _vtk.vtkStructuredGrid):
            if deep:
                self.deep_copy(uinput)
            else:
                self.shallow_copy(uinput)  # type: ignore[arg-type]
        elif isinstance(uinput, (str, Path)):
            self._from_file(uinput, **kwargs)
        elif (
            isinstance(uinput, np.ndarray)
            and isinstance(y, np.ndarray)
            and isinstance(z, np.ndarray)
        ):
            self._from_arrays(uinput, y, z, **kwargs)
        elif isinstance(uinput, np.ndarray) and y is None and z is None:
            self.points = uinput
        elif uinput is None:
            # do nothing, initialize as empty structured grid
            pass
        else:
            msg = (
                'Invalid parameters. Expecting one of the following:\n'
                ' - No arguments\n'
                ' - Filename as the only argument\n'
                ' - StructuredGrid as the only argument\n'
                ' - Single `numpy.ndarray` as the only argument'
                ' - Three `numpy.ndarray` as the first three arguments'
            )
            raise TypeError(msg)

    def __repr__(self):
        """Return the standard representation."""
        return DataSet.__repr__(self)

    def __str__(self):
        """Return the standard str representation."""
        return DataSet.__str__(self)

    def _from_arrays(self, x, y, z, *, force_float: bool = True):
        """Create VTK structured grid directly from numpy arrays.

        Parameters
        ----------
        x : numpy.ndarray
            Position of the points in x direction.

        y : numpy.ndarray
            Position of the points in y direction.

        z : numpy.ndarray
            Position of the points in z direction.

        force_float : bool, optional
            Casts the datatype to ``float32`` if points datatype is
            non-float.  Default ``True``. Set this to ``False`` to allow
            non-float types, though this may lead to truncation of
            intermediate floats when transforming datasets.

        """
        if not (x.shape == y.shape == z.shape):
            msg = 'Input point array shapes must match exactly'
            raise ValueError(msg)

        # make the output points the same precision as the input arrays
        points = np.empty((x.size, 3), x.dtype)
        points[:, 0] = x.ravel('F')
        points[:, 1] = y.ravel('F')
        points[:, 2] = z.ravel('F')

        # ensure that the inputs are 3D
        dim = list(x.shape)
        while len(dim) < 3:
            dim.append(1)

        # Create structured grid
        self.SetDimensions(dim)
        self.SetPoints(vtk_points(points, force_float=force_float))

    @property
    def dimensions(self):  # numpydoc ignore=RT01
        """Return a length 3 tuple of the grid's dimensions.

        Returns
        -------
        tuple
            Grid dimensions.

        Examples
        --------
        >>> import pyvista as pv
        >>> import numpy as np
        >>> xrng = np.arange(-10, 10, 1, dtype=np.float32)
        >>> yrng = np.arange(-10, 10, 2, dtype=np.float32)
        >>> zrng = np.arange(-10, 10, 5, dtype=np.float32)
        >>> x, y, z = np.meshgrid(xrng, yrng, zrng, indexing='ij')
        >>> grid = pv.StructuredGrid(x, y, z)
        >>> grid.dimensions
        (20, 10, 4)

        """
        dims = [0, 0, 0]
        self.GetDimensions(dims)
        return tuple(dims)

    @dimensions.setter
    def dimensions(self, dims) -> None:
        nx, ny, nz = dims[0], dims[1], dims[2]
        self.SetDimensions(nx, ny, nz)
        self.Modified()

    @property
    def x(self):  # numpydoc ignore=RT01
        """Return the X coordinates of all points.

        Returns
        -------
        numpy.ndarray
            Numpy array of all X coordinates.

        Examples
        --------
        >>> import pyvista as pv
        >>> import numpy as np
        >>> xrng = np.arange(-10, 10, 1, dtype=np.float32)
        >>> yrng = np.arange(-10, 10, 2, dtype=np.float32)
        >>> zrng = np.arange(-10, 10, 5, dtype=np.float32)
        >>> x, y, z = np.meshgrid(xrng, yrng, zrng, indexing='ij')
        >>> grid = pv.StructuredGrid(x, y, z)
        >>> grid.x.shape
        (20, 10, 4)

        """
        return self._reshape_point_array(self.points[:, 0])

    @property
    def y(self):  # numpydoc ignore=RT01
        """Return the Y coordinates of all points."""
        return self._reshape_point_array(self.points[:, 1])

    @property
    def z(self):  # numpydoc ignore=RT01
        """Return the Z coordinates of all points."""
        return self._reshape_point_array(self.points[:, 2])

    @property
    def points_matrix(self):  # numpydoc ignore=RT01
        """Points as a 4-D matrix, with x/y/z along the last dimension."""
        return self.points.reshape((*self.dimensions, 3), order='F')

    def _get_attrs(self):
        """Return the representation methods (internal helper)."""
        attrs = PointGrid._get_attrs(self)
        attrs.append(('Dimensions', self.dimensions, '{:d}, {:d}, {:d}'))
        return attrs

    def __getitem__(self, key):
        """Slice subsets of the StructuredGrid, or extract an array field."""
        # legacy behavior which looks for a point or cell array
        if not isinstance(key, tuple):
            return super().__getitem__(key)

        # convert slice to VOI specification - only "basic indexing" is supported
        voi = []  # type: ignore[var-annotated]
        rate = []
        if len(key) != 3:
            msg = 'Slices must have exactly 3 dimensions.'
            raise RuntimeError(msg)
        for i, k in enumerate(key):
            if isinstance(k, Iterable):
                msg = 'Fancy indexing with iterable is not supported.'
                raise TypeError(msg)
            if isinstance(k, numbers.Integral):
                start = stop = k
                step = 1
            elif isinstance(k, slice):
                start = k.start if k.start is not None else 0  # type: ignore[assignment]
                stop = k.stop - 1 if k.stop is not None else self.dimensions[i]
                step = k.step if k.step is not None else 1
            voi.extend((start, stop))
            rate.append(step)

        return self.extract_subset(voi, rate, boundary=False)

    @_deprecate_positional_args(allowed=['ind'])
    def hide_cells(self, ind, inplace: bool = False) -> Self:  # noqa: FBT001, FBT002
        """Hide cells without deleting them.

        Hides cells by setting the ghost_cells array to ``HIDDEN_CELL``.

        Parameters
        ----------
        ind : sequence[int]
            List or array of cell indices to be hidden.  The array can
            also be a boolean array of the same size as the number of
            cells.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        pyvista.StructuredGrid
            Structured grid with hidden cells.

        Examples
        --------
        Hide part of the middle of a structured surface.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> x = np.arange(-10, 10, 0.25)
        >>> y = np.arange(-10, 10, 0.25)
        >>> z = 0
        >>> x, y, z = np.meshgrid(x, y, z)
        >>> grid = pv.StructuredGrid(x, y, z)
        >>> grid = grid.hide_cells(range(79 * 30, 79 * 50))
        >>> grid.plot(color=True, show_edges=True)

        """
        if not inplace:
            return self.copy().hide_cells(ind, inplace=True)
        if isinstance(ind, np.ndarray):
            if ind.dtype == np.bool_ and ind.size != self.n_cells:
                msg = f'Boolean array size must match the number of cells ({self.n_cells})'
                raise ValueError(msg)
        ghost_cells = np.zeros(self.n_cells, np.uint8)
        ghost_cells[ind] = _vtk.vtkDataSetAttributes.HIDDENCELL

        # NOTE: cells cannot be removed from a structured grid, only
        # hidden setting ghost_cells to a value besides
        # vtk.vtkDataSetAttributes.HIDDENCELL will not hide them
        # properly, additionally, calling self.RemoveGhostCells will
        # have no effect

        # add but do not make active
        self.cell_data.set_array(ghost_cells, _vtk.vtkDataSetAttributes.GhostArrayName())  # type: ignore[arg-type]
        return self

    def hide_points(self, ind: VectorLike[bool] | VectorLike[int]) -> None:
        """Hide points without deleting them.

        Hides points by setting the ghost_points array to ``HIDDEN_CELL``.

        Parameters
        ----------
        ind : VectorLike[bool] | VectorLike[int]
            Vector of point indices to be hidden. The vector can also be a
            boolean array of the same size as the number of points.

        Examples
        --------
        Hide part of the middle of a structured surface.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> x = np.arange(-10, 10, 0.25)
        >>> y = np.arange(-10, 10, 0.25)
        >>> z = 0
        >>> x, y, z = np.meshgrid(x, y, z)
        >>> grid = pv.StructuredGrid(x, y, z)
        >>> grid.hide_points(range(80 * 30, 80 * 50))
        >>> grid.plot(color=True, show_edges=True)

        """
        if isinstance(ind, np.ndarray):
            if ind.dtype == np.bool_ and ind.size != self.n_points:
                msg = f'Boolean array size must match the number of points ({self.n_points})'
                raise ValueError(msg)
        ghost_points = np.zeros(self.n_points, np.uint8)
        ghost_points[ind] = _vtk.vtkDataSetAttributes.HIDDENPOINT

        # add but do not make active
        self.point_data.set_array(ghost_points, _vtk.vtkDataSetAttributes.GhostArrayName())  # type: ignore[arg-type]

    def cast_to_explicit_structured_grid(self) -> ExplicitStructuredGrid:
        """Cast to an explicit structured grid.

        Returns
        -------
        pyvista.ExplicitStructuredGrid
            An explicit structured grid.

        Raises
        ------
        TypeError
            If the structured grid is not 3D (i.e., any dimension is 1).

        """
        if any(n == 1 for n in self.dimensions):
            msg = 'Only 3D structured grid can be casted to an explicit structured grid.'
            raise TypeError(msg)

        ni, nj, nk = self.dimensions
        grid = self.cast_to_unstructured_grid()

        s1 = {'BLOCK_I', 'BLOCK_J', 'BLOCK_K'}
        if not s1.issubset(self.cell_data):
            i, j, k = np.unravel_index(
                np.arange(self.n_cells),
                shape=(ni - 1, nj - 1, nk - 1),
                order='F',
            )
            grid.cell_data['BLOCK_I'] = i
            grid.cell_data['BLOCK_J'] = j
            grid.cell_data['BLOCK_K'] = k

        grid = grid.cast_to_explicit_structured_grid()

        if not s1.issubset(self.cell_data):
            for key in s1:
                grid.cell_data.pop(key, None)

        return grid

    def _reshape_point_array(self, array: NumpyArray[float]) -> NumpyArray[float]:
        """Reshape point data to a 3-D matrix."""
        return array.reshape(self.dimensions, order='F')

    def _reshape_cell_array(self, array: NumpyArray[float]) -> NumpyArray[float]:
        """Reshape cell data to a 3-D matrix."""
        cell_dims = np.array(self.dimensions) - 1
        cell_dims[cell_dims == 0] = 1
        return array.reshape(cell_dims, order='F')


class ExplicitStructuredGrid(PointGrid, _vtk.vtkExplicitStructuredGrid):
    """Extend the functionality of the :vtk:`vtkExplicitStructuredGrid` class.

    Can be initialized by the following:

    - Creating an empty grid
    - From a :vtk:`vtkStructuredGrid`, :vtk:`vtkExplicitStructuredGrid` or
      :vtk:`vtkUnstructuredGrid` object
    - From a VTU or VTK file
    - From ``dims`` and ``corners`` arrays
    - From ``dims``, ``cells`` and ``points`` arrays

    Parameters
    ----------
    args : :vtk:`vtkExplicitStructuredGrid`, :vtk:`vtkUnstructuredGrid`, str, Sequence
        See examples below.
    deep : bool, default: False
        Whether to deep copy a :vtk:`vtkUnstructuredGrid` object.

    See Also
    --------
    :ref:`create_explicit_structured_grid_example`

    Examples
    --------
    >>> import numpy as np
    >>> import pyvista as pv
    >>>
    >>> # grid size: ni*nj*nk cells; si, sj, sk steps
    >>> ni, nj, nk = 4, 5, 6
    >>> si, sj, sk = 20, 10, 1
    >>>
    >>> # create raw coordinate grid
    >>> grid_ijk = np.mgrid[
    ...     : (ni + 1) * si : si,
    ...     : (nj + 1) * sj : sj,
    ...     : (nk + 1) * sk : sk,
    ... ]
    >>>
    >>> # repeat array along each Cartesian axis for connectivity
    >>> for axis in range(1, 4):
    ...     grid_ijk = grid_ijk.repeat(2, axis=axis)
    >>>
    >>> # slice off unnecessarily doubled edge coordinates
    >>> grid_ijk = grid_ijk[:, 1:-1, 1:-1, 1:-1]
    >>>
    >>> # reorder and reshape to VTK order
    >>> corners = grid_ijk.transpose().reshape(-1, 3)
    >>>
    >>> dims = np.array([ni, nj, nk]) + 1
    >>> grid = pv.ExplicitStructuredGrid(dims, corners)
    >>> grid = grid.compute_connectivity()
    >>> grid.plot(show_edges=True)

    """

    _WRITERS: ClassVar[dict[str, type[BaseWriter]]] = {
        '.vtu': XMLUnstructuredGridWriter,
        '.vtk': UnstructuredGridWriter,
    }

    def __init__(self, *args, deep: bool = False, **kwargs):  # noqa: ARG002
        """Initialize the explicit structured grid."""
        super().__init__()
        n = len(args)
        if n > 3:
            msg = 'Too many args to create ExplicitStructuredGrid.'
            raise ValueError(msg)
        if n == 1:
            arg0 = args[0]
            if isinstance(arg0, _vtk.vtkExplicitStructuredGrid):
                if deep:
                    self.deep_copy(arg0)
                else:
                    self.shallow_copy(arg0)  # type: ignore[arg-type]
            elif isinstance(arg0, (_vtk.vtkStructuredGrid, _vtk.vtkUnstructuredGrid)):
                grid = arg0.cast_to_explicit_structured_grid()  # type: ignore[union-attr]
                self.shallow_copy(grid)
            elif isinstance(arg0, (str, Path)):
                grid = UnstructuredGrid(arg0)
                grid = grid.cast_to_explicit_structured_grid()
                self.shallow_copy(grid)
        elif n == 2:
            arg0, arg1 = args
            if isinstance(arg0, tuple):
                arg0 = np.asarray(arg0)
            if isinstance(arg1, list):
                arg1 = np.asarray(arg1)
            arg0_is_arr = isinstance(arg0, np.ndarray)
            arg1_is_arr = isinstance(arg1, np.ndarray)
            if all([arg0_is_arr, arg1_is_arr]):
                self._from_arrays(arg0, arg1)
        elif n == 3:
            arg0, arg1, arg2 = args
            arg0 = np.asarray(arg0)
            arg1 = np.asarray(arg1) if not isinstance(arg1, dict) else arg1
            arg2 = np.asarray(arg2)
            self._from_cells_points(arg0, arg1, arg2)

    def __repr__(self) -> str:
        """Return the standard representation."""
        return DataSet.__repr__(self)

    def __str__(self) -> str:
        """Return the standard ``str`` representation."""
        return DataSet.__str__(self)

    def _from_arrays(self, dims: VectorLike[int], corners: MatrixLike[float]) -> None:
        """Create a VTK explicit structured grid from NumPy arrays.

        Parameters
        ----------
        dims : VectorLike[int]
            A sequence of integers with shape (3,) containing the
            topological dimensions of the grid.

        corners : MatrixLike[float]
            A sequence of numbers with shape ``(number of corners, 3)``
            containing the coordinates of the corner points.

        """
        if len(dims) != 3:
            msg = 'Expected dimensions to be length 3.'
            raise ValueError(msg)

        ni, nj, nk = np.asanyarray(dims) - 1
        corners = np.reshape(corners, (2 * ni, 2 * nj, 2 * nk, 3), order='F')
        points = np.column_stack(
            [
                np.column_stack(
                    (
                        corners_[::2, ::2, ::2].ravel(order='F'),
                        corners_[1::2, ::2, ::2].ravel(order='F'),
                        corners_[1::2, 1::2, ::2].ravel(order='F'),
                        corners_[::2, 1::2, ::2].ravel(order='F'),
                        corners_[::2, ::2, 1::2].ravel(order='F'),
                        corners_[1::2, ::2, 1::2].ravel(order='F'),
                        corners_[1::2, 1::2, 1::2].ravel(order='F'),
                        corners_[::2, 1::2, 1::2].ravel(order='F'),
                    )
                ).ravel()
                for corners_ in corners.transpose((3, 0, 1, 2))
            ]
        )
        cells = np.arange(8 * ni * nj * nk).reshape((ni * nj * nk, 8))
        self._from_cells_points(dims, {CellType.HEXAHEDRON: cells}, points)

    def _from_cells_points(
        self,
        dims: VectorLike[int],
        cells: VectorLike[int] | dict[int, MatrixLike[int]],
        points: MatrixLike[float],
    ) -> None:
        """Create a VTK explicit structured grid from cells and points arrays.

        Parameters
        ----------
        dims : VectorLike[int]
            A sequence of integers with shape (3,) containing the
            topological dimensions of the grid.

        cells : VectorLike[int] | dict[int, MatrixLike[int]]
            Array of cells.  Each cell contains the number of points in the
            cell and the node numbers of the cell.

        points : MatrixLike[float]
            Numpy array containing point locations.

        """
        if len(dims) != 3:
            msg = 'Expected dimensions to be length 3.'
            raise ValueError(msg)

        else:
            n_cells = np.prod([n - 1 for n in dims])  # type: ignore[arg-type]

        if isinstance(cells, dict):
            celltypes = list(cells)

            if not (len(celltypes) == 1 and celltypes[0] == CellType.HEXAHEDRON):
                msg = f'Expected cells to be a single cell of type {CellType.HEXAHEDRON}.'
                raise ValueError(msg)

            cells = np.asarray(cells[celltypes[0]])
            if cells.shape != (n_cells, 8):
                msg = f'Expected cells to be of shape ({n_cells}, 8)'
                raise ValueError(msg)

            cells = np.column_stack((np.full(n_cells, 8), cells)).flatten()

        elif len(cells) != 9 * n_cells:
            msg = f'Expected cells to be length {9 * n_cells}'
            raise ValueError(msg)

        self.SetDimensions(dims[0], dims[1], dims[2])  # type: ignore[arg-type]
        self.SetCells(CellArray(cells))
        self.SetPoints(vtk_points(points))

    def cast_to_unstructured_grid(self) -> UnstructuredGrid:
        """Cast to an unstructured grid.

        Returns
        -------
        UnstructuredGrid
            An unstructured grid. VTK adds the ``'BLOCK_I'``,
            ``'BLOCK_J'`` and ``'BLOCK_K'`` cell arrays. These arrays
            are required to restore the explicit structured grid.

        See Also
        --------
        pyvista.DataSetFilters.extract_cells : Extract a subset of a dataset.
        pyvista.UnstructuredGrid.cast_to_explicit_structured_grid
            Cast an unstructured grid to an explicit structured grid.

        Notes
        -----
        The ghost cell array is disabled before casting the
        unstructured grid in order to allow the original structure
        and attributes data of the explicit structured grid to be
        restored. If you don't need to restore the explicit
        structured grid later or want to extract an unstructured
        grid from the visible subgrid, use the ``extract_cells``
        filter and the cell indices where the ghost cell array is
        ``0``.

        Examples
        --------
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)

        >>> grid = grid.hide_cells(range(80, 120))
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)

        >>> grid = grid.cast_to_unstructured_grid()
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)

        >>> grid = grid.cast_to_explicit_structured_grid()
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)

        """
        grid = ExplicitStructuredGrid()
        grid.copy_structure(self)
        alg = _vtk.vtkExplicitStructuredGridToUnstructuredGrid()
        alg.SetInputDataObject(grid)
        alg.Update()
        ugrid = _get_output(alg)
        ugrid.cell_data.remove('vtkOriginalCellIds')  # unrequired
        ugrid.copy_attributes(self)  # copy ghost cell array and other arrays
        return ugrid

    @_deprecate_positional_args
    def clean(  # noqa: PLR0917
        self,
        tolerance=0,
        remove_unused_points: bool = True,  # noqa: FBT001, FBT002
        produce_merge_map: bool = True,  # noqa: FBT001, FBT002
        average_point_data: bool = True,  # noqa: FBT001, FBT002
        merging_array_name=None,
        progress_bar: bool = False,  # noqa: FBT001, FBT002
    ) -> ExplicitStructuredGrid:
        """Merge duplicate points and remove unused points in an ExplicitStructuredGrid.

        This filter, merging coincident points as defined by a merging
        tolerance and optionally removes unused points. The filter does not
        modify the topology of the input dataset, nor change the types of
        cells. It may however, renumber the cell connectivity ids.

        This filter casts the grid to an UnstructuredGrid to clean it, then
        casts the cleaned unstructured grid to an explicit structured grid.

        Parameters
        ----------
        tolerance : float, default: 0.0
            The absolute point merging tolerance.

        remove_unused_points : bool, default: True
            Indicate whether points unused by any cell are removed from the
            output. Note that when this is off, the filter can successfully
            process datasets with no cells (and just points). If on in this
            case, and there are no cells, the output will be empty.

        produce_merge_map : bool, default: False
            Indicate whether a merge map should be produced on output.
            The merge map, if requested, maps each input point to its
            output point id, or provides a value of -1 if the input point
            is not used in the output. The merge map is associated with
            the filter's output field data and is named ``"PointMergeMap"``.

        average_point_data : bool, default: True
            Indicate whether point coordinates and point data of merged points
            are averaged. When ``True``, the data coordinates and attribute
            values of all merged points are averaged. When ``False``, the point
            coordinate and data of the single remaining merged point is
            retained.

        merging_array_name : str, optional
            If a ``merging_array_name`` is specified and exists in the
            ``point_data``, then point merging will switch into a mode where
            merged points must be both geometrically coincident and have
            matching point data. When set, ``tolerance`` has no effect.

        progress_bar : bool, default: False
            Display a progress bar to indicate progress.

        Returns
        -------
        ExplicitStructuredGrid
            Cleaned explicit structured grid.

        """
        grid = (
            self.cast_to_unstructured_grid()
            .clean(
                tolerance=tolerance,
                remove_unused_points=remove_unused_points,
                produce_merge_map=produce_merge_map,
                average_point_data=average_point_data,
                merging_array_name=merging_array_name,
                progress_bar=progress_bar,
            )
            .cast_to_explicit_structured_grid()
        )

        s1 = {'BLOCK_I', 'BLOCK_J', 'BLOCK_K'}
        if not s1.issubset(self.cell_data):
            for key in s1:
                grid.cell_data.pop(key, None)

        return grid

    @_deprecate_positional_args(allowed=['filename'])
    def save(  # noqa: PLR0917
        self,
        filename: Path | str,
        binary: bool = True,  # noqa: FBT001, FBT002
        texture: NumpyArray[np.uint8] | str | None = None,
        compression: _CompressionOptions = 'zlib',
    ) -> None:
        """Save this VTK object to file.

        Parameters
        ----------
        filename : Path, str
            Output file name. VTU and VTK extensions are supported.

        binary : bool, default: True
            If ``True``, write as binary, else ASCII.

        texture : np.ndarray, str, None
            Ignored argument. Kept to maintain compatibility with supertype.

        compression : str or None, default: 'zlib'
            The compression type to use when ``binary`` is ``True``
            and VTK writer is of type :vtk:`vtkXMLWriter`. This
            argument has no effect otherwise. Acceptable values are
            ``'zlib'``, ``'lz4'``, ``'lzma'``, and ``None``. ``None``
            indicates no compression.

            .. versionadded:: 0.47

        Notes
        -----
        VTK adds the ``'BLOCK_I'``, ``'BLOCK_J'`` and ``'BLOCK_K'``
        cell arrays. These arrays are required to restore the explicit
        structured grid.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()  # doctest:+SKIP
        >>> grid = grid.hide_cells(range(80, 120))  # doctest:+SKIP
        >>> grid.save('grid.vtu')  # doctest:+SKIP

        >>> grid = pv.ExplicitStructuredGrid('grid.vtu')  # doctest:+SKIP
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)  # doctest:+SKIP

        >>> grid.show_cells()  # doctest:+SKIP
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)  # doctest:+SKIP

        """
        if texture is not None:
            msg = 'Cannot save texture of a pointset.'
            raise ValueError(msg)
        grid = self.cast_to_unstructured_grid()
        grid.save(filename, binary=binary, compression=compression)

    @_deprecate_positional_args(allowed=['ind'])
    def hide_cells(self, ind: VectorLike[int], inplace: bool = False) -> Self:  # noqa: FBT001, FBT002
        """Hide specific cells.

        Hides cells by setting the ghost cell array to ``HIDDENCELL``.

        Parameters
        ----------
        ind : sequence[int]
            Cell indices to be hidden. A boolean array of the same
            size as the number of cells also is acceptable.

        inplace : bool, default: False
            This method is applied to this grid if ``True``
            or to a copy otherwise.

        Returns
        -------
        output : ExplicitStructuredGrid | None
            A deep copy of this grid if ``inplace=False`` with the
            hidden cells, or this grid with the hidden cells if
            otherwise.

        Examples
        --------
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()
        >>> grid = grid.hide_cells(range(80, 120))
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)

        """
        ind_arr = np.asanyarray(ind)

        if inplace:
            array = np.zeros(self.n_cells, dtype=np.uint8)
            array[ind_arr] = _vtk.vtkDataSetAttributes.HIDDENCELL
            name = _vtk.vtkDataSetAttributes.GhostArrayName()
            self.cell_data[name] = array
            return self

        grid = self.copy()
        grid.hide_cells(ind, inplace=True)
        return grid

    @_deprecate_positional_args
    def show_cells(self, inplace: bool = False) -> Self:  # noqa: FBT001, FBT002
        """Show hidden cells.

        Shows hidden cells by setting the ghost cell array to ``0``
        where ``HIDDENCELL``.

        Parameters
        ----------
        inplace : bool, default: False
            This method is applied to this grid if ``True``
            or to a copy otherwise.

        Returns
        -------
        ExplicitStructuredGrid
            A deep copy of this grid if ``inplace=False`` with the
            hidden cells shown.  Otherwise, this dataset with the
            shown cells.

        Examples
        --------
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()
        >>> grid = grid.hide_cells(range(80, 120))
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)

        >>> grid = grid.show_cells()
        >>> grid.plot(color='w', show_edges=True, show_bounds=True)

        """
        if inplace:
            name = _vtk.vtkDataSetAttributes.GhostArrayName()
            if name in self.cell_data.keys():
                array = self.cell_data[name]
                ind = np.argwhere(array == _vtk.vtkDataSetAttributes.HIDDENCELL)
                array[ind] = 0
            return self
        else:
            grid = self.copy()
            grid.show_cells(inplace=True)
            return grid

    def _dimensions(self) -> tuple[int, int, int]:
        # This method is required to avoid conflict if a developer extends `ExplicitStructuredGrid`
        # and reimplements `dimensions` to return, for example, the number of cells in the I, J and
        dims = np.reshape(self.GetExtent(), (3, 2))  # K directions.
        dims = np.diff(dims, axis=1)
        dims = dims.flatten() + 1  # type: ignore[assignment]
        return int(dims[0]), int(dims[1]), int(dims[2])

    @property
    def dimensions(self) -> tuple[int, int, int]:  # numpydoc ignore=RT01
        """Return the topological dimensions of the grid.

        Returns
        -------
        tuple[int, int, int]
            Number of sampling points in the I, J and Z directions respectively.

        Examples
        --------
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()
        >>> grid.dimensions
        (5, 6, 7)

        """
        return self._dimensions()

    @property
    def visible_bounds(self) -> BoundsTuple:  # numpydoc ignore=RT01
        """Return the bounding box of the visible cells.

        Different from `bounds`, which returns the bounding box of the
        complete grid, this method returns the bounding box of the
        visible cells, where the ghost cell array is not
        ``HIDDENCELL``.

        Returns
        -------
        tuple[float, float, float]
            The limits of the visible grid in the X, Y and Z
            directions respectively.

        Examples
        --------
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()
        >>> grid = grid.hide_cells(range(80, 120))
        >>> grid.bounds
        BoundsTuple(x_min =  0.0,
                    x_max = 80.0,
                    y_min =  0.0,
                    y_max = 50.0,
                    z_min =  0.0,
                    z_max =  6.0)

        >>> grid.visible_bounds
        BoundsTuple(x_min =  0.0,
                    x_max = 80.0,
                    y_min =  0.0,
                    y_max = 50.0,
                    z_min =  0.0,
                    z_max =  4.0)

        """
        name = _vtk.vtkDataSetAttributes.GhostArrayName()
        if name in self.cell_data:
            array = self.cell_data[name]
            grid = self.extract_cells(array == 0)
            return grid.bounds
        else:
            return self.bounds

    def cell_id(self, coords: ArrayLike[int]) -> int | NumpyArray[int] | None:
        """Return the cell ID.

        Parameters
        ----------
        coords : ArrayLike[int]
            Cell structured coordinates.

        Returns
        -------
        output : int | numpy.ndarray | None
            Cell IDs. ``None`` if ``coords`` is outside the grid extent.

        See Also
        --------
        pyvista.ExplicitStructuredGrid.cell_coords : Return the cell structured coordinates.

        Examples
        --------
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()
        >>> grid.cell_id((3, 4, 0))
        np.int64(19)

        >>> coords = [(3, 4, 0), (3, 2, 1), (1, 0, 2), (2, 3, 2)]
        >>> grid.cell_id(coords)
        array([19, 31, 41, 54])

        """
        # `vtk.vtkExplicitStructuredGrid.ComputeCellId` is not used
        # here because this method returns invalid cell IDs when
        # `coords` is outside the grid extent.
        if isinstance(coords, Sequence):
            coords = np.asarray(coords)

        if coords.ndim == 2:
            ncol = coords.shape[1]
            coords = [coords[:, c] for c in range(ncol)]
            coords = tuple(coords)
        dims = self._dimensions()
        try:
            ind = np.ravel_multi_index(coords, np.array(dims) - 1, order='F')
        except ValueError:
            return None
        else:
            return ind

    def cell_coords(
        self,
        ind: int | VectorLike[int],
    ) -> None | MatrixLike[int]:
        """Return the cell structured coordinates.

        Parameters
        ----------
        ind : int | VectorLike[int]
            Cell IDs.

        Returns
        -------
        output : numpy.ndarray | None
            Cell structured coordinates. ``None`` if ``ind`` is
            outside the grid extent.

        See Also
        --------
        pyvista.ExplicitStructuredGrid.cell_id : Return the cell ID.

        Examples
        --------
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()
        >>> grid.cell_coords(19)
        array([3, 4, 0])

        >>> grid.cell_coords((19, 31, 41, 54))
        array([[3, 4, 0],
               [3, 2, 1],
               [1, 0, 2],
               [2, 3, 2]])

        """
        dims = self._dimensions()
        try:
            coords = np.unravel_index(ind, np.array(dims) - 1, order='F')
        except ValueError:
            return None
        else:
            if isinstance(coords[0], np.ndarray):
                return np.stack(coords, axis=1)
            return np.asanyarray(coords)  # type: ignore[unreachable]

    def neighbors(self, ind: int | VectorLike[int], rel: str = 'connectivity') -> list[int]:
        """Return the indices of neighboring cells.

        Parameters
        ----------
        ind : int | VectorLike[int]
            Cell IDs.

        rel : str, default: "connectivity"
            Defines the neighborhood relationship. If
            ``'topological'``, returns the ``(i-1, j, k)``, ``(i+1, j,
            k)``, ``(i, j-1, k)``, ``(i, j+1, k)``, ``(i, j, k-1)``
            and ``(i, j, k+1)`` cells. If ``'connectivity'``
            (default), returns only the topological neighbors
            considering faces connectivity. If ``'geometric'``,
            returns the cells in the ``(i-1, j)``, ``(i+1, j)``,
            ``(i,j-1)`` and ``(i, j+1)`` vertical cell groups whose
            faces intersect.

        Returns
        -------
        list[int]
            Indices of neighboring cells.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()
        >>> cell = grid.extract_cells(31)
        >>> ind = grid.neighbors(31)
        >>> neighbors = grid.extract_cells(ind)
        >>> pl = pv.Plotter()
        >>> _ = pl.add_axes()
        >>> _ = pl.add_mesh(cell, color='r', show_edges=True)
        >>> _ = pl.add_mesh(neighbors, color='w', show_edges=True)
        >>> pl.show()

        """

        def connectivity(ind):
            indices = []
            cell_coords = self.cell_coords(ind)
            cell_points = self.get_cell(ind).points
            if cell_points.shape[0] == 8:
                faces = [
                    [(-1, 0, 0), (0, 4, 7, 3), (1, 5, 6, 2)],
                    [(+1, 0, 0), (1, 2, 6, 5), (0, 3, 7, 4)],
                    [(0, -1, 0), (0, 1, 5, 4), (3, 2, 6, 7)],
                    [(0, +1, 0), (3, 7, 6, 2), (0, 4, 5, 1)],
                    [(0, 0, -1), (0, 3, 2, 1), (4, 7, 6, 5)],
                    [(0, 0, +1), (4, 5, 6, 7), (0, 1, 2, 3)],
                ]
                for f in faces:
                    coords = np.sum([cell_coords, f[0]], axis=0)  # type: ignore[arg-type]
                    ind = self.cell_id(coords)
                    if ind:
                        points = self.get_cell(ind).points
                        if points.shape[0] == 8:
                            a1 = cell_points[f[1], :]
                            a2 = points[f[2], :]
                            if np.array_equal(a1, a2):
                                indices.append(ind)
            return indices

        def topological(ind):
            indices = []
            cell_coords = self.cell_coords(ind)
            cell_neighbors = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
            for n in cell_neighbors:
                coords = np.sum([cell_coords, n], axis=0)  # type: ignore[arg-type]
                ind = self.cell_id(coords)
                if ind:
                    indices.append(ind)
            return indices

        def geometric(ind):
            indices = []
            cell_coords = self.cell_coords(ind)
            cell_points = self.get_cell(ind).points
            if cell_points.shape[0] == 8:
                for k in [-1, 1]:
                    coords = np.sum([cell_coords, (0, 0, k)], axis=0)  # type: ignore[arg-type]
                    ind = self.cell_id(coords)
                    if ind:
                        indices.append(ind)
                faces = [
                    [(-1, 0, 0), (0, 4, 3, 7), (1, 5, 2, 6)],
                    [(+1, 0, 0), (2, 6, 1, 5), (3, 7, 0, 4)],
                    [(0, -1, 0), (1, 5, 0, 4), (2, 6, 3, 7)],
                    [(0, +1, 0), (3, 7, 2, 6), (0, 4, 1, 5)],
                ]
                nk = self.dimensions[2]
                for f in faces:
                    cell_z = cell_points[f[1], 2]
                    cell_z = np.abs(cell_z)
                    cell_z = cell_z.reshape((2, 2))
                    cell_zmin = cell_z.min(axis=1)
                    cell_zmax = cell_z.max(axis=1)
                    coords = np.sum([cell_coords, f[0]], axis=0)  # type: ignore[arg-type]
                    for k in range(nk):
                        coords[2] = k
                        ind = self.cell_id(coords)
                        if ind:
                            points = self.get_cell(ind).points
                            if points.shape[0] == 8:
                                z = points[f[2], 2]
                                z = np.abs(z)
                                z = z.reshape((2, 2))
                                zmin = z.min(axis=1)
                                zmax = z.max(axis=1)
                                if (
                                    (zmax[0] > cell_zmin[0] and zmin[0] < cell_zmax[0])
                                    or (zmax[1] > cell_zmin[1] and zmin[1] < cell_zmax[1])
                                    or (zmin[0] > cell_zmax[0] and zmax[1] < cell_zmin[1])
                                    or (zmin[1] > cell_zmax[1] and zmax[0] < cell_zmin[0])
                                ):
                                    indices.append(ind)
            return indices

        if isinstance(ind, int):
            ind = [ind]

        rel_map = {
            'connectivity': connectivity,
            'geometric': geometric,
            'topological': topological,
        }

        if rel not in rel_map:
            msg = (
                f'Invalid value for `rel` of {rel}. Should be one of the '
                f'following\n{rel_map.keys()}'
            )
            raise ValueError(msg)
        rel_func = rel_map[rel]

        indices = set()
        for i in ind:
            indices.update(rel_func(i))
        return sorted(indices)

    @_deprecate_positional_args
    def compute_connectivity(self, inplace: bool = False) -> Self:  # noqa: FBT001, FBT002
        """Compute the faces connectivity flags array.

        This method checks the faces connectivity of the cells with
        their topological neighbors.  The result is stored in the
        array of integers ``'ConnectivityFlags'``. Each value in this
        array must be interpreted as a binary number, where the digits
        shows the faces connectivity of a cell with its topological
        neighbors -Z, +Z, -Y, +Y, -X and +X respectively. For example,
        a cell with ``'ConnectivityFlags'`` equal to ``27``
        (``011011``) indicates that this cell is connected by faces
        with their neighbors ``(0, 0, 1)``, ``(0, -1, 0)``,
        ``(-1, 0, 0)`` and ``(1, 0, 0)``.

        Parameters
        ----------
        inplace : bool, default: False
            This method is applied to this grid if ``True``
            or to a copy otherwise.

        Returns
        -------
        ExplicitStructuredGrid
            A deep copy of this grid if ``inplace=False``, or this
            DataSet if otherwise.

        See Also
        --------
        ExplicitStructuredGrid.compute_connections
            Compute an array with the number of connected cell faces.

        Examples
        --------
        >>> from pyvista import examples
        >>>
        >>> grid = examples.load_explicit_structured()
        >>> grid = grid.compute_connectivity()
        >>> grid.plot(show_edges=True)

        """
        if inplace:
            self.ComputeFacesConnectivityFlagsArray()
            return self
        else:
            grid = self.copy()
            grid.compute_connectivity(inplace=True)
            return grid

    @_deprecate_positional_args
    def compute_connections(self, inplace: bool = False) -> Self:  # noqa: FBT001, FBT002
        """Compute an array with the number of connected cell faces.

        This method calculates the number of topological cell
        neighbors connected by faces. The results are stored in the
        ``'number_of_connections'`` cell array.

        Parameters
        ----------
        inplace : bool, default: False
            This method is applied to this grid if ``True`` or to a copy
            otherwise.

        Returns
        -------
        ExplicitStructuredGrid
            A deep copy of this grid if ``inplace=False`` or this
            DataSet if otherwise.

        See Also
        --------
        ExplicitStructuredGrid.compute_connectivity : Compute the faces connectivity flags array.

        Examples
        --------
        >>> from pyvista import examples
        >>> grid = examples.load_explicit_structured()
        >>> grid = grid.compute_connections()
        >>> grid.plot(show_edges=True)

        """
        if inplace:
            if 'ConnectivityFlags' in self.cell_data:
                array = self.cell_data['ConnectivityFlags']
            else:
                grid = self.compute_connectivity(inplace=False)
                array = grid.cell_data['ConnectivityFlags']
            array = array.reshape((-1, 1))  # type: ignore[assignment]
            array = array.astype(np.uint8)  # type: ignore[assignment]
            array = np.unpackbits(array, axis=1)  # type: ignore[assignment]
            array = array.sum(axis=1)
            self.cell_data['number_of_connections'] = array
            return self
        else:
            return self.copy().compute_connections(inplace=True)
