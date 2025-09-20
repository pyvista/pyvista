"""Attributes common to PolyData and Grid Objects."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import cast
from typing import overload
import warnings

import numpy as np

import pyvista
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.typing.mypy_plugin import promote_type

from . import _vtk_core as _vtk
from ._typing_core import BoundsTuple
from .dataobject import DataObject
from .datasetattributes import DataSetAttributes
from .errors import PyVistaDeprecationWarning
from .filters import DataSetFilters
from .filters import _get_output
from .pyvista_ndarray import pyvista_ndarray
from .utilities.arrays import CellLiteral
from .utilities.arrays import FieldAssociation
from .utilities.arrays import FieldLiteral
from .utilities.arrays import PointLiteral
from .utilities.arrays import _coerce_pointslike_arg
from .utilities.arrays import get_array
from .utilities.arrays import get_array_association
from .utilities.arrays import raise_not_matching
from .utilities.arrays import vtk_id_list_to_array
from .utilities.helpers import is_pyvista_dataset
from .utilities.misc import _NoNewAttrMixin
from .utilities.misc import abstract_class
from .utilities.points import vtk_points

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator
    from collections.abc import Iterator

    from typing_extensions import Self

    from ._typing_core import MatrixLike
    from ._typing_core import NumpyArray
    from ._typing_core import VectorLike

# vector array names
DEFAULT_VECTOR_KEY = '_vectors'


class ActiveArrayInfoTuple(NamedTuple):
    """Active array info tuple.

    Parameters
    ----------
    association : pyvista.core.utilities.arrays.FieldAssociation
        Association of the array.

    name : str
        The name of the array.

    """

    association: FieldAssociation
    name: str | None

    def copy(self: ActiveArrayInfoTuple) -> ActiveArrayInfoTuple:
        """Return a copy of this object.

        Returns
        -------
        ActiveArrayInfo
            A copy of this object.

        """
        return ActiveArrayInfoTuple(self.association, self.name)


class _ActiveArrayExistsInfoTuple(NamedTuple):
    """Active array info tuple for arrays that exist.

    This named tuple is similar to ActiveArrayInfoTuple except the
    `name` attribute cannot be `None`.
    """

    association: FieldAssociation
    name: str


class ActiveArrayInfo(_NoNewAttrMixin):
    """Active array info class with support for pickling.

    .. deprecated:: 0.45

        Use :class:`pyvista.core.dataset.ActiveArrayInfoTuple` instead.

    Parameters
    ----------
    association : pyvista.core.utilities.arrays.FieldAssociation
        Array association.
        Association of the array.

    name : str
        The name of the array.

    """

    def __init__(self: ActiveArrayInfo, association: FieldAssociation, name: str | None) -> None:
        """Initialize."""
        self.association = association
        self.name = name
        # Deprecated on v0.45.0, estimated removal on v0.48.0
        warnings.warn(
            'ActiveArrayInfo is deprecated. Use ActiveArrayInfoTuple instead.',
            PyVistaDeprecationWarning,
            stacklevel=2,
        )

    def copy(self: ActiveArrayInfo) -> ActiveArrayInfo:
        """Return a copy of this object.

        Returns
        -------
        ActiveArrayInfo
            A copy of this object.

        """
        return ActiveArrayInfo(self.association, self.name)

    def __getstate__(self: ActiveArrayInfo) -> dict[str, Any]:
        """Support pickling."""
        state = self.__dict__.copy()
        state['association'] = int(self.association.value)
        return state

    def __setstate__(self: ActiveArrayInfo, state: dict[str, Any]) -> None:
        """Support unpickling."""
        self.__dict__ = state.copy()
        self.association = FieldAssociation(state['association'])

    @property
    def _namedtuple(self: ActiveArrayInfo) -> ActiveArrayInfoTuple:
        """Build a namedtuple on the fly to provide legacy support."""
        return ActiveArrayInfoTuple(self.association, self.name)

    def __iter__(self: ActiveArrayInfo) -> Iterator[FieldAssociation | str | None]:
        """Provide namedtuple-like __iter__."""
        return self._namedtuple.__iter__()

    def __repr__(self: ActiveArrayInfo) -> str:
        """Provide namedtuple-like __repr__."""
        return self._namedtuple.__repr__()

    def __getitem__(self: ActiveArrayInfo, item: int) -> FieldAssociation | str | None:
        """Provide namedtuple-like __getitem__."""
        return self._namedtuple.__getitem__(item)

    def __eq__(self: ActiveArrayInfo, other: object) -> bool:
        """Check equivalence (useful for serialize/deserialize tests)."""
        if isinstance(other, ActiveArrayInfo):
            same_association = int(self.association.value) == int(other.association.value)
            return self.name == other.name and same_association
        return False

    __hash__ = None  # type: ignore[assignment]  # https://github.com/pyvista/pyvista/pull/7671


@promote_type(_vtk.vtkDataSet)
@abstract_class
class DataSet(DataSetFilters, DataObject):
    """Methods in common to spatially referenced objects.

    Parameters
    ----------
    *args :
        Any extra args are passed as option to spatially referenced objects.

    **kwargs :
        Any extra keyword args are passed as option to spatially referenced objects.

    """

    plot = pyvista._plot.plot

    def __init__(self: Self, *args, **kwargs) -> None:
        """Initialize the common object."""
        super().__init__(*args, **kwargs)
        self._last_active_scalars_name: str | None = None
        self._active_scalars_info = ActiveArrayInfoTuple(FieldAssociation.POINT, name=None)
        self._active_vectors_info = ActiveArrayInfoTuple(FieldAssociation.POINT, name=None)
        self._active_tensors_info = ActiveArrayInfoTuple(FieldAssociation.POINT, name=None)

        # Used by glyph filter and plotter legend
        self._glyph_geom: Sequence[_vtk.vtkDataSet] | None = None

    def __getattr__(self: Self, item: str) -> Any:
        """Get attribute from base class if not found."""
        return super().__getattribute__(item)

    @property
    def active_scalars_info(self: Self) -> ActiveArrayInfoTuple:
        """Return the active scalar's association and name.

        Association refers to the data association (e.g. point, cell, or
        field) of the active scalars.

        Returns
        -------
        ActiveArrayInfo
            The scalars info in an object with namedtuple semantics,
            with attributes ``association`` and ``name``.

        Notes
        -----
        If both cell and point scalars are present and neither have
        been set active within at the dataset level, point scalars
        will be made active.

        Examples
        --------
        Create a mesh, add scalars to the mesh, and return the active
        scalars info.  Note how when the scalars are added, they
        automatically become the active scalars.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh['Z Height'] = mesh.points[:, 2]
        >>> mesh.active_scalars_info
        ActiveArrayInfoTuple(association=<FieldAssociation.POINT: 0>, name='Z Height')

        """
        field, name = self._active_scalars_info
        exclude = {'__custom_rgba', 'Normals', 'vtkOriginalPointIds', 'TCoords'}
        if name in exclude:
            name = self._last_active_scalars_name

        # verify this field is still valid
        if name is not None:
            if field is FieldAssociation.CELL:
                if self.cell_data.active_scalars_name != name:
                    name = None
            elif field is FieldAssociation.POINT:
                if self.point_data.active_scalars_name != name:
                    name = None

        if name is None:
            # check for the active scalars in point or cell arrays
            self._active_scalars_info = ActiveArrayInfoTuple(field, None)
            for attr in [self.point_data, self.cell_data]:
                if attr.active_scalars_name is not None:
                    self._active_scalars_info = ActiveArrayInfoTuple(
                        attr.association,
                        attr.active_scalars_name,
                    )
                    break

        return self._active_scalars_info

    @property
    def active_vectors_info(self: Self) -> ActiveArrayInfoTuple:
        """Return the active vector's association and name.

        Association refers to the data association (e.g. point, cell, or
        field) of the active vectors.

        Returns
        -------
        ActiveArrayInfo
            The vectors info in an object with namedtuple semantics,
            with attributes ``association`` and ``name``.

        Notes
        -----
        If both cell and point vectors are present and neither have
        been set active within at the dataset level, point vectors
        will be made active.

        Examples
        --------
        Create a mesh, compute the normals inplace, set the active
        vectors to the normals, and show that the active vectors are
        the ``'Normals'`` array associated with points.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> _ = mesh.compute_normals(inplace=True)
        >>> mesh.active_vectors_name = 'Normals'
        >>> mesh.active_vectors_info
        ActiveArrayInfoTuple(association=<FieldAssociation.POINT: 0>, name='Normals')

        """
        field, name = self._active_vectors_info

        # verify this field is still valid
        if name is not None:
            if field is FieldAssociation.POINT:
                if self.point_data.active_vectors_name != name:
                    name = None
            if field is FieldAssociation.CELL:
                if self.cell_data.active_vectors_name != name:
                    name = None

        if name is None:
            # check for the active vectors in point or cell arrays
            self._active_vectors_info = ActiveArrayInfoTuple(field, None)
            for attr in [self.point_data, self.cell_data]:
                name = attr.active_vectors_name
                if name is not None:
                    self._active_vectors_info = ActiveArrayInfoTuple(attr.association, name)
                    break

        return self._active_vectors_info

    @property
    def active_tensors_info(self: Self) -> ActiveArrayInfoTuple:
        """Return the active tensor's field and name: [field, name].

        Returns
        -------
        ActiveArrayInfo
            Active tensor's field and name: [field, name].

        """
        return self._active_tensors_info

    @property
    def active_vectors(self: Self) -> pyvista_ndarray | None:
        """Return the active vectors array.

        Returns
        -------
        Optional[pyvista_ndarray]
            Active vectors array.

        Examples
        --------
        Create a mesh, compute the normals inplace, and return the
        normals vector array.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> _ = mesh.compute_normals(inplace=True)
        >>> mesh.active_vectors  # doctest:+SKIP
        pyvista_ndarray([[-2.48721432e-10, -1.08815623e-09, -1.00000000e+00],
                         [-2.48721432e-10, -1.08815623e-09,  1.00000000e+00],
                         [-1.18888125e-01,  3.40539310e-03, -9.92901802e-01],
                         ...,
                         [-3.11940581e-01, -6.81432486e-02,  9.47654784e-01],
                         [-2.09880397e-01, -4.65070531e-02,  9.76620376e-01],
                         [-1.15582108e-01, -2.80492082e-02,  9.92901802e-01]],
                        dtype=float32)

        """
        field, name = self.active_vectors_info
        if name is not None:
            try:
                if field is FieldAssociation.POINT:
                    return self.point_data[name]
                if field is FieldAssociation.CELL:
                    return self.cell_data[name]
            except KeyError:
                return None
        return None

    @property
    def active_tensors(self: Self) -> NumpyArray[float] | None:
        """Return the active tensors array.

        Returns
        -------
        Optional[np.ndarray]
            Active tensors array.

        """
        field: FieldAssociation = self.active_tensors_info.association
        name: str | None = self.active_tensors_info.name
        if name is not None:
            try:
                if field is FieldAssociation.POINT:
                    return self.point_data[name]
                if field is FieldAssociation.CELL:
                    return self.cell_data[name]
            except KeyError:
                return None
        return None

    @property
    def active_tensors_name(self: Self) -> str | None:
        """Return the name of the active tensor array.

        Returns
        -------
        str
            Name of the active tensor array.

        """
        return self.active_tensors_info.name

    @active_tensors_name.setter
    def active_tensors_name(self: Self, name: str | None) -> None:
        """Set the name of the active tensor array.

        Parameters
        ----------
        name : str
            Name of the active tensor array.

        """
        self.set_active_tensors(name)

    @property
    def active_vectors_name(self: Self) -> str | None:
        """Return the name of the active vectors array.

        Returns
        -------
        str
            Name of the active vectors array.

        Examples
        --------
        Create a mesh, compute the normals, set them as active, and
        return the name of the active vectors.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh_w_normals = mesh.compute_normals()
        >>> mesh_w_normals.active_vectors_name = 'Normals'
        >>> mesh_w_normals.active_vectors_name
        'Normals'

        """
        return self.active_vectors_info.name

    @active_vectors_name.setter
    def active_vectors_name(self: Self, name: str | None) -> None:
        """Set the name of the active vectors array.

        Parameters
        ----------
        name : str
            Name of the active vectors array.

        """
        self.set_active_vectors(name)

    @property
    def active_scalars_name(self: Self) -> str | None:
        """Return the name of the active scalars.

        Returns
        -------
        str
            Name of the active scalars.

        Examples
        --------
        Create a mesh, add scalars to the mesh, and return the name of
        the active scalars.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh['Z Height'] = mesh.points[:, 2]
        >>> mesh.active_scalars_name
        'Z Height'

        """
        return self.active_scalars_info.name

    @active_scalars_name.setter
    def active_scalars_name(self: Self, name: str | None) -> None:
        """Set the name of the active scalars.

        Parameters
        ----------
        name : str
             Name of the active scalars.

        """
        self.set_active_scalars(name)

    @property
    def points(self: Self) -> pyvista_ndarray:
        """Return a reference to the points as a numpy object.

        Returns
        -------
        pyvista_ndarray
            Reference to the points as a numpy object.

        Examples
        --------
        Create a mesh and return the points of the mesh as a numpy
        array.

        >>> import pyvista as pv
        >>> cube = pv.Cube()
        >>> points = cube.points
        >>> points
        pyvista_ndarray([[-0.5, -0.5, -0.5],
                         [-0.5, -0.5,  0.5],
                         [-0.5,  0.5,  0.5],
                         [-0.5,  0.5, -0.5],
                         [ 0.5, -0.5, -0.5],
                         [ 0.5,  0.5, -0.5],
                         [ 0.5,  0.5,  0.5],
                         [ 0.5, -0.5,  0.5]], dtype=float32)

        Shift these points in the z direction and show that their
        position is reflected in the mesh points.

        >>> points[:, 2] += 1
        >>> cube.points
        pyvista_ndarray([[-0.5, -0.5,  0.5],
                         [-0.5, -0.5,  1.5],
                         [-0.5,  0.5,  1.5],
                         [-0.5,  0.5,  0.5],
                         [ 0.5, -0.5,  0.5],
                         [ 0.5,  0.5,  0.5],
                         [ 0.5,  0.5,  1.5],
                         [ 0.5, -0.5,  1.5]], dtype=float32)

        You can also update the points in-place:

        >>> cube.points[...] = 2 * points
        >>> cube.points
        pyvista_ndarray([[-1., -1.,  1.],
                         [-1., -1.,  3.],
                         [-1.,  1.,  3.],
                         [-1.,  1.,  1.],
                         [ 1., -1.,  1.],
                         [ 1.,  1.,  1.],
                         [ 1.,  1.,  3.],
                         [ 1., -1.,  3.]], dtype=float32)

        """
        _points = self.GetPoints()
        try:
            _points = _points.GetData()
        except AttributeError:
            # create an empty array
            vtkpts = vtk_points(np.empty((0, 3)), deep=False)
            self.SetPoints(vtkpts)
            _points = self.GetPoints().GetData()
        return pyvista_ndarray(_points, dataset=self)

    @points.setter
    def points(self: Self, points: MatrixLike[float] | _vtk.vtkPoints) -> None:
        """Set a reference to the points as a numpy object.

        Parameters
        ----------
        points : MatrixLike[float] | :vtk:`vtkPoints`
            Points as a array object.

        """
        pdata = self.GetPoints()
        if isinstance(points, pyvista_ndarray):
            # simply set the underlying data
            if points.VTKObject is not None and pdata is not None:
                pdata.SetData(points.VTKObject)
                pdata.Modified()
                self.Modified()
                return
        # directly set the data if vtk object
        if isinstance(points, _vtk.vtkPoints):
            self.SetPoints(points)
            if pdata is not None:
                pdata.Modified()
            self.Modified()
            return
        # otherwise, wrap and use the array
        points, _ = _coerce_pointslike_arg(points, copy=False)
        vtkpts = vtk_points(points, deep=False)
        if not pdata:
            self.SetPoints(vtkpts)
        else:
            pdata.SetData(vtkpts.GetData())
        self.GetPoints().Modified()
        self.Modified()

    @property
    def arrows(
        self: Self,
    ) -> pyvista.PolyData | None:
        """Return a glyph representation of the active vector data as arrows.

        Arrows will be located at the points or cells of the mesh and
        their size will be dependent on the norm of the vector.
        Their direction will be the "direction" of the vector.

        If there are both active point and cell vectors, preference is
        given to the point vectors.

        Returns
        -------
        pyvista.PolyData
            Active vectors represented as arrows.

        Examples
        --------
        Create a mesh, compute the normals and set them active.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> mesh_w_normals = mesh.compute_normals()
        >>> mesh_w_normals.active_vectors_name = 'Normals'

        Plot the active vectors as arrows. Show the original mesh as wireframe for
        context.

        >>> arrows = mesh_w_normals.arrows
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, style='wireframe')
        >>> _ = pl.add_mesh(arrows, color='red')
        >>> pl.show()

        """
        vectors = self.active_vectors
        if vectors is None:
            return None

        if vectors.ndim != 2:
            msg = 'Active vectors are not vectors.'
            raise ValueError(msg)

        field, vectors_name = self.active_vectors_info
        # Cast type since we know name is not None since vectors is not None at this point
        vectors_name = cast('str', vectors_name)

        scale_name = f'{vectors_name} Magnitude'
        scale = np.linalg.norm(vectors, axis=1)
        if field == FieldAssociation.POINT:
            self.point_data.set_array(scale, scale_name)
        else:
            self.cell_data.set_array(scale, scale_name)
        return self.glyph(orient=vectors_name, scale=scale_name)

    def set_active_scalars(
        self: Self,
        name: str | None,
        preference: PointLiteral | CellLiteral = 'cell',
    ) -> tuple[FieldAssociation, NumpyArray[float] | None]:
        """Find the scalars by name and appropriately sets it as active.

        To deactivate any active scalars, pass ``None`` as the ``name``.

        Parameters
        ----------
        name : str, optional
            Name of the scalars array to assign as active.  If
            ``None``, deactivates active scalars for both point and
            cell data.

        preference : str, default: "cell"
            If there are two arrays of the same name associated with
            points or cells, it will prioritize an array matching this
            type.  Can be either ``'cell'`` or ``'point'``.

        Returns
        -------
        pyvista.core.utilities.arrays.FieldAssociation
            Association of the scalars matching ``name``.

        pyvista_ndarray
            An array from the dataset matching ``name``.

        """
        if preference not in [
            'point',
            'cell',
            FieldAssociation.CELL,
            FieldAssociation.POINT,
        ]:
            msg = '``preference`` must be either "point" or "cell"'
            raise ValueError(msg)
        if name is None:
            self.GetCellData().SetActiveScalars(None)
            self.GetPointData().SetActiveScalars(None)
            return FieldAssociation.NONE, np.array([])
        field = get_array_association(self, name, preference=preference)
        if field == FieldAssociation.NONE:
            if name in self.field_data:
                msg = f'Data named "{name}" is a field array which cannot be active.'
                raise ValueError(msg)
            else:
                msg = f'Data named "{name}" does not exist in this dataset.'
                raise KeyError(msg)
        self._last_active_scalars_name = self.active_scalars_info.name
        if field == FieldAssociation.POINT:
            ret = self.GetPointData().SetActiveScalars(name)
        elif field == FieldAssociation.CELL:
            ret = self.GetCellData().SetActiveScalars(name)
        else:
            msg = f'Data field ({name}) with type ({field}) not usable'
            raise ValueError(msg)

        if ret < 0:
            msg = f'Data field "{name}" with type ({field}) could not be set as the active scalars'
            raise ValueError(msg)

        self._active_scalars_info = ActiveArrayInfoTuple(field, name)

        if field == FieldAssociation.POINT:
            return field, self.point_data.active_scalars
        else:  # must be cell
            return field, self.cell_data.active_scalars

    def set_active_vectors(
        self: Self, name: str | None, preference: PointLiteral | CellLiteral = 'point'
    ) -> None:
        """Find the vectors by name and appropriately sets it as active.

        To deactivate any active vectors, pass ``None`` as the ``name``.

        Parameters
        ----------
        name : str, optional
            Name of the vectors array to assign as active.

        preference : str, default: "point"
            If there are two arrays of the same name associated with
            points, cells, or field data, it will prioritize an array
            matching this type.  Can be either ``'cell'``,
            ``'field'``, or ``'point'``.

        """
        if name is None:
            self.GetCellData().SetActiveVectors(None)
            self.GetPointData().SetActiveVectors(None)
            field = FieldAssociation.POINT
        else:
            field = get_array_association(self, name, preference=preference)
            if field == FieldAssociation.POINT:
                ret = self.GetPointData().SetActiveVectors(name)
            elif field == FieldAssociation.CELL:
                ret = self.GetCellData().SetActiveVectors(name)
            else:
                msg = f'Data field ({name}) with type ({field}) not usable'
                raise ValueError(msg)

            if ret < 0:
                msg = (
                    f'Data field ({name}) with type ({field}) could not be set as the '
                    f'active vectors'
                )
                raise ValueError(msg)

        self._active_vectors_info = ActiveArrayInfoTuple(field, name)

    def set_active_tensors(
        self: Self, name: str | None, preference: PointLiteral | CellLiteral = 'point'
    ) -> None:
        """Find the tensors by name and appropriately sets it as active.

        To deactivate any active tensors, pass ``None`` as the ``name``.

        Parameters
        ----------
        name : str, optional
            Name of the tensors array to assign as active.

        preference : str, default: "point"
            If there are two arrays of the same name associated with
            points, cells, or field data, it will prioritize an array
            matching this type.  Can be either ``'cell'``,
            ``'field'``, or ``'point'``.

        """
        if name is None:
            self.GetCellData().SetActiveTensors(None)
            self.GetPointData().SetActiveTensors(None)
            field = FieldAssociation.POINT
        else:
            field = get_array_association(self, name, preference=preference)
            if field == FieldAssociation.POINT:
                ret = self.GetPointData().SetActiveTensors(name)
            elif field == FieldAssociation.CELL:
                ret = self.GetCellData().SetActiveTensors(name)
            else:
                msg = f'Data field ({name}) with type ({field}) not usable'
                raise ValueError(msg)

            if ret < 0:
                msg = (
                    f'Data field ({name}) with type ({field}) could not be set as the '
                    f'active tensors'
                )
                raise ValueError(msg)

        self._active_tensors_info = ActiveArrayInfoTuple(field, name)

    def rename_array(
        self: Self,
        old_name: str,
        new_name: str,
        preference: PointLiteral | CellLiteral | FieldLiteral = 'cell',
    ) -> None:
        """Change array name by searching for the array then renaming it.

        Parameters
        ----------
        old_name : str
            Name of the array to rename.

        new_name : str
            Name to rename the array to.

        preference : str, default: "cell"
            If there are two arrays of the same name associated with
            points, cells, or field data, it will prioritize an array
            matching this type.  Can be either ``'cell'``,
            ``'field'``, or ``'point'``.

        Examples
        --------
        Create a cube, assign a point array to the mesh named
        ``'my_array'``, and rename it to ``'my_renamed_array'``.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> cube = pv.Cube()
        >>> cube['my_array'] = range(cube.n_points)
        >>> cube.rename_array('my_array', 'my_renamed_array')
        >>> cube['my_renamed_array']
        pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7])

        """
        field = get_array_association(self, old_name, preference=preference)

        was_active = False
        if self.active_scalars_name == old_name:
            was_active = True
        if field == FieldAssociation.POINT:
            data = self.point_data
        elif field == FieldAssociation.CELL:
            data = self.cell_data
        elif field == FieldAssociation.NONE:
            data = self.field_data
        else:
            msg = f'Array with name {old_name} not found.'
            raise KeyError(msg)

        arr = data.pop(old_name)
        # Update the array's name before reassigning. This prevents taking a copy of the array in
        # `DataSetAttributes._prepare_array` which can lead to the array being garbage collected.
        # See issue #5244.
        arr.VTKObject.SetName(new_name)  # type: ignore[union-attr]
        data[new_name] = arr

        if was_active and field != FieldAssociation.NONE:
            self.set_active_scalars(new_name, preference=field)

    @property
    def active_scalars(self: Self) -> pyvista_ndarray | None:
        """Return the active scalars as an array.

        Returns
        -------
        Optional[pyvista_ndarray]
            Active scalars as an array.

        """
        field, name = self.active_scalars_info
        if name is not None:
            try:
                if field == FieldAssociation.POINT:
                    return self.point_data[name]
                if field == FieldAssociation.CELL:
                    return self.cell_data[name]
            except KeyError:
                return None
        return None

    @property
    def active_normals(self: Self) -> pyvista_ndarray | None:
        """Return the active normals as an array.

        Returns
        -------
        pyvista_ndarray
            Active normals of this dataset.

        Notes
        -----
        If both point and cell normals exist, this returns point
        normals by default.

        Examples
        --------
        Compute normals on an example sphere mesh and return the
        active normals for the dataset.  Show that this is the same size
        as the number of points.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh = mesh.compute_normals()
        >>> normals = mesh.active_normals
        >>> normals.shape
        (842, 3)
        >>> mesh.n_points
        842

        """
        if self.point_data.active_normals is not None:
            return self.point_data.active_normals
        return self.cell_data.active_normals

    def get_data_range(  # type: ignore[override]
        self: Self,
        arr_var: str | NumpyArray[float] | None = None,
        preference: PointLiteral | CellLiteral | FieldLiteral = 'cell',
    ) -> tuple[float, float]:
        """Get the min and max of a named array.

        Parameters
        ----------
        arr_var : str, np.ndarray, optional
            The name of the array to get the range. If ``None``, the
            active scalars is used.

        preference : str, default: "cell"
            When scalars is specified, this is the preferred array type
            to search for in the dataset.  Must be either ``'point'``,
            ``'cell'``, or ``'field'``.

        Returns
        -------
        tuple
            ``(min, max)`` of the named array.

        """
        if arr_var is None:  # use active scalars array
            arr_var = self.active_scalars_info.name
            if arr_var is None:
                return (np.nan, np.nan)

        if isinstance(arr_var, str):
            name = arr_var
            arr = get_array(self, name, preference=preference, err=True)
        else:
            arr = arr_var  # type: ignore[assignment]

        # If array has no tuples return a NaN range
        if arr is None:
            return (np.nan, np.nan)
        if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
            return (np.nan, np.nan)
        # Use the array range
        return np.nanmin(arr), np.nanmax(arr)

    @_deprecate_positional_args(allowed=['ido'])
    def copy_meta_from(self: Self, ido: DataSet, deep: bool = True) -> None:  # noqa: FBT001, FBT002
        """Copy pyvista meta data onto this object from another object.

        Parameters
        ----------
        ido : pyvista.DataSet
            Dataset to copy the metadata from.

        deep : bool, default: True
            Deep or shallow copy.

        """
        if deep:
            self._association_complex_names = deepcopy(ido._association_complex_names)
            self._association_bitarray_names = deepcopy(ido._association_bitarray_names)
            self._active_scalars_info = ido.active_scalars_info.copy()
            self._active_vectors_info = ido.active_vectors_info.copy()
            self._active_tensors_info = ido.active_tensors_info.copy()
        else:
            # pass by reference
            self._association_complex_names = ido._association_complex_names
            self._association_bitarray_names = ido._association_bitarray_names
            self._active_scalars_info = ido.active_scalars_info
            self._active_vectors_info = ido.active_vectors_info
            self._active_tensors_info = ido.active_tensors_info

    @property
    def point_data(self: Self) -> DataSetAttributes:
        """Return point data as DataSetAttributes.

        Returns
        -------
        DataSetAttributes
            Point data as DataSetAttributes.

        Examples
        --------
        Add point arrays to a mesh and list the available ``point_data``.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> mesh = pv.Cube()
        >>> mesh.clear_data()
        >>> mesh.point_data['my_array'] = np.random.default_rng().random(mesh.n_points)
        >>> mesh.point_data['my_other_array'] = np.arange(mesh.n_points)
        >>> mesh.point_data
        pyvista DataSetAttributes
        Association     : POINT
        Active Scalars  : my_array
        Active Vectors  : None
        Active Texture  : None
        Active Normals  : None
        Contains arrays :
            my_array                float64    (8,)                 SCALARS
            my_other_array          int64      (8,)

        Access an array from ``point_data``.

        >>> mesh.point_data['my_other_array']
        pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7])

        Or access it directly from the mesh.

        >>> mesh['my_array'].shape
        (8,)

        """
        return DataSetAttributes(
            self.GetPointData(),
            dataset=self,
            association=FieldAssociation.POINT,
        )

    def clear_point_data(self: Self) -> None:
        """Remove all point arrays.

        Examples
        --------
        Clear all point arrays from a mesh.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> mesh = pv.Sphere()
        >>> mesh.point_data.keys()
        ['Normals']
        >>> mesh.clear_point_data()
        >>> mesh.point_data.keys()
        []

        """
        self.point_data.clear()

    def clear_cell_data(self: Self) -> None:
        """Remove all cell arrays."""
        self.cell_data.clear()

    def clear_data(self: Self) -> None:
        """Remove all arrays from point/cell/field data.

        Examples
        --------
        Clear all arrays from a mesh.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> mesh = pv.Sphere()
        >>> mesh.point_data.keys()
        ['Normals']
        >>> mesh.clear_data()
        >>> mesh.point_data.keys()
        []

        """
        self.clear_point_data()
        self.clear_cell_data()
        self.clear_field_data()

    @property
    def cell_data(self: Self) -> DataSetAttributes:
        """Return cell data as DataSetAttributes.

        Returns
        -------
        DataSetAttributes
            Cell data as DataSetAttributes.

        Examples
        --------
        Add cell arrays to a mesh and list the available ``cell_data``.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> mesh = pv.Cube()
        >>> mesh.clear_data()
        >>> mesh.cell_data['my_array'] = np.random.default_rng().random(mesh.n_cells)
        >>> mesh.cell_data['my_other_array'] = np.arange(mesh.n_cells)
        >>> mesh.cell_data
        pyvista DataSetAttributes
        Association     : CELL
        Active Scalars  : my_array
        Active Vectors  : None
        Active Texture  : None
        Active Normals  : None
        Contains arrays :
            my_array                float64    (6,)                 SCALARS
            my_other_array          int64      (6,)

        Access an array from ``cell_data``.

        >>> mesh.cell_data['my_other_array']
        pyvista_ndarray([0, 1, 2, 3, 4, 5])

        Or access it directly from the mesh.

        >>> mesh['my_array'].shape
        (6,)

        """
        return DataSetAttributes(
            self.GetCellData(),
            dataset=self,
            association=FieldAssociation.CELL,
        )

    @property
    def n_points(self: Self) -> int:
        """Return the number of points in the entire dataset.

        Returns
        -------
        int
            Number of points in the entire dataset.

        Examples
        --------
        Create a mesh and return the number of points in the
        mesh.

        >>> import pyvista as pv
        >>> cube = pv.Cube()
        >>> cube.n_points
        8

        """
        return self.GetNumberOfPoints()

    @property
    def n_cells(self: Self) -> int:
        """Return the number of cells in the entire dataset.

        Returns
        -------
        int :
             Number of cells in the entire dataset.

        Notes
        -----
        This returns the total number of cells -- for :class:`pyvista.PolyData`
        this includes vertices, lines, triangle strips and polygonal faces.

        Examples
        --------
        Create a mesh and return the number of cells in the
        mesh.

        >>> import pyvista as pv
        >>> cube = pv.Cube()
        >>> cube.n_cells
        6

        """
        return self.GetNumberOfCells()

    @property
    def number_of_points(self: Self) -> int:  # pragma: no cover
        """Return the number of points.

        Returns
        -------
        int :
             Number of points.

        """
        return self.GetNumberOfPoints()

    @property
    def number_of_cells(self: Self) -> int:  # pragma: no cover
        """Return the number of cells.

        Returns
        -------
        int :
             Number of cells.

        """
        return self.GetNumberOfCells()

    @property
    def bounds(self: Self) -> BoundsTuple:
        """Return the bounding box of this dataset.

        Returns
        -------
        BoundsLike
            Bounding box of this dataset.
            The form is: ``(x_min, x_max, y_min, y_max, z_min, z_max)``.

        Examples
        --------
        Create a cube and return the bounds of the mesh.

        >>> import pyvista as pv
        >>> cube = pv.Cube()
        >>> cube.bounds
        BoundsTuple(x_min = -0.5,
                    x_max =  0.5,
                    y_min = -0.5,
                    y_max =  0.5,
                    z_min = -0.5,
                    z_max =  0.5)

        """
        return BoundsTuple(*self.GetBounds())

    @property
    def length(self: Self) -> float:
        """Return the length of the diagonal of the bounding box.

        Returns
        -------
        float
            Length of the diagonal of the bounding box.

        Examples
        --------
        Get the length of the bounding box of a cube.  This should
        match ``3**(1/2)`` since it is the diagonal of a cube that is
        ``1 x 1 x 1``.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> mesh.length
        1.7320508075688772

        """
        return self.GetLength()

    @property
    def center(self: Self) -> tuple[float, float, float]:
        """Return the center of the bounding box.

        Returns
        -------
        tuple[float, float, float]
            Center of the bounding box.

        Examples
        --------
        Get the center of a mesh.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere(center=(1, 2, 0))
        >>> mesh.center
        (1.0, 2.0, 0.0)

        """
        return self.GetCenter()

    @property
    def volume(
        self: Self,
    ) -> float:
        """Return the mesh volume.

        This will return 0 for meshes with 2D cells.

        Returns
        -------
        float
            Total volume of the mesh.

        Examples
        --------
        Get the volume of a cube of size 4x4x4.
        Note that there are 5 points in each direction.

        >>> import pyvista as pv
        >>> mesh = pv.ImageData(dimensions=(5, 5, 5))
        >>> mesh.volume
        64.0

        A mesh with 2D cells has no volume.

        >>> mesh = pv.ImageData(dimensions=(5, 5, 1))
        >>> mesh.volume
        0.0

        :class:`pyvista.PolyData` is special as a 2D surface can
        enclose a 3D volume. This case uses a different methodology,
        see :func:`pyvista.PolyData.volume`.

        >>> mesh = pv.Sphere()
        >>> mesh.volume
        0.51825

        """
        sizes = self.compute_cell_sizes(length=False, area=False, volume=True)
        return sizes.cell_data['Volume'].sum().item()

    @property
    def area(
        self: Self,
    ) -> float:
        """Return the mesh area if 2D.

        This will return 0 for meshes with 3D cells.

        Returns
        -------
        float
            Total area of the mesh.

        Examples
        --------
        Get the area of a square of size 2x2.
        Note 5 points in each direction.

        >>> import pyvista as pv
        >>> mesh = pv.ImageData(dimensions=(5, 5, 1))
        >>> mesh.area
        16.0

        A mesh with 3D cells does not have an area.  To get
        the outer surface area, first extract the surface using
        :func:`pyvista.DataSetFilters.extract_surface`.

        >>> mesh = pv.ImageData(dimensions=(5, 5, 5))
        >>> mesh.area
        0.0

        Get the area of a sphere. Discretization error results
        in slight difference from ``pi``.

        >>> mesh = pv.Sphere()
        >>> mesh.area
        3.13

        """
        sizes = self.compute_cell_sizes(length=False, area=True, volume=False)
        return sizes.cell_data['Area'].sum().item()

    def get_array(
        self: Self,
        name: str,
        preference: CellLiteral | PointLiteral | FieldLiteral = 'cell',
    ) -> pyvista.pyvista_ndarray:
        """Search both point, cell and field data for an array.

        Parameters
        ----------
        name : str
            Name of the array.

        preference : str, default: "cell"
            When scalars is specified, this is the preferred array
            type to search for in the dataset.  Must be either
            ``'point'``, ``'cell'``, or ``'field'``.

        Returns
        -------
        pyvista.pyvista_ndarray
            Requested array.

        Examples
        --------
        Create a DataSet with a variety of arrays.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> mesh.clear_data()
        >>> mesh.point_data['point-data'] = range(mesh.n_points)
        >>> mesh.cell_data['cell-data'] = range(mesh.n_cells)
        >>> mesh.field_data['field-data'] = ['a', 'b', 'c']
        >>> mesh.array_names
        ['point-data', 'field-data', 'cell-data']

        Get the point data array.

        >>> mesh.get_array('point-data')
        pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7])

        Get the cell data array.

        >>> mesh.get_array('cell-data')
        pyvista_ndarray([0, 1, 2, 3, 4, 5])

        Get the field data array.

        >>> mesh.get_array('field-data')
        pyvista_ndarray(['a', 'b', 'c'], dtype='<U1')

        """
        arr = get_array(self, name, preference=preference, err=True)
        if arr is None:  # pragma: no cover
            raise RuntimeError  # this should never be reached with err=True
        return arr

    def get_array_association(
        self: Self,
        name: str,
        preference: Literal['cell', 'point', 'field'] = 'cell',
    ) -> FieldAssociation:
        """Get the association of an array.

        Parameters
        ----------
        name : str
            Name of the array.

        preference : str, default: "cell"
            When ``name`` is specified, this is the preferred array
            association to search for in the dataset.  Must be either
            ``'point'``, ``'cell'``, or ``'field'``.

        Returns
        -------
        pyvista.core.utilities.arrays.FieldAssociation
            Field association of the array.

        Examples
        --------
        Create a DataSet with a variety of arrays.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> mesh.clear_data()
        >>> mesh.point_data['point-data'] = range(mesh.n_points)
        >>> mesh.cell_data['cell-data'] = range(mesh.n_cells)
        >>> mesh.field_data['field-data'] = ['a', 'b', 'c']
        >>> mesh.array_names
        ['point-data', 'field-data', 'cell-data']

        Get the point data array association.

        >>> mesh.get_array_association('point-data')
        <FieldAssociation.POINT: 0>

        Get the cell data array association.

        >>> mesh.get_array_association('cell-data')
        <FieldAssociation.CELL: 1>

        Get the field data array association.

        >>> mesh.get_array_association('field-data')
        <FieldAssociation.NONE: 2>

        """
        return get_array_association(self, name, preference=preference, err=True)

    def __getitem__(
        self: Self,
        index: tuple[str, Literal['cell', 'point', 'field']] | str,
    ) -> pyvista_ndarray:
        """Search both point, cell, and field data for an array."""
        if isinstance(index, tuple):
            name, preference = index
        elif isinstance(index, str):
            name = index
            preference = 'cell'
        else:
            msg = (  # type: ignore[unreachable]
                f'Index ({index}) not understood.'
                ' Index must be a string name or a tuple of string name and string preference.'
            )
            raise KeyError(msg)
        return self.get_array(name, preference=preference)

    def _ipython_key_completions_(self: Self) -> list[str]:
        """Tab completion of IPython."""
        return self.array_names

    def __setitem__(
        self: Self,
        name: str,
        scalars: NumpyArray[float] | Sequence[float] | float,
    ) -> None:  # numpydoc ignore=PR01,RT01
        """Add/set an array in the point_data, or cell_data accordingly.

        It depends on the array's length, or specified mode.

        """
        # First check points - think of case with vertex cells
        #   there would be the same number of cells as points but we'd want
        #   the data to be on the nodes.
        if scalars is None:
            msg = 'Empty array unable to be added.'  # type: ignore[unreachable]
            raise TypeError(msg)
        else:
            scalars = np.asanyarray(scalars)

        if scalars.ndim == 0:
            if np.issubdtype(scalars.dtype, np.str_):
                # Always set scalar strings as field data
                self.field_data[name] = scalars
                return
            # reshape single scalar values from 0D to 1D so that shape[0] can be indexed
            scalars = scalars.reshape((1,))

        # Now check array size to determine which field to place array
        if scalars.shape[0] == self.n_points:
            self.point_data[name] = scalars
        elif scalars.shape[0] == self.n_cells:
            self.cell_data[name] = scalars
        else:
            # Field data must be set explicitly as it could be a point of
            # confusion for new users
            raise_not_matching(scalars, self)
        return

    @property
    def n_arrays(self: Self) -> int:
        """Return the number of arrays present in the dataset.

        Returns
        -------
        int
           Number of arrays present in the dataset.

        """
        n = self.GetPointData().GetNumberOfArrays()
        n += self.GetCellData().GetNumberOfArrays()
        n += self.GetFieldData().GetNumberOfArrays()
        return n

    @property
    def array_names(self: Self) -> list[str]:
        """Return a list of array names for the dataset.

        This makes sure to put the active scalars' name first in the list.

        Returns
        -------
        list[str]
            List of array names for the dataset.

        Examples
        --------
        Return the array names for a mesh.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh.point_data['my_array'] = range(mesh.n_points)
        >>> mesh.array_names
        ['my_array', 'Normals']

        """
        names: list[str] = []
        names.extend(self.field_data.keys())
        names.extend(self.point_data.keys())
        names.extend(self.cell_data.keys())
        if self.active_scalars_name is not None:
            names.remove(self.active_scalars_name)
            names.insert(0, self.active_scalars_name)
        return names

    def _get_attrs(self: Self) -> list[tuple[str, Any, str]]:
        """Return the representation methods (internal helper)."""
        attrs: list[tuple[str, Any, str]] = []
        attrs.append(('N Cells', self.GetNumberOfCells(), '{}'))
        attrs.append(('N Points', self.GetNumberOfPoints(), '{}'))
        if isinstance(self, pyvista.PolyData):
            attrs.append(('N Strips', self.n_strips, '{}'))
        bds = self.bounds
        fmt = f'{pyvista.FLOAT_FORMAT}, {pyvista.FLOAT_FORMAT}'
        attrs.append(('X Bounds', (bds.x_min, bds.x_max), fmt))
        attrs.append(('Y Bounds', (bds.y_min, bds.y_max), fmt))
        attrs.append(('Z Bounds', (bds.z_min, bds.z_max), fmt))
        # if self.n_cells <= pyvista.REPR_VOLUME_MAX_CELLS and self.n_cells > 0:
        #     attrs.append(("Volume", (self.volume), pyvista.FLOAT_FORMAT))
        return attrs

    def _repr_html_(self: Self) -> str:
        """Return a pretty representation for Jupyter notebooks.

        It includes header details and information about all arrays.

        """
        fmt = ''
        if self.n_arrays > 0:
            fmt += "<table style='width: 100%;'>"
            fmt += '<tr><th>Header</th><th>Data Arrays</th></tr>'
            fmt += '<tr><td>'
        # Get the header info
        fmt += self.head(display=False, html=True)
        # Fill out arrays
        if self.n_arrays > 0:
            fmt += '</td><td>'
            fmt += '\n'
            fmt += "<table style='width: 100%;'>\n"
            titles = ['Name', 'Field', 'Type', 'N Comp', 'Min', 'Max']
            fmt += '<tr>' + ''.join([f'<th>{t}</th>' for t in titles]) + '</tr>\n'
            row = '<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>\n'
            row = '<tr>' + ''.join(['<td>{}</td>' for i in range(len(titles))]) + '</tr>\n'

            def format_array(
                name: str,
                arr: str | pyvista_ndarray,
                field: Literal['Points', 'Cells', 'Fields'],
            ) -> str:
                """Format array information for printing (internal helper)."""
                if isinstance(arr, str):
                    # Convert string scalar into a numpy array. Otherwise, get_data_range
                    # will treat the string as an array name, not an array value.
                    arr = pyvista.pyvista_ndarray(arr)  # type: ignore[arg-type]
                dl, dh = self.get_data_range(arr)
                dl = pyvista.FLOAT_FORMAT.format(dl)  # type: ignore[assignment]
                dh = pyvista.FLOAT_FORMAT.format(dh)  # type: ignore[assignment]
                if name == self.active_scalars_info.name:
                    name = f'<b>{name}</b>'
                ncomp = arr.shape[1] if arr.ndim > 1 else 1
                return row.format(name, field, arr.dtype, ncomp, dl, dh)

            for key, arr in self.point_data.items():
                fmt += format_array(key, arr, 'Points')
            for key, arr in self.cell_data.items():
                fmt += format_array(key, arr, 'Cells')
            for key, arr in self.field_data.items():
                fmt += format_array(key, arr, 'Fields')

            fmt += '</table>\n'
            fmt += '\n'
            fmt += '</td></tr> </table>'
        return fmt

    def __repr__(self: Self) -> str:
        """Return the object representation."""
        return self.head(display=False, html=False)

    def __str__(self: Self) -> str:
        """Return the object string representation."""
        return self.head(display=False, html=False)

    @_deprecate_positional_args(allowed=['mesh'])
    def copy_from(self: Self, mesh: _vtk.vtkDataSet, deep: bool = True) -> None:  # noqa: FBT001, FBT002
        """Overwrite this dataset inplace with the new dataset's geometries and data.

        Parameters
        ----------
        mesh : :vtk:`vtkDataSet`
            The overwriting mesh.

        deep : bool, default: True
            Whether to perform a deep or shallow copy.

        Examples
        --------
        Create two meshes and overwrite ``mesh_a`` with ``mesh_b``.
        Show that ``mesh_a`` is equal to ``mesh_b``.

        >>> import pyvista as pv
        >>> mesh_a = pv.Sphere()
        >>> mesh_b = pv.Cube()
        >>> mesh_a.copy_from(mesh_b)
        >>> mesh_a == mesh_b
        True

        """
        # Allow child classes to overwrite parent classes
        if not isinstance(self, type(mesh)):
            msg = (
                f'The Input DataSet type {type(mesh)} must be '
                f'compatible with the one being overwritten {type(self)}'
            )
            raise TypeError(msg)
        if deep:  # type: ignore[unreachable]
            self.deep_copy(mesh)
        else:
            self.shallow_copy(mesh)
        if is_pyvista_dataset(mesh):
            self.copy_meta_from(mesh, deep=deep)

    def cast_to_unstructured_grid(self: Self) -> pyvista.UnstructuredGrid:
        """Get a new representation of this object as a :class:`pyvista.UnstructuredGrid`.

        Returns
        -------
        pyvista.UnstructuredGrid
            Dataset cast into a :class:`pyvista.UnstructuredGrid`.

        Examples
        --------
        Cast a :class:`pyvista.PolyData` to a
        :class:`pyvista.UnstructuredGrid`.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> type(mesh)
        <class 'pyvista.core.pointset.PolyData'>
        >>> grid = mesh.cast_to_unstructured_grid()
        >>> type(grid)
        <class 'pyvista.core.pointset.UnstructuredGrid'>

        """
        alg = _vtk.vtkAppendFilter()
        alg.AddInputData(self)
        alg.Update()
        return _get_output(alg)

    @_deprecate_positional_args
    def cast_to_pointset(self: Self, pass_cell_data: bool = False) -> pyvista.PointSet:  # noqa: FBT001, FBT002
        """Extract the points of this dataset and return a :class:`pyvista.PointSet`.

        Parameters
        ----------
        pass_cell_data : bool, default: False
            Run the :func:`cell_data_to_point_data()
            <pyvista.DataObjectFilters.cell_data_to_point_data>` filter and pass
            cell data fields to the new pointset.

        Returns
        -------
        pyvista.PointSet
            Dataset cast into a :class:`pyvista.PointSet`.

        Notes
        -----
        This will produce a deep copy of the points and point/cell data of
        the original mesh.

        See Also
        --------
        :ref:`create_pointset_example`

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Wavelet()
        >>> pointset = mesh.cast_to_pointset()
        >>> type(pointset)
        <class 'pyvista.core.pointset.PointSet'>

        """
        pset = pyvista.PointSet()
        pset.points = self.points.copy()
        out = self.cell_data_to_point_data() if pass_cell_data else self
        pset.GetPointData().DeepCopy(out.GetPointData())
        pset.active_scalars_name = out.active_scalars_name
        return pset

    @_deprecate_positional_args
    def cast_to_poly_points(self: Self, pass_cell_data: bool = False) -> pyvista.PolyData:  # noqa: FBT001, FBT002
        """Extract the points of this dataset and return a :class:`pyvista.PolyData`.

        Parameters
        ----------
        pass_cell_data : bool, default: False
            Run the :func:`cell_data_to_point_data()
            <pyvista.DataObjectFilters.cell_data_to_point_data>` filter and pass
            cell data fields to the new pointset.

        Returns
        -------
        pyvista.PolyData
            Dataset cast into a :class:`pyvista.PolyData`.

        Notes
        -----
        This will produce a deep copy of the points and point/cell data of
        the original mesh.

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_uniform()
        >>> points = mesh.cast_to_poly_points(pass_cell_data=True)
        >>> type(points)
        <class 'pyvista.core.pointset.PolyData'>
        >>> points.n_arrays
        2
        >>> points.point_data
        pyvista DataSetAttributes
        Association     : POINT
        Active Scalars  : Spatial Point Data
        Active Vectors  : None
        Active Texture  : None
        Active Normals  : None
        Contains arrays :
            Spatial Point Data      float64    (1000,)              SCALARS
        >>> points.cell_data
        pyvista DataSetAttributes
        Association     : CELL
        Active Scalars  : None
        Active Vectors  : None
        Active Texture  : None
        Active Normals  : None
        Contains arrays :
            Spatial Cell Data       float64    (1000,)

        """
        pset = pyvista.PolyData(self.points.copy())
        if pass_cell_data:
            cell_data = self.copy()
            cell_data.clear_point_data()
            cell_data = cell_data.cell_data_to_point_data()
            pset.GetCellData().DeepCopy(cell_data.GetPointData())
        pset.GetPointData().DeepCopy(self.GetPointData())
        pset.active_scalars_name = self.active_scalars_name
        return pset

    @overload
    def find_closest_point(self: Self, point: Iterable[float], n: Literal[1] = 1) -> int: ...
    @overload
    def find_closest_point(
        self: Self, point: Iterable[float], n: int = ...
    ) -> VectorLike[int]: ...
    def find_closest_point(
        self: Self, point: Iterable[float], n: int = 1
    ) -> int | VectorLike[int]:
        """Find index of closest point in this mesh to the given point.

        If wanting to query many points, use a KDTree with scipy or another
        library as those implementations will be easier to work with.

        See: https://github.com/pyvista/pyvista-support/issues/107

        Parameters
        ----------
        point : sequence[float]
            Length 3 coordinate of the point to query.

        n : int, optional
            If greater than ``1``, returns the indices of the ``n`` closest
            points.

        Returns
        -------
        int
            The index of the point in this mesh that is closest to the given point.

        See Also
        --------
        DataSet.find_closest_cell
        DataSet.find_containing_cell
        DataSet.find_cells_along_line
        DataSet.find_cells_within_bounds

        Examples
        --------
        Find the index of the closest point to ``(0, 1, 0)``.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> index = mesh.find_closest_point((0, 1, 0))
        >>> index
        239

        Get the coordinate of that point.

        >>> mesh.points[index]
        pyvista_ndarray([-0.05218758,  0.49653167,  0.02706946], dtype=float32)

        """
        if not isinstance(point, (np.ndarray, Sequence)) or len(point) != 3:
            msg = 'Given point must be a length three sequence.'
            raise TypeError(msg)
        if not isinstance(n, int):
            msg = '`n` must be a positive integer.'  # type: ignore[unreachable]
            raise TypeError(msg)
        if n < 1:
            msg = '`n` must be a positive integer.'
            raise ValueError(msg)

        locator = _vtk.vtkPointLocator()
        locator.SetDataSet(self)
        locator.BuildLocator()
        if n > 1:
            id_list = _vtk.vtkIdList()
            locator.FindClosestNPoints(n, point, id_list)  # type: ignore[arg-type]
            return vtk_id_list_to_array(id_list)
        return locator.FindClosestPoint(point)  # type: ignore[arg-type]

    @_deprecate_positional_args(allowed=['point'])
    def find_closest_cell(
        self: Self,
        point: VectorLike[float] | MatrixLike[float],
        return_closest_point: bool = False,  # noqa: FBT001, FBT002
    ) -> int | NumpyArray[int] | tuple[int | NumpyArray[int], NumpyArray[int]]:
        """Find index of closest cell in this mesh to the given point.

        Parameters
        ----------
        point : VectorLike[float] | MatrixLike[float]
            Coordinates of point to query (length 3) or a
            :class:`numpy.ndarray` of ``n`` points with shape ``(n, 3)``.

        return_closest_point : bool, default: False
            If ``True``, the closest point within a mesh cell to that point is
            returned.  This is not necessarily the closest nodal point on the
            mesh.  Default is ``False``.

        Returns
        -------
        int or numpy.ndarray
            Index or indices of the cell in this mesh that is/are closest
            to the given point(s).

            .. versionchanged:: 0.35.0
               Inputs of shape ``(1, 3)`` now return a :class:`numpy.ndarray`
               of shape ``(1,)``.

        numpy.ndarray
            Point or points inside a cell of the mesh that is/are closest
            to the given point(s).  Only returned if
            ``return_closest_point=True``.

            .. versionchanged:: 0.35.0
               Inputs of shape ``(1, 3)`` now return a :class:`numpy.ndarray`
               of the same shape.

        Warnings
        --------
        This method may still return a valid cell index even if the point
        contains a value like ``numpy.inf`` or ``numpy.nan``.

        See Also
        --------
        DataSet.find_closest_point
        DataSet.find_containing_cell
        DataSet.find_cells_along_line
        DataSet.find_cells_within_bounds
        :ref:`distance_between_surfaces_example`

        Examples
        --------
        Find nearest cell on a sphere centered on the
        origin to the point ``[0.1, 0.2, 0.3]``.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> point = [0.1, 0.2, 0.3]
        >>> index = mesh.find_closest_cell(point)
        >>> index
        338

        Make sure that this cell indeed is the closest to
        ``[0.1, 0.2, 0.3]``.

        >>> import numpy as np
        >>> cell_centers = mesh.cell_centers()
        >>> relative_position = cell_centers.points - point
        >>> distance = np.linalg.norm(relative_position, axis=1)
        >>> np.argmin(distance)
        np.int64(338)

        Find the nearest cells to several random points that
        are centered on the origin.

        >>> points = 2 * np.random.default_rng().random((5000, 3)) - 1
        >>> indices = mesh.find_closest_cell(points)
        >>> indices.shape
        (5000,)

        For the closest cell, find the point inside the cell that is
        closest to the supplied point.  The rectangle is a unit square
        with 1 cell and 4 nodal points at the corners in the plane with
        ``z`` normal and ``z=0``.  The closest point inside the cell is
        not usually at a nodal point.

        >>> unit_square = pv.Rectangle()
        >>> index, closest_point = unit_square.find_closest_cell(
        ...     [0.25, 0.25, 0.5], return_closest_point=True
        ... )
        >>> closest_point
        array([0.25, 0.25, 0.  ])

        But, the closest point can be a nodal point, although the index of
        that point is not returned.  If the closest nodal point by index is
        desired, see :func:`DataSet.find_closest_point`.

        >>> index, closest_point = unit_square.find_closest_cell(
        ...     [1.0, 1.0, 0.5], return_closest_point=True
        ... )
        >>> closest_point
        array([1., 1., 0.])

        """
        point, singular = _coerce_pointslike_arg(point, copy=False)

        locator = _vtk.vtkCellLocator()
        locator.SetDataSet(self)
        locator.BuildLocator()

        cell = _vtk.vtkGenericCell()

        closest_cells: list[int] = []
        closest_points: list[list[float]] = []

        for node in point:
            closest_point = [0.0, 0.0, 0.0]
            cell_id = _vtk.mutable(0)
            sub_id = _vtk.mutable(0)
            dist2 = _vtk.mutable(0.0)

            locator.FindClosestPoint(node, closest_point, cell, cell_id, sub_id, dist2)  # type: ignore[call-overload]
            closest_cells.append(int(cell_id))
            closest_points.append(closest_point)

        out_cells: int | NumpyArray[int] = (
            closest_cells[0] if singular else np.array(closest_cells)
        )
        out_points = np.array(closest_points[0]) if singular else np.array(closest_points)

        if return_closest_point:
            return out_cells, out_points
        return out_cells

    def find_containing_cell(
        self: Self,
        point: VectorLike[float] | MatrixLike[float],
    ) -> int | NumpyArray[int]:
        """Find index of a cell that contains the given point.

        Parameters
        ----------
        point : VectorLike[float] | MatrixLike[float],
            Coordinates of point to query (length 3) or a
            :class:`numpy.ndarray` of ``n`` points with shape ``(n, 3)``.

        Returns
        -------
        int or numpy.ndarray
            Index or indices of the cell in this mesh that contains
            the given point.

            .. versionchanged:: 0.35.0
               Inputs of shape ``(1, 3)`` now return a :class:`numpy.ndarray`
               of shape ``(1,)``.

        See Also
        --------
        DataSet.find_closest_point
        DataSet.find_closest_cell
        DataSet.find_cells_along_line
        DataSet.find_cells_within_bounds

        Examples
        --------
        A unit square with 16 equal sized cells is created and a cell
        containing the point ``[0.3, 0.3, 0.0]`` is found.

        >>> import pyvista as pv
        >>> mesh = pv.ImageData(dimensions=[5, 5, 1], spacing=[1 / 4, 1 / 4, 0])
        >>> mesh
        ImageData...
        >>> mesh.find_containing_cell([0.3, 0.3, 0.0])
        5

        A point outside the mesh domain will return ``-1``.

        >>> mesh.find_containing_cell([0.3, 0.3, 1.0])
        -1

        Find the cells that contain 1000 random points inside the mesh.

        >>> import numpy as np
        >>> points = np.random.default_rng().random((1000, 3))
        >>> indices = mesh.find_containing_cell(points)
        >>> indices.shape
        (1000,)

        """
        point, singular = _coerce_pointslike_arg(point, copy=False)

        locator = _vtk.vtkCellLocator()
        locator.SetDataSet(self)
        locator.BuildLocator()

        containing_cells = [locator.FindCell(node) for node in point]
        return containing_cells[0] if singular else np.array(containing_cells)

    def find_cells_along_line(
        self: Self,
        pointa: VectorLike[float],
        pointb: VectorLike[float],
        tolerance: float = 0.0,
    ) -> NumpyArray[int]:
        """Find the index of cells whose bounds intersect a line.

        Line is defined from ``pointa`` to ``pointb``.

        Parameters
        ----------
        pointa : VectorLike
            Length 3 coordinate of the start of the line.

        pointb : VectorLike
            Length 3 coordinate of the end of the line.

        tolerance : float, default: 0.0
            The absolute tolerance to use to find cells along line.

        Returns
        -------
        numpy.ndarray
            Index or indices of the cell(s) whose bounds intersect
            the line.

        Warnings
        --------
        This method returns cells whose bounds intersect the line.
        This means that the line may not intersect the cell itself.
        To obtain cells that intersect the line, use
        :func:`pyvista.DataSet.find_cells_intersecting_line`.

        See Also
        --------
        DataSet.find_closest_point
        DataSet.find_closest_cell
        DataSet.find_containing_cell
        DataSet.find_cells_within_bounds
        DataSet.find_cells_intersecting_line

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh.find_cells_along_line([0.0, 0, 0], [1.0, 0, 0])
        array([  86,   87, 1652, 1653])

        """
        if np.array(pointa).size != 3:
            msg = 'Point A must be a length three tuple of floats.'
            raise TypeError(msg)
        if np.array(pointb).size != 3:
            msg = 'Point B must be a length three tuple of floats.'
            raise TypeError(msg)
        locator = _vtk.vtkCellLocator()
        locator.SetDataSet(self)
        locator.BuildLocator()
        id_list = _vtk.vtkIdList()
        locator.FindCellsAlongLine(
            cast('Sequence[float]', pointa),
            cast('Sequence[float]', pointb),
            tolerance,
            id_list,
        )
        return vtk_id_list_to_array(id_list)

    def find_cells_intersecting_line(
        self: Self,
        pointa: VectorLike[float],
        pointb: VectorLike[float],
        tolerance: float = 0.0,
    ) -> NumpyArray[int]:
        """Find the index of cells that intersect a line.

        Line is defined from ``pointa`` to ``pointb``.  This
        method requires vtk version >=9.2.0.

        Parameters
        ----------
        pointa : sequence[float]
            Length 3 coordinate of the start of the line.

        pointb : sequence[float]
            Length 3 coordinate of the end of the line.

        tolerance : float, default: 0.0
            The absolute tolerance to use to find cells along line.

        Returns
        -------
        numpy.ndarray
            Index or indices of the cell(s) that intersect
            the line.

        See Also
        --------
        DataSet.find_closest_point
        DataSet.find_closest_cell
        DataSet.find_containing_cell
        DataSet.find_cells_within_bounds
        DataSet.find_cells_along_line

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh.find_cells_intersecting_line([0.0, 0, 0], [1.0, 0, 0])
        array([  86, 1653])

        """
        if np.array(pointa).size != 3:
            msg = 'Point A must be a length three tuple of floats.'
            raise TypeError(msg)
        if np.array(pointb).size != 3:
            msg = 'Point B must be a length three tuple of floats.'
            raise TypeError(msg)
        locator = _vtk.vtkCellLocator()
        locator.SetDataSet(cast('_vtk.vtkDataSet', self))
        locator.BuildLocator()
        id_list = _vtk.vtkIdList()
        points = _vtk.vtkPoints()
        cell = _vtk.vtkGenericCell()
        locator.IntersectWithLine(
            cast('Sequence[float]', pointa),
            cast('Sequence[float]', pointb),
            tolerance,
            points,
            id_list,
            cell,
        )
        return vtk_id_list_to_array(id_list)

    def find_cells_within_bounds(self: Self, bounds: BoundsTuple) -> NumpyArray[int]:
        """Find the index of cells in this mesh within bounds.

        Parameters
        ----------
        bounds : sequence[float]
            Bounding box. The form is: ``(x_min, x_max, y_min, y_max, z_min, z_max)``.

        Returns
        -------
        numpy.ndarray
            Index or indices of the cell in this mesh that are closest
            to the given point.

        See Also
        --------
        DataSet.find_closest_point
        DataSet.find_closest_cell
        DataSet.find_containing_cell
        DataSet.find_cells_along_line

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> index = mesh.find_cells_within_bounds([-2.0, 2.0, -2.0, 2.0, -2.0, 2.0])

        """
        if np.array(bounds).size != 6:
            msg = 'Bounds must be a length six tuple of floats.'
            raise TypeError(msg)
        locator = _vtk.vtkCellTreeLocator()
        locator.SetDataSet(cast('_vtk.vtkDataSet', self))
        locator.BuildLocator()
        id_list = _vtk.vtkIdList()
        locator.FindCellsWithinBounds(list(bounds), id_list)
        return vtk_id_list_to_array(id_list)

    def get_cell(self: Self, index: int) -> pyvista.Cell:
        """Return a :class:`pyvista.Cell` object.

        Parameters
        ----------
        index : int
            Cell ID.

        Returns
        -------
        pyvista.Cell
            The i-th pyvista.Cell.

        Notes
        -----
        Cells returned from this method are deep copies of the original
        cells. Changing properties (for example, ``points``) will not affect
        the dataset they originated from.

        Examples
        --------
        Get the 0-th cell.

        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()
        >>> cell = mesh.get_cell(0)
        >>> cell
        Cell ...

        Get the point ids of the first cell

        >>> cell.point_ids
        [0, 1, 2]

        Get the point coordinates of the first cell

        >>> cell.points
        array([[897.0,  48.8,  82.3],
               [906.6,  48.8,  80.7],
               [907.5,  55.5,  83.7]])

        For the first cell, get the points associated with the first edge

        >>> cell.edges[0].point_ids
        [0, 1]

        For a Tetrahedron, get the point ids of the last face

        >>> mesh = examples.cells.Tetrahedron()
        >>> cell = mesh.get_cell(0)
        >>> cell.faces[-1].point_ids
        [0, 2, 1]

        """
        # must check upper bounds, otherwise segfaults (on Linux, 9.2)
        if index + 1 > self.n_cells:
            msg = f'Invalid index {index} for a dataset with {self.n_cells} cells.'
            raise IndexError(msg)

        # Note: we have to use vtkGenericCell here since
        # GetCell(vtkIdType cellId, vtkGenericCell* cell) is thread-safe,
        # while GetCell(vtkIdType cellId) is not.
        cell = pyvista.Cell()
        self.GetCell(index, cell)
        cell.SetCellType(self.GetCellType(index))
        return cell

    @property
    def cell(self: Self) -> Iterator[pyvista.Cell]:
        """A generator that provides an easy way to loop over all cells.

        To access a single cell, use :func:`pyvista.DataSet.get_cell`.

        .. versionchanged:: 0.39.0
            Now returns a generator instead of a list.
            Use ``get_cell(i)`` instead of ``cell[i]``.

        Yields
        ------
        pyvista.Cell

        See Also
        --------
        pyvista.DataSet.get_cell

        Examples
        --------
        Loop over the cells

        >>> import pyvista as pv
        >>> # Create a grid with 9 points and 4 cells
        >>> mesh = pv.ImageData(dimensions=(3, 3, 1))
        >>> for cell in mesh.cell:  # doctest: +SKIP
        ...     cell

        """
        for i in range(self.n_cells):
            yield self.get_cell(i)

    def cell_neighbors(self: Self, ind: int, connections: str = 'points') -> list[int]:
        """Get the cell neighbors of the ind-th cell.

        Concrete implementation of :vtk:`vtkDataSet.GetCellNeighbors`.

        Parameters
        ----------
        ind : int
            Cell ID.

        connections : str, default: "points"
            Describe how the neighbor cell(s) must be connected to the current
            cell to be considered as a neighbor.
            Can be either ``'points'``, ``'edges'`` or ``'faces'``.

        Returns
        -------
        list[int]
            List of neighbor cells IDs for the ind-th cell.

        Warnings
        --------
        For a :class:`pyvista.ExplicitStructuredGrid`, use
        :func:`pyvista.ExplicitStructuredGrid.neighbors`.

        See Also
        --------
        pyvista.DataSet.cell_neighbors_levels

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()

        Get the neighbor cell ids that have at least one point in common with
        the 0-th cell.

        >>> mesh.cell_neighbors(0, 'points')
        [1, 2, 3, 388, 389, 11, 12, 395, 14, 209, 211, 212]

        Get the neighbor cell ids that have at least one edge in common with
        the 0-th cell.

        >>> mesh.cell_neighbors(0, 'edges')
        [1, 3, 12]

        For unstructured grids with cells of dimension 3 (Tetrahedron for example),
        cell neighbors can be defined using faces.

        >>> mesh = examples.download_tetrahedron()
        >>> mesh.cell_neighbors(0, 'faces')
        [1, 5, 7]

        Show a visual example.

        >>> from functools import partial
        >>> import pyvista as pv
        >>> mesh = pv.Sphere(theta_resolution=10)
        >>>
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> pl.link_views()
        >>> add_point_labels = partial(
        ...     pl.add_point_labels,
        ...     text_color='white',
        ...     font_size=20,
        ...     shape=None,
        ...     show_points=False,
        ... )
        >>>
        >>> for i, connection in enumerate(['points', 'edges']):
        ...     pl.subplot(0, i)
        ...     pl.view_xy()
        ...     _ = pl.add_title(
        ...         f'{connection.capitalize()} neighbors',
        ...         color='red',
        ...         shadow=True,
        ...         font_size=8,
        ...     )
        ...
        ...     # Add current cell
        ...     i_cell = 0
        ...     current_cell = mesh.extract_cells(i_cell)
        ...     _ = pl.add_mesh(current_cell, show_edges=True, color='blue')
        ...     _ = add_point_labels(
        ...         current_cell.cell_centers().points,
        ...         labels=[f'{i_cell}'],
        ...     )
        ...
        ...     # Add neighbors
        ...     ids = mesh.cell_neighbors(i_cell, connection)
        ...     cells = mesh.extract_cells(ids)
        ...     _ = pl.add_mesh(cells, color='red', show_edges=True)
        ...     _ = add_point_labels(
        ...         cells.cell_centers().points,
        ...         labels=[f'{i}' for i in ids],
        ...     )
        ...
        ...     # Add other cells
        ...     ids.append(i_cell)
        ...     others = mesh.extract_cells(ids, invert=True)
        ...     _ = pl.add_mesh(others, show_edges=True)
        >>> pl.show()

        """
        if isinstance(self, _vtk.vtkExplicitStructuredGrid):
            msg = 'For an ExplicitStructuredGrid, use the `neighbors` method'  # type: ignore[unreachable]
            raise TypeError(msg)

        # Build links as recommended:
        # https://vtk.org/doc/nightly/html/classvtkPolyData.html#adf9caaa01f72972d9a986ba997af0ac7
        if hasattr(self, 'BuildLinks'):
            self.BuildLinks()

        needed = ['points', 'edges', 'faces']
        if connections not in needed:
            msg = f'`connections` must be one of: {needed} (got "{connections}")'
            raise ValueError(msg)

        cell = self.get_cell(ind)

        iterators = {
            'points': cell.point_ids,
            'edges': range(cell.n_edges),
            'faces': range(cell.n_faces),
        }

        def generate_ids(i: int, connections: str) -> _vtk.vtkIdList | None:
            if connections == 'points':
                ids = _vtk.vtkIdList()
                ids.InsertNextId(i)
                return ids
            elif connections == 'edges':
                return cell.get_edge(i).GetPointIds()
            elif connections == 'faces':
                return cell.get_face(i).GetPointIds()
            return None  # pragma: no cover

        neighbors = set()
        for i in iterators[connections]:
            point_ids = generate_ids(i, connections)
            cell_ids = _vtk.vtkIdList()
            self.GetCellNeighbors(ind, point_ids, cell_ids)

            neighbors.update([cell_ids.GetId(i) for i in range(cell_ids.GetNumberOfIds())])

        return list(neighbors)

    def point_neighbors(self: Self, ind: int) -> list[int]:
        """Get the point neighbors of the ind-th point.

        Parameters
        ----------
        ind : int
            Point ID.

        Returns
        -------
        list[int]
            List of neighbor points IDs for the ind-th point.

        See Also
        --------
        pyvista.DataSet.point_neighbors_levels

        Examples
        --------
        Get the point neighbors of the 0-th point.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere(theta_resolution=10)
        >>> mesh.point_neighbors(0)
        [2, 226, 198, 170, 142, 114, 86, 254, 58, 30]

        Plot them.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, show_edges=True)
        >>>
        >>> # Label the 0-th point
        >>> _ = pl.add_point_labels(
        ...     mesh.points[0], ['0'], text_color='blue', font_size=40
        ... )
        >>>
        >>> # Get the point neighbors and plot them
        >>> neighbors = mesh.point_neighbors(0)
        >>> _ = pl.add_point_labels(
        ...     mesh.points[neighbors],
        ...     labels=[f'{i}' for i in neighbors],
        ...     text_color='red',
        ...     font_size=40,
        ... )
        >>> pl.camera_position = 'xy'
        >>> pl.camera.zoom(7.0)
        >>> pl.show()

        """
        if ind + 1 > self.n_points:
            msg = f'Invalid index {ind} for a dataset with {self.n_points} points.'
            raise IndexError(msg)

        out = []
        for cell in self.point_cell_ids(ind):
            out.extend([i for i in self.get_cell(cell).point_ids if i != ind])
        return list(set(out))

    def point_neighbors_levels(
        self: Self,
        ind: int,
        n_levels: int = 1,
    ) -> Generator[list[int], None, None]:
        """Get consecutive levels of point neighbors.

        Parameters
        ----------
        ind : int
            Point ID.

        n_levels : int, default: 1
            Number of levels to search for point neighbors.
            When equal to 1, it is equivalent to :func:`pyvista.DataSet.point_neighbors`.

        Returns
        -------
        generator[list[[int]]
            A generator of list of neighbor points IDs for the ind-th point.

        See Also
        --------
        pyvista.DataSet.point_neighbors

        Examples
        --------
        Get the point neighbors IDs starting from the 0-th point
        up until the third level.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere(theta_resolution=10)
        >>> pt_nbr_levels = mesh.point_neighbors_levels(0, 3)
        >>> pt_nbr_levels = list(pt_nbr_levels)
        >>> pt_nbr_levels[0]
        [2, 226, 198, 170, 142, 114, 86, 30, 58, 254]
        >>> pt_nbr_levels[1]
        [3, 227, 255, 199, 171, 143, 115, 87, 59, 31]
        >>> pt_nbr_levels[2]
        [256, 32, 4, 228, 200, 172, 144, 116, 88, 60]

        Visualize these points IDs.

        >>> from functools import partial
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, show_edges=True)
        >>>
        >>> # Define partial function to add point labels
        >>> add_point_labels = partial(
        ...     pl.add_point_labels,
        ...     text_color='white',
        ...     font_size=40,
        ...     point_size=10,
        ... )
        >>>
        >>> # Add the first point label
        >>> _ = add_point_labels(mesh.points[0], labels=['0'], text_color='blue')
        >>>
        >>> # Add the neighbors to the plot
        >>> neighbors = mesh.point_neighbors_levels(0, n_levels=3)
        >>> for i, ids in enumerate(neighbors, start=1):
        ...     _ = add_point_labels(
        ...         mesh.points[ids],
        ...         labels=[f'{i}'] * len(ids),
        ...         text_color='red',
        ...     )
        >>>
        >>> pl.view_xy()
        >>> pl.camera.zoom(4.0)
        >>> pl.show()

        """
        method = self.point_neighbors
        return self._get_levels_neihgbors(ind, n_levels, method)

    def cell_neighbors_levels(
        self: Self,
        ind: int,
        connections: str = 'points',
        n_levels: int = 1,
    ) -> Generator[list[int], None, None]:
        """Get consecutive levels of cell neighbors.

        Parameters
        ----------
        ind : int
            Cell ID.

        connections : str, default: "points"
            Describe how the neighbor cell(s) must be connected to the current
            cell to be considered as a neighbor.
            Can be either ``'points'``, ``'edges'`` or ``'faces'``.

        n_levels : int, default: 1
            Number of levels to search for cell neighbors.
            When equal to 1, it is equivalent to :func:`pyvista.DataSet.cell_neighbors`.

        Returns
        -------
        generator[list[int]]
            A generator of list of cell IDs for each level.

        Warnings
        --------
        For a :class:`pyvista.ExplicitStructuredGrid`, use
        :func:`pyvista.ExplicitStructuredGrid.neighbors`.

        See Also
        --------
        pyvista.DataSet.cell_neighbors

        Examples
        --------
        Get the cell neighbors IDs starting from the 0-th cell
        up until the third level.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere(theta_resolution=10)
        >>> nbr_levels = mesh.cell_neighbors_levels(0, connections='edges', n_levels=3)
        >>> nbr_levels = list(nbr_levels)
        >>> nbr_levels[0]
        [1, 21, 9]
        >>> nbr_levels[1]
        [2, 8, 74, 75, 20, 507]
        >>> nbr_levels[2]
        [128, 129, 3, 453, 7, 77, 23, 506]

        Visualize these cells IDs.

        >>> from functools import partial
        >>> pv.global_theme.color_cycler = [
        ...     'red',
        ...     'green',
        ...     'blue',
        ...     'purple',
        ... ]
        >>> pl = pv.Plotter()
        >>>
        >>> # Define partial function to add point labels
        >>> add_point_labels = partial(
        ...     pl.add_point_labels,
        ...     text_color='white',
        ...     font_size=40,
        ...     shape=None,
        ...     show_points=False,
        ... )
        >>>
        >>> # Add the 0-th cell to the plotter
        >>> cell = mesh.extract_cells(0)
        >>> _ = pl.add_mesh(cell, show_edges=True)
        >>> _ = add_point_labels(cell.cell_centers().points, labels=['0'])
        >>> other_ids = [0]
        >>>
        >>> # Add the neighbors to the plot
        >>> neighbors = mesh.cell_neighbors_levels(0, connections='edges', n_levels=3)
        >>> for i, ids in enumerate(neighbors, start=1):
        ...     cells = mesh.extract_cells(ids)
        ...     _ = pl.add_mesh(cells, show_edges=True)
        ...     _ = add_point_labels(
        ...         cells.cell_centers().points, labels=[f'{i}'] * len(ids)
        ...     )
        ...     other_ids.extend(ids)
        >>>
        >>> # Add the cell IDs that are not neighbors (ie. the rest of the sphere)
        >>> cells = mesh.extract_cells(other_ids, invert=True)
        >>> _ = pl.add_mesh(cells, color='white', show_edges=True)
        >>>
        >>> pl.view_xy()
        >>> pl.camera.zoom(6.0)
        >>> pl.show()

        """
        method = partial(self.cell_neighbors, connections=connections)
        return self._get_levels_neihgbors(ind, n_levels, method)

    def _get_levels_neihgbors(
        self: Self,
        ind: int,
        n_levels: int,
        method: Callable[[Any], Any],
    ) -> Generator[list[int], None, None]:  # numpydoc ignore=PR01,RT01
        """Provide helper method that yields neighbors ids."""
        neighbors = set(method(ind))
        yield list(neighbors)

        # Keep track of visited points or cells
        all_visited = neighbors.copy()
        all_visited.add(ind)

        for _ in range(n_levels - 1):
            # Get the neighbors for the next level.
            new_visited = set()
            for n in neighbors:
                new_neighbors = method(n)
                new_visited.update(new_neighbors)
            neighbors = new_visited

            # Only return the ones that have not been visited yet
            yield list(neighbors.difference(all_visited))
            all_visited.update(neighbors)

    def point_cell_ids(self: Self, ind: int) -> list[int]:
        """Get the cell IDs that use the ind-th point.

        Implements :vtk:`vtkDataSet.GetPointCells`.

        Parameters
        ----------
        ind : int
            Point ID.

        Returns
        -------
        list[int]
            List of cell IDs using the ind-th point.

        Examples
        --------
        Get the cell ids using the 0-th point.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere(theta_resolution=10)
        >>> mesh.point_cell_ids(0)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        Plot them.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, show_edges=True)
        >>>
        >>> # Label the 0-th point
        >>> _ = pl.add_point_labels(
        ...     mesh.points[0], ['0'], text_color='blue', font_size=20
        ... )
        >>>
        >>> # Get the cells ids using the 0-th point
        >>> ids = mesh.point_cell_ids(0)
        >>> cells = mesh.extract_cells(ids)
        >>> _ = pl.add_mesh(cells, color='red', show_edges=True)
        >>> centers = cells.cell_centers().points
        >>> _ = pl.add_point_labels(
        ...     centers,
        ...     labels=[f'{i}' for i in ids],
        ...     text_color='white',
        ...     font_size=20,
        ...     shape=None,
        ...     show_points=False,
        ... )
        >>>
        >>> # Plot the other cells
        >>> others = mesh.extract_cells(
        ...     [i for i in range(mesh.n_cells) if i not in ids]
        ... )
        >>> _ = pl.add_mesh(others, show_edges=True)
        >>>
        >>> pl.camera_position = 'xy'
        >>> pl.camera.zoom(7.0)
        >>> pl.show()

        """
        # Build links as recommended:
        # https://vtk.org/doc/nightly/html/classvtkPolyData.html#adf9caaa01f72972d9a986ba997af0ac7
        if hasattr(self, 'BuildLinks'):
            self.BuildLinks()

        ids = _vtk.vtkIdList()
        self.GetPointCells(ind, ids)
        out = [ids.GetId(i) for i in range(ids.GetNumberOfIds())]
        if (9, 4, 0) <= pyvista.vtk_version_info < (9, 5, 0):
            # Need to reverse the order
            return out[::-1]
        return out

    def point_is_inside_cell(
        self: Self,
        ind: int,
        point: VectorLike[float] | MatrixLike[float],
    ) -> bool | NumpyArray[np.bool_]:
        """Return whether one or more points are inside a cell.

        .. versionadded:: 0.35.0

        Parameters
        ----------
        ind : int
            Cell ID.

        point : VectorLike[float] | MatrixLike[float]
            Point or points to query if are inside a cell.

        Returns
        -------
        bool or numpy.ndarray
            Whether point(s) is/are inside cell. A single bool is only returned if
            the input point has shape ``(3,)``.

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_hexbeam()
        >>> mesh.get_cell(0).bounds
        BoundsTuple(x_min = 0.0,
                    x_max = 0.5,
                    y_min = 0.0,
                    y_max = 0.5,
                    z_min = 0.0,
                    z_max = 0.5)
        >>> mesh.point_is_inside_cell(0, [0.2, 0.2, 0.2])
        True

        """
        if not isinstance(ind, (int, np.integer)):
            msg = f'ind must be an int, got {type(ind)}'  # type: ignore[unreachable]
            raise TypeError(msg)

        if not 0 <= ind < self.n_cells:
            msg = f'ind must be >= 0 and < {self.n_cells}, got {ind}'
            raise ValueError(msg)

        co_point, singular = _coerce_pointslike_arg(point, copy=False)

        cell = self.GetCell(ind)
        npoints = cell.GetPoints().GetNumberOfPoints()

        closest_point = [0.0, 0.0, 0.0]
        sub_id = _vtk.mutable(0)
        pcoords = [0.0, 0.0, 0.0]
        dist2 = _vtk.mutable(0.0)
        weights = [0.0] * npoints

        in_cell = np.empty(shape=co_point.shape[0], dtype=np.bool_)
        for i, node in enumerate(co_point):
            is_inside = cell.EvaluatePosition(node, closest_point, sub_id, pcoords, dist2, weights)
            if not 0 <= is_inside <= 1:
                msg = f'Computational difficulty encountered for point {node} in cell {ind}'
                raise RuntimeError(msg)
            in_cell[i] = bool(is_inside)

        if singular:
            return in_cell[0].item()
        return in_cell

    @property
    def active_texture_coordinates(self: Self) -> pyvista_ndarray | None:
        """Return the active texture coordinates on the points.

        Returns
        -------
        Optional[pyvista_ndarray]
            Active texture coordinates on the points.

        Examples
        --------
        Return the active texture coordinates from the globe example.

        >>> from pyvista import examples
        >>> globe = examples.load_globe()
        >>> globe.active_texture_coordinates
        pyvista_ndarray([[0.        , 0.        ],
                         [0.        , 0.07142857],
                         [0.        , 0.14285714],
                         ...,
                         [1.        , 0.85714286],
                         [1.        , 0.92857143],
                         [1.        , 1.        ]], shape=(540, 2))

        """
        return self.point_data.active_texture_coordinates

    @active_texture_coordinates.setter
    def active_texture_coordinates(
        self: Self,
        texture_coordinates: NumpyArray[float],
    ) -> None:
        """Set the active texture coordinates on the points.

        Parameters
        ----------
        texture_coordinates : np.ndarray
            Active texture coordinates on the points.

        """
        self.point_data.active_texture_coordinates = texture_coordinates

    @property
    def is_empty(self) -> bool:  # numpydoc ignore=RT01
        """Return ``True`` if there are no points.

        .. versionadded:: 0.45

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.PolyData()
        >>> mesh.is_empty
        True

        >>> mesh = pv.Sphere()
        >>> mesh.is_empty
        False

        """
        return self.n_points == 0
