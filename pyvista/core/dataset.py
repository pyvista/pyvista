"""Attributes common to PolyData and Grid Objects."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import Sequence
from typing import cast
import warnings

import numpy as np

import pyvista

from . import _vtk_core as _vtk
from ._typing_core import BoundsLike
from ._typing_core import MatrixLike
from ._typing_core import Number
from ._typing_core import NumpyArray
from ._typing_core import VectorLike
from .dataobject import DataObject
from .datasetattributes import DataSetAttributes
from .errors import PyVistaDeprecationWarning
from .errors import VTKVersionError
from .filters import DataSetFilters
from .filters import _get_output
from .pyvista_ndarray import pyvista_ndarray
from .utilities import transformations
from .utilities.arrays import FieldAssociation
from .utilities.arrays import _coerce_pointslike_arg
from .utilities.arrays import get_array
from .utilities.arrays import get_array_association
from .utilities.arrays import raise_not_matching
from .utilities.arrays import vtk_id_list_to_array
from .utilities.helpers import is_pyvista_dataset
from .utilities.misc import abstract_class
from .utilities.misc import check_valid_vector
from .utilities.points import vtk_points

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from collections.abc import Generator
    from collections.abc import Iterator

# vector array names
DEFAULT_VECTOR_KEY = '_vectors'


class ActiveArrayInfoTuple(NamedTuple):
    """Active array info tuple to provide legacy support."""

    association: FieldAssociation
    name: str


class ActiveArrayInfo:
    """Active array info class with support for pickling.

    Parameters
    ----------
    association : pyvista.core.utilities.arrays.FieldAssociation
        Array association.
        Association of the array.

    name : str
        The name of the array.
    """

    def __init__(self, association, name):
        """Initialize."""
        self.association = association
        self.name = name

    def copy(self) -> ActiveArrayInfo:
        """Return a copy of this object.

        Returns
        -------
        ActiveArrayInfo
            A copy of this object.

        """
        return ActiveArrayInfo(self.association, self.name)

    def __getstate__(self):
        """Support pickling."""
        state = self.__dict__.copy()
        state['association'] = int(self.association.value)
        return state

    def __setstate__(self, state):
        """Support unpickling."""
        self.__dict__ = state.copy()
        self.association = FieldAssociation(state['association'])

    @property
    def _namedtuple(self):
        """Build a namedtuple on the fly to provide legacy support."""
        return ActiveArrayInfoTuple(self.association, self.name)

    def __iter__(self):
        """Provide namedtuple-like __iter__."""
        return self._namedtuple.__iter__()

    def __repr__(self):
        """Provide namedtuple-like __repr__."""
        return self._namedtuple.__repr__()

    def __getitem__(self, item):
        """Provide namedtuple-like __getitem__."""
        return self._namedtuple.__getitem__(item)

    def __setitem__(self, key, value):
        """Provide namedtuple-like __setitem__."""
        self._namedtuple.__setitem__(key, value)

    def __getattr__(self, item):
        """Provide namedtuple-like __getattr__."""
        self._namedtuple.__getattr__(item)

    def __eq__(self, other):
        """Check equivalence (useful for serialize/deserialize tests)."""
        same_association = int(self.association.value) == int(other.association.value)
        return self.name == other.name and same_association


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

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the common object."""
        super().__init__(*args, **kwargs)
        self._last_active_scalars_name: str | None = None
        self._active_scalars_info = ActiveArrayInfo(FieldAssociation.POINT, name=None)
        self._active_vectors_info = ActiveArrayInfo(FieldAssociation.POINT, name=None)
        self._active_tensors_info = ActiveArrayInfo(FieldAssociation.POINT, name=None)

    def __getattr__(self, item) -> Any:
        """Get attribute from base class if not found."""
        return super().__getattribute__(item)

    @property
    def active_scalars_info(self) -> ActiveArrayInfo:
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
            self._active_scalars_info = ActiveArrayInfo(field, None)
            for attr in [self.point_data, self.cell_data]:
                if attr.active_scalars_name is not None:
                    self._active_scalars_info = ActiveArrayInfo(
                        attr.association,
                        attr.active_scalars_name,
                    )
                    break

        return self._active_scalars_info

    @property
    def active_vectors_info(self) -> ActiveArrayInfo:
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
            self._active_vectors_info = ActiveArrayInfo(field, None)
            for attr in [self.point_data, self.cell_data]:
                name = attr.active_vectors_name
                if name is not None:
                    self._active_vectors_info = ActiveArrayInfo(attr.association, name)
                    break

        return self._active_vectors_info

    @property
    def active_tensors_info(self) -> ActiveArrayInfo:
        """Return the active tensor's field and name: [field, name].

        Returns
        -------
        ActiveArrayInfo
            Active tensor's field and name: [field, name].

        """
        return self._active_tensors_info

    @property
    def active_vectors(self) -> pyvista_ndarray | None:
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
    def active_tensors(self) -> NumpyArray[float] | None:
        """Return the active tensors array.

        Returns
        -------
        Optional[np.ndarray]
            Active tensors array.

        """
        field, name = self.active_tensors_info
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
    def active_tensors_name(self) -> str:
        """Return the name of the active tensor array.

        Returns
        -------
        str
            Name of the active tensor array.

        """
        return self.active_tensors_info.name

    @active_tensors_name.setter
    def active_tensors_name(self, name: str):  # numpydoc ignore=GL08
        """Set the name of the active tensor array.

        Parameters
        ----------
        name : str
            Name of the active tensor array.

        """
        self.set_active_tensors(name)

    @property
    def active_vectors_name(self) -> str:
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
    def active_vectors_name(self, name: str):  # numpydoc ignore=GL08
        """Set the name of the active vectors array.

        Parameters
        ----------
        name : str
            Name of the active vectors array.

        """
        self.set_active_vectors(name)

    @property  # type: ignore[explicit-override, override]
    def active_scalars_name(self) -> str:
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
    def active_scalars_name(self, name: str):  # numpydoc ignore=GL08
        """Set the name of the active scalars.

        Parameters
        ----------
        name : str
             Name of the active scalars.

        """
        self.set_active_scalars(name)

    @property
    def points(self) -> pyvista_ndarray:
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
            vtkpts = vtk_points(np.empty((0, 3)), False)
            self.SetPoints(vtkpts)
            _points = self.GetPoints().GetData()
        return pyvista_ndarray(_points, dataset=self)

    @points.setter
    def points(self, points: MatrixLike[float] | _vtk.vtkPoints):  # numpydoc ignore=GL08
        """Set a reference to the points as a numpy object.

        Parameters
        ----------
        points : MatrixLike[float] | vtk.vtkPoints
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
        vtkpts = vtk_points(points, False)
        if not pdata:
            self.SetPoints(vtkpts)
        else:
            pdata.SetData(vtkpts.GetData())
        self.GetPoints().Modified()
        self.Modified()

    @property
    def arrows(self) -> pyvista.PolyData | None:
        """Return a glyph representation of the active vector data as arrows.

        Arrows will be located at the points of the mesh and
        their size will be dependent on the norm of the vector.
        Their direction will be the "direction" of the vector

        Returns
        -------
        pyvista.PolyData
            Active vectors represented as arrows.

        Examples
        --------
        Create a mesh, compute the normals and set them active, and
        plot the active vectors.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> mesh_w_normals = mesh.compute_normals()
        >>> mesh_w_normals.active_vectors_name = 'Normals'
        >>> arrows = mesh_w_normals.arrows
        >>> arrows.plot(show_scalar_bar=False)

        """
        vectors, vectors_name = self.active_vectors, self.active_vectors_name
        if vectors is None or vectors_name is None:
            return None

        if vectors.ndim != 2:
            raise ValueError('Active vectors are not vectors.')

        scale_name = f'{vectors_name} Magnitude'
        scale = np.linalg.norm(vectors, axis=1)
        self.point_data.set_array(scale, scale_name)
        return self.glyph(orient=vectors_name, scale=scale_name)

    @property
    def active_t_coords(self) -> pyvista_ndarray | None:
        """Return the active texture coordinates on the points.

        Returns
        -------
        Optional[pyvista_ndarray]
            Active texture coordinates on the points.

        """
        warnings.warn(
            "Use of `DataSet.active_t_coords` is deprecated. Use `DataSet.active_texture_coordinates` instead.",
            PyVistaDeprecationWarning,
        )
        return self.active_texture_coordinates

    @active_t_coords.setter
    def active_t_coords(self, t_coords: NumpyArray[float]):  # numpydoc ignore=GL08
        """Set the active texture coordinates on the points.

        Parameters
        ----------
        t_coords : np.ndarray
            Active texture coordinates on the points.
        """
        warnings.warn(
            "Use of `DataSet.active_t_coords` is deprecated. Use `DataSet.active_texture_coordinates` instead.",
            PyVistaDeprecationWarning,
        )
        self.active_texture_coordinates = t_coords  # type: ignore[assignment]

    def set_active_scalars(
        self,
        name: str | None,
        preference='cell',
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
        if preference not in ['point', 'cell', FieldAssociation.CELL, FieldAssociation.POINT]:
            raise ValueError('``preference`` must be either "point" or "cell"')
        if name is None:
            self.GetCellData().SetActiveScalars(None)
            self.GetPointData().SetActiveScalars(None)
            return FieldAssociation.NONE, np.array([])
        field = get_array_association(self, name, preference=preference)
        if field == FieldAssociation.NONE:
            if name in self.field_data:
                raise ValueError(f'Data named "{name}" is a field array which cannot be active.')
            else:
                raise KeyError(f'Data named "{name}" does not exist in this dataset.')
        self._last_active_scalars_name = self.active_scalars_info.name
        if field == FieldAssociation.POINT:
            ret = self.GetPointData().SetActiveScalars(name)
        elif field == FieldAssociation.CELL:
            ret = self.GetCellData().SetActiveScalars(name)
        else:
            raise ValueError(f'Data field ({name}) with type ({field}) not usable')

        if ret < 0:
            raise ValueError(
                f'Data field "{name}" with type ({field}) could not be set as the active scalars',
            )

        self._active_scalars_info = ActiveArrayInfo(field, name)

        if field == FieldAssociation.POINT:
            return field, self.point_data.active_scalars
        else:  # must be cell
            return field, self.cell_data.active_scalars

    def set_active_vectors(self, name: str | None, preference: str = 'point') -> None:
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
                raise ValueError(f'Data field ({name}) with type ({field}) not usable')

            if ret < 0:
                raise ValueError(
                    f'Data field ({name}) with type ({field}) could not be set as the active vectors',
                )

        self._active_vectors_info = ActiveArrayInfo(field, name)

    def set_active_tensors(self, name: str | None, preference: str = 'point') -> None:
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
                raise ValueError(f'Data field ({name}) with type ({field}) not usable')

            if ret < 0:
                raise ValueError(
                    f'Data field ({name}) with type ({field}) could not be set as the active tensors',
                )

        self._active_tensors_info = ActiveArrayInfo(field, name)

    def rename_array(self, old_name: str, new_name: str, preference='cell') -> None:
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
            raise KeyError(f'Array with name {old_name} not found.')

        arr = data.pop(old_name)
        # Update the array's name before reassigning. This prevents taking a copy of the array
        # in `DataSetAttributes._prepare_array` which can lead to the array being garbage collected.
        # See issue #5244.
        arr.VTKObject.SetName(new_name)
        data[new_name] = arr

        if was_active and field != FieldAssociation.NONE:
            self.set_active_scalars(new_name, preference=field)

    @property
    def active_scalars(self) -> pyvista_ndarray | None:
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
    def active_normals(self) -> pyvista_ndarray | None:
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

    def get_data_range(
        self,
        arr_var: str | NumpyArray[float] | None = None,
        preference='cell',
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
            _, arr_var = self.active_scalars_info
            if arr_var is None:
                return (np.nan, np.nan)

        if isinstance(arr_var, str):
            name = arr_var
            arr = get_array(self, name, preference=preference, err=True)
        else:
            arr = arr_var

        # If array has no tuples return a NaN range
        if arr is None:
            return (np.nan, np.nan)
        if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
            return (np.nan, np.nan)
        # Use the array range
        return np.nanmin(arr), np.nanmax(arr)

    def rotate_x(
        self,
        angle: float,
        point: VectorLike[float] = (0.0, 0.0, 0.0),
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ):
        """Rotate mesh about the x-axis.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
            under the hood.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the x-axis.

        point : Vector, default: (0.0, 0.0, 0.0)
            Point to rotate about. Defaults to origin.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Rotated dataset.

        Examples
        --------
        Rotate a mesh 30 degrees about the x-axis.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> rot = mesh.rotate_x(30, inplace=False)

        Plot the rotated mesh.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(rot)
        >>> _ = pl.add_mesh(mesh, style='wireframe', line_width=3)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        """
        check_valid_vector(point, "point")
        t = transformations.axis_angle_rotation((1, 0, 0), angle, point=point, deg=True)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    def rotate_y(
        self,
        angle: float,
        point: VectorLike[float] = (0.0, 0.0, 0.0),
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ):
        """Rotate mesh about the y-axis.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
            under the hood.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the y-axis.

        point : Vector, default: (0.0, 0.0, 0.0)
            Point to rotate about.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are transformed. Otherwise, only
            the points, normals and active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Rotated dataset.

        Examples
        --------
        Rotate a cube 30 degrees about the y-axis.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> rot = mesh.rotate_y(30, inplace=False)

        Plot the rotated mesh.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(rot)
        >>> _ = pl.add_mesh(mesh, style='wireframe', line_width=3)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        """
        check_valid_vector(point, "point")
        t = transformations.axis_angle_rotation((0, 1, 0), angle, point=point, deg=True)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    def rotate_z(
        self,
        angle: float,
        point: VectorLike[float] = (0.0, 0.0, 0.0),
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ):
        """Rotate mesh about the z-axis.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
            under the hood.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the z-axis.

        point : Vector, default: (0.0, 0.0, 0.0)
            Point to rotate about.  Defaults to origin.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Rotated dataset.

        Examples
        --------
        Rotate a mesh 30 degrees about the z-axis.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> rot = mesh.rotate_z(30, inplace=False)

        Plot the rotated mesh.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(rot)
        >>> _ = pl.add_mesh(mesh, style='wireframe', line_width=3)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        """
        check_valid_vector(point, "point")
        t = transformations.axis_angle_rotation((0, 0, 1), angle, point=point, deg=True)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    def rotate_vector(
        self,
        vector: VectorLike[float],
        angle: float,
        point: VectorLike[float] = (0.0, 0.0, 0.0),
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ):
        """Rotate mesh about a vector.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
            under the hood.

        Parameters
        ----------
        vector : Vector
            Vector to rotate about.

        angle : float
            Angle to rotate.

        point : Vector, default: (0.0, 0.0, 0.0)
            Point to rotate about. Defaults to origin.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Rotated dataset.

        Examples
        --------
        Rotate a mesh 30 degrees about the ``(1, 1, 1)`` axis.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> rot = mesh.rotate_vector((1, 1, 1), 30, inplace=False)

        Plot the rotated mesh.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(rot)
        >>> _ = pl.add_mesh(mesh, style='wireframe', line_width=3)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        """
        check_valid_vector(vector)
        check_valid_vector(point, "point")
        t = transformations.axis_angle_rotation(vector, angle, point=point, deg=True)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    def translate(
        self,
        xyz: VectorLike[float],
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ):
        """Translate the mesh.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
            under the hood.

        Parameters
        ----------
        xyz : Vector
            A vector of three floats.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Translated dataset.

        Examples
        --------
        Create a sphere and translate it by ``(2, 1, 2)``.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mesh.center
        [0.0, 0.0, 0.0]
        >>> trans = mesh.translate((2, 1, 2), inplace=False)
        >>> trans.center
        [2.0, 1.0, 2.0]

        """
        transform = _vtk.vtkTransform()
        transform.Translate(cast(Sequence[float], xyz))
        return self.transform(
            transform,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    def scale(
        self,
        xyz: Number | VectorLike[float],
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ):
        """Scale the mesh.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
            under the hood.

        Parameters
        ----------
        xyz : Number | Vector
            A vector sequence defining the scale factors along x, y, and z. If
            a scalar, the same uniform scale is used along all three axes.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are transformed. Otherwise, only
            the points, normals and active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Scaled dataset.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> pl.show_axes()
        >>> _ = pl.show_grid()
        >>> mesh1 = examples.download_teapot()
        >>> _ = pl.add_mesh(mesh1)
        >>> pl.subplot(0, 1)
        >>> pl.show_axes()
        >>> _ = pl.show_grid()
        >>> mesh2 = mesh1.scale([10.0, 10.0, 10.0], inplace=False)
        >>> _ = pl.add_mesh(mesh2)
        >>> pl.show(cpos="xy")

        """
        if isinstance(xyz, (float, int, np.number)):
            xyz = [xyz] * 3

        transform = _vtk.vtkTransform()
        transform.Scale(xyz)
        return self.transform(
            transform,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    def flip_x(
        self,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ):
        """Flip mesh about the x-axis.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
            under the hood.

        Parameters
        ----------
        point : sequence[float], optional
            Point to rotate about.  Defaults to center of mesh at
            :attr:`center <pyvista.DataSet.center>`.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Flipped dataset.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> pl.show_axes()
        >>> mesh1 = examples.download_teapot()
        >>> _ = pl.add_mesh(mesh1)
        >>> pl.subplot(0, 1)
        >>> pl.show_axes()
        >>> mesh2 = mesh1.flip_x(inplace=False)
        >>> _ = pl.add_mesh(mesh2)
        >>> pl.show(cpos="xy")

        """
        if point is None:
            point = self.center
        check_valid_vector(point, 'point')
        t = transformations.reflection((1, 0, 0), point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    def flip_y(
        self,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ):
        """Flip mesh about the y-axis.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
            under the hood.

        Parameters
        ----------
        point : sequence[float], optional
            Point to rotate about.  Defaults to center of mesh at
            :attr:`center <pyvista.DataSet.center>`.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Flipped dataset.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> pl.show_axes()
        >>> mesh1 = examples.download_teapot()
        >>> _ = pl.add_mesh(mesh1)
        >>> pl.subplot(0, 1)
        >>> pl.show_axes()
        >>> mesh2 = mesh1.flip_y(inplace=False)
        >>> _ = pl.add_mesh(mesh2)
        >>> pl.show(cpos="xy")

        """
        if point is None:
            point = self.center
        check_valid_vector(point, 'point')
        t = transformations.reflection((0, 1, 0), point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    def flip_z(
        self,
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ):
        """Flip mesh about the z-axis.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
            under the hood.

        Parameters
        ----------
        point : Vector, optional
            Point to rotate about.  Defaults to center of mesh at
            :attr:`center <pyvista.DataSet.center>`.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Flipped dataset.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> pl.show_axes()
        >>> mesh1 = examples.download_teapot().rotate_x(90, inplace=False)
        >>> _ = pl.add_mesh(mesh1)
        >>> pl.subplot(0, 1)
        >>> pl.show_axes()
        >>> mesh2 = mesh1.flip_z(inplace=False)
        >>> _ = pl.add_mesh(mesh2)
        >>> pl.show(cpos="xz")

        """
        if point is None:
            point = self.center
        check_valid_vector(point, 'point')
        t = transformations.reflection((0, 0, 1), point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    def flip_normal(
        self,
        normal: VectorLike[float],
        point: VectorLike[float] | None = None,
        transform_all_input_vectors: bool = False,
        inplace: bool = False,
    ):
        """Flip mesh about the normal.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
            under the hood.

        Parameters
        ----------
        normal : sequence[float]
           Normal vector to flip about.

        point : sequence[float]
            Point to rotate about.  Defaults to center of mesh at
            :attr:`center <pyvista.DataSet.center>`.

        transform_all_input_vectors : bool, default: False
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, default: False
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Dataset flipped about its normal.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> pl.show_axes()
        >>> mesh1 = examples.download_teapot()
        >>> _ = pl.add_mesh(mesh1)
        >>> pl.subplot(0, 1)
        >>> pl.show_axes()
        >>> mesh2 = mesh1.flip_normal([1.0, 1.0, 1.0], inplace=False)
        >>> _ = pl.add_mesh(mesh2)
        >>> pl.show(cpos="xy")

        """
        if point is None:
            point = self.center
        check_valid_vector(normal, 'normal')
        check_valid_vector(point, 'point')
        t = transformations.reflection(normal, point=point)
        return self.transform(
            t,
            transform_all_input_vectors=transform_all_input_vectors,
            inplace=inplace,
        )

    def copy_meta_from(self, ido: DataSet, deep: bool = True) -> None:
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
    def point_data(self) -> DataSetAttributes:
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
        >>> mesh.point_data['my_array'] = np.random.default_rng().random(
        ...     mesh.n_points
        ... )
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

    def clear_point_data(self) -> None:
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

    def clear_cell_data(self) -> None:
        """Remove all cell arrays."""
        self.cell_data.clear()

    def clear_data(self) -> None:
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
    def cell_data(self) -> DataSetAttributes:
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
        >>> mesh.cell_data['my_array'] = np.random.default_rng().random(
        ...     mesh.n_cells
        ... )
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
    def n_points(self) -> int:
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
    def n_cells(self) -> int:
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
    def number_of_points(self) -> int:  # pragma: no cover
        """Return the number of points.

        Returns
        -------
        int :
             Number of points.

        """
        return self.GetNumberOfPoints()

    @property
    def number_of_cells(self) -> int:  # pragma: no cover
        """Return the number of cells.

        Returns
        -------
        int :
             Number of cells.

        """
        return self.GetNumberOfCells()

    @property
    def bounds(self) -> BoundsLike:
        """Return the bounding box of this dataset.

        Returns
        -------
        BoundsLike
            Bounding box of this dataset.
            The form is: ``(xmin, xmax, ymin, ymax, zmin, zmax)``.

        Examples
        --------
        Create a cube and return the bounds of the mesh.

        >>> import pyvista as pv
        >>> cube = pv.Cube()
        >>> cube.bounds
        (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)

        """
        return cast(BoundsLike, self.GetBounds())

    @property
    def length(self) -> float:
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
    def center(self) -> list[float]:
        """Return the center of the bounding box.

        Returns
        -------
        Vector
            Center of the bounding box.

        Examples
        --------
        Get the center of a mesh.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere(center=(1, 2, 0))
        >>> mesh.center
        [1.0, 2.0, 0.0]

        """
        return list(self.GetCenter())

    @property
    def volume(self) -> float:
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
    def area(self) -> float:
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
        self,
        name: str,
        preference: Literal['cell', 'point', 'field'] = 'cell',
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
        self,
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

    def __getitem__(self, index: Iterable[Any] | str) -> NumpyArray[float]:
        """Search both point, cell, and field data for an array."""
        if isinstance(index, Iterable) and not isinstance(index, str):
            name, preference = tuple(index)
        elif isinstance(index, str):
            name = index
            preference = 'cell'
        else:
            raise KeyError(
                f'Index ({index}) not understood.'
                ' Index must be a string name or a tuple of string name and string preference.',
            )
        return self.get_array(name, preference=preference)

    def _ipython_key_completions_(self) -> list[str]:
        """Tab completion of IPython."""
        return self.array_names

    def __setitem__(
        self,
        name: str,
        scalars: NumpyArray[float] | Sequence[float] | float,
    ):  # numpydoc ignore=PR01,RT01
        """Add/set an array in the point_data, or cell_data accordingly.

        It depends on the array's length, or specified mode.

        """
        # First check points - think of case with vertex cells
        #   there would be the same number of cells as points but we'd want
        #   the data to be on the nodes.
        if scalars is None:
            raise TypeError('Empty array unable to be added.')
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
    def n_arrays(self) -> int:
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
    def array_names(self) -> list[str]:
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
        names = []
        names.extend(self.field_data.keys())
        names.extend(self.point_data.keys())
        names.extend(self.cell_data.keys())
        try:
            names.remove(self.active_scalars_name)
            names.insert(0, self.active_scalars_name)
        except ValueError:
            pass
        return names

    def _get_attrs(self):
        """Return the representation methods (internal helper)."""
        attrs = []
        attrs.append(("N Cells", self.GetNumberOfCells(), "{}"))
        attrs.append(("N Points", self.GetNumberOfPoints(), "{}"))
        if isinstance(self, pyvista.PolyData):
            attrs.append(("N Strips", self.n_strips, "{}"))
        bds = self.bounds
        fmt = f"{pyvista.FLOAT_FORMAT}, {pyvista.FLOAT_FORMAT}"
        attrs.append(("X Bounds", (bds[0], bds[1]), fmt))
        attrs.append(("Y Bounds", (bds[2], bds[3]), fmt))
        attrs.append(("Z Bounds", (bds[4], bds[5]), fmt))
        # if self.n_cells <= pyvista.REPR_VOLUME_MAX_CELLS and self.n_cells > 0:
        #     attrs.append(("Volume", (self.volume), pyvista.FLOAT_FORMAT))
        return attrs

    def _repr_html_(self) -> str:
        """Return a pretty representation for Jupyter notebooks.

        It includes header details and information about all arrays.

        """
        fmt = ""
        if self.n_arrays > 0:
            fmt += "<table style='width: 100%;'>"
            fmt += "<tr><th>Header</th><th>Data Arrays</th></tr>"
            fmt += "<tr><td>"
        # Get the header info
        fmt += self.head(display=False, html=True)
        # Fill out arrays
        if self.n_arrays > 0:
            fmt += "</td><td>"
            fmt += "\n"
            fmt += "<table style='width: 100%;'>\n"
            titles = ["Name", "Field", "Type", "N Comp", "Min", "Max"]
            fmt += "<tr>" + "".join([f"<th>{t}</th>" for t in titles]) + "</tr>\n"
            row = "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>\n"
            row = "<tr>" + "".join(["<td>{}</td>" for i in range(len(titles))]) + "</tr>\n"

            def format_array(name, arr, field):
                """Format array information for printing (internal helper)."""
                dl, dh = self.get_data_range(arr)
                dl = pyvista.FLOAT_FORMAT.format(dl)
                dh = pyvista.FLOAT_FORMAT.format(dh)
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

            fmt += "</table>\n"
            fmt += "\n"
            fmt += "</td></tr> </table>"
        return fmt

    def __repr__(self) -> str:
        """Return the object representation."""
        return self.head(display=False, html=False)

    def __str__(self) -> str:
        """Return the object string representation."""
        return self.head(display=False, html=False)

    def copy_from(self, mesh: _vtk.vtkDataSet, deep: bool = True) -> None:
        """Overwrite this dataset inplace with the new dataset's geometries and data.

        Parameters
        ----------
        mesh : vtk.vtkDataSet
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
            raise TypeError(
                f'The Input DataSet type {type(mesh)} must be '
                f'compatible with the one being overwritten {type(self)}',
            )
        if deep:
            self.deep_copy(mesh)
        else:
            self.shallow_copy(mesh)
        if is_pyvista_dataset(mesh):
            self.copy_meta_from(mesh, deep=deep)

    def cast_to_unstructured_grid(self) -> pyvista.UnstructuredGrid:
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

    def cast_to_pointset(self, pass_cell_data: bool = False) -> pyvista.PointSet:
        """Extract the points of this dataset and return a :class:`pyvista.PointSet`.

        Parameters
        ----------
        pass_cell_data : bool, default: False
            Run the :func:`cell_data_to_point_data()
            <pyvista.DataSetFilters.cell_data_to_point_data>` filter and pass
            cell data fields to the new pointset.

        Returns
        -------
        pyvista.PointSet
            Dataset cast into a :class:`pyvista.PointSet`.

        Notes
        -----
        This will produce a deep copy of the points and point/cell data of
        the original mesh.

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
        if pass_cell_data:
            self = self.cell_data_to_point_data()
        pset.GetPointData().DeepCopy(self.GetPointData())
        pset.active_scalars_name = self.active_scalars_name
        return pset

    def cast_to_poly_points(self, pass_cell_data: bool = False) -> pyvista.PolyData:
        """Extract the points of this dataset and return a :class:`pyvista.PolyData`.

        Parameters
        ----------
        pass_cell_data : bool, default: False
            Run the :func:`cell_data_to_point_data()
            <pyvista.DataSetFilters.cell_data_to_point_data>` filter and pass
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

    def find_closest_point(self, point: Iterable[float], n=1) -> int:
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
            raise TypeError("Given point must be a length three sequence.")
        if not isinstance(n, int):
            raise TypeError("`n` must be a positive integer.")
        if n < 1:
            raise ValueError("`n` must be a positive integer.")

        locator = _vtk.vtkPointLocator()
        locator.SetDataSet(self)
        locator.BuildLocator()
        if n > 1:
            id_list = _vtk.vtkIdList()
            locator.FindClosestNPoints(n, point, id_list)
            return vtk_id_list_to_array(id_list)
        return locator.FindClosestPoint(point)

    def find_closest_cell(
        self,
        point: VectorLike[float] | MatrixLike[float],
        return_closest_point: bool = False,
    ) -> int | NumpyArray[int] | tuple[int | NumpyArray[int], NumpyArray[int]]:
        """Find index of closest cell in this mesh to the given point.

        Parameters
        ----------
        point : Vector | Matrix
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

            locator.FindClosestPoint(node, closest_point, cell, cell_id, sub_id, dist2)
            closest_cells.append(int(cell_id))
            closest_points.append(closest_point)

        out_cells: int | NumpyArray[int] = closest_cells[0] if singular else np.array(closest_cells)
        out_points = np.array(closest_points[0]) if singular else np.array(closest_points)

        if return_closest_point:
            return out_cells, out_points
        return out_cells

    def find_containing_cell(
        self,
        point: VectorLike[float] | MatrixLike[float],
    ) -> int | NumpyArray[int]:
        """Find index of a cell that contains the given point.

        Parameters
        ----------
        point : Vector, Matrix
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
        >>> mesh = pv.ImageData(
        ...     dimensions=[5, 5, 1], spacing=[1 / 4, 1 / 4, 0]
        ... )
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
        self,
        pointa: VectorLike[float],
        pointb: VectorLike[float],
        tolerance: float = 0.0,
    ) -> NumpyArray[int]:
        """Find the index of cells whose bounds intersect a line.

        Line is defined from ``pointa`` to ``pointb``.

        Parameters
        ----------
        pointa : Vector
            Length 3 coordinate of the start of the line.

        pointb : Vector
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
            raise TypeError("Point A must be a length three tuple of floats.")
        if np.array(pointb).size != 3:
            raise TypeError("Point B must be a length three tuple of floats.")
        locator = _vtk.vtkCellLocator()
        locator.SetDataSet(self)
        locator.BuildLocator()
        id_list = _vtk.vtkIdList()
        locator.FindCellsAlongLine(
            cast(Sequence[float], pointa),
            cast(Sequence[float], pointb),
            tolerance,
            id_list,
        )
        return vtk_id_list_to_array(id_list)

    def find_cells_intersecting_line(
        self,
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
        if pyvista.vtk_version_info < (9, 2, 0):
            raise VTKVersionError("pyvista.PointSet requires VTK >= 9.2.0")

        if np.array(pointa).size != 3:
            raise TypeError("Point A must be a length three tuple of floats.")
        if np.array(pointb).size != 3:
            raise TypeError("Point B must be a length three tuple of floats.")
        locator = _vtk.vtkCellLocator()
        locator.SetDataSet(cast(_vtk.vtkDataSet, self))
        locator.BuildLocator()
        id_list = _vtk.vtkIdList()
        points = _vtk.vtkPoints()
        cell = _vtk.vtkGenericCell()
        locator.IntersectWithLine(
            cast(Sequence[float], pointa),
            cast(Sequence[float], pointb),
            tolerance,
            points,
            id_list,
            cell,
        )
        return vtk_id_list_to_array(id_list)

    def find_cells_within_bounds(self, bounds: BoundsLike) -> NumpyArray[int]:
        """Find the index of cells in this mesh within bounds.

        Parameters
        ----------
        bounds : sequence[float]
            Bounding box. The form is: ``[xmin, xmax, ymin, ymax, zmin, zmax]``.

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
        >>> index = mesh.find_cells_within_bounds(
        ...     [-2.0, 2.0, -2.0, 2.0, -2.0, 2.0]
        ... )

        """
        if np.array(bounds).size != 6:
            raise TypeError("Bounds must be a length six tuple of floats.")
        locator = _vtk.vtkCellTreeLocator()
        locator.SetDataSet(cast(_vtk.vtkDataSet, self))
        locator.BuildLocator()
        id_list = _vtk.vtkIdList()
        locator.FindCellsWithinBounds(list(bounds), id_list)
        return vtk_id_list_to_array(id_list)

    def get_cell(self, index: int) -> pyvista.Cell:
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
            raise IndexError(f'Invalid index {index} for a dataset with {self.n_cells} cells.')

        # Note: we have to use vtkGenericCell here since
        # GetCell(vtkIdType cellId, vtkGenericCell* cell) is thread-safe,
        # while GetCell(vtkIdType cellId) is not.
        cell = pyvista.Cell()
        self.GetCell(index, cell)
        cell.SetCellType(self.GetCellType(index))
        return cell

    @property
    def cell(self) -> Iterator[pyvista.Cell]:
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
        ...
        """
        for i in range(self.n_cells):
            yield self.get_cell(i)

    def cell_neighbors(self, ind: int, connections: str = "points") -> list[int]:
        """Get the cell neighbors of the ind-th cell.

        Concrete implementation of vtkDataSet's `GetCellNeighbors
        <https://vtk.org/doc/nightly/html/classvtkDataSet.html#ae1ba413c15802ef50d9b1955a66521e4>`_.

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
        For a :class:`pyvista.ExplicitStructuredGrid`, use :func:`pyvista.ExplicitStructuredGrid.neighbors`.

        See Also
        --------
        pyvista.DataSet.cell_neighbors_levels

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()

        Get the neighbor cell ids that have at least one point in common with
        the 0-th cell.

        >>> mesh.cell_neighbors(0, "points")
        [1, 2, 3, 388, 389, 11, 12, 395, 14, 209, 211, 212]

        Get the neighbor cell ids that have at least one edge in common with
        the 0-th cell.

        >>> mesh.cell_neighbors(0, "edges")
        [1, 3, 12]

        For unstructured grids with cells of dimension 3 (Tetrahedron for example),
        cell neighbors can be defined using faces.

        >>> mesh = examples.download_tetrahedron()
        >>> mesh.cell_neighbors(0, "faces")
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
        ...     text_color="white",
        ...     font_size=20,
        ...     shape=None,
        ...     show_points=False,
        ... )
        >>>
        >>> for i, connection in enumerate(["points", "edges"]):
        ...     pl.subplot(0, i)
        ...     pl.view_xy()
        ...     _ = pl.add_title(
        ...         f"{connection.capitalize()} neighbors",
        ...         color="red",
        ...         shadow=True,
        ...         font_size=8,
        ...     )
        ...
        ...     # Add current cell
        ...     i_cell = 0
        ...     current_cell = mesh.extract_cells(i_cell)
        ...     _ = pl.add_mesh(
        ...         current_cell, show_edges=True, color="blue"
        ...     )
        ...     _ = add_point_labels(
        ...         current_cell.cell_centers().points,
        ...         labels=[f"{i_cell}"],
        ...     )
        ...
        ...     # Add neighbors
        ...     ids = mesh.cell_neighbors(i_cell, connection)
        ...     cells = mesh.extract_cells(ids)
        ...     _ = pl.add_mesh(cells, color="red", show_edges=True)
        ...     _ = add_point_labels(
        ...         cells.cell_centers().points,
        ...         labels=[f"{i}" for i in ids],
        ...     )
        ...
        ...     # Add other cells
        ...     ids.append(i_cell)
        ...     others = mesh.extract_cells(ids, invert=True)
        ...     _ = pl.add_mesh(others, show_edges=True)
        ...
        >>> pl.show()
        """
        if isinstance(self, _vtk.vtkExplicitStructuredGrid):
            raise TypeError("For an ExplicitStructuredGrid, use the `neighbors` method")

        # Build links as recommended:
        # https://vtk.org/doc/nightly/html/classvtkPolyData.html#adf9caaa01f72972d9a986ba997af0ac7
        if hasattr(self, "BuildLinks"):
            self.BuildLinks()

        needed = ["points", "edges", "faces"]
        if connections not in needed:
            raise ValueError(f'`connections` must be one of: {needed} (got "{connections}")')

        cell = self.get_cell(ind)

        iterators = {
            "points": cell.point_ids,
            "edges": range(cell.n_edges),
            "faces": range(cell.n_faces),
        }

        def generate_ids(i: int, connections: str):  # numpydoc ignore=GL08
            if connections == "points":
                ids = _vtk.vtkIdList()
                ids.InsertNextId(i)
                return ids
            elif connections == "edges":
                return cell.get_edge(i).GetPointIds()
            elif connections == "faces":
                return cell.get_face(i).GetPointIds()
            return None  # pragma: no cover

        neighbors = set()
        for i in iterators[connections]:
            point_ids = generate_ids(i, connections)
            cell_ids = _vtk.vtkIdList()
            self.GetCellNeighbors(ind, point_ids, cell_ids)

            neighbors.update([cell_ids.GetId(i) for i in range(cell_ids.GetNumberOfIds())])

        return list(neighbors)

    def point_neighbors(self, ind: int) -> list[int]:
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
        ...     mesh.points[0], ["0"], text_color="blue", font_size=40
        ... )
        >>>
        >>> # Get the point neighbors and plot them
        >>> neighbors = mesh.point_neighbors(0)
        >>> _ = pl.add_point_labels(
        ...     mesh.points[neighbors],
        ...     labels=[f"{i}" for i in neighbors],
        ...     text_color="red",
        ...     font_size=40,
        ... )
        >>> pl.camera_position = "xy"
        >>> pl.camera.zoom(7.0)
        >>> pl.show()

        """
        if ind + 1 > self.n_points:
            raise IndexError(f'Invalid index {ind} for a dataset with {self.n_points} points.')

        out = []
        for cell in self.point_cell_ids(ind):
            out.extend([i for i in self.get_cell(cell).point_ids if i != ind])
        return list(set(out))

    def point_neighbors_levels(
        self,
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
        ...     text_color="white",
        ...     font_size=40,
        ...     point_size=10,
        ... )
        >>>
        >>> # Add the first point label
        >>> _ = add_point_labels(
        ...     mesh.points[0], labels=["0"], text_color="blue"
        ... )
        >>>
        >>> # Add the neighbors to the plot
        >>> neighbors = mesh.point_neighbors_levels(0, n_levels=3)
        >>> for i, ids in enumerate(neighbors, start=1):
        ...     _ = add_point_labels(
        ...         mesh.points[ids],
        ...         labels=[f"{i}"] * len(ids),
        ...         text_color="red",
        ...     )
        ...
        >>>
        >>> pl.view_xy()
        >>> pl.camera.zoom(4.0)
        >>> pl.show()
        """
        method = self.point_neighbors
        return self._get_levels_neihgbors(ind, n_levels, method)

    def cell_neighbors_levels(
        self,
        ind: int,
        connections: str = "points",
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
        For a :class:`pyvista.ExplicitStructuredGrid`, use :func:`pyvista.ExplicitStructuredGrid.neighbors`.

        See Also
        --------
        pyvista.DataSet.cell_neighbors

        Examples
        --------
        Get the cell neighbors IDs starting from the 0-th cell
        up until the third level.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere(theta_resolution=10)
        >>> nbr_levels = mesh.cell_neighbors_levels(
        ...     0, connections="edges", n_levels=3
        ... )
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
        ...     text_color="white",
        ...     font_size=40,
        ...     shape=None,
        ...     show_points=False,
        ... )
        >>>
        >>> # Add the 0-th cell to the plotter
        >>> cell = mesh.extract_cells(0)
        >>> _ = pl.add_mesh(cell, show_edges=True)
        >>> _ = add_point_labels(cell.cell_centers().points, labels=["0"])
        >>> other_ids = [0]
        >>>
        >>> # Add the neighbors to the plot
        >>> neighbors = mesh.cell_neighbors_levels(
        ...     0, connections="edges", n_levels=3
        ... )
        >>> for i, ids in enumerate(neighbors, start=1):
        ...     cells = mesh.extract_cells(ids)
        ...     _ = pl.add_mesh(cells, show_edges=True)
        ...     _ = add_point_labels(
        ...         cells.cell_centers().points, labels=[f"{i}"] * len(ids)
        ...     )
        ...     other_ids.extend(ids)
        ...
        >>>
        >>> # Add the cell IDs that are not neighbors (ie. the rest of the sphere)
        >>> cells = mesh.extract_cells(other_ids, invert=True)
        >>> _ = pl.add_mesh(cells, color="white", show_edges=True)
        >>>
        >>> pl.view_xy()
        >>> pl.camera.zoom(6.0)
        >>> pl.show()
        """
        method = partial(self.cell_neighbors, connections=connections)
        return self._get_levels_neihgbors(ind, n_levels, method)

    def _get_levels_neihgbors(
        self,
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

    def point_cell_ids(self, ind: int) -> list[int]:
        """Get the cell IDs that use the ind-th point.

        Implements vtkDataSet's `GetPointCells <https://vtk.org/doc/nightly/html/classvtkDataSet.html#a36d1d8f67ad67adf4d1a9cfb30dade49>`_.

        Parameters
        ----------
        ind : int
            Point ID.

        Returns
        -------
        listint]
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
        ...     mesh.points[0], ["0"], text_color="blue", font_size=20
        ... )
        >>>
        >>> # Get the cells ids using the 0-th point
        >>> ids = mesh.point_cell_ids(0)
        >>> cells = mesh.extract_cells(ids)
        >>> _ = pl.add_mesh(cells, color="red", show_edges=True)
        >>> centers = cells.cell_centers().points
        >>> _ = pl.add_point_labels(
        ...     centers,
        ...     labels=[f"{i}" for i in ids],
        ...     text_color="white",
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
        >>> pl.camera_position = "yx"
        >>> pl.camera.zoom(7.0)
        >>> pl.show()
        """
        # Build links as recommended:
        # https://vtk.org/doc/nightly/html/classvtkPolyData.html#adf9caaa01f72972d9a986ba997af0ac7
        if hasattr(self, "BuildLinks"):
            self.BuildLinks()

        ids = _vtk.vtkIdList()
        self.GetPointCells(ind, ids)
        return [ids.GetId(i) for i in range(ids.GetNumberOfIds())]

    def point_is_inside_cell(
        self,
        ind: int,
        point: VectorLike[float] | MatrixLike[float],
    ) -> bool | NumpyArray[np.bool_]:
        """Return whether one or more points are inside a cell.

        .. versionadded:: 0.35.0

        Parameters
        ----------
        ind : int
            Cell ID.

        point : Matrix
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
        (0.0, 0.5, 0.0, 0.5, 0.0, 0.5)
        >>> mesh.point_is_inside_cell(0, [0.2, 0.2, 0.2])
        True

        """
        if not isinstance(ind, (int, np.integer)):
            raise TypeError(f"ind must be an int, got {type(ind)}")

        if not 0 <= ind < self.n_cells:
            raise ValueError(f"ind must be >= 0 and < {self.n_cells}, got {ind}")

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
                raise RuntimeError(
                    f"Computational difficulty encountered for point {node} in cell {ind}",
                )
            in_cell[i] = bool(is_inside)

        if singular:
            return in_cell[0].item()
        return in_cell

    @property
    def active_texture_coordinates(self) -> pyvista_ndarray | None:
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
                         [1.        , 1.        ]])

        """
        return self.point_data.active_texture_coordinates

    @active_texture_coordinates.setter
    def active_texture_coordinates(
        self,
        texture_coordinates: NumpyArray[float],
    ):  # numpydoc ignore=GL08
        """Set the active texture coordinates on the points.

        Parameters
        ----------
        texture_coordinates : np.ndarray
            Active texture coordinates on the points.
        """
        self.point_data.active_texture_coordinates = texture_coordinates  # type: ignore[assignment]
