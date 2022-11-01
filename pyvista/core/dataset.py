"""Attributes common to PolyData and Grid Objects."""

import collections.abc
from copy import deepcopy
import logging
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import warnings

if sys.version_info >= (3, 8):
    from typing import Literal
else:  # pragma: no cover
    from typing_extensions import Literal

import numpy as np

import pyvista
from pyvista import _vtk
from pyvista.utilities import (
    FieldAssociation,
    abstract_class,
    get_array,
    get_array_association,
    is_pyvista_dataset,
    raise_not_matching,
    transformations,
    vtk_id_list_to_array,
)
from pyvista.utilities.arrays import _coerce_pointslike_arg
from pyvista.utilities.errors import check_valid_vector
from pyvista.utilities.misc import PyVistaDeprecationWarning

from .._typing import Number, NumericArray, Vector, VectorArray
from .dataobject import DataObject
from .datasetattributes import DataSetAttributes
from .filters import DataSetFilters, _get_output
from .pyvista_ndarray import pyvista_ndarray

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')

# vector array names
DEFAULT_VECTOR_KEY = '_vectors'
ActiveArrayInfoTuple = collections.namedtuple('ActiveArrayInfoTuple', ['association', 'name'])


class ActiveArrayInfo:
    """Active array info class with support for pickling."""

    def __init__(self, association, name):
        """Initialize."""
        self.association = association
        self.name = name

    def copy(self):
        """Return a copy of this object."""
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
    """Methods in common to spatially referenced objects."""

    # Simply bind pyvista.plotting.plot to the object
    plot = pyvista.plot

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the common object."""
        super().__init__()
        self._last_active_scalars_name: Optional[str] = None
        self._active_scalars_info = ActiveArrayInfo(FieldAssociation.POINT, name=None)
        self._active_vectors_info = ActiveArrayInfo(FieldAssociation.POINT, name=None)
        self._active_tensors_info = ActiveArrayInfo(FieldAssociation.POINT, name=None)
        self._textures: Dict[str, _vtk.vtkTexture] = {}

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

        >>> import pyvista
        >>> mesh = pyvista.Sphere()
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
                        attr.association, attr.active_scalars_name
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

        >>> import pyvista
        >>> mesh = pyvista.Sphere()
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
        """Return the active tensor's field and name: [field, name]."""
        return self._active_tensors_info

    @property
    def active_vectors(self) -> Optional[pyvista_ndarray]:
        """Return the active vectors array.

        Examples
        --------
        Create a mesh, compute the normals inplace, and return the
        normals vector array.

        >>> import pyvista
        >>> mesh = pyvista.Sphere()
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
    def active_tensors(self) -> Optional[np.ndarray]:
        """Return the active tensors array."""
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
        """Return the name of the active tensor array."""
        return self.active_tensors_info.name

    @active_tensors_name.setter
    def active_tensors_name(self, name: str):
        self.set_active_tensors(name)

    @property
    def active_vectors_name(self) -> str:
        """Return the name of the active vectors array.

        Examples
        --------
        Create a mesh, compute the normals, set them as active, and
        return the name of the active vectors.

        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh_w_normals = mesh.compute_normals()
        >>> mesh_w_normals.active_vectors_name = 'Normals'
        >>> mesh_w_normals.active_vectors_name
        'Normals'

        """
        return self.active_vectors_info.name

    @active_vectors_name.setter
    def active_vectors_name(self, name: str):
        self.set_active_vectors(name)

    @property  # type: ignore
    def active_scalars_name(self) -> str:  # type: ignore
        """Return the name of the active scalars.

        Examples
        --------
        Create a mesh, add scalars to the mesh, and return the name of
        the active scalars.

        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh['Z Height'] = mesh.points[:, 2]
        >>> mesh.active_scalars_name
        'Z Height'

        """
        return self.active_scalars_info.name

    @active_scalars_name.setter
    def active_scalars_name(self, name: str):
        self.set_active_scalars(name)

    @property
    def points(self) -> pyvista_ndarray:
        """Return a reference to the points as a numpy object.

        Examples
        --------
        Create a mesh and return the points of the mesh as a numpy
        array.

        >>> import pyvista
        >>> cube = pyvista.Cube()
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

        >>> cube.points[...] = 2*points
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
            vtk_points = pyvista.vtk_points(np.empty((0, 3)), False)
            self.SetPoints(vtk_points)
            _points = self.GetPoints().GetData()
        return pyvista_ndarray(_points, dataset=self)

    @points.setter
    def points(self, points: Union[VectorArray, NumericArray, _vtk.vtkPoints]):
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
        vtk_points = pyvista.vtk_points(points, False)
        if not pdata:
            self.SetPoints(vtk_points)
        else:
            pdata.SetData(vtk_points.GetData())
        self.GetPoints().Modified()
        self.Modified()

    @property
    def arrows(self) -> Optional['pyvista.PolyData']:
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

        >>> import pyvista
        >>> mesh = pyvista.Cube()
        >>> mesh_w_normals = mesh.compute_normals()
        >>> mesh_w_normals.active_vectors_name = 'Normals'
        >>> arrows = mesh_w_normals.arrows
        >>> arrows.plot(show_scalar_bar=False)

        """
        vectors_name = self.active_vectors_name
        if vectors_name is None:
            return

        if self.active_vectors.ndim != 2:  # type: ignore
            raise ValueError('Active vectors are not vectors.')

        scale_name = f'{vectors_name} Magnitude'
        scale = np.linalg.norm(self.active_vectors, axis=1)  # type: ignore
        self.point_data.set_array(scale, scale_name)
        return self.glyph(orient=vectors_name, scale=scale_name)

    @property
    def vectors(self) -> Optional[pyvista_ndarray]:  # pragma: no cover
        """Return active vectors.

        .. deprecated:: 0.32.0
           Use of `DataSet.vectors` to return vector data is deprecated.

        """
        warnings.warn(
            "Use of `DataSet.vectors` is deprecated. Use `DataSet.active_vectors` instead.",
            PyVistaDeprecationWarning,
        )
        return self.active_vectors

    @vectors.setter
    def vectors(self, array: np.ndarray):  # pragma: no cover
        warnings.warn(
            "Use of `DataSet.vectors` to add vector data is deprecated. "
            "Use `DataSet['vector_name'] = data`. "
            "Use `DataSet.active_vectors_name = 'vector_name' to make active.",
            PyVistaDeprecationWarning,
        )
        if array.ndim != 2:
            raise ValueError('vector array must be a 2-dimensional array')
        elif array.shape[1] != 3:
            raise ValueError('vector array must be 3D')
        elif array.shape[0] != self.n_points:
            raise ValueError('Number of vectors be the same as the number of points')

        self.point_data[DEFAULT_VECTOR_KEY] = array
        self.active_vectors_name = DEFAULT_VECTOR_KEY

    @property
    def t_coords(self) -> Optional[pyvista_ndarray]:  # pragma: no cover
        """Return the active texture coordinates on the points.

        .. deprecated:: 0.32.0
            Use :attr:`DataSet.active_t_coords` to return the active
            texture coordinates.

        """
        warnings.warn(
            "Use of `DataSet.t_coords` is deprecated. Use `DataSet.active_t_coords` instead.",
            PyVistaDeprecationWarning,
        )
        return self.active_t_coords

    @t_coords.setter
    def t_coords(self, t_coords: np.ndarray):  # pragma: no cover
        warnings.warn(
            "Use of `DataSet.t_coords` is deprecated. Use `DataSet.active_t_coords` instead.",
            PyVistaDeprecationWarning,
        )
        self.active_t_coords = t_coords  # type: ignore

    @property
    def active_t_coords(self) -> Optional[pyvista_ndarray]:
        """Return or set the active texture coordinates on the points.

        Examples
        --------
        Return the active texture coordinates from the globe example.

        >>> from pyvista import examples
        >>> globe = examples.load_globe()
        >>> globe.active_t_coords
        pyvista_ndarray([[0.        , 0.        ],
                         [0.        , 0.07142857],
                         [0.        , 0.14285714],
                         ...,
                         [1.        , 0.85714286],
                         [1.        , 0.92857143],
                         [1.        , 1.        ]])

        """
        return self.point_data.active_t_coords

    @active_t_coords.setter
    def active_t_coords(self, t_coords: np.ndarray):
        self.point_data.active_t_coords = t_coords  # type: ignore

    @property
    def textures(self) -> Dict[str, _vtk.vtkTexture]:
        """Return a dictionary to hold compatible ``vtk.vtkTexture`` objects.

        When casting back to a VTK dataset or filtering this dataset,
        these textures will not be passed.

        Examples
        --------
        Return the active texture datasets from the globe example.

        >>> from pyvista import examples
        >>> globe = examples.load_globe()
        >>> globe.textures
        {'2k_earth_daymap': ...}

        """
        return self._textures

    def clear_textures(self):
        """Clear the textures from this mesh.

        Examples
        --------
        Clear the texture from the globe example.

        >>> from pyvista import examples
        >>> globe = examples.load_globe()
        >>> globe.textures
        {'2k_earth_daymap': ...}
        >>> globe.clear_textures()
        >>> globe.textures
        {}

        """
        self._textures.clear()

    def _activate_texture(mesh, name: str) -> _vtk.vtkTexture:
        """Grab a texture and update the active texture coordinates.

        This makes sure to not destroy old texture coordinates.

        Parameters
        ----------
        name : str
            The name of the texture and texture coordinates to activate

        Returns
        -------
        vtk.vtkTexture
            The active texture

        """
        if name is True or isinstance(name, int):
            keys = list(mesh.textures.keys())
            # Grab the first name available if True
            idx = 0 if not isinstance(name, int) or name is True else name
            if idx > len(keys):  # is this necessary?
                idx = 0
            try:
                name = keys[idx]
            except IndexError:
                logging.warning('No textures associated with input mesh.')
                return None
        # Grab the texture object by name
        try:
            texture = mesh.textures[name]
        except KeyError:
            logging.warning(f'Texture ({name}) not associated with this dataset')
            texture = None
        else:
            # Be sure to reset the tcoords if present
            # Grab old coordinates
            if name in mesh.array_names:
                old_tcoord = mesh.GetPointData().GetTCoords()
                mesh.GetPointData().SetTCoords(mesh.GetPointData().GetAbstractArray(name))
                mesh.GetPointData().AddArray(old_tcoord)
                mesh.Modified()
        return texture

    def set_active_scalars(self, name: Optional[str], preference='cell'):
        """Find the scalars by name and appropriately sets it as active.

        To deactivate any active scalars, pass ``None`` as the ``name``.

        Parameters
        ----------
        name : str or None
            Name of the scalars array to assign as active.  If
            ``None``, deactivates active scalars for both point and
            cell data.

        preference : str, optional
            If there are two arrays of the same name associated with
            points or cells, it will prioritize an array matching this
            type.  Can be either ``'cell'`` or ``'point'``.

        Returns
        -------
        pyvista.FieldAssociation
            Association of the scalars matching ``name``.

        numpy.ndarray
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
                f'Data field "{name}" with type ({field}) could not be set as the active scalars'
            )

        self._active_scalars_info = ActiveArrayInfo(field, name)

        if field == FieldAssociation.POINT:
            return field, self.point_data.active_scalars
        else:  # must be cell
            return field, self.cell_data.active_scalars

    def set_active_vectors(self, name: Optional[str], preference='point'):
        """Find the vectors by name and appropriately sets it as active.

        To deactivate any active vectors, pass ``None`` as the ``name``.

        Parameters
        ----------
        name : str
            Name of the vectors array to assign as active.

        preference : str, optional
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
                    f'Data field ({name}) with type ({field}) could not be set as the active vectors'
                )

        self._active_vectors_info = ActiveArrayInfo(field, name)

    def set_active_tensors(self, name: Optional[str], preference='point'):
        """Find the tensors by name and appropriately sets it as active.

        To deactivate any active tensors, pass ``None`` as the ``name``.

        Parameters
        ----------
        name : str
            Name of the tensors array to assign as active.

        preference : str, optional
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
                    f'Data field ({name}) with type ({field}) could not be set as the active tensors'
                )

        self._active_tensors_info = ActiveArrayInfo(field, name)

    def rename_array(self, old_name: str, new_name: str, preference='cell'):
        """Change array name by searching for the array then renaming it.

        Parameters
        ----------
        old_name : str
            Name of the array to rename.

        new_name : str
            Name to rename the array to.

        preference : str, optional
            If there are two arrays of the same name associated with
            points, cells, or field data, it will prioritize an array
            matching this type.  Can be either ``'cell'``,
            ``'field'``, or ``'point'``.

        Examples
        --------
        Create a cube, assign a point array to the mesh named
        ``'my_array'``, and rename it to ``'my_renamed_array'``.

        >>> import pyvista
        >>> import numpy as np
        >>> cube = pyvista.Cube()
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
            self.point_data[new_name] = self.point_data.pop(old_name)
        elif field == FieldAssociation.CELL:
            self.cell_data[new_name] = self.cell_data.pop(old_name)
        elif field == FieldAssociation.NONE:
            self.field_data[new_name] = self.field_data.pop(old_name)
        else:
            raise KeyError(f'Array with name {old_name} not found.')
        if was_active and field != FieldAssociation.NONE:
            self.set_active_scalars(new_name, preference=field)

    @property
    def active_scalars(self) -> Optional[pyvista_ndarray]:
        """Return the active scalars as an array."""
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
    def active_normals(self) -> Optional[pyvista_ndarray]:
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

        >>> import pyvista
        >>> mesh = pyvista.Sphere()
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
        self, arr_var: Optional[Union[str, np.ndarray]] = None, preference='cell'
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Get the non-NaN min and max of a named array.

        Parameters
        ----------
        arr_var : str, np.ndarray, optional
            The name of the array to get the range. If ``None``, the
            active scalars is used.

        preference : str, optional
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
        self, angle: float, point=(0.0, 0.0, 0.0), transform_all_input_vectors=False, inplace=False
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

        point : list, optional
            Point to rotate about.  Defaults to origin ``(0.0, 0.0, 0.0)``.

        transform_all_input_vectors : bool, optional
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Rotated dataset.

        Examples
        --------
        Rotate a mesh 30 degrees about the x-axis.

        >>> import pyvista
        >>> mesh = pyvista.Cube()
        >>> rot = mesh.rotate_x(30, inplace=False)

        Plot the rotated mesh.

        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(rot)
        >>> _ = pl.add_mesh(mesh, style='wireframe', line_width=3)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        """
        check_valid_vector(point, "point")
        t = transformations.axis_angle_rotation((1, 0, 0), angle, point=point, deg=True)
        return self.transform(
            t, transform_all_input_vectors=transform_all_input_vectors, inplace=inplace
        )

    def rotate_y(
        self, angle: float, point=(0.0, 0.0, 0.0), transform_all_input_vectors=False, inplace=False
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

        point : float, optional
            Point to rotate about.

        transform_all_input_vectors : bool, optional
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Rotated dataset.

        Examples
        --------
        Rotate a cube 30 degrees about the y-axis.

        >>> import pyvista
        >>> mesh = pyvista.Cube()
        >>> rot = mesh.rotate_y(30, inplace=False)

        Plot the rotated mesh.

        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(rot)
        >>> _ = pl.add_mesh(mesh, style='wireframe', line_width=3)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        """
        check_valid_vector(point, "point")
        t = transformations.axis_angle_rotation((0, 1, 0), angle, point=point, deg=True)
        return self.transform(
            t, transform_all_input_vectors=transform_all_input_vectors, inplace=inplace
        )

    def rotate_z(
        self, angle: float, point=(0.0, 0.0, 0.0), transform_all_input_vectors=False, inplace=False
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

        point : list, optional
            Point to rotate about.  Defaults to origin ``(0.0, 0.0, 0.0)``.

        transform_all_input_vectors : bool, optional
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Rotated dataset.

        Examples
        --------
        Rotate a mesh 30 degrees about the z-axis.

        >>> import pyvista
        >>> mesh = pyvista.Cube()
        >>> rot = mesh.rotate_z(30, inplace=False)

        Plot the rotated mesh.

        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(rot)
        >>> _ = pl.add_mesh(mesh, style='wireframe', line_width=3)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        """
        check_valid_vector(point, "point")
        t = transformations.axis_angle_rotation((0, 0, 1), angle, point=point, deg=True)
        return self.transform(
            t, transform_all_input_vectors=transform_all_input_vectors, inplace=inplace
        )

    def rotate_vector(
        self,
        vector: Iterable[float],
        angle: float,
        point=(0.0, 0.0, 0.0),
        transform_all_input_vectors=False,
        inplace=False,
    ):
        """Rotate mesh about a vector.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
            under the hood.

        Parameters
        ----------
        vector : Iterable
            Axes to rotate about.

        angle : float
            Angle in degrees to rotate about the vector.

        point : list, optional
            Point to rotate about.  Defaults to origin ``(0.0, 0.0, 0.0)``.

        transform_all_input_vectors : bool, optional
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Rotated dataset.

        Examples
        --------
        Rotate a mesh 30 degrees about the ``(1, 1, 1)`` axis.

        >>> import pyvista
        >>> mesh = pyvista.Cube()
        >>> rot = mesh.rotate_vector((1, 1, 1), 30, inplace=False)

        Plot the rotated mesh.

        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(rot)
        >>> _ = pl.add_mesh(mesh, style='wireframe', line_width=3)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        """
        check_valid_vector(vector)
        check_valid_vector(point, "point")
        t = transformations.axis_angle_rotation(vector, angle, point=point, deg=True)
        return self.transform(
            t, transform_all_input_vectors=transform_all_input_vectors, inplace=inplace
        )

    def translate(
        self, xyz: Union[list, tuple, np.ndarray], transform_all_input_vectors=False, inplace=False
    ):
        """Translate the mesh.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
            under the hood.

        Parameters
        ----------
        xyz : list or tuple or np.ndarray
            Length 3 list, tuple or array.

        transform_all_input_vectors : bool, optional
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Translated dataset.

        Examples
        --------
        Create a sphere and translate it by ``(2, 1, 2)``.

        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> mesh.center
        [0.0, 0.0, 0.0]
        >>> trans = mesh.translate((2, 1, 2), inplace=False)
        >>> trans.center
        [2.0, 1.0, 2.0]

        """
        transform = _vtk.vtkTransform()
        transform.Translate(xyz)
        return self.transform(
            transform, transform_all_input_vectors=transform_all_input_vectors, inplace=inplace
        )

    def scale(
        self,
        xyz: Union[Number, list, tuple, np.ndarray],
        transform_all_input_vectors=False,
        inplace=False,
    ):
        """Scale the mesh.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
            under the hood.

        Parameters
        ----------
        xyz : float or list or tuple or np.ndarray
            A scalar or length 3 list, tuple or array defining the scale
            factors along x, y, and z. If a scalar, the same uniform scale is
            used along all three axes.

        transform_all_input_vectors : bool, optional
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Scaled dataset.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> pl = pyvista.Plotter(shape=(1, 2))
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
            transform, transform_all_input_vectors=transform_all_input_vectors, inplace=inplace
        )

    def flip_x(self, point=None, transform_all_input_vectors=False, inplace=False):
        """Flip mesh about the x-axis.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
            under the hood.

        Parameters
        ----------
        point : list, optional
            Point to rotate about.  Defaults to center of mesh at
            :attr:`center <pyvista.DataSet.center>`.

        transform_all_input_vectors : bool, optional
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Flipped dataset.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> pl = pyvista.Plotter(shape=(1, 2))
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
            t, transform_all_input_vectors=transform_all_input_vectors, inplace=inplace
        )

    def flip_y(self, point=None, transform_all_input_vectors=False, inplace=False):
        """Flip mesh about the y-axis.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
            under the hood.

        Parameters
        ----------
        point : list, optional
            Point to rotate about.  Defaults to center of mesh at
            :attr:`center <pyvista.DataSet.center>`.

        transform_all_input_vectors : bool, optional
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Flipped dataset.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> pl = pyvista.Plotter(shape=(1, 2))
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
            t, transform_all_input_vectors=transform_all_input_vectors, inplace=inplace
        )

    def flip_z(self, point=None, transform_all_input_vectors=False, inplace=False):
        """Flip mesh about the z-axis.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
            under the hood.

        Parameters
        ----------
        point : list, optional
            Point to rotate about.  Defaults to center of mesh at
            :attr:`center <pyvista.DataSet.center>`.

        transform_all_input_vectors : bool, optional
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Flipped dataset.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> pl = pyvista.Plotter(shape=(1, 2))
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
            t, transform_all_input_vectors=transform_all_input_vectors, inplace=inplace
        )

    def flip_normal(
        self, normal: List[float], point=None, transform_all_input_vectors=False, inplace=False
    ):
        """Flip mesh about the normal.

        .. note::
            See also the notes at :func:`transform()
            <DataSetFilters.transform>` which is used by this filter
            under the hood.

        Parameters
        ----------
        normal : tuple
           Normal vector to flip about.

        point : list, optional
            Point to rotate about.  Defaults to center of mesh at
            :attr:`center <pyvista.DataSet.center>`.

        transform_all_input_vectors : bool, optional
            When ``True``, all input vectors are
            transformed. Otherwise, only the points, normals and
            active vectors are transformed.

        inplace : bool, optional
            Updates mesh in-place.

        Returns
        -------
        pyvista.DataSet
            Dataset flipped about its normal.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> pl = pyvista.Plotter(shape=(1, 2))
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
            t, transform_all_input_vectors=transform_all_input_vectors, inplace=inplace
        )

    def copy_meta_from(self, ido: 'DataSet', deep: bool = True):
        """Copy pyvista meta data onto this object from another object.

        Parameters
        ----------
        ido : pyvista.DataSet
            Dataset to copy the metadata from.

        deep : bool, optional
            Deep or shallow copy.

        """
        self.clear_textures()

        if deep:
            self._association_complex_names = deepcopy(ido._association_complex_names)
            self._association_bitarray_names = deepcopy(ido._association_bitarray_names)
            self._active_scalars_info = ido.active_scalars_info.copy()
            self._active_vectors_info = ido.active_vectors_info.copy()
            self._active_tensors_info = ido.active_tensors_info.copy()
            self._textures = {name: tex.copy() for name, tex in ido.textures.items()}
        else:
            # pass by reference
            self._association_complex_names = ido._association_complex_names
            self._association_bitarray_names = ido._association_bitarray_names
            self._active_scalars_info = ido.active_scalars_info
            self._active_vectors_info = ido.active_vectors_info
            self._active_tensors_info = ido.active_tensors_info
            self._textures = ido.textures

    @property
    def point_arrays(self) -> DataSetAttributes:  # pragma: no cover
        """Return vtkPointData as DataSetAttributes.

        .. deprecated:: 0.32.0
            Use :attr:`DataSet.point_data` to return point data.

        """
        warnings.warn(
            "Use of `point_arrays` is deprecated. Use `point_data` instead.",
            PyVistaDeprecationWarning,
        )
        return self.point_data

    @property
    def point_data(self) -> DataSetAttributes:
        """Return vtkPointData as DataSetAttributes.

        Examples
        --------
        Add point arrays to a mesh and list the available ``point_data``.

        >>> import pyvista
        >>> import numpy as np
        >>> mesh = pyvista.Cube()
        >>> mesh.clear_data()
        >>> mesh.point_data['my_array'] = np.random.random(mesh.n_points)
        >>> mesh.point_data['my_other_array'] = np.arange(mesh.n_points)
        >>> mesh.point_data
        pyvista DataSetAttributes
        Association     : POINT
        Active Scalars  : my_other_array
        Active Vectors  : None
        Active Texture  : None
        Active Normals  : None
        Contains arrays :
            my_array                float64    (8,)
            my_other_array          int64      (8,)                 SCALARS

        Access an array from ``point_data``.

        >>> mesh.point_data['my_other_array']
        pyvista_ndarray([0, 1, 2, 3, 4, 5, 6, 7])

        Or access it directly from the mesh.

        >>> mesh['my_array'].shape
        (8,)

        """
        return DataSetAttributes(
            self.GetPointData(), dataset=self, association=FieldAssociation.POINT
        )

    def clear_point_arrays(self):  # pragma: no cover
        """Remove all point data.

        .. deprecated:: 0.32.0
            Use :func:`DataSet.clear_point_data` instead.

        """
        warnings.warn(
            "Use of `clear_point_arrays` is deprecated. Use `clear_point_data` instead.",
            PyVistaDeprecationWarning,
        )
        self.clear_point_data()

    def clear_point_data(self):
        """Remove all point arrays.

        Examples
        --------
        Clear all point arrays from a mesh.

        >>> import pyvista
        >>> import numpy as np
        >>> mesh = pyvista.Sphere()
        >>> mesh.point_data.keys()
        ['Normals']
        >>> mesh.clear_point_data()
        >>> mesh.point_data.keys()
        []

        """
        self.point_data.clear()

    def clear_cell_arrays(self):  # pragma: no cover
        """Remove all cell data.

        .. deprecated:: 0.32.0
            Use :func:`DataSet.clear_cell_data` instead.

        """
        warnings.warn(
            "Use of `clear_cell_arrays` is deprecated. Use `clear_cell_data` instead.",
            PyVistaDeprecationWarning,
        )
        self.clear_cell_data()

    def clear_cell_data(self):
        """Remove all cell arrays."""
        self.cell_data.clear()

    def clear_arrays(self):  # pragma: no cover
        """Remove all arrays from point/cell/field data.

        .. deprecated:: 0.32.0
            Use :func:`DataSet.clear_data` instead.

        """
        warnings.warn(
            "Use of `clear_arrays` is deprecated. Use `clear_data` instead.",
            PyVistaDeprecationWarning,
        )
        self.clear_data()

    def clear_data(self):
        """Remove all arrays from point/cell/field data.

        Examples
        --------
        Clear all arrays from a mesh.

        >>> import pyvista
        >>> import numpy as np
        >>> mesh = pyvista.Sphere()
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
    def cell_arrays(self) -> DataSetAttributes:  # pragma: no cover
        """Return vtkCellData as DataSetAttributes.

        .. deprecated:: 0.32.0
            Use :attr:`DataSet.cell_data` to return cell data.

        """
        warnings.warn(
            "Use of `cell_arrays` is deprecated. Use `cell_data` instead.",
            PyVistaDeprecationWarning,
        )
        return self.cell_data

    @property
    def cell_data(self) -> DataSetAttributes:
        """Return vtkCellData as DataSetAttributes.

        Examples
        --------
        Add cell arrays to a mesh and list the available ``cell_data``.

        >>> import pyvista
        >>> import numpy as np
        >>> mesh = pyvista.Cube()
        >>> mesh.clear_data()
        >>> mesh.cell_data['my_array'] = np.random.random(mesh.n_cells)
        >>> mesh.cell_data['my_other_array'] = np.arange(mesh.n_cells)
        >>> mesh.cell_data
        pyvista DataSetAttributes
        Association     : CELL
        Active Scalars  : my_other_array
        Active Vectors  : None
        Active Texture  : None
        Active Normals  : None
        Contains arrays :
            my_array                float64    (6,)
            my_other_array          int64      (6,)                 SCALARS

        Access an array from ``cell_data``.

        >>> mesh.cell_data['my_other_array']
        pyvista_ndarray([0, 1, 2, 3, 4, 5])

        Or access it directly from the mesh.

        >>> mesh['my_array'].shape
        (6,)

        """
        return DataSetAttributes(
            self.GetCellData(), dataset=self, association=FieldAssociation.CELL
        )

    @property
    def n_points(self) -> int:
        """Return the number of points in the entire dataset.

        Examples
        --------
        Create a mesh and return the number of points in the
        mesh.

        >>> import pyvista
        >>> cube = pyvista.Cube()
        >>> cube.n_points
        8

        """
        return self.GetNumberOfPoints()

    @property
    def n_cells(self) -> int:
        """Return the number of cells in the entire dataset.

        Notes
        -----
        This is identical to :attr:`n_faces <pyvista.PolyData.n_faces>`
        in :class:`pyvista.PolyData`.

        Examples
        --------
        Create a mesh and return the number of cells in the
        mesh.

        >>> import pyvista
        >>> cube = pyvista.Cube()
        >>> cube.n_cells
        6

        """
        return self.GetNumberOfCells()

    @property
    def number_of_points(self) -> int:  # pragma: no cover
        """Return the number of points."""
        return self.GetNumberOfPoints()

    @property
    def number_of_cells(self) -> int:  # pragma: no cover
        """Return the number of cells."""
        return self.GetNumberOfCells()

    @property
    def bounds(self) -> Tuple[float, float, float, float, float, float]:
        """Return the bounding box of this dataset.

        The form is: ``(xmin, xmax, ymin, ymax, zmin, zmax)``.

        Examples
        --------
        Create a cube and return the bounds of the mesh.

        >>> import pyvista
        >>> cube = pyvista.Cube()
        >>> cube.bounds
        (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)

        """
        return self.GetBounds()

    @property
    def length(self) -> float:
        """Return the length of the diagonal of the bounding box.

        Examples
        --------
        Get the length of the bounding box of a cube.  This should
        match ``3**(1/2)`` since it is the diagonal of a cube that is
        ``1 x 1 x 1``.

        >>> import pyvista
        >>> mesh = pyvista.Cube()
        >>> mesh.length
        1.7320508075688772

        """
        return self.GetLength()

    @property
    def center(self) -> Vector:
        """Return the center of the bounding box.

        Examples
        --------
        Get the center of a mesh.

        >>> import pyvista
        >>> mesh = pyvista.Sphere(center=(1, 2, 0))
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
        >>> mesh = pv.UniformGrid(dimensions=(5, 5, 5))
        >>> mesh.volume
        64.0

        A mesh with 2D cells has no volume.

        >>> mesh = pv.UniformGrid(dimensions=(5, 5, 1))
        >>> mesh.volume
        0.0

        :class:`pyvista.PolyData` is special as a 2D surface can
        enclose a 3D volume.

        >>> mesh = pv.Sphere()
        >>> mesh.volume
        0.51825

        """
        sizes = self.compute_cell_sizes(length=False, area=False, volume=True)
        return sizes.cell_data['Volume'].sum()

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
        >>> mesh = pv.UniformGrid(dimensions=(5, 5, 1))
        >>> mesh.area
        16.0

        A mesh with 3D cells does not have an area.  To get
        the outer surface area, first extract the surface using
        :func:`pyvista.DataSetFilters.extract_surface`.

        >>> mesh = pv.UniformGrid(dimensions=(5, 5, 5))
        >>> mesh.area
        0.0

        Get the area of a sphere.

        >>> mesh = pv.Sphere()
        >>> mesh.volume
        0.51825

        """
        sizes = self.compute_cell_sizes(length=False, area=True, volume=False)
        return sizes.cell_data['Area'].sum()

    def get_array(
        self, name: str, preference: Literal['cell', 'point', 'field'] = 'cell'
    ) -> 'pyvista.pyvista_ndarray':
        """Search both point, cell and field data for an array.

        Parameters
        ----------
        name : str
            Name of the array.

        preference : str, optional
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

        >>> import pyvista
        >>> mesh = pyvista.Cube()
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
        self, name: str, preference: Literal['cell', 'point', 'field'] = 'cell'
    ) -> FieldAssociation:
        """Get the association of an array.

        Parameters
        ----------
        name : str
            Name of the array.

        preference : str, optional
            When ``name`` is specified, this is the preferred array
            association to search for in the dataset.  Must be either
            ``'point'``, ``'cell'``, or ``'field'``.

        Returns
        -------
        pyvista.FieldAssociation
            Field association of the array.

        Examples
        --------
        Create a DataSet with a variety of arrays.

        >>> import pyvista
        >>> mesh = pyvista.Cube()
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

    def __getitem__(self, index: Union[Iterable, str]) -> np.ndarray:
        """Search both point, cell, and field data for an array."""
        if isinstance(index, collections.abc.Iterable) and not isinstance(index, str):
            name, preference = tuple(index)
        elif isinstance(index, str):
            name = index
            preference = 'cell'
        else:
            raise KeyError(
                f'Index ({index}) not understood.'
                ' Index must be a string name or a tuple of string name and string preference.'
            )
        return self.get_array(name, preference=preference)

    def _ipython_key_completions_(self) -> List[str]:
        return self.array_names

    def __setitem__(self, name: str, scalars: np.ndarray):
        """Add/set an array in the point_data, or cell_data accordingly.

        It depends on the array's length, or specified mode.

        """
        # First check points - think of case with vertex cells
        #   there would be the same number of cells as points but we'd want
        #   the data to be on the nodes.
        if scalars is None:
            raise TypeError('Empty array unable to be added.')
        if not isinstance(scalars, np.ndarray):
            scalars = np.array(scalars)
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
        """Return the number of arrays present in the dataset."""
        n = self.GetPointData().GetNumberOfArrays()
        n += self.GetCellData().GetNumberOfArrays()
        n += self.GetFieldData().GetNumberOfArrays()
        return n

    @property
    def array_names(self) -> List[str]:
        """Return a list of array names for the dataset.

        This makes sure to put the active scalars' name first in the list.

        Examples
        --------
        Return the array names for a mesh.

        >>> import pyvista
        >>> mesh = pyvista.Sphere()
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
            fmt += "<table>"
            fmt += "<tr><th>Header</th><th>Data Arrays</th></tr>"
            fmt += "<tr><td>"
        # Get the header info
        fmt += self.head(display=False, html=True)
        # Fill out arrays
        if self.n_arrays > 0:
            fmt += "</td><td>"
            fmt += "\n"
            fmt += "<table>\n"
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
                if arr.ndim > 1:
                    ncomp = arr.shape[1]
                else:
                    ncomp = 1
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

    def copy_from(self, mesh: _vtk.vtkDataSet):
        """Overwrite this dataset inplace with the new dataset's geometries and data.

        Parameters
        ----------
        mesh : vtk.vtkDataSet
            The overwriting mesh.

        Examples
        --------
        Create two meshes and overwrite ``mesh_a`` with ``mesh_b``.
        Show that ``mesh_a`` is equal to ``mesh_b``.

        >>> import pyvista
        >>> mesh_a = pyvista.Sphere()
        >>> mesh_b = pyvista.Cube()
        >>> mesh_a.copy_from(mesh_b)
        >>> mesh_a == mesh_b
        True

        """
        # Allow child classes to overwrite parent classes
        if not isinstance(self, type(mesh)):
            raise TypeError(
                f'The Input DataSet type {type(mesh)} must be '
                f'compatible with the one being overwritten {type(self)}'
            )
        self.deep_copy(mesh)
        if is_pyvista_dataset(mesh):
            self.copy_meta_from(mesh, deep=True)

    def overwrite(self, mesh: _vtk.vtkDataSet):
        """Overwrite this dataset inplace with the new dataset's geometries and data.

        .. deprecated:: 0.37.0
            Use :func:`DataSet.copy_from` instead.

        Parameters
        ----------
        mesh : vtk.vtkDataSet
            The overwriting mesh.

        """
        warnings.warn(
            "Use of `DataSet.overwrite` is deprecated. Use `DataSet.copy_from` instead.",
            PyVistaDeprecationWarning,
        )
        self.copy_from(mesh)

    def cast_to_unstructured_grid(self) -> 'pyvista.UnstructuredGrid':
        """Get a new representation of this object as a :class:`pyvista.UnstructuredGrid`.

        Returns
        -------
        pyvista.UnstructuredGrid
            Dataset cast into a :class:`pyvista.UnstructuredGrid`.

        Examples
        --------
        Cast a :class:`pyvista.PolyData` to a
        :class:`pyvista.UnstructuredGrid`.

        >>> import pyvista
        >>> mesh = pyvista.Sphere()
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

    def cast_to_pointset(self, deep: bool = False) -> 'pyvista.PointSet':
        """Get a new representation of this object as a :class:`pyvista.PointSet`.

        Parameters
        ----------
        deep : bool, optional
            When ``True`` makes a full copy of the object.  When ``False``,
            performs a shallow copy where the points and data arrays are
            references to the original object.

        Returns
        -------
        pyvista.PointSet
            Dataset cast into a :class:`pyvista.PointSet`.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> pointset = mesh.cast_to_pointset()
        >>> type(pointset)
        <class 'pyvista.core.pointset.PointSet'>

        """
        pset = pyvista.PointSet()
        pset.SetPoints(self.GetPoints())
        pset.GetPointData().ShallowCopy(self.GetPointData())
        if deep:
            return pset.copy(deep=True)
        return pset

    def find_closest_point(self, point: Iterable[float], n=1) -> int:
        """Find index of closest point in this mesh to the given point.

        If wanting to query many points, use a KDTree with scipy or another
        library as those implementations will be easier to work with.

        See: https://github.com/pyvista/pyvista-support/issues/107

        Parameters
        ----------
        point : iterable(float)
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

        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> index = mesh.find_closest_point((0, 1, 0))
        >>> index
        212

        Get the coordinate of that point.

        >>> mesh.points[index]
        pyvista_ndarray([-0.05218758,  0.49653167,  0.02706946], dtype=float32)

        """
        if not isinstance(point, (np.ndarray, collections.abc.Sequence)) or len(point) != 3:
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
        point: Union[VectorArray, NumericArray],
        return_closest_point: bool = False,
    ) -> Union[int, np.ndarray, Tuple[Union[int, np.ndarray], np.ndarray]]:
        """Find index of closest cell in this mesh to the given point.

        Parameters
        ----------
        point : Sequence(float) or np.ndarray
            Coordinates of point to query (length 3) or a ``numpy`` array of ``n``
            points with shape ``(n, 3)``.

        return_closest_point : bool, optional
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

        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> point = [0.1, 0.2, 0.3]
        >>> index = mesh.find_closest_cell(point)
        >>> index
        591

        Make sure that this cell indeed is the closest to
        ``[0.1, 0.2, 0.3]``.

        >>> import numpy as np
        >>> cell_centers = mesh.cell_centers()
        >>> relative_position = cell_centers.points - point
        >>> distance = np.linalg.norm(relative_position, axis=1)
        >>> np.argmin(distance)
        591

        Find the nearest cells to several random points that
        are centered on the origin.

        >>> points = 2 * np.random.random((5000, 3)) - 1
        >>> indices = mesh.find_closest_cell(points)
        >>> indices.shape
        (5000,)

        For the closest cell, find the point inside the cell that is
        closest to the supplied point.  The rectangle is a unit square
        with 1 cell and 4 nodal points at the corners in the plane with
        ``z`` normal and ``z=0``.  The closest point inside the cell is
        not usually at a nodal point.

        >>> unit_square = pyvista.Rectangle()
        >>> index, closest_point = unit_square.find_closest_cell(
        ...     [0.25, 0.25, 0.5],
        ...     return_closest_point=True
        ... )
        >>> closest_point
        array([0.25, 0.25, 0.  ])

        But, the closest point can be a nodal point, although the index of
        that point is not returned.  If the closest nodal point by index is
        desired, see :func:`DataSet.find_closest_point`.

        >>> index, closest_point = unit_square.find_closest_cell(
        ...     [1.0, 1.0, 0.5],
        ...     return_closest_point=True
        ... )
        >>> closest_point
        array([1., 1., 0.])

        """
        point, singular = _coerce_pointslike_arg(point, copy=False)

        locator = _vtk.vtkCellLocator()
        locator.SetDataSet(self)
        locator.BuildLocator()

        cell = _vtk.vtkGenericCell()

        closest_cells: List[int] = []
        closest_points: List[List[float]] = []

        for node in point:
            closest_point = [0.0, 0.0, 0.0]
            cell_id = _vtk.mutable(0)
            sub_id = _vtk.mutable(0)
            dist2 = _vtk.mutable(0.0)

            locator.FindClosestPoint(node, closest_point, cell, cell_id, sub_id, dist2)
            closest_cells.append(int(cell_id))
            closest_points.append(closest_point)

        out_cells: Union[int, np.ndarray] = (
            closest_cells[0] if singular else np.array(closest_cells)
        )
        out_points = np.array(closest_points[0]) if singular else np.array(closest_points)

        if return_closest_point:
            return out_cells, out_points
        return out_cells

    def find_containing_cell(
        self, point: Union[VectorArray, NumericArray]
    ) -> Union[int, np.ndarray]:
        """Find index of a cell that contains the given point.

        Parameters
        ----------
        point : Sequence(float) or np.ndarray
            Coordinates of point to query (length 3) or a ``numpy`` array of ``n``
            points with shape ``(n, 3)``.

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

        >>> import pyvista
        >>> mesh = pyvista.UniformGrid(dimensions=[5, 5, 1], spacing=[1/4, 1/4, 0])
        >>> mesh
        UniformGrid...
        >>> mesh.find_containing_cell([0.3, 0.3, 0.0])
        5

        A point outside the mesh domain will return ``-1``.

        >>> mesh.find_containing_cell([0.3, 0.3, 1.0])
        -1

        Find the cells that contain 1000 random points inside the mesh.

        >>> import numpy as np
        >>> points = np.random.random((1000, 3))
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
        pointa: Iterable[float],
        pointb: Iterable[float],
        tolerance=0.0,
    ) -> np.ndarray:
        """Find the index of cells in this mesh along a line.

        Line is defined from ``pointa`` to ``pointb``.

        Parameters
        ----------
        pointa : iterable(float)
            Length 3 coordinate of the start of the line.

        pointb : iterable(float)
            Length 3 coordinate of the end of the line.

        tolerance : float, optional
            The absolute tolerance to use to find cells along line.

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
        DataSet.find_cells_within_bounds

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> index = mesh.find_cells_along_line([0, 0, 0], [0, 0, 1.0])

        """
        if np.array(pointa).size != 3:
            raise TypeError("Point A must be a length three tuple of floats.")
        if np.array(pointb).size != 3:
            raise TypeError("Point B must be a length three tuple of floats.")
        locator = _vtk.vtkCellLocator()
        locator.SetDataSet(self)
        locator.BuildLocator()
        id_list = _vtk.vtkIdList()
        locator.FindCellsAlongLine(pointa, pointb, tolerance, id_list)
        return vtk_id_list_to_array(id_list)

    def find_cells_within_bounds(self, bounds: Iterable[float]) -> np.ndarray:
        """Find the index of cells in this mesh within bounds.

        Parameters
        ----------
        bounds : iterable(float)
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
        >>> import pyvista
        >>> mesh = pyvista.Cube()
        >>> index = mesh.find_cells_within_bounds([-2.0, 2.0, -2.0, 2.0, -2.0, 2.0])

        """
        if np.array(bounds).size != 6:
            raise TypeError("Bounds must be a length three tuple of floats.")
        locator = _vtk.vtkCellTreeLocator()
        locator.SetDataSet(self)
        locator.BuildLocator()
        id_list = _vtk.vtkIdList()
        locator.FindCellsWithinBounds(list(bounds), id_list)
        return vtk_id_list_to_array(id_list)

    def cell_n_points(self, ind: int) -> int:
        """Return the number of points in a cell.

        Parameters
        ----------
        ind : int
            Cell ID.

        Returns
        -------
        int
            Number of points in the cell.

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()
        >>> mesh.cell_n_points(0)
        3

        """
        return self.GetCell(ind).GetPoints().GetNumberOfPoints()

    def cell_points(self, ind: int) -> np.ndarray:
        """Return the points in a cell.

        Parameters
        ----------
        ind : int
            Cell ID.

        Returns
        -------
        numpy.ndarray
            An array of floats with shape (number of points, 3) containing the coordinates of the
            cell corners.

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()
        >>> mesh.cell_points(0)  # doctest:+SKIP
        [[896.99401855  48.76010132  82.26560211]
         [906.59301758  48.76010132  80.74520111]
         [907.53900146  55.49020004  83.65809631]]

        """
        # A copy of the points must be returned to avoid overlapping them since the
        # `vtk.vtkExplicitStructuredGrid.GetCell` is an override method.
        points = self.GetCell(ind).GetPoints().GetData()
        points = _vtk.vtk_to_numpy(points)
        return points.copy()

    def cell_bounds(self, ind: int) -> Tuple[float, float, float, float, float, float]:
        """Return the bounding box of a cell.

        Parameters
        ----------
        ind : int
            Cell ID.

        Returns
        -------
        tuple(float)
            The limits of the cell in the X, Y and Z directions respectively.

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()
        >>> mesh.cell_bounds(0)
        (896.9940185546875, 907.5390014648438, 48.760101318359375, 55.49020004272461, 80.74520111083984, 83.65809631347656)

        """
        return self.GetCell(ind).GetBounds()

    def cell_type(self, ind: int) -> int:
        """Return the type of a cell.

        Parameters
        ----------
        ind : int
            Cell type ID.

        Returns
        -------
        int
            VTK cell type. See `vtkCellType.h <https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html>`_ .

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()
        >>> mesh.cell_type(0)
        5

        """
        return self.GetCellType(ind)

    def cell_point_ids(self, ind: int) -> List[int]:
        """Return the point ids in a cell.

        Parameters
        ----------
        ind : int
            Cell ID.

        Returns
        -------
        list[int]
            Point Ids that are associated with the cell.

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()
        >>> mesh.cell_type(0)
        5

        Cell type 5 is a triangular cell with three points.

        >>> mesh.cell_point_ids(0)
        [0, 1, 2]

        """
        cell = self.GetCell(ind)
        point_ids = cell.GetPointIds()
        return [point_ids.GetId(i) for i in range(point_ids.GetNumberOfIds())]

    def point_is_inside_cell(
        self, ind: int, point: Union[VectorArray, NumericArray]
    ) -> Union[int, np.ndarray]:
        """Return whether one or more points are inside a cell.

        .. versionadded:: 0.35.0

        Parameters
        ----------
        ind : int
            Cell ID.

        point : Sequence[float] or np.ndarray
            Coordinates of point to query (length 3) or a ``numpy`` array of ``n``
            points with shape ``(n, 3)``.

        Returns
        -------
        bool or numpy.ndarray
            Whether point(s) is/are inside cell. A scalar bool is only returned if
            the input point has shape ``(3,)``.

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_hexbeam()
        >>> mesh.cell_bounds(0)
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
                    f"Computational difficulty encountered for point {node} in cell {ind}"
                )
            in_cell[i] = bool(is_inside)

        if singular:
            return in_cell[0]
        return in_cell
