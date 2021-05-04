"""Attributes common to PolyData and Grid Objects."""

import collections.abc
import logging
from typing import Optional, List, Tuple, Iterable, Union, Any, Dict

import numpy as np

import pyvista
from pyvista import _vtk
from pyvista.utilities import (FieldAssociation, get_array, is_pyvista_dataset,
                               raise_not_matching, vtk_id_list_to_array,
                               abstract_class, axis_rotation, transformations)
from .dataobject import DataObject
from .datasetattributes import DataSetAttributes
from .filters import DataSetFilters, _get_output
from .pyvista_ndarray import pyvista_ndarray
from .._typing import Vector

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')

# vector array names
DEFAULT_VECTOR_KEY = '_vectors'


class ActiveArrayInfo:
    """Active array info class with support for pickling."""

    def __init__(self, association, name):
        """Initialize."""
        self.association = association
        self.name = name

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
        named_tuple = collections.namedtuple('ActiveArrayInfo', ['association', 'name'])
        return named_tuple(self.association, self.name)

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
        return self.name == other.name and \
               int(self.association.value) == int(other.association.value)


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
        """Return the active scalar's field and name: [field, name]."""
        field, name = self._active_scalars_info
        exclude = {'__custom_rgba', 'Normals', 'vtkOriginalPointIds', 'TCoords'}
        if name in exclude:
            name = self._last_active_scalars_name

        all_arrays = self.point_arrays.keys() + self.cell_arrays.keys()
        if name is None or name not in all_arrays:
            # find first available array name
            for attributes in (self.point_arrays, self.cell_arrays):
                first_arr = next((arr for arr in attributes if arr not in exclude), None)
                if first_arr is not None:
                    self._active_scalars_info = ActiveArrayInfo(attributes.association, first_arr)
                    attributes.active_scalars = first_arr  # type: ignore
                    break
            else:
                self._active_scalars_info = ActiveArrayInfo(field, None)
        return self._active_scalars_info

    @property
    def active_vectors_info(self) -> ActiveArrayInfo:
        """Return the active scalar's field and name: [field, name]."""
        if self._active_vectors_info.name is None:
            # Sometimes, precomputed normals aren't set as active
            if 'Normals' in self.array_names:
                self.set_active_vectors('Normals')
        return self._active_vectors_info

    @property
    def active_tensors_info(self) -> ActiveArrayInfo:
        """Return the active tensor's field and name: [field, name]."""
        return self._active_tensors_info

    @property
    def active_vectors(self) -> Optional[pyvista_ndarray]:
        """Return the active vectors array."""
        field, name = self.active_vectors_info
        try:
            if field is FieldAssociation.POINT:
                return self.point_arrays[name]
            if field is FieldAssociation.CELL:
                return self.cell_arrays[name]
        except KeyError:
            return None
        return None

    @property
    def active_tensors(self) -> Optional[np.ndarray]:
        """Return the active tensors array."""
        field, name = self.active_tensors_info
        try:
            if field is FieldAssociation.POINT:
                return self.point_arrays[name]
            if field is FieldAssociation.CELL:
                return self.cell_arrays[name]
        except KeyError:
            return None
        return None

    @property
    def active_tensors_name(self) -> str:
        """Return the name of the active tensor array."""
        return self.active_tensors_info.name

    @active_tensors_name.setter
    def active_tensors_name(self, name: str):
        """Set the name of the active tensor."""
        self.set_active_tensors(name)

    @property
    def active_vectors_name(self) -> str:
        """Return the name of the active vectors array."""
        return self.active_vectors_info.name

    @active_vectors_name.setter
    def active_vectors_name(self, name: str):
        """Set the name of the active vector."""
        self.set_active_vectors(name)

    @property
    def active_scalars_name(self) -> str:
        """Return the active scalar's name."""
        return self.active_scalars_info.name

    @active_scalars_name.setter
    def active_scalars_name(self, name: str):
        """Set the name of the active scalar."""
        self.set_active_scalars(name)

    @property
    def points(self) -> pyvista_ndarray:
        """Return a pointer to the points as a numpy object."""
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
    def points(self, points: np.ndarray):
        """Set points without copying."""
        pdata = self.GetPoints()
        if isinstance(points, pyvista_ndarray):
            # simply set the underlying data
            if points.VTKObject is not None and pdata is not None:
                pdata.SetData(points.VTKObject)
                pdata.Modified()
                self.Modified()
                return

        # otherwise, wrap and use the array
        if not isinstance(points, np.ndarray):
            raise TypeError('Points must be a numpy array')
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
        their size will be dependent on the length of the vector.
        Their direction will be the "direction" of the vector

        Returns
        -------
        arrows : pyvista.PolyData
            Active scalars represented as arrows.

        """
        name = self.active_vectors_name
        return None if name is None else self.glyph(scale=name, orient=name)

    @property
    def vectors(self) -> Optional[pyvista_ndarray]:
        """Return active vectors."""
        return self.active_vectors

    @vectors.setter
    def vectors(self, array: np.ndarray):
        """Set the active vector."""
        if array.ndim != 2:
            raise ValueError('vector array must be a 2-dimensional array')
        elif array.shape[1] != 3:
            raise ValueError('vector array must be 3D')
        elif array.shape[0] != self.n_points:
            raise ValueError('Number of vectors be the same as the number of points')

        self.point_arrays[DEFAULT_VECTOR_KEY] = array
        self.active_vectors_name = DEFAULT_VECTOR_KEY

    @property
    def t_coords(self) -> Optional[pyvista_ndarray]:
        """Return the active texture coordinates on the points."""
        return self.point_arrays.t_coords

    @t_coords.setter
    def t_coords(self, t_coords: np.ndarray):
        """Set the array to use as the texture coordinates."""
        self.point_arrays.t_coords = t_coords  # type: ignore

    @property
    def textures(self) -> Dict[str, _vtk.vtkTexture]:
        """Return a dictionary to hold compatible ``vtk.vtkTexture`` objects.

        When casting back to a VTK dataset or filtering this dataset, these textures
        will not be passed.

        """
        return self._textures

    def clear_textures(self):
        """Clear the textures from this mesh."""
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
        vtk.vtkTexture : The active texture

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

    def set_active_scalars(self, name: str, preference='cell'):
        """Find the scalars by name and appropriately sets it as active.

        To deactivate any active scalars, pass ``None`` as the ``name``.

        """
        if name is None:
            self.GetCellData().SetActiveScalars(None)
            self.GetPointData().SetActiveScalars(None)
            return
        _, field = get_array(self, name, preference=preference, info=True)
        self._last_active_scalars_name = self.active_scalars_info.name
        if field == FieldAssociation.POINT:
            ret = self.GetPointData().SetActiveScalars(name)
        elif field == FieldAssociation.CELL:
            ret = self.GetCellData().SetActiveScalars(name)
        else:
            raise ValueError(f'Data field ({field}) not usable')

        if ret < 0:
            raise ValueError(f'Data field ({field}) could not be set as the active scalars')

        self._active_scalars_info = ActiveArrayInfo(field, name)

    def set_active_vectors(self, name: str, preference='point'):
        """Find the vectors by name and appropriately sets it as active.

        To deactivate any active vectors, pass ``None`` as the ``name``.
        """
        if name is None:
            self.GetCellData().SetActiveVectors(None)
            self.GetPointData().SetActiveVectors(None)
            field = FieldAssociation.POINT
        else:
            _, field = get_array(self, name, preference=preference, info=True)
            if field == FieldAssociation.POINT:
                ret = self.GetPointData().SetActiveVectors(name)
            elif field == FieldAssociation.CELL:
                ret = self.GetCellData().SetActiveVectors(name)
            else:
                raise ValueError(f'Data field ({field}) not usable')

            if ret < 0:
                raise ValueError(f'Data field ({field}) could not be set as the active vectors')

        self._active_vectors_info = ActiveArrayInfo(field, name)

    def set_active_tensors(self, name: str, preference='point'):
        """Find the tensors by name and appropriately sets it as active.

        To deactivate any active tensors, pass ``None`` as the ``name``.
        """
        if name is None:
            self.GetCellData().SetActiveTensors(None)
            self.GetPointData().SetActiveTensors(None)
            field = FieldAssociation.POINT
        else:
            _, field = get_array(self, name, preference=preference, info=True)
            if field == FieldAssociation.POINT:
                ret = self.GetPointData().SetActiveTensors(name)
            elif field == FieldAssociation.CELL:
                ret = self.GetCellData().SetActiveTensors(name)
            else:
                raise ValueError(f'Data field ({field}) not usable')

            if ret < 0:
                raise ValueError(f'Data field ({field}) could not be set as the active tensors')

        self._active_tensors_info = ActiveArrayInfo(field, name)

    def rename_array(self, old_name: str, new_name: str, preference='cell'):
        """Change array name by searching for the array then renaming it."""
        _, field = get_array(self, old_name, preference=preference, info=True)
        was_active = False
        if self.active_scalars_name == old_name:
            was_active = True
        if field == FieldAssociation.POINT:
            self.point_arrays[new_name] = self.point_arrays.pop(old_name)
        elif field == FieldAssociation.CELL:
            self.cell_arrays[new_name] = self.cell_arrays.pop(old_name)
        elif field == FieldAssociation.NONE:
            self.field_arrays[new_name] = self.field_arrays.pop(old_name)
        else:
            raise KeyError(f'Array with name {old_name} not found.')
        if was_active:
            self.set_active_scalars(new_name, preference=field)

    @property
    def active_scalars(self) -> Optional[pyvista_ndarray]:
        """Return the active scalars as an array."""
        field, name = self.active_scalars_info
        try:
            if field == FieldAssociation.POINT:
                return self.point_arrays[name]
            if field == FieldAssociation.CELL:
                return self.cell_arrays[name]
        except KeyError:
            return None
        return None

    def get_data_range(self,
                       arr_var: Optional[Union[str, np.ndarray]] = None,
                       preference='cell') -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
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

        """
        if arr_var is None:  # use active scalars array
            _, arr_var = self.active_scalars_info
            if arr_var is None:
                return (np.nan, np.nan)

        if isinstance(arr_var, str):
            name = arr_var
            # This can return None when an array is not found - expected
            arr = get_array(self, name, preference=preference)
            if arr is None:
                # Raise a value error if fetching the range of an unknown array
                raise ValueError(f'Array `{name}` not present.')
        else:
            arr = arr_var

        # If array has no tuples return a NaN range
        if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
            return (np.nan, np.nan)
        # Use the array range
        return np.nanmin(arr), np.nanmax(arr)

    def points_to_double(self):
        """Make points double precision."""
        if self.points.dtype != np.double:
            self.points = self.points.astype(np.double)

    def rotate_x(self, angle: float, transform_all_input_vectors=False):
        """Rotate mesh about the x-axis.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the x-axis.

        """
        t = transformations.axis_angle_rotation((1, 0, 0), angle, deg=True)
        self.transform(t, transform_all_input_vectors=transform_all_input_vectors, inplace=True)

    def rotate_y(self, angle: float, transform_all_input_vectors=False):
        """Rotate mesh about the y-axis.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the y-axis.

        """
        t = transformations.axis_angle_rotation((0, 1, 0), angle, deg=True)
        self.transform(t, transform_all_input_vectors=transform_all_input_vectors, inplace=True)

    def rotate_z(self, angle: float, transform_all_input_vectors=False):
        """Rotate mesh about the z-axis.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the z-axis.

        """
        t = transformations.axis_angle_rotation((0, 0, 1), angle, deg=True)
        self.transform(t, transform_all_input_vectors=transform_all_input_vectors, inplace=True)

    def translate(self, xyz: Union[list, tuple, np.ndarray]):
        """Translate the mesh.

        Parameters
        ----------
        xyz : list or tuple or np.ndarray
            Length 3 list, tuple or array.

        """
        self.points += np.asarray(xyz)

    def copy_meta_from(self, ido: 'DataSet'):
        """Copy pyvista meta data onto this object from another object."""
        self._active_scalars_info = ido.active_scalars_info
        self._active_vectors_info = ido.active_vectors_info
        self.clear_textures()
        self._textures = {name: tex.copy() for name, tex in ido.textures.items()}

    @property
    def point_arrays(self) -> DataSetAttributes:
        """Return vtkPointData as DataSetAttributes."""
        return DataSetAttributes(self.GetPointData(), dataset=self, association=FieldAssociation.POINT)

    def clear_point_arrays(self):
        """Remove all point arrays."""
        self.point_arrays.clear()

    def clear_cell_arrays(self):
        """Remove all cell arrays."""
        self.cell_arrays.clear()

    def clear_arrays(self):
        """Remove all arrays from point/cell/field data."""
        self.clear_point_arrays()
        self.clear_cell_arrays()
        self.clear_field_arrays()

    @property
    def cell_arrays(self) -> DataSetAttributes:
        """Return vtkCellData as DataSetAttributes."""
        return DataSetAttributes(self.GetCellData(), dataset=self, association=FieldAssociation.CELL)

    @property
    def n_points(self) -> int:
        """Return the number of points in the entire dataset."""
        return self.GetNumberOfPoints()

    @property
    def n_cells(self) -> int:
        """Return the number of cells in the entire dataset."""
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
    def bounds(self) -> List[float]:
        """Return the bounding box of this dataset.

        The form is: (xmin,xmax, ymin,ymax, zmin,zmax).

        """
        return list(self.GetBounds())

    @property
    def length(self) -> float:
        """Return the length of the diagonal of the bounding box."""
        return self.GetLength()

    @property
    def center(self) -> Vector:
        """Return the center of the bounding box."""
        return list(self.GetCenter())

    @property
    def extent(self) -> Optional[list]:
        """Return the range of the bounding box."""
        try:
            _extent = list(self.GetExtent())
        except AttributeError:
            return None
        return _extent

    @extent.setter
    def extent(self, extent: List[float]):
        """Set the range of the bounding box."""
        if hasattr(self, 'SetExtent'):
            if len(extent) != 6:
                raise ValueError('Extent must be a vector of 6 values.')
            self.SetExtent(extent)
        else:
            raise AttributeError('This mesh type does not handle extents.')

    @property
    def volume(self) -> float:
        """Return the mesh volume.

        Returns
        -------
        volume : float
            Total volume of the mesh.

        """
        sizes = self.compute_cell_sizes(length=False, area=False, volume=True)
        return np.sum(sizes.cell_arrays['Volume'])

    def get_array(self, name: str, preference='cell', info=False) -> Union[Tuple, np.ndarray]:
        """Search both point, cell and field data for an array."""
        return get_array(self, name, preference=preference, info=info, err=True)

    def __getitem__(self, index: Union[Iterable, str]) -> Union[Tuple, np.ndarray]:
        """Search both point, cell, and field data for an array."""
        if isinstance(index, collections.abc.Iterable) and not isinstance(index, str):
            name, preference = tuple(index)
        elif isinstance(index, str):
            name = index
            preference = 'cell'
        else:
            raise KeyError(f'Index ({index}) not understood.'
                           ' Index must be a string name or a tuple of string name and string preference.')
        return self.get_array(name, preference=preference, info=False)

    def _ipython_key_completions_(self) -> List[str]:
        return self.array_names

    def __setitem__(self, name: str, scalars: np.ndarray):
        """Add/set an array in the point_arrays, or cell_arrays accordingly.

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
            self.point_arrays[name] = scalars
        elif scalars.shape[0] == self.n_cells:
            self.cell_arrays[name] = scalars
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

        """
        names = []
        names.extend(self.field_arrays.keys())
        names.extend(self.point_arrays.keys())
        names.extend(self.cell_arrays.keys())
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

            for key, arr in self.point_arrays.items():
                fmt += format_array(key, arr, 'Points')
            for key, arr in self.cell_arrays.items():
                fmt += format_array(key, arr, 'Cells')
            for key, arr in self.field_arrays.items():
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

    def overwrite(self, mesh: _vtk.vtkDataSet):
        """Overwrite this mesh inplace with the new mesh's geometries and data.

        Parameters
        ----------
        mesh : vtk.vtkDataSet
            The overwriting mesh.

        """
        if not isinstance(mesh, type(self)):
            raise TypeError('The Input DataSet type must match '
                            f'the one being overwritten {type(self)}')
        self.deep_copy(mesh)
        if is_pyvista_dataset(mesh):
            self.copy_meta_from(mesh)

    def cast_to_unstructured_grid(self) -> 'pyvista.UnstructuredGrid':
        """Get a new representation of this object as an :class:`pyvista.UnstructuredGrid`."""
        alg = _vtk.vtkAppendFilter()
        alg.AddInputData(self)
        alg.Update()
        return _get_output(alg)

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
        int : the index of the point in this mesh that is closes to the given point.
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

    def find_closest_cell(self, point: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        """Find index of closest cell in this mesh to the given point.

        Parameters
        ----------
        point : iterable(float) or np.ndarray
            Length 3 coordinate of the point to query or a ``numpy`` array
            of coordinates.

        Returns
        -------
        index : int or np.ndarray
            Index or indices of the cell in this mesh that is closest
            to the given point.

        Examples
        --------
        Find nearest cell to a point on a sphere

        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> index = mesh.find_closest_cell([0, 0, 0.5])
        >>> index
        59

        Find the nearest cells to several random points.  Note that
        ``-1`` indicates that the locator was not able to find a
        reasonably close cell.

        >>> import numpy as np
        >>> points = np.random.random((1000, 3))
        >>> indices = mesh.find_closest_cell(points)
        >>> print(indices.shape)
        (1000,)
        """
        if isinstance(point, collections.abc.Sequence):
            point = np.array(point)
        # check if this is an array of points
        if isinstance(point, np.ndarray):
            if point.ndim > 2:
                raise ValueError("Array of points must be 2D")
            if point.ndim == 2:
                if point.shape[1] != 3:
                    raise ValueError("Array of points must have three values per point")
            else:
                if point.size != 3:
                    raise ValueError("Given point must have three values")
                point = np.array([point])
        else:
            raise TypeError("Given point must be an iterable or an array.")

        locator = _vtk.vtkCellLocator()
        locator.SetDataSet(self)
        locator.BuildLocator()
        closest_cells = np.array([locator.FindCell(node) for node in point])
        return int(closest_cells[0]) if len(closest_cells) == 1 else closest_cells

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

    def cell_bounds(self, ind: int) -> List[float]:
        """Return the bounding box of a cell.

        Parameters
        ----------
        ind : int
            Cell ID.

        Returns
        -------
        list(float)
            The limits of the cell in the X, Y and Z directions respectivelly.

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()
        >>> mesh.cell_bounds(0)
        [896.9940185546875, 907.5390014648438, 48.760101318359375, 55.49020004272461, 80.74520111083984, 83.65809631347656]

        """
        return list(self.GetCell(ind).GetBounds())

    def cell_type(self, ind: int) -> int:
        """Return the type of a cell.

        Parameters
        ----------
        ind : int
            Cell ID.

        Returns
        -------
        int
            VTK cell type. See <https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html>.

        Examples
        --------
        >>> from pyvista import examples
        >>> mesh = examples.load_airplane()
        >>> mesh.cell_type(0)
        5

        """
        return self.GetCellType(ind)
