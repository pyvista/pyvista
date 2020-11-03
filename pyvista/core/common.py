"""Attributes common to PolyData and Grid Objects."""

import collections.abc
import logging
import warnings
from pathlib import Path

import numpy as np
import vtk

import pyvista
from pyvista.utilities import (FieldAssociation, get_array, is_pyvista_dataset,
                               raise_not_matching, vtk_id_list_to_array, fileio,
                               abstract_class, axis_rotation)
from .datasetattributes import DataSetAttributes
from .filters import DataSetFilters

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')

# vector array names
DEFAULT_VECTOR_KEY = '_vectors'

ActiveArrayInfo = collections.namedtuple('ActiveInfo', field_names=['association', 'name'])


@abstract_class
class DataObject:
    """Methods common to all wrapped data objects."""

    _READERS = None
    _WRITERS = None

    def __init__(self, *args, **kwargs):
        """Initialize the data object."""
        super().__init__()
        # Remember which arrays come from numpy.bool arrays, because there is no direct
        # conversion from bool to vtkBitArray, such arrays are stored as vtkCharArray.
        self.association_bitarray_names = collections.defaultdict(set)

    def shallow_copy(self, to_copy):
        """Shallow copy the given mesh to this mesh."""
        return self.ShallowCopy(to_copy)

    def deep_copy(self, to_copy):
        """Overwrite this mesh with the given mesh as a deep copy."""
        return self.DeepCopy(to_copy)

    def _load_file(self, filename):
        """Generically load a vtk object from file.

        Parameters
        ----------
        filename : str, pathlib.Path
            Filename of object to be loaded.  File/reader type is inferred from the
            extension of the filename.

        Notes
        -----
        Binary files load much faster than ASCII.

        """
        if self._READERS is None:
            raise NotImplementedError(f'{self.__class__.__name__} readers are not specified,'
                                      ' this should be a dict of (file extension: vtkReader type)')

        file_path = Path(filename).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f'File {filename} does not exist')

        file_ext = file_path.suffix
        if file_ext not in self._READERS:
            valid_extensions = ', '.join(self._READERS.keys())
            raise ValueError(f'Invalid file extension for {self.__class__.__name__}({file_ext}).'
                             f' Must be one of: {valid_extensions}')

        reader = self._READERS[file_ext]()
        if file_ext == ".case":
            reader.SetCaseFileName(str(file_path))
        else:
            reader.SetFileName(str(file_path))
        reader.Update()
        return reader.GetOutputDataObject(0)

    def _from_file(self, filename):
        self.shallow_copy(self._load_file(filename))

    def save(self, filename, binary=True):
        """Save this vtk object to file.

        Parameters
        ----------
        filename : str, pathlib.Path
         Filename of output file. Writer type is inferred from
         the extension of the filename.

        binary : bool, optional
         If True, write as binary, else ASCII.

        Notes
        -----
        Binary files write much faster than ASCII and have a smaller
        file size.

        """
        if self._WRITERS is None:
            raise NotImplementedError(f'{self.__class__.__name__} writers are not specified,'
                                      ' this should be a dict of (file extension: vtkWriter type)')

        file_path = Path(filename).resolve()
        file_ext = file_path.suffix
        if file_ext not in self._WRITERS:
            raise ValueError('Invalid file extension for this data type.'
                             f' Must be one of: {self._WRITERS.keys()}')

        writer = self._WRITERS[file_ext]()
        fileio.set_vtkwriter_mode(vtk_writer=writer, use_binary=binary)
        writer.SetFileName(str(file_path))
        writer.SetInputData(self)
        writer.Write()

    def get_data_range(self, arr=None, preference='field'):  # pragma: no cover
        """Get the non-NaN min and max of a named array.

        Parameters
        ----------
        arr : str, np.ndarray, optional
            The name of the array to get the range. If None, the
            active scalar is used

        preference : str, optional
            When scalars is specified, this is the preferred array type
            to search for in the dataset.  Must be either ``'point'``,
            ``'cell'``, or ``'field'``.

        """
        raise NotImplementedError(f'{type(self)} mesh type does not have a `get_data_range` method.')

    def _get_attrs(self):  # pragma: no cover
        """Return the representation methods (internal helper)."""
        raise NotImplementedError('Called only by the inherited class')

    def head(self, display=True, html=None):
        """Return the header stats of this dataset.

        If in IPython, this will be formatted to HTML. Otherwise returns a console friendly string.

        """
        # Generate the output
        if html:
            fmt = ""
            # HTML version
            fmt += "\n"
            fmt += "<table>\n"
            fmt += f"<tr><th>{type(self).__name__}</th><th>Information</th></tr>\n"
            row = "<tr><td>{}</td><td>{}</td></tr>\n"
            # now make a call on the object to get its attributes as a list of len 2 tuples
            for attr in self._get_attrs():
                try:
                    fmt += row.format(attr[0], attr[2].format(*attr[1]))
                except:
                    fmt += row.format(attr[0], attr[2].format(attr[1]))
            if hasattr(self, 'n_arrays'):
                fmt += row.format('N Arrays', self.n_arrays)
            fmt += "</table>\n"
            fmt += "\n"
            if display:
                from IPython.display import display, HTML
                display(HTML(fmt))
                return
            return fmt
        # Otherwise return a string that is Python console friendly
        fmt = f"{type(self).__name__} ({hex(id(self))})\n"
        # now make a call on the object to get its attributes as a list of len 2 tuples
        row = "  {}:\t{}\n"
        for attr in self._get_attrs():
            try:
                fmt += row.format(attr[0], attr[2].format(*attr[1]))
            except:
                fmt += row.format(attr[0], attr[2].format(attr[1]))
        if hasattr(self, 'n_arrays'):
            fmt += row.format('N Arrays', self.n_arrays)
        return fmt

    def _repr_html_(self):  # pragma: no cover
        """Return a pretty representation for Jupyter notebooks.

        This includes header details and information about all arrays.

        """
        raise NotImplementedError('Called only by the inherited class')

    def copy_meta_from(self, ido):  # pragma: no cover
        """Copy pyvista meta data onto this object from another object."""
        pass  # called only by the inherited class

    def copy(self, deep=True):
        """Return a copy of the object.

        Parameters
        ----------
        deep : bool, optional
            When True makes a full copy of the object.

        Return
        ------
        newobject : same as input
           Deep or shallow copy of the input.

        """
        thistype = type(self)
        newobject = thistype()
        if deep:
            newobject.deep_copy(self)
        else:
            newobject.shallow_copy(self)
        newobject.copy_meta_from(self)
        return newobject

    def add_field_array(self, scalars, name, deep=True):
        """Add a field array."""
        self.field_arrays.append(scalars, name, deep_copy=deep)

    @property
    def field_arrays(self):
        """Return vtkFieldData as DataSetAttributes."""
        return DataSetAttributes(self.GetFieldData(), dataset=self, association=FieldAssociation.NONE)

    def clear_field_arrays(self):
        """Remove all field arrays."""
        self.field_arrays.clear()

    @property
    def memory_address(self):
        """Get address of the underlying C++ object in format 'Addr=%p'."""
        return self.GetInformation().GetAddressAsString("")


@abstract_class
class Common(DataSetFilters, DataObject):
    """Methods in common to spatially referenced objects."""

    # Simply bind pyvista.plotting.plot to the object
    plot = pyvista.plot

    def __init__(self, *args, **kwargs):
        """Initialize the common object."""
        super().__init__()
        self._last_active_scalars_name = None
        self._active_scalars_info = ActiveArrayInfo(FieldAssociation.POINT, name=None)
        self._active_vectors_info = ActiveArrayInfo(FieldAssociation.POINT, name=None)
        self._active_tensors_info = ActiveArrayInfo(FieldAssociation.POINT, name=None)
        self._textures = {}

    @property
    def active_scalars_info(self):
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
                    attributes.active_scalars = first_arr
                    break
            else:
                self._active_scalars_info = ActiveArrayInfo(field, None)
        return self._active_scalars_info

    @property
    def active_vectors_info(self):
        """Return the active scalar's field and name: [field, name]."""
        if self._active_vectors_info.name is None:
            # Sometimes, precomputed normals aren't set as active
            if 'Normals' in self.array_names:
                self.set_active_vectors('Normals')
        return self._active_vectors_info

    @property
    def active_tensors_info(self):
        """Return the active tensor's field and name: [field, name]."""
        return self._active_tensors_info

    @property
    def active_vectors(self):
        """Return the active vectors array."""
        field, name = self.active_vectors_info
        if name:
            if field is FieldAssociation.POINT:
                return self.point_arrays[name]
            if field is FieldAssociation.CELL:
                return self.cell_arrays[name]

    @property
    def active_tensors(self):
        """Return the active tensors array."""
        field, name = self.active_tensors_info
        if name:
            if field is FieldAssociation.POINT:
                return self.point_arrays[name]
            if field is FieldAssociation.CELL:
                return self.cell_arrays[name]

    @property
    def active_tensors_name(self):
        """Return the name of the active tensor array."""
        return self.active_tensors_info.name

    @active_tensors_name.setter
    def active_tensors_name(self, name):
        """Set the name of the active tensor."""
        self.set_active_tensors(name)

    @property
    def active_vectors_name(self):
        """Return the name of the active vectors array."""
        return self.active_vectors_info.name

    @active_vectors_name.setter
    def active_vectors_name(self, name):
        """Set the name of the active vector."""
        self.set_active_vectors(name)

    @property
    def active_scalars_name(self):
        """Return the active scalar's name."""
        return self.active_scalars_info.name

    @active_scalars_name.setter
    def active_scalars_name(self, name):
        """Set the name of the active scalar."""
        self.set_active_scalars(name)

    @property
    def points(self):
        """Return a pointer to the points as a numpy object."""
        pts = self.GetPoints()
        if pts is None:
            return None
        vtk_data = pts.GetData()
        # arr = vtk_to_numpy(vtk_data)
        return pyvista.pyvista_ndarray(vtk_data, dataset=self)

    @points.setter
    def points(self, points):
        """Set points without copying."""
        if not isinstance(points, np.ndarray):
            raise TypeError('Points must be a numpy array')
        vtk_points = pyvista.vtk_points(points, False)
        pdata = self.GetPoints()
        if not pdata:
            self.SetPoints(vtk_points)
        else:
            pdata.SetData(vtk_points.GetData())
        self.GetPoints().Modified()
        self.Modified()

    @property
    def arrows(self):
        """Return a glyph representation of the active vector data as arrows.

        Arrows will be located at the points of the mesh and
        their size will be dependent on the length of the vector.
        Their direction will be the "direction" of the vector

        Return
        ------
        arrows : pyvista.PolyData
            Active scalars represented as arrows.

        """
        if self.active_vectors is not None:
            name = self.active_vectors_name
            return self.glyph(scale=name, orient=name)

    @property
    def vectors(self):
        """Return active vectors."""
        return self.active_vectors

    @vectors.setter
    def vectors(self, array):
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
    def t_coords(self):
        """Return the active texture coordinates on the points."""
        return self.point_arrays.t_coords

    @t_coords.setter
    def t_coords(self, t_coords):
        """Set the array to use as the texture coordinates."""
        self.point_arrays.t_coords = t_coords

    @property
    def textures(self):
        """Return a dictionary to hold compatible ``vtk.vtkTexture`` objects.

        When casting back to a VTK dataset or filtering this dataset, these textures
        will not be passed.

        """
        return self._textures

    def clear_textures(self):
        """Clear the textures from this mesh."""
        self._textures.clear()

    def _activate_texture(mesh, name):
        """Grab a texture and update the active texture coordinates.

        This makes sure to not destroy old texture coordinates.

        Parameters
        ----------
        name : str
            The name of the texture and texture coordinates to activate

        Return
        ------
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

    def set_active_scalars(self, name, preference='cell'):
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

    def set_active_vectors(self, name, preference='point'):
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

    def set_active_tensors(self, name, preference='point'):
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

    def rename_array(self, old_name, new_name, preference='cell'):
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
    def active_scalars(self):
        """Return the active scalars as an array."""
        field, name = self.active_scalars_info
        if name is not None:
            if field == FieldAssociation.POINT:
                return self.point_arrays[name]
            elif field == FieldAssociation.CELL:
                return self.cell_arrays[name]

    def get_data_range(self, arr=None, preference='cell'):
        """Get the non-NaN min and max of a named array.

        Parameters
        ----------
        arr : str, np.ndarray, optional
            The name of the array to get the range. If None, the
            active scalars is used.

        preference : str, optional
            When scalars is specified, this is the preferred array type
            to search for in the dataset.  Must be either ``'point'``,
            ``'cell'``, or ``'field'``.

        """
        if arr is None:
            # use active scalars array
            _, arr = self.active_scalars_info
        if isinstance(arr, str):
            name = arr
            # This can return None when an array is not found - expected
            arr = get_array(self, name, preference=preference)
            if arr is None:
                # Raise a value error if fetching the range of an unknown array
                raise ValueError(f'Array `{name}` not present.')
        # If array has no tuples return a NaN range
        if arr is None or arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
            return (np.nan, np.nan)
        # Use the array range
        return np.nanmin(arr), np.nanmax(arr)

    def points_to_double(self):
        """Make points double precision."""
        if self.points.dtype != np.double:
            self.points = self.points.astype(np.double)

    def rotate_x(self, angle):
        """Rotate mesh about the x-axis.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the x-axis.

        """
        axis_rotation(self.points, angle, inplace=True, axis='x')

    def rotate_y(self, angle):
        """Rotate mesh about the y-axis.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the y-axis.

        """
        axis_rotation(self.points, angle, inplace=True, axis='y')

    def rotate_z(self, angle):
        """Rotate mesh about the z-axis.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the z-axis.

        """
        axis_rotation(self.points, angle, inplace=True, axis='z')

    def translate(self, xyz):
        """Translate the mesh.

        Parameters
        ----------
        xyz : list or np.ndarray
            Length 3 list or array.

        """
        self.points += np.asarray(xyz)

    def transform(self, trans):
        """Compute a transformation in place using a 4x4 transform.

        Parameters
        ----------
        trans : vtk.vtkMatrix4x4, vtk.vtkTransform, or np.ndarray
            Accepts a vtk transformation object or a 4x4 transformation matrix.

        """
        if isinstance(trans, vtk.vtkMatrix4x4):
            t = pyvista.trans_from_matrix(trans)
        elif isinstance(trans, vtk.vtkTransform):
            t = pyvista.trans_from_matrix(trans.GetMatrix())
        elif isinstance(trans, np.ndarray):
            if trans.ndim != 2:
                raise ValueError('Transformation array must be 4x4')
            elif trans.shape[0] != 4 or trans.shape[1] != 4:
                raise ValueError('Transformation array must be 4x4')
            t = trans
        else:
            raise TypeError('Input transform must be either:\n'
                            '\tvtk.vtkMatrix4x4\n'
                            '\tvtk.vtkTransform\n'
                            '\t4x4 np.ndarray\n')

        if t[3, 3] == 0:
            raise ValueError(
                "Transform element (3,3), the inverse scale term, is zero")

        # Normalize the transformation to account for the scale
        t /= t[3, 3]

        x = (self.points*t[0, :3]).sum(1) + t[0, -1]
        y = (self.points*t[1, :3]).sum(1) + t[1, -1]
        z = (self.points*t[2, :3]).sum(1) + t[2, -1]

        # overwrite points
        self.points[:, 0] = x
        self.points[:, 1] = y
        self.points[:, 2] = z

    def copy_meta_from(self, ido):
        """Copy pyvista meta data onto this object from another object."""
        self._active_scalars_info = ido.active_scalars_info
        self._active_vectors_info = ido.active_vectors_info
        self.clear_textures()
        self._textures = {name: tex.copy() for name, tex in ido.textures.items()}

    @property
    def point_arrays(self):
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
    def cell_arrays(self):
        """Return vtkCellData as DataSetAttributes."""
        return DataSetAttributes(self.GetCellData(), dataset=self, association=FieldAssociation.CELL)

    @property
    def n_points(self):
        """Return the number of points in the entire dataset."""
        return self.GetNumberOfPoints()

    @property
    def n_cells(self):
        """Return the number of cells in the entire dataset."""
        return self.GetNumberOfCells()

    @property
    def number_of_points(self):  # pragma: no cover
        """Return the number of points."""
        return self.GetNumberOfPoints()

    @property
    def number_of_cells(self):  # pragma: no cover
        """Return the number of cells."""
        return self.GetNumberOfCells()

    @property
    def bounds(self):
        """Return the bounding box of this dataset.

        The form is: (xmin,xmax, ymin,ymax, zmin,zmax).

        """
        return list(self.GetBounds())

    @property
    def length(self):
        """Return the length of the diagonal of the bounding box."""
        return self.GetLength()

    @property
    def center(self):
        """Return the center of the bounding box."""
        return list(self.GetCenter())

    @property
    def extent(self):
        """Return the range of the bounding box."""
        if hasattr(self, 'GetExtent'):
            return list(self.GetExtent())

    @extent.setter
    def extent(self, extent):
        """Set the range of the bounding box."""
        if hasattr(self, 'SetExtent'):
            if len(extent) != 6:
                raise ValueError('Extent must be a vector of 6 values.')
            self.SetExtent(extent)
        else:
            raise AttributeError('This mesh type does not handle extents.')

    @property
    def volume(self):
        """Return the mesh volume.

        Return
        ------
        volume : float
            Total volume of the mesh.

        """
        sizes = self.compute_cell_sizes(length=False, area=False, volume=True)
        return np.sum(sizes.cell_arrays['Volume'])

    def get_array(self, name, preference='cell', info=False):
        """Search both point, cell and field data for an array."""
        return get_array(self, name, preference=preference, info=info)

    def __getitem__(self, index):
        """Search both point, cell, and field data for an array."""
        if isinstance(index, collections.abc.Iterable) and not isinstance(index, str):
            name, preference = index
        elif isinstance(index, str):
            name = index
            preference = 'cell'
        else:
            raise KeyError(f'Index ({index}) not understood.'
                           ' Index must be a string name or a tuple of string name and string preference.')
        return self.get_array(name, preference=preference, info=False)

    def _ipython_key_completions_(self):
        return self.array_names

    def __setitem__(self, name, scalars):
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
    def n_arrays(self):
        """Return the number of arrays present in the dataset."""
        n = self.GetPointData().GetNumberOfArrays()
        n += self.GetCellData().GetNumberOfArrays()
        n += self.GetFieldData().GetNumberOfArrays()
        return n

    @property
    def array_names(self):
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

    def _repr_html_(self):
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

    def __repr__(self):
        """Return the object representation."""
        return self.head(display=False, html=False)

    def __str__(self):
        """Return the object string representation."""
        return self.head(display=False, html=False)

    def overwrite(self, mesh):
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

    def cast_to_unstructured_grid(self):
        """Get a new representation of this object as an :class:`pyvista.UnstructuredGrid`."""
        alg = vtk.vtkAppendFilter()
        alg.AddInputData(self)
        alg.Update()
        return pyvista.filters._get_output(alg)

    def find_closest_point(self, point, n=1):
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

        Return
        ------
        int : the index of the point in this mesh that is closes to the given point.
        """
        if not isinstance(point, (np.ndarray, collections.abc.Sequence)) or len(point) != 3:
            raise TypeError("Given point must be a length three sequence.")
        if not isinstance(n, int):
            raise TypeError("`n` must be a positive integer.")
        if n < 1:
            raise ValueError("`n` must be a positive integer.")

        locator = vtk.vtkPointLocator()
        locator.SetDataSet(self)
        locator.BuildLocator()
        if n > 1:
            id_list = vtk.vtkIdList()
            locator.FindClosestNPoints(n, point, id_list)
            return vtk_id_list_to_array(id_list)
        return locator.FindClosestPoint(point)

    def find_closest_cell(self, point):
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

        locator = vtk.vtkCellLocator()
        locator.SetDataSet(self)
        locator.BuildLocator()
        closest_cells = np.array([locator.FindCell(node) for node in point])
        return int(closest_cells[0]) if len(closest_cells) == 1 else closest_cells
