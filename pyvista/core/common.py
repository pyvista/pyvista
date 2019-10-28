"""
Attributes common to PolyData and Grid Objects
"""
import collections
import logging
import warnings
from weakref import proxy

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import pyvista
from pyvista.utilities import (CELL_DATA_FIELD, FIELD_DATA_FIELD,
                               POINT_DATA_FIELD, convert_array, get_array,
                               is_pyvista_dataset, parse_field_choice,
                               raise_not_matching, vtk_bit_array_to_char)

from .filters import DataSetFilters

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')

# vector array names
DEFAULT_VECTOR_KEY = '_vectors'



class DataObject(object):
    """ Methods common to all wrapped data objects """
    def __init__(self, *args, **kwargs):
        self._field_bool_array_names = []


    def __new__(cls, *args, **kwargs):
        if cls is DataObject:
            raise TypeError("pyvista.DataObject is an abstract class and may not be instantiated.")
        return object.__new__(cls, *args, **kwargs)


    def shallow_copy(self, to_copy):
        """Shallow copy the given mesh to this mesh"""
        return self.ShallowCopy(to_copy)


    def deep_copy(self, to_copy):
        """Overwrite this mesh with the given mesh as a deep copy"""
        return self.DeepCopy(to_copy)


    def save(self, filename, binary=True):
        """
        Writes this mesh to a file.

        Parameters
        ----------
        filename : str
         Filename of mesh to be written.  File type is inferred from
         the extension of the filename unless overridden with
         ftype.

        binary : bool, optional
         Writes the file as binary when True and ASCII when False.

        Notes
        -----
        Binary files write much faster than ASCII and have a smaller
        file size.
        """
        raise NotImplementedError('{} mesh type does not have a save method.'.format(type(self)))


    def get_data_range(self, arr=None, preference='field'):
        """Get the non-NaN min and max of a named scalar array

        Parameters
        ----------
        arr : str, np.ndarray, optional
            The name of the array to get the range. If None, the active scalar
            is used

        preference : str, optional
            When scalars is specified, this is the perfered scalar type to
            search for in the dataset.  Must be either ``'point'``, ``'cell'``,
            or ``'field'``.

        """
        raise NotImplementedError('{} mesh type does not have a `get_data_range` method.'.format(type(self)))


    def _get_attrs(self):
        """An internal helper for the representation methods"""
        raise NotImplementedError


    def head(self, display=True, html=None):
        """Return the header stats of this dataset. If in IPython, this will
        be formatted to HTML. Otherwise returns a console friendly string"""
        # Generate the output
        if html:
            fmt = ""
            # HTML version
            fmt += "\n"
            fmt += "<table>\n"
            fmt += "<tr><th>{}</th><th>Information</th></tr>\n".format(type(self).__name__)
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
        fmt = "{} ({})\n".format(type(self).__name__, hex(id(self)))
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


    def _repr_html_(self):
        """A pretty representation for Jupyter notebooks that includes header
        details and information about all scalar arrays"""
        raise NotImplemented


    def copy_meta_from(self, ido):
        """Copies pyvista meta data onto this object from another object"""
        pass


    def copy(self, deep=True):
        """
        Returns a copy of the object

        Parameters
        ----------
        deep : bool, optional
            When True makes a full copy of the object.

        Returns
        -------
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


    def _field_array(self, name=None):
        """
        Returns field scalars of a vtk object

        Parameters
        ----------
        name : str
            Name of field scalars to retrive.

        Returns
        -------
        scalars : np.ndarray
            Numpy array of scalars

        """
        if name is None:
            raise RuntimeError('Must specify an array to fetch.')
        vtkarr = self.GetFieldData().GetAbstractArray(name)
        if vtkarr is None:
            raise AssertionError('({}) is not a valid field scalar array'.format(name))

        # numpy does not support bit array data types
        if isinstance(vtkarr, vtk.vtkBitArray):
            vtkarr = vtk_bit_array_to_char(vtkarr)
            if name not in self._point_bool_array_names:
                self._field_bool_array_names.append(name)

        array = convert_array(vtkarr)
        if array.dtype == np.uint8 and name in self._field_bool_array_names:
            array = array.view(np.bool)
        return array


    def _add_field_array(self, scalars, name, deep=True):
        """
        Adds field scalars to the mesh

        Parameters
        ----------
        scalars : numpy.ndarray
            Numpy array of scalars.  Does not have to match number of points or
            numbers of cells.

        name : str
            Name of field scalars to add.

        deep : bool, optional
            Does not copy scalars when False.  A reference to the scalars
            must be kept to avoid a segfault.

        """
        if scalars is None:
            raise TypeError('Empty array unable to be added')

        if not isinstance(scalars, np.ndarray):
            scalars = np.array(scalars)

        # need to track which arrays are boolean as all boolean arrays
        # must be stored as uint8
        if scalars.dtype == np.bool:
            scalars = scalars.view(np.uint8)
            if name not in self._field_bool_array_names:
                self._field_bool_array_names.append(name)

        if not scalars.flags.c_contiguous:
            scalars = np.ascontiguousarray(scalars)

        vtkarr = convert_array(scalars, deep=deep)
        vtkarr.SetName(name)
        self.GetFieldData().AddArray(vtkarr)


    def _add_field_scalar(self, scalars, name, set_active=False, deep=True):
        """DEPRECATED: Please use `_add_field_array`"""
        warnings.warn('Deprecation Warning: `_add_field_scalar` is now `_add_field_array`', RuntimeWarning)
        return self._add_field_array(scalars, name, set_active=set_active, deep=deep)


    def add_field_array(self, scalars, name, deep=True):
        self._add_field_array(scalars, name, deep=deep)


    @property
    def field_arrays(self):
        """ Returns all field arrays """
        fdata = self.GetFieldData()
        narr = fdata.GetNumberOfArrays()

        # just return if unmodified
        if hasattr(self, '_field_arrays'):
            keys = list(self._field_arrays.keys())
            if narr == len(keys):
                return self._field_arrays

        # dictionary with callbacks
        self._field_arrays = FieldScalarsDict(self)

        for i in range(narr):
            name = fdata.GetArrayName(i)
            if name is None or len(name) < 1:
                name = 'Field Array {}'.format(i)
                fdata.GetAbstractArray(i).SetName(name)
            self._field_arrays[name] = self._field_array(name)

        self._field_arrays.enable_callback()
        return self._field_arrays


    def clear_field_arrays(self):
        """ removes all field arrays """
        keys = self.field_arrays.keys()
        for key in keys:
            self._remove_array(FIELD_DATA_FIELD, key)



class Common(DataSetFilters, DataObject):
    """ Methods in common to spatially referenced objects"""

    # Simply bind pyvista.plotting.plot to the object
    plot = pyvista.plot


    def __new__(cls, *args, **kwargs):
        if cls is Common:
            raise TypeError("pyvista.Common is an abstract class and may not be instantiated.")
        return object.__new__(cls, *args, **kwargs)


    def __init__(self, *args, **kwargs):
        super(Common, self).__init__()
        self.references = []
        self._point_bool_array_names = []
        self._cell_bool_array_names = []


    @property
    def active_scalar_info(self):
        """Return the active scalar's field and name: [field, name]"""
        if not hasattr(self, '_active_scalar_info'):
            self._active_scalar_info = [POINT_DATA_FIELD, None] # field and name
        if not hasattr(self, '_last_active_scalar_name'):
            self._last_active_scalar_name = None
        field, name = self._active_scalar_info

        # rare error where scalar name isn't a valid scalar
        if name not in self.point_arrays:
            if name not in self.cell_arrays:
                if name in self.field_arrays:
                    raise RuntimeError('Field arrays cannot be made active. '
                                       'Convert to point/cell arrays if possible.')
                else:
                    name = None

        exclude = ['__custom_rgba', 'Normals', 'vtkOriginalPointIds',
                   'TCoords']

        def search_for_array(data):
            arr = None
            for i in range(data.GetNumberOfArrays()):
                name = data.GetArrayName(i)
                if name not in exclude:
                    arr = name
                    break
            return arr

        if name in exclude:
            name = self._last_active_scalar_name

        if name is None:
            if self.n_arrays < 1:
                return field, name
            # find some array in the set field
            parr = search_for_array(self.GetPointData())
            carr = search_for_array(self.GetCellData())
            if parr is not None:
                self._active_scalar_info = [POINT_DATA_FIELD, parr]
                self.GetPointData().SetActiveScalars(parr)
            elif carr is not None:
                self._active_scalar_info = [CELL_DATA_FIELD, carr]
                self.GetCellData().SetActiveScalars(carr)

        return self._active_scalar_info


    @property
    def active_vectors_info(self):
        """Return the active scalar's field and name: [field, name]"""
        if not hasattr(self, '_active_vectors_info'):
            # Sometimes, precomputed normals aren't set as active
            if 'Normals' in self.array_names:
                self.set_active_vectors('Normals')
            else:
                self._active_vectors_info = [POINT_DATA_FIELD, None] # field and name
        _, name = self._active_vectors_info

        # rare error where scalar name isn't a valid scalar
        if name not in self.point_arrays:
            if name not in self.cell_arrays:
                if name in self.field_arrays:
                    raise RuntimeError('Field arrays cannot be made active. '
                                       'Convert to point/cell array if possible.')
                else:
                    name = None

        return self._active_vectors_info


    @property
    def active_vectors(self):
        """The active vectors array"""
        field, name = self.active_vectors_info
        if name:
            if field is POINT_DATA_FIELD:
                return self.point_arrays[name]
            if field is CELL_DATA_FIELD:
                return self.cell_arrays[name]


    @property
    def active_vectors_name(self):
        """The name of the active vectors array"""
        return self.active_vectors_info[1]


    @active_vectors_name.setter
    def active_vectors_name(self, name):
        """Set the name of the active vector"""
        return self.set_active_vectors(name)


    @property
    def active_scalar_name(self):
        """Returns the active scalar's name"""
        return self.active_scalar_info[1]


    @active_scalar_name.setter
    def active_scalar_name(self, name):
        """Set the name of the active scalar"""
        return self.set_active_scalar(name)


    @property
    def points(self):
        """ returns a pointer to the points as a numpy object """
        pts = self.GetPoints()
        if pts is None:
            return None
        vtk_data = pts.GetData()
        arr = vtk_to_numpy(vtk_data)
        return pyvista_ndarray(arr, vtk_data)


    @points.setter
    def points(self, points):
        """ set points without copying """
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
        """
        Returns a glyph representation of the active vector data as
        arrows.  Arrows will be located at the points of the mesh and
        their size will be dependent on the length of the vector.
        Their direction will be the "direction" of the vector

        Returns
        -------
        arrows : pyvista.PolyData
            Active scalars represented as arrows.
        """
        if self.active_vectors is None:
            return
        name = self.active_vectors_name
        return self.glyph(scale=name, orient=name)


    @property
    def vectors(self):
        """ Returns active vectors """
        return self.active_vectors


    @vectors.setter
    def vectors(self, array):
        """ Sets the active vector  """
        if array.ndim != 2:
            raise AssertionError('vector array must be a 2-dimensional array')
        elif array.shape[1] != 3:
            raise RuntimeError('vector array must be 3D')
        elif array.shape[0] != self.n_points:
            raise RuntimeError('Number of vectors be the same as the number of points')

        self.point_arrays[DEFAULT_VECTOR_KEY] = array
        self.active_vectors_name = DEFAULT_VECTOR_KEY


    @property
    def t_coords(self):
        """The active texture coordinates on the points"""
        if self.GetPointData().GetTCoords() is not None:
            return vtk_to_numpy(self.GetPointData().GetTCoords())
        return None


    @t_coords.setter
    def t_coords(self, t_coords):
        """Set the array to use as the texture coordinates"""
        if not isinstance(t_coords, np.ndarray):
            raise TypeError('Texture coordinates must be a numpy array')
        if t_coords.ndim != 2:
            raise AssertionError('Texture coordinates must be a 2-dimensional array')
        if t_coords.shape[0] != self.n_points:
            raise AssertionError('Number of texture coordinates ({}) must match number of points ({})'.format(t_coords.shape[0], self.n_points))
        if t_coords.shape[1] != 2:
            raise AssertionError('Texture coordinates must only have 2 components, not ({})'.format(t_coords.shape[1]))
        # if np.min(t_coords) < 0.0 or np.max(t_coords) > 1.0:
        #     warnings.warn('Texture coordinates are typically within (0, 1) range. Textures will repeat on this mesh.', RuntimeWarning)
        # convert the array
        vtkarr = numpy_to_vtk(t_coords)
        vtkarr.SetName('Texture Coordinates')
        self.GetPointData().SetTCoords(vtkarr)
        self.GetPointData().Modified()
        return


    @property
    def textures(self):
        """A dictionary to hold ``vtk.vtkTexture`` objects that can be
        associated with this dataset. When casting back to a VTK dataset or
        filtering this dataset, these textures will not be passed.
        """
        if not hasattr(self, '_textures'):
            self._textures = {}
        return self._textures


    def clear_textures(self):
        """Clear the textures from this mesh"""
        if hasattr(self, '_textures'):
            del self._textures


    def _activate_texture(mesh, name):
        """Grab a texture and update the active texture coordinates. This makes
        sure to not destroy old texture coordinates

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
            # Grab the first name availabe if True
            idx = 0 if not isinstance(name, int) or name is True else name
            if idx > len(keys):
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
            logging.warning('Texture ({}) not associated with this dataset'.format(name))
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


    def set_active_scalar(self, name, preference='cell'):
        """Finds the scalar by name and appropriately sets it as active.
        To deactivate any active scalars, pass ``None`` as the ``name``.
        """
        if name is None:
            self.GetCellData().SetActiveScalars(None)
            self.GetPointData().SetActiveScalars(None)
            return
        _, field = get_array(self, name, preference=preference, info=True)
        self._last_active_scalar_name = self.active_scalar_info[1]
        if field == POINT_DATA_FIELD:
            self.GetPointData().SetActiveScalars(name)
        elif field == CELL_DATA_FIELD:
            self.GetCellData().SetActiveScalars(name)
        else:
            raise RuntimeError('Data field ({}) not useable'.format(field))
        self._active_scalar_info = [field, name]


    def set_active_vectors(self, name, preference='point'):
        """Finds the vectors by name and appropriately sets it as active
        To deactivate any active scalars, pass ``None`` as the ``name``.
        """
        if name is None:
            self.GetCellData().SetActiveVectors(None)
            self.GetPointData().SetActiveVectors(None)
            return
        _, field = get_array(self, name, preference=preference, info=True)
        if field == POINT_DATA_FIELD:
            self.GetPointData().SetActiveVectors(name)
        elif field == CELL_DATA_FIELD:
            self.GetCellData().SetActiveVectors(name)
        else:
            raise RuntimeError('Data field ({}) not useable'.format(field))
        self._active_vectors_info = [field, name]


    def rename_scalar(self, old_name, new_name, preference='cell'):
        """Changes array name by searching for the array then renaming it"""
        _, field = get_array(self, old_name, preference=preference, info=True)
        if field == POINT_DATA_FIELD:
            self.point_arrays[new_name] = self.point_arrays.pop(old_name)
        elif field == CELL_DATA_FIELD:
            self.cell_arrays[new_name] = self.cell_arrays.pop(old_name)
        elif field == FIELD_DATA_FIELD:
            self.field_arrays[new_name] = self.field_arrays.pop(old_name)
        else:
            raise RuntimeError('Array not found.')
        if self.active_scalar_info[1] == old_name:
            self.set_active_scalar(new_name, preference=field)


    @property
    def active_scalar(self):
        """Returns the active scalar as an array"""
        field, name = self.active_scalar_info
        if name is None:
            return None
        if field == POINT_DATA_FIELD:
            return self._point_array(name)
        elif field == CELL_DATA_FIELD:
            return self._cell_array(name)


    def _point_array(self, name=None):
        """
        Returns point scalars of a vtk object

        Parameters
        ----------
        name : str
            Name of point scalars to retrive.

        Returns
        -------
        scalars : np.ndarray
            Numpy array of scalars

        """
        if name is None:
            # use active scalar array
            field, name = self.active_scalar_info
            if field != POINT_DATA_FIELD:
                raise RuntimeError('Must specify an array to fetch.')
        vtkarr = self.GetPointData().GetAbstractArray(name)
        if vtkarr is None:
            raise AssertionError('({}) is not a point scalar'.format(name))

        # numpy does not support bit array data types
        if isinstance(vtkarr, vtk.vtkBitArray):
            vtkarr = vtk_bit_array_to_char(vtkarr)
            if name not in self._point_bool_array_names:
                self._point_bool_array_names.append(name)

        array = convert_array(vtkarr)
        if array.dtype == np.uint8 and name in self._point_bool_array_names:
            array = array.view(np.bool)
        return array


    def _add_point_array(self, scalars, name, set_active=False, deep=True):
        """
        Adds point scalars to the mesh

        Parameters
        ----------
        scalars : numpy.ndarray
            Numpy array of scalars.  Must match number of points.

        name : str
            Name of point scalars to add.

        set_active : bool, optional
            Sets the scalars to the active plotting scalars.  Default False.

        deep : bool, optional
            Does not copy scalars when False.  A reference to the scalars
            must be kept to avoid a segfault.

        """
        if scalars is None:
            raise TypeError('Empty array unable to be added')

        if not isinstance(scalars, np.ndarray):
            scalars = np.array(scalars)

        if scalars.shape[0] != self.n_points:
            raise Exception('Number of scalars must match the number of points')

        # need to track which arrays are boolean as all boolean arrays
        # must be stored as uint8
        if scalars.dtype == np.bool:
            scalars = scalars.view(np.uint8)
            if name not in self._point_bool_array_names:
                self._point_bool_array_names.append(name)

        if not scalars.flags.c_contiguous:
            scalars = np.ascontiguousarray(scalars)

        vtkarr = convert_array(scalars, deep=deep)
        vtkarr.SetName(name)
        self.GetPointData().AddArray(vtkarr)
        if set_active or self.active_scalar_info[1] is None:
            self.GetPointData().SetActiveScalars(name)
            self._active_scalar_info = [POINT_DATA_FIELD, name]


    def _add_point_scalar(self, scalars, name, set_active=False, deep=True):
        """DEPRECATED: Please use `_add_point_array`"""
        warnings.warn('Deprecation Warning: `_add_point_scalar` is now `_add_point_array`', RuntimeWarning)
        return self._add_point_array(scalars, name, set_active=set_active, deep=deep)


    def get_data_range(self, arr=None, preference='cell'):
        """Get the non-NaN min and max of a named scalar array

        Parameters
        ----------
        arr : str, np.ndarray, optional
            The name of the array to get the range. If None, the active scalar
            is used

        preference : str, optional
            When scalars is specified, this is the perfered scalar type to
            search for in the dataset.  Must be either ``'point'``, ``'cell'``,
            or ``'field'``.

        """
        if arr is None:
            # use active scalar array
            _, arr = self.active_scalar_info
        if isinstance(arr, str):
            arr = get_array(self, arr, preference=preference)
        # If array has no tuples return a NaN range
        if arr is None or arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
            return (np.nan, np.nan)
        # Use the array range
        return np.nanmin(arr), np.nanmax(arr)


    def points_to_double(self):
        """ Makes points double precision """
        if self.points.dtype != np.double:
            self.points = self.points.astype(np.double)


    def rotate_x(self, angle):
        """
        Rotates mesh about the x-axis.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the x-axis.

        """
        axis_rotation(self.points, angle, inplace=True, axis='x')


    def rotate_y(self, angle):
        """
        Rotates mesh about the y-axis.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the y-axis.

        """
        axis_rotation(self.points, angle, inplace=True, axis='y')


    def rotate_z(self, angle):
        """
        Rotates mesh about the z-axis.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the z-axis.

        """
        axis_rotation(self.points, angle, inplace=True, axis='z')


    def translate(self, xyz):
        """
        Translates the mesh.

        Parameters
        ----------
        xyz : list or np.ndarray
            Length 3 list or array.

        """
        self.points += np.asarray(xyz)


    def transform(self, trans):
        """
        Compute a transformation in place using a 4x4 transform.

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
            if trans.shape[0] != 4 or trans.shape[1] != 4:
                raise Exception('Transformation array must be 4x4')
            t = trans
        else:
            raise TypeError('Input transform must be either:\n'
                            '\tvtk.vtkMatrix4x4\n'
                            '\tvtk.vtkTransform\n'
                            '\t4x4 np.ndarray\n')

        x = (self.points*t[0, :3]).sum(1) + t[0, -1]
        y = (self.points*t[1, :3]).sum(1) + t[1, -1]
        z = (self.points*t[2, :3]).sum(1) + t[2, -1]

        # overwrite points
        self.points[:, 0] = x
        self.points[:, 1] = y
        self.points[:, 2] = z


    def _cell_array(self, name=None):
        """
        Returns the cell scalars of a vtk object

        Parameters
        ----------
        name : str
            Name of cell scalars to retrive.

        Returns
        -------
        scalars : np.ndarray
            Numpy array of scalars

        """
        if name is None:
            # use active scalar array
            field, name = self.active_scalar_info
            if field != CELL_DATA_FIELD:
                raise RuntimeError('Must specify an array to fetch.')

        vtkarr = self.GetCellData().GetAbstractArray(name)
        if vtkarr is None:
            raise AssertionError('({}) is not a cell scalar'.format(name))

        # numpy does not support bit array data types
        if isinstance(vtkarr, vtk.vtkBitArray):
            vtkarr = vtk_bit_array_to_char(vtkarr)
            if name not in self._cell_bool_array_names:
                self._cell_bool_array_names.append(name)

        array = convert_array(vtkarr)
        if array.dtype == np.uint8 and name in self._cell_bool_array_names:
            array = array.view(np.bool)
        return array


    def _add_cell_array(self, scalars, name, set_active=False, deep=True):
        """
        Adds cell scalars to the vtk object.

        Parameters
        ----------
        scalars : numpy.ndarray
            Numpy array of scalars.  Must match number of points.

        name : str
            Name of point scalars to add.

        set_active : bool, optional
            Sets the scalars to the active plotting scalars.  Default False.

        deep : bool, optional
            Does not copy scalars when False.  A reference to the scalars
            must be kept to avoid a segfault.

        """
        if scalars is None:
            raise TypeError('Empty array unable to be added')

        if not isinstance(scalars, np.ndarray):
            scalars = np.array(scalars)

        if scalars.shape[0] != self.n_cells:
            raise Exception('Number of scalars must match the number of cells (%d)'
                            % self.n_cells)

        if not scalars.flags.c_contiguous:
            raise AssertionError('Array must be contigious')
        if scalars.dtype == np.bool:
            scalars = scalars.view(np.uint8)
            self._cell_bool_array_names.append(name)

        vtkarr = convert_array(scalars, deep=deep)
        vtkarr.SetName(name)
        self.GetCellData().AddArray(vtkarr)
        if set_active or self.active_scalar_info[1] is None:
            self.GetCellData().SetActiveScalars(name)
            self._active_scalar_info = [CELL_DATA_FIELD, name]


    def _add_cell_scalar(self, scalars, name, set_active=False, deep=True):
        """DEPRECATED: Please use `_add_cell_array`"""
        warnings.warn('Deprecation Warning: `_add_cell_scalar` is now `_add_cell_array`', RuntimeWarning)
        return self._add_cell_array(scalars, name, set_active=set_active, deep=deep)


    def copy_meta_from(self, ido):
        """Copies pyvista meta data onto this object from another object"""
        self._active_scalar_info = ido.active_scalar_info
        self._active_vectors_info = ido.active_vectors_info
        if hasattr(ido, '_textures'):
            self._textures = ido._textures


    @property
    def point_arrays(self):
        """ Returns the all point arrays """
        pdata = self.GetPointData()
        narr = pdata.GetNumberOfArrays()

        # Update data if necessary
        if hasattr(self, '_point_arrays'):
            keys = list(self._point_arrays.keys())
            if narr == len(keys):
                if keys:
                    if self._point_arrays[keys[0]].shape[0] == self.n_points:
                        return self._point_arrays
                else:
                    return self._point_arrays

        # dictionary with callbacks
        self._point_arrays = PointScalarsDict(self)

        for i in range(narr):
            name = pdata.GetArrayName(i)
            if name is None or len(name) < 1:
                name = 'Point Array {}'.format(i)
                pdata.GetAbstractArray(i).SetName(name)
            self._point_arrays[name] = self._point_array(name)

        self._point_arrays.enable_callback()
        return self._point_arrays



    def _remove_array(self, field, key):
        """internal helper to remove a single array by name from each field"""
        field = parse_field_choice(field)
        if field == POINT_DATA_FIELD:
            self.GetPointData().RemoveArray(key)
        elif field == CELL_DATA_FIELD:
            self.GetCellData().RemoveArray(key)
        elif field == FIELD_DATA_FIELD:
            self.GetFieldData().RemoveArray(key)
        else:
            raise NotImplementedError('Not able to remove arrays from the ({}) data fiedl'.format(field))
        return


    def clear_point_arrays(self):
        """ removes all point arrays """
        keys = self.point_arrays.keys()
        for key in keys:
            self._remove_array(POINT_DATA_FIELD, key)


    def clear_cell_arrays(self):
        """ removes all cell arrays """
        keys = self.cell_arrays.keys()
        for key in keys:
            self._remove_array(CELL_DATA_FIELD, key)


    def clear_arrays(self):
        """ removes all arrays from point/cell/field data """
        self.clear_point_arrays()
        self.clear_cell_arrays()
        self.clear_field_arrays()


    @property
    def cell_arrays(self):
        """ Returns the all cell arrays """
        cdata = self.GetCellData()
        narr = cdata.GetNumberOfArrays()

        # Update data if necessary
        if hasattr(self, '_cell_arrays'):
            keys = list(self._cell_arrays.keys())
            if narr == len(keys):
                if keys:
                    if self._cell_arrays[keys[0]].shape[0] == self.n_cells:
                        return self._cell_arrays
                else:
                    return self._cell_arrays

        # dictionary with callbacks
        self._cell_arrays = CellScalarsDict(self)

        for i in range(narr):
            name = cdata.GetArrayName(i)
            if name is None or len(name) < 1:
                name = 'Cell Array {}'.format(i)
                cdata.GetAbstractArray(i).SetName(name)
            self._cell_arrays[name] = self._cell_array(name)

        self._cell_arrays.enable_callback()
        return self._cell_arrays


    @property
    def n_points(self):
        """The number of points in the entire dataset"""
        return self.GetNumberOfPoints()


    @property
    def n_cells(self):
        """The number of cells in the entire dataset"""
        return self.GetNumberOfCells()


    @property
    def number_of_points(self):  # pragma: no cover
        """ returns the number of points """
        return self.GetNumberOfPoints()


    @property
    def number_of_cells(self):  # pragma: no cover
        """ returns the number of cells """
        return self.GetNumberOfCells()


    @property
    def bounds(self):
        """
        bounding box of this dataset in the form
        (xmin,xmax, ymin,ymax, zmin,zmax)
        """
        return list(self.GetBounds())


    @property
    def length(self):
        """the length of the diagonal of the bounding box"""
        return self.GetLength()


    @property
    def center(self):
        """ Center of the bounding box """
        return list(self.GetCenter())


    @property
    def extent(self):
        """ The range of the bounding box """
        if hasattr(self, 'GetExtent'):
            return list(self.GetExtent())


    @extent.setter
    def extent(self, extent):
        """ The range of the bounding box """
        if hasattr(self, 'SetExtent'):
            return self.SetExtent(extent)
        else:
            raise AttributeError('This mesh type does not handle extents.')


    @property
    def volume(self):
        """
        Mesh volume

        Returns
        -------
        volume : float
            Total volume of the mesh.

        """
        sizes = self.compute_cell_sizes(length=False, area=False, volume=True)
        return np.sum(sizes.cell_arrays['Volume'])


    def get_array(self, name, preference='cell', info=False):
        """ Searches both point, cell and field data for an array """
        return get_array(self, name, preference=preference, info=info)


    def __getitem__(self, index):
        """ Searches both point, cell, and field data for an array """
        if isinstance(index, collections.Iterable) and not isinstance(index, str):
            name, preference = index[0], index[1]
        elif isinstance(index, str):
            name = index
            preference = 'cell'
        else:
            raise KeyError('Index ({}) not understood. Index must be a string name or a tuple of string name and string preference.'.format(index))
        return self.get_array(name, preference=preference, info=False)


    def __setitem__(self, name, scalars):
        """Add/set an array in the point_arrays, or cell_arrays depending on the
        array's length, or specified mode.
        """
        # First check points - think of case with vertex cells
        #   there would be the same number of cells as points but we'd want
        #   the data to be on the nodes.
        if scalars is None:
            raise TypeError('Empty array unable to be added')
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
        """The number of scalar arrays present in the dataset"""
        n = self.GetPointData().GetNumberOfArrays()
        n += self.GetCellData().GetNumberOfArrays()
        n += self.GetFieldData().GetNumberOfArrays()
        return n


    @property
    def n_scalars(self):
        """DEPRECATED: Please use `n_arrays`"""
        warnings.warn('Deprecation Warning: `n_scalars` is now `n_arrays`', RuntimeWarning)
        return self.n_arrays


    @property
    def array_names(self):
        """A list of scalar names for the dataset. This makes
        sure to put the active scalar's name first in the list."""
        names = []
        for i in range(self.GetPointData().GetNumberOfArrays()):
            names.append(self.GetPointData().GetArrayName(i))
        for i in range(self.GetCellData().GetNumberOfArrays()):
            names.append(self.GetCellData().GetArrayName(i))
        for i in range(self.GetFieldData().GetNumberOfArrays()):
            names.append(self.GetFieldData().GetArrayName(i))
        try:
            names.remove(self.active_scalar_name)
            names.insert(0, self.active_scalar_name)
        except ValueError:
            pass
        return names


    @property
    def scalar_names(self):
        """DEPRECATED: Please use `array_names`"""
        warnings.warn('Deprecation Warning: `scalar_names` is now `array_names`', RuntimeWarning)
        return self.array_names


    def _get_attrs(self):
        """An internal helper for the representation methods"""
        attrs = []
        attrs.append(("N Cells", self.GetNumberOfCells(), "{}"))
        attrs.append(("N Points", self.GetNumberOfPoints(), "{}"))
        bds = self.bounds
        fmt = "{}, {}".format(pyvista.FLOAT_FORMAT, pyvista.FLOAT_FORMAT)
        attrs.append(("X Bounds", (bds[0], bds[1]), fmt))
        attrs.append(("Y Bounds", (bds[2], bds[3]), fmt))
        attrs.append(("Z Bounds", (bds[4], bds[5]), fmt))
        # if self.n_cells <= pyvista.REPR_VOLUME_MAX_CELLS and self.n_cells > 0:
        #     attrs.append(("Volume", (self.volume), pyvista.FLOAT_FORMAT))
        return attrs


    def _repr_html_(self):
        """A pretty representation for Jupyter notebooks that includes header
        details and information about all scalar arrays"""
        fmt = ""
        if self.n_arrays > 0:
            fmt += "<table>"
            fmt += "<tr><th>Header</th><th>Data Arrays</th></tr>"
            fmt += "<tr><td>"
        # Get the header info
        fmt += self.head(display=False, html=True)
        # Fill out scalar arrays
        if self.n_arrays > 0:
            fmt += "</td><td>"
            fmt += "\n"
            fmt += "<table>\n"
            titles = ["Name", "Field", "Type", "N Comp", "Min", "Max"]
            fmt += "<tr>" + "".join(["<th>{}</th>".format(t) for t in titles]) + "</tr>\n"
            row = "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>\n"
            row = "<tr>" + "".join(["<td>{}</td>" for i in range(len(titles))]) + "</tr>\n"

            def format_array(name, arr, field):
                """internal helper to foramt array information for printing"""
                dl, dh = self.get_data_range(arr)
                dl = pyvista.FLOAT_FORMAT.format(dl)
                dh = pyvista.FLOAT_FORMAT.format(dh)
                if name == self.active_scalar_info[1]:
                    name = '<b>{}</b>'.format(name)
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
        """Object representation"""
        return self.head(display=False, html=False)


    def __str__(self):
        """Object string representation"""
        return self.head(display=False, html=False)


    def overwrite(self, mesh):
        """
        Overwrites this mesh inplace with the new mesh's geometries and data

        Parameters
        ----------
        mesh : vtk.vtkDataSet
            The overwriting mesh.

        """
        self.deep_copy(mesh)
        if is_pyvista_dataset(mesh):
            self.copy_meta_from(mesh)


    def cast_to_unstructured_grid(self):
        """Get a new representation of this object as an
        :class:`pyvista.UnstructuredGrid`
        """
        alg = vtk.vtkAppendFilter()
        alg.AddInputData(self)
        alg.Update()
        return pyvista.filters._get_output(alg)


    @property
    def quality(self):
        """
        Returns cell quality using PyANSYS. Computes the minimum scaled
        jacobian of each cell.  Cells that have values below 0 are invalid for
        a finite element analysis.

        Note
        ----
        This casts the input to an unstructured grid

        Returns
        -------
        cellquality : np.ndarray
            Minimum scaled jacobian of each cell.  Ranges from -1 to 1.

        Notes
        -----
        Requires pyansys to be installed.

        """
        try:
            import pyansys
        except ImportError:
            raise Exception('Install pyansys for this function')
        if not isinstance(self, pyvista.UnstructuredGrid):
            dataset = self.cast_to_unstructured_grid()
        else:
            dataset = self
        return pyansys.CellQuality(dataset)



class _ScalarsDict(dict):
    """Internal helper for scalars dictionaries"""
    def __init__(self, data):
        self.data = proxy(data)
        dict.__init__(self)
        self.callback_enabled = False
        self.remover = None
        self.modifier = None


    def enable_callback(self):
        """Enable callbacks to be set True"""
        self.callback_enabled = True


    def adder(self, scalars, name, set_active=False, deep=True):
        raise NotImplementedError()


    def pop(self, key):
        """Get and remove an element by key name"""
        arr = dict.pop(self, key).copy()
        self.remover(key)
        return arr


    def update(self, data):
        """
        Update this dictionary with th key-value pairs from a given
        dictionary
        """
        if not isinstance(data, (dict, pyvista.Table)):
            raise TypeError('Data to update must be in a dictionary or PyVista Table.')
        for k, v in data.items():
            arr = np.array(v)
            try:
                self[k] = arr
            except TypeError:
                logging.warning("Values under key ({}) not supported by VTK".format(k))
        return


    def __setitem__(self, key, val):
        """ overridden to assure data is contigious """
        if isinstance(val, (list, tuple)):
            val = np.array(val)
        if self.callback_enabled:
            self.adder(val, key, deep=False)
        dict.__setitem__(self, key, val)
        self.modifier()


    def __delitem__(self, key):
        """Remove item by key name"""
        self.remover(key)
        return dict.__delitem__(self, key)


class CellScalarsDict(_ScalarsDict):
    """
    Updates internal cell data when an array is added or removed from
    the dictionary.
    """
    def __init__(self, data):
        _ScalarsDict.__init__(self, data)
        self.remover = lambda key: self.data._remove_array(CELL_DATA_FIELD, key)
        self.modifier = lambda *args: self.data.GetCellData().Modified()


    def adder(self, scalars, name, set_active=False, deep=True):
        self.data._add_cell_array(scalars, name, set_active=False, deep=deep)


class PointScalarsDict(_ScalarsDict):
    """
    Updates internal point data when an array is added or removed from
    the dictionary.
    """
    def __init__(self, data):
        _ScalarsDict.__init__(self, data)
        self.remover = lambda key: self.data._remove_array(POINT_DATA_FIELD, key)
        self.modifier = lambda *args: self.data.GetPointData().Modified()


    def adder(self, scalars, name, set_active=False, deep=True):
        self.data._add_point_array(scalars, name, set_active=False, deep=deep)


class FieldScalarsDict(_ScalarsDict):
    """
    Updates internal field data when an array is added or removed from
    the dictionary.
    """
    def __init__(self, data):
        _ScalarsDict.__init__(self, data)
        self.remover = lambda key: self.data._remove_array(FIELD_DATA_FIELD, key)
        self.modifier = lambda *args: self.data.GetFieldData().Modified()


    def adder(self, scalars, name, set_active=False, deep=True):
        self.data._add_field_array(scalars, name, deep=deep)


def axis_rotation(points, angle, inplace=False, deg=True, axis='z'):
    """ Rotates points angle ang (in deg) about an axis """
    axis = axis.lower()

    # Copy original array to if not inplace
    if not inplace:
        points = points.copy()

    # Convert angle to radians
    if deg:
        angle *= np.pi / 180

    if axis == 'x':
        y = points[:, 1] * np.cos(angle) - points[:, 2] * np.sin(angle)
        z = points[:, 1] * np.sin(angle) + points[:, 2] * np.cos(angle)
        points[:, 1] = y
        points[:, 2] = z
    elif axis == 'y':
        x = points[:, 0] * np.cos(angle) + points[:, 2] * np.sin(angle)
        z = - points[:, 0] * np.sin(angle) + points[:, 2] * np.cos(angle)
        points[:, 0] = x
        points[:, 2] = z
    elif axis == 'z':
        x = points[:, 0] * np.cos(angle) - points[:, 1] * np.sin(angle)
        y = points[:, 0] * np.sin(angle) + points[:, 1] * np.cos(angle)
        points[:, 0] = x
        points[:, 1] = y
    else:
        raise Exception('invalid axis.  Must be either "x", "y", or "z"')

    if not inplace:
        return points


class pyvista_ndarray(np.ndarray):
    """
    Links a numpy array with the vtk object the data is attached to.

    When the array is changed it triggers "Modified()" which updates
    all upstream objects, including any render windows holding the
    object.

    """

    def __new__(cls, input_array, proxy):
        obj = np.asarray(input_array).view(cls)
        cls.proxy = proxy
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __setitem__(self, coords, value):
        """ Update the array and update the vtk object """
        super(pyvista_ndarray, self).__setitem__(coords, value)
        self.proxy.Modified()
