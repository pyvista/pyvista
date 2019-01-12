"""
Attributes common to PolyData and Grid Objects
"""
import logging
from weakref import proxy

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')

import vtki
from vtki.utilities import get_scalar, POINT_DATA_FIELD, CELL_DATA_FIELD
from vtki import DataSetFilters



class Common(DataSetFilters):
    """ Methods in common to grid and surface objects"""

    def __init__(self, *args, **kwargs):
        self.references = []

    @property
    def active_scalar_info(self):
        if not hasattr(self, '_active_scalar_info'):
            self._active_scalar_info = [POINT_DATA_FIELD, None] # field and name
        field, name = self._active_scalar_info

        # rare error where scalar name isn't a valid scalar
        if name not in self.point_arrays:
            if name not in self.cell_arrays:
                name = None

        if name is None:
            if self.n_scalars < 1:
                return field, name
            # find some array in the set field
            parr = self.GetPointData().GetArrayName(0)
            carr = self.GetCellData().GetArrayName(0)
            if parr is not None:
                self._active_scalar_info = [POINT_DATA_FIELD, parr]
            elif carr is not None:
                self._active_scalar_info = [CELL_DATA_FIELD, carr]
        return self._active_scalar_info

    @property
    def points(self):
        """ returns a pointer to the points as a numpy object """
        return vtk_to_numpy(self.GetPoints().GetData())

    @points.setter
    def points(self, points):
        """ set points without copying """
        if not isinstance(points, np.ndarray):
            raise TypeError('Points must be a numpy array')
        vtk_points = vtki.vtk_points(points, False)
        self.SetPoints(vtk_points)
        #self._point_ref = points

    def set_active_scalar(self, name, preference='cell'):
        """Finds the scalar by name and appropriately sets it as active"""
        arr, field = get_scalar(self, name, preference=preference, info=True)
        if field == POINT_DATA_FIELD:
            self.GetPointData().SetActiveScalars(name)
        elif field == CELL_DATA_FIELD:
            self.GetCellData().SetActiveScalars(name)
        else:
            raise RuntimeError('Data field ({}) no useable'.format(field))
        self._active_scalar_info = [field, name]

    def change_scalar_name(self, old_name, new_name, preference='cell'):
        """Changes array name by searching for the array then renaming it"""
        _, field = get_scalar(self, old_name, preference=preference, info=True)
        if field == POINT_DATA_FIELD:
            self.GetPointData().GetArray(old_name).SetName(new_name)
        elif field == CELL_DATA_FIELD:
            self.GetCellData().GetArray(old_name).SetName(new_name)
        else:
            raise RuntimeError('Array not found.')
        if self.active_scalar_info[1] == old_name:
            self.set_active_scalar(new_name, preference=field)


    @property
    def active_scalar(self):
        field, name = self.active_scalar_info
        if name is None:
            return None
        if field == POINT_DATA_FIELD:
            return self._point_scalar(name)
        elif field == CELL_DATA_FIELD:
            return self._cell_scalar(name)

    def _point_scalar(self, name=None):
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
        vtkarr = self.GetPointData().GetArray(name)
        assert vtkarr is not None, '%s is not a point scalar' % name
        array = vtk_to_numpy(vtkarr)
        if array.dtype == np.uint8:
            array = array.view(np.bool)
        return array

    def _add_point_scalar(self, scalars, name, setactive=False, deep=True):
        """
        Adds point scalars to the mesh

        Parameters
        ----------
        scalars : numpy.ndarray
            Numpy array of scalars.  Must match number of points.

        name : str
            Name of point scalars to add.

        setactive : bool, optional
            Sets the scalars to the active plotting scalars.  Default False.

        deep : bool, optional
            Does not copy scalars when False.  A reference to the scalars
            must be kept to avoid a segfault.

        """
        if not isinstance(scalars, np.ndarray):
            raise TypeError('Input must be a numpy.ndarray')

        if scalars.shape[0] != self.n_points:
            raise Exception('Number of scalars must match the number of ' +
                            'points')
        if scalars.dtype == np.bool:
            scalars = scalars.view(np.uint8)

        # assert scalars.flags.c_contiguous, 'array must be contigious'
        if not scalars.flags.c_contiguous:
            scalars = np.ascontiguousarray(scalars)

        vtkarr = numpy_to_vtk(scalars, deep=deep)
        vtkarr.SetName(name)
        self.GetPointData().AddArray(vtkarr)
        if setactive:
            self.GetPointData().SetActiveScalars(name)
            self._active_scalar_info = [POINT_DATA_FIELD, name]

    def plot(self, **args):
        """
        Adds a vtk unstructured, structured, or polymesh to the plotting object

        Parameters
        ----------
        mesh : vtk unstructured, structured, or polymesh
            A vtk unstructured, structured, or polymesh to plot.

        color : string or 3 item list, optional, defaults to white
            Either a string, rgb list, or hex color string.  For example:
                color='white'
                color='w'
                color=[1, 1, 1]
                color='#FFFFFF'

            Color will be overridden when scalars are input.

        style : string, optional
            Visualization style of the vtk mesh.  One for the following:
                style='surface'
                style='wireframe'
                style='points'

            Defaults to 'surface'

        scalars : numpy array, optional
            Scalars used to "color" the mesh.  Accepts an array equal to the
            number of cells or the number of points in the mesh.  Array should
            be sized as a single vector.

        rng : 2 item list, optional
            Range of mapper for scalars.  Defaults to minimum and maximum of
            scalars array.  Example: [-1, 2]

        stitle : string, optional
            Scalar title.  By default there is no scalar legend bar.  Setting
            this creates the legend bar and adds a title to it.  To create a
            bar with no title, use an empty string (i.e. '').

        showedges : bool, optional
            Shows the edges of a mesh.  Does not apply to a wireframe
            representation.

        psize : float, optional
            Point size.  Applicable when style='points'.  Default 5.0

        opacity : float, optional
            Opacity of mesh.  Should be between 0 and 1.  Default 1.0

        linethick : float, optional
            Thickness of lines.  Only valid for wireframe and surface
            representations.  Default None.

        flipscalars : bool, optional
            Flip scalar display approach.  Default is red is minimum and blue
            is maximum.

        lighting : bool, optional
            Enable or disable Z direction lighting.  True by default.

        ncolors : int, optional
            Number of colors to use when displaying scalars.

        interpolatebeforemap : bool, default False
            Enabling makes for a smoother scalar display.  Default False

        screenshot : str, default None
            Takes a screenshot when window is closed when a filename is
            entered as this parameter.

        full_screen : bool, optional
            Opens window in full screen.  When enabled, ignores window_size.
            Default False.

        Returns
        -------
        cpos : list
            Camera position, focal point, and view up.

        Examples
        --------
        >>> # take screenshot without opening window
        >>> mesh.plot(off_screen=True, screenshot='mesh_picture.png')

        """
        return vtki.plot(self, **args)

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
            t = vtki.trans_from_matrix(trans)
        elif isinstance(trans, vtk.vtkTransform):
            t = vtki.trans_from_matrix(trans.GetMatrix())
        elif isinstance(trans, np.ndarray):
            if trans.shape[0] != 4 or trans.shape[1] != 4:
                raise Exception('Transformation array must be 4x4')
            t = trans
        else:
            raise TypeError('Input transform must be either:\n'
                            + '\tvtk.vtkMatrix4x4\n'
                            + '\tvtk.vtkTransform\n'
                            + '\t4x4 np.ndarray\n')

        x = (self.points*t[0, :3]).sum(1) + t[0, -1]
        y = (self.points*t[1, :3]).sum(1) + t[1, -1]
        z = (self.points*t[2, :3]).sum(1) + t[2, -1]

        # overwrite points
        self.points[:, 0] = x
        self.points[:, 1] = y
        self.points[:, 2] = z

    def _cell_scalar(self, name=None):
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
        vtkarr = self.GetCellData().GetArray(name)
        array = vtk_to_numpy(vtkarr)
        if array.dtype == np.uint8:
            array = array.view(np.bool)
        return array

    def _add_cell_scalar(self, scalars, name, setactive=False, deep=True):
        """
        Adds cell scalars to the vtk object.

        Parameters
        ----------
        scalars : numpy.ndarray
            Numpy array of scalars.  Must match number of points.

        name : str
            Name of point scalars to add.

        setactive : bool, optional
            Sets the scalars to the active plotting scalars.  Default False.

        deep : bool, optional
            Does not copy scalars when False.  A reference to the scalars
            must be kept to avoid a segfault.

        """
        if not isinstance(scalars, np.ndarray):
            raise TypeError('Input must be a numpy.ndarray')

        if scalars.shape[0] != self.n_cells:
            raise Exception('Number of scalars must match the number of cells (%d)'
                            % self.n_cells)

        assert scalars.flags.c_contiguous, 'Array must be contigious'
        if scalars.dtype == np.bool:
            scalars = scalars.view(np.uint8)

        vtkarr = numpy_to_vtk(scalars, deep=deep)
        vtkarr.SetName(name)
        self.GetCellData().AddArray(vtkarr)
        if setactive:
            self.GetCellData().SetActiveScalars(name)
            self._active_scalar_info = [CELL_DATA_FIELD, name]

    def copy_meta_from(self, ido):
        """Copies vtki meta data onto this object from another object"""
        self._active_scalar_info = ido.active_scalar_info

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
            newobject.DeepCopy(self)
        else:
            newobject.ShallowCopy(self)
        newobject.copy_meta_from(self)
        return newobject


    def _remove_point_scalar(self, key):
        """ removes point scalars from point data """
        self.GetPointData().RemoveArray(key)

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
                    if self._point_arrays[keys[0]].size == self.n_points:
                        return self._point_arrays
                else:
                    return self._point_arrays

        # dictionary with callbacks
        self._point_arrays = PointScalarsDict(self)

        for i in range(narr):
            name = pdata.GetArrayName(i)
            self._point_arrays[name] = self._point_scalar(name)

        self._point_arrays.enable_callback()
        return self._point_arrays

    def _remove_cell_scalar(self, key):
        """ removes cell scalars """
        self.GetCellData().RemoveArray(key)

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
                    if self._cell_arrays[keys[0]].size == self.n_cells:
                        return self._cell_arrays
                else:
                    return self._cell_arrays

        # dictionary with callbacks
        self._cell_arrays = CellScalarsDict(self)

        for i in range(narr):
            name = cdata.GetArrayName(i)
            self._cell_arrays[name] = self._cell_scalar(name)

        self._cell_arrays.enable_callback()
        return self._cell_arrays

    @property
    def n_points(self):
        return self.GetNumberOfPoints()

    @property
    def n_cells(self):
        return self.GetNumberOfCells()

    @property
    def number_of_points(self):
        """ returns the number of points """
        return self.GetNumberOfPoints()

    @property
    def number_of_cells(self):
        """ returns the number of cells """
        return self.GetNumberOfCells()

    @property
    def bounds(self):
        return self.GetBounds()

    @property
    def center(self):
        return self.GetCenter()

    @property
    def extent(self):
        return self.GetExtent()

    def get_data_range(self, arr=None, preference='cell'):
        if arr is None:
            # use active scalar array
            _, arr = self.active_scalar_info
        if isinstance(arr, str):
            arr = get_scalar(self, arr, preference=preference)
        return np.nanmin(arr), np.nanmax(arr)

    @property
    def n_scalars(self):
        return self.GetPointData().GetNumberOfArrays() + \
               self.GetCellData().GetNumberOfArrays()

    def _get_attrs(self):
        """An internal helper for the representation methods"""
        attrs = []
        attrs.append(("N Cells", self.GetNumberOfCells(), "{}"))
        attrs.append(("N Points", self.GetNumberOfPoints(), "{}"))
        bds = self.bounds
        attrs.append(("X Bounds", (bds[0], bds[1]), "{:.3f}, {:.3f}"))
        attrs.append(("Y Bounds", (bds[2], bds[3]), "{:.3f}, {:.3f}"))
        attrs.append(("Z Bounds", (bds[4], bds[5]), "{:.3f}, {:.3f}"))
        return attrs

    def _repr_html_(self):
        """A pretty representation for Jupyter notebooks"""
        fmt = ""
        if self.n_scalars > 0:
            fmt += "<table>"
            fmt += "<tr><th>Information</th><th>Data Arrays</th></tr>"
            fmt += "<tr><td>"
        fmt += "\n"
        fmt += "<table>\n"
        fmt += "<tr><th>{}</th><th>Values</th></tr>\n".format(self.GetClassName())
        row = "<tr><td>{}</td><td>{}</td></tr>\n"

        # now make a call on the object to get its attributes as a list of len 2 tuples
        for attr in self._get_attrs():
            try:
                fmt += row.format(attr[0], attr[2].format(*attr[1]))
            except:
                fmt += row.format(attr[0], attr[2].format(attr[1]))

        fmt += "</table>\n"
        fmt += "\n"
        if self.n_scalars > 0:
            fmt += "</td><td>"
            fmt += "\n"
            fmt += "<table>\n"
            row = "<tr><th>{}</th><th>{}</th><th>{}</th><th>{}</th><th>{}</th></tr>\n"
            fmt += row.format("Name", "Field", "Type", "Min", "Max")
            row = "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.3e}</td><td>{:.3e}</td></tr>\n"

            def format_array(key, field):
                arr = get_scalar(self, key)
                dl, dh = self.get_data_range(key)
                if key == self.active_scalar_info[1]:
                    key = '<b>{}</b>'.format(key)
                return row.format(key, field, arr.dtype, dl, dh)

            for i in range(self.GetPointData().GetNumberOfArrays()):
                key = self.GetPointData().GetArrayName(i)
                fmt += format_array(key, field='Points')
            for i in range(self.GetCellData().GetNumberOfArrays()):
                key = self.GetCellData().GetArrayName(i)
                fmt += format_array(key, field='Cells')
            fmt += "</table>\n"
            fmt += "\n"
            fmt += "</td></tr> </table>"
        return fmt


class CellScalarsDict(dict):
    """
    Updates internal cell data when an array is added or removed from
    the dictionary.
    """

    def __init__(self, data):
        self.data = proxy(data)
        dict.__init__(self)
        self.callback_enabled = False

    def enable_callback(self):
        self.callback_enabled = True

    def __setitem__(self, key, val):
        """ overridden to assure data is contigious """
        if self.callback_enabled:
            self.data._add_cell_scalar(val, key, deep=False)
        dict.__setitem__(self, key, val)

    def __delitem__(self, key):
        self.data._remove_cell_scalar(key)
        return dict.__delitem__(self, key)


class PointScalarsDict(dict):
    """
    Updates internal point data when an array is added or removed from
    the dictionary.
    """

    def __init__(self, data):
        self.data = proxy(data)
        dict.__init__(self)
        self.callback_enabled = False

    def enable_callback(self):
        self.callback_enabled = True

    def __setitem__(self, key, val):
        """ overridden to assure data is contigious """
        if self.callback_enabled:
            self.data._add_point_scalar(val, key, deep=False)
        dict.__setitem__(self, key, val)

    def __delitem__(self, key):
        self.data._remove_point_scalar(key)
        return dict.__delitem__(self, key)


def axis_rotation(p, ang, inplace=False, deg=True, axis='z'):
    """ Rotates points p angle ang (in deg) about an axis """
    axis = axis.lower()

    # Copy original array to if not inplace
    if not inplace:
        p = p.copy()

    # Convert angle to radians
    if deg:
        ang *= np.pi / 180

    if axis == 'x':
        y = p[:, 1] * np.cos(ang) - p[:, 2] * np.sin(ang)
        z = p[:, 1] * np.sin(ang) + p[:, 2] * np.cos(ang)
        p[:, 1] = y
        p[:, 2] = z
    elif axis == 'y':
        x = p[:, 0] * np.cos(ang) + p[:, 2] * np.sin(ang)
        z = - p[:, 0] * np.sin(ang) + p[:, 2] * np.cos(ang)
        p[:, 0] = x
        p[:, 2] = z
    elif axis == 'z':
        x = p[:, 0] * np.cos(ang) - p[:, 1] * np.sin(ang)
        y = p[:, 0] * np.sin(ang) + p[:, 1] * np.cos(ang)
        p[:, 0] = x
        p[:, 1] = y
    else:
        raise Exception('invalid axis.  Must be either "x", "y", or "z"')

    if not inplace:
        return p
