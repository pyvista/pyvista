"""
Attributes common to PolyData and Grid Objects
"""
import numpy as np
import vtkInterface
import logging

log = logging.getLogger(__name__)
log.setLevel('CRITICAL')

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    from vtk.util.numpy_support import numpy_to_vtk
except ImportError:
    pass


class Common(object):
    """ Methods in common to grid and surface objects"""

    def __init__(self, *args, **kwargs):
        self.references = []

    @property
    def points(self):
        """ returns a pointer to the points as a numpy object """
        return vtk_to_numpy(self.GetPoints().GetData())

    @points.setter
    def points(self, newpoints):
        """ set the new points without copying """
        if not isinstance(newpoints, np.ndarray):
            raise TypeError('Points must be a numpy array')
        self.SetNumpyPoints(newpoints, False)

        # keep a reference as pointer might be deleted
        if not hasattr(self, 'references'):
            self.references = [newpoints]
        else:
            self.references.append(newpoints)

    def GetNumpyPoints(self, dtype=None, deep=False):
        """
        Returns points as a numpy array

        This performs the same actions as self.points with default parameters

        Parameters
        ----------
        dtype : optional
            Data type to specify.  Will not copy data if points are already
            in the specified data type.

        deep : bool, optional
            Copies points when True

        Returns
        -------
        points : np.ndarray
            Numpy array of points.

        """
        points = vtk_to_numpy(self.GetPoints().GetData())

        if dtype:
            if points.dtype != dtype:
                return points.astype(dtype)

        # Copy if requested
        if deep:
            return points.copy()
        else:
            return points

    def SetNumpyPoints(self, points, deep=True):
        """
        Overwrites existing points with new points.

        Parameters
        ----------
        points : np.ndarray
            Points to set to.

        deep : bool, optional
            Copies points when True.  When False, does not copy, but
            a reference to the original points must be kept or else
            Python will segfault.

        """
        vtkPoints = vtkInterface.MakevtkPoints(points, deep)
        self.SetPoints(vtkPoints)

    def GetPointScalars(self, name):
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
        vtkarr = self.GetPointData().GetArray(name)

        if vtkarr:
            array = vtk_to_numpy(vtkarr)
            if array.dtype == np.int8:
                array = array.astype(np.bool)
            return array
        else:
            return None

    def AddPointScalars(self, scalars, name, setactive=False, deep=True):
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
        if scalars.shape[0] != self.GetNumberOfPoints():
            raise Exception('Number of scalars must match the number of ' +
                            'points')
        if scalars.dtype == np.bool:
            scalars = scalars.astype(np.int8)

        if not scalars.flags.c_contiguous:
            scalars = np.ascontiguousarray(scalars)
        vtkarr = numpy_to_vtk(scalars, deep=deep)
        vtkarr.SetName(name)
        self.GetPointData().AddArray(vtkarr)
        if setactive:
            self.GetPointData().SetActiveScalars(name)

    def Plot(self, **args):
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

        Returns
        -------
        cpos : list
            Camera position.
        """
        return vtkInterface.Plot(self, **args)

    def MakeDouble(self):
        """ Makes points double precision """
        if self.points.dtype != np.double:
            self.points = self.points.astype(np.double)

    def RotateX(self, angle):
        """
        Rotates mesh about the x-axis.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the x-axis.

        """
        AxisRotation(self.points, angle, inplace=True, axis='x')

    def RotateY(self, angle):
        """
        Rotates mesh about the y-axis.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the y-axis.

        """
        AxisRotation(self.points, angle, inplace=True, axis='y')

    def RotateZ(self, angle):
        """
        Rotates mesh about the z-axis.

        Parameters
        ----------
        angle : float
            Angle in degrees to rotate about the z-axis.

        """
        AxisRotation(self.points, angle, inplace=True, axis='z')

    def Translate(self, xyz):
        """
        Translates the mesh.

        Parameters
        ----------
        xyz : list or np.ndarray
            Length 3 list or array.

        """
        self.points += np.asarray(xyz)

    def ApplyTransformationInPlace(self, trans):
        """
        Apply a transformation in place.

        Parameters
        ----------
        trans : vtk.vtkMatrix4x4, vtk.vtkTransform, or np.ndarray
            Accepts a vtk transformation object or a 4x4 transformation matrix.

        """
        # work with mulitple input types
        if isinstance(trans, vtk.vtkMatrix4x4):
            t = vtkInterface.TransFromMatrix(trans)
        elif isinstance(trans, vtk.vtkTransform):
            t = vtkInterface.TransFromMatrix(trans.GetMatrix())
        elif isinstance(trans, np.ndarray):
            if trans.shape[1] != 4:
                raise Exception('Invalid input shape')
            t = trans
        else:
            raise Exception('Input transform must be either:\n'
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

    def CheckArrayExists(self, name):
        """
        Returns True if a point array exists in a vtk object

        Parameters
        ----------
        name : str
            String of the point scalar to check.

        Returns
        -------
        exists : bool
            True when array exists and False when it does not.
        """
        return bool(self.GetPointData().GetArray(name))

    def GetCellScalars(self, name):
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
        vtkarr = self.GetCellData().GetArray(name)
        if vtkarr:
            array = vtk_to_numpy(vtkarr)
            if array.dtype == np.int8:
                array = array.astype(np.bool)
            return array
        else:
            return None

    def AddCellScalars(self, scalars, name, setactive=True, deep=True):
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
        if scalars.shape[0] != self.GetNumberOfCells():
            raise Exception('Number of scalars must match the number of cells')

        if not scalars.flags.c_contiguous:
            scalars = np.ascontiguousarray(scalars)

        if scalars.dtype == np.bool:
            scalars = scalars.astype(np.int8)

        vtkarr = numpy_to_vtk(scalars, deep=deep)
        vtkarr.SetName(name)
        self.GetCellData().AddArray(vtkarr)
        if setactive:
            self.GetCellData().SetActiveScalars(name)

    def Copy(self, deep=True):
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
        return newobject

    def __del__(self):
        log.debug('Object collected')


def AxisRotation(p, ang, inplace=False, deg=True, axis='z'):
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
