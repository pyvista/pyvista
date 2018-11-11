"""
Supporting functions for polydata and grid objects

"""
import warnings
import vtkInterface
import numpy as np
import ctypes

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtkIdTypeArray
    from vtk.util.numpy_support import numpy_to_vtk
except Exception as e:
    warnings.warn(str(e))


"""
Various utility functions for plotting and vtkInterface objects
"""
def GetPointScalars(mesh, name):
    """ Returns point scalars of a vtk object """
    vtkarr = mesh.GetPointData().GetArray(name)

    if vtkarr:
        array = vtk_to_numpy(vtkarr)
        if array.dtype == np.int8:
            array = array.astype(np.bool)
        return array
    else:
        return None


def MakevtkPoints(points, deep=True):
    """ Convert numpy points to a vtkPoints object """

    # Data checking
    if not points.flags['C_CONTIGUOUS']:
        points = np.ascontiguousarray(points)

    vtkpts = vtk.vtkPoints()
    vtkpts.SetData(numpy_to_vtk(points, deep=deep))
    return vtkpts


def MakePointMesh(points, deep=True):
    """ Convert numpy points to vtkPoints """

    # Data checking
    if not points.flags['C_CONTIGUOUS']:
        points = np.ascontiguousarray(points)

    # Convert to vtk objects
    vtkpoints = vtk.vtkPoints()
    vtkpoints.SetData(numpy_to_vtk(points, deep=True))

    npoints = points.shape[0]

    pcell = np.vstack((np.ones(npoints, dtype=ctypes.c_int64),
                       np.arange(npoints, dtype=ctypes.c_int64))).ravel('F')

    # Convert to a vtk array
    vtkcells = vtk.vtkCellArray()
    vtkcells.SetCells(npoints, numpy_to_vtkIdTypeArray(pcell,
                                                       deep=True))

    # Create polydata object
    mesh = vtkInterface.PolyData()
    mesh.SetPoints(vtkpoints)
    mesh.SetPolys(vtkcells)

    # return cleaned mesh
    return mesh


def MakeLine(points):
    """
    Generates line from points.  Assumes points are ordered as line segments.

    Parameters
    ----------
    points : np.ndarray
        Points representing line segments.  For example, two line segments 
        would be represented as:

        np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])

    Returns
    -------
    lines : vtki.PolyData
        PolyData with lines and cells.

    Examples
    --------
    This example plots two line segments at right angles to each other line.

    >>> import vtkInterface as vtki
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])
    >>> lines = vtki.MakeLine(points)
    >>> lines.Plot()

    """
    # Assuming ordered points, create array defining line order
    npoints = points.shape[0] - 1
    lines = np.vstack((2 * np.ones(npoints, np.int),
                       np.arange(npoints),
                       np.arange(1, npoints + 1))).T.ravel()

    return vtkInterface.PolyData(points, lines)


def MeshfromVF(points, triangles_in, clean=True, deep_points=True):
    """ Generates mesh from points and triangles """

    # Add face padding if necessary
    if triangles_in.shape[1] == 3:
        triangles = np.empty((triangles_in.shape[0], 4), dtype=ctypes.c_int64)
        triangles[:, -3:] = triangles_in
        triangles[:, 0] = 3
    else:
        triangles = triangles_in

    mesh = vtkInterface.PolyData(points, triangles.ravel())

    if clean:
        mesh.Clean()
    return mesh


def CreateVectorPolyData(orig, vec):
    """ Creates a vtkPolyData object composed of vectors """

    # shape, dimention checking
    if not isinstance(orig, np.ndarray):
        orig = np.asarray(orig)

    if not isinstance(vec, np.ndarray):
        vec = np.asarray(vec)

    if orig.ndim != 2:
        orig = orig.reshape((-1, 3))
    elif orig.shape[1] != 3:
        raise Exception('orig array must be 3D')

    if vec.ndim != 2:
        vec = vec.reshape((-1, 3))
    elif vec.shape[1] != 3:
        raise Exception('vec array must be 3D')

    # Create vtk points and cells objects
    vpts = vtk.vtkPoints()
    vpts.SetData(numpy_to_vtk(np.ascontiguousarray(orig), deep=True))

    npts = orig.shape[0]
    cells = np.hstack((np.ones((npts, 1), 'int'),
                       np.arange(npts).reshape((-1, 1))))

    if cells.dtype != ctypes.c_int64 or cells.flags.c_contiguous:
        cells = np.ascontiguousarray(cells, ctypes.c_int64)
    vcells = vtk.vtkCellArray()
    vcells.SetCells(npts, numpy_to_vtkIdTypeArray(cells, deep=True))

    # Create vtkPolyData object
    pdata = vtk.vtkPolyData()
    pdata.SetPoints(vpts)
    pdata.SetVerts(vcells)

    # Add vectors to polydata
    name = 'vectors'
    vtkfloat = numpy_to_vtk(np.ascontiguousarray(vec), deep=True)
    vtkfloat.SetName(name)
    pdata.GetPointData().AddArray(vtkfloat)
    pdata.GetPointData().SetActiveVectors(name)

    # Add magnitude of vectors to polydata
    name = 'mag'
    scalars = (vec * vec).sum(1)**0.5
    vtkfloat = numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
    vtkfloat.SetName(name)
    pdata.GetPointData().AddArray(vtkfloat)
    pdata.GetPointData().SetActiveScalars(name)

    return vtkInterface.PolyData(pdata)


def ReadG3D(filename):
    """ Reads data from a *.g3d file and outputs points and triangle arrays
    """

    # open file and skip header and part of triangle header
    with open(filename) as f:
        f.seek(96 + 144)

        # Number of points, poitner to first point, and size of point
        pt_dat = np.fromstring(f.read(4 * 3), dtype=np.uint32)

        # Number of triangles, pointer to first triangle, and size of triangle
        tri_dat = np.fromstring(f.read(4 * 3), dtype=np.uint32)

        # Read in points
        f.seek(pt_dat[1])  # Seek to start of point data
        points = np.zeros((pt_dat[0], 3))
        for i in range(pt_dat[0]):
            points[i] = np.fromstring(f.read(24), dtype=np.float64)
            f.seek(f.tell() + 4)

        # Read in triangles
        tri = np.fromstring(f.read(12 * tri_dat[0]), dtype=np.uint32)
        triangles = np.zeros((tri_dat[0], 4), dtype=ctypes.c_int64)
        triangles[:, 0] = 3
        triangles[:, 1:] = tri.reshape((-1, 3))

        # Close file
        f.close()

    return points, triangles


def TransFromMatrix(matrix, rigid=True):
    """ Convert a vtk matrix to a np.array """
    if rigid:
        n = 3
    else:
        n = 4

    t = np.empty((n, 4))
    for i in range(n):
        for j in range(4):
            t[i, j] = matrix.GetElement(i, j)

    return t


def MakeVTKPointsMesh(points):
    """ Create a PolyData object from a numpy array containing just points """
    if points.ndim != 2:
        points = points.reshape((-1, 3))

    npoints = points.shape[0]

    # Make VTK cells array
    cells = np.hstack((np.ones((npoints, 1)),
                       np.arange(npoints).reshape(-1, 1)))
    cells = np.ascontiguousarray(cells, dtype=ctypes.c_int64)
    vtkcells = vtk.vtkCellArray()
    vtkcells.SetCells(npoints, numpy_to_vtkIdTypeArray(cells, deep=True))

    # Convert points to vtk object
    vtkPoints = vtkInterface.MakevtkPoints(points)

    # Create polydata
    pdata = vtkInterface.PolyData()
    pdata.SetPoints(vtkPoints)
    pdata.SetVerts(vtkcells)
    return pdata


def TriUnitNormal(p1, p2, p3):
    ''' Retrieve the unit normal of a triangle polygon.

    Parameters
    ----------
    p1, p2, p3 : float, float, float
        Coordinates for triangle vertices

    Returns
    -------
    normal : float
        Unit normal for triangle
    '''
    x = np.linalg.det([[1, p1[1], p1[2]],
                       [1, p2[1], p2[2]],
                       [1, p3[1], p3[2]]])
    y = np.linalg.det([[p1[0], 1, p1[2]],
                       [p2[0], 1, p2[2]],
                       [p3[0], 1, p3[2]]])
    z = np.linalg.det([[p1[0], p1[1], 1],
                       [p2[0], p2[1], 1],
                       [p3[0], p3[1], 1]])

    magnitude = (x**2 + y**2 + z**2)**.5
    normal = np.array([x, y, z]) / magnitude

    return normal


def TriangleArea(p1, p2, p3):
    ''' Calculate triangle area using the Shoelace formula.

    Parameters
    ----------
    p1, p2, p3 : float, float, float
        Coordinates for triangle vertices

    Returns
    ----------
    result : float
        Area of triangle
    '''

    poly = np.array([p1, p2, p3])
    total = np.zeros((3,))
    for ii in range(len(poly)):
        vi1 = poly[ii]
        vi2 = poly[(ii + 1) % len(poly)]
        prod = np.cross(vi1, vi2)
        total += prod
    result = np.abs(np.dot(total, TriUnitNormal(*poly))) / 2

    return result
