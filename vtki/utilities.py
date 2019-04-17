"""
Supporting functions for polydata and grid objects

"""
import collections
import ctypes
import logging
import os

import imageio
import numpy as np
import vtk
from vtk.util.numpy_support import (numpy_to_vtk, numpy_to_vtkIdTypeArray,
                                    vtk_to_numpy)

import vtki
from vtki.readers import standard_reader_routine, get_ext, get_reader

POINT_DATA_FIELD = 0
CELL_DATA_FIELD = 1


def vtk_bit_array_to_char(vtkarr_bint):
    """ Cast vtk bit array to a char array """
    vtkarr = vtk.vtkCharArray()
    vtkarr.DeepCopy(vtkarr_bint)
    return vtkarr


def is_vtki_obj(obj):
    """ Return True if the Object is a ``vtki`` wrapped dataset """
    return isinstance(obj, (vtki.Common, vtki.MultiBlock))


def point_scalar(mesh, name):
    """ Returns point scalars of a vtk object """
    vtkarr = mesh.GetPointData().GetArray(name)
    if vtkarr:
        if isinstance(vtkarr, vtk.vtkBitArray):
            vtkarr = vtk_bit_array_to_char(vtkarr)
        return vtk_to_numpy(vtkarr)


def cell_scalar(mesh, name):
    """ Returns cell scalars of a vtk object """
    vtkarr = mesh.GetCellData().GetArray(name)
    if vtkarr:
        if isinstance(vtkarr, vtk.vtkBitArray):
            vtkarr = vtk_bit_array_to_char(vtkarr)
        return vtk_to_numpy(vtkarr)


def get_scalar(mesh, name, preference='cell', info=False, err=False):
    """ Searches both point and cell data for an array

    Parameters
    ----------
    name : str
        The name of the array to get the range.

    preference : str, optional
        When scalars is specified, this is the perfered scalar type to
        search for in the dataset.  Must be either ``'point'`` or ``'cell'``

    info : bool
        Return info about the scalar rather than the array itself.

    err : bool
        Boolean to control whether to throw an error if array is not present.

    """
    parr = point_scalar(mesh, name)
    carr = cell_scalar(mesh, name)
    if isinstance(preference, str):
        if preference in ['cell', 'c', 'cells']:
            preference = CELL_DATA_FIELD
        elif preference in ['point', 'p', 'points']:
            preference = POINT_DATA_FIELD
        else:
            raise RuntimeError('Data field ({}) not supported.'.format(preference))
    if all([parr is not None, carr is not None]):
        if preference == CELL_DATA_FIELD:
            if info:
                return carr, CELL_DATA_FIELD
            else:
                return carr
        elif preference == POINT_DATA_FIELD:
            if info:
                return parr, POINT_DATA_FIELD
            else:
                return parr
        else:
            raise RuntimeError('Data field ({}) not supported.'.format(preference))
    arr = None
    field = None
    if parr is not None:
        arr = parr
        field = 0
    elif carr is not None:
        arr = carr
        field = 1
    elif err:
        raise KeyError('Data scalar ({}) not present in this dataset.'.format(name))
    if info:
        return arr, field
    return arr


def vtk_points(points, deep=True):
    """ Convert numpy points to a vtkPoints object """
    if not points.flags['C_CONTIGUOUS']:
        points = np.ascontiguousarray(points)
    vtkpts = vtk.vtkPoints()
    vtkpts.SetData(numpy_to_vtk(points, deep=deep))
    return vtkpts


def lines_from_points(points):
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

    >>> import vtki
    >>> import numpy as np
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])
    >>> lines = vtki.lines_from_points(points)
    >>> lines.plot() # doctest:+SKIP

    """
    # Assuming ordered points, create array defining line order
    npoints = points.shape[0] - 1
    lines = np.vstack((2 * np.ones(npoints, np.int),
                       np.arange(npoints),
                       np.arange(1, npoints + 1))).T.ravel()

    return vtki.PolyData(points, lines)


def vector_poly_data(orig, vec):
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
    cells = np.reshape(cells, (2*npts))
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

    return vtki.PolyData(pdata)


def trans_from_matrix(matrix):
    """ Convert a vtk matrix to a numpy.ndarray """
    t = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            t[i, j] = matrix.GetElement(i, j)
    return t


def wrap(vtkdataset):
    """This is a convenience method to safely wrap any given VTK data object
    to its appropriate ``vtki`` data object.
    """
    wrappers = {
        'vtkUnstructuredGrid' : vtki.UnstructuredGrid,
        'vtkRectilinearGrid' : vtki.RectilinearGrid,
        'vtkStructuredGrid' : vtki.StructuredGrid,
        'vtkPolyData' : vtki.PolyData,
        'vtkImageData' : vtki.UniformGrid,
        'vtkStructuredPoints' : vtki.UniformGrid,
        'vtkMultiBlockDataSet' : vtki.MultiBlock,
        }
    key = vtkdataset.GetClassName()
    try:
        wrapped = wrappers[key](vtkdataset)
    except:
        logging.warning('VTK data type ({}) is not currently supported by vtki.'.format(key))
        return vtkdataset # if not supported just passes the VTK data object
    return wrapped


def image_to_texture(image):
    """Converts ``vtkImageData`` to a ``vtkTexture``"""
    vtex = vtk.vtkTexture()
    vtex.SetInputDataObject(image)
    vtex.Update()
    return vtex


def numpy_to_texture(image):
    """Convert a NumPy image array to a vtk.vtkTexture"""
    if not isinstance(image, np.ndarray):
        raise TypeError('Unknown input type ({})'.format(type(image)))
    if image.ndim != 3 or image.shape[2] != 3:
        raise AssertionError('Input image must be nn by nm by RGB')
    grid = vtki.UniformGrid((image.shape[1], image.shape[0], 1))
    grid.point_arrays['Image'] = np.flip(image.swapaxes(0,1), axis=1).reshape((-1, 3), order='F')
    grid.set_active_scalar('Image')
    return image_to_texture(grid)


def is_inside_bounds(point, bounds):
    """ Checks if a point is inside a set of bounds. This is implemented
    through recursion so that this is N-dimensional.
    """
    if isinstance(point, (int, float)):
        point = [point]
    if isinstance(point, collections.Iterable) and not isinstance(point, collections.deque):
        if len(bounds) < 2 * len(point) or len(bounds) % 2 != 0:
            raise AssertionError('Bounds mismatch point dimensionality')
        point = collections.deque(point)
        bounds = collections.deque(bounds)
        return is_inside_bounds(point, bounds)
    if not isinstance(point, collections.deque):
        raise TypeError('Unknown input data type ({}).'.format(type(point)))
    if len(point) < 1:
        return True
    p = point.popleft()
    lower, upper = bounds.popleft(), bounds.popleft()
    if lower <= p <= upper:
        return is_inside_bounds(point, bounds)
    return False


def fit_plane_to_points(points, return_meta=False):
    """
    Fits a plane to a set of points

    Parameters
    ----------
    points : np.ndarray
        Size n by 3 array of points to fit a plane through

    return_meta : bool
        If true, also returns the center and normal used to generate the plane
    """
    data = np.array(points)
    center = data.mean(axis=0)
    result = np.linalg.svd(data - center)
    normal = np.cross(result[2][0], result[2][1])
    plane = vtki.Plane(center=center, direction=normal)
    if return_meta:
        return plane, center, normal
    return plane


def _raise_not_matching(scalars, mesh):
    raise Exception('Number of scalars (%d) ' % scalars.size +
                    'must match either the number of points ' +
                    '(%d) ' % mesh.n_points +
                    'or the number of cells ' +
                    '(%d) ' % mesh.n_cells)
