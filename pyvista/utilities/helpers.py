"""Supporting functions for polydata and grid objects."""

import collections.abc
import enum
import logging
import os
import signal
import sys
import threading
from threading import Thread
import traceback
from typing import Optional
import warnings

import numpy as np

import pyvista
from pyvista import _vtk

from . import transformations
from .fileio import from_meshio


class FieldAssociation(enum.Enum):
    """Represents which type of vtk field a scalar or vector array is associated with."""

    POINT = _vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
    CELL = _vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS
    NONE = _vtk.vtkDataObject.FIELD_ASSOCIATION_NONE
    ROW = _vtk.vtkDataObject.FIELD_ASSOCIATION_ROWS


def get_vtk_type(typ):
    """Look up the VTK type for a given numpy data type.

    Corrects for string type mapping issues.

    Parameters
    ----------
    typ : numpy.dtype
        Numpy data type.

    Returns
    -------
    int
        Integer type id specified in ``vtkType.h``

    """
    typ = _vtk.get_vtk_array_type(typ)
    # This handles a silly string type bug
    if typ == 3:
        return 13
    return typ


def vtk_bit_array_to_char(vtkarr_bint):
    """Cast vtk bit array to a char array.

    Parameters
    ----------
    vtkarr_bint : vtk.vtkBitArray
        VTK binary array.

    Returns
    -------
    vtk.vtkCharArray
        VTK char array.

    Notes
    -----
    This performs a copy.

    """
    vtkarr = _vtk.vtkCharArray()
    vtkarr.DeepCopy(vtkarr_bint)
    return vtkarr


def vtk_id_list_to_array(vtk_id_list):
    """Convert a vtkIdList to a NumPy array.

    Parameters
    ----------
    vtk_id_list : vtk.vtkIdList
        VTK ID list.

    Returns
    -------
    numpy.ndarray
        Array of IDs.

    """
    return np.array([vtk_id_list.GetId(i) for i in range(vtk_id_list.GetNumberOfIds())])


def convert_string_array(arr, name=None):
    """Convert a numpy array of strings to a vtkStringArray or vice versa.

    Parameters
    ----------
    arr : numpy.ndarray
        Numpy string array to convert.

    name : str, optional
        Name to set the vtkStringArray to.

    Returns
    -------
    vtkStringArray
        VTK string array.

    Notes
    -----
    Note that this is terribly inefficient. If you have ideas on how
    to make this faster, please consider opening a pull request.

    """
    if isinstance(arr, np.ndarray):
        vtkarr = _vtk.vtkStringArray()
        ########### OPTIMIZE ###########
        for val in arr:
            vtkarr.InsertNextValue(val)
        ################################
        if isinstance(name, str):
            vtkarr.SetName(name)
        return vtkarr
    # Otherwise it is a vtk array and needs to be converted back to numpy
    ############### OPTIMIZE ###############
    nvalues = arr.GetNumberOfValues()
    return np.array([arr.GetValue(i) for i in range(nvalues)], dtype='|U')
    ########################################


def convert_array(arr, name=None, deep=False, array_type=None):
    """Convert a NumPy array to a vtkDataArray or vice versa.

    Parameters
    ----------
    arr : np.ndarray or vtkDataArray
        A numpy array or vtkDataArry to convert.
    name : str, optional
        The name of the data array for VTK.
    deep : bool, optional
        If input is numpy array then deep copy values.
    array_type : int, optional
        VTK array type ID as specified in specified in ``vtkType.h``.

    Returns
    -------
    vtkDataArray, numpy.ndarray, or DataFrame
        The converted array.  If input is a :class:`numpy.ndarray` then
        returns ``vtkDataArray`` or is input is ``vtkDataArray`` then
        returns NumPy ``ndarray``.

    """
    if arr is None:
        return
    if isinstance(arr, np.ndarray):
        if arr.dtype is np.dtype('O'):
            arr = arr.astype('|S')
        arr = np.ascontiguousarray(arr)
        if arr.dtype.type in (np.str_, np.bytes_):
            # This handles strings
            vtk_data = convert_string_array(arr)
        else:
            # This will handle numerical data
            arr = np.ascontiguousarray(arr)
            vtk_data = _vtk.numpy_to_vtk(num_array=arr, deep=deep,
                                         array_type=array_type)
        if isinstance(name, str):
            vtk_data.SetName(name)
        return vtk_data
    # Otherwise input must be a vtkDataArray
    if not isinstance(arr, (_vtk.vtkDataArray, _vtk.vtkBitArray, _vtk.vtkStringArray)):
        raise TypeError(f'Invalid input array type ({type(arr)}).')
    # Handle booleans
    if isinstance(arr, _vtk.vtkBitArray):
        arr = vtk_bit_array_to_char(arr)
    # Handle string arrays
    if isinstance(arr, _vtk.vtkStringArray):
        return convert_string_array(arr)
    # Convert from vtkDataArry to NumPy
    return _vtk.vtk_to_numpy(arr)


def is_pyvista_dataset(obj):
    """Return ``True`` if the object is a PyVista wrapped dataset.

    Parameters
    ----------
    obj : anything
        Any object to test.

    Returns
    -------
    bool
        ``True`` when the object is a :class:`pyvista.DataSet`.

    """
    return isinstance(obj, (pyvista.DataSet, pyvista.MultiBlock))


def point_array(obj, name):
    """Return point array of a pyvista or vtk object.

    Parameters
    ----------
    obj : pyvista.DataSet or vtk.vtkDataSet
        PyVista or VTK dataset.

    name : str
        Name of the array.

    Returns
    -------
    numpy.ndarray
        Wrapped array.

    """
    vtkarr = obj.GetPointData().GetAbstractArray(name)
    return convert_array(vtkarr)


def field_array(obj, name):
    """Return field data of a pyvista or vtk object.

    Parameters
    ----------
    obj : pyvista.DataSet or vtk.vtkDataSet
        PyVista or VTK dataset.

    name : str
        Name of the array.

    Returns
    -------
    numpy.ndarray
        Wrapped array.

    """
    vtkarr = obj.GetFieldData().GetAbstractArray(name)
    return convert_array(vtkarr)


def cell_array(obj, name):
    """Return cell array of a pyvista or vtk object.

    Parameters
    ----------
    obj : pyvista.DataSet or vtk.vtkDataSet
        PyVista or VTK dataset.

    name : str
        Name of the array.

    Returns
    -------
    numpy.ndarray
        Wrapped array.

    """
    vtkarr = obj.GetCellData().GetAbstractArray(name)
    return convert_array(vtkarr)


def row_array(obj, name):
    """Return row array of a vtk object.

    Parameters
    ----------
    obj : vtk.vtkDataSet
        PyVista or VTK dataset.

    name : str
        Name of the array.

    Returns
    -------
    numpy.ndarray
        Wrapped array.

    """
    vtkarr = obj.GetRowData().GetAbstractArray(name)
    return convert_array(vtkarr)


def parse_field_choice(field):
    """Return a field association object for a given field type string.

    Parameters
    ----------
    field : str, FieldAssociation
        Name of the field (e.g, ``'cell'``, ``'field'``, ``'point'``,
        ``'row'``).

    Returns
    -------
    pyvista.FieldAssociation
        Field association.

    """
    if isinstance(field, str):
        field = field.strip().lower()
        if field in ['cell', 'c', 'cells']:
            field = FieldAssociation.CELL
        elif field in ['point', 'p', 'points']:
            field = FieldAssociation.POINT
        elif field in ['field', 'f', 'fields']:
            field = FieldAssociation.NONE
        elif field in ['row', 'r']:
            field = FieldAssociation.ROW
        else:
            raise ValueError(f'Data field ({field}) not supported.')
    elif isinstance(field, FieldAssociation):
        pass
    else:
        raise ValueError(f'Data field ({field}) not supported.')
    return field


def get_array(mesh, name, preference='cell', err=False) -> Optional[np.ndarray]:
    """Search point, cell and field data for an array.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Dataset to get the array from.

    name : str
        The name of the array to get the range.

    preference : str, optional
        When scalars is specified, this is the preferred array type to
        search for in the dataset.  Must be either ``'point'``,
        ``'cell'``, or ``'field'``.

    err : bool, optional
        Whether to throw an error if array is not present.

    Returns
    -------
    pyvista.pyvista_ndarray or ``None``
        Requested array.  Return ``None`` if there is no array
        matching the ``name`` and ``err=False``.

    """
    if isinstance(mesh, _vtk.vtkTable):
        arr = row_array(mesh, name)
        if arr is None and err:
            raise KeyError(f'Data array ({name}) not present in this dataset.')
        return arr

    parr = point_array(mesh, name)
    carr = cell_array(mesh, name)
    farr = field_array(mesh, name)
    preference = parse_field_choice(preference)
    if sum([array is not None for array in (parr, carr, farr)]) > 1:
        if preference == FieldAssociation.CELL:
            return carr
        elif preference == FieldAssociation.POINT:
            return parr
        elif preference == FieldAssociation.NONE:
            return farr
        else:
            raise ValueError(f'Data field ({preference}) not supported.')

    if parr is not None:
        return parr
    elif carr is not None:
        return carr
    elif farr is not None:
        return farr
    elif err:
        raise KeyError(f'Data array ({name}) not present in this dataset.')
    return None


def get_array_association(mesh, name, preference='cell', err=False) -> FieldAssociation:
    """Return the array association.

    Parameters
    ----------
    mesh : Dataset
        Dataset to get the array association from.

    name : str
        The name of the array.

    preference : str, optional
        When scalars is specified, this is the preferred array type to
        search for in the dataset.  Must be either ``'point'``,
        ``'cell'``, or ``'field'``.

    err : bool, optional
        Boolean to control whether to throw an error if array is not
        present.

    Returns
    -------
    pyvista.FieldAssociation
        Association of the array. If array is not present and ``err`` is
        ``False``, ``FieldAssociation.NONE`` is returned.

    """
    if isinstance(mesh, _vtk.vtkTable):
        arr = row_array(mesh, name)
        if arr is None and err:
            raise KeyError(f'Data array ({name}) not present in this dataset.')
        return FieldAssociation.ROW

    # with multiple arrays, return the array preference if possible
    parr = point_array(mesh, name)
    carr = cell_array(mesh, name)
    farr = field_array(mesh, name)
    arrays = [parr, carr, farr]
    preferences = [FieldAssociation.POINT, FieldAssociation.CELL, FieldAssociation.NONE]
    preference = parse_field_choice(preference)
    if preference not in preferences:
        raise ValueError(f'Data field ({preference}) not supported.')

    matches = [pref for pref, array in zip(preferences, arrays) if array is not None]
    # optionally raise if no match
    if not matches:
        if err:
            raise KeyError(f'Data array ({name}) not present in this dataset.')
        return FieldAssociation.NONE
    # use preference if it applies
    if preference in matches:
        return preference
    # otherwise return first in order of point -> cell -> field
    return matches[0]


def vtk_points(points, deep=True, force_float=False):
    """Convert numpy array or array-like to a ``vtkPoints`` object.

    Parameters
    ----------
    points : numpy.ndarray or sequence
        Points to convert.  Should be 1 or 2 dimensional.  Accepts a
        single point or several points.

    deep : bool, optional
        Perform a deep copy of the array.  Only applicable if
        ``points`` is a :class:`numpy.ndarray`.

    force_float : bool, optional
        Casts the datatype to ``float32`` if points datatype is
        non-float.  Set this to ``False`` to allow non-float types,
        though this may lead to truncation of intermediate floats
        when transforming datasets.

    Returns
    -------
    vtk.vtkPoints
        The vtkPoints object.

    Examples
    --------
    >>> import pyvista
    >>> import numpy as np
    >>> points = np.random.random((10, 3))
    >>> vpoints = pyvista.vtk_points(points)
    >>> vpoints  # doctest:+SKIP
    (vtkmodules.vtkCommonCore.vtkPoints)0x7f0c2e26af40

    """
    points = np.asanyarray(points)

    # verify is numeric
    if not np.issubdtype(points.dtype, np.number):
        raise TypeError('Points must be a numeric type')

    if force_float:
        if not np.issubdtype(points.dtype, np.floating):
            warnings.warn(
                'Points is not a float type. This can cause issues when '
                'transforming or applying filters. Casting to '
                '``np.float32``. Disable this by passing '
                '``force_float=False``.'
            )
            points = points.astype(np.float32)

    # check dimensionality
    if points.ndim == 1:
        points = points.reshape(-1, 3)
    elif points.ndim > 2:
        raise ValueError('Dimension of ``points`` should be 1 or 2, not '
                         f'{points.ndim}')

    # verify shape
    if points.shape[1] != 3:
        raise ValueError('Points array must contain three values per point. '
                         f'Shape is {points.shape} and should be (X, 3)')

    # points must be contiguous
    points = np.require(points, requirements=['C'])
    vtkpts = _vtk.vtkPoints()
    vtk_arr = _vtk.numpy_to_vtk(points, deep=deep)
    vtkpts.SetData(vtk_arr)
    return vtkpts


def line_segments_from_points(points):
    """Generate non-connected line segments from points.

    Assumes points are ordered as line segments and an even number of
    points.

    Parameters
    ----------
    points : numpy.ndarray
        Points representing line segments. An even number must be
        given as every two vertices represent a single line
        segment. For example, two line segments would be represented
        as ``np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])``.

    Returns
    -------
    pyvista.PolyData
        PolyData with lines and cells.

    Examples
    --------
    This example plots two line segments at right angles to each other.

    >>> import pyvista
    >>> import numpy as np
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])
    >>> lines = pyvista.lines_from_points(points)
    >>> lines.plot()

    """
    if len(points) % 2 != 0:
        raise ValueError("An even number of points must be given to define each segment.")
    # Assuming ordered points, create array defining line order
    n_points = len(points)
    n_lines = n_points // 2
    lines = np.c_[(2 * np.ones(n_lines, np.int_),
                   np.arange(0, n_points-1, step=2),
                   np.arange(1, n_points+1, step=2))]
    poly = pyvista.PolyData()
    poly.points = points
    poly.lines = lines
    return poly


def lines_from_points(points, close=False):
    """Make a connected line set given an array of points.

    Parameters
    ----------
    points : np.ndarray
        Points representing the vertices of the connected
        segments. For example, two line segments would be represented
        as ``np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])``.

    close : bool, optional
        If ``True``, close the line segments into a loop.

    Returns
    -------
    pyvista.PolyData
        PolyData with lines and cells.

    Examples
    --------
    >>> import numpy as np
    >>> import pyvista
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    >>> poly = pyvista.lines_from_points(points)
    >>> poly.plot(line_width=5)

    """
    poly = pyvista.PolyData()
    poly.points = points
    cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    if close:
        cells = np.append(cells, [[2, len(points)-1, 0]], axis=0)
    poly.lines = cells
    return poly


def make_tri_mesh(points, faces):
    """Construct a ``pyvista.PolyData`` mesh using points and faces arrays.

    Construct a mesh from an Nx3 array of points and an Mx3 array of
    triangle indices, resulting in a mesh with N vertices and M
    triangles.  This function does not require the standard VTK
    "padding" column and simplifies mesh creation.

    Parameters
    ----------
    points : np.ndarray
        Array of points with shape ``(N, 3)`` storing the vertices of the
        triangle mesh.

    faces : np.ndarray
        Array of indices with shape ``(M, 3)`` containing the triangle
        indices.

    Returns
    -------
    pyvista.PolyData
        PolyData instance containing the triangle mesh.

    Examples
    --------
    This example discretizes the unit square into a triangle mesh with
    nine vertices and eight faces.

    >>> import numpy as np
    >>> import pyvista
    >>> points = np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [0, 0.5, 0],
    ...                    [0.5, 0.5, 0], [1, 0.5, 0], [0, 1, 0], [0.5, 1, 0],
    ...                    [1, 1, 0]])
    >>> faces = np.array([[0, 1, 4], [4, 7, 6], [2, 5, 4], [4, 5, 8],
    ...                   [0, 4, 3], [3, 4, 6], [1, 2, 4], [4, 8, 7]])
    >>> tri_mesh = pyvista.make_tri_mesh(points, faces)
    >>> tri_mesh.plot(show_edges=True, line_width=5)

    """
    if points.shape[1] != 3:
        raise ValueError("Points array should have shape (N, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Face array should have shape (M, 3).")
    cells = np.empty((faces.shape[0], 4), dtype=faces.dtype)
    cells[:, 0] = 3
    cells[:, 1:] = faces
    return pyvista.PolyData(points, cells)


def vector_poly_data(orig, vec):
    """Create a pyvista.PolyData object composed of vectors.

    Parameters
    ----------
    orig : numpy.ndarray
        Array of vector origins.

    vec : numpy.ndarray
        Array of vectors.

    Returns
    -------
    pyvista.PolyData
        Mesh containing the ``orig`` points along with the
        ``'vectors'`` and ``'mag'`` point arrays representing the
        vectors and magnitude of the the vectors at each point.

    Examples
    --------
    Create basic vector field.  This is a point cloud where each point
    has a vector and magnitude attached to it.

    >>> import pyvista
    >>> import numpy as np
    >>> x, y = np.meshgrid(np.linspace(-5,5,10),np.linspace(-5,5,10))
    >>> points = np.vstack((x.ravel(), y.ravel(), np.zeros(x.size))).T
    >>> u = x/np.sqrt(x**2 + y**2)
    >>> v = y/np.sqrt(x**2 + y**2)
    >>> vectors = np.vstack((u.ravel()**3, v.ravel()**3, np.zeros(u.size))).T
    >>> pdata = pyvista.vector_poly_data(points, vectors)
    >>> pdata.point_data.keys()
    ['vectors', 'mag']

    Convert these to arrows and plot it.

    >>> pdata.glyph(orient='vectors', scale='mag').plot()

    """
    # shape, dimension checking
    if not isinstance(orig, np.ndarray):
        orig = np.asarray(orig)

    if not isinstance(vec, np.ndarray):
        vec = np.asarray(vec)

    if orig.ndim != 2:
        orig = orig.reshape((-1, 3))
    elif orig.shape[1] != 3:
        raise ValueError('orig array must be 3D')

    if vec.ndim != 2:
        vec = vec.reshape((-1, 3))
    elif vec.shape[1] != 3:
        raise ValueError('vec array must be 3D')

    # Create vtk points and cells objects
    vpts = _vtk.vtkPoints()
    vpts.SetData(_vtk.numpy_to_vtk(np.ascontiguousarray(orig), deep=True))

    npts = orig.shape[0]
    cells = np.empty((npts, 2), dtype=pyvista.ID_TYPE)
    cells[:, 0] = 1
    cells[:, 1] = np.arange(npts, dtype=pyvista.ID_TYPE)
    vcells = pyvista.utilities.cells.CellArray(cells, npts)

    # Create vtkPolyData object
    pdata = _vtk.vtkPolyData()
    pdata.SetPoints(vpts)
    pdata.SetVerts(vcells)

    # Add vectors to polydata
    name = 'vectors'
    vtkfloat = _vtk.numpy_to_vtk(np.ascontiguousarray(vec), deep=True)
    vtkfloat.SetName(name)
    pdata.GetPointData().AddArray(vtkfloat)
    pdata.GetPointData().SetActiveVectors(name)

    # Add magnitude of vectors to polydata
    name = 'mag'
    scalars = (vec * vec).sum(1)**0.5
    vtkfloat = _vtk.numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
    vtkfloat.SetName(name)
    pdata.GetPointData().AddArray(vtkfloat)
    pdata.GetPointData().SetActiveScalars(name)

    return pyvista.PolyData(pdata)


def trans_from_matrix(matrix):  # pragma: no cover
    """Convert a vtk matrix to a numpy.ndarray.

    DEPRECATED: Please use ``array_from_vtkmatrix``.

    """
    # import needs to happen here to prevent a circular import
    from pyvista.core.errors import DeprecationError
    raise DeprecationError('DEPRECATED: Please use ``array_from_vtkmatrix``.')


def array_from_vtkmatrix(matrix):
    """Convert a vtk matrix to an array.

    Parameters
    ----------
    matrix : vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4
        The vtk matrix to be converted to a ``numpy.ndarray``.
        Returned ndarray has shape (3, 3) or (4, 4) as appropriate.

    Returns
    -------
    numpy.ndarray
        Numpy array containing the data from ``matrix``.

    """
    if isinstance(matrix, _vtk.vtkMatrix3x3):
        shape = (3, 3)
    elif isinstance(matrix, _vtk.vtkMatrix4x4):
        shape = (4, 4)
    else:
        raise TypeError('Expected vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4 input,'
                        f' got {type(matrix).__name__} instead.')
    array = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            array[i, j] = matrix.GetElement(i, j)
    return array


def vtkmatrix_from_array(array):
    """Convert a ``numpy.ndarray`` or array-like to a vtk matrix.

    Parameters
    ----------
    array : numpy.ndarray or array-like
        The array or array-like to be converted to a vtk matrix.
        Shape (3, 3) gets converted to a ``vtk.vtkMatrix3x3``, shape (4, 4)
        gets converted to a ``vtk.vtkMatrix4x4``. No other shapes are valid.

    Returns
    -------
    vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4
        VTK matrix.

    """
    array = np.asarray(array)
    if array.shape == (3, 3):
        matrix = _vtk.vtkMatrix3x3()
    elif array.shape == (4, 4):
        matrix = _vtk.vtkMatrix4x4()
    else:
        raise ValueError(f'Invalid shape {array.shape}, must be (3, 3) or (4, 4).')
    m, n = array.shape
    for i in range(m):
        for j in range(n):
            matrix.SetElement(i, j, array[i, j])
    return matrix


def is_meshio_mesh(obj):
    """Test if passed object is instance of ``meshio.Mesh``.

    Parameters
    ----------
    obj
        Any object.

    Returns
    -------
    bool
        ``True`` if ``obj`` is an ``meshio.Mesh``.

    """
    try:
        import meshio
        return isinstance(obj, meshio.Mesh)
    except ImportError:
        return False


def wrap(dataset):
    """Wrap any given VTK data object to its appropriate PyVista data object.

    Other formats that are supported include:

    * 2D :class:`numpy.ndarray` of XYZ vertices
    * 3D :class:`numpy.ndarray` representing a volume. Values will be scalars.
    * 3D :class:`trimesh.Trimesh` mesh.
    * 3D :class:`meshio.Mesh` mesh.

    Parameters
    ----------
    dataset : :class:`numpy.ndarray`, :class:`trimesh.Trimesh`, or VTK object
        Dataset to wrap.

    Returns
    -------
    pyvista.DataSet
        The PyVista wrapped dataset.

    Examples
    --------
    Wrap a numpy array representing a random point cloud.

    >>> import numpy as np
    >>> import pyvista
    >>> points = np.random.random((10, 3))
    >>> cloud = pyvista.wrap(points)
    >>> cloud  # doctest:+SKIP
    PolyData (0x7fc52db83d70)
      N Cells:  10
      N Points: 10
      X Bounds: 1.123e-01, 7.457e-01
      Y Bounds: 1.009e-01, 9.877e-01
      Z Bounds: 2.346e-03, 9.640e-01
      N Arrays: 0

    Wrap a Trimesh object.

    >>> import trimesh
    >>> import pyvista
    >>> points = [[0, 0, 0], [0, 0, 1], [0, 1, 0]]
    >>> faces = [[0, 1, 2]]
    >>> tmesh = trimesh.Trimesh(points, faces=faces, process=False)
    >>> mesh = pyvista.wrap(tmesh)
    >>> mesh  # doctest:+SKIP
    PolyData (0x7fc55ff27ad0)
      N Cells:  1
      N Points: 3
      X Bounds: 0.000e+00, 0.000e+00
      Y Bounds: 0.000e+00, 1.000e+00
      Z Bounds: 0.000e+00, 1.000e+00
      N Arrays: 0

    Wrap a VTK object.

    >>> import pyvista
    >>> import vtk
    >>> points = vtk.vtkPoints()
    >>> p = [1.0, 2.0, 3.0]
    >>> vertices = vtk.vtkCellArray()
    >>> pid = points.InsertNextPoint(p)
    >>> _ = vertices.InsertNextCell(1)
    >>> _ = vertices.InsertCellPoint(pid)
    >>> point = vtk.vtkPolyData()
    >>> _ = point.SetPoints(points)
    >>> _ = point.SetVerts(vertices)
    >>> mesh = pyvista.wrap(point)
    >>> mesh  # doctest:+SKIP
    PolyData (0x7fc55ff27ad0)
      N Cells:  1
      N Points: 3
      X Bounds: 0.000e+00, 0.000e+00
      Y Bounds: 0.000e+00, 1.000e+00
      Z Bounds: 0.000e+00, 1.000e+00
      N Arrays: 0

    """
    # Return if None
    if dataset is None:
        return

    # Check if dataset is a numpy array.  We do this first since
    # pyvista_ndarray contains a VTK type that we don't want to
    # directly wrap.
    if isinstance(dataset, (np.ndarray, pyvista.pyvista_ndarray)):
        if dataset.ndim == 1 and dataset.shape[0] == 3:
            return pyvista.PolyData(dataset)
        if dataset.ndim > 1 and dataset.ndim < 3 and dataset.shape[1] == 3:
            return pyvista.PolyData(dataset)
        elif dataset.ndim == 3:
            mesh = pyvista.UniformGrid(dims=dataset.shape)
            mesh['values'] = dataset.ravel(order='F')
            mesh.active_scalars_name = 'values'
            return mesh
        else:
            raise NotImplementedError('NumPy array could not be wrapped pyvista.')

    # wrap VTK arrays as pyvista_ndarray
    if isinstance(dataset, _vtk.vtkDataArray):
        return pyvista.pyvista_ndarray(dataset)

    # Check if a dataset is a VTK type
    if hasattr(dataset, 'GetClassName'):
        key = dataset.GetClassName()
        try:
            return pyvista._wrappers[key](dataset)
        except KeyError:
            logging.warning(f'VTK data type ({key}) is not currently supported by pyvista.')
        return

    # wrap meshio
    if is_meshio_mesh(dataset):
        return from_meshio(dataset)

    # wrap trimesh
    if dataset.__class__.__name__ == 'Trimesh':
        # trimesh doesn't pad faces
        n_face = dataset.faces.shape[0]
        faces = np.empty((n_face, 4), dataset.faces.dtype)
        faces[:, 1:] = dataset.faces
        faces[:, 0] = 3
        return pyvista.PolyData(np.asarray(dataset.vertices), faces)

    # otherwise, flag tell the user we can't wrap this object
    raise NotImplementedError(f'Unable to wrap ({type(dataset)}) into a pyvista type.')


def image_to_texture(image):
    """Convert ``vtkImageData`` (:class:`pyvista.UniformGrid`) to a ``vtkTexture``.

    Parameters
    ----------
    image : pyvista.UniformGrid or vtkImageData
        Image to convert.

    Returns
    -------
    vtkTexture
        VTK texture.

    """
    return pyvista.Texture(image)


def numpy_to_texture(image):
    """Convert a NumPy image array to a vtk.vtkTexture.

    Parameters
    ----------
    image : numpy.ndarray
        Numpy image array.

    Returns
    -------
    vtkTexture
        VTK texture.

    """
    return pyvista.Texture(image)


def is_inside_bounds(point, bounds):
    """Check if a point is inside a set of bounds.

    This is implemented through recursion so that this is N-dimensional.

    Parameters
    ----------
    point : sequence
        Three item cartesian point (i.e. ``[x, y, z]``).

    bounds : sequence
        Six item bounds in the form of ``(xMin, xMax, yMin, yMax, zMin, zMax)``.

    Returns
    -------
    bool
        ``True`` when ``point`` is inside ``bounds``.

    """
    if isinstance(point, (int, float)):
        point = [point]
    if isinstance(point, (np.ndarray, collections.abc.Sequence)) and not isinstance(point, collections.deque):
        if len(bounds) < 2 * len(point) or len(bounds) % 2 != 0:
            raise ValueError('Bounds mismatch point dimensionality')
        point = collections.deque(point)
        bounds = collections.deque(bounds)
        return is_inside_bounds(point, bounds)
    if not isinstance(point, collections.deque):
        raise TypeError(f'Unknown input data type ({type(point)}).')
    if len(point) < 1:
        return True
    p = point.popleft()
    lower, upper = bounds.popleft(), bounds.popleft()
    if lower <= p <= upper:
        return is_inside_bounds(point, bounds)
    return False


def fit_plane_to_points(points, return_meta=False):
    """Fit a plane to a set of points using the SVD algorithm.

    Parameters
    ----------
    points : sequence
        Size ``[N x 3]`` sequence of points to fit a plane through.

    return_meta : bool, optional
        If ``True``, also returns the center and normal used to
        generate the plane.

    Returns
    -------
    pyvista.PolyData
        Plane mesh.

    numpy.ndarray
        Plane center if ``return_meta=True``.

    numpy.ndarray
        Plane normal if ``return_meta=True``.

    Examples
    --------
    Fit a plane to a random point cloud.

    >>> import pyvista
    >>> import numpy as np
    >>> cloud = np.random.random((10, 3))
    >>> cloud[:, 2] *= 0.1
    >>> plane, center, normal = pyvista.fit_plane_to_points(cloud, return_meta=True)

    Plot the fitted plane.

    >>> pl = pyvista.Plotter()
    >>> _ = pl.add_mesh(plane, color='tan', style='wireframe', line_width=4)
    >>> _ = pl.add_points(cloud, render_points_as_spheres=True,
    ...                   color='r', point_size=30)
    >>> pl.show()

    """
    data = np.array(points)
    center = data.mean(axis=0)
    result = np.linalg.svd(data - center)
    normal = np.cross(result[2][0], result[2][1])
    plane = pyvista.Plane(center=center, direction=normal)
    if return_meta:
        return plane, center, normal
    return plane


def raise_not_matching(scalars, dataset):
    """Raise exception about inconsistencies.

    Parameters
    ----------
    scalars : numpy.ndarray
        Array of scalars.

    dataset : pyvista.DataSet
        Dataset to check against.

    Raises
    ------
    ValueError
        Raises a ValueError if the size of scalars does not the dataset.
    """
    if isinstance(dataset, _vtk.vtkTable):
        raise ValueError(f'Number of scalars ({scalars.size}) must match number of rows ({dataset.n_rows}).')
    raise ValueError(f'Number of scalars ({scalars.size}) '
                     f'must match either the number of points ({dataset.n_points}) '
                     f'or the number of cells ({dataset.n_cells}).')


def generate_plane(normal, origin):
    """Return a _vtk.vtkPlane.

    Parameters
    ----------
    normal : sequence
        Three item sequence representing the normal of the plane.

    origin : sequence
        Three item sequence representing the origin of the plane.

    Returns
    -------
    vtk.vtkPlane
        VTK plane.

    """
    plane = _vtk.vtkPlane()
    # NORMAL MUST HAVE MAGNITUDE OF 1
    normal = normal / np.linalg.norm(normal)
    plane.SetNormal(normal)
    plane.SetOrigin(origin)
    return plane


def try_callback(func, *args):
    """Wrap a given callback in a try statement.

    Parameters
    ----------
    func : callable
        Callable object.

    *args
        Any arguments.

    """
    try:
        func(*args)
    except Exception:
        etype, exc, tb = sys.exc_info()
        stack = traceback.extract_tb(tb)[1:]
        formatted_exception = \
            'Encountered issue in callback (most recent call last):\n' + \
            ''.join(traceback.format_list(stack) +
                    traceback.format_exception_only(etype, exc)).rstrip('\n')
        logging.warning(formatted_exception)


def check_depth_peeling(number_of_peels=100, occlusion_ratio=0.0):
    """Check if depth peeling is available.

    Attempts to use depth peeling to see if it is available for the
    current environment. Returns ``True`` if depth peeling is
    available and has been successfully leveraged, otherwise
    ``False``.

    Parameters
    ----------
    number_of_peels : int, optional
        Maximum number of depth peels.

    occlusion_ratio : float, optional
        Occlusion ratio.

    Returns
    -------
    bool
        ``True`` when system supports depth peeling with the specified
        settings.

    """
    # Try Depth Peeling with a basic scene
    source = _vtk.vtkSphereSource()
    mapper = _vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    actor = _vtk.vtkActor()
    actor.SetMapper(mapper)
    # requires opacity < 1
    actor.GetProperty().SetOpacity(0.5)
    renderer = _vtk.vtkRenderer()
    renderWindow = _vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetOffScreenRendering(True)
    renderWindow.SetAlphaBitPlanes(True)
    renderWindow.SetMultiSamples(0)
    renderer.AddActor(actor)
    renderer.SetUseDepthPeeling(True)
    renderer.SetMaximumNumberOfPeels(number_of_peels)
    renderer.SetOcclusionRatio(occlusion_ratio)
    renderWindow.Render()
    return renderer.GetLastRenderingUsedDepthPeeling() == 1


def threaded(fn):
    """Call a function using a thread.

    Parameters
    ----------
    fn : callable
        Callable object.

    Returns
    -------
    function
        Wrapped function.

    """

    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper


class conditional_decorator:
    """Conditional decorator for methods.

    Parameters
    ----------
    dec
        Decorator
    condition
        Condition to match.

    """

    def __init__(self, dec, condition):
        """Initialize."""
        self.decorator = dec
        self.condition = condition

    def __call__(self, func):
        """Call the decorated function if condition is matched."""
        if not self.condition:
            # Return the function unchanged, not decorated.
            return func
        return self.decorator(func)


class ProgressMonitor():
    """A standard class for monitoring the progress of a VTK algorithm.

    This must be use in a ``with`` context and it will block keyboard
    interrupts from happening until the exit event as interrupts will crash
    the kernel if the VTK algorithm is still executing.

    Parameters
    ----------
    algorithm
        VTK algorithm or filter.

    message : str, optional
        Message to display in the progress bar.

    scaling : float, optional
        Unused keyword argument.

    """

    def __init__(self, algorithm, message="", scaling=100):
        """Initialize observer."""
        try:
            from tqdm import tqdm  # noqa
        except ImportError:
            raise ImportError("Please install `tqdm` to monitor algorithms.")
        self.event_type = _vtk.vtkCommand.ProgressEvent
        self.progress = 0.0
        self._last_progress = self.progress
        self.algorithm = algorithm
        self.message = message
        self._interrupt_signal_received = False
        self._old_progress = 0
        self._old_handler = None
        self._progress_bar = None

    def handler(self, sig, frame):
        """Pass signal to custom interrupt handler."""
        self._interrupt_signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt until '
                      'VTK algorithm finishes.')

    def __call__(self, obj, event, *args):
        """Call progress update callback.

        On an event occurrence, this function executes.
        """
        if self._interrupt_signal_received:
            obj.AbortExecuteOn()
        else:
            progress = obj.GetProgress()
            step = progress - self._old_progress
            self._progress_bar.update(step)
            self._old_progress = progress

    def __enter__(self):
        """Enter event for ``with`` context."""
        from tqdm import tqdm

        # check if in main thread
        if threading.current_thread().__class__.__name__ == '_MainThread':
            self._old_handler = signal.signal(signal.SIGINT, self.handler)
        self._progress_bar = tqdm(total=1, leave=True,
                                  bar_format='{l_bar}{bar}[{elapsed}<{remaining}]')
        self._progress_bar.set_description(self.message)
        self.algorithm.AddObserver(self.event_type, self)
        return self._progress_bar

    def __exit__(self, type, value, traceback):
        """Exit event for ``with`` context."""
        self._progress_bar.total = 1
        self._progress_bar.refresh()
        self._progress_bar.close()
        self.algorithm.RemoveObservers(self.event_type)
        if threading.current_thread().__class__.__name__ == '_MainThread':
            signal.signal(signal.SIGINT, self._old_handler)


def abstract_class(cls_):
    """Decorate a class, overriding __new__.

    Preventing a class from being instantiated similar to abc.ABCMeta
    but does not require an abstract method.
    """

    def __new__(cls, *args, **kwargs):
        if cls is cls_:
            raise TypeError(f'{cls.__name__} is an abstract class and may not be instantiated.')
        return object.__new__(cls)
    cls_.__new__ = __new__
    return cls_


def axis_rotation(points, angle, inplace=False, deg=True, axis='z'):
    """Rotate points by angle about an axis.

    Parameters
    ----------
    points : numpy.ndarray
        Array of points with shape ``(N, 3)``.

    angle : float
        Rotation angle.

    inplace : bool, optional
        Updates points in-place while returning nothing.

    deg : bool, optional
        If ``True``, the angle is interpreted as degrees instead of
        radians. Default is ``True``.

    axis : str, optional
        Name of axis to rotate about. Valid options are ``'x'``, ``'y'``,
        and ``'z'``. Default value is ``'z'``.

    Returns
    -------
    numpy.ndarray
        Rotated points.

    Examples
    --------
    Rotate a set of points by 90 degrees about the x-axis in-place.

    >>> import numpy as np
    >>> import pyvista
    >>> from pyvista import examples
    >>> points = examples.load_airplane().points
    >>> points_orig = points.copy()
    >>> pyvista.axis_rotation(points, 90, axis='x', deg=True, inplace=True)
    >>> assert np.all(np.isclose(points[:, 0], points_orig[:, 0]))
    >>> assert np.all(np.isclose(points[:, 1], -points_orig[:, 2]))
    >>> assert np.all(np.isclose(points[:, 2], points_orig[:, 1]))
    """
    axis = axis.lower()
    axis_to_vec = {
        'x': (1, 0, 0),
        'y': (0, 1, 0),
        'z': (0, 0, 1)
    }

    if axis not in axis_to_vec:
        raise ValueError('Invalid axis. Must be either "x", "y", or "z"')

    rot_mat = transformations.axis_angle_rotation(axis_to_vec[axis], angle, deg=deg)
    return transformations.apply_transformation_to_points(rot_mat, points, inplace=inplace)


def cubemap(path='', prefix='', ext='.jpg'):
    """Construct a cubemap from 6 images.

    Each of the 6 images must be in the following format:

    - <prefix>negx<ext>
    - <prefix>negy<ext>
    - <prefix>negz<ext>
    - <prefix>posx<ext>
    - <prefix>posy<ext>
    - <prefix>posz<ext>

    Prefix may be empty, and extension will default to ``'.jpg'``

    For example, if you have 6 images with the skybox2 prefix:

    - ``'skybox2-negx.jpg'``
    - ``'skybox2-negy.jpg'``
    - ``'skybox2-negz.jpg'``
    - ``'skybox2-posx.jpg'``
    - ``'skybox2-posy.jpg'``
    - ``'skybox2-posz.jpg'``

    Parameters
    ----------
    path : str, optional
        Directory containing the cubemap images.

    prefix : str, optional
        Prefix to the filename.

    ext : str, optional
        The filename extension.  For example ``'.jpg'``.

    Returns
    -------
    pyvista.Texture
        Texture with cubemap.

    Examples
    --------
    >>> import pyvista
    >>> skybox = pyvista.cubemap('my_directory', 'skybox', '.jpeg')  # doctest:+SKIP
    """
    sets = ['posx', 'negx', 'posy', 'negy', 'posz', 'negz']
    image_paths = [os.path.join(path, f'{prefix}{suffix}{ext}') for suffix in sets]

    for image_path in image_paths:
        if not os.path.isfile(image_path):
            file_str = '\n'.join(image_paths)
            raise FileNotFoundError(f'Unable to locate {image_path}\n'
                                    'Expected to find the following files:\n'
                                    f'{file_str}')

    texture = pyvista.Texture()
    texture.SetMipmap(True)
    texture.SetInterpolate(True)
    texture.cube_map = True  # Must be set prior to setting images

    # add each image to the cubemap
    for i, fn in enumerate(image_paths):
        image = pyvista.read(fn)
        flip = _vtk.vtkImageFlip()
        flip.SetInputDataObject(image)
        flip.SetFilteredAxis(1)  # flip y axis
        flip.Update()
        texture.SetInputDataObject(i, flip.GetOutput())

    return texture
