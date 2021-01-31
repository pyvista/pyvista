"""Supporting functions for polydata and grid objects."""

import collections.abc
import enum
import logging
import signal
import sys
import warnings
from threading import Thread
import threading
import traceback

import numpy as np
import scooby
import vtk
import vtk.util.numpy_support as nps

import pyvista
from .fileio import from_meshio
from . import transformations


class FieldAssociation(enum.Enum):
    """Represents which type of vtk field a scalar or vector array is associated with."""

    POINT = vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
    CELL = vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS
    NONE = vtk.vtkDataObject.FIELD_ASSOCIATION_NONE
    ROW = vtk.vtkDataObject.FIELD_ASSOCIATION_ROWS


def get_vtk_type(typ):
    """Look up the VTK type for a give python data type.

    Corrects for string type mapping issues.

    Returns
    -------
        int : the integer type id specified in vtkType.h

    """
    typ = nps.get_vtk_array_type(typ)
    # This handles a silly string type bug
    if typ == 3:
        return 13
    return typ


def vtk_bit_array_to_char(vtkarr_bint):
    """Cast vtk bit array to a char array."""
    vtkarr = vtk.vtkCharArray()
    vtkarr.DeepCopy(vtkarr_bint)
    return vtkarr


def vtk_id_list_to_array(vtk_id_list):
    """Convert a vtkIdList to a NumPy array."""
    return np.array([vtk_id_list.GetId(i) for i in range(vtk_id_list.GetNumberOfIds())])


def convert_string_array(arr, name=None):
    """Convert a numpy array of strings to a vtkStringArray or vice versa.

    Note that this is terribly inefficient - inefficient support
    is better than no support :). If you have ideas on how to make this faster,
    please consider opening a pull request.

    """
    if isinstance(arr, np.ndarray):
        vtkarr = vtk.vtkStringArray()
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


def convert_array(arr, name=None, deep=0, array_type=None):
    """Convert a NumPy array to a vtkDataArray or vice versa.

    Parameters
    -----------
    arr : ndarray or vtkDataArry
        A numpy array or vtkDataArry to convert
    name : str
        The name of the data array for VTK
    deep : bool
        if input is numpy array then deep copy values

    Returns
    -------
    vtkDataArray, ndarray, or DataFrame:
        the converted array (if input is a NumPy ndaray then returns
        ``vtkDataArray`` or is input is ``vtkDataArray`` then returns NumPy
        ``ndarray``). If pdf==True and the input is ``vtkDataArry``,
        return a pandas DataFrame.

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
            vtk_data = nps.numpy_to_vtk(num_array=arr, deep=deep, array_type=array_type)

        if isinstance(name, str):
            vtk_data.SetName(name)
        return vtk_data
    # Otherwise input must be a vtkDataArray
    if not isinstance(arr, (vtk.vtkDataArray, vtk.vtkBitArray, vtk.vtkStringArray)):
        raise TypeError(f'Invalid input array type ({type(arr)}).')
    # Handle booleans
    if isinstance(arr, vtk.vtkBitArray):
        arr = vtk_bit_array_to_char(arr)
    # Handle string arrays
    if isinstance(arr, vtk.vtkStringArray):
        return convert_string_array(arr)
    # Convert from vtkDataArry to NumPy
    return nps.vtk_to_numpy(arr)


def is_pyvista_dataset(obj):
    """Return True if the Object is a PyVista wrapped dataset."""
    return isinstance(obj, (pyvista.Common, pyvista.MultiBlock))


def point_array(mesh, name):
    """Return point array of a vtk object."""
    vtkarr = mesh.GetPointData().GetAbstractArray(name)
    return convert_array(vtkarr)


def field_array(mesh, name):
    """Return field array of a vtk object."""
    vtkarr = mesh.GetFieldData().GetAbstractArray(name)
    return convert_array(vtkarr)


def cell_array(mesh, name):
    """Return cell array of a vtk object."""
    vtkarr = mesh.GetCellData().GetAbstractArray(name)
    return convert_array(vtkarr)


def row_array(data_object, name):
    """Return row array of a vtk object."""
    vtkarr = data_object.GetRowData().GetAbstractArray(name)
    return convert_array(vtkarr)


def parse_field_choice(field):
    """Return the id of the given field."""
    if isinstance(field, str):
        field = field.strip().lower()
        if field in ['cell', 'c', 'cells']:
            field = FieldAssociation.CELL
        elif field in ['point', 'p', 'points']:
            field = FieldAssociation.POINT
        elif field in ['field', 'f', 'fields']:
            field = FieldAssociation.NONE
        elif field in ['row', 'r',]:
            field = FieldAssociation.ROW
        else:
            raise ValueError(f'Data field ({field}) not supported.')
    elif isinstance(field, FieldAssociation):
        pass
    else:
        raise ValueError(f'Data field ({field}) not supported.')
    return field


def get_array(mesh, name, preference='cell', info=False, err=False):
    """Search point, cell and field data for an array.

    Parameters
    ----------
    name : str
        The name of the array to get the range.

    preference : str, optional
        When scalars is specified, this is the preferred array type to
        search for in the dataset.  Must be either ``'point'``,
        ``'cell'``, or ``'field'``

    info : bool
        Return info about the array rather than the array itself.

    err : bool
        Boolean to control whether to throw an error if array is not present.

    """
    if isinstance(mesh, vtk.vtkTable):
        arr = row_array(mesh, name)
        if arr is None and err:
            raise KeyError(f'Data array ({name}) not present in this dataset.')
        field = FieldAssociation.ROW
        if info:
            return arr, field
        return arr

    parr = point_array(mesh, name)
    carr = cell_array(mesh, name)
    farr = field_array(mesh, name)
    preference = parse_field_choice(preference)
    if np.sum([parr is not None, carr is not None, farr is not None]) > 1:
        if preference == FieldAssociation.CELL:
            if info:
                return carr, FieldAssociation.CELL
            else:
                return carr
        elif preference == FieldAssociation.POINT:
            if info:
                return parr, FieldAssociation.POINT
            else:
                return parr
        elif preference == FieldAssociation.NONE:
            if info:
                return farr, FieldAssociation.NONE
            else:
                return farr
        else:
            raise ValueError(f'Data field ({preference}) not supported.')
    arr = None
    field = None
    if parr is not None:
        arr = parr
        field = FieldAssociation.POINT
    elif carr is not None:
        arr = carr
        field = FieldAssociation.CELL
    elif farr is not None:
        arr = farr
        field = FieldAssociation.NONE
    elif err:
        raise KeyError(f'Data array ({name}) not present in this dataset.')
    if info:
        return arr, field
    return arr


def vtk_points(points, deep=True):
    """Convert numpy points to a vtkPoints object."""
    if not points.flags['C_CONTIGUOUS']:
        points = np.ascontiguousarray(points)
    vtkpts = vtk.vtkPoints()
    vtkpts.SetData(nps.numpy_to_vtk(points, deep=deep))
    return vtkpts


def line_segments_from_points(points):
    """Generate non-connected line segments from points.

    Assumes points are ordered as line segments and an even number of points
    are

    Parameters
    ----------
    points : np.ndarray
        Points representing line segments. An even number must be given as
        every two vertices represent a single line segment. For example, two
        line segments would be represented as:

        np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])

    Returns
    -------
    lines : pyvista.PolyData
        PolyData with lines and cells.

    Examples
    --------
    This example plots two line segments at right angles to each other line.

    >>> import pyvista
    >>> import numpy as np
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])
    >>> lines = pyvista.lines_from_points(points)
    >>> lines.plot() # doctest:+SKIP

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
        Points representing the vertices of the connected segments. For
        example, two line segments would be represented as:

        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])

    close : bool, optional
        If True, close the line segments into a loop

    Returns
    -------
    lines : pyvista.PolyData
        PolyData with lines and cells.

    """
    poly = pyvista.PolyData()
    poly.points = points
    cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    if close:
        cells = np.append(cells, [[2, len(points)-1, 0],], axis=0)
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
        Array of points with shape (N, 3) storing the vertices of the
        triangle mesh.

    faces : np.ndarray
        Array of indices with shape (M, 3) containing the triangle
        indices.

    Returns
    -------
    tri_mesh : pyvista.PolyData
        PolyData instance containing the triangle mesh.

    Examples
    --------
    This example discretizes the unit square into a triangle mesh with
    nine vertices and eight faces.

    >>> import numpy as np
    >>> import pyvista as pv
    >>> points = np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [0, 0.5, 0],
    ...                    [0.5, 0.5, 0], [1, 0.5, 0], [0, 1, 0], [0.5, 1, 0],
    ...                    [1, 1, 0]])
    >>> faces = np.array([[0, 1, 4], [4, 7, 6], [2, 5, 4], [4, 5, 8],
    ...                   [0, 4, 3], [3, 4, 6], [1, 2, 4], [4, 8, 7]])
    >>> tri_mesh = pyvista.make_tri_mesh(points, faces)
    >>> tri_mesh.plot(show_edges=True) # doctest:+SKIP

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
    """Create a vtkPolyData object composed of vectors."""
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
    vpts = vtk.vtkPoints()
    vpts.SetData(nps.numpy_to_vtk(np.ascontiguousarray(orig), deep=True))

    npts = orig.shape[0]
    cells = np.empty((npts, 2), dtype=pyvista.ID_TYPE)
    cells[:, 0] = 1
    cells[:, 1] = np.arange(npts, dtype=pyvista.ID_TYPE)
    vcells = pyvista.utilities.cells.CellArray(cells, npts)

    # Create vtkPolyData object
    pdata = vtk.vtkPolyData()
    pdata.SetPoints(vpts)
    pdata.SetVerts(vcells)

    # Add vectors to polydata
    name = 'vectors'
    vtkfloat = nps.numpy_to_vtk(np.ascontiguousarray(vec), deep=True)
    vtkfloat.SetName(name)
    pdata.GetPointData().AddArray(vtkfloat)
    pdata.GetPointData().SetActiveVectors(name)

    # Add magnitude of vectors to polydata
    name = 'mag'
    scalars = (vec * vec).sum(1)**0.5
    vtkfloat = nps.numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
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
    """Convert a vtk matrix to a ``numpy.ndarray``.

    Parameters
    ----------
    matrix : vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4
        The vtk matrix to be converted to a ``numpy.ndarray``.
        Returned ndarray has shape (3, 3) or (4, 4) as appropriate.

    """
    if isinstance(matrix, vtk.vtkMatrix3x3):
        shape = (3, 3)
    elif isinstance(matrix, vtk.vtkMatrix4x4):
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

    """
    array = np.asarray(array)
    if array.shape == (3, 3):
        matrix = vtk.vtkMatrix3x3()
    elif array.shape == (4, 4):
        matrix = vtk.vtkMatrix4x4()
    else:
        raise ValueError(f'Invalid shape {array.shape}, must be (3, 3) or (4, 4).')
    m, n = array.shape
    for i in range(m):
        for j in range(n):
            matrix.SetElement(i, j, array[i, j])
    return matrix


def is_meshio_mesh(mesh):
    """Test if passed object is instance of ``meshio.Mesh``."""
    try:
        import meshio
        return isinstance(mesh, meshio.Mesh)
    except ImportError:
        return False


def wrap(dataset):
    """Wrap any given VTK data object to its appropriate PyVista data object.

    Other formats that are supported include:
    * 2D :class:`numpy.ndarray` of XYZ vertices
    * 3D :class:`numpy.ndarray` representing a volume. Values will be scalars.
    * 3D :class:`trimesh.Trimesh` mesh.

    Parameters
    ----------
    dataset : :class:`numpy.ndarray`, :class:`trimesh.Trimesh`, or VTK object
        Dataset to wrap.

    Returns
    -------
    wrapped_dataset : pyvista class
        The `pyvista` wrapped dataset.

    Examples
    --------
    Wrap a numpy array representing a random point cloud

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

    Wrap a Trimesh object

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

    Wrap a VTK object

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
    wrappers = {
        'vtkUnstructuredGrid': pyvista.UnstructuredGrid,
        'vtkRectilinearGrid': pyvista.RectilinearGrid,
        'vtkStructuredGrid': pyvista.StructuredGrid,
        'vtkPolyData': pyvista.PolyData,
        'vtkImageData': pyvista.UniformGrid,
        'vtkStructuredPoints': pyvista.UniformGrid,
        'vtkMultiBlockDataSet': pyvista.MultiBlock,
        'vtkTable': pyvista.Table,
        # 'vtkParametricSpline': pyvista.Spline,
    }
    # Otherwise, we assume a VTK data object was passed
    if hasattr(dataset, 'GetClassName'):
        key = dataset.GetClassName()
    elif dataset is None:
        return None
    elif isinstance(dataset, np.ndarray):
        if dataset.ndim == 1 and dataset.shape[0] == 3:
            return pyvista.PolyData(dataset)
        if dataset.ndim > 1 and dataset.ndim < 3 and dataset.shape[1] == 3:
            return pyvista.PolyData(dataset)
        elif dataset.ndim == 3:
            mesh = pyvista.UniformGrid(dataset.shape)
            mesh['values'] = dataset.ravel(order='F')
            mesh.active_scalars_name = 'values'
            return mesh
        else:
            print(dataset.shape, dataset)
            raise NotImplementedError('NumPy array could not be converted to PyVista.')
    elif is_meshio_mesh(dataset):
        return from_meshio(dataset)
    elif dataset.__class__.__name__ == 'Trimesh':
        # trimesh doesn't pad faces
        n_face = dataset.faces.shape[0]
        faces = np.empty((n_face, 4), dataset.faces.dtype)
        faces[:, 1:] = dataset.faces
        faces[:, 0] = 3
        return pyvista.PolyData(np.asarray(dataset.vertices), faces)
    else:
        raise NotImplementedError(f'Type ({type(dataset)}) not able to be wrapped into a PyVista mesh.')
    try:
        wrapped = wrappers[key](dataset)
    except KeyError:
        logging.warning(f'VTK data type ({key}) is not currently supported by pyvista.')
        return dataset  # if not supported just passes the VTK data object
    return wrapped


def image_to_texture(image):
    """Convert ``vtkImageData`` (:class:`pyvista.UniformGrid`) to a ``vtkTexture``."""
    return pyvista.Texture(image)


def numpy_to_texture(image):
    """Convert a NumPy image array to a vtk.vtkTexture."""
    return pyvista.Texture(image)


def is_inside_bounds(point, bounds):
    """Check if a point is inside a set of bounds.

    This is implemented through recursion so that this is N-dimensional.

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
    """Fit a plane to a set of points.

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
    plane = pyvista.Plane(center=center, direction=normal)
    if return_meta:
        return plane, center, normal
    return plane


def raise_not_matching(scalars, mesh):
    """Raise exception about inconsistencies."""
    if isinstance(mesh, vtk.vtkTable):
        raise ValueError(f'Number of scalars ({scalars.size}) must match number of rows ({mesh.n_rows}).')
    raise ValueError(f'Number of scalars ({scalars.size}) ' +
                     f'must match either the number of points ({mesh.n_points}) ' +
                     f'or the number of cells ({mesh.n_cells}).')


def generate_plane(normal, origin):
    """Return a vtk.vtkPlane."""
    plane = vtk.vtkPlane()
    # NORMAL MUST HAVE MAGNITUDE OF 1
    normal = normal / np.linalg.norm(normal)
    plane.SetNormal(normal)
    plane.SetOrigin(origin)
    return plane


def try_callback(func, *args):
    """Wrap a given callback in a try statement."""
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
    return


def check_depth_peeling(number_of_peels=100, occlusion_ratio=0.0):
    """Check if depth peeling is available.

    Attempts to use depth peeling to see if it is available for the current
    environment. Returns ``True`` if depth peeling is available and has been
    successfully leveraged, otherwise ``False``.

    """
    # Try Depth Peeling with a basic scene
    source = vtk.vtkSphereSource()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # requires opacity < 1
    actor.GetProperty().SetOpacity(0.5)
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
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
    """Call a function using a thread."""

    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper


class conditional_decorator:
    """Conditional decorator for methods."""

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

    """

    def __init__(self, algorithm, message="", scaling=100):
        """Initialize observer."""
        try:
            from tqdm import tqdm
        except ImportError:
            raise ImportError("Please install `tqdm` to monitor algorithms.")
        self.event_type = vtk.vtkCommand.ProgressEvent
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
    """Rotate points angle (in deg) about an axis.

    Parameters
    ----------
    points : numpy.ndarray
        Array of points with shape ``(N, 3)``

    angle : float
        Rotation angle.

    inplace : bool, optional
        Updates points in-place while returning nothing.

    deg : bool, optional
        If `True`, the angle is interpreted as degrees instead of
        radians. Default is `True`.

    axis : str, optional
        Name of axis to rotate about. Valid options are ``'x'``, ``'y'``,
        and ``'z'``. Default value is ``'z'``.

    Returns
    -------
    points : numpy.ndarray
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
