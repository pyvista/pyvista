"""Points related utilities."""
import warnings

import numpy as np

import pyvista
from pyvista.core import _vtk_core as _vtk


def vtk_points(points, deep=True, force_float=False):
    """Convert numpy array or array-like to a ``vtkPoints`` object.

    Parameters
    ----------
    points : numpy.ndarray or sequence
        Points to convert.  Should be 1 or 2 dimensional.  Accepts a
        single point or several points.

    deep : bool, default: True
        Perform a deep copy of the array.  Only applicable if
        ``points`` is a :class:`numpy.ndarray`.

    force_float : bool, default: False
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
        raise ValueError(f'Dimension of ``points`` should be 1 or 2, not {points.ndim}')

    # verify shape
    if points.shape[1] != 3:
        raise ValueError(
            'Points array must contain three values per point. '
            f'Shape is {points.shape} and should be (X, 3)'
        )

    # use the underlying vtk data if present to avoid memory leaks
    if not deep and isinstance(points, pyvista.pyvista_ndarray):
        if points.VTKObject is not None:
            vtk_object = points.VTKObject

            # we can only use the underlying data if `points` is not a slice of
            # the VTK data object
            if vtk_object.GetSize() == points.size:
                vtkpts = _vtk.vtkPoints()
                vtkpts.SetData(points.VTKObject)
                return vtkpts
            else:
                deep = True

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
    points : array_like[float]
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
    >>> lines = pyvista.line_segments_from_points(points)
    >>> lines.plot()

    """
    if len(points) % 2 != 0:
        raise ValueError("An even number of points must be given to define each segment.")
    # Assuming ordered points, create array defining line order
    n_points = len(points)
    n_lines = n_points // 2
    lines = np.c_[
        (
            2 * np.ones(n_lines, np.int_),
            np.arange(0, n_points - 1, step=2),
            np.arange(1, n_points + 1, step=2),
        )
    ]
    poly = pyvista.PolyData()
    poly.points = points
    poly.lines = lines
    return poly


def lines_from_points(points, close=False):
    """Make a connected line set given an array of points.

    Parameters
    ----------
    points : array_like[float]
        Points representing the vertices of the connected
        segments. For example, two line segments would be represented
        as ``np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])``.

    close : bool, default: False
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
    cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    if close:
        cells = np.append(cells, [[2, len(points) - 1, 0]], axis=0)
    poly.lines = cells
    return poly


def fit_plane_to_points(points, return_meta=False):
    """Fit a plane to a set of points using the SVD algorithm.

    Parameters
    ----------
    points : array_like[float]
        Size ``[N x 3]`` sequence of points to fit a plane through.

    return_meta : bool, default: False
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
    >>> plane, center, normal = pyvista.fit_plane_to_points(
    ...     cloud, return_meta=True
    ... )

    Plot the fitted plane.

    >>> pl = pyvista.Plotter()
    >>> _ = pl.add_mesh(
    ...     plane, color='lightblue', style='wireframe', line_width=4
    ... )
    >>> _ = pl.add_points(
    ...     cloud,
    ...     render_points_as_spheres=True,
    ...     color='r',
    ...     point_size=30,
    ... )
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
    >>> points = np.array(
    ...     [
    ...         [0, 0, 0],
    ...         [0.5, 0, 0],
    ...         [1, 0, 0],
    ...         [0, 0.5, 0],
    ...         [0.5, 0.5, 0],
    ...         [1, 0.5, 0],
    ...         [0, 1, 0],
    ...         [0.5, 1, 0],
    ...         [1, 1, 0],
    ...     ]
    ... )
    >>> faces = np.array(
    ...     [
    ...         [0, 1, 4],
    ...         [4, 7, 6],
    ...         [2, 5, 4],
    ...         [4, 5, 8],
    ...         [0, 4, 3],
    ...         [3, 4, 6],
    ...         [1, 2, 4],
    ...         [4, 8, 7],
    ...     ]
    ... )
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
    orig : array_like[float]
        Array of vector origins.

    vec : array_like[float]
        Array of vectors.

    Returns
    -------
    pyvista.PolyData
        Mesh containing the ``orig`` points along with the
        ``'vectors'`` and ``'mag'`` point arrays representing the
        vectors and magnitude of the vectors at each point.

    Examples
    --------
    Create basic vector field.  This is a point cloud where each point
    has a vector and magnitude attached to it.

    >>> import pyvista
    >>> import numpy as np
    >>> x, y = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
    >>> points = np.vstack((x.ravel(), y.ravel(), np.zeros(x.size))).T
    >>> u = x / np.sqrt(x**2 + y**2)
    >>> v = y / np.sqrt(x**2 + y**2)
    >>> vectors = np.vstack(
    ...     (u.ravel() ** 3, v.ravel() ** 3, np.zeros(u.size))
    ... ).T
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
    vcells = pyvista.core.cell.CellArray(cells, npts)

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
    scalars = (vec * vec).sum(1) ** 0.5
    vtkfloat = _vtk.numpy_to_vtk(np.ascontiguousarray(scalars), deep=True)
    vtkfloat.SetName(name)
    pdata.GetPointData().AddArray(vtkfloat)
    pdata.GetPointData().SetActiveScalars(name)

    return pyvista.PolyData(pdata)
