"""Points related utilities."""
from typing import Literal
import warnings

import numpy as np

import pyvista
from pyvista.core import _vtk_core as _vtk
from pyvista.core.utilities.arrays import _coerce_pointslike_arg
from pyvista.core.utilities.geometric_objects import NORMALS
from pyvista.core.utilities.misc import check_valid_vector


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


def principal_axes_transform(
    points,
    axis_0_direction=None,
    axis_1_direction=None,
    axis_2_direction=None,
):
    """Compute the principal axes transform.

    This function uses :func:`~pyvista.principal_axes_vectors` to get
    the transformation matrix which will:
        1. Translate ``points`` such that their centroid is at the origin, then
        2. Rotate ``points`` to align their principal axes to the XYZ axes.

    .. versionadded:: 0.43.0

    Notes
    -----
        If the transform cannot be computed, the identity matrix is returned.

    See Also
    --------
        :func:`~pyvista.principal_axes_vectors`, :attr:`~pyvista.DataSet.principal_axes`

    Parameters
    ----------
    points : array_like[float]
        Points array. Accepts a single point or several points as a
        Nx3 array.

    axis_0_direction : sequence[float] | str, optional
        Approximate direction vector of the first axis. If set, the
        direction of the first orthonormal axis will be flipped such
        that it best aligns with this vector. Has no effect if this
        vector is perpendicular to the applied axis. Can be a sequence
        of three elements specifying the ``(x, y, z)`` direction or a
        string specifying a conventional direction (e.g. ``'x'`` for
        ``(1, 0, 0)`` or ``'-x'`` for ``(-1, 0, 0)``, etc.).

    axis_1_direction : sequence[float] | str, optional
        Approximate direction vector of the second axis. If set, the
        direction of the second orthonormal axis will be flipped such
        that it best aligns with this vector. Has no effect if this
        vector is perpendicular to the second axis.

    axis_2_direction : sequence[float] | str, optional
        Approximate direction vector of the third axis. If set, the
        direction of the third orthonormal axis will be flipped such
        that it best aligns with this vector. Has no effect if this
        vector is perpendicular to the second axis, or if
        ``axis_0_direction`` and ``axis_1_direction`` are set.

    Returns
    -------
    numpy.ndarray
        The 4x4 transformation matrix which aligns the points to the XYZ axes
        at the origin.

    """
    return principal_axes_vectors(
        points,
        axis_0_direction=axis_0_direction,
        axis_1_direction=axis_1_direction,
        axis_2_direction=axis_2_direction,
        as_transform=True,
    )


def principal_axes_vectors(
    points,
    axis_0_direction=None,
    axis_1_direction=None,
    axis_2_direction=None,
    as_transform=False,
):
    """Compute the principal axes vectors from a set of points.

    The mesh's principal axes are orthonormal row vectors that best
    fit its points. The axes are computed using Singular Value
    Decomposition (SVD) and the points are centered at their mean prior
    to the computation. The first axis direction explains the most
    variance in the points, and the third axis direction explains
    the least variance in the points.

    The computed axes are not unique and their directions are arbitrary.
    As such, approximate direction vectors may optionally be specified
    to control the axis directions. This can be used, for example, to
    define a local coordinate frame where one or two axis directions
    have a clear physical meaning.

    .. versionadded:: 0.43.0

    Notes
    -----
        If the axes cannot be computed, the identity matrix is returned.

    See Also
    --------
        :attr:`~pyvista.DataSet.principal_axes`,
        :func:`~pyvista.principal_axes_transform`,
        :func:`~pyvista.fit_plane_to_points`

    Parameters
    ----------
    points : array_like[float]
        Points array. Accepts a single point or several points as a
        Nx3 array.

    axis_0_direction : sequence[float] | str, optional
        Approximate direction vector of the first axis. If set, the
        direction of the first orthonormal axis will be flipped such
        that it best aligns with this vector. Has no effect if this
        vector is perpendicular to the applied axis. Can be a sequence
        of three elements specifying the ``(x, y, z)`` direction or a
        string specifying a conventional direction (e.g. ``'x'`` for
        ``(1, 0, 0)`` or ``'-x'`` for ``(-1, 0, 0)``, etc.).

    axis_1_direction : sequence[float] | str, optional
        Approximate direction vector of the second axis. If set, the
        direction of the second orthonormal axis will be flipped such
        that it best aligns with this vector. Has no effect if this
        vector is perpendicular to the second axis.

    axis_2_direction : sequence[float] | str, optional
        Approximate direction vector of the third axis. If set, the
        direction of the third orthonormal axis will be flipped such
        that it best aligns with this vector. Has no effect if this
        vector is perpendicular to the second axis, or if
        ``axis_0_direction`` and ``axis_1_direction`` are set.

    as_transform : bool, False
        If ``True``, the axes are used to compute a 4x4 transformation
        matrix. The transform translates the points to be centered at the
        origin, then rotates the points to align the orthonormal axes to
        the XYZ axes.

    Returns
    -------
    numpy.ndarray
        A 3x3 array with the principal axes as row vectors or a 4x4
        transformation matrix if ``as_transform=True``.

    """

    def _validate_vector(vector):
        if vector is not None:
            if isinstance(vector, str):
                vector = vector.lower()
                valid_strings = list(NORMALS.keys())
                if vector not in valid_strings:
                    raise ValueError(
                        f"Vector string must be one of {valid_strings}, got {vector} instead."
                    )
                vector = NORMALS[vector.lower()]
            check_valid_vector(vector)
        return vector

    axis_0_direction = _validate_vector(axis_0_direction)
    axis_1_direction = _validate_vector(axis_1_direction)
    axis_2_direction = _validate_vector(axis_2_direction)

    # Compare all direction vectors
    directions = [vec for vec in [axis_0_direction, axis_1_direction, axis_2_direction] if vec]
    for i, vec1 in enumerate(directions):
        for j, vec2 in enumerate(directions[i + 1 : :]):
            if np.allclose(vec1, vec2):
                raise ValueError("Direction vectors must be distinct.")

    # Initialize output
    if as_transform:
        default_output = np.eye(4)
    else:
        default_output = np.eye(3)

    # Validate points
    data, _ = _coerce_pointslike_arg(points, copy=True)
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)
    if len(data) == 0:
        return default_output

    # Center data
    centroid = data.mean(axis=0)
    data -= centroid

    try:
        # Use SVD as it's numerically more stable than using PCA (below)
        _, _, axes_vectors = np.linalg.svd(data)

        ## Equivalently (up to a difference in axis sign and numerical error),
        ## the axes may also be computed using PCA (i.e. using covariance and
        ## eigenvalue decomposition).
        # covariance = np.cov(data, rowvar=False)
        # _, axes_vectors = np.linalg.eigh(covariance)  # column vectors, ascending order
        # axes_vectors = axes_vectors.T[::-1]  # row vectors, descending order

    except np.linalg.LinAlgError:
        return default_output

    # Normalize to unit-length, flip directions, and ensure vectors form
    # a right-hand coordinate system
    i_vector = axes_vectors[0] / np.linalg.norm(axes_vectors[0])
    if axis_0_direction:
        sign = np.sign(np.dot(i_vector, axis_0_direction))
        if sign != 0:  # Only change sign if not perpendicular
            i_vector *= sign

    j_vector = axes_vectors[1] / np.linalg.norm(axes_vectors[1])
    if axis_1_direction:
        sign = np.sign(np.dot(j_vector, axis_1_direction))
        if sign != 0:
            j_vector *= sign

    k_vector = np.cross(i_vector, j_vector)
    if axis_2_direction:
        sign = np.sign(np.dot(k_vector, axis_2_direction))
        if axis_0_direction and axis_1_direction:
            pass  # k direction is already pre-determined
        else:
            # Need to modify two vectors to keep system as right-handed
            if axis_1_direction:
                # Do not modify j vector
                i_vector *= sign
                k_vector *= sign
            else:
                # Do not modify i vector
                j_vector *= sign
                k_vector *= sign

    axes_vectors = np.row_stack((i_vector, j_vector, k_vector))

    if as_transform:
        # Create a 4x4 transformation matrix to align orthonormal axes
        # to the XYZ axes
        rotate_to_xyz = np.eye(4)
        rotate_to_xyz[:3, :3] = axes_vectors
        translate_to_origin = np.eye(4)
        translate_to_origin[:3, 3] = -centroid
        transform = rotate_to_xyz @ translate_to_origin
        return transform

    return axes_vectors


def fit_plane_to_points(
    points,
    return_meta=False,
    i_resolution=10,
    j_resolution=10,
    normal_direction=None,
):
    """Fit a plane to a set of points.

    The plane is fitted to the points using :func:~pyvista.principal_axes_vectors,
    and is automatically sized to fit the extents of the points.

    Optionally, the sign of the normal can be controlled by specifying
    an approximate normal direction. This can be useful, for example,
    in cases where the normal direction has a clear physical meaning.

    See Also
    --------
        :func:`~pyvista.principal_axes_vectors`

    Parameters
    ----------
    points : array_like[float]
        Size ``[N x 3]`` sequence of points to fit a plane through.

    return_meta : bool, default: False
        If ``True``, also returns the center and normal of the
        generated plane.

    i_resolution : int, default: 10
        Number of points on the plane mesh in the direction of its long
        edge.

        .. versionadded:: 0.43.0

    j_resolution : int, default: 10
        Number of points on the plane mesh in the direction of its short
        edge.

        .. versionadded:: 0.43.0

    normal_direction : sequence[float] | str, optional
        Approximate direction vector of the plane's normal. If set, the
        direction of the plane's normal will be flipped suc that it best
        aligns with this vector. Can be a sequence of three elements
        specifying the ``(x, y, z)`` direction or a string specifying
        a conventional direction (e.g. ``'x'`` for ``(1, 0, 0)`` or
        ``'-x'`` for ``(-1, 0, 0)``, etc.).

        .. versionadded:: 0.43.0

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
    >>>
    >>> # Create point cloud
    >>> cloud = np.random.random((10, 3))
    >>> cloud[:, 2] *= 0.1
    >>>
    >>> # Fit plane
    >>> plane, center, normal = pyvista.fit_plane_to_points(
    ...     cloud, return_meta=True
    ... )
    >>>
    >>> # Plot the fitted plane
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

    Fit a plane to a mesh.

    >>> import pyvista
    >>> from pyvista import examples
    >>>
    >>> # Create mesh
    >>> mesh = examples.download_shark()
    >>>
    >>> # Fit plane
    >>> plane = pyvista.fit_plane_to_points(
    ...     mesh.points, i_resolution=1, j_resolution=1
    ... )
    >>>
    >>> # Plot the fitted plane
    >>> pl = pyvista.Plotter()
    >>> _ = pl.add_mesh(
    ...     plane, show_edges=True, color='lightblue', opacity=0.25
    ... )
    >>> _ = pl.add_mesh(mesh, color='gray')
    >>> pl.camera_position = [
    ...     (-117, 76, 235),
    ...     (1.69, -1.38, 0),
    ...     (0.189, 0.957, -0.22),
    ... ]
    >>> pl.show()


    """
    vectors = principal_axes_vectors(points, axis_2_direction=normal_direction)
    normal = vectors[2]

    # Create rotation matrix from basis vectors
    rotate_transform = np.eye(4)
    rotate_transform[:3, :3] = vectors
    rotate_transform_inv = rotate_transform.T

    # Project and transform points to align and center data to the XY plane
    poly = pyvista.PolyData(points)
    data_center = points.mean(axis=0)
    projected = poly.project_points_to_plane(origin=data_center, normal=normal)
    projected.points -= data_center
    projected.transform(rotate_transform)

    # Compute size of the plane
    i_size = projected.bounds[1] - projected.bounds[0]
    j_size = projected.bounds[3] - projected.bounds[2]

    # The center of the input data does not necessarily coincide with
    # the center of the plane. The true center of the plane is the
    # middle of the bounding box of the projected + transformed data
    # relative to the input data's center
    center = rotate_transform_inv[:3, :3] @ projected.center + data_center

    # Initialize plane then move to final position
    plane = pyvista.Plane(
        center=(0, 0, 0),
        direction=(0, 0, 1),
        i_size=i_size,
        j_size=j_size,
        i_resolution=i_resolution,
        j_resolution=j_resolution,
    )
    plane.transform(rotate_transform_inv)
    plane.points += center

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
