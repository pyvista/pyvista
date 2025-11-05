"""Points related utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal
from typing import overload
import warnings

import numpy as np

import pyvista
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core import _validation
from pyvista.core import _vtk_core as _vtk

if TYPE_CHECKING:
    from pyvista import PolyData
    from pyvista.core._typing_core import MatrixLike
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core import VectorLike


@_deprecate_positional_args(allowed=['points'])
def vtk_points(  # noqa: PLR0917
    points: VectorLike[float] | MatrixLike[float],
    deep: bool = True,  # noqa: FBT001, FBT002
    force_float: bool = False,  # noqa: FBT001, FBT002
    allow_empty: bool = True,  # noqa: FBT001, FBT002
) -> _vtk.vtkPoints:
    """Convert numpy array or array-like to a :vtk:`vtkPoints` object.

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

    allow_empty : bool, default: True
        Allow ``points`` to be an empty array. If ``False``, points
        must be strictly one- or two-dimensional.

        .. versionadded:: 0.45

    Returns
    -------
    :vtk:`vtkPoints`
        The :vtk:`vtkPoints` object.

    Examples
    --------
    >>> import pyvista as pv
    >>> import numpy as np
    >>> points = np.random.default_rng().random((10, 3))
    >>> vpoints = pv.vtk_points(points)
    >>> vpoints  # doctest:+SKIP
    (vtkmodules.vtkCommonCore.vtkPoints)0x7f0c2e26af40

    """
    try:
        points_ = _validation.validate_arrayNx3(points, name='points')
    except ValueError as e:
        if 'points has shape (0,)' in repr(e) and allow_empty:
            points_ = np.empty(shape=(0, 3), dtype=np.array(points).dtype)
        else:
            raise

    if force_float and not np.issubdtype(points_.dtype, np.floating):
        warnings.warn(
            'Points is not a float type. This can cause issues when '
            'transforming or applying filters. Casting to '
            '``np.float32``. Disable this by passing '
            '``force_float=False``.',
            stacklevel=2,
        )
        points_ = points_.astype(np.float32)

    # use the underlying vtk data if present to avoid memory leaks
    if not deep and isinstance(points_, pyvista.pyvista_ndarray) and points_.VTKObject is not None:
        vtk_object = points_.VTKObject

        # we can only use the underlying data if `points` is not a slice of
        # the VTK data object
        if vtk_object.GetSize() == points_.size:
            vtkpts = _vtk.vtkPoints()
            vtkpts.SetData(points_.VTKObject)
            return vtkpts
        else:
            deep = True

    # points must be contiguous
    points_ = np.require(points_, requirements=['C'])
    vtkpts = _vtk.vtkPoints()
    vtk_arr = _vtk.numpy_to_vtk(points_, deep=deep)
    vtkpts.SetData(vtk_arr)

    return vtkpts


def line_segments_from_points(points: VectorLike[float] | MatrixLike[float]) -> PolyData:
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

    >>> import pyvista as pv
    >>> import numpy as np
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])
    >>> lines = pv.line_segments_from_points(points)
    >>> lines.plot()

    """
    if len(points) % 2 != 0:
        msg = 'An even number of points must be given to define each segment.'
        raise ValueError(msg)
    # Assuming ordered points, create array defining line order
    n_points = len(points)
    n_lines = n_points // 2
    lines = np.c_[
        2 * np.ones(n_lines, np.int_),
        np.arange(0, n_points - 1, step=2),
        np.arange(1, n_points + 1, step=2),
    ]
    poly = pyvista.PolyData()
    poly.points = points
    poly.lines = lines
    return poly


@_deprecate_positional_args(allowed=['points'])
def lines_from_points(
    points: VectorLike[float] | MatrixLike[float],
    close: bool = False,  # noqa: FBT001, FBT002
) -> PolyData:
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
    >>> import pyvista as pv
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    >>> poly = pv.lines_from_points(points)
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


@_deprecate_positional_args(allowed=['points'])
def fit_plane_to_points(  # noqa: PLR0917
    points: MatrixLike[float],
    return_meta: bool = False,  # noqa: FBT001, FBT002
    resolution: int = 10,
    init_normal: VectorLike[float] | None = None,
) -> PolyData | tuple[PolyData, float, NumpyArray[float]]:
    """Fit a plane to points using its :func:`principal_axes`.

    The plane is automatically sized and oriented to fit the extents of
    the points.

    .. versionchanged:: 0.42.0
        The generated plane is now sized and oriented to match the points.

    .. versionchanged:: 0.42.0
        The center of the plane (returned if ``return_meta=True``) is now
        computed as the center of the generated plane mesh. In previous
        versions, the center of the input points was returned.

    .. versionchanged:: 0.45.0
        The internal method used for fitting the plane has changed. Previously, singular
        value decomposition (SVD) was used, but eigenvectors are now used instead.
        See warning below.

    .. warning::
        The sign of the plane's normal vector prior to version 0.45 may differ
        from the latest version. This may impact methods which rely on the plane's
        direction. Use ``init_normal`` to control the sign explicitly.

    Parameters
    ----------
    points : array_like[float]
        Size ``[N x 3]`` sequence of points to fit a plane through.

    return_meta : bool, default: False
        If ``True``, also returns the center and normal of the
        generated plane.

    resolution : int, default: 10
        Number of points on the plane mesh along its edges. Specify two numbers to
        set the resolution along the plane's long and short edge (respectively) or
        a single number to set both edges to have the same resolution.

        .. versionadded:: 0.45.0

    init_normal : VectorLike[float] | str, optional
        Flip the normal of the plane such that it best aligns with this vector. Can be
        a vector or string specifying the axis by name (e.g. ``'x'`` or ``'-x'``, etc.).

        .. versionadded:: 0.45.0

    Returns
    -------
    pyvista.PolyData
        Plane mesh.

    pyvista.pyvista_ndarray
        Plane center if ``return_meta=True``.

    pyvista.pyvista_ndarray
        Plane normal if ``return_meta=True``.

    See Also
    --------
    fit_line_to_points
        Fit a line using the first principal axis of the points.

    principal_axes
        Compute axes vectors which best fit a set of points.

    Examples
    --------
    Fit a plane to a random point cloud.

    >>> import pyvista as pv
    >>> import numpy as np
    >>> from pyvista import examples
    >>>
    >>> rng = np.random.default_rng(seed=0)
    >>> cloud = rng.random((10, 3))
    >>> cloud[:, 2] *= 0.1
    >>>
    >>> plane = pv.fit_plane_to_points(cloud)

    Plot the point cloud and fitted plane.

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(plane, style='wireframe', line_width=4)
    >>> _ = pl.add_points(
    ...     cloud,
    ...     render_points_as_spheres=True,
    ...     color='r',
    ...     point_size=30,
    ... )
    >>> pl.show()

    Fit a plane to a mesh and return its metadata. Set the plane resolution to 1
    so that the plane has no internal points or edges.

    >>> mesh = examples.download_shark()
    >>> plane, center, normal = pv.fit_plane_to_points(
    ...     mesh.points, return_meta=True, resolution=1
    ... )

    Plot the mesh and fitted plane.

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(plane, show_edges=True, opacity=0.25)
    >>> _ = pl.add_mesh(mesh, color='gray')
    >>> pl.camera_position = pv.CameraPosition(
    ...     position=(-117, 76, 235),
    ...     focal_point=(1.69, -1.38, 0),
    ...     viewup=(0.189, 0.957, -0.22),
    ... )
    >>> pl.show()

    Use the metadata with :meth:`pyvista.DataObjectFilters.clip` to split the mesh into
    two.

    >>> first_half, second_half = mesh.clip(
    ...     origin=center, normal=normal, return_clipped=True
    ... )

    Plot the two halves of the clipped mesh.

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(first_half, color='red')
    >>> _ = pl.add_mesh(second_half, color='blue')
    >>> pl.camera_position = pv.CameraPosition(
    ...     position=(-143, 43, 40),
    ...     focal_point=(-8.7, -11, -14),
    ...     viewup=(0.25, 0.92, -0.29),
    ... )
    >>> pl.show()

    Note that it is pointing in the positive z-direction.

    >>> normal  # doctest:+SKIP
    pyvista_ndarray([0.0, 0.0, 1.0], dtype=float32)

    Use ``init_normal`` to flip the sign and make it negative instead.

    >>> _, _, normal = pv.fit_plane_to_points(
    ...     mesh.points, return_meta=True, init_normal='-z'
    ... )
    >>> normal  # doctest:+SKIP
    pyvista_ndarray([0.0, 0.0, -1.0], dtype=float32)

    """
    valid_resolution = _validation.validate_array(
        resolution,
        must_have_shape=[(), (2,)],
        must_be_integer=True,
        broadcast_to=(2,),
        dtype_out=int,
    )
    i_resolution, j_resolution = valid_resolution

    # Align points to the xyz-axes
    aligned, matrix = pyvista.PolyData(points).align_xyz(
        return_matrix=True, axis_2_direction=init_normal
    )

    # Fit plane to xyz-aligned mesh
    i_size, j_size, _ = aligned.bounds_size
    plane = pyvista.Plane(
        i_size=i_size,
        j_size=j_size,
        i_resolution=i_resolution,
        j_resolution=j_resolution,
    )

    # Transform plane back to input points positioning
    inverse_matrix = pyvista.Transform(matrix).inverse_matrix
    plane.transform(inverse_matrix, inplace=True)

    if return_meta:
        # Compute center and normal from the plane's points and normals
        center = np.mean(plane.points, axis=0)
        normal = np.mean(plane.point_normals, axis=0)
        return plane, center, normal
    return plane


def fit_line_to_points(
    points: MatrixLike[float],
    *,
    resolution: int = 1,
    init_direction: VectorLike[float] | None = None,
    return_meta: bool = False,
) -> PolyData | tuple[PolyData, float, NumpyArray[float]]:
    """Fit a line to points using its :func:`principal_axes`.

    The line is automatically sized and oriented to fit the extents of
    the points.

    .. versionadded:: 0.45.0

    Parameters
    ----------
    points : MatrixLike[float]
        Size ``[N x 3]`` array of points to fit a line through.

    resolution : int, default: 1
        Number of pieces to divide the line into.

    init_direction : VectorLike[float], optional
        Flip the direction of the line's points such that it best aligns with this
        vector. Can be a vector or string specifying the axis by name (e.g. ``'x'``
        or ``'-x'``, etc.).

    return_meta : bool, default: False
        If ``True``, also returns the length (magnitude) and direction of the line.

    See Also
    --------
    fit_plane_to_points
        Fit a plane using the first two principal axes of the points.

    principal_axes
        Compute axes vectors which best fit a set of points.

    Returns
    -------
    pyvista.PolyData
        Line mesh.

    float
        Line length if ``return_meta=True``.

    numpy.ndarray
        Line direction (unit vector) if ``return_meta=True``.

    Examples
    --------
    Download a point cloud. The points trace a path along topographical surface.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> mesh = examples.download_gpr_path()

    Fit a line to the points and plot the result. The line of best fit is colored red.

    >>> line = pv.fit_line_to_points(mesh.points)

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(mesh, color='black', line_width=10)
    >>> _ = pl.add_mesh(line, color='red', line_width=5)
    >>> pl.show()

    Fit a line to a mesh and return the metadata.

    >>> mesh = examples.download_human()
    >>> line, length, direction = pv.fit_line_to_points(
    ...     mesh.points, return_meta=True
    ... )

    Show the length of the line.

    >>> length
    167.6145

    Plot the line as an arrow to show its direction.

    >>> arrow = pv.Arrow(
    ...     start=line.points[0],
    ...     direction=direction,
    ...     scale=length,
    ...     tip_length=0.2,
    ...     tip_radius=0.04,
    ...     shaft_radius=0.01,
    ... )

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(mesh, opacity=0.5)
    >>> _ = pl.add_mesh(arrow, color='red')
    >>> pl.show()

    Set ``init_direction`` to the positive z-axis to flip the line's direction.

    >>> mesh = examples.download_human()
    >>> line, length, direction = pv.fit_line_to_points(
    ...     mesh.points, init_direction='z', return_meta=True
    ... )

    Plot the results again with an arrow.

    >>> arrow = pv.Arrow(
    ...     start=line.points[0],
    ...     direction=direction,
    ...     scale=length,
    ...     tip_length=0.2,
    ...     tip_radius=0.04,
    ...     shaft_radius=0.01,
    ... )

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(mesh, opacity=0.5)
    >>> _ = pl.add_mesh(arrow, color='red')
    >>> pl.show()

    """
    # Align points to the xyz-axes
    aligned, matrix = pyvista.PolyData(points).align_xyz(
        axis_0_direction=init_direction, return_matrix=True
    )

    # Fit line to xyz-aligned mesh
    point_a = (aligned.bounds.x_min, 0, 0)
    point_b = (aligned.bounds.x_max, 0, 0)
    line_mesh = pyvista.LineSource(point_a, point_b, resolution=resolution).output

    # Transform line back to input points positioning
    inverse_matrix = pyvista.Transform(matrix).inverse_matrix
    line_mesh.transform(inverse_matrix, inplace=True)

    if return_meta:
        return line_mesh, line_mesh.length, matrix[0, :3]
    return line_mesh


def make_tri_mesh(points: NumpyArray[float], faces: NumpyArray[int]) -> PolyData:
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
    >>> import pyvista as pv
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
    >>> tri_mesh = pv.make_tri_mesh(points, faces)
    >>> tri_mesh.plot(show_edges=True, line_width=5)

    """
    if points.shape[1] != 3:
        msg = 'Points array should have shape (N, 3).'
        raise ValueError(msg)
    if faces.ndim != 2 or faces.shape[1] != 3:
        msg = 'Face array should have shape (M, 3).'
        raise ValueError(msg)
    cells = np.empty((faces.shape[0], 4), dtype=faces.dtype)
    cells[:, 0] = 3
    cells[:, 1:] = faces
    return pyvista.PolyData(points, cells)


def vector_poly_data(
    orig: VectorLike[float] | MatrixLike[float], vec: VectorLike[float] | MatrixLike[float]
) -> PolyData:
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

    >>> import pyvista as pv
    >>> import numpy as np
    >>> x, y = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
    >>> points = np.vstack((x.ravel(), y.ravel(), np.zeros(x.size))).T
    >>> u = x / np.sqrt(x**2 + y**2)
    >>> v = y / np.sqrt(x**2 + y**2)
    >>> vectors = np.vstack((u.ravel() ** 3, v.ravel() ** 3, np.zeros(u.size))).T
    >>> pdata = pv.vector_poly_data(points, vectors)
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
        msg = 'orig array must be 3D'
        raise ValueError(msg)

    if vec.ndim != 2:
        vec = vec.reshape((-1, 3))
    elif vec.shape[1] != 3:
        msg = 'vec array must be 3D'
        raise ValueError(msg)

    # Create vtk points and cells objects
    vpts = _vtk.vtkPoints()
    vpts.SetData(_vtk.numpy_to_vtk(np.ascontiguousarray(orig), deep=True))

    npts = orig.shape[0]
    vcells = pyvista.core.cell.CellArray.from_regular_cells(
        np.arange(npts, dtype=pyvista.ID_TYPE).reshape((npts, 1)),
    )

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


@overload
def principal_axes(points: MatrixLike[float]) -> NumpyArray[float]: ...
@overload
def principal_axes(
    points: MatrixLike[float],
    *,
    return_std: Literal[True] = True,
) -> tuple[NumpyArray[float], NumpyArray[float]]: ...
@overload
def principal_axes(
    points: MatrixLike[float],
    *,
    return_std: Literal[False] = False,
) -> NumpyArray[float]: ...
@overload
def principal_axes(
    points: MatrixLike[float], *, return_std: bool = ...
) -> NumpyArray[float] | tuple[NumpyArray[float], NumpyArray[float]]: ...
def principal_axes(
    points: MatrixLike[float], *, return_std: bool = False
) -> NumpyArray[float] | tuple[NumpyArray[float], NumpyArray[float]]:
    """Compute the principal axes of a set of points.

    Principal axes are orthonormal vectors that best fit a set of points. The axes
    are also known as the principal components in Principal Component Analysis (PCA),
    or the right singular vectors from the Singular Value Decomposition (SVD).

    The axes are computed as the eigenvectors of the covariance matrix from the
    mean-centered points, and are processed to ensure that they form a right-handed
    coordinate frame.

    The axes explain the total variance of the points. The first axis explains the
    largest percentage of variance, followed by the second axis, followed again by
    the third axis which explains the smallest percentage of variance.

    The axes may be used to build an oriented bounding box or to align the points to
    another set of axes (e.g. the world XYZ axes).

    .. note::
        The computed axes are not unique, and the sign of each axis direction can be
        arbitrarily changed.

    .. note::
        This implementation creates a temporary array of the same size as the input
        array, and is therefore not optimal in terms of its memory requirements.
        A more memory-efficient computation may be supported in a future release.

    .. versionadded:: 0.45.0

    See Also
    --------
    fit_plane_to_points
        Fit a plane to points using the first two principal axes.

    pyvista.DataSetFilters.align_xyz
        Filter which aligns principal axes to the x-y-z axes.

    Parameters
    ----------
    points : MatrixLike[float]
        Nx3 array of points.

    return_std : bool, default: False
        If ``True``, also returns the standard deviation of the points along each axis.
        Standard deviation is computed as the square root of the eigenvalues of the
        mean-centered covariance matrix.

    Returns
    -------
    numpy.ndarray
        3x3 orthonormal array with the principal axes as row vectors.

    numpy.ndarray
        Three-item array of the standard deviations along each axis.

    Examples
    --------
    >>> import pyvista as pv
    >>> import numpy as np
    >>> rng = np.random.default_rng(seed=0)  # only seeding for the example

    Create a mesh with points that have the largest variation in ``X``,
    followed by ``Y``, then ``Z``.

    >>> radii = np.array((6, 3, 1))  # x-y-z radii
    >>> mesh = pv.ParametricEllipsoid(
    ...     xradius=radii[0], yradius=radii[1], zradius=radii[2]
    ... )

    Plot the mesh and highlight its points in black.

    >>> p = pv.Plotter()
    >>> _ = p.add_mesh(mesh)
    >>> _ = p.add_points(mesh, color='black')
    >>> _ = p.show_grid()
    >>> p.show()

    Compute its principal axes and return the standard deviations.

    >>> axes, std = pv.principal_axes(mesh.points, return_std=True)
    >>> axes  # doctest:+SKIP
    pyvista_ndarray([[-1.,  0.,  0.],
                     [ 0.,  1.,  0.],
                     [ 0.,  0., -1.]], dtype=float32)

    Note that the principal axes have ones along the diagonal and zeros
    in the off-diagonal. This indicates that the first principal axis is
    aligned with the x-axis, the second with the y-axis, and third with
    the z-axis. This is expected, since the mesh is already axis-aligned.

    However, since the signs of the principal axes are arbitrary, the
    first and third axes in this case have a negative direction.

    Show the standard deviation along each axis.

    >>> std  # doctest:+SKIP
    array([3.0149 , 1.5074 , 0.7035], dtype=float32)

    Compare this to using :meth:`numpy.std` for the computation.

    >>> np.std(mesh.points, axis=0)
    pyvista_ndarray([3.0149572, 1.5074761, 0.7035699], dtype=float32)

    Since the points are axis-aligned, the two results agree in this case. In general,
    however, these two methods differ in that :meth:`numpy.std` with `axis=0` computes
    the standard deviation along the `x-y-z` axes, whereas the standard deviation
    returned by :meth:`principal_axes` is computed along the principal axes.

    Convert the values to proportions for analysis.

    >>> std / sum(std)  # doctest:+SKIP
    array([0.5769149 , 0.28845742, 0.1346276 ], dtype=float32)

    From this result, we can determine that the axes explain approximately
    58%, 29%, and 13% of the total variance in the points, respectively.

    Let's compare this to the proportions of the known radii of the ellipsoid.

    >>> radii / sum(radii)
    array([0.6, 0.3, 0.1])

    Note how the two ratios are similar, but do not match exactly. This is
    because the points of the ellipsoid are prolate and are denser near the
    poles. If the points were normally distributed, however, the proportions
    would match exactly.

    Create an array of normally distributed points scaled along the x-y-z axes.
    Use the same scaling as the radii of the ellipsoid from the previous example.

    >>> normal_points = rng.normal(size=(1000, 3))
    >>> scaled_points = normal_points * radii
    >>> axes, std = pv.principal_axes(scaled_points, return_std=True)
    >>> axes
    array([[-0.99997578,  0.00682346,  0.00136972],
           [ 0.00681368,  0.99995213, -0.00702282],
           [-0.00141757, -0.00701331, -0.9999744 ]])

    Once again, the axes have ones along the diagonal as expected since the
    points are already axis-aligned. Now let's examine the standard deviation
    and compare the relative proportions.

    >>> std
    array([5.94466738, 2.89590334, 1.02103169])

    >>> std / sum(std)
    array([0.60280948, 0.29365444, 0.10353608])

    >>> radii / sum(radii)
    array([0.6, 0.3, 0.1])

    Since the points are normally distributed, the relative proportion of
    the standard deviation matches the scaling of the axes almost perfectly.

    """
    points = _validation.validate_arrayNx3(points)

    points_centered = points - np.mean(points, axis=0)
    eig_vals, eig_vectors = np.linalg.eigh(points_centered.T @ points_centered)
    axes = eig_vectors.T[::-1]  # columns, ascending order -> rows, descending order

    # Ensure axes form a right-handed coordinate frame
    if np.linalg.det(axes) < 0:
        axes[2] *= -1

    if return_std:
        # Compute standard deviation and swap order from ascending -> descending
        std = np.sqrt(np.abs(eig_vals) / len(points))[::-1]
        return axes, std
    return axes
