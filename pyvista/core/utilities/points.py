"""Points related utilities."""

import random
import warnings

import numpy as np

import pyvista
from pyvista.core import _vtk_core as _vtk
from pyvista.core.utilities.arrays import _coerce_pointslike_arg
from pyvista.core.utilities.geometric_objects import NORMALS
from pyvista.core.utilities.misc import check_valid_vector
from pyvista.core.utilities.transformations import (
    apply_transformation_to_points,
    axes_rotation_matrix,
)


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
    >>> import pyvista as pv
    >>> import numpy as np
    >>> points = np.random.default_rng().random((10, 3))
    >>> vpoints = pv.vtk_points(points)
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
                '``force_float=False``.',
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
            f'Shape is {points.shape} and should be (X, 3)',
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

    >>> import pyvista as pv
    >>> import numpy as np
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])
    >>> lines = pv.line_segments_from_points(points)
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


def principal_axes_transform(points, transformed_center="origin", return_inverse=False, **kwargs):
    """Compute the principal axes transform.

    The principal axes transform uses the :func:`~pyvista.principal_axes_vectors`
    to compute a 4x4 transformation matrix which aligns the principal axes
    of the points to the XYZ axes. The transformed center can be controlled
    to create origin-centered points (with ``transformed_center="origin"``)
    or to align the principal axes with the XYZ axes but keep the points
    centered at their initial center (with ``transformed_center="centroid"``).

    The transform applies the following transformations in sequence:
        * translation from the centroid of ``points`` to the origin
        * rotation defined by the principal axes`
        * translation from the origin to ``transformed_center``

    See :func:`~pyvista.principal_axes_vectors` for more information and
    for additional keyword arguments.

    .. versionadded:: 0.43.0

    Notes
    -----
    If the transform cannot be computed, the identity matrix is returned.

    See Also
    --------
    :func:`~pyvista.axes_rotation`
        Apply a rotation by axes vectors.

    Parameters
    ----------
    points : array_like[float]
        Points array. Accepts a single point or several points as a
        Nx3 array.

    transformed_center : sequence[float] | str, default: "origin"
        Desired location of the points center after the transformation
        is applied. May be a sequence of three values or one of:
        * ``"origin"`` : Alias for ``(0, 0, 0)``
        * ``"centroid"`` : Alias for ``np.mean(points, axis=0)``
        Has no effect if ``return_transforms`` is ``False``.

    return_inverse : bool, False
        If ``True``, the inverse of the transform is also returned.

    **kwargs : dict, optional
        Keyword arguments passed to :func:`~pyvista.principal_axes_vectors`.

    Returns
    -------
    numpy.ndarray
        4x4 transformation matrix which aligns the points to the XYZ axes
        at the origin.

    numpy.ndarray
        4x4 inverse transformation matrix if ``return_inverse`` is ``True``.
    """
    # Examples
    # --------
    # Compute the principal axes transform for a mesh.
    # >>> import pyvista as pv
    # >>> from pyvista import examples
    # >>> mesh = examples.download_face()
    # >>> mesh.points *= 5  # scale mesh for visualization
    # >>> matrix = pv.principal_axes_transform(mesh.points)
    # >>> matrix
    # array([[-5.79430342e-01, -3.02252942e-04, -8.15021694e-01,
    #         -8.62259164e-02],
    #        [ 6.74928480e-04,  9.99999404e-01, -8.50685057e-04,
    #          6.39482565e-01],
    #        [ 8.15021455e-01, -1.04299409e-03, -5.79429805e-01,
    #         -4.70775854e-01],
    #        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    #          1.00000000e+00]])
    #
    # Apply the transformation and compare the result to the input. Notice
    # that the transformed mesh is centered at the origin and aligned with
    # the XYZ axes.
    # >>> mesh_transformed = mesh.transform(matrix, inplace=False)
    # >>> def plot_meshes():
    # ...     p = pv.Plotter()
    # ...     _ = p.add_mesh(
    # ...         mesh, label='Input', color='lightblue', show_edges=True
    # ...     )
    # ...     _ = p.add_mesh(
    # ...         mesh_transformed,
    # ...         label='Transformed',
    # ...         color='goldenrod',
    # ...         show_edges=True,
    # ...     )
    # ...     _ = p.add_axes_at_origin()
    # ...     _ = p.add_legend()
    # ...     _ = p.camera.zoom(2)
    # ...     p.show()
    # ...
    # >>> plot_meshes()
    #
    # It is possible to adjust the transform so that the sign of one
    # or more principal axes have a meaningful interpretation. For example,
    # the face of the original mesh is generally "looking" towards the
    # ``+X`` direction, and we can see from the transformed mesh that this
    # direction correlates with the third principal axis (i.e. the z-axis
    # of the transformed mesh). Therefore, if we want the face of the
    # transformed mesh to be "looking" down instead of up, we can specify
    # an approximate direction vector for the third principal axis as the
    # ``-X`` direction with ``axis_2_direction='-x'``.
    # >>> matrix = pv.principal_axes_transform(
    # ...     mesh.points, axis_2_direction='-x'
    # ... )
    # >>> mesh_transformed = mesh.transform(matrix, inplace=False)
    # >>> plot_meshes()
    #
    # The face is now looking down in the ``-Z`` direction as desired.
    # However, the top of the face has now flipped in the new transform
    # relative to the previous one, and is pointing in the ``-Y``
    # direction. Similar to above, we can adjust the transform so that the
    # direction of the second principal axis (which corresponds to the
    # y-axis) is such that the top of the face points in the ``+Y``
    # direction. Since the top of the face in the original mesh
    # points approximately in the ``+Y`` direction, we set
    # ``axis_1_direction='y'.
    # >>> matrix = pv.principal_axes_transform(
    # ...     mesh.points, axis_1_direction='y', axis_2_direction='-x'
    # ... )
    # >>> mesh_transformed = mesh.transform(matrix, inplace=False)
    # >>> plot_meshes()

    axes, transform, inverse = principal_axes_vectors(
        points,
        transformed_center=transformed_center,
        return_transforms=True,
        **kwargs,
    )
    if return_inverse:
        return transform, inverse
    return transform


def principal_axes_vectors(
    points,
    axis_0_direction=None,
    axis_1_direction=None,
    axis_2_direction=None,
    swap_equal_axes=None,
    project_xyz=False,
    return_transforms=False,
    transformed_center="origin",
    sample_count=10000,
    precision=np.double,
):
    """Compute the principal axes vectors of a set of points.

    Principal axes are orthonormal vectors that best fit a set of
    points. The axes are also known as the principal components of the
    points in Principal Component Analysis (PCA). For numerical
    stability, the axes are computed as the right singular vectors from
    the Singular Value Decomposition (SVD) of the mean-centered points.

    The axes explain the total variance of the points. The first axis
    explains the largest percentage of variance, followed by the second
    axis, followed again by the third axis which explains the smallest
    percentage of variance. The axes can be used to build a
    transformation matrix to align a mesh to the XYZ axes.

    The computed axes are not unique, and the sign of each axis direction
    can be arbitrarily changed (as long the axes define a right-handed
    coordinate frame). Similarly, axes which explain variance equally can
    be arbitrarily reordered. As such, approximate direction vectors may
    optionally be specified to control the axis directions, and equally-
    weighted axes may optionally be reordered. This can be useful for
    cases where axis directions for a local coordinate frame have a
    clear physical meaning.

    .. warning::

        This function has a large memory storage requirement of``O(N^2)``
        which can be an issue for large point arrays. To reduce this
        requirement, the points are randomly sampled by default. Sampling
        can be disabled, however, if desired. For reference, the
        approximate memory requirement for double precision (default) is:

        .. list-table::
            :widths: 25 25
            :header-rows: 1

            * - Number of points
              - Minimum memory required
            * - 10,000
              - < 1 GiB
            * - 40,000
              - 12 GiB
            * - 100,000
              - 75 GiB

        If using single precision, the requirement is halved.

    .. versionadded:: 0.43.0

    See Also
    --------
    :attr:`~pyvista.DataSet.principal_axes`
        Compute the principal axes of a mesh.
    :func:`~pyvista.principal_axes_transform`
        Compute the principal axes transform.
    :func:`~pyvista.fit_plane_to_points`
        Use the principal axes to fit a plane.
    :func:`~pyvista.axes_rotation`
        Apply a rotation by axes vectors.

    Parameters
    ----------
    points : array_like[float]
        Points array. Accepts a single point or several points as a
        Nx3 array.

    axis_0_direction : sequence[float] | str, optional
        Approximate direction vector of the first axis. If set, the
        sign of the first principal axis will be flipped such that it
        best aligns with this vector. Has no effect if this vector is
        perpendicular to the first principal axis. Can be a sequence
        of three elements specifying the ``(x, y, z)`` direction or a
        string specifying a conventional direction (e.g. ``'x'`` for
        ``(1, 0, 0)`` or ``'-x'`` for ``(-1, 0, 0)``, etc.).

    axis_1_direction : sequence[float] | str, optional
        Approximate direction vector of the second axis. If set, the
        sign of the second principal axis will be flipped such that it
        best aligns with this vector. Has no effect if this vector is
        perpendicular to the second principal axis.

    axis_2_direction : sequence[float] | str, optional
        Approximate direction vector of the third axis. If set, the
        sign of the third principal axis will be flipped such that it
        best aligns with this vector. Has no effect if this vector is
        perpendicular to the third principal axis. Has no effect if
        ``axis_0_direction`` and ``axis_1_direction`` are set.

    swap_equal_axes : bool, optional
        If ``True``, principal axes which explain variance equally
        (e.g. when points have reflection symmetry) may be swapped based
        on their relative alignment (projection) onto each of the X, Y,
        and Z axes. Swapping is performed as follows: first, principal
        axes are labelled according to the axis (``X``, ``Y``, or ``Z``)
        which they are most closely aligned with; then, the principal
        axes are sorted by their label. For example, if principal axis 1
        maps to ``Z`` and principal axis 2 maps to ``X``, the axes are
        swapped so that the order is ``X-Z`` instead of ``Z-X``.

        .. note::
            Swapping may cause the sign of a principal axis to be
            flipped to ensure the axes form a right-handed coordinate
            frame. Use the ``axis_#_direction`` parameters to control
            the axes signs if needed.

    project_xyz : bool, False
        If ``True``, the following default values are set:
        * ``axis_0_direction='x'``
        * ``axis_1_direction='y'``
        * ``axis_2_direction='z'``
        * ``reorder_equal_axes=True``
        Default values are only applied if the respective parameter has
        not been set. As such, these values can be overridden.
        This parameter can be used, for example, to project the
        principal axes of X-Y planar data onto the positive XYZ axes to
        ensure the principal axes all have a positive direction.

        .. note::

            The ``project_xyz`` and ``axis_#_direction`` parameters only control
            the signs of the individual principal axes and do not apply a
            transformation to reorient the axes as a set.

    return_transforms : bool, False
        If ``True``, two 4x4 transformation matrices are also returned.
        The first matrix applies the following transformations in sequence:
        * translation from the centroid of ``points`` to the origin
        * rotation defined by the principal axes`
        * translation from the origin to ``transformed_center``
        The second transform is the inverse of the first.

    transformed_center : sequence[float] | str, default: "origin"
        Desired location of the points center after the transformation
        is applied. May be a sequence of three values or one of:
        * ``"origin"``: Alias for ``(0, 0, 0)``
        * ``"centroid"``: Alias for ``np.mean(points, axis=0)``
        Has no effect if ``return_transforms`` is ``False``.

    sample_count : int, default: 10000
        If set, the input points may be randomly sampled to reduce the
        time and memory required for the computation. Has no effect if
        the number of input points is less than this value. Can be set
        to ``None`` to disable sampling.

    precision : str, default: np.double
        Control the precision of the SVD computation. Use ``np.double``
        for better precision and ``np.single`` to reduce the time and
        memory requirements for the computation. The returned values
        will also have this type.

    Raises
    ------
    MemoryError
        If there is insufficient memory to compute the principal axes.

    Returns
    -------
    numpy.ndarray
        3x3 array with the principal axes as row vectors.

    numpy.ndarray
        4x4 transformation matrix if ``return_transforms=True``.

    numpy.ndarray
        4x4 inverse transformation matrix if ``return_transforms=True``.
    """
    # Examples
    # --------
    # Create a mesh with points that have the largest variation in ``X``,
    # followed by ``Y``, then ``Z``.
    # >>> import pyvista as pv
    # >>> mesh = pv.ParametricEllipsoid(xradius=10, yradius=5, zradius=1)
    # >>> p = pv.Plotter()
    # >>> _ = p.add_mesh(mesh)
    # >>> _ = p.show_grid()
    # >>> p.show()
    #
    # Compute its principal axes
    # >>> principal_axes = pv.principal_axes_vectors(mesh.points)
    #
    # Note that the principal axes have ones along the diagonal and zeros
    # in the off diagonals. This indicates that the first principal axis is
    # aligned with the x-axis, the second with the y-axis, and third with
    # the z-axis, as expected, since the mesh is already axis-aligned.
    # >>> principal_axes
    # array([[-1.0000000e+00,  5.7725526e-11, -9.1508944e-19],
    #        [ 5.7725526e-11,  1.0000000e+00, -3.8939370e-18],
    #        [ 9.1508944e-19, -3.8939370e-18, -1.0000000e+00]], dtype=float32)
    #
    # However, since the signs of the principal axes are arbitrary, the
    # first and third axes in this case have a negative direction. To
    # project the positive XYZ axes directions onto the principal axes,
    # use ``project_xyz=True``.
    # >>> principal_axes = pv.principal_axes_vectors(
    # ...     mesh.points, project_xyz=True
    # ... )
    # >>> principal_axes
    # array([[ 1.0000000e+00, -5.7725526e-11,  9.1508944e-19],
    #        [ 5.7725526e-11,  1.0000000e+00, -3.8939370e-18],
    #        [-9.1508944e-19,  3.8939370e-18,  1.0000000e+00]], dtype=float32)
    #
    # The signs of the principal axes can also be controlled by specifying
    # approximate axis directions.
    # >>> principal_axes = pv.principal_axes_vectors(
    # ...     mesh.points, axis_0_direction='-x', axis_1_direction='-y'
    # ... )
    # >>> principal_axes
    # array([[-1.0000000e+00,  5.7725526e-11, -9.1508944e-19],
    #        [-5.7725526e-11, -1.0000000e+00,  3.8939370e-18],
    #        [-9.1508944e-19,  3.8939370e-18,  1.0000000e+00]], dtype=float32)
    #
    # Note, however, that since the ``project_xyz`` and ``axis_#_direction``
    # parameters only control the signs of the axes, they cannot be used
    # to reorient them. For example, the following code does not orient
    # the first principal axes to point in a specified direction.
    # >>> principal_axes = pv.principal_axes_vectors(
    # ...     mesh.points, axis_0_direction=[4, 5, 6]
    # ... )
    # >>> principal_axes
    # array([[ 1.0000000e+00, -5.7725526e-11,  9.1508944e-19],
    #        [ 5.7725526e-11,  1.0000000e+00, -3.8939370e-18],
    #        [-9.1508944e-19,  3.8939370e-18,  1.0000000e+00]], dtype=float32)
    from numpy.core._exceptions import _ArrayMemoryError

    def _default_values():
        default_axes = np.eye(3)
        default_transform = np.eye(4)
        if return_transforms:
            return default_axes, default_transform, default_transform
        return default_axes

    def _validate_vector(vector, name):
        if vector is not None:
            if isinstance(vector, str):
                vector = vector.lower()
                valid_strings = list(NORMALS.keys())
                if vector not in valid_strings:
                    raise ValueError(
                        f"Vector string for {name} must be one of {valid_strings}, got {vector} instead.",
                    )
                vector = NORMALS[vector]
            check_valid_vector(vector, name=name)
        return vector

    axis_0_direction = _validate_vector(axis_0_direction, name='axis_0_direction')
    axis_1_direction = _validate_vector(axis_1_direction, name='axis_1_direction')
    axis_2_direction = _validate_vector(axis_2_direction, name='axis_2_direction')

    # Compare all direction vectors with each other
    directions = [axis_0_direction, axis_1_direction, axis_2_direction]
    for i, vec1 in enumerate(directions):
        for vec2 in directions[i + 1 : :]:
            if vec1 is not None and vec2 is not None and np.allclose(vec1, vec2):
                raise ValueError("Direction vectors must be distinct.")

    if project_xyz:
        # Set values only if not yet set
        axis_0_direction = [1, 0, 0] if not axis_0_direction else None
        axis_1_direction = [0, 1, 0] if not axis_1_direction else None
        axis_2_direction = [0, 0, 1] if not axis_2_direction else None
        swap_equal_axes = True if swap_equal_axes is None else None

    # Validate points
    data, _ = _coerce_pointslike_arg(points, copy=True)
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)
    if len(data) == 0:
        return _default_values()

    # Validate precision
    precision = np.array(np.empty(0), dtype=precision).dtype.type
    if precision not in [np.single, np.double]:
        raise TypeError(f"Precision must be np.single or np.double, got {precision} instead.")

    # Center data
    centroid = np.mean(data, axis=0)
    data -= centroid

    # Validate transformed center
    if return_transforms:
        if isinstance(transformed_center, str):
            if transformed_center == "origin":
                transformed_center = (0, 0, 0)
            elif transformed_center == "centroid":
                transformed_center = centroid
            else:
                raise ValueError(
                    f"Expected one of {['origin', 'centroid']}, got {transformed_center} instead.",
                )
        check_valid_vector(transformed_center, name="transform_location")

    # Sample data
    if sample_count is not None:
        data = random_sample_points(data, count=sample_count, pass_through=True, seed=42)

    # Compute principal axes
    try:
        if precision is np.double:
            # Use SVD as it's numerically more stable than using PCA (below)
            _, axes_values, axes_vectors = np.linalg.svd(data)  # row vectors, descending order

            ## Equivalently (up to a difference in axis sign, non-uniqueness of
            ## vectors, and numerical error), the axes may also be computed using
            ## PCA, i.e. using covariance and eigenvalue decomposition.
            # covariance = np.cov(data, rowvar=False)
            # _, axes_vectors = np.linalg.eigh(covariance)  # column vectors, ascending order
            # axes_vectors = axes_vectors.T[::-1]  # row vectors, descending order

        elif precision is np.single:
            _, axes_values, axes_vectors = _svd_single(data)
        else:
            # This line should not be reachable
            raise NotImplementedError(f"Type {precision} precision is not valid.")

    except _ArrayMemoryError as e:
        msg = (
            f"Failed to allocate memory to compute the principal axes for "
            f"N={len(data)} points. Consider reducing the number of points"
            f"with `sample_count` or using some other method."
        )
        raise MemoryError(msg) from e

    if swap_equal_axes:
        # Note: Swapping may create a left-handed coordinate frame. This
        # is fixed later with a cross-product
        axes_vectors = _swap_axes(axes_vectors, axes_values)

    # Normalize to unit-length, flip directions, and ensure vectors form
    # a right-handed coordinate system
    i_vector = axes_vectors[0] / np.linalg.norm(axes_vectors[0])
    if axis_0_direction is not None:
        sign = np.sign(np.dot(i_vector, axis_0_direction))
        if sign != 0:  # Only change sign if not perpendicular
            i_vector *= sign

    j_vector = axes_vectors[1] / np.linalg.norm(axes_vectors[1])
    if axis_1_direction is not None:
        sign = np.sign(np.dot(j_vector, axis_1_direction))
        if sign != 0:
            j_vector *= sign

    k_vector = np.cross(i_vector, j_vector)
    if axis_2_direction is not None:
        sign = np.sign(np.dot(k_vector, axis_2_direction))
        cannot_be_changed = axis_0_direction is not None and axis_1_direction is not None
        if sign == 0 or cannot_be_changed:
            pass
        else:
            # Need to modify two vectors to keep system as right-handed
            if axis_1_direction is not None:
                # Do not modify j vector
                i_vector *= sign
                k_vector *= sign
            else:
                # Do not modify i vector
                j_vector *= sign
                k_vector *= sign

    axes_vectors = np.vstack((i_vector, j_vector, k_vector))

    if return_transforms:
        transform, inverse = axes_rotation_matrix(
            axes_vectors,
            point_initial=centroid,
            point_final=transformed_center,
            return_inverse=True,
        )
        return axes_vectors, transform, inverse
    return axes_vectors


def _swap_axes(vectors, values):
    """Swap axes vectors based on their respective values.

    This function is intended to be used by :func:`principal_axes_vectors`
    and is only exposed as a module-level function for testing purposes.

    """

    def _swap(axis_a, axis_b):
        axis_order = np.argmax(np.abs(vectors), axis=1)
        if axis_order[axis_a] > axis_order[axis_b]:
            vectors[[axis_a, axis_b]] = vectors[[axis_b, axis_a]]

    if np.isclose(values[0], values[1]) and np.isclose(values[1], values[2]):
        # Sort all axes by largest 'x' component
        vectors = vectors[np.argsort(np.abs(vectors)[:, 0])[::-1]]
        _swap(1, 2)
    else:
        if np.isclose(values[0], values[1]):
            _swap(0, 1)
        elif np.isclose(values[1], values[2]):
            _swap(1, 2)
    return vectors


def _svd_single(a):
    """Compute single-precision SVD.

    The function is a customized recreating of `numpy.linalg.svd` with
    modifications for single floats. Stock SVD always computes with
    double floats, which is not always practical.

    In contrast to numpy's general SVD, this version:
        - assumes that we have an array of real numbers
        - assumes that M > N
        - always computes uv (compute_uv==True)
        - always returns full matrices (full_matrices==True)

    Warning: No checks are done to validate the assumptions above.

    """
    # Import locally instead of at module level to avoid cluttering
    # the namespace with internal NumPy functions
    from numpy.linalg.linalg import (
        _assert_stacked_2d,
        _makearray,
        _raise_linalgerror_svd_nonconvergence,
        _umath_linalg,
        get_linalg_error_extobj,
    )

    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    result_t = np.single

    extobj = get_linalg_error_extobj(_raise_linalgerror_svd_nonconvergence)

    gufunc = _umath_linalg.svd_n_f  # compute full uv with M < N

    # default signature is 'd->ddd' for double floats
    # swap 'd' with 'f' for single floats
    signature = 'f->fff'
    u, s, vh = gufunc(a, signature=signature, extobj=extobj)
    u = u.astype(result_t, copy=False)
    s = s.astype(result_t, copy=False)
    vh = vh.astype(result_t, copy=False)
    return wrap(u), s, wrap(vh)


def fit_plane_to_points(
    points,
    return_meta=False,
    i_resolution=10,
    j_resolution=10,
    normal_direction=None,
):
    """Fit a plane to a set of points.

    The plane is fitted to the points using :func:~pyvista.principal_axes_vectors,
    and is automatically sized to fit the extent of the points.

    Optionally, the sign of the normal can be controlled by specifying
    an approximate normal direction. This can be useful, for example,
    in cases where the normal direction has a clear physical meaning.

    .. versionchanged:: 0.42.0
        The center of the plane (returned if ``return_meta=True``) is now
        computed as the center of the generated plane mesh. In previous
        versions, the center of the input points was returned.

    .. versionchanged:: 0.43.0
        If ``points`` is type ``numpy.double``, the points of the generated
        plane will also be type ``numpy.double``

    See Also
    --------
    :func:`~pyvista.principal_axes_vectors`
        Compute the best fit axes to a set of points.

    Parameters
    ----------
    points : array_like[float]
        Size ``[N x 3]`` sequence of points to fit a plane through.

    return_meta : bool, default: False
        If ``True``, also returns the center and normal of the
        generated plane.

        .. note::
            The center of the generated plane mesh may not coincide with
            the center of the points.

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
        sign of the plane's normal will be flipped such that it best
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

    >>> import pyvista as pv
    >>> import numpy as np
    >>>
    >>> # Create point cloud
    >>> rng = np.random.default_rng(3)
    >>> cloud = rng.random((10, 3))
    >>> cloud[:, 2] *= 0.1
    >>>
    >>> # Fit plane
    >>> plane, center, normal = pv.fit_plane_to_points(
    ...     cloud, return_meta=True
    ... )
    >>>
    >>> # Plot the fitted plane
    >>> pl = pv.Plotter()
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

    The plane's normal in this example points in the negative z direction
    >>> normal
    array([-0.02878024, -0.01055844, -0.99953   ])

    To control the sign of the normal, specify the approximate normal
    direction when fitting the plane
    >>> plane, center, normal = pv.fit_plane_to_points(
    ...     cloud, return_meta=True, normal_direction='z'
    ... )
    >>> normal
    array([0.02878024, 0.01055844, 0.99953   ])

    Fit a plane to a mesh.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>>
    >>> # Create mesh
    >>> mesh = examples.download_shark()
    >>>
    >>> # Fit plane. Set the plane resolution to one to only extract
    >>> # the plane's corner points
    >>> plane = pv.fit_plane_to_points(
    ...     mesh.points, i_resolution=1, j_resolution=1
    ... )
    >>>
    >>> # Plot the fitted plane
    >>> pl = pv.Plotter()
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
    # Get best-fit axes and transforms
    axes_vectors, transform, inverse = principal_axes_vectors(
        points,
        axis_2_direction=normal_direction,
        return_transforms=True,
    )
    dtype = axes_vectors.dtype

    # Align points to XYZ axes
    points_aligned = apply_transformation_to_points(transform, points)

    # Compute plane size and center from XY bounds
    xmin, xmax = np.min(points_aligned[:, 0]), np.max(points_aligned[:, 0])
    ymin, ymax = np.min(points_aligned[:, 1]), np.max(points_aligned[:, 1])
    i_size = xmax - xmin
    j_size = ymax - ymin
    center_aligned = np.array([(xmax + xmin) / 2, (ymax + ymin) / 2, 0])

    # Compute plane normal direction sign using axes as a rotation matrix
    normal_aligned = axes_vectors @ axes_vectors[2]
    sign = np.sign(normal_aligned[2])

    # Initialize plane aligned with XYZ axes
    # Set center and direction manually afterward to preserve precision
    # and set the correct orientation
    plane = pyvista.Plane(
        center=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1 * sign),
        i_size=i_size,
        j_size=j_size,
        i_resolution=i_resolution,
        j_resolution=j_resolution,
    )
    if dtype.type is np.double:
        plane.points_to_double()

    # Shift plane to its axis-aligned center
    plane.points += center_aligned

    # Transform plane to the points' original coordinate frame
    plane.transform(inverse)

    if return_meta:
        # Recompute center from the actual plane being returned.
        # This is done to remove any error from the transformations and
        # ensure the plane's geometry matches the meta variable
        center = np.mean(plane.points, axis=0)

        # Unlike with center, for the normal we return the vector computed
        # directly from the principal axes since any normals derived
        # from the actual plane will have errors from the edge points,
        # are limited to float32, and have a variable number of cell normals
        normal = axes_vectors[2]
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

    >>> import pyvista as pv
    >>> import numpy as np
    >>> x, y = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
    >>> points = np.vstack((x.ravel(), y.ravel(), np.zeros(x.size))).T
    >>> u = x / np.sqrt(x**2 + y**2)
    >>> v = y / np.sqrt(x**2 + y**2)
    >>> vectors = np.vstack(
    ...     (u.ravel() ** 3, v.ravel() ** 3, np.zeros(u.size))
    ... ).T
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
        raise ValueError('orig array must be 3D')

    if vec.ndim != 2:
        vec = vec.reshape((-1, 3))
    elif vec.shape[1] != 3:
        raise ValueError('vec array must be 3D')

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


def random_sample_points(points, count, seed=None, pass_through=False):
    """Randomly sample a set of points.

    Parameters
    ----------
    points : array_like[float]
        Nx3 points array to sample.

    count : int
        Number of points to return.

    seed : int
        Value to initialize random number generator.

    pass_through : bool, default: False
        If ``True``, the points are returned as-is without sampling
        when ``count`` is greater than the number of ``points``.

    Returns
    -------
    numpy.ndarray
        Array of sampled points.

    """
    if seed is not None:
        random.seed(seed)  # make sampling deterministic
    points, _ = _coerce_pointslike_arg(points)
    N = len(points)
    if count > N and pass_through:
        return points
    sample_idx = random.sample(range(N), count)
    return points[sample_idx]
