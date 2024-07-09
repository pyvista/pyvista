"""Module implementing point transformations and their matrices."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyvista.core import _validation

if TYPE_CHECKING:  # pragma: no cover
    import scipy

    from pyvista.core import _vtk_core as _vtk
    from pyvista.core._typing_core import MatrixLike
    from pyvista.core._typing_core import VectorLike


def axis_angle_rotation(axis, angle, point=None, deg=True):
    r"""Return a 4x4 matrix for rotation about any axis by given angle.

    Rotations around an axis that contains the origin can easily be
    computed using Rodrigues' rotation formula. The key quantity is
    the ``K`` cross product matrix for the unit vector ``n`` defining
    the axis of the rotation:

             /   0  -nz   ny \
        K =  |  nz    0  -nx |
             \ -ny   nx    0 /

    For a rotation angle ``phi`` around the vector ``n`` the rotation
    matrix is given by

        R = I + sin(phi) K  + (1 - cos(phi)) K^2

    where ``I`` is the 3-by-3 unit matrix and ``K^2`` denotes the matrix
    square of ``K``.

    If the rotation axis doesn't contain the origin, we have to first
    shift real space to transform the axis' ``p0`` reference point into
    the origin, then shift the points back after rotation:

        p' = R @ (p - p0) + p0 = R @ p + (p0 - R @ p0)

    This means that the rotation in general consists of a 3-by-3
    rotation matrix ``R``, and a translation given by
    ``b = p0 - R @ p0``. These can be encoded in a 4-by-4 transformation
    matrix by filling the 3-by-3 leading principal submatrix with ``R``,
    and filling the top 3 values in the last column with ``b``.

    Parameters
    ----------
    axis : sequence[float]
        The direction vector of the rotation axis. It need not be a
        unit vector, but it must not be a zero vector.

    angle : float
        Angle of rotation around the axis. The angle is defined as a
        counterclockwise rotation when facing the normal vector of the
        rotation axis. Passed either in degrees or radians depending on
        the value of ``deg``.

    point : sequence[float], optional
        The origin of the rotation (a reference point through which the
        rotation axis passes). By default the rotation axis contains the
        origin.

    deg : bool, default: True
        Whether the angle is specified in degrees. ``False`` implies
        radians.

    Returns
    -------
    numpy.ndarray
        The ``(4, 4)`` rotation matrix.

    Examples
    --------
    Generate a transformation matrix for rotation around a cube's body
    diagonal by 120 degrees.

    >>> import numpy as np
    >>> from pyvista import transformations
    >>> trans = transformations.axis_angle_rotation([1, 1, 1], 120)

    Check that the transformation cycles the cube's three corners.

    >>> corners = np.array(
    ...     [
    ...         [1, 0, 0],
    ...         [0, 1, 0],
    ...         [0, 0, 1],
    ...     ]
    ... )
    >>> rotated = transformations.apply_transformation_to_points(
    ...     trans, corners
    ... )
    >>> np.allclose(rotated, corners[[1, 2, 0], :])
    True

    """
    if deg:
        # convert to radians
        angle *= np.pi / 180

    # return early for no rotation; play it safe and check only exact equality
    if angle % (2 * np.pi) == 0:
        return np.eye(4)

    axis = np.asarray(axis, dtype='float64')
    if axis.shape != (3,):
        raise ValueError('Axis must be a 3-length array-like.')
    if point is not None:
        point = np.asarray(point)
        if point.shape != (3,):
            raise ValueError('Rotation center must be a 3-length array-like.')

    # check and normalize
    axis_norm = np.linalg.norm(axis)
    if np.isclose(axis_norm, 0):
        raise ValueError('Cannot rotate around zero vector axis.')
    if not np.isclose(axis_norm, 1):
        axis = axis / axis_norm

    # build Rodrigues' rotation matrix
    K = np.zeros((3, 3))
    K[[2, 0, 1], [1, 2, 0]] = axis
    K += -K.T

    # the cos and sin functions can introduce some numerical error
    # round the elements to exact values for special cases where we know
    # sin/cos should evaluate exactly to 0 or 1
    sin_angle = np.sin(angle)
    cos_angle = np.cos(angle)
    if angle % (np.pi / 2) == 0:
        cos_angle = round(cos_angle)
        sin_angle = round(sin_angle)

    R = np.eye(3) + sin_angle * K + (1 - cos_angle) * K @ K

    augmented = np.eye(4)
    augmented[:-1, :-1] = R

    if point is not None:
        # rotation of point p would be R @ (p - point) + point
        # which is R @ p + (point - R @ point)
        augmented[:-1, -1] = point - R @ point

    return augmented


def reflection(normal, point=None):
    """Return a 4x4 matrix for reflection across a normal about a point.

    Projection to a unit vector ``n`` can be computed using the dyadic
    product (or outer product) ``P`` of ``n`` with itself, which is a
    3-by-3 symmetric matrix.

    Reflection across a plane that contains the origin amounts to
    reversing the components of real space points that are perpendicular
    to the reflection plane. This gives us the transformation ``R``
    acting on a point ``p`` as

        p' = R @ p = p - 2 P @ p = (I - 2 P) @ p

    so the reflection's transformation matrix is the unit matrix minus
    twice the dyadic product ``P``.

    If additionally we want to compute a reflection to a plane that does
    not contain the origin, we can we can first shift every point in
    real space by ``-p0`` (if ``p0`` is a point that lies on the plane)

        p' = R @ (p - p0) + p0 = R @ p + (p0 - R @ p0)

    This means that the reflection in general consists of a 3-by-3
    reflection matrix ``R``, and a translation given by
    ``b = p0 - R @ p0``. These can be encoded in a 4-by-4 transformation
    matrix by filling the 3-by-3 leading principal submatrix with ``R``,
    and filling the top 3 values in the last column with ``b``.

    Parameters
    ----------
    normal : sequence[float]
        The normal vector of the reflection plane. It need not be a unit
        vector, but it must not be a zero vector.

    point : sequence[float], optional
        The origin of the reflection (a reference point through which
        the reflection plane passes). By default the reflection plane
        contains the origin.

    Returns
    -------
    ndarray
        A ``(4, 4)`` transformation matrix for reflecting points across the
        plane defined by the given normal and point.

    Examples
    --------
    Generate a transformation matrix for reflection over the XZ plane.

    >>> import numpy as np
    >>> from pyvista import transformations
    >>> trans = transformations.reflection([0, 1, 0])

    Check that the reflection transforms corners of a cube among one
    another.

    >>> verts = np.array(
    ...     [
    ...         [1, -1, 1],
    ...         [-1, -1, 1],
    ...         [-1, -1, -1],
    ...         [-1, -1, 1],
    ...         [1, 1, 1],
    ...         [-1, 1, 1],
    ...         [-1, 1, -1],
    ...         [-1, 1, 1],
    ...     ]
    ... )
    >>> mirrored = transformations.apply_transformation_to_points(
    ...     trans, verts
    ... )
    >>> np.allclose(mirrored, verts[[np.r_[4:8, 0:4]], :])
    True

    """
    normal = np.asarray(normal, dtype='float64')
    if normal.shape != (3,):
        raise ValueError('Normal must be a 3-length array-like.')
    if point is not None:
        point = np.asarray(point)
        if point.shape != (3,):
            raise ValueError('Plane reference point must be a 3-length array-like.')

    # check and normalize
    normal_norm = np.linalg.norm(normal)
    if np.isclose(normal_norm, 0):
        raise ValueError('Plane normal cannot be zero.')
    if not np.isclose(normal_norm, 1):
        normal = normal / normal_norm

    # build reflection matrix
    projection = np.outer(normal, normal)
    R = np.eye(3) - 2 * projection
    augmented = np.eye(4)
    augmented[:-1, :-1] = R

    if point is not None:
        # reflection of point p would be R @ (p - point) + point
        # which is R @ p + (point - R @ point)
        augmented[:-1, -1] = point - R @ point

    return augmented


def apply_transformation_to_points(transformation, points, inplace=False):
    """Apply a given transformation matrix (3x3 or 4x4) to a set of points.

    Parameters
    ----------
    transformation : np.ndarray
        Transformation matrix of shape (3, 3) or (4, 4).

    points : np.ndarray
        Array of points to be transformed of shape (N, 3).

    inplace : bool, default: False
        Updates points in-place while returning nothing.

    Returns
    -------
    numpy.ndarray
        Transformed points.

    Examples
    --------
    Scale a set of points in-place.

    >>> import numpy as np
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> points = examples.load_airplane().points
    >>> points_orig = points.copy()
    >>> scale_factor = 2
    >>> tf = scale_factor * np.eye(4)
    >>> tf[3, 3] = 1
    >>> pv.core.utilities.transformations.apply_transformation_to_points(
    ...     tf, points, inplace=True
    ... )
    >>> assert np.all(np.isclose(points, scale_factor * points_orig))

    """
    transformation_shape = transformation.shape
    if transformation_shape not in ((3, 3), (4, 4)):
        raise ValueError('`transformation` must be of shape (3, 3) or (4, 4).')

    if points.shape[1] != 3:
        raise ValueError('`points` must be of shape (N, 3).')

    if transformation_shape[0] == 4:
        # Divide by scale factor when homogeneous
        transformation /= transformation[3, 3]

        # Add the homogeneous coordinate
        # `points_2` is a copy of the data, not a view
        points_2 = np.empty((len(points), 4))
        points_2[:, :-1] = points
        points_2[:, -1] = 1
    else:
        points_2 = points

    # Paged matrix multiplication. For arrays with ndim > 2, matmul assumes
    # that the matrices to be multiplied lie in the last two dimensions.
    points_2 = (transformation[np.newaxis, :, :] @ points_2.T)[0, :3, :].T

    # If inplace, set the points
    if inplace:
        points[:] = points_2
        return None
    else:
        # otherwise return the new points
        return points_2


def rotation(
    rotation: MatrixLike[float] | _vtk.vtkMatrix3x3 | scipy.spatial.transform.Rotation,
    point: VectorLike[float] | None = None,
):
    """Return a 4x4 matrix for rotation about a point.

    The transformation is computed as follows:

    #. Translation to make ``point`` the origin of the rotation.
    #. Rotation specified by ``rotation``.
    #. Translation by ``point`` to restore initial positioning.

    Parameters
    ----------
    rotation : NumpyArray[float] | vtkMatrix3x3 | scipy.spatial.transform.Rotation
        3x3 rotation matrix or a SciPy ``Rotation`` object.

    point : Vector, default: (0.0, 0.0, 0.0)
        Point to rotate about. Defaults to origin.

    Returns
    -------
    numpy.ndarray
        4x4 rotation matrix.

    Examples
    --------
    Apply a rotation about a point with a 3x3 rotation matrix.

    >>> import numpy as np
    >>> from pyvista import transformations
    >>> point = [1, 1, 1]
    >>> rotation = np.array(
    ...     [
    ...         [1.0, 0.0, 0.0],
    ...         [0.0, 0.70710678, -0.70710678],
    ...         [0.0, 0.70710678, 0.70710678],
    ...     ]
    ... )
    >>> trans = transformations.rotation(rotation, point=point)
    >>> trans
    array([[ 1.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.70710678, -0.70710678,  1.        ],
           [ 0.        ,  0.70710678,  0.70710678, -0.41421356],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])

    Compute the inverse transformation using rotation's transpose.

    >>> trans_inv = transformations.rotation(rotation.T, point=point)
    >>> trans_inv
    array([[ 1.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.70710678,  0.70710678, -0.41421356],
           [ 0.        , -0.70710678,  0.70710678,  1.        ],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])

    Verify the transform and its inverse combine to produce the identity matrix.

    >>> np.allclose(trans_inv @ trans, np.eye(4))
    True
    """
    valid_rotation = _validation.validate_transform3x3(rotation, name='rotation')
    rotate = np.eye(4)
    rotate[:3, :3] = valid_rotation
    if point is None:
        return rotate
    else:
        valid_point = _validation.validate_array3(point, name='point')

        translate_to_origin = np.eye(4)
        translate_to_origin[:3, 3] = valid_point * -1

        translate_from_origin = np.eye(4)
        translate_from_origin[:3, 3] = valid_point

        return translate_from_origin @ rotate @ translate_to_origin
