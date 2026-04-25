"""Module implementing point transformations and their matrices."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal
from typing import TypeAlias
from typing import overload

import numpy as np

from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core import _validation
from pyvista.core.utilities.misc import _reciprocal

if TYPE_CHECKING:
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core import TransformLike
    from pyvista.core._typing_core import VectorLike

    _FiveArrays: TypeAlias = tuple[
        NumpyArray[float],
        NumpyArray[float],
        NumpyArray[float],
        NumpyArray[float],
        NumpyArray[float],
    ]


@_deprecate_positional_args(allowed=['axis', 'angle'])
def axis_angle_rotation(  # noqa: PLR0917
    axis: VectorLike[float],
    angle: float,
    point: VectorLike[float] | None = None,
    deg: bool = True,  # noqa: FBT001, FBT002
) -> NumpyArray[float]:
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

    axis_ = _validation.validate_array3(axis, dtype_out=float, name='axis')
    if point is not None:
        point_ = _validation.validate_array3(point, dtype_out=float, name='point')

    # check and normalize
    axis_norm = np.linalg.norm(axis_)
    if np.isclose(axis_norm, 0):
        msg = 'Cannot rotate around zero vector axis.'
        raise ValueError(msg)
    if not np.isclose(axis_norm, 1):
        axis_ = axis_ / axis_norm

    # build Rodrigues' rotation matrix
    K = np.zeros((3, 3))
    K[[2, 0, 1], [1, 2, 0]] = axis_
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
        augmented[:-1, -1] = point_ - R @ point_

    return augmented


def reflection(
    normal: VectorLike[float], point: VectorLike[float] | None = None
) -> NumpyArray[float]:
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
    >>> mirrored = transformations.apply_transformation_to_points(trans, verts)
    >>> np.allclose(mirrored, verts[[np.r_[4:8, 0:4]], :])
    True

    """
    normal = np.asarray(normal, dtype='float64')
    if normal.shape != (3,):
        msg = 'Normal must be a 3-length array-like.'
        raise ValueError(msg)
    if point is not None:
        point = np.asarray(point)
        if point.shape != (3,):
            msg = 'Plane reference point must be a 3-length array-like.'
            raise ValueError(msg)

    # check and normalize
    normal_norm = np.linalg.norm(normal)
    if np.isclose(normal_norm, 0):
        msg = 'Plane normal cannot be zero.'
        raise ValueError(msg)
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


@overload
def apply_transformation_to_points(
    transformation: NumpyArray[float],
    points: NumpyArray[float],
    inplace: Literal[True] = True,  # noqa: FBT002
) -> None: ...
@overload
def apply_transformation_to_points(
    transformation: NumpyArray[float],
    points: NumpyArray[float],
    inplace: Literal[False] = False,  # noqa: FBT002
) -> NumpyArray[float]: ...
@overload
def apply_transformation_to_points(
    transformation: NumpyArray[float],
    points: NumpyArray[float],
    inplace: bool = ...,  # noqa: FBT001
) -> NumpyArray[float] | None: ...
@_deprecate_positional_args(allowed=['transformation', 'points'])
def apply_transformation_to_points(
    transformation: NumpyArray[float],
    points: NumpyArray[float],
    inplace: Literal[True, False] = False,  # noqa: FBT002
) -> NumpyArray[float] | None:
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
        msg = '`transformation` must be of shape (3, 3) or (4, 4).'
        raise ValueError(msg)

    if points.shape[1] != 3:
        msg = '`points` must be of shape (N, 3).'
        raise ValueError(msg)

    if transformation_shape[0] == 4:
        # Divide by scale factor when homogeneous
        transformation /= transformation[3, 3]

        # Add the homogeneous coordinate
        # `points_2` is a copy of the data, not a view
        points_2 = np.empty((len(points), 4))
        points_2[:, :-1] = points
        points_2[:, -1] = 1
    else:
        points_2 = points  # type: ignore[assignment]

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


def decomposition(transformation: TransformLike, *, homogeneous: bool = False) -> _FiveArrays:
    """Decompose a transformation into its components.

    The transformation matrix ``M`` is decomposed into five components:

    - translation ``T``
    - rotation ``R``
    - reflection ``N``
    - scaling ``S``
    - shearing ``K``

    such that, when represented as 4x4 matrices, ``M = TRNSK``. The decomposition is
    unique and is computed with polar matrix decomposition.

    By default, compact representations of the transformations are returned (e.g. as a
    3-element vector or a 3x3 matrix). Optionally, 4x4 matrices may be returned instead.

    .. note::

        - The rotation is orthonormal and right-handed with positive determinant.
        - The scaling factors are positive.
        - The reflection is either ``1`` (no reflection) or ``-1`` (has reflection)
          and can be used like a scaling factor.

    Parameters
    ----------
    transformation : TransformLike
        Array or transform to decompose.

    homogeneous : bool, default: False
        If ``True``, return the components (translation, rotation, etc.) as 4x4
        homogeneous matrices. By default, reflection is a scalar, translation and
        scaling are length-3 vectors, and rotation and shear are 3x3 matrices.

    Returns
    -------
    numpy.ndarray
        Translation component ``T``. Returned as a 3-element vector (or a 4x4
        translation matrix if ``homogeneous`` is ``True``).

    numpy.ndarray
        Rotation component ``R``. Returned as a 3x3 orthonormal rotation matrix of row
        vectors (or a 4x4 rotation matrix if ``homogeneous`` is ``True``).

    numpy.ndarray
        Reflection component ``N``. Returned as a NumPy scalar (or a 4x4 reflection
        matrix if ``homogeneous`` is ``True``).

    numpy.ndarray
        Scaling component ``S``. Returned as a 3-element vector (or a 4x4 scaling matrix
        if ``homogeneous`` is ``True``).

    numpy.ndarray
        Shear component ``K``. Returned as a 3x3 matrix with ones on the diagonal and
        shear values in the off-diagonals (or as a 4x4 shearing matrix if ``homogeneous``
        is ``True``).

    Examples
    --------
    Decompose a transformation matrix which has scaling, rotation, and translation.

    >>> import pyvista as pv
    >>> matrix = [
    ...     [0.0, -2.0, 0.0, 4.0],
    ...     [1.0, 0.0, 0.0, 5.0],
    ...     [0.0, 0.0, 3.0, 6.0],
    ...     [0.0, 0.0, 0.0, 1.0],
    ... ]
    >>> T, R, N, S, K = pv.transformations.decomposition(matrix)

    Since the input has no shear, this component is the identity matrix.

    >>> K  # shear
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    >>> S  # scale
    array([1., 2., 3.])

    There is no reflection so this component is ``1``.

    >>> N  # reflection
    array(1.)

    >>> R  # rotation
    array([[ 0., -1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.]])

    >>> T  # translation
    array([4., 5., 6.])

    Repeat the example, but this time with a small shear component of 0.1. Note how the
    presence of shear also affects the values of the scaling and rotation components.

    >>> matrix = [
    ...     [0.0, -2.0, 0.0, 4.0],
    ...     [1.0, 0.1, 0.0, 5.0],
    ...     [0.0, 0.0, 3.0, 6.0],
    ...     [0.0, 0.0, 0.0, 1.0],
    ... ]
    >>> T, R, N, S, K = pv.transformations.decomposition(matrix)

    >>> K  # shear
    array([[1.        , 0.03333333, 0.        ],
           [0.01663894, 1.        , 0.        ],
           [0.        , 0.        , 1.        ]])

    >>> S  # scale
    array([0.99944491, 2.0022213 , 3.        ])

    >>> N  # reflection
    array(1.)

    >>> R  # rotation
    array([[ 0.03331483, -0.99944491,  0.        ],
           [ 0.99944491,  0.03331483,  0.        ],
           [ 0.        ,  0.        ,  1.        ]])

    >>> T  # translation
    array([4., 5., 6.])

    """
    matrix4x4 = _validation.validate_transform4x4(transformation)

    dtype_out = matrix4x4.dtype
    I3 = np.eye(3, dtype=dtype_out)
    matrix3x3 = matrix4x4[:3, :3]

    T = matrix4x4[:3, 3]
    RN, SK = _polar_decomposition(matrix3x3)

    # Get scale from diagonals and shear from off-diagonals
    S = np.diagonal(SK).copy()  # Copy since it's read only

    # Avoid division by zero for cases with rank < 3
    inv_S = _reciprocal(S, tol=1e-12, value_if_division_by_zero=0.0)
    K = (SK * (I3 == 0.0)) * inv_S[:, np.newaxis] + I3

    # Get reflection and ensure rotation is right-handed
    if np.linalg.det(RN) < 0:
        # Reflections are present
        R = RN * -1
        N = np.array(-1, dtype=dtype_out)
    else:
        R = RN
        N = np.array(1, dtype=dtype_out)

    if homogeneous:
        return _decomposition_as_homogeneous(T, R, N, S, K)
    return T, R, N, S, K


def _decomposition_as_homogeneous(  # noqa: PLR0917
    T: NumpyArray[float],  # noqa: N803
    R: NumpyArray[float],  # noqa: N803
    N: NumpyArray[float],  # noqa: N803
    S: NumpyArray[float],  # noqa: N803
    K: NumpyArray[float],  # noqa: N803
) -> _FiveArrays:
    """Return TRNSK decomposition as homogeneous matrices."""
    dtype_out = T.dtype  # Assume all inputs have the same dtype
    I3 = np.eye(3, dtype=dtype_out)
    I4 = np.eye(4, dtype=dtype_out)

    T4 = I4.copy()
    T4[:3, 3] = T

    R4 = I4.copy()
    R4[:3, :3] = R

    N4 = I4.copy()
    N4[:3, :3] = I3 * N

    S4 = I4.copy()
    S4[:3, :3] = I3 * S

    K4 = I4.copy()
    K4[:3, :3] = K

    return T4, R4, N4, S4, K4


def _polar_decomposition(a: NumpyArray[float]) -> tuple[NumpyArray[float], NumpyArray[float]]:
    # Decompose `a=up` where u is orthonormal and p is positive semi-definite
    # See scipy.linalg.polar for details
    w, s, vh = np.linalg.svd(a, full_matrices=False)
    u = w.dot(vh)
    p = (vh.T.conj() * s).dot(vh)
    return u, p
