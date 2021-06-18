"""Transformations leveraging transforms3d library."""
import numpy as np



def axis_angle_rotation_tf3d(axis, angle, point=None, deg=True):
    """Return a 4x4 matrix for rotation about an axis by given angle, optionally about a given point."""
    if deg:
        angle *= np.pi / 180
    import transforms3d as tf3d
    return tf3d.axangles.axangle2aff(axis, angle, point=point)


def reflection_tf3d(normal, point=None):
    """Return a 4x4 matrix for reflection across a normal about a point."""
    import transforms3d as tf3d
    return tf3d.reflections.rfnorm2aff(normal, point=point)


def axis_angle_rotation(axis, angle, point=None, deg=True):
    """Return a 4x4 matrix for rotation about an axis by given angle, optionally about a given point."""
    if deg:
        # convert to radians
        angle *= np.pi / 180

    # return early for no rotation
    if np.isclose(angle % (2 * np.pi), [0, 2 * np.pi], rtol=0, atol=1e-10).any():
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
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    augmented = np.eye(4)
    augmented[:-1, :-1] = R

    if point is not None:
        # rotation of point p would be R @ (p - point) + point
        # which is R @ p + (point - R @ point)
        augmented[:-1, -1] = point - R @ point

    return augmented


def reflection(normal, point=None):
    """Return a 4x4 matrix for reflection across a normal about a point."""
    normal = np.asarray(normal, dtype='float64')
    if normal.shape != (3,):
        raise ValueError('Normal must be a 3-length array-like.')
    if point is not None:
        point = np.asarray(point)
        if point.shape != (3,):
            raise ValueError('Plane reference point must '
                             'be a 3-length array-like.')

    # check and normalize
    normal_norm = np.linalg.norm(normal)
    if np.isclose(normal_norm, 0):
        raise ValueError('Plane normal cannot be zero.')
    if not np.isclose(normal_norm, 1):
        normal = normal / normal_norm

    # build reflection matrix
    projection = np.outer(normal, normal)
    T = np.eye(3) - 2 * projection
    augmented = np.eye(4)
    augmented[:-1, :-1] = T

    if point is not None:
        # reflection of point p would be T @ (p - point) + point
        # which is T @ p + (point - T @ point)
        augmented[:-1, -1] = point - T @ point

    return augmented


def apply_transformation_to_points(transformation, points, inplace=False):
    """Apply a given transformation matrix (3x3 or 4x4) to a set of points.

    Parameters
    ----------
    transformation : np.ndarray
        Transformation matrix of shape (3, 3) or (4, 4).

    points : np.ndarray
        Array of points to be transformed of shape (N, 3).

    inplace : bool, optional
        Updates points in-place while returning nothing.

    Returns
    -------
    new_points : np.ndarray
        Transformed points.

    Examples
    --------
    Scale a set of points in-place.

    >>> import numpy as np
    >>> import pyvista
    >>> from pyvista import examples
    >>> points = examples.load_airplane().points
    >>> points_orig = points.copy()
    >>> scale_factor = 2
    >>> tf = scale_factor * np.eye(4)
    >>> tf[3, 3,] = 1
    >>> pyvista.transformations.apply_transformation_to_points(tf, points, inplace=True)
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
    else:
        # otherwise return the new points
        return points_2
