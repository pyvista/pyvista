"""Transformations leveraging transforms3d library."""
import numpy as np



def axis_angle_rotation(axis, angle, point=None, deg=True):
    """Return a 4x4 matrix for rotation about an axis by given angle, optionally about a given point."""
    if deg:
        angle *= np.pi / 180
    import transforms3d as tf3d
    return tf3d.axangles.axangle2aff(axis, angle, point=point)


def reflection(normal, point=None):
    """Return a 4x4 matrix for reflection across a normal about a point."""
    import transforms3d as tf3d
    return tf3d.reflections.rfnorm2aff(normal, point=point)


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
