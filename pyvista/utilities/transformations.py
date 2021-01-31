"""Transformations leveraging transforms3d library."""
import numpy as np
import transforms3d as tf3d


def axis_angle_rotation(axis, angle, point=None, deg=True):
    """Return a 4x4 matrix for rotation about an axis by given angle, optionally about a given point."""
    if deg:
        angle *= np.pi / 180
    return tf3d.axangles.axangle2aff(axis, angle, point=point)


def reflection(normal, point=None):
    """Return a 4x4 matrix for reflection across a normal about a point."""
    return tf3d.reflections.rfnorm2aff(normal, point=None)


def apply_transformation_to_points(transformation, points, inplace=False):
    """Apply a given transformation matrix (3x3 or 4x4) to a set of points."""
    if transformation.shape not in ((3, 3), (4, 4)):
        raise RuntimeError('`transformation` must be of shape (3, 3) or (4, 4)')

    if transformation.shape[1] == 4:
        # a stack is a copy
        points_2 = np.hstack((points, np.ones((len(points), 1))))
    else:
        points_2 = points

    # Paged matrix multiplication. For arrays with ndim > 2, matmul assumes
    # that the matrices to be multiplied lie in the last two dimensions.
    points_2 = np.matmul(transformation[np.newaxis, :, :],
                         points_2.T)[0, :3, :].T

    # If inplace, set the points
    if inplace:
        points[:] = points_2
    else:
        # otherwise return the new points
        return points_2
