"""Transformations leveraging transforms3d library."""
import numpy as np
import transforms3d as tf3d


def axis_angle_rotation(axis, angle, point=None, deg=True):
    """Return a 4x4 matrix for rotation about an axis by given angle, optionally about a given point."""
    if deg:
        angle *= np.pi / 180
    return tf3d.axangles.axangle2aff(axis, angle, point=point)


def apply_transformation_to_points(transformation, points, inplace=False):
    """Apply a given transformation matrix (3x3 or 4x4) to a set of points."""
    if transformation.shape not in ((3, 3), (4, 4)):
        raise RuntimeError('`transformation` must be of shape (3, 3) or (4, 4)')

    # Copy original array to if not inplace
    if not inplace:
        points = points.copy()

    if transformation.shape[1] == 4:
        points_homogeneous = np.hstack((points, np.ones((len(points), 1))))
    points[:, :3] = np.matmul(transformation[np.newaxis, :, :], points_homogeneous.T)[0, :3, :].T

    if not inplace:
        return points
