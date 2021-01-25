"""Module containing pyvista implementation of vtkCamera."""

import numpy as np
import vtk


class Camera(vtk.vtkCamera):
    """PyVista wrapper for the VTK Camera class.

    Examples
    --------
    Create a camera at the pyvista module level

    >>> import pyvista
    >>> camera = pyvista.Camera()

    Access the active camera of a plotter and get the position of the
    camera.

    >>> pl = pyvista.Plotter()
    >>> pl.camera.position
    (1.0, 1.0, 1.0)

    """

    def __init__(self):
        """Initialize a new camera descriptor."""
        self._focus = self.GetFocalPoint()
        self._is_parallel_projection = False

    @property
    def position(self):
        """Position of the camera in world coordinates.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.position
        (1.0, 1.0, 1.0)

        """
        return self.GetPosition()

    @position.setter
    def position(self, value):
        """Set the position of the camera.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.position = 2.0, 1.0, 1.0
        """
        self.SetPosition(value)

    @property
    def focal_point(self):
        """Location of the camera's focus in world coordinates.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.focal_point
        (0.0, 0.0, 0.0)
        """
        return self._focus

    @focal_point.setter
    def focal_point(self, point):
        """Set the location of the camera's focus in world coordinates.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.focal_point = (2.0, 0.0, 0.0)
        """
        self.SetFocalPoint(point)
        self._focus = self.GetFocalPoint()

    @property
    def model_transform_matrix(self):
        """Return the camera's model transformation matrix.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.model_transform_matrix
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])
        """
        vtk_matrix = self.GetModelTransformMatrix()
        matrix = np.empty((4, 4))
        vtk_matrix.DeepCopy(matrix.ravel(), vtk_matrix)
        return matrix

    @model_transform_matrix.setter
    def model_transform_matrix(self, matrix):
        """Set the camera's model transformation matrix.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> trans_mat = np.array([[1., 0., 0., 0.],
                                  [0., 1., 0., 0.],
                                  [0., 0., 1., 0.],
                                  [0., 0., 0., 1.]])
        >>> pl.camera.model_transform_matrix = trans_mat
        """
        vtk_matrix = vtk.vtkMatrix4x4()
        vtk_matrix.DeepCopy(matrix.ravel())
        self.SetModelTransformMatrix(vtk_matrix)

    @property
    def is_parallel_projection(self):
        """Return True if parallel projection is set."""
        return self._is_parallel_projection

    @property
    def distance(self):
        """Distance from the camera position to the focal point.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.distance  # doctest:+SKIP
        1.732050807568
        """
        return self.GetDistance()

    @property
    def thickness(self):
        """Return the distance between clipping planes.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.thickness
        1000.0
        """
        return self.GetThickness()

    @thickness.setter
    def thickness(self, length):
        """Set the distance between clipping planes.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.thickness = 100
        """
        self.SetThickness(length)

    @property
    def parallel_scale(self):
        """Scaling used for a parallel projection.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.parallel_scale
        1.0
        """
        return self.GetParallelScale()

    @parallel_scale.setter
    def parallel_scale(self, scale):
        """Set the scaling used for parallel projection.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.parallel_scale = 2.0
        """
        self.SetParallelScale(scale)

    def zoom(self, value):
        """Set the zoom of the camera.

        In perspective mode, decrease the view angle by the specified
        factor.

        In parallel mode, decrease the parallel scale by the specified
        factor. A value greater than 1 is a zoom-in, a value less than
        1 is a zoom-out.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.zoom(2.0)

        """
        self.Zoom(value)

    @property
    def up(self):
        """Return the "up" of the camera.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.up
        (0.0, 0.0, 1.0)

        """
        return self.GetViewUp()

    @up.setter
    def up(self, vector):
        """Set the "up" of the camera.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.up = (0.410018, 0.217989, 0.885644)
        """
        self.SetViewUp(vector)

    def enable_parallel_projection(self, flag=True):
        """Enable parallel projection.

        The camera will have a parallel projection. Parallel projection is
        often useful when viewing images or 2D datasets.

        """
        self._is_parallel_projection = flag
        self.SetParallelProjection(flag)

    def disable_parallel_projection(self):
        """Disable the use of perspective projection."""
        self.enable_parallel_projection(False)

    @property
    def clipping_range(self):
        """Return the Clipping range.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.clipping_range
        (0.01, 1000.01)
        """
        return self.GetClippingRange()

    @clipping_range.setter
    def clipping_range(self, points):
        """Set the clipping range.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.clipping_range = (1, 10)
        """
        if points[0] > points[1]:
            raise ValueError(f'Near point must be lower than the far point.')
        self.SetClippingRange(points[0], points[1])

    def __del__(self):
        """Delete the camera."""
        self.RemoveAllObservers()
        self.parent = None
