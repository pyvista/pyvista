"""Module containing pyvista implementation of vtkCamera."""
import warnings
from weakref import proxy

import numpy as np

import pyvista
from pyvista import _vtk
from pyvista.utilities.misc import PyvistaDeprecationWarning


class Camera(_vtk.vtkCamera):
    """PyVista wrapper for the VTK Camera class.

    Parameters
    ----------
    renderer : pyvista.Renderer, optional
        Renderer to attach the camera to.

    Examples
    --------
    Create a camera at the pyvista module level.

    >>> import pyvista
    >>> camera = pyvista.Camera()

    Access the active camera of a plotter and get the position of the
    camera.

    >>> pl = pyvista.Plotter()
    >>> pl.camera.position
    (1.0, 1.0, 1.0)

    """

    def __init__(self, renderer=None):
        """Initialize a new camera descriptor."""
        self._parallel_projection = False
        self._elevation = 0.0
        self._azimuth = 0.0

        if renderer:
            if not isinstance(renderer, pyvista.Renderer):
                raise TypeError('Camera only accepts a pyvista.Renderer or None as '
                                'the ``renderer`` argument')
            self._renderer = proxy(renderer)
        else:
            self._renderer = None

    def __repr__(self):
       """Print a repr specifying the id of the camera and its camera type."""
       return (f'<{self.__class__.__name__} at {hex(id(self))}>')

    def __eq__(self, other):
        """Compare whether the relevant attributes of two cameras are equal."""
        # attributes which are native python types and thus implement __eq__

        native_attrs = [
            'position',
            'focal_point',
            'parallel_projection',
            'distance',
            'thickness',
            'parallel_scale',
            'clipping_range',
            'view_angle',
            'roll',
        ]
        for attr in native_attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False

        this_trans = self.model_transform_matrix
        that_trans = other.model_transform_matrix
        trans_count = sum(1 for trans in [this_trans, that_trans] if trans is not None)
        if trans_count == 1:
            # either but not both are None
            return False
        if trans_count == 2:
            if not np.array_equal(this_trans, that_trans):
                return False

        return True

    def __del__(self):
        """Delete the camera."""
        self.RemoveAllObservers()
        self.parent = None

    @property
    def position(self):
        """Return or set the position of the camera in world coordinates.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.position
        (1.0, 1.0, 1.0)
        >>> pl.camera.position = (2.0, 1.0, 1.0)
        >>> pl.camera.position
        (2.0, 1.0, 1.0)

        """
        return self.GetPosition()

    @position.setter
    def position(self, value):
        """Set the position of the camera."""
        self.SetPosition(value)
        self._elevation = 0.0
        self._azimuth = 0.0
        if self._renderer:
            self.reset_clipping_range()

    def reset_clipping_range(self):
        """Reset the camera clipping range based on the bounds of the visible actors.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(pyvista.Sphere())
        >>> pl.camera.clipping_range = (1, 2)
        >>> pl.camera.reset_clipping_range()  # doctest:+SKIP
        (0.0039213485598532955, 3.9213485598532953)

        """
        if self._renderer is None:
            raise AttributeError('Camera is must be associated with a renderer to '
                                 'reset its clipping range.')
        self._renderer.reset_camera_clipping_range()

    @property
    def focal_point(self):
        """Location of the camera's focus in world coordinates.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.focal_point
        (0.0, 0.0, 0.0)
        >>> pl.camera.focal_point = (2.0, 0.0, 0.0)
        >>> pl.camera.focal_point
        (2.0, 0.0, 0.0)
        """
        return self.GetFocalPoint()

    @focal_point.setter
    def focal_point(self, point):
        """Set the location of the camera's focus in world coordinates."""
        self.SetFocalPoint(point)

    @property
    def model_transform_matrix(self):
        """Return or set the camera's model transformation matrix.

        Examples
        --------
        >>> import pyvista
        >>> import numpy as np
        >>> pl = pyvista.Plotter()
        >>> pl.camera.model_transform_matrix
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])
        >>> pl.camera.model_transform_matrix = np.array([[1., 0., 0., 0.],
        ...                                              [0., 1., 0., 0.],
        ...                                              [0., 0., 1., 0.],
        ...                                              [0., 0., 0., 0.5]])
        >>>
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 0.5]])

        """
        vtk_matrix = self.GetModelTransformMatrix()
        matrix = np.empty((4, 4))
        vtk_matrix.DeepCopy(matrix.ravel(), vtk_matrix)
        return matrix

    @model_transform_matrix.setter
    def model_transform_matrix(self, matrix):
        """Set the camera's model transformation matrix."""
        vtk_matrix = _vtk.vtkMatrix4x4()
        vtk_matrix.DeepCopy(matrix.ravel())
        self.SetModelTransformMatrix(vtk_matrix)

    @property
    def is_parallel_projection(self):
        """Return True if parallel projection is set."""
        warnings.warn( "Use of `Camera.is_parallel_projection` is deprecated. "
            "Use `Camera.parallel_projection` instead.",
            PyvistaDeprecationWarning
        )
        return self._parallel_projection

    @property
    def distance(self):
        """Return or set the distance of the focal point from the camera.
        
        Notes
        -----
        Setting the distance keeps the camera fixed and moves the focal point.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.distance
        1.73205
        >>> pl.camera.distance = 2.0
        >>> pl.camera.distance
        2.0

        """
        return self.GetDistance()

    @distance.setter
    def distance(self, distance):
        """Set the distance from the camera position to the focal point."""
        self.SetDistance(distance)

    @property
    def thickness(self):
        """Return or set the distance between clipping planes.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.thickness
        1000.0
        >>> pl.camera.thickness = 100
        >>> pl.camera.thickness
        100.0

        """
        return self.GetThickness()

    @thickness.setter
    def thickness(self, length):
        """Set the distance between clipping planes."""
        self.SetThickness(length)

    @property
    def parallel_scale(self):
        """Return or set the scaling used for a parallel projection.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.parallel_scale
        1.0
        >>> pl.camera.parallel_scale = 2.0
        >>> pl.camera.parallel_scale
        2.0

        """
        return self.GetParallelScale()

    @parallel_scale.setter
    def parallel_scale(self, scale):
        """Set the scaling used for parallel projection."""
        self.SetParallelScale(scale)

    def zoom(self, value):
        """Set the zoom of the camera.

        In perspective mode, decrease the view angle by the specified
        factor.

        In parallel mode, decrease the parallel scale by the specified
        factor. A value greater than 1 is a zoom-in, a value less than
        1 is a zoom-out.

        Parameters
        ----------
        value : float
            Zoom of the camera.  Must be greater than 0.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.zoom(2.0)

        """
        self.Zoom(value)

    @property
    def up(self):
        """Return or set the "up" of the camera.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.up
        (0.0, 0.0, 1.0)
        >>> pl.camera.up = (0.410018, 0.217989, 0.885644)
        >>> pl.camera.up
        (0.410018, 0.217989, 0.885644)

        """
        return self.GetViewUp()

    @up.setter
    def up(self, vector):
        """Set the "up" of the camera."""
        self.SetViewUp(vector)

    def enable_parallel_projection(self):
        """Enable parallel projection.

        The camera will have a parallel projection. Parallel
        projection is often useful when viewing images or 2D datasets,
        but will look odd when viewing 3D datasets.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import demos
        >>> pl = pyvista.demos.orientation_plotter()
        >>> pl.enable_parallel_projection()
        >>> pl.show()

        """
        self._parallel_projection = True
        self.SetParallelProjection(True)

    def disable_parallel_projection(self):
        """Disable the use of perspective projection.

        This is default behavior.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import demos
        >>> pl = pyvista.demos.orientation_plotter()
        >>> pl.disable_parallel_projection()
        >>> pl.show()
        """
        self._parallel_projection = False
        self.SetParallelProjection(False)

    @property
    def parallel_projection(self):
        """Return the state of the parallel projection.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import demos
        >>> pl = pyvista.Plotter()
        >>> pl.disable_parallel_projection()
        >>> pl.parallel_projection
        False
        """
        return self._parallel_projection

    @parallel_projection.setter
    def parallel_projection(self, state):
        """Return the state of the parallel projection.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import demos
        >>> pl = pyvista.Plotter()
        >>> pl.disable_parallel_projection()
        >>> pl.parallel_projection
        False
        """
        if state:
            self.enable_parallel_projection()
        else:
            self.disable_parallel_projection()

    @property
    def clipping_range(self):
        """Return or set the location of the clipping planes.

        Clipping planes are the the near and far clipping planes along
        the direction of projection.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.clipping_range
        (0.01, 1000.01)
        >>> pl.camera.clipping_range = (1, 10)
        >>> pl.camera.clipping_range
        (1.0, 10.0)

        """
        return self.GetClippingRange()

    @clipping_range.setter
    def clipping_range(self, points):
        """Set the clipping planes."""
        if points[0] > points[1]:
            raise ValueError(f'Near point must be lower than the far point.')
        self.SetClippingRange(points[0], points[1])

    @property
    def view_angle(self):
        """Return or set the camera view angle.

        Examples
        --------
        >>> import pyvista
        >>> plotter = pyvista.Plotter()
        >>> plotter.camera.view_angle
        30.0
        >>> plotter.camera.view_angle = 60.0
        >>> plotter.camera.view_angle
        60.0

        """
        return self.GetViewAngle()

    @view_angle.setter
    def view_angle(self, value):
        """Set the camera view angle."""
        self.SetViewAngle(value)

    @property
    def direction(self):
        """Vector from the camera position to the focal point.

        Examples
        --------
        >>> import pyvista
        >>> plotter = pyvista.Plotter()
        >>> plotter.camera.direction  # doctest:+SKIP
        (0.0, 0.0, -1.0)

        """
        return self.GetDirectionOfProjection()

    def view_frustum(self, aspect=1.0):
        """Get the view frustum.

        Parameters
        ----------
        aspect : float, optional
            The aspect of the viewport to compute the planes. Defaults
            to 1.0.

        Returns
        -------
        pyvista.PolyData
            View frustum.

        Examples
        --------
        >>> import pyvista
        >>> plotter = pyvista.Plotter()
        >>> frustum = plotter.camera.view_frustum(1.0)
        >>> frustum.n_points
        8
        >>> frustum.n_cells
        6

        """
        frustum_planes = [0] * 24
        self.GetFrustumPlanes(aspect, frustum_planes)
        planes = _vtk.vtkPlanes()
        planes.SetFrustumPlanes(frustum_planes)

        frustum_source = _vtk.vtkFrustumSource()
        frustum_source.ShowLinesOff()
        frustum_source.SetPlanes(planes)
        frustum_source.Update()

        frustum = pyvista.wrap(frustum_source.GetOutput())
        return frustum

    @property
    def roll(self):
        """Rotate the camera about the direction of projection.

        This will spin the camera about its axis.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.roll
        -120.00000000000001
        >>> pl.camera.roll = 45.0
        >>> pl.camera.roll
        45.0
        """
        return self.GetRoll()

    @roll.setter
    def roll(self, angle):
        """Set the rotate of the camera about the direction of projection."""
        self.SetRoll(angle)

    @property
    def elevation(self):
        """Vertical rotation of the scene.

        Rotate the camera about the cross product of the negative of
        the direction of projection and the view up vector, using the
        focal point as the center of rotation.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.elevation
        0.0
        >>> pl.camera.elevation = 45.0
        >>> pl.camera.elevation
        45.0

        """
        return self._elevation

    @elevation.setter
    def elevation(self, angle):
        """Set the vertical rotation of the scene."""
        if self._elevation:
            self.Elevation(-self._elevation)
        self._elevation = angle
        self.Elevation(angle)

    @property
    def azimuth(self):
        """Azimuth of the camera.

        Rotate the camera about the view up vector centered at the
        focal point. Note that the view up vector is whatever was set
        via SetViewUp, and is not necessarily perpendicular to the
        direction of projection.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera.azimuth
        0.0
        >>> pl.camera.azimuth = 45.0
        >>> pl.camera.azimuth
        45.0

        """
        return self._azimuth

    @azimuth.setter
    def azimuth(self, angle):
        """Set the azimuth rotation of the camera."""
        if self._azimuth:
            self.Azimuth(-self._azimuth)
        self._azimuth = angle
        self.Azimuth(angle)

    def copy(self):
        """Return a deep copy of the camera.

        Returns
        -------
        pyvista.Camera
            Deep copy of the camera.

        Examples
        --------
        Create a camera and check that it shares a transformation
        matrix with its shallow copy.

        >>> import pyvista as pv
        >>> import numpy as np
        >>> camera = pv.Camera()
        >>> camera.model_transform_matrix = np.array([[1., 0., 0., 0.],
        ...                                           [0., 1., 0., 0.],
        ...                                           [0., 0., 1., 0.],
        ...                                           [0., 0., 0., 1.]])
        >>> copied_camera = camera.copy()
        >>> copied_camera == camera
        True
        >>> camera.model_transform_matrix = np.array([[1., 0., 0., 0.],
        ...                                           [0., 1., 0., 0.],
        ...                                           [0., 0., 1., 0.],
        ...                                           [0., 0., 0., 0.5]])
        >>> copied_camera == camera
        False
        """
        immutable_attrs = [
            'position',
            'focal_point',
            'model_transform_matrix',
            'distance',
            'thickness',
            'parallel_scale',
            'up',
            'clipping_range',
            'view_angle',
            'roll',
            'parallel_projection',
        ]
        new_camera = Camera()

        for attr in immutable_attrs:
            value = getattr(self, attr)
            setattr(new_camera, attr, value)

        return new_camera
