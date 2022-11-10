"""Module containing pyvista implementation of vtkCamera."""
import warnings
from weakref import proxy

import numpy as np

import pyvista
from pyvista import _vtk
from pyvista.utilities.misc import PyVistaDeprecationWarning

from .helpers import view_vectors


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
                raise TypeError(
                    'Camera only accepts a pyvista.Renderer or None as the ``renderer`` argument'
                )
            self._renderer = proxy(renderer)
        else:
            self._renderer = None

    def __repr__(self):
        """Print a repr specifying the id of the camera and its camera type."""
        return f'<{self.__class__.__name__} at {hex(id(self))}>'

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
            raise AttributeError(
                'Camera is must be associated with a renderer to reset its clipping range.'
            )
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
        warnings.warn(
            "Use of `Camera.is_parallel_projection` is deprecated. "
            "Use `Camera.parallel_projection` instead.",
            PyVistaDeprecationWarning,
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
        value : float or str
            Zoom of the camera. If a float, must be greater than 0. Otherwise,
            if a string, must be ``"tight"``. If tight, the plot will be zoomed
            such that the actors fill the entire viewport.

        Examples
        --------
        Show the Default zoom.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Sphere())
        >>> pl.camera.zoom(1.0)
        >>> pl.show()

        Show 2x zoom.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Sphere())
        >>> pl.camera.zoom(2.0)
        >>> pl.show()

        Zoom so the actor fills the entire render window.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Sphere())
        >>> pl.camera.zoom('tight')
        >>> pl.show()

        """
        if isinstance(value, str):
            if not value == 'tight':
                raise ValueError('If a string, ``zoom`` can only be "tight"')
            self.tight()
            return

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
        """Disable the use of parallel projection.

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
            raise ValueError('Near point must be lower than the far point.')
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

    def tight(self, padding=0.0, adjust_render_window=True, view='xy', negative=False):
        """Adjust the camera position so that the actors fill the entire renderer.

        The camera view direction is reoriented to be normal to the ``view``
        plane. When ``negative=False``, The first letter of ``view`` refers
        to the axis that points to the right. The second letter of ``view``
        refers to axis that points up.  When ``negative=True``, the first
        letter refers to the axis that points left.  The up direction is
        unchanged.

        Parallel projection is enabled when using this function.

        Parameters
        ----------
        padding : float, optional
            Additional padding around the actor(s). This is effectively a zoom,
            where a value of 0.01 results in a zoom out of 1%.

        adjust_render_window : bool, optional
            Adjust the size of the render window as to match the dimensions of
            the visible actors.

        view : {'xy', 'yx', 'xz', 'zx', 'yz', 'zy'}
            Plane to which the view is oriented. Default 'xy'.

        negative : bool
            Whether to view in opposite direction. Default ``False``.

        Notes
        -----
        This resets the view direction to look at a plane with parallel projection.

        Examples
        --------
        Display the puppy image with a tight view.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> puppy = examples.download_puppy()
        >>> pl = pv.Plotter(border=True, border_width=5)
        >>> _ = pl.add_mesh(puppy, rgb=True)
        >>> pl.camera.tight()
        >>> pl.show()

        Set the background to blue use a 5% padding around the image.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(puppy, rgb=True)
        >>> pl.background_color = 'b'
        >>> pl.camera.tight(padding=0.05)
        >>> pl.show()

        """
        # inspired by vedo resetCamera. Thanks @marcomusy!
        x0, x1, y0, y1, z0, z1 = self._renderer.ComputeVisiblePropBounds()

        self.enable_parallel_projection()

        self._renderer.ComputeAspect()
        aspect = self._renderer.GetAspect()

        position0 = np.array([x0, y0, z0])
        position1 = np.array([x1, y1, z1])
        objects_size = position1 - position0
        position = position0 + objects_size / 2

        direction, viewup = view_vectors(view, negative)
        horizontal = np.cross(direction, viewup)

        vert_dist = abs(objects_size @ viewup)
        horiz_dist = abs(objects_size @ horizontal)

        # set focal point to objects' center
        # offset camera position from objects center by dist in opposite of viewing direction
        # (actual distance doesn't matter due to parallel projection)
        dist = 1
        camera_position = position + dist * direction

        self.SetViewUp(*viewup)
        self.SetPosition(*camera_position)
        self.SetFocalPoint(*position)

        ps = max(horiz_dist / aspect[0], vert_dist) / 2
        self.parallel_scale = ps * (1 + padding)
        self._renderer.ResetCameraClippingRange(x0, x1, y0, y1, z0, z1)

        if adjust_render_window:
            ren_win = self._renderer.GetRenderWindow()
            size = list(ren_win.GetSize())
            size_ratio = size[0] / size[1]
            tight_ratio = horiz_dist / vert_dist
            resize_ratio = tight_ratio / size_ratio
            if resize_ratio < 1:
                size[0] = round(size[0] * resize_ratio)
            else:
                size[1] = round(size[1] / resize_ratio)

            ren_win.SetSize(size)

            # simply call tight again to reset the parallel scale due to the
            # resized window
            self.tight(padding=padding, adjust_render_window=False, view=view, negative=negative)
