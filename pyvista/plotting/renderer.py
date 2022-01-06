"""Module containing pyvista implementation of vtkRenderer."""

import collections.abc
from functools import partial
from typing import Sequence
from weakref import proxy

import numpy as np

import pyvista
from pyvista import MAX_N_COLOR_BARS, _vtk
from pyvista.utilities import check_depth_peeling, try_callback, wrap

from .camera import Camera
from .charts import Charts
from .tools import create_axes_marker, create_axes_orientation_box, parse_color, parse_font_family

ACTOR_LOC_MAP = [
    'upper right',
    'upper left',
    'lower left',
    'lower right',
    'center left',
    'center right',
    'lower center',
    'upper center',
    'center',
]


def map_loc_to_pos(loc, size, border=0.05):
    """Map location and size to a VTK position and position2.

    Attempt to place 2d actor in a sensible position.

    """
    if not isinstance(size, Sequence) or len(size) != 2:
        raise ValueError(
            f'`size` must be a list of length 2. Passed value is {size}'
        )

    if 'right' in loc:
        x = 1 - size[1] - border
    elif 'left' in loc:
        x = border
    else:
        x = 0.5 - size[1]/2

    if 'upper' in loc:
        y = 1 - size[1] - border
    elif 'lower' in loc:
        y = border
    else:
        y = 0.5 - size[1]/2

    return x, y, size


def make_legend_face(face):
    """Create the legend face."""
    if face is None:
        legendface = pyvista.PolyData([0.0, 0.0, 0.0])
    elif face in ["-", "line"]:
        legendface = _line_for_legend()
    elif face in ["^", "triangle"]:
        legendface = pyvista.Triangle()
    elif face in ["o", "circle"]:
        legendface = pyvista.Circle()
    elif face in ["r", "rectangle"]:
        legendface = pyvista.Rectangle()
    elif isinstance(face, pyvista.PolyData):
        legendface = face
    else:
        raise ValueError(f'Invalid face "{face}".  Must be one of the following:\n'
                         '\t"triangle"\n'
                         '\t"circle"\n'
                         '\t"rectangle"\n'
                         '\tNone'
                         '\tpyvista.PolyData')
    return legendface

def scale_point(camera, point, invert=False):
    """Scale a point using the camera's transform matrix.

    Parameters
    ----------
    camera : Camera
        The camera who's matrix to use.

    point : tuple(float)
        Length 3 tuple of the point coordinates.

    invert : bool
        If ``True``, invert the matrix to transform the point out of
        the camera's transformed space. Default is ``False`` to
        transform a point from world coordinates to the camera's
        transformed space.

    Returns
    -------
    tuple
        Scaling of the camera in ``(x, y, z)``.

    """
    if invert:
        mtx = _vtk.vtkMatrix4x4()
        mtx.DeepCopy(camera.GetModelTransformMatrix())
        mtx.Invert()
    else:
        mtx = camera.GetModelTransformMatrix()
    scaled = mtx.MultiplyDoublePoint((point[0], point[1], point[2], 0.0))
    return (scaled[0], scaled[1], scaled[2])


class CameraPosition:
    """Container to hold camera location attributes."""

    def __init__(self, position, focal_point, viewup):
        """Initialize a new camera position descriptor."""
        self._position = position
        self._focal_point = focal_point
        self._viewup = viewup

    def to_list(self):
        """Convert to a list of the position, focal point, and viewup.

        Returns
        -------
        list
            List of the position, focal point, and view up of the camera.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.camera_position.to_list()
        [(0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

        """
        return [self._position, self._focal_point, self._viewup]

    def __repr__(self):
        """List representation method."""
        return "[{},\n {},\n {}]".format(*self.to_list())

    def __getitem__(self, index):
        """Fetch a component by index location like a list."""
        return self.to_list()[index]

    def __eq__(self, other):
        """Comparison operator to act on list version of CameraPosition object."""
        if isinstance(other, CameraPosition):
            return self.to_list() == other.to_list()
        return self.to_list() == other

    @property
    def position(self):
        """Location of the camera in world coordinates."""
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def focal_point(self):
        """Location of the camera's focus in world coordinates."""
        return self._focal_point

    @focal_point.setter
    def focal_point(self, value):
        self._focal_point = value

    @property
    def viewup(self):
        """Viewup vector of the camera."""
        return self._viewup

    @viewup.setter
    def viewup(self, value):
        self._viewup = value


class Renderer(_vtk.vtkRenderer):
    """Renderer class."""

    # map camera_position string to an attribute
    CAMERA_STR_ATTR_MAP = {'xy': 'view_xy', 'xz': 'view_xz',
                           'yz': 'view_yz', 'yx': 'view_yx',
                           'zx': 'view_zx', 'zy': 'view_zy',
                           'iso': 'view_isometric'}

    def __init__(self, parent, border=True, border_color=(1, 1, 1),
                 border_width=2.0):
        """Initialize the renderer."""
        super().__init__()
        self._actors = {}
        self.parent = parent  # the plotter
        self._theme = parent.theme
        self.camera_set = False
        self.bounding_box_actor = None
        self.scale = [1.0, 1.0, 1.0]
        self.AutomaticLightCreationOff()
        self._labels = {}  # tracks labeled actors
        self._legend = None
        self._floor = None
        self._floors = []
        self._floor_kwargs = []
        # this keeps track of lights added manually to prevent garbage collection
        self._lights = []
        self._camera = Camera(self)
        self.SetActiveCamera(self._camera)
        self._empty_str = None  # used to track reference to a vtkStringArray
        self._shadow_pass = None

        # This is a private variable to keep track of how many colorbars exist
        # This allows us to keep adding colorbars without overlapping
        self._scalar_bar_slots = set(range(MAX_N_COLOR_BARS))
        self._scalar_bar_slot_lookup = {}
        self.__charts = None

        self._border_actor = None
        if border:
            self.add_border(border_color, border_width)

    @property
    def _charts(self):
        """Return the charts collection."""
        # lazy instantiation here to avoid creating the charts object unless needed.
        if self.__charts is None:
            self.__charts = Charts(self)
            self.AddObserver("StartEvent", partial(try_callback, self._render_event))
        return self.__charts

    @property
    def camera_position(self):
        """Return camera position of active render window.

        Returns
        -------
        pyvista.CameraPosition
            Camera position.

        """
        return CameraPosition(
            scale_point(self.camera, self.camera.position, invert=True),
            scale_point(self.camera, self.camera.focal_point, invert=True),
            self.camera.up)

    @camera_position.setter
    def camera_position(self, camera_location):
        """Set camera position of all active render windows."""
        if camera_location is None:
            return
        elif isinstance(camera_location, str):
            camera_location = camera_location.lower()
            if camera_location not in self.CAMERA_STR_ATTR_MAP:
                err = pyvista.core.errors.InvalidCameraError
                raise err('Invalid view direction.  '
                          'Use one of the following:\n   '
                          f'{", ".join(self.CAMERA_STR_ATTR_MAP)}')

            getattr(self, self.CAMERA_STR_ATTR_MAP[camera_location])()

        elif isinstance(camera_location[0], (int, float)):
            if len(camera_location) != 3:
                raise pyvista.core.errors.InvalidCameraError
            self.view_vector(camera_location)
        else:
            # check if a valid camera position
            if not isinstance(camera_location, CameraPosition):
                if not len(camera_location) == 3:
                    raise pyvista.core.errors.InvalidCameraError
                elif any([len(item) != 3 for item in camera_location]):
                    raise pyvista.core.errors.InvalidCameraError

            # everything is set explicitly
            self.camera.position = scale_point(self.camera, camera_location[0], invert=False)
            self.camera.focal_point = scale_point(self.camera, camera_location[1], invert=False)
            self.camera.up = camera_location[2]

        # reset clipping range
        self.reset_camera_clipping_range()
        self.camera_set = True
        self.Modified()

    def reset_camera_clipping_range(self):
        """Reset the camera clipping range based on the bounds of the visible actors.

        This ensures that no props are cut off
        """
        self.ResetCameraClippingRange()

    @property
    def camera(self):
        """Return the active camera for the rendering scene."""
        return self._camera

    @camera.setter
    def camera(self, source):
        """Set the active camera for the rendering scene."""
        self._camera = source
        self.SetActiveCamera(self._camera)
        self.camera_position = CameraPosition(
            scale_point(source, source.position, invert=True),
            scale_point(source, source.focal_point, invert=True),
            source.up
        )
        self.Modified()
        self.camera_set = True

    @property
    def bounds(self):
        """Return the bounds of all actors present in the rendering window."""
        the_bounds = np.array([np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf])

        def _update_bounds(bounds):
            def update_axis(ax):
                if bounds[ax*2] < the_bounds[ax*2]:
                    the_bounds[ax*2] = bounds[ax*2]
                if bounds[ax*2+1] > the_bounds[ax*2+1]:
                    the_bounds[ax*2+1] = bounds[ax*2+1]
            for ax in range(3):
                update_axis(ax)
            return

        for actor in self._actors.values():
            if isinstance(actor, (_vtk.vtkCubeAxesActor, _vtk.vtkLightActor)):
                continue
            if (hasattr(actor, 'GetBounds') and actor.GetBounds() is not None
                 and id(actor) != id(self.bounding_box_actor)):
                _update_bounds(actor.GetBounds())

        if np.any(np.abs(the_bounds)):
            the_bounds[the_bounds == np.inf] = -1.0
            the_bounds[the_bounds == -np.inf] = 1.0

        return the_bounds.tolist()

    @property
    def length(self):
        """Return the length of the diagonal of the bounding box of the scene.

        Returns
        -------
        float
            Length of the diagional of the bounding box.
        """
        return pyvista.Box(self.bounds).length

    @property
    def center(self):
        """Return the center of the bounding box around all data present in the scene.

        Returns
        -------
        list
            Cartesian coordinates of the center.

        """
        bounds = self.bounds
        x = (bounds[1] + bounds[0])/2
        y = (bounds[3] + bounds[2])/2
        z = (bounds[5] + bounds[4])/2
        return [x, y, z]

    @property
    def background_color(self):
        """Return the background color of this renderer."""
        return self.GetBackground()

    @background_color.setter
    def background_color(self, color):
        """Set the background color of this renderer."""
        self.set_background(color)
        self.Modified()

    def _render_event(self, *args, **kwargs):
        """Notify all charts about render event."""
        for chart in self._charts:
            chart._render_event(*args, **kwargs)

    def enable_depth_peeling(self, number_of_peels=None, occlusion_ratio=None):
        """Enable depth peeling to improve rendering of translucent geometry.

        Parameters
        ----------
        number_of_peels : int
            The maximum number of peeling layers. Initial value is 4
            and is set in the ``pyvista.global_theme``. A special value of
            0 means no maximum limit.  It has to be a positive value.

        occlusion_ratio : float
            The threshold under which the depth peeling algorithm
            stops to iterate over peel layers. This is the ratio of
            the number of pixels that have been touched by the last
            layer over the total number of pixels of the viewport
            area. Initial value is 0.0, meaning rendering has to be
            exact. Greater values may speed up the rendering with
            small impact on the quality.

        Returns
        -------
        bool
            If depth peeling is supported.

        """
        if number_of_peels is None:
            number_of_peels = self._theme.depth_peeling.number_of_peels
        if occlusion_ratio is None:
            occlusion_ratio = self._theme.depth_peeling.occlusion_ratio
        depth_peeling_supported = check_depth_peeling(number_of_peels,
                                                      occlusion_ratio)
        if depth_peeling_supported:
            self.SetUseDepthPeeling(True)
            self.SetMaximumNumberOfPeels(number_of_peels)
            self.SetOcclusionRatio(occlusion_ratio)
        self.Modified()
        return depth_peeling_supported

    def disable_depth_peeling(self):
        """Disable depth peeling."""
        self.SetUseDepthPeeling(False)
        self.Modified()

    def enable_anti_aliasing(self):
        """Enable anti-aliasing using FXAA.

        This tends to make edges appear softer and less pixelated.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.enable_anti_aliasing()
        >>> _ = pl.add_mesh(pyvista.Sphere(), show_edges=True)
        >>> pl.show()

        """
        self.SetUseFXAA(True)
        self.Modified()

    def disable_anti_aliasing(self):
        """Disable anti-aliasing.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.disable_anti_aliasing()
        >>> _ = pl.add_mesh(pyvista.Sphere(), show_edges=True)
        >>> pl.show()

        """
        self.SetUseFXAA(False)
        self.Modified()

    def add_border(self, color=[1, 1, 1], width=2.0):
        """Add borders around the frame.

        Parameters
        ----------
        color : str or sequence, optional
            Color of the border.

        width : float, optional
            Width of the border.

        Returns
        -------
        vtk.vtkActor2D
            Border actor.

        """
        points = np.array([[1., 1., 0.],
                           [0., 1., 0.],
                           [0., 0., 0.],
                           [1., 0., 0.]])

        lines = np.array([[2, 0, 1],
                          [2, 1, 2],
                          [2, 2, 3],
                          [2, 3, 0]]).ravel()

        poly = pyvista.PolyData()
        poly.points = points
        poly.lines = lines

        coordinate = _vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToNormalizedViewport()

        mapper = _vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(poly)
        mapper.SetTransformCoordinate(coordinate)

        actor = _vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(parse_color(color))
        actor.GetProperty().SetLineWidth(width)

        self.AddViewProp(actor)
        self.Modified()

        self._border_actor = actor
        return actor

    @property
    def has_border(self):
        """Return if the renderer has a border."""
        return self._border_actor is not None

    @property
    def border_width(self):
        """Return the border width."""
        if self.has_border:
            return self._border_actor.GetProperty().GetLineWidth()
        return 0

    @property
    def border_color(self):
        """Return the border color."""
        if self.has_border:
            return self._border_actor.GetProperty().GetColor()
        return None

    def add_chart(self, chart, *charts):
        """Add a chart to this renderer.

        Parameters
        ----------
        chart : Chart2D, ChartBox, ChartPie or ChartMPL
            Chart to add to renderer.

        *charts : Chart2D, ChartBox, ChartPie or ChartMPL
            Charts to add to renderer.

        Examples
        --------
        >>> import pyvista
        >>> chart = pyvista.Chart2D()
        >>> _ = chart.plot(range(10), range(10))
        >>> pl = pyvista.Plotter()
        >>> pl.add_chart(chart)
        >>> pl.show()

        """
        self._charts.add_chart(chart, *charts)

    def remove_chart(self, chart_or_index):
        """Remove a chart from this renderer.

        Parameters
        ----------
        chart_or_index : Chart2D, ChartBox, ChartPie, ChartMPL or int
            Either the chart to remove from this renderer or its index in the collection of charts.

        Examples
        --------
        First define a function to add two charts to a renderer.

        >>> import pyvista
        >>> def plotter_with_charts():
        ...     pl = pyvista.Plotter()
        ...     pl.background_color = 'w'
        ...     chart_left = pyvista.Chart2D(size=(0.5, 1))
        ...     _ = chart_left.line([0, 1, 2], [2, 1, 3])
        ...     pl.add_chart(chart_left)
        ...     chart_right = pyvista.Chart2D(size=(0.5, 1), loc=(0.5, 0))
        ...     _ = chart_right.line([0, 1, 2], [3, 1, 2])
        ...     pl.add_chart(chart_right)
        ...     return pl, chart_left, chart_right
        ...
        >>> pl, *_ = plotter_with_charts()
        >>> pl.show()

        Now reconstruct the same plotter but remove the right chart by index.

        >>> pl, *_ = plotter_with_charts()
        >>> pl.remove_chart(1)
        >>> pl.show()

        Finally, remove the left chart by reference.

        >>> pl, chart_left, chart_right = plotter_with_charts()
        >>> pl.remove_chart(chart_left)
        >>> pl.show()

        """
        self._charts.remove_chart(chart_or_index)

    @property
    def actors(self):
        """Return a dictionary of actors assigned to this renderer."""
        return self._actors

    def add_actor(self, uinput, reset_camera=False, name=None, culling=False,
                  pickable=True, render=True):
        """Add an actor to render window.

        Creates an actor if input is a mapper.

        Parameters
        ----------
        uinput : vtk.vtkMapper or vtk.vtkActor
            Vtk mapper or vtk actor to be added.

        reset_camera : bool, optional
            Resets the camera when ``True``.

        name : str, optional
            Name to assign to the actor.  Defaults to the memory address.

        culling : str, optional
            Does not render faces that are culled. Options are
            ``'front'`` or ``'back'``. This can be helpful for dense
            surface meshes, especially when edges are visible, but can
            cause flat meshes to be partially displayed.  Default
            ``False``.

        pickable : bool, optional
            Whether to allow this actor to be pickable within the
            render window.

        render : bool, optional
            If the render window is being shown, trigger a render
            after adding the actor.

        Returns
        -------
        actor : vtk.vtkActor
            The actor.

        actor_properties : vtk.Properties
            Actor properties.
        """
        # Remove actor by that name if present
        rv = self.remove_actor(name, reset_camera=False, render=False)

        if isinstance(uinput, _vtk.vtkMapper):
            actor = _vtk.vtkActor()
            actor.SetMapper(uinput)
        else:
            actor = uinput

        self.AddActor(actor)
        actor.renderer = proxy(self)

        if name is None:
            name = actor.GetAddressAsString("")

        self._actors[name] = actor

        if reset_camera:
            self.reset_camera(render)
        elif not self.camera_set and reset_camera is None and not rv:
            self.reset_camera(render)
        elif render:
            self.parent.render()

        self.update_bounds_axes()

        if isinstance(culling, str):
            culling = culling.lower()

        if culling:
            if culling in [True, 'back', 'backface', 'b']:
                try:
                    actor.GetProperty().BackfaceCullingOn()
                except AttributeError:  # pragma: no cover
                    pass
            elif culling in ['front', 'frontface', 'f']:
                try:
                    actor.GetProperty().FrontfaceCullingOn()
                except AttributeError:  # pragma: no cover
                    pass
            else:
                raise ValueError(f'Culling option ({culling}) not understood.')

        actor.SetPickable(pickable)
        self.ResetCameraClippingRange()
        self.Modified()

        prop = None
        if hasattr(actor, 'GetProperty'):
            prop = actor.GetProperty()

        return actor, prop

    def add_axes_at_origin(self, x_color=None, y_color=None, z_color=None,
                           xlabel='X', ylabel='Y', zlabel='Z', line_width=2,
                           labels_off=False):
        """Add axes actor at origin.

        Parameters
        ----------
        x_color : str or 3 item sequence, optional
            The color of the x axes arrow.

        y_color : str or 3 item sequence, optional
            The color of the y axes arrow.

        z_color : str or 3 item sequence, optional
            The color of the z axes arrow.

        xlabel : str, optional
            The label of the x axes arrow.

        ylabel : str, optional
            The label of the y axes arrow.

        zlabel : str, optional
            The label of the z axes arrow.

        line_width : int, optional
            Width of the arrows.

        labels_off : bool, optional
            Disables the label text when ``True``.

        Returns
        -------
        vtk.vtkAxesActor
            Actor of the axes.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(pyvista.Sphere(center=(2, 0, 0)), color='r')
        >>> _ = pl.add_mesh(pyvista.Sphere(center=(0, 2, 0)), color='g')
        >>> _ = pl.add_mesh(pyvista.Sphere(center=(0, 0, 2)), color='b')
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        """
        self._marker_actor = create_axes_marker(line_width=line_width,
            x_color=x_color, y_color=y_color, z_color=z_color,
            xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, labels_off=labels_off)
        self.AddActor(self._marker_actor)
        memory_address = self._marker_actor.GetAddressAsString("")
        self._actors[memory_address] = self._marker_actor
        self.Modified()
        return self._marker_actor

    def add_orientation_widget(self, actor, interactive=None, color=None,
                               opacity=1.0):
        """Use the given actor in an orientation marker widget.

        Color and opacity are only valid arguments if a mesh is passed.

        Parameters
        ----------
        actor : vtk.vtkActor or pyvista.DataSet
            The mesh or actor to use as the marker.

        interactive : bool, optional
            Control if the orientation widget is interactive.  By
            default uses the value from
            :attr:`pyvista.global_theme.interactive
            <pyvista.themes.DefaultTheme.interactive>`.

        color : str or sequence, optional
            The color of the actor.  This only applies if ``actor`` is
            a :class:`pyvista.DataSet`.

        opacity : int or float, optional
            Opacity of the marker.

        Returns
        -------
        vtk.vtkOrientationMarkerWidget
            Orientation marker widget.

        Examples
        --------
        Use an Arrow as the orientation widget.

        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> actor = pl.add_mesh(pyvista.Cube(), show_edges=True)
        >>> actor = pl.add_orientation_widget(pyvista.Arrow(), color='r')
        >>> pl.show()

        """
        if isinstance(actor, pyvista.DataSet):
            mapper = _vtk.vtkDataSetMapper()
            mesh = actor.copy()
            mesh.clear_data()
            mapper.SetInputData(mesh)
            actor = _vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            if color is not None:
                prop.SetColor(parse_color(color))
            prop.SetOpacity(opacity)
        if hasattr(self, 'axes_widget'):
            # Delete the old one
            self.axes_widget.EnabledOff()
            self.Modified()
            del self.axes_widget
        if interactive is None:
            interactive = self._theme.interactive
        self.axes_widget = _vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(actor)
        if hasattr(self.parent, 'iren'):
            self.axes_widget.SetInteractor(self.parent.iren.interactor)
            self.axes_widget.SetEnabled(1)
            self.axes_widget.SetInteractive(interactive)
        self.axes_widget.SetCurrentRenderer(self)
        self.Modified()
        return self.axes_widget

    def add_axes(self, interactive=None, line_width=2,
                 color=None, x_color=None, y_color=None, z_color=None,
                 xlabel='X', ylabel='Y', zlabel='Z', labels_off=False,
                 box=None, box_args=None):
        """Add an interactive axes widget in the bottom left corner.

        Parameters
        ----------
        interactive : bool, optional
            Enable this orientation widget to be moved by the user.

        line_width : int, optional
            The width of the marker lines.

        color : str or sequence, optional
            Color of the labels.

        x_color : str or sequence, optional
            Color used for the x axis arrow.  Defaults to theme axes parameters.

        y_color : str or sequence, optional
            Color used for the y axis arrow.  Defaults to theme axes parameters.

        z_color : str or sequence, optional
            Color used for the z axis arrow.  Defaults to theme axes parameters.

        xlabel : str, optional
            Text used for the x axis.

        ylabel : str, optional
            Text used for the y axis.

        zlabel : str, optional
            Text used for the z axis.

        labels_off : bool, optional
            Enable or disable the text labels for the axes.

        box : bool, optional
            Show a box orientation marker. Use ``box_args`` to adjust.
            See :func:`pyvista.create_axes_orientation_box` for details.

        box_args : dict, optional
            Parameters for the orientation box widget when
            ``box=True``. See the parameters of
            :func:`pyvista.create_axes_orientation_box`.

        Returns
        -------
        vtk.vtkAxesActor
            Axes actor.

        Examples
        --------
        Show axes without labels and with thick lines.

        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> actor = pl.add_mesh(pyvista.Box(), show_edges=True)
        >>> _ = pl.add_axes(line_width=5, labels_off=True)
        >>> pl.show()

        Use the axes orientation widget instead of the default arrows.

        >>> pl = pyvista.Plotter()
        >>> actor = pl.add_mesh(pyvista.Sphere())
        >>> _ = pl.add_axes(box=True)
        >>> pl.show()

        """
        if interactive is None:
            interactive = self._theme.interactive
        if hasattr(self, 'axes_widget'):
            self.axes_widget.EnabledOff()
            self.Modified()
            del self.axes_widget
        if box is None:
            box = self._theme.axes.box
        if box:
            if box_args is None:
                box_args = {}
            self.axes_actor = create_axes_orientation_box(
                label_color=color, line_width=line_width,
                x_color=x_color, y_color=y_color, z_color=z_color,
                xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                labels_off=labels_off, **box_args)
        else:
            self.axes_actor = create_axes_marker(
                label_color=color, line_width=line_width,
                x_color=x_color, y_color=y_color, z_color=z_color,
                xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, labels_off=labels_off)
        self.add_orientation_widget(self.axes_actor, interactive=interactive,
                                    color=None)
        return self.axes_actor

    def hide_axes(self):
        """Hide the axes orientation widget.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.hide_axes()

        """
        if hasattr(self, 'axes_widget') and self.axes_widget.GetEnabled():
            self.axes_widget.EnabledOff()
            self.Modified()

    def show_axes(self):
        """Show the axes orientation widget.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.show_axes()

        """
        if hasattr(self, 'axes_widget'):
            self.axes_widget.EnabledOn()
            self.axes_widget.SetCurrentRenderer(self)
        else:
            self.add_axes()
        self.Modified()

    @property
    def axes_enabled(self):
        """Return ``True`` when axes are enabled.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.hide_axes()
        >>> pl.renderer.axes_enabled
        False

        """
        if hasattr(self, 'axes_widget'):
            return bool(self.axes_widget.GetEnabled())
        return False

    def show_bounds(self, mesh=None, bounds=None, show_xaxis=True,
                    show_yaxis=True, show_zaxis=True,
                    show_xlabels=True, show_ylabels=True,
                    show_zlabels=True, bold=True, font_size=None,
                    font_family=None, color=None, xlabel='X Axis',
                    ylabel='Y Axis', zlabel='Z Axis', use_2d=False,
                    grid=None, location='closest', ticks=None,
                    all_edges=False, corner_factor=0.5, fmt=None,
                    minor_ticks=False, padding=0.0, render=None):
        """Add bounds axes.

        Shows the bounds of the most recent input mesh unless mesh is
        specified.

        Parameters
        ----------
        mesh : pyvista.DataSet or pyvista.MultiBlock
            Input mesh to draw bounds axes around.

        bounds : list or tuple, optional
            Bounds to override mesh bounds in the form ``[xmin, xmax,
            ymin, ymax, zmin, zmax]``.

        show_xaxis : bool, optional
            Makes x axis visible.  Default ``True``.

        show_yaxis : bool, optional
            Makes y axis visible.  Default ``True``.

        show_zaxis : bool, optional
            Makes z axis visible.  Default ``True``.

        show_xlabels : bool, optional
            Shows x labels.  Default ``True``.

        show_ylabels : bool, optional
            Shows y labels.  Default ``True``.

        show_zlabels : bool, optional
            Shows z labels.  Default ``True``.

        bold : bool, optional
            Bolds axis labels and numbers.  Default ``True``.

        font_size : float, optional
            Sets the size of the label font.  Defaults to 16.

        font_family : str, optional
            Font family.  Must be either ``'courier'``, ``'times'``,
            or ``'arial'``.

        color : str or 3 item list, optional
            Color of all labels and axis titles.  Default white.
            Either a string, rgb list, or hex color string.  For
            example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1, 1, 1]``
            * ``color='#FFFFFF'``

        xlabel : str, optional
            Title of the x axis.  Default ``"X Axis"``.

        ylabel : str, optional
            Title of the y axis.  Default ``"Y Axis"``.

        zlabel : str, optional
            Title of the z axis.  Default ``"Z Axis"``.

        use_2d : bool, optional
            This can be enabled for smoother plotting.

            .. warning::
               A bug with vtk 6.3 in Windows seems to cause this
               function to crash.

        grid : bool or str, optional
            Add grid lines to the backface (``True``, ``'back'``, or
            ``'backface'``) or to the frontface (``'front'``,
            ``'frontface'``) of the axes actor.

        location : str, optional
            Set how the axes are drawn: either static (``'all'``),
            closest triad (``front``), furthest triad (``'back'``),
            static closest to the origin (``'origin'``), or outer
            edges (``'outer'``) in relation to the camera
            position. Options include: ``'all', 'front', 'back',
            'origin', 'outer'``.

        ticks : str, optional
            Set how the ticks are drawn on the axes grid. Options include:
            ``'inside', 'outside', 'both'``.

        all_edges : bool, optional
            Adds an unlabeled and unticked box at the boundaries of
            plot. Useful for when wanting to plot outer grids while
            still retaining all edges of the boundary.

        corner_factor : float, optional
            If ``all_edges````, this is the factor along each axis to
            draw the default box. Default is 0.5 to show the full box.

        fmt : str, optional
            A format string defining how tick labels are generated from
            tick positions. A default is looked up on the active theme.

        minor_ticks : bool, optional
            If ``True``, also plot minor ticks on all axes.

        padding : float, optional
            An optional percent padding along each axial direction to
            cushion the datasets in the scene from the axes
            annotations. Defaults to 0 (no padding).

        render : bool, optional
            If the render window is being shown, trigger a render
            after showing bounds.

        Returns
        -------
        vtk.vtkCubeAxesActor
            Bounds actor.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> plotter = pyvista.Plotter()
        >>> actor = plotter.add_mesh(mesh)
        >>> actor = plotter.show_bounds(grid='front', location='outer',
        ...                             all_edges=True)
        >>> plotter.show()

        """
        self.remove_bounds_axes()

        if font_family is None:
            font_family = self._theme.font.family
        if font_size is None:
            font_size = self._theme.font.size
        if color is None:
            color = self._theme.font.color
        if fmt is None:
            fmt = self._theme.font.fmt

        color = parse_color(color)

        # Use the bounds of all data in the rendering window
        if mesh is None and bounds is None:
            bounds = self.bounds

        # create actor
        cube_axes_actor = _vtk.vtkCubeAxesActor()
        if use_2d or not np.allclose(self.scale, [1.0, 1.0, 1.0]):
            cube_axes_actor.SetUse2DMode(True)
        else:
            cube_axes_actor.SetUse2DMode(False)

        if grid:
            if isinstance(grid, str) and grid.lower() in ('front', 'frontface'):
                cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_CLOSEST)
            if isinstance(grid, str) and grid.lower() in ('both', 'all'):
                cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_ALL)
            else:
                cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_FURTHEST)
            # Only show user desired grid lines
            cube_axes_actor.SetDrawXGridlines(show_xaxis)
            cube_axes_actor.SetDrawYGridlines(show_yaxis)
            cube_axes_actor.SetDrawZGridlines(show_zaxis)
            # Set the colors
            cube_axes_actor.GetXAxesGridlinesProperty().SetColor(color)
            cube_axes_actor.GetYAxesGridlinesProperty().SetColor(color)
            cube_axes_actor.GetZAxesGridlinesProperty().SetColor(color)

        if isinstance(ticks, str):
            ticks = ticks.lower()
            if ticks in ('inside'):
                cube_axes_actor.SetTickLocationToInside()
            elif ticks in ('outside'):
                cube_axes_actor.SetTickLocationToOutside()
            elif ticks in ('both'):
                cube_axes_actor.SetTickLocationToBoth()
            else:
                raise ValueError(f'Value of ticks ({ticks}) not understood.')

        if isinstance(location, str):
            location = location.lower()
            if location in ('all'):
                cube_axes_actor.SetFlyModeToStaticEdges()
            elif location in ('origin'):
                cube_axes_actor.SetFlyModeToStaticTriad()
            elif location in ('outer'):
                cube_axes_actor.SetFlyModeToOuterEdges()
            elif location in ('default', 'closest', 'front'):
                cube_axes_actor.SetFlyModeToClosestTriad()
            elif location in ('furthest', 'back'):
                cube_axes_actor.SetFlyModeToFurthestTriad()
            else:
                raise ValueError(f'Value of location ({location}) not understood.')

        # set bounds
        if bounds is None:
            bounds = np.array(mesh.GetBounds())
        if isinstance(padding, (int, float)) and 0.0 <= padding < 1.0:
            if not np.any(np.abs(bounds) == np.inf):
                cushion = np.array([np.abs(bounds[1] - bounds[0]),
                                    np.abs(bounds[3] - bounds[2]),
                                    np.abs(bounds[5] - bounds[4])]) * padding
                bounds[::2] -= cushion
                bounds[1::2] += cushion
        else:
            raise ValueError(f'padding ({padding}) not understood. Must be float between 0 and 1')
        cube_axes_actor.SetBounds(bounds)

        # show or hide axes
        cube_axes_actor.SetXAxisVisibility(show_xaxis)
        cube_axes_actor.SetYAxisVisibility(show_yaxis)
        cube_axes_actor.SetZAxisVisibility(show_zaxis)

        # disable minor ticks
        if not minor_ticks:
            cube_axes_actor.XAxisMinorTickVisibilityOff()
            cube_axes_actor.YAxisMinorTickVisibilityOff()
            cube_axes_actor.ZAxisMinorTickVisibilityOff()

        cube_axes_actor.SetCamera(self.camera)

        # set color
        cube_axes_actor.GetXAxesLinesProperty().SetColor(color)
        cube_axes_actor.GetYAxesLinesProperty().SetColor(color)
        cube_axes_actor.GetZAxesLinesProperty().SetColor(color)

        # empty string used for clearing axis labels
        self._empty_str = _vtk.vtkStringArray()
        self._empty_str.InsertNextValue('')

        # show lines
        if show_xaxis:
            cube_axes_actor.SetXTitle(xlabel)
            if not show_xlabels:
                cube_axes_actor.SetAxisLabels(0, self._empty_str)
        else:
            cube_axes_actor.SetXTitle('')
            cube_axes_actor.SetAxisLabels(0, self._empty_str)

        if show_yaxis:
            cube_axes_actor.SetYTitle(ylabel)
            if not show_ylabels:
                cube_axes_actor.SetAxisLabels(1, self._empty_str)
        else:
            cube_axes_actor.SetYTitle('')
            cube_axes_actor.SetAxisLabels(1, self._empty_str)

        if show_zaxis:
            cube_axes_actor.SetZTitle(zlabel)
            if not show_zlabels:
                cube_axes_actor.SetAxisLabels(2, self._empty_str)
        else:
            cube_axes_actor.SetZTitle('')
            cube_axes_actor.SetAxisLabels(2, self._empty_str)

        # set font
        font_family = parse_font_family(font_family)
        for i in range(3):
            cube_axes_actor.GetTitleTextProperty(i).SetFontSize(font_size)
            cube_axes_actor.GetTitleTextProperty(i).SetColor(color)
            cube_axes_actor.GetTitleTextProperty(i).SetFontFamily(font_family)
            cube_axes_actor.GetTitleTextProperty(i).SetBold(bold)

            cube_axes_actor.GetLabelTextProperty(i).SetFontSize(font_size)
            cube_axes_actor.GetLabelTextProperty(i).SetColor(color)
            cube_axes_actor.GetLabelTextProperty(i).SetFontFamily(font_family)
            cube_axes_actor.GetLabelTextProperty(i).SetBold(bold)

        self.add_actor(cube_axes_actor, reset_camera=False, pickable=False,
                       render=render)
        self.cube_axes_actor = cube_axes_actor

        if all_edges:
            self.add_bounding_box(color=color, corner_factor=corner_factor)

        if fmt is not None:
            cube_axes_actor.SetXLabelFormat(fmt)
            cube_axes_actor.SetYLabelFormat(fmt)
            cube_axes_actor.SetZLabelFormat(fmt)

        self.Modified()
        return cube_axes_actor

    def show_grid(self, **kwargs):
        """Show gridlines and axes labels.

        A wrapped implementation of ``show_bounds`` to change default
        behaviour to use gridlines and showing the axes labels on the
        outer edges. This is intended to be similar to
        ``matplotlib``'s ``grid`` function.

        Parameters
        ----------
        **kwargs : dict, optional
            See :func:`Renderer.show_bounds` for additional keyword
            arguments.

        Returns
        -------
        vtk.vtkAxesActor
            Bounds actor.

         Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(examples.download_guitar())
        >>> _ = pl.show_grid()
        >>> pl.show()

        """
        kwargs.setdefault('grid', 'back')
        kwargs.setdefault('location', 'outer')
        kwargs.setdefault('ticks', 'both')
        return self.show_bounds(**kwargs)

    def remove_bounding_box(self, render=True):
        """Remove bounding box.

        Parameters
        ----------
        render : bool, optional
            Trigger a render once the bounding box is removed.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_bounding_box()
        >>> pl.remove_bounding_box()

        """
        if hasattr(self, '_box_object'):
            actor = self.bounding_box_actor
            self.bounding_box_actor = None
            del self._box_object
            self.remove_actor(actor, reset_camera=False, render=render)
            self.Modified()

    def add_bounding_box(self, color="grey", corner_factor=0.5, line_width=None,
                         opacity=1.0, render_lines_as_tubes=False,
                         lighting=None, reset_camera=None, outline=True,
                         culling='front'):
        """Add an unlabeled and unticked box at the boundaries of plot.

        Useful for when wanting to plot outer grids while still
        retaining all edges of the boundary.

        Parameters
        ----------
        color : str or sequence, optional
            Color of all labels and axis titles.  Default white.
            Either a string, rgb sequence, or hex color string.  For
            example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1, 1, 1]``
            * ``color='#FFFFFF'``

        corner_factor : float, optional
            This is the factor along each axis to draw the default
            box. Default is 0.5 to show the full box.

        line_width : float, optional
            Thickness of lines.

        opacity : float, optional
            Opacity of mesh.  Default 1.0 and should be between 0 and 1.

        render_lines_as_tubes : bool, optional
            Show lines as thick tubes rather than flat lines.  Control
            the width with ``line_width``.

        lighting : bool, optional
            Enable or disable directional lighting for this actor.

        reset_camera : bool, optional
            Reset camera position when ``True`` to include all actors.

        outline : bool
            Default is ``True``. when ``False``, a box with faces is
            shown with the specified culling.

        culling : str, optional
            Does not render faces that are culled. Options are
            ``'front'`` or ``'back'``. Default is ``'front'`` for
            bounding box.

        Returns
        -------
        vtk.vtkActor
            VTK actor of the bounding box.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(pyvista.Sphere())
        >>> _ = pl.add_bounding_box(line_width=5, color='black')
        >>> pl.show()

        """
        if lighting is None:
            lighting = self._theme.lighting

        self.remove_bounding_box()
        if color is None:
            color = self._theme.outline_color
        rgb_color = parse_color(color)
        if outline:
            self._bounding_box = _vtk.vtkOutlineCornerSource()
            self._bounding_box.SetCornerFactor(corner_factor)
        else:
            self._bounding_box = _vtk.vtkCubeSource()
        self._bounding_box.SetBounds(self.bounds)
        self._bounding_box.Update()
        self._box_object = wrap(self._bounding_box.GetOutput())
        name = f'BoundingBox({hex(id(self._box_object))})'

        mapper = _vtk.vtkDataSetMapper()
        mapper.SetInputData(self._box_object)
        self.bounding_box_actor, prop = self.add_actor(mapper,
                                                       reset_camera=reset_camera,
                                                       name=name, culling=culling,
                                                       pickable=False)

        prop.SetColor(rgb_color)
        prop.SetOpacity(opacity)
        if render_lines_as_tubes:
            prop.SetRenderLinesAsTubes(render_lines_as_tubes)

        # lighting display style
        if lighting is False:
            prop.LightingOff()

        # set line thickness
        if line_width:
            prop.SetLineWidth(line_width)

        prop.SetRepresentationToSurface()
        self.Modified()
        return self.bounding_box_actor

    def add_floor(self, face='-z', i_resolution=10, j_resolution=10,
                  color=None, line_width=None, opacity=1.0, show_edges=False,
                  lighting=False, edge_color=None, reset_camera=None, pad=0.0,
                  offset=0.0, pickable=False, store_floor_kwargs=True):
        """Show a floor mesh.

        This generates planes at the boundaries of the scene to behave
        like floors or walls.

        Parameters
        ----------
        face : str, optional
            The face at which to place the plane. Options are
            (``'-z'``, ``'-y'``, ``'-x'``, ``'+z'``, ``'+y'``, and
            ``'+z'``). Where the ``-/+`` sign indicates on which side of
            the axis the plane will lie.  For example, ``'-z'`` would
            generate a floor on the XY-plane and the bottom of the
            scene (minimum z).

        i_resolution : int, optional
            Number of points on the plane in the i direction.

        j_resolution : int, optional
            Number of points on the plane in the j direction.

        color : str or 3 item list, optional
            Color of all labels and axis titles.  Default gray.
            Either a string, rgb list, or hex color string.

        line_width : int, optional
            Thickness of the edges. Only if ``show_edges`` is
            ``True``.

        opacity : float, optional
            The opacity of the generated surface.

        show_edges : bool, optional
            Flag on whether to show the mesh edges for tiling.

        line_width : float, optional
            Thickness of lines.  Only valid for wireframe and surface
            representations.  Default ``None``.

        lighting : bool, optional
            Enable or disable view direction lighting.  Default
            ``False``.

        edge_color : str or sequence, optional
            Color of of the edges of the mesh.

        reset_camera : bool, optional
            Resets the camera when ``True`` after adding the floor.

        pad : float, optional
            Percentage padding between 0 and 1.

        offset : float, optional
            Percentage offset along plane normal.

        pickable : bool, optional
            Make this floor actor pickable in the renderer.

        store_floor_kwargs : bool, optional
            Stores the keyword arguments used when adding this floor.
            Useful when updating the bounds and regenerating the
            floor.

        Returns
        -------
        vtk.vtkActor
            VTK actor of the floor.

        Examples
        --------
        Add a floor below a sphere and plot it.

        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> actor = pl.add_mesh(pyvista.Sphere())
        >>> actor = pl.add_floor()
        >>> pl.show()

        """
        if store_floor_kwargs:
            kwargs = locals()
            kwargs.pop('self')
            self._floor_kwargs.append(kwargs)
        ranges = np.array(self.bounds).reshape(-1, 2).ptp(axis=1)
        ranges += (ranges * pad)
        center = np.array(self.center)
        if face.lower() in '-z':
            center[2] = self.bounds[4] - (ranges[2] * offset)
            normal = (0, 0, 1)
            i_size = ranges[0]
            j_size = ranges[1]
        elif face.lower() in '-y':
            center[1] = self.bounds[2] - (ranges[1] * offset)
            normal = (0, 1, 0)
            i_size = ranges[0]
            j_size = ranges[2]
        elif face.lower() in '-x':
            center[0] = self.bounds[0] - (ranges[0] * offset)
            normal = (1, 0, 0)
            i_size = ranges[2]
            j_size = ranges[1]
        elif face.lower() in '+z':
            center[2] = self.bounds[5] + (ranges[2] * offset)
            normal = (0, 0, -1)
            i_size = ranges[0]
            j_size = ranges[1]
        elif face.lower() in '+y':
            center[1] = self.bounds[3] + (ranges[1] * offset)
            normal = (0, -1, 0)
            i_size = ranges[0]
            j_size = ranges[2]
        elif face.lower() in '+x':
            center[0] = self.bounds[1] + (ranges[0] * offset)
            normal = (-1, 0, 0)
            i_size = ranges[2]
            j_size = ranges[1]
        else:
            raise NotImplementedError(f'Face ({face}) not implementd')
        self._floor = pyvista.Plane(center=center, direction=normal,
                                    i_size=i_size, j_size=j_size,
                                    i_resolution=i_resolution,
                                    j_resolution=j_resolution)
        self._floor.clear_data()

        if lighting is None:
            lighting = self._theme.lighting

        if edge_color is None:
            edge_color = self._theme.edge_color

        self.remove_bounding_box()
        if color is None:
            color = self._theme.floor_color
        rgb_color = parse_color(color)
        mapper = _vtk.vtkDataSetMapper()
        mapper.SetInputData(self._floor)
        actor, prop = self.add_actor(mapper,
                                     reset_camera=reset_camera,
                                     name=f'Floor({face})', pickable=pickable)

        prop.SetColor(rgb_color)
        prop.SetOpacity(opacity)

        # edge display style
        if show_edges:
            prop.EdgeVisibilityOn()
        prop.SetEdgeColor(parse_color(edge_color))

        # lighting display style
        if lighting is False:
            prop.LightingOff()

        # set line thickness
        if line_width:
            prop.SetLineWidth(line_width)

        prop.SetRepresentationToSurface()
        self._floors.append(actor)
        return actor

    def remove_floors(self, clear_kwargs=True, render=True):
        """Remove all floor actors.

        Parameters
        ----------
        clear_kwargs : bool, optional
            Clear default floor arguments.  Default ``True``.

        render : bool, optional
            Render upon removing the floor.  Default ``True``.

        Examples
        --------
        Add a floor below a sphere, remove it, and then plot it.

        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> actor = pl.add_mesh(pyvista.Sphere())
        >>> actor = pl.add_floor()
        >>> pl.remove_floors()
        >>> pl.show()

        """
        if getattr(self, '_floor', None) is not None:
            self._floor.ReleaseData()
            self._floor = None
        for actor in self._floors:
            self.remove_actor(actor, reset_camera=False, render=render)
        self._floors.clear()
        if clear_kwargs:
            self._floor_kwargs.clear()

    def remove_bounds_axes(self):
        """Remove bounds axes.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> actor = pl.add_mesh(pyvista.Sphere())
        >>> actor = pl.show_bounds(grid='front', location='outer')
        >>> pl.subplot(0, 1)
        >>> actor = pl.add_mesh(pyvista.Sphere())
        >>> actor = pl.show_bounds(grid='front', location='outer')
        >>> actor = pl.remove_bounds_axes()
        >>> pl.show()

        """
        if hasattr(self, 'cube_axes_actor'):
            self.remove_actor(self.cube_axes_actor)
            self.Modified()

    def add_light(self, light):
        """Add a light to the renderer.

        Parameters
        ----------
        light : vtk.vtkLight or pyvista.Light
            Light to add.

        """
        # convert from a vtk type if applicable
        if isinstance(light, _vtk.vtkLight) and not isinstance(light, pyvista.Light):
            light = pyvista.Light.from_vtk(light)

        if not isinstance(light, pyvista.Light):
            raise TypeError(f'Expected Light instance, got {type(light).__name__} instead.')
        self._lights.append(light)
        self.AddLight(light)
        self.Modified()

        # we add the renderer to add/remove the light actor if
        # positional or cone angle is modified
        light.add_renderer(self)

    @property
    def lights(self):
        """Return a list of all lights in the renderer.

        Returns
        -------
        list
            Lights in the renderer.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.renderer.lights   # doctest:+SKIP
        [<Light (Headlight) at 0x7f1dd8155820>,
         <Light (Camera Light) at 0x7f1dd8155760>,
         <Light (Camera Light) at 0x7f1dd8155340>,
         <Light (Camera Light) at 0x7f1dd8155460>,
         <Light (Camera Light) at 0x7f1dd8155f40>]

        """
        return list(self.GetLights())

    def remove_all_lights(self):
        """Remove all lights from the renderer."""
        self.RemoveAllLights()
        self._lights.clear()

    def clear(self):
        """Remove all actors and properties."""
        if self._actors:
            for actor in list(self._actors):
                try:
                    self.remove_actor(actor, reset_camera=False, render=False)
                except KeyError:
                    pass

        self.remove_all_lights()
        self.RemoveAllViewProps()
        self.Modified()

        self._scalar_bar_slots = set(range(MAX_N_COLOR_BARS))
        self._scalar_bar_slot_lookup = {}

    def set_focus(self, point):
        """Set focus to a point.

        Parameters
        ----------
        point : sequence
            Cartesian point to focus on in the form of ``[x, y, z]``.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Cube()
        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(mesh, show_edges=True)
        >>> _ = pl.add_point_labels([mesh.points[1]], ["Focus"])
        >>> _ = pl.camera  # this initializes the camera
        >>> pl.set_focus(mesh.points[1])
        >>> pl.show()

        """
        if isinstance(point, np.ndarray):
            if point.ndim != 1:
                point = point.ravel()
        self.camera.focal_point = scale_point(self.camera, point, invert=False)
        self.camera_set = True
        self.Modified()

    def set_position(self, point, reset=False):
        """Set camera position to a point.

        Parameters
        ----------
        point : sequence
            Cartesian point to focus on in the form of ``[x, y, z]``.

        reset : bool, optional
            Whether to reset the camera after setting the camera
            position.

        Examples
        --------
        Move the camera far away to ``[7, 7, 7]``.

        >>> import pyvista
        >>> mesh = pyvista.Cube()
        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(mesh, show_edges=True)
        >>> pl.set_position([7, 7, 7])
        >>> pl.show()

        """
        if isinstance(point, np.ndarray):
            if point.ndim != 1:
                point = point.ravel()
        self.camera.position = scale_point(self.camera, point, invert=False)
        if reset:
            self.reset_camera()
        self.camera_set = True
        self.Modified()

    def set_viewup(self, vector):
        """Set camera viewup vector.

        Parameters
        ----------
        vector : sequence
            New 3 value camera viewup vector.

        Examples
        --------
        Look from the top down by setting view up to ``[0, 1, 0]``.
        Notice how the Y axis appears vertical.

        >>> from pyvista import demos
        >>> pl = demos.orientation_plotter()
        >>> pl.set_viewup([0, 1, 0])
        >>> pl.show()

        """
        if isinstance(vector, np.ndarray):
            if vector.ndim != 1:
                vector = vector.ravel()
        self.camera.up = vector
        self.camera_set = True
        self.Modified()

    def enable_parallel_projection(self):
        """Enable parallel projection.

        The camera will have a parallel projection. Parallel projection is
        often useful when viewing images or 2D datasets.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import demos
        >>> pl = pyvista.demos.orientation_plotter()
        >>> pl.enable_parallel_projection()
        >>> pl.show()

        """
        # Fix the 'reset camera' effect produced by the VTK when parallel
        # projection is enabled.
        angle = np.radians(self.camera.view_angle)
        self.camera.parallel_scale = self.camera.distance * np.sin(.5 * angle)

        self.camera.enable_parallel_projection()
        self.Modified()

    def disable_parallel_projection(self):
        """Reset the camera to use perspective projection.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import demos
        >>> pl = pyvista.demos.orientation_plotter()
        >>> pl.disable_parallel_projection()
        >>> pl.show()

        """
        # Fix the 'reset camera' effect produced by the VTK when parallel
        # projection is disabled.
        focus = self.camera.focal_point
        angle = np.radians(self.camera.view_angle)
        distance = self.camera.parallel_scale / np.sin(.5 * angle)
        direction = self.camera.direction
        x = focus[0] - distance * direction[0]
        y = focus[1] - distance * direction[1]
        z = focus[2] - distance * direction[2]
        self.camera.position = (x, y, z)
        self.ResetCameraClippingRange()

        self.camera.disable_parallel_projection()
        self.Modified()

    @property
    def parallel_projection(self):
        """Return parallel projection state of active render window.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.parallel_projection = False
        >>> pl.parallel_projection
        False
        """
        return self.camera.parallel_projection

    @parallel_projection.setter
    def parallel_projection(self, state):
        """Set parallel projection state of all active render windows."""
        self.camera.parallel_projection = state
        self.Modified()

    @property
    def parallel_scale(self):
        """Return parallel scale of active render window.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.parallel_scale = 2
        """
        return self.camera.parallel_scale

    @parallel_scale.setter
    def parallel_scale(self, value):
        """Set parallel scale of all active render windows."""
        self.camera.parallel_scale = value
        self.Modified()

    def remove_actor(self, actor, reset_camera=False, render=True):
        """Remove an actor from the Renderer.

        Parameters
        ----------
        actor : str, vtk.vtkActor, list or tuple
            If the type is ``str``, removes the previously added actor
            with the given name. If the type is ``vtk.vtkActor``,
            removes the actor if it's previously added to the
            Renderer. If ``list`` or ``tuple``, removes iteratively
            each actor.

        reset_camera : bool, optional
            Resets camera so all actors can be seen.

        render : bool, optional
            Render upon actor removal.  Set this to ``False`` to stop
            the render window from rendering when an actor is removed.

        Returns
        -------
        bool
            ``True`` when actor removed.  ``False`` when actor has not
            been removed.

        Examples
        --------
        Add two meshes to a plotter and then remove the sphere actor.

        >>> import pyvista
        >>> mesh = pyvista.Cube()
        >>> pl = pyvista.Plotter()
        >>> cube_actor = pl.add_mesh(pyvista.Cube(), show_edges=True)
        >>> sphere_actor = pl.add_mesh(pyvista.Sphere(), show_edges=True)
        >>> _ = pl.remove_actor(cube_actor)
        >>> pl.show()

        """
        name = None
        if isinstance(actor, str):
            name = actor
            keys = list(self._actors.keys())
            names = []
            for k in keys:
                if k.startswith(f'{name}-'):
                    names.append(k)
            if len(names) > 0:
                self.remove_actor(names, reset_camera=reset_camera)
            try:
                actor = self._actors[name]
            except KeyError:
                # If actor of that name is not present then return success
                return False
        if isinstance(actor, collections.abc.Iterable):
            success = False
            for a in actor:
                rv = self.remove_actor(a, reset_camera=reset_camera)
                if rv or success:
                    success = True
            return success
        if actor is None:
            return False

        # remove any labels associated with the actor
        self._labels.pop(actor.GetAddressAsString(""), None)

        # ensure any scalar bars associated with this actor are removed
        self.parent.scalar_bars._remove_mapper_from_plotter(actor)
        self.RemoveActor(actor)

        if name is None:
            for k, v in self._actors.items():
                if v == actor:
                    name = k
        self._actors.pop(name, None)
        self.update_bounds_axes()
        if reset_camera:
            self.reset_camera()
        elif not self.camera_set and reset_camera is None:
            self.reset_camera()
        elif render:
            self.parent.render()

        self.Modified()
        return True

    def set_scale(self, xscale=None, yscale=None, zscale=None, reset_camera=True):
        """Scale all the datasets in the scene.

        Scaling in performed independently on the X, Y and Z axis.
        A scale of zero is illegal and will be replaced with one.

        Parameters
        ----------
        xscale : float, optional
            Scaling in the x direction.  Default is ``None``, which
            does not change existing scaling.

        yscale : float, optional
            Scaling in the y direction.  Default is ``None``, which
            does not change existing scaling.

        zscale : float, optional
            Scaling in the z direction.  Default is ``None``, which
            does not change existing scaling.

        reset_camera : bool, optional
            Resets camera so all actors can be seen.  Default ``True``.

        Examples
        --------
        Set the scale in the z direction to be 5 times that of
        nominal.  Leave the other axes unscaled.

        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.set_scale(zscale=5)
        >>> _ = pl.add_mesh(pyvista.Sphere())  # perfect sphere
        >>> pl.show()

        """
        if xscale is None:
            xscale = self.scale[0]
        if yscale is None:
            yscale = self.scale[1]
        if zscale is None:
            zscale = self.scale[2]
        self.scale = [xscale, yscale, zscale]

        # Update the camera's coordinate system
        transform = np.diag([xscale, yscale, zscale, 1.0])
        self.camera.model_transform_matrix = transform
        self.parent.render()
        if reset_camera:
            self.update_bounds_axes()
            self.reset_camera()
        self.Modified()

    def get_default_cam_pos(self, negative=False):
        """Return the default focal points and viewup.

        Uses ResetCamera to make a useful view.

        Parameters
        ----------
        negative : bool
            View from the opposite direction.

        Returns
        -------
        list
            List of camera position:

            * Position
            * Focal point
            * View up

        """
        focal_pt = self.center
        if any(np.isnan(focal_pt)):
            focal_pt = (0.0, 0.0, 0.0)
        position = np.array(self._theme.camera['position']).astype(float)
        if negative:
            position *= -1
        position = position / np.array(self.scale).astype(float)
        cpos = [position + np.array(focal_pt),
                focal_pt, self._theme.camera['viewup']]
        return cpos

    def update_bounds_axes(self):
        """Update the bounds axes of the render window."""
        if (hasattr(self, '_box_object') and self._box_object is not None
                and self.bounding_box_actor is not None):
            if not np.allclose(self._box_object.bounds, self.bounds):
                color = self.bounding_box_actor.GetProperty().GetColor()
                self.remove_bounding_box()
                self.add_bounding_box(color=color)
                self.remove_floors(clear_kwargs=False)
                for floor_kwargs in self._floor_kwargs:
                    floor_kwargs['store_floor_kwargs'] = False
                    self.add_floor(**floor_kwargs)
        if hasattr(self, 'cube_axes_actor'):
            self.cube_axes_actor.SetBounds(self.bounds)
            if not np.allclose(self.scale, [1.0, 1.0, 1.0]):
                self.cube_axes_actor.SetUse2DMode(True)
            else:
                self.cube_axes_actor.SetUse2DMode(False)
            self.Modified()

    def reset_camera(self, render=True, bounds=None):
        """Reset the camera of the active render window.

        The camera slides along the vector defined from camera
        position to focal point until all of the actors can be seen.

        Parameters
        ----------
        render : bool
            Trigger a render after resetting the camera.
        bounds : iterable(int)
            Automatically set up the camera based on a specified bounding box
            ``(xmin, xmax, ymin, ymax, zmin, zmax)``.

        Examples
        --------
        Add a mesh and place the camera position too close to the
        mesh.  Then reset the camera and show the mesh.

        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> actor = pl.add_mesh(pyvista.Sphere(), show_edges=True)
        >>> pl.set_position((0, 0.1, 0.1))
        >>> pl.reset_camera()
        >>> pl.show()

        """
        if bounds is not None:
            self.ResetCamera(*bounds)
        else:
            self.ResetCamera()
        if render:
            self.parent.render()
        self.Modified()

    def isometric_view(self):
        """Reset the camera to a default isometric view.

        DEPRECATED: Please use ``view_isometric``.

        """
        self.view_isometric()

    def view_isometric(self, negative=False):
        """Reset the camera to a default isometric view.

        The view will show all the actors in the scene.

        Parameters
        ----------
        negative : bool, optional
            View from the other isometric direction.

        Examples
        --------
        Isometric view.

        >>> from pyvista import demos
        >>> pl = demos.orientation_plotter()
        >>> pl.view_isometric()
        >>> pl.show()

        Negative isometric view.

        >>> from pyvista import demos
        >>> pl = demos.orientation_plotter()
        >>> pl.view_isometric(negative=True)
        >>> pl.show()

        """
        position = self.get_default_cam_pos(negative=negative)
        self.camera_position = CameraPosition(*position)
        self.camera_set = negative
        self.reset_camera()

    def view_vector(self, vector, viewup=None):
        """Point the camera in the direction of the given vector.

        Parameters
        ----------
        vector : sequence
            Three item sequence to point the camera in.

        viewup : sequence, optional
            Three item sequence describing the view up of the camera.

        """
        focal_pt = self.center
        if viewup is None:
            viewup = self._theme.camera['viewup']
        cpos = CameraPosition(vector + np.array(focal_pt),
                focal_pt, viewup)
        self.camera_position = cpos
        self.reset_camera()

    def view_xy(self, negative=False):
        """View the XY plane.

        Parameters
        ----------
        negative : bool, optional
            View from the opposite direction.

        Examples
        --------
        View the XY plane of a built-in mesh example.

        >>> from pyvista import examples
        >>> import pyvista as pv
        >>> airplane = examples.load_airplane()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(airplane)
        >>> pl.view_xy()
        >>> pl.show()

        """
        vec = np.array([0,0,1])
        viewup = np.array([0,1,0])
        if negative:
            vec *= -1
        self.view_vector(vec, viewup)

    def view_yx(self, negative=False):
        """View the YX plane.

        Parameters
        ----------
        negative : bool, optional
            View from the opposite direction.

        Examples
        --------
        View the YX plane of a built-in mesh example.

        >>> from pyvista import examples
        >>> import pyvista as pv
        >>> airplane = examples.load_airplane()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(airplane)
        >>> pl.view_yx()
        >>> pl.show()

        """
        vec = np.array([0,0,-1])
        viewup = np.array([1,0,0])
        if negative:
            vec *= -1
        self.view_vector(vec, viewup)

    def view_xz(self, negative=False):
        """View the XZ plane.

        Parameters
        ----------
        negative : bool, optional
            View from the opposite direction.

        Examples
        --------
        View the XZ plane of a built-in mesh example.

        >>> from pyvista import examples
        >>> import pyvista as pv
        >>> airplane = examples.load_airplane()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(airplane)
        >>> pl.view_xz()
        >>> pl.show()

        """
        vec = np.array([0,-1,0])
        viewup = np.array([0,0,1])
        if negative:
            vec *= -1
        self.view_vector(vec, viewup)

    def view_zx(self, negative=False):
        """View the ZX plane.

        Parameters
        ----------
        negative : bool, optional
            View from the opposite direction.

        Examples
        --------
        View the ZX plane of a built-in mesh example.

        >>> from pyvista import examples
        >>> import pyvista as pv
        >>> airplane = examples.load_airplane()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(airplane)
        >>> pl.view_zx()
        >>> pl.show()

        """
        vec = np.array([0,1,0])
        viewup = np.array([1,0,0])
        if negative:
            vec *= -1
        self.view_vector(vec, viewup)

    def view_yz(self, negative=False):
        """View the YZ plane.

        Parameters
        ----------
        negative : bool, optional
            View from the opposite direction.

        Examples
        --------
        View the YZ plane of a built-in mesh example.

        >>> from pyvista import examples
        >>> import pyvista as pv
        >>> airplane = examples.load_airplane()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(airplane)
        >>> pl.view_yz()
        >>> pl.show()

        """
        vec = np.array([1,0,0])
        viewup = np.array([0,0,1])
        if negative:
            vec *= -1
        self.view_vector(vec, viewup)

    def view_zy(self, negative=False):
        """View the ZY plane.

        Parameters
        ----------
        negative : bool, optional
            View from the opposite direction.

        Examples
        --------
        View the ZY plane of a built-in mesh example.

        >>> from pyvista import examples
        >>> import pyvista as pv
        >>> airplane = examples.load_airplane()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(airplane)
        >>> pl.view_zy()
        >>> pl.show()

        """
        vec = np.array([-1,0,0])
        viewup = np.array([0,1,0])
        if negative:
            vec *= -1
        self.view_vector(vec, viewup)

    def disable(self):
        """Disable this renderer's camera from being interactive."""
        self.SetInteractive(0)

    def enable(self):
        """Enable this renderer's camera to be interactive."""
        self.SetInteractive(1)

    def enable_eye_dome_lighting(self):
        """Enable eye dome lighting (EDL).

        Returns
        -------
        vtk.vtkOpenGLRenderer
            VTK renderer with eye dome lighting pass.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> _ = pl.enable_eye_dome_lighting()

        """
        if hasattr(self, 'edl_pass'):
            return self
        # create the basic VTK render steps
        basic_passes = _vtk.vtkRenderStepsPass()
        # blur the resulting image
        # The blur delegates rendering the unblured image to the basic_passes
        self.edl_pass = _vtk.vtkEDLShading()
        self.edl_pass.SetDelegatePass(basic_passes)

        # tell the renderer to use our render pass pipeline
        self.glrenderer = _vtk.vtkOpenGLRenderer.SafeDownCast(self)
        self.glrenderer.SetPass(self.edl_pass)
        self.Modified()
        return self.glrenderer

    def disable_eye_dome_lighting(self):
        """Disable eye dome lighting (EDL).

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.disable_eye_dome_lighting()

        """
        if not hasattr(self, 'edl_pass'):
            return
        self.SetPass(None)
        self.edl_pass.ReleaseGraphicsResources(self.parent.ren_win)
        del self.edl_pass
        self.Modified()

    def enable_shadows(self):
        """Enable shadows.

        Examples
        --------
        First, plot without shadows enabled (default)

        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> pl = pyvista.Plotter(lighting='none', window_size=(1000, 1000))
        >>> light = pyvista.Light()
        >>> light.set_direction_angle(20, -20)
        >>> pl.add_light(light)
        >>> _ = pl.add_mesh(mesh, color='white', smooth_shading=True)
        >>> _ = pl.add_mesh(pyvista.Box((-1.2, -1, -1, 1, -1, 1)))
        >>> pl.show()

        Now, enable shadows.

        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> pl = pyvista.Plotter(lighting='none', window_size=(1000, 1000))
        >>> light = pyvista.Light()
        >>> light.set_direction_angle(20, -20)
        >>> pl.add_light(light)
        >>> _ = pl.add_mesh(mesh, color='white', smooth_shading=True)
        >>> _ = pl.add_mesh(pyvista.Box((-1.2, -1, -1, 1, -1, 1)))
        >>> pl.enable_shadows()
        >>> pl.show()

        """
        if self._shadow_pass is not None:
            # shadows are already enabled for this renderer
            return

        shadows = _vtk.vtkShadowMapPass()

        passes = _vtk.vtkRenderPassCollection()
        passes.AddItem(shadows.GetShadowMapBakerPass())
        passes.AddItem(shadows)

        seq = _vtk.vtkSequencePass()
        seq.SetPasses(passes)

        # Tell the renderer to use our render pass pipeline
        self._shadow_pass = _vtk.vtkCameraPass()
        self._shadow_pass.SetDelegatePass(seq)
        self.SetPass(self._shadow_pass)
        self.Modified()

    def disable_shadows(self):
        """Disable shadows.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.disable_shadows()

        """
        if self._shadow_pass is None:
            # shadows are already disabled
            return

        self.SetPass(None)
        if hasattr(self.parent, 'ren_win'):
            self._shadow_pass.ReleaseGraphicsResources(self.parent.ren_win)
        self._shadow_pass = None
        self.Modified()

    def get_pick_position(self):
        """Get the pick position/area as ``x0, y0, x1, y1``.

        Returns
        -------
        tuple
            Pick position as ``x0, y0, x1, y1``.

        """
        x0 = int(self.GetPickX1())
        x1 = int(self.GetPickX2())
        y0 = int(self.GetPickY1())
        y1 = int(self.GetPickY2())
        return x0, y0, x1, y1

    def set_background(self, color, top=None):
        """Set the background color of this renderer.

        Parameters
        ----------
        color : str or 3 item list, optional
            Either a string, rgb list, or hex color string.  Defaults
            to theme default.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1, 1, 1]``
            * ``color='#FFFFFF'``

        top : str or 3 item list, optional
            If given, this will enable a gradient background where the
            ``color`` argument is at the bottom and the color given in
            ``top`` will be the color at the top of the renderer.

        Examples
        --------
        Set the background color to black with a gradient to white at
        the top of the plot.

        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> actor = pl.add_mesh(pyvista.Cone())
        >>> pl.set_background('black', top='white')
        >>> pl.show()

        """
        if color is None:
            color = self._theme.background

        use_gradient = False
        if top is not None:
            use_gradient = True

        self.SetBackground(parse_color(color))
        if use_gradient:
            self.GradientBackgroundOn()
            self.SetBackground2(parse_color(top))
        else:
            self.GradientBackgroundOff()
        self.Modified()

    def set_environment_texture(self, texture):
        """Set the environment texture used for image based lighting.

        This texture is supposed to represent the scene background. If
        it is not a cubemap, the texture is supposed to represent an
        equirectangular projection. If used with raytracing backends,
        the texture must be an equirectangular projection and must be
        constructed with a valid vtkImageData. Warning, this texture
        must be expressed in linear color space. If the texture is in
        sRGB color space, set the color flag on the texture or set the
        argument isSRGB to true.

        Parameters
        ----------
        texture : vtk.vtkTexture
            Texture.
        """
        self.UseImageBasedLightingOn()
        self.SetEnvironmentTexture(texture)
        self.Modified()

    def close(self):
        """Close out widgets and sensitive elements."""
        self.RemoveAllObservers()
        if hasattr(self, 'axes_widget'):
            self.hide_axes()  # Necessary to avoid segfault
            self.axes_actor = None
            del self.axes_widget

        if self._empty_str is not None:
            self._empty_str.SetReferenceCount(0)
            self._empty_str = None

    def deep_clean(self, render=False):
        """Clean the renderer of the memory.

        Parameters
        ----------
        render : bool, optional
            Render the render window after removing the bounding box
            (if applicable).

        """
        if hasattr(self, 'cube_axes_actor'):
            del self.cube_axes_actor
        if hasattr(self, 'edl_pass'):
            del self.edl_pass
        if hasattr(self, '_box_object'):
            self.remove_bounding_box(render=render)
        if self._shadow_pass is not None:
            self.disable_shadows()
        if self.__charts is not None:
            self.__charts.deep_clean()

        self.remove_floors(render=render)
        self.remove_legend(render=render)
        self.RemoveAllViewProps()
        self._actors = {}
        self._camera = None
        self._bounding_box = None
        self._marker_actor = None
        self._border_actor = None
        # remove reference to parent last
        self.parent = None

    def __del__(self):
        """Delete the renderer."""
        self.deep_clean()

    def enable_hidden_line_removal(self):
        """Enable hidden line removal."""
        self.UseHiddenLineRemovalOn()

    def disable_hidden_line_removal(self):
        """Disable hidden line removal."""
        self.UseHiddenLineRemovalOff()

    @property
    def layer(self):
        """Return or set the current layer of this renderer."""
        return self.GetLayer()

    @layer.setter
    def layer(self, layer):
        self.SetLayer(layer)

    @property
    def viewport(self):
        """Viewport of the renderer.

        Viewport describes the ``(xstart, ystart, xend, yend)`` square
        of the renderer relative to the main renderer window.

        For example, a renderer taking up the entire window will have
        a viewport of ``(0.0, 0.0, 1.0, 1.0)``, while the viewport of
        a renderer on the left-hand side of a horizontally split window
        would be ``(0.0, 0.0, 0.5, 1.0)``.

        Returns
        -------
        tuple
            Viewport in the form ``(xstart, ystart, xend, yend)``.

        Examples
        --------
        Show the viewport of a renderer taking up half the render
        window.

        >>> import pyvista
        >>> pl = pyvista.Plotter(shape=(1, 2))
        >>> pl.renderers[0].viewport
        (0.0, 0.0, 0.5, 1.0)

        """
        return self.GetViewport()

    @property
    def width(self):
        """Width of the renderer."""
        xmin, _, xmax, _ = self.viewport
        return self.parent.window_size[0]*(xmax - xmin)

    @property
    def height(self):
        """Height of the renderer."""
        _, ymin, _, ymax = self.viewport
        return self.parent.window_size[1]*(ymax - ymin)

    def add_legend(self, labels=None, bcolor=(0.5, 0.5, 0.5),
                   border=False, size=(0.2, 0.2), name=None,
                   loc='upper right', face='triangle'):
        """Add a legend to render window.

        Entries must be a list containing one string and color entry for each
        item.

        Parameters
        ----------
        labels : list, optional
            When set to ``None``, uses existing labels as specified by

            - :func:`add_mesh <BasePlotter.add_mesh>`
            - :func:`add_lines <BasePlotter.add_lines>`
            - :func:`add_points <BasePlotter.add_points>`

            List containing one entry for each item to be added to the
            legend.  Each entry must contain two strings, [label,
            color], where label is the name of the item to add, and
            color is the color of the label to add.

        bcolor : list or str, optional
            Background color, either a three item 0 to 1 RGB color
            list, or a matplotlib color string (e.g. ``'w'`` or ``'white'``
            for a white color).  If None, legend background is
            disabled.

        border : bool, optional
            Controls if there will be a border around the legend.
            Default False.

        size : sequence, optional
            Two float sequence, each float between 0 and 1.  For example
            ``(0.1, 0.1)`` would make the legend 10% the size of the
            entire figure window.

        name : str, optional
            The name for the added actor so that it can be easily
            updated.  If an actor of this name already exists in the
            rendering window, it will be replaced by the new actor.

        loc : str, optional
            Location string.  One of the following:

            * ``'upper right'``
            * ``'upper left'``
            * ``'lower left'``
            * ``'lower right'``
            * ``'center left'``
            * ``'center right'``
            * ``'lower center'``
            * ``'upper center'``
            * ``'center'``

        face : str or pyvista.PolyData, optional
            Face shape of legend face.  One of the following:

            * None: ``None``
            * Line: ``"-"`` or ``"line"``
            * Triangle: ``"^"`` or ``'triangle'``
            * Circle: ``"o"`` or ``'circle'``
            * Rectangle: ``"r"`` or ``'rectangle'``
            * Custom: :class:`pyvista.PolyData`

            Default is ``'triangle'``.  Passing ``None`` removes the
            legend face.  A custom face can be created using
            :class:`pyvista.PolyData`.  This will be rendered from the
            XY plane.

        Returns
        -------
        vtk.vtkLegendBoxActor
            Actor for the legend.

        Examples
        --------
        Create a legend by labeling the meshes when using ``add_mesh``

        >>> import pyvista
        >>> from pyvista import examples
        >>> sphere = pyvista.Sphere(center=(0, 0, 1))
        >>> cube = pyvista.Cube()
        >>> plotter = pyvista.Plotter()
        >>> _ = plotter.add_mesh(sphere, 'grey', smooth_shading=True, label='Sphere')
        >>> _ = plotter.add_mesh(cube, 'r', label='Cube')
        >>> _ = plotter.add_legend(bcolor='w', face=None)
        >>> plotter.show()

        Alternatively provide labels in the plotter.

        >>> plotter = pyvista.Plotter()
        >>> _ = plotter.add_mesh(sphere, 'grey', smooth_shading=True)
        >>> _ = plotter.add_mesh(cube, 'r')
        >>> legend_entries = []
        >>> legend_entries.append(['My Mesh', 'w'])
        >>> legend_entries.append(['My Other Mesh', 'k'])
        >>> _ = plotter.add_legend(legend_entries)
        >>> plotter.show()

        """
        if self.legend is not None:
            self.remove_legend()
        self._legend = _vtk.vtkLegendBoxActor()

        if labels is None:
            # use existing labels
            if not self._labels:
                raise ValueError('No labels input.\n\n'
                                 'Add labels to individual items when adding them to'
                                 'the plotting object with the "label=" parameter.  '
                                 'or enter them as the "labels" parameter.')

            self._legend.SetNumberOfEntries(len(self._labels))
            for i, (vtk_object, text, color) in enumerate(self._labels.values()):

                if face is None:
                    # dummy vtk object
                    vtk_object = pyvista.PolyData([0.0, 0.0, 0.0])

                self._legend.SetEntry(i, vtk_object, text, parse_color(color))

        else:
            self._legend.SetNumberOfEntries(len(labels))

            legend_face = make_legend_face(face)
            for i, (text, color) in enumerate(labels):
                self._legend.SetEntry(i, legend_face, text, parse_color(color))

        if loc is not None:
            if loc not in ACTOR_LOC_MAP:
                allowed = '\n'.join([f'\t * "{item}"' for item in ACTOR_LOC_MAP])
                raise ValueError(
                    f'Invalid loc "{loc}".  Expected one of the following:\n{allowed}'
                )
            x, y, size = map_loc_to_pos(loc, size, border=0.05)
            self._legend.SetPosition(x, y)
            self._legend.SetPosition2(size[0], size[1])

        if bcolor is None:
            self._legend.UseBackgroundOff()
        else:
            self._legend.UseBackgroundOn()
            self._legend.SetBackgroundColor(parse_color(bcolor))

        self._legend.SetBorder(border)

        self.add_actor(self._legend, reset_camera=False, name=name, pickable=False)
        return self._legend

    def remove_legend(self, render=True):
        """Remove the legend actor.

        Parameters
        ----------
        render : bool, optional
            Render upon actor removal.  Set this to ``False`` to stop
            the render window from rendering when a the legend is removed.

        Examples
        --------
        >>> import pyvista
        >>> mesh = pyvista.Sphere()
        >>> pl = pyvista.Plotter()
        >>> _ = pl.add_mesh(mesh, label='sphere')
        >>> _ = pl.add_legend()
        >>> pl.remove_legend()

        """
        if self.legend is not None:
            self.remove_actor(self.legend, reset_camera=False, render=render)
            self._legend = None

    @property
    def legend(self):
        """Legend actor."""
        return self._legend


def _line_for_legend():
    """Create a simple line-like rectangle for the legend."""
    points = [
        [0, 0, 0],
        [0.4, 0, 0],
        [0.4, 0.07, 0],
        [0, 0.07, 0],
        [0.5, 0, 0],  # last point needed to expand the bounds of the PolyData to be rendered smaller
    ]
    legendface = pyvista.PolyData()
    legendface.points = np.array(points)
    legendface.faces = [4, 0, 1, 2, 3]
    return legendface
