"""Module containing pyvista implementation of vtkRenderer."""

import collections.abc
from functools import partial, wraps
from typing import Sequence, cast
import warnings

import numpy as np

import pyvista
from pyvista import MAX_N_COLOR_BARS
from pyvista.core._typing_core import BoundsLike
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.misc import assert_empty_kwargs, try_callback

from . import _vtk
from .actor import Actor
from .camera import Camera
from .charts import Charts
from .colors import Color, get_cycler
from .errors import InvalidCameraError
from .helpers import view_vectors
from .render_passes import RenderPasses
from .tools import create_axes_marker, create_axes_orientation_box, parse_font_family
from .utilities.gl_checks import check_depth_peeling, uses_egl

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
        raise ValueError(f'`size` must be a list of length 2. Passed value is {size}')

    if 'right' in loc:
        x = 1 - size[1] - border
    elif 'left' in loc:
        x = border
    else:
        x = 0.5 - size[1] / 2

    if 'upper' in loc:
        y = 1 - size[1] - border
    elif 'lower' in loc:
        y = border
    else:
        y = 0.5 - size[1] / 2

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
        raise ValueError(
            f'Invalid face "{face}".  Must be one of the following:\n'
            '\t"triangle"\n'
            '\t"circle"\n'
            '\t"rectangle"\n'
            '\tNone'
            '\tpyvista.PolyData'
        )
    return legendface


def scale_point(camera, point, invert=False):
    """Scale a point using the camera's transform matrix.

    Parameters
    ----------
    camera : Camera
        The camera who's matrix to use.

    point : sequence[float]
        Scale point coordinates.

    invert : bool, default: False
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
    """Container to hold camera location attributes.

    Parameters
    ----------
    position : sequence[float]
        Position of the camera.

    focal_point : sequence[float]
        The focal point of the camera.

    viewup : sequence[float]
        View up of the camera.

    """

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


class Renderer(_vtk.vtkOpenGLRenderer):
    """Renderer class."""

    # map camera_position string to an attribute
    CAMERA_STR_ATTR_MAP = {
        'xy': 'view_xy',
        'xz': 'view_xz',
        'yz': 'view_yz',
        'yx': 'view_yx',
        'zx': 'view_zx',
        'zy': 'view_zy',
        'iso': 'view_isometric',
    }

    def __init__(self, parent, border=True, border_color='w', border_width=2.0):
        """Initialize the renderer."""
        super().__init__()
        self._actors = {}
        self.parent = parent  # weakref.proxy to the plotter from Renderers
        self._theme = parent.theme
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
        self._render_passes = RenderPasses(self)
        self.cube_axes_actor = None

        # This is a private variable to keep track of how many colorbars exist
        # This allows us to keep adding colorbars without overlapping
        self._scalar_bar_slots = set(range(MAX_N_COLOR_BARS))
        self._scalar_bar_slot_lookup = {}
        self.__charts = None

        self._border_actor = None
        if border:
            self.add_border(border_color, border_width)

        self.set_color_cycler(self._theme.color_cycler)

    @property
    def camera_set(self) -> bool:
        """Get or set whether this camera has been configured."""
        if self.camera is None:  # pragma: no cover
            return False
        return self.camera.is_set

    @camera_set.setter
    def camera_set(self, is_set: bool):
        self.camera.is_set = is_set

    def set_color_cycler(self, color_cycler):
        """Set or reset this renderer's color cycler.

        This color cycler is iterated over by each sequential :class:`add_mesh() <pyvista.Plotter.add_mesh>`
        call to set the default color of the dataset being plotted.

        When setting, the value must be either a list of color-like objects,
        or a cycler of color-like objects. If the value passed is a single
        string, it must be one of:

            * ``'default'`` - Use the default color cycler (matches matplotlib's default)
            * ``'matplotlib`` - Dynamically get matplotlib's current theme's color cycler.
            * ``'all'`` - Cycle through all of the available colors in ``pyvista.plotting.colors.hexcolors``

        Setting to ``None`` will disable the use of the color cycler on this
        renderer.

        Parameters
        ----------
        color_cycler : str | cycler.Cycler | sequence[ColorLike]
            The colors to cycle through.

        Examples
        --------
        Set the default color cycler to iterate through red, green, and blue.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.renderer.set_color_cycler(['red', 'green', 'blue'])
        >>> _ = pl.add_mesh(pv.Cone(center=(0, 0, 0)))  # red
        >>> _ = pl.add_mesh(pv.Cube(center=(1, 0, 0)))  # green
        >>> _ = pl.add_mesh(pv.Sphere(center=(1, 1, 0)))  # blue
        >>> _ = pl.add_mesh(pv.Cylinder(center=(0, 1, 0)))  # red again
        >>> pl.show()

        """
        cycler = get_cycler(color_cycler)
        if cycler is not None:
            # Color cycler - call object to generate `cycle` instance
            self._color_cycle = cycler()
        else:
            self._color_cycle = None

    @property
    def next_color(self):
        """Return next color from this renderer's color cycler."""
        if self._color_cycle is None:
            return self._theme.color
        return next(self._color_cycle)['color']

    @property
    def _charts(self):
        """Return the charts collection."""
        # lazy instantiation here to avoid creating the charts object unless needed.
        if self.__charts is None:
            self.__charts = Charts(self)
            self.AddObserver("StartEvent", partial(try_callback, self._before_render_event))
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
            self.camera.up,
        )

    @camera_position.setter
    def camera_position(self, camera_location):
        """Set camera position of all active render windows."""
        if camera_location is None:
            return
        elif isinstance(camera_location, str):
            camera_location = camera_location.lower()
            if camera_location not in self.CAMERA_STR_ATTR_MAP:
                raise InvalidCameraError(
                    'Invalid view direction.  '
                    'Use one of the following:\n   '
                    f'{", ".join(self.CAMERA_STR_ATTR_MAP)}'
                )

            getattr(self, self.CAMERA_STR_ATTR_MAP[camera_location])()

        elif isinstance(camera_location[0], (int, float)):
            if len(camera_location) != 3:
                raise InvalidCameraError
            self.view_vector(camera_location)
        else:
            # check if a valid camera position
            if not isinstance(camera_location, CameraPosition):
                if not len(camera_location) == 3:
                    raise InvalidCameraError
                elif any([len(item) != 3 for item in camera_location]):
                    raise InvalidCameraError

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
            source.up,
        )
        self.Modified()
        self.camera_set = True

    @property
    def bounds(self) -> BoundsLike:
        """Return the bounds of all actors present in the rendering window."""
        the_bounds = np.array([np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf])

        def _update_bounds(bounds):
            def update_axis(ax):
                if bounds[ax * 2] < the_bounds[ax * 2]:
                    the_bounds[ax * 2] = bounds[ax * 2]
                if bounds[ax * 2 + 1] > the_bounds[ax * 2 + 1]:
                    the_bounds[ax * 2 + 1] = bounds[ax * 2 + 1]

            for ax in range(3):
                update_axis(ax)
            return

        for actor in self._actors.values():
            if isinstance(actor, (_vtk.vtkCubeAxesActor, _vtk.vtkLightActor)):
                continue
            if (
                hasattr(actor, 'GetBounds')
                and actor.GetBounds() is not None
                and id(actor) != id(self.bounding_box_actor)
            ):
                _update_bounds(actor.GetBounds())

        if np.any(np.abs(the_bounds)):
            the_bounds[the_bounds == np.inf] = -1.0
            the_bounds[the_bounds == -np.inf] = 1.0

        return cast(BoundsLike, tuple(the_bounds))

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
        x = (bounds[1] + bounds[0]) / 2
        y = (bounds[3] + bounds[2]) / 2
        z = (bounds[5] + bounds[4]) / 2
        return [x, y, z]

    @property
    def background_color(self):
        """Return the background color of this renderer."""
        return Color(self.GetBackground())

    @background_color.setter
    def background_color(self, color):
        """Set the background color of this renderer."""
        self.set_background(color)
        self.Modified()

    def _before_render_event(self, *args, **kwargs):
        """Notify all charts about render event."""
        for chart in self._charts:
            chart._render_event(*args, **kwargs)

    def enable_depth_peeling(self, number_of_peels=None, occlusion_ratio=None):
        """Enable depth peeling to improve rendering of translucent geometry.

        Parameters
        ----------
        number_of_peels : int, optional
            The maximum number of peeling layers. Initial value is 4
            and is set in the ``pyvista.global_theme``. A special value of
            0 means no maximum limit.  It has to be a positive value.

        occlusion_ratio : float, optional
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
        depth_peeling_supported = check_depth_peeling(number_of_peels, occlusion_ratio)
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

    def enable_anti_aliasing(self, aa_type='ssaa'):
        """Enable anti-aliasing.

        Parameters
        ----------
        aa_type : str, default: 'ssaa'
            Anti-aliasing type. Either ``"fxaa"`` or ``"ssaa"``.

        """
        if not isinstance(aa_type, str):
            raise TypeError(f'`aa_type` must be a string, not {type(aa_type)}')
        aa_type = aa_type.lower()

        if aa_type == 'fxaa':
            if uses_egl():  # pragma: no cover
                # only display the warning when not building documentation
                if not pyvista.BUILDING_GALLERY:
                    warnings.warn(
                        "VTK compiled with OSMesa/EGL does not properly support "
                        "FXAA anti-aliasing and SSAA will be used instead."
                    )
                self._render_passes.enable_ssaa_pass()
                return
            self._enable_fxaa()

        elif aa_type == 'ssaa':
            self._render_passes.enable_ssaa_pass()

        else:
            raise ValueError(f'Invalid `aa_type` "{aa_type}". Should be either "fxaa" or "ssaa"')

    def disable_anti_aliasing(self):
        """Disable all anti-aliasing."""
        self._render_passes.disable_ssaa_pass()
        self.SetUseFXAA(False)
        self.Modified()

    def _enable_fxaa(self):
        """Enable FXAA anti-aliasing."""
        self.SetUseFXAA(True)
        self.Modified()

    def _disable_fxaa(self):
        """Disable FXAA anti-aliasing."""
        self.SetUseFXAA(False)
        self.Modified()

    def add_border(self, color='white', width=2.0):
        """Add borders around the frame.

        Parameters
        ----------
        color : ColorLike, default: "white"
            Color of the border.

        width : float, default: 2.0
            Width of the border.

        Returns
        -------
        vtk.vtkActor2D
            Border actor.

        """
        points = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        lines = np.array([[2, 0, 1], [2, 1, 2], [2, 2, 3], [2, 3, 0]]).ravel()

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
        actor.GetProperty().SetColor(Color(color).float_rgb)
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
            return Color(self._border_actor.GetProperty().GetColor())
        return None

    def add_chart(self, chart, *charts):
        """Add a chart to this renderer.

        Parameters
        ----------
        chart : Chart
            Chart to add to renderer.

        *charts : Chart
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
        if _vtk.vtkRenderingContextOpenGL2 is None:  # pragma: no cover
            from pyvista.core.errors import VTKVersionError

            raise VTKVersionError(
                "VTK is missing vtkRenderingContextOpenGL2. Try installing VTK v9.1.0 or newer."
            )
        self._charts.add_chart(chart, *charts)

    @property
    def has_charts(self):
        """Return whether this renderer has charts."""
        return self.__charts is not None

    @wraps(Charts.set_interaction)
    def set_chart_interaction(self, interactive, toggle=False):
        """Wrap ``Charts.set_interaction``."""
        # Make sure we don't create the __charts object if this renderer has no charts yet.
        return self._charts.set_interaction(interactive, toggle) if self.has_charts else []

    @wraps(Charts.get_charts_by_pos)
    def _get_charts_by_pos(self, pos):
        """Wrap ``Charts.get_charts_by_pos``."""
        # Make sure we don't create the __charts object if this renderer has no charts yet.
        return self._charts.get_charts_by_pos(pos) if self.has_charts else []

    def remove_chart(self, chart_or_index):
        """Remove a chart from this renderer.

        Parameters
        ----------
        chart_or_index : Chart or int
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
        # Make sure we don't create the __charts object if this renderer has no charts yet.
        if self.has_charts:
            self._charts.remove_chart(chart_or_index)

    @property
    def actors(self):
        """Return a dictionary of actors assigned to this renderer."""
        return self._actors

    def add_actor(
        self,
        actor,
        reset_camera=False,
        name=None,
        culling=False,
        pickable=True,
        render=True,
        remove_existing_actor=True,
    ):
        """Add an actor to render window.

        Creates an actor if input is a mapper.

        Parameters
        ----------
        actor : vtk.vtkActor | vtk.vtkMapper | pyvista.Actor
            The actor to be added. Can be either ``vtkActor`` or ``vtkMapper``.

        reset_camera : bool, default: False
            Resets the camera when ``True``.

        name : str, optional
            Name to assign to the actor.  Defaults to the memory address.

        culling : str, default: False
            Does not render faces that are culled. Options are
            ``'front'`` or ``'back'``. This can be helpful for dense
            surface meshes, especially when edges are visible, but can
            cause flat meshes to be partially displayed.

        pickable : bool, default: True
            Whether to allow this actor to be pickable within the
            render window.

        render : bool, default: True
            If the render window is being shown, trigger a render
            after adding the actor.

        remove_existing_actor : bool, default: True
            Removes any existing actor if the named actor ``name`` is already
            present.

        Returns
        -------
        actor : vtk.vtkActor or pyvista.Actor
            The actor.

        actor_properties : vtk.Properties
            Actor properties.
        """
        # Remove actor by that name if present
        rv = None
        if name and remove_existing_actor:
            rv = self.remove_actor(name, reset_camera=False, render=False)

        if isinstance(actor, _vtk.vtkMapper):
            actor = Actor(mapper=actor, name=name)

        if isinstance(actor, Actor) and name:
            # WARNING: this will override the name if already set on Actor
            actor.name = name

        if name is None:
            if isinstance(actor, Actor):
                name = actor.name
            else:
                # Fallback for non-wrapped actors
                # e.g., vtkScalarBarActor
                name = actor.GetAddressAsString("")

        actor.SetPickable(pickable)
        # Apply this renderer's scale to the actor (which can be further scaled)
        if hasattr(actor, 'SetScale'):
            actor.SetScale(np.array(actor.GetScale()) * np.array(self.scale))
        self.AddActor(actor)  # must add actor before resetting camera
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

        self.Modified()

        prop = None
        if hasattr(actor, 'GetProperty'):
            prop = actor.GetProperty()

        return actor, prop

    def add_axes_at_origin(
        self,
        x_color=None,
        y_color=None,
        z_color=None,
        xlabel='X',
        ylabel='Y',
        zlabel='Z',
        line_width=2,
        labels_off=False,
    ):
        """Add axes actor at origin.

        Parameters
        ----------
        x_color : ColorLike, optional
            The color of the x axes arrow.

        y_color : ColorLike, optional
            The color of the y axes arrow.

        z_color : ColorLike, optional
            The color of the z axes arrow.

        xlabel : str, default: "X"
            The label of the x axes arrow.

        ylabel : str, default: "Y"
            The label of the y axes arrow.

        zlabel : str, default: "Z"
            The label of the z axes arrow.

        line_width : int, default: 2
            Width of the arrows.

        labels_off : bool, default: False
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
        self._marker_actor = create_axes_marker(
            line_width=line_width,
            x_color=x_color,
            y_color=y_color,
            z_color=z_color,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            labels_off=labels_off,
        )
        self.AddActor(self._marker_actor)
        memory_address = self._marker_actor.GetAddressAsString("")
        self._actors[memory_address] = self._marker_actor
        self.Modified()
        return self._marker_actor

    def add_orientation_widget(
        self, actor, interactive=None, color=None, opacity=1.0, viewport=None
    ):
        """Use the given actor in an orientation marker widget.

        Color and opacity are only valid arguments if a mesh is passed.

        Parameters
        ----------
        actor : vtk.vtkActor | pyvista.DataSet
            The mesh or actor to use as the marker.

        interactive : bool, optional
            Control if the orientation widget is interactive.  By
            default uses the value from
            :attr:`pyvista.global_theme.interactive
            <pyvista.plotting.themes.Theme.interactive>`.

        color : ColorLike, optional
            The color of the actor.  This only applies if ``actor`` is
            a :class:`pyvista.DataSet`.

        opacity : int | float, default: 1.0
            Opacity of the marker.

        viewport : sequence[float], optional
            Viewport ``(xstart, ystart, xend, yend)`` of the widget.

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
            actor = pyvista.Actor(mapper=mapper)
            if color is not None:
                actor.prop.color = color
            actor.prop.opacity = opacity
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
        if viewport is not None:
            self.axes_widget.SetViewport(viewport)
        self.Modified()
        return self.axes_widget

    def add_axes(
        self,
        interactive=None,
        line_width=2,
        color=None,
        x_color=None,
        y_color=None,
        z_color=None,
        xlabel='X',
        ylabel='Y',
        zlabel='Z',
        labels_off=False,
        box=None,
        box_args=None,
        viewport=(0, 0, 0.2, 0.2),
        marker_args=None,
        **kwargs,
    ):
        """Add an interactive axes widget in the bottom left corner.

        Parameters
        ----------
        interactive : bool, optional
            Enable this orientation widget to be moved by the user.

        line_width : int, default: 2
            The width of the marker lines.

        color : ColorLike, optional
            Color of the labels.

        x_color : ColorLike, optional
            Color used for the x axis arrow.  Defaults to theme axes parameters.

        y_color : ColorLike, optional
            Color used for the y axis arrow.  Defaults to theme axes parameters.

        z_color : ColorLike, optional
            Color used for the z axis arrow.  Defaults to theme axes parameters.

        xlabel : str, default: "X"
            Text used for the x axis.

        ylabel : str, default: "Y"
            Text used for the y axis.

        zlabel : str, default: "Z"
            Text used for the z axis.

        labels_off : bool, default: false
            Enable or disable the text labels for the axes.

        box : bool, optional
            Show a box orientation marker. Use ``box_args`` to adjust.
            See :func:`pyvista.create_axes_orientation_box` for details.

        box_args : dict, optional
            Parameters for the orientation box widget when
            ``box=True``. See the parameters of
            :func:`pyvista.create_axes_orientation_box`.

        viewport : sequence[float], default: (0, 0, 0.2, 0.2)
            Viewport ``(xstart, ystart, xend, yend)`` of the widget.

        marker_args : dict, optional
            Marker arguments.

            .. deprecated:: 0.37.0
               Use ``**kwargs`` for passing parameters for the orientation
               marker widget. See the parameters of
               :func:`pyvista.create_axes_marker`.

        **kwargs : dict, optional
            Used for passing parameters for the orientation marker
            widget. See the parameters of :func:`pyvista.create_axes_marker`.

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

        Specify more parameters for the axes marker.

        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> actor = pl.add_mesh(pyvista.Box(), show_edges=True)
        >>> _ = pl.add_axes(
        ...     line_width=5,
        ...     cone_radius=0.6,
        ...     shaft_length=0.7,
        ...     tip_length=0.3,
        ...     ambient=0.5,
        ...     label_size=(0.4, 0.16),
        ... )
        >>> pl.show()

        """
        # Deprecated on v0.37.0, estimated removal on v0.40.0
        if marker_args is not None:  # pragma: no cover
            warnings.warn(
                "Use of `marker_args` is deprecated. Use `**kwargs` instead.",
                PyVistaDeprecationWarning,
            )
            kwargs.update(marker_args)

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
                label_color=color,
                line_width=line_width,
                x_color=x_color,
                y_color=y_color,
                z_color=z_color,
                xlabel=xlabel,
                ylabel=ylabel,
                zlabel=zlabel,
                labels_off=labels_off,
                **box_args,
            )
        else:
            self.axes_actor = create_axes_marker(
                label_color=color,
                line_width=line_width,
                x_color=x_color,
                y_color=y_color,
                z_color=z_color,
                xlabel=xlabel,
                ylabel=ylabel,
                zlabel=zlabel,
                labels_off=labels_off,
                **kwargs,
            )
        axes_widget = self.add_orientation_widget(
            self.axes_actor, interactive=interactive, color=None
        )
        axes_widget.SetViewport(viewport)
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

    def show_bounds(
        self,
        mesh=None,
        bounds=None,
        axes_ranges=None,
        show_xaxis=True,
        show_yaxis=True,
        show_zaxis=True,
        show_xlabels=True,
        show_ylabels=True,
        show_zlabels=True,
        bold=True,
        font_size=None,
        font_family=None,
        color=None,
        xtitle='X Axis',
        ytitle='Y Axis',
        ztitle='Z Axis',
        n_xlabels=5,
        n_ylabels=5,
        n_zlabels=5,
        use_2d=False,
        grid=None,
        location='closest',
        ticks=None,
        all_edges=False,
        corner_factor=0.5,
        fmt=None,
        minor_ticks=False,
        padding=0.0,
        use_3d_text=True,
        render=None,
        **kwargs,
    ):
        """Add bounds axes.

        Shows the bounds of the most recent input mesh unless mesh is
        specified.

        Parameters
        ----------
        mesh : pyvista.DataSet | pyvista.MultiBlock, optional
            Input mesh to draw bounds axes around.

        bounds : sequence[float], optional
            Bounds to override mesh bounds in the form ``[xmin, xmax,
            ymin, ymax, zmin, zmax]``.

        axes_ranges : sequence[float], optional
            When set, these values override the values that are shown on the
            axes. This can be useful when plotting scaled datasets or if you wish
            to manually display different values. These values must be in the
            form:

            ``[xmin, xmax, ymin, ymax, zmin, zmax]``.

        show_xaxis : bool, default: True
            Makes X axis visible.

        show_yaxis : bool, default: True
            Makes Y axis visible.

        show_zaxis : bool, default: True
            Makes Z axis visible.

        show_xlabels : bool, default: True
            Shows X labels.

        show_ylabels : bool, default: True
            Shows Y labels.

        show_zlabels : bool, default: True
            Shows Z labels.

        bold : bool, default: True
            Bolds axis labels and numbers.

        font_size : float, optional
            Sets the size of the label font. Defaults to
            :attr:`pyvista.global_theme.font.size
            <pyvista.plotting.themes._Font.size>`.

        font_family : str, optional
            Font family.  Must be either ``'courier'``, ``'times'``,
            or ``'arial'``. Defaults to :attr:`pyvista.global_theme.font.family
            <pyvista.plotting.themes._Font.family>`.

        color : ColorLike, optional
            Color of all labels and axis titles.  Defaults to
            :attr:`pyvista.global_theme.font.color
            <pyvista.plotting.themes._Font.color>`.

            Either a string, RGB list, or hex color string.  For
            example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

        xtitle : str, default: "X Axis"
            Title of the X axis.  Default ``"X Axis"``.

        ytitle : str, default: "Y Axis"
            Title of the Y axis.  Default ``"Y Axis"``.

        ztitle : str, default: "Z Axis"
            Title of the Z axis.  Default ``"Z Axis"``.

        n_xlabels : int, default: 5
            Number of labels for the X axis.

        n_ylabels : int, default: 5
            Number of labels for the Y axis.

        n_zlabels : int, default: 5
            Number of labels for the Z axis.

        use_2d : bool, default: False
            This can be enabled for smoother plotting.

        grid : bool or str, optional
            Add grid lines to the backface (``True``, ``'back'``, or
            ``'backface'``) or to the frontface (``'front'``,
            ``'frontface'``) of the axes actor.

        location : str, default: "closest"
            Set how the axes are drawn: either static (``'all'``), closest
            triad (``'front'``, ``'closest'``, ``'default'``), furthest triad
            (``'back'``, ``'furthest'``), static closest to the origin
            (``'origin'``), or outer edges (``'outer'``) in relation to the
            camera position.

        ticks : str, optional
            Set how the ticks are drawn on the axes grid. Options include:
            ``'inside', 'outside', 'both'``.

        all_edges : bool, default: False
            Adds an unlabeled and unticked box at the boundaries of
            plot. Useful for when wanting to plot outer grids while
            still retaining all edges of the boundary.

        corner_factor : float, default: 0.5
            If ``all_edges``, this is the factor along each axis to
            draw the default box. Default shows the full box.

        fmt : str, optional
            A format string defining how tick labels are generated from
            tick positions. A default is looked up on the active theme.

        minor_ticks : bool, default: False
            If ``True``, also plot minor ticks on all axes.

        padding : float, default: 0.0
            An optional percent padding along each axial direction to
            cushion the datasets in the scene from the axes
            annotations. Defaults no padding.

        use_3d_text : bool, default: True
            Use ``vtkTextActor3D`` for titles and labels.

        render : bool, optional
            If the render window is being shown, trigger a render
            after showing bounds.

        **kwargs : dict, optional
            Deprecated keyword arguments.

        Returns
        -------
        vtk.vtkCubeAxesActor
            Bounds actor.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import examples

        >>> mesh = pv.Sphere()
        >>> plotter = pv.Plotter()
        >>> actor = plotter.add_mesh(mesh)
        >>> actor = plotter.show_bounds(
        ...     grid='front',
        ...     location='outer',
        ...     all_edges=True,
        ... )
        >>> plotter.show()

        Control how many labels are displayed.

        >>> mesh = examples.load_random_hills()

        >>> plotter = pv.Plotter()
        >>> actor = plotter.add_mesh(
        ...     mesh, cmap='terrain', show_scalar_bar=False
        ... )
        >>> actor = plotter.show_bounds(
        ...     grid='back',
        ...     location='outer',
        ...     ticks='both',
        ...     n_xlabels=2,
        ...     n_ylabels=2,
        ...     n_zlabels=2,
        ...     xtitle='Easting',
        ...     ytitle='Northing',
        ...     ztitle='Elevation',
        ... )
        >>> plotter.show()

        Hide labels, but still show axis titles.

        >>> plotter = pv.Plotter()
        >>> actor = plotter.add_mesh(
        ...     mesh, cmap='terrain', show_scalar_bar=False
        ... )
        >>> actor = plotter.show_bounds(
        ...     grid='back',
        ...     location='outer',
        ...     ticks='both',
        ...     show_xlabels=False,
        ...     show_ylabels=False,
        ...     show_zlabels=False,
        ...     xtitle='Easting',
        ...     ytitle='Northing',
        ...     ztitle='Elevation',
        ... )
        >>> plotter.show()

        """
        self.remove_bounds_axes()

        if font_family is None:
            font_family = self._theme.font.family
        if font_size is None:
            font_size = self._theme.font.size
        if fmt is None:
            fmt = self._theme.font.fmt
        if fmt is None:
            fmt = '%.1f'  # fallback

        if 'xlabel' in kwargs:  # pragma: no cover
            xtitle = kwargs.pop('xlabel')
            warnings.warn(
                "`xlabel` is deprecated. Use `xtitle` instead.",
                PyVistaDeprecationWarning,
            )
        if 'ylabel' in kwargs:  # pragma: no cover
            ytitle = kwargs.pop('ylabel')
            warnings.warn(
                "`ylabel` is deprecated. Use `ytitle` instead.",
                PyVistaDeprecationWarning,
            )
        if 'zlabel' in kwargs:  # pragma: no cover
            ztitle = kwargs.pop('zlabel')
            warnings.warn(
                "`zlabel` is deprecated. Use `ztitle` instead.",
                PyVistaDeprecationWarning,
            )
        assert_empty_kwargs(**kwargs)

        color = Color(color, default_color=self._theme.font.color)

        if mesh is None and bounds is None:
            # Use the bounds of all data in the rendering window
            bounds = np.array(self.bounds)
        elif bounds is None:
            # otherwise, use the bounds of the mesh (if available)
            bounds = np.array(mesh.bounds)
        else:
            bounds = np.asanyarray(bounds, dtype=float)

        # create actor
        cube_axes_actor = pyvista.CubeAxesActor(
            self.camera,
            minor_ticks=minor_ticks,
            tick_location=ticks,
            x_title=xtitle,
            y_title=ytitle,
            z_title=ztitle,
            x_axis_visibility=show_xaxis,
            y_axis_visibility=show_yaxis,
            z_axis_visibility=show_zaxis,
            x_label_format=fmt,
            y_label_format=fmt,
            z_label_format=fmt,
            x_label_visibility=show_xlabels,
            y_label_visibility=show_ylabels,
            z_label_visibility=show_zlabels,
            n_xlabels=n_xlabels,
            n_ylabels=n_ylabels,
            n_zlabels=n_zlabels,
        )

        cube_axes_actor.use_2d_mode = use_2d or not np.allclose(self.scale, [1.0, 1.0, 1.0])

        if grid:
            grid = 'back' if grid is True else grid
            if not isinstance(grid, str):
                raise TypeError(f'`grid` must be a str, not {type(grid)}')
            grid = grid.lower()
            if grid in ('front', 'frontface'):
                cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_CLOSEST)
            elif grid in ('both', 'all'):
                cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_ALL)
            elif grid in ('back', True):
                cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_FURTHEST)
            else:
                raise ValueError(f'`grid` must be either "front", "back, or, "all", not {grid}')
            # Only show user desired grid lines
            cube_axes_actor.SetDrawXGridlines(show_xaxis)
            cube_axes_actor.SetDrawYGridlines(show_yaxis)
            cube_axes_actor.SetDrawZGridlines(show_zaxis)
            # Set the colors
            cube_axes_actor.GetXAxesGridlinesProperty().SetColor(color.float_rgb)
            cube_axes_actor.GetYAxesGridlinesProperty().SetColor(color.float_rgb)
            cube_axes_actor.GetZAxesGridlinesProperty().SetColor(color.float_rgb)

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
                raise ValueError(
                    f'Value of location ("{location}") should be either "all", "origin",'
                    ' "outer", "default", "closest", "front", "furthest", or "back".'
                )
        elif location is not None:
            raise TypeError('location must be a string')

        if isinstance(padding, (int, float)) and 0.0 <= padding < 1.0:
            if not np.any(np.abs(bounds) == np.inf):
                cushion = (
                    np.array(
                        [
                            np.abs(bounds[1] - bounds[0]),
                            np.abs(bounds[3] - bounds[2]),
                            np.abs(bounds[5] - bounds[4]),
                        ]
                    )
                    * padding
                )
                bounds[::2] -= cushion
                bounds[1::2] += cushion
        else:
            raise ValueError(f'padding ({padding}) not understood. Must be float between 0 and 1')
        cube_axes_actor.bounds = bounds

        # set axes ranges if input
        if axes_ranges is not None:
            if isinstance(axes_ranges, (collections.abc.Sequence, np.ndarray)):
                axes_ranges = np.asanyarray(axes_ranges)
            else:
                raise TypeError('Input axes_ranges must be a numeric sequence.')

            if not np.issubdtype(axes_ranges.dtype, np.number):
                raise TypeError('All of the elements of axes_ranges must be numbers.')

            # set the axes ranges
            if axes_ranges.shape != (6,):
                raise ValueError(
                    '`axes_ranges` must be passed as a [xmin, xmax, ymin, ymax, zmin, zmax] sequence.'
                )

            cube_axes_actor.x_axis_range = axes_ranges[0], axes_ranges[1]
            cube_axes_actor.y_axis_range = axes_ranges[2], axes_ranges[3]
            cube_axes_actor.z_axis_range = axes_ranges[4], axes_ranges[5]

        # set color
        cube_axes_actor.GetXAxesLinesProperty().SetColor(color.float_rgb)
        cube_axes_actor.GetYAxesLinesProperty().SetColor(color.float_rgb)
        cube_axes_actor.GetZAxesLinesProperty().SetColor(color.float_rgb)

        # set font
        font_family = parse_font_family(font_family)

        if not use_3d_text or not np.allclose(self.scale, [1.0, 1.0, 1.0]):
            use_3d_text = False
            cube_axes_actor.SetUseTextActor3D(False)
        else:
            cube_axes_actor.SetUseTextActor3D(True)

        props = [
            cube_axes_actor.GetTitleTextProperty(0),
            cube_axes_actor.GetTitleTextProperty(1),
            cube_axes_actor.GetTitleTextProperty(2),
            cube_axes_actor.GetLabelTextProperty(0),
            cube_axes_actor.GetLabelTextProperty(1),
            cube_axes_actor.GetLabelTextProperty(2),
        ]

        for prop in props:
            prop.SetColor(color.float_rgb)
            prop.SetFontFamily(font_family)
            prop.SetBold(bold)

            # this merely makes the font sharper
            if use_3d_text:
                prop.SetFontSize(50)

        # Note: font_size does nothing as a property, use SetScreenSize instead
        # Here, we normalize relative to 12 to give the user an illusion of
        # just changing the font size relative to a font size of 12. 10 is used
        # here since it's the default "screen size".
        cube_axes_actor.SetScreenSize(font_size / 12 * 10.0)

        self.add_actor(cube_axes_actor, reset_camera=False, pickable=False, render=render)
        self.cube_axes_actor = cube_axes_actor

        if all_edges:
            self.add_bounding_box(color=color, corner_factor=corner_factor)

        self.Modified()
        return cube_axes_actor

    def show_grid(self, **kwargs):
        """Show grid lines and bounds axes labels.

        A wrapped implementation of :func:`show_bounds()
        <pyvista.Renderer.show_bounds>` to change default behavior to use
        grid lines and showing the axes labels on the outer edges.

        This is intended to be similar to :func:`matplotlib.pyplot.grid`.

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
        >>> import pyvista as pv
        >>> mesh = pv.Cone()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh)
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
        render : bool, default: True
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

    def add_bounding_box(
        self,
        color="grey",
        corner_factor=0.5,
        line_width=None,
        opacity=1.0,
        render_lines_as_tubes=False,
        lighting=None,
        reset_camera=None,
        outline=True,
        culling='front',
    ):
        """Add an unlabeled and unticked box at the boundaries of plot.

        Useful for when wanting to plot outer grids while still
        retaining all edges of the boundary.

        Parameters
        ----------
        color : ColorLike, default: "grey"
            Color of all labels and axis titles.  Default white.
            Either a string, rgb sequence, or hex color string.  For
            example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

        corner_factor : float, default: 0.5
            This is the factor along each axis to draw the default
            box. Default is 0.5 to show the full box.

        line_width : float, optional
            Thickness of lines.

        opacity : float, default: 1.0
            Opacity of mesh. Should be between 0 and 1.

        render_lines_as_tubes : bool, default: False
            Show lines as thick tubes rather than flat lines.  Control
            the width with ``line_width``.

        lighting : bool, optional
            Enable or disable directional lighting for this actor.

        reset_camera : bool, optional
            Reset camera position when ``True`` to include all actors.

        outline : bool, default: True
            Default is ``True``. when ``False``, a box with faces is
            shown with the specified culling.

        culling : str, default: "front"
            Does not render faces on the bounding box that are culled. Options
            are ``'front'`` or ``'back'``.

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
        self.bounding_box_actor, prop = self.add_actor(
            mapper, reset_camera=reset_camera, name=name, culling=culling, pickable=False
        )

        prop.SetColor(Color(color, default_color=self._theme.outline_color).float_rgb)
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

    def add_floor(
        self,
        face='-z',
        i_resolution=10,
        j_resolution=10,
        color=None,
        line_width=None,
        opacity=1.0,
        show_edges=False,
        lighting=False,
        edge_color=None,
        reset_camera=None,
        pad=0.0,
        offset=0.0,
        pickable=False,
        store_floor_kwargs=True,
    ):
        """Show a floor mesh.

        This generates planes at the boundaries of the scene to behave
        like floors or walls.

        Parameters
        ----------
        face : str, default: "-z"
            The face at which to place the plane. Options are
            (``'-z'``, ``'-y'``, ``'-x'``, ``'+z'``, ``'+y'``, and
            ``'+z'``). Where the ``-/+`` sign indicates on which side of
            the axis the plane will lie.  For example, ``'-z'`` would
            generate a floor on the XY-plane and the bottom of the
            scene (minimum z).

        i_resolution : int, default: 10
            Number of points on the plane in the i direction.

        j_resolution : int, default: 10
            Number of points on the plane in the j direction.

        color : ColorLike, optional
            Color of all labels and axis titles.  Default gray.
            Either a string, rgb list, or hex color string.

        line_width : int, optional
            Thickness of the edges. Only if ``show_edges`` is
            ``True``.

        opacity : float, default: 1.0
            The opacity of the generated surface.

        show_edges : bool, default: False
            Flag on whether to show the mesh edges for tiling.

        line_width : float, default: False
            Thickness of lines.  Only valid for wireframe and surface
            representations.

        lighting : bool, default: False
            Enable or disable view direction lighting.

        edge_color : ColorLike, optional
            Color of the edges of the mesh.

        reset_camera : bool, optional
            Resets the camera when ``True`` after adding the floor.

        pad : float, default: 0.0
            Percentage padding between 0 and 1.

        offset : float, default: 0.0
            Percentage offset along plane normal.

        pickable : bool, default: false
            Make this floor actor pickable in the renderer.

        store_floor_kwargs : bool, default: True
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
        ranges += ranges * pad
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
        self._floor = pyvista.Plane(
            center=center,
            direction=normal,
            i_size=i_size,
            j_size=j_size,
            i_resolution=i_resolution,
            j_resolution=j_resolution,
        )
        self._floor.clear_data()

        if lighting is None:
            lighting = self._theme.lighting

        self.remove_bounding_box()
        mapper = _vtk.vtkDataSetMapper()
        mapper.SetInputData(self._floor)
        actor, prop = self.add_actor(
            mapper, reset_camera=reset_camera, name=f'Floor({face})', pickable=pickable
        )

        prop.SetColor(Color(color, default_color=self._theme.floor_color).float_rgb)
        prop.SetOpacity(opacity)

        # edge display style
        if show_edges:
            prop.EdgeVisibilityOn()
        prop.SetEdgeColor(Color(edge_color, default_color=self._theme.edge_color).float_rgb)

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
        clear_kwargs : bool, default: True
            Clear default floor arguments.

        render : bool, default: True
            Render upon removing the floor.

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
        if self.cube_axes_actor is not None:
            self.remove_actor(self.cube_axes_actor)
            self.cube_axes_actor = None
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
        >>> pl.renderer.lights  # doctest:+SKIP
        [<Light (Headlight) at ...>,
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

    def clear_actors(self):
        """Remove all actors (keep lights and properties)."""
        if self._actors:
            for actor in list(self._actors):
                try:
                    self.remove_actor(actor, reset_camera=False, render=False)
                except KeyError:
                    pass
            self.Modified()

    def clear(self):
        """Remove all actors and properties."""
        self.clear_actors()
        if self.__charts is not None:
            self._charts.deep_clean()
        self.remove_all_lights()
        self.RemoveAllViewProps()
        self.Modified()

        self._scalar_bar_slots = set(range(MAX_N_COLOR_BARS))
        self._scalar_bar_slot_lookup = {}

    def set_focus(self, point):
        """Set focus to a point.

        Parameters
        ----------
        point : sequence[float]
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

    def set_position(self, point, reset=False, render=True):
        """Set camera position to a point.

        Parameters
        ----------
        point : sequence
            Cartesian point to focus on in the form of ``[x, y, z]``.

        reset : bool, default: False
            Whether to reset the camera after setting the camera
            position.

        render : bool, default: True
            If the render window is being shown, trigger a render
            after setting the position.

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
            self.reset_camera(render=render)
        self.camera_set = True
        self.Modified()

    def set_viewup(self, vector, reset=True, render=True):
        """Set camera viewup vector.

        Parameters
        ----------
        vector : sequence[float]
            New camera viewup vector.

        reset : bool, default: True
            Whether to reset the camera after setting the camera
            position.

        render : bool, default: True
            If the render window is being shown, trigger a render
            after setting the viewup.

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
        if reset:
            self.reset_camera(render=render)

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
        self.camera.parallel_scale = self.camera.distance * np.sin(0.5 * angle)

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
        distance = self.camera.parallel_scale / np.sin(0.5 * angle)
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
        try:
            self.parent.scalar_bars._remove_mapper_from_plotter(actor)
        except (AttributeError, ReferenceError):
            pass
        self.RemoveActor(actor)

        if name is None:
            for k, v in self._actors.items():
                if v == actor:
                    name = k
        self._actors.pop(name, None)
        self.update_bounds_axes()
        if reset_camera:
            self.reset_camera(render=render)
        elif not self.camera_set and reset_camera is None:
            self.reset_camera(render=render)
        elif render:
            self.parent.render()

        self.Modified()
        return True

    def set_scale(self, xscale=None, yscale=None, zscale=None, reset_camera=True, render=True):
        """Scale all the actors in the scene.

        Scaling in performed independently on the X, Y and Z axis.
        A scale of zero is illegal and will be replaced with one.

        .. warning::
            Setting the scale on the renderer is a convenience method to
            individually scale each of the actors in the scene. If a scale
            was set on an actor previously, it will be reset to the scale
            of this Renderer.

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

        reset_camera : bool, default: True
            Resets camera so all actors can be seen.

        render : bool, default: True
            If the render window is being shown, trigger a render
            after setting the scale.

        Examples
        --------
        Set the scale in the z direction to be 2 times that of
        nominal.  Leave the other axes unscaled.

        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.set_scale(zscale=2)
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

        # Reset all actors to match this scale
        for actor in self.actors.values():
            if hasattr(actor, 'SetScale'):
                actor.SetScale(self.scale)

        self.parent.render()
        if reset_camera:
            self.update_bounds_axes()
            self.reset_camera(render=render)
        self.Modified()

    def get_default_cam_pos(self, negative=False):
        """Return the default focal points and viewup.

        Uses ResetCamera to make a useful view.

        Parameters
        ----------
        negative : bool, default: False
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
        cpos = [position + np.array(focal_pt), focal_pt, self._theme.camera['viewup']]
        return cpos

    def update_bounds_axes(self):
        """Update the bounds axes of the render window."""
        if (
            hasattr(self, '_box_object')
            and self._box_object is not None
            and self.bounding_box_actor is not None
        ):
            if not np.allclose(self._box_object.bounds, self.bounds):
                color = self.bounding_box_actor.GetProperty().GetColor()
                self.remove_bounding_box()
                self.add_bounding_box(color=color)
                self.remove_floors(clear_kwargs=False)
                for floor_kwargs in self._floor_kwargs:
                    floor_kwargs['store_floor_kwargs'] = False
                    self.add_floor(**floor_kwargs)
        if self.cube_axes_actor is not None:
            self.cube_axes_actor.update_bounds(self.bounds)
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
        render : bool, default: True
            Trigger a render after resetting the camera.

        bounds : iterable(int), optional
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

        self.reset_camera_clipping_range()

        if render:
            self.parent.render()
        self.Modified()

    def isometric_view(self):
        """Reset the camera to a default isometric view.

        DEPRECATED: Please use ``view_isometric``.

        """
        self.view_isometric()

    def view_isometric(self, negative=False, render=True):
        """Reset the camera to a default isometric view.

        The view will show all the actors in the scene.

        Parameters
        ----------
        negative : bool, default: False
            View from the other isometric direction.

        render : bool, default: True
            If the render window is being shown, trigger a render
            after setting the camera position.

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
        self.reset_camera(render=render)

    def view_vector(self, vector, viewup=None, render=True):
        """Point the camera in the direction of the given vector.

        Parameters
        ----------
        vector : sequence[float]
            Direction to point the camera in.

        viewup : sequence[float], optional
            Sequence describing the view up of the camera.

        render : bool, default: True
            If the render window is being shown, trigger a render
            after setting the camera position.

        """
        focal_pt = self.center
        if viewup is None:
            viewup = self._theme.camera['viewup']
        cpos = CameraPosition(vector + np.array(focal_pt), focal_pt, viewup)
        self.camera_position = cpos
        self.reset_camera(render=render)

    def view_xy(self, negative=False, render=True):
        """View the XY plane.

        Parameters
        ----------
        negative : bool, default: False
            View from the opposite direction.

        render : bool, default: True
            If the render window is being shown, trigger a render
            after setting the camera position.

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
        self.view_vector(*view_vectors('xy', negative=negative), render=render)

    def view_yx(self, negative=False, render=True):
        """View the YX plane.

        Parameters
        ----------
        negative : bool, default: False
            View from the opposite direction.

        render : bool, default: True
            If the render window is being shown, trigger a render
            after setting the camera position.

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
        self.view_vector(*view_vectors('yx', negative=negative), render=render)

    def view_xz(self, negative=False, render=True):
        """View the XZ plane.

        Parameters
        ----------
        negative : bool, default: False
            View from the opposite direction.

        render : bool, default: True
            If the render window is being shown, trigger a render
            after setting the camera position.

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
        self.view_vector(*view_vectors('xz', negative=negative), render=render)

    def view_zx(self, negative=False, render=True):
        """View the ZX plane.

        Parameters
        ----------
        negative : bool, default: False
            View from the opposite direction.

        render : bool, default: True
            If the render window is being shown, trigger a render
            after setting the camera position.

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
        self.view_vector(*view_vectors('zx', negative=negative), render=render)

    def view_yz(self, negative=False, render=True):
        """View the YZ plane.

        Parameters
        ----------
        negative : bool, default: False
            View from the opposite direction.

        render : bool, default: True
            If the render window is being shown, trigger a render
            after setting the camera position.

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
        self.view_vector(*view_vectors('yz', negative=negative), render=render)

    def view_zy(self, negative=False, render=True):
        """View the ZY plane.

        Parameters
        ----------
        negative : bool, default: False
            View from the opposite direction.

        render : bool, default: True
            If the render window is being shown, trigger a render
            after setting the camera position.

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
        self.view_vector(*view_vectors('zy', negative=negative), render=render)

    def disable(self):
        """Disable this renderer's camera from being interactive."""
        self.SetInteractive(0)

    def enable(self):
        """Enable this renderer's camera to be interactive."""
        self.SetInteractive(1)

    def add_blurring(self):
        """Add blurring.

        This can be added several times to increase the degree of blurring.

        Examples
        --------
        Add two blurring passes to the plotter and show it.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Sphere(), show_edges=True)
        >>> pl.add_blurring()
        >>> pl.add_blurring()
        >>> pl.show()

        See :ref:`blur_example` for a full example using this method.

        """
        self._render_passes.add_blur_pass()

    def remove_blurring(self):
        """Remove a single blurring pass.

        You will need to run this multiple times to remove all blurring passes.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Sphere())
        >>> pl.add_blurring()
        >>> pl.remove_blurring()
        >>> pl.show()

        """
        self._render_passes.remove_blur_pass()

    def enable_depth_of_field(self, automatic_focal_distance=True):
        """Enable depth of field plotting.

        Parameters
        ----------
        automatic_focal_distance : bool, default: True
            Use automatic focal distance calculation. When enabled, the center
            of the viewport will always be in focus regardless of where the
            focal point is.

        Examples
        --------
        Create five spheres and demonstrate the effect of depth of field.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> pl = pv.Plotter(lighting="three lights")
        >>> pl.background_color = "w"
        >>> for i in range(5):
        ...     mesh = pv.Sphere(center=(-i * 4, 0, 0))
        ...     color = [0, 255 - i * 20, 30 + i * 50]
        ...     _ = pl.add_mesh(
        ...         mesh,
        ...         show_edges=False,
        ...         pbr=True,
        ...         metallic=1.0,
        ...         color=color,
        ...     )
        ...
        >>> pl.camera.zoom(1.8)
        >>> pl.camera_position = [
        ...     (4.74, 0.959, 0.525),
        ...     (0.363, 0.3116, 0.132),
        ...     (-0.088, -0.0075, 0.996),
        ... ]
        >>> pl.enable_depth_of_field()
        >>> pl.show()

        See :ref:`depth_of_field_example` for a full example using this method.

        """
        self._render_passes.enable_depth_of_field_pass(automatic_focal_distance)

    def disable_depth_of_field(self):
        """Disable depth of field plotting.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter(lighting="three lights")
        >>> pl.enable_depth_of_field()
        >>> pl.disable_depth_of_field()

        """
        self._render_passes.disable_depth_of_field_pass()

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
        self._render_passes.enable_edl_pass()

    def disable_eye_dome_lighting(self):
        """Disable eye dome lighting (EDL).

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.disable_eye_dome_lighting()

        """
        self._render_passes.disable_edl_pass()

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
        self._render_passes.enable_shadow_pass()

    def disable_shadows(self):
        """Disable shadows.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.disable_shadows()

        """
        self._render_passes.disable_shadow_pass()

    def enable_ssao(self, radius=0.5, bias=0.005, kernel_size=256, blur=True):
        """Enable surface space ambient occlusion (SSAO).

        SSAO can approximate shadows more efficiently than ray-tracing
        and produce similar results. Use this when you wish to plot the
        occlusion effect that nearby meshes have on each other by blocking
        nearby light sources.

        See `Kitware: Screen-Space Ambient Occlusion
        <https://www.kitware.com/ssao/>`_ for more details

        Parameters
        ----------
        radius : float, default: 0.5
            Neighbor pixels considered when computing the occlusion.

        bias : float, default: 0.005
            Tolerance factor used when comparing pixel depth.

        kernel_size : int, default: 256
            Number of samples used. This controls the quality where a higher
            number increases the quality at the expense of computation time.

        blur : bool, default: True
            Controls if occlusion buffer should be blurred before combining it
            with the color buffer.

        Examples
        --------
        Generate a :class:`pyvista.UnstructuredGrid` with many tetrahedrons
        nearby each other and plot it without SSAO.

        >>> import pyvista as pv
        >>> ugrid = pv.ImageData(dimensions=(3, 2, 2)).to_tetrahedra(12)
        >>> exploded = ugrid.explode()
        >>> exploded.plot()

        Enable SSAO with the default parameters.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(exploded)
        >>> pl.enable_ssao()
        >>> pl.show()

        """
        self._render_passes.enable_ssao_pass(radius, bias, kernel_size, blur)

    def disable_ssao(self):
        """Disable surface space ambient occlusion (SSAO)."""
        self._render_passes.disable_ssao_pass()

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
        color : ColorLike, optional
            Either a string, rgb list, or hex color string.  Defaults
            to theme default.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1.0, 1.0, 1.0]``
            * ``color='#FFFFFF'``

        top : ColorLike, optional
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
        use_gradient = False
        if top is not None:
            use_gradient = True

        self.SetBackground(Color(color, default_color=self._theme.background).float_rgb)
        if use_gradient:
            self.SetGradientBackground(True)
            self.SetBackground2(Color(top).float_rgb)
        else:
            self.SetGradientBackground(False)
        self.Modified()

    def set_environment_texture(self, texture, is_srgb=False):
        """Set the environment texture used for image based lighting.

        This texture is supposed to represent the scene background. If
        it is not a cubemap, the texture is supposed to represent an
        equirectangular projection. If used with raytracing backends,
        the texture must be an equirectangular projection and must be
        constructed with a valid ``vtk.vtkImageData``.

        Parameters
        ----------
        texture : pyvista.Texture
            Texture.

        is_srgb : bool, default: False
            If the texture is in sRGB color space, set the color flag on the
            texture or set this parameter to ``True``. Textures are assumed
            to be in linear color space by default.

        Examples
        --------
        Add a skybox cubemap as an environment texture and show that the
        lighting from the texture is mapped on to a sphere dataset. Note how
        even when disabling the default lightkit, the scene lighting will still
        be mapped onto the actor.

        >>> from pyvista import examples
        >>> import pyvista as pv
        >>> pl = pv.Plotter(lighting=None)
        >>> cubemap = examples.download_sky_box_cube_map()
        >>> _ = pl.add_mesh(
        ...     pv.Sphere(), pbr=True, metallic=0.9, roughness=0.4
        ... )
        >>> pl.set_environment_texture(cubemap)
        >>> pl.camera_position = 'xy'
        >>> pl.show()

        """
        # cube_map textures cannot use spherical harmonics
        if texture.cube_map:
            self.AutomaticLightCreationOff()
            # disable spherical harmonics was added in 9.1.0
            if hasattr(self, 'UseSphericalHarmonicsOff'):
                self.UseSphericalHarmonicsOff()

        self.UseImageBasedLightingOn()
        self.SetEnvironmentTexture(texture, is_srgb)
        self.Modified()

    def remove_environment_texture(self):
        """Remove the environment texture.

        Examples
        --------
        >>> from pyvista import examples
        >>> import pyvista as pv
        >>> pl = pv.Plotter(lighting=None)
        >>> cubemap = examples.download_sky_box_cube_map()
        >>> _ = pl.add_mesh(
        ...     pv.Sphere(), pbr=True, metallic=0.9, roughness=0.4
        ... )
        >>> pl.set_environment_texture(cubemap)
        >>> pl.remove_environment_texture()
        >>> pl.camera_position = 'xy'
        >>> pl.show()

        """
        self.UseImageBasedLightingOff()
        self.SetEnvironmentTexture(None)
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

    def on_plotter_render(self):
        """Notify renderer components of explicit plotter render call."""
        if self.__charts is not None:
            for chart in self.__charts:
                # Notify Charts that plotter.render() is called
                chart._render_event(plotter_render=True)

    def deep_clean(self, render=False):
        """Clean the renderer of the memory.

        Parameters
        ----------
        render : bool, optional
            Render the render window after removing the bounding box
            (if applicable).

        """
        if self.cube_axes_actor is not None:
            self.cube_axes_actor = None

        if hasattr(self, 'edl_pass'):
            del self.edl_pass
        if hasattr(self, '_box_object'):
            self.remove_bounding_box(render=render)
        if hasattr(self, '_shadow_pass') and self._shadow_pass is not None:
            self.disable_shadows()
        try:
            if self.__charts is not None:
                self.__charts.deep_clean()
                self.__charts = None
        except AttributeError:  # pragma: no cover
            pass

        self._render_passes.deep_clean()
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
        return self.parent.window_size[0] * (xmax - xmin)

    @property
    def height(self):
        """Height of the renderer."""
        _, ymin, _, ymax = self.viewport
        return self.parent.window_size[1] * (ymax - ymin)

    def add_legend(
        self,
        labels=None,
        bcolor=(0.5, 0.5, 0.5),
        border=False,
        size=(0.2, 0.2),
        name=None,
        loc='upper right',
        face='triangle',
    ):
        """Add a legend to render window.

        Entries must be a list containing one string and color entry for each
        item.

        Parameters
        ----------
        labels : list, optional
            When set to ``None``, uses existing labels as specified by

            - :func:`add_mesh <Plotter.add_mesh>`
            - :func:`add_lines <Plotter.add_lines>`
            - :func:`add_points <Plotter.add_points>`

            List containing one entry for each item to be added to the
            legend.  Each entry must contain two strings, [label,
            color], where label is the name of the item to add, and
            color is the color of the label to add.

        bcolor : ColorLike, default: (0.5, 0.5, 0.5)
            Background color, either a three item 0 to 1 RGB color
            list, or a matplotlib color string (e.g. ``'w'`` or ``'white'``
            for a white color).  If None, legend background is
            disabled.

        border : bool, default: False
            Controls if there will be a border around the legend.
            Default False.

        size : sequence[float], default: (0.2, 0.2)
            Two float sequence, each float between 0 and 1.  For example
            ``(0.1, 0.1)`` would make the legend 10% the size of the
            entire figure window.

        name : str, optional
            The name for the added actor so that it can be easily
            updated.  If an actor of this name already exists in the
            rendering window, it will be replaced by the new actor.

        loc : str, default: "upper right"
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

        face : str | pyvista.PolyData | NoneType, default: "triangle"
            Face shape of legend face.  One of the following:

            * None: ``None``
            * Line: ``"-"`` or ``"line"``
            * Triangle: ``"^"`` or ``'triangle'``
            * Circle: ``"o"`` or ``'circle'``
            * Rectangle: ``"r"`` or ``'rectangle'``
            * Custom: :class:`pyvista.PolyData`

            Passing ``None`` removes the legend face.  A custom face can be
            created using :class:`pyvista.PolyData`.  This will be rendered
            from the XY plane.

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
        >>> _ = plotter.add_mesh(
        ...     sphere, 'grey', smooth_shading=True, label='Sphere'
        ... )
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
                raise ValueError(
                    'No labels input.\n\n'
                    'Add labels to individual items when adding them to'
                    'the plotting object with the "label=" parameter.  '
                    'or enter them as the "labels" parameter.'
                )

            self._legend.SetNumberOfEntries(len(self._labels))
            for i, (vtk_object, text, color) in enumerate(self._labels.values()):
                if face is None:
                    # dummy vtk object
                    vtk_object = pyvista.PolyData([0.0, 0.0, 0.0])

                self._legend.SetEntry(i, vtk_object, text, color.float_rgb)

        else:
            self._legend.SetNumberOfEntries(len(labels))

            legend_face = make_legend_face(face)
            for i, (text, color) in enumerate(labels):
                self._legend.SetEntry(i, legend_face, text, Color(color).float_rgb)

        if loc is not None:
            if loc not in ACTOR_LOC_MAP:
                allowed = '\n'.join([f'\t * "{item}"' for item in ACTOR_LOC_MAP])
                raise ValueError(f'Invalid loc "{loc}".  Expected one of the following:\n{allowed}')
            x, y, size = map_loc_to_pos(loc, size, border=0.05)
            self._legend.SetPosition(x, y)
            self._legend.SetPosition2(size[0], size[1])

        if bcolor is None:
            self._legend.SetUseBackground(False)
        else:
            self._legend.SetUseBackground(True)
            self._legend.SetBackgroundColor(Color(bcolor).float_rgb)

        self._legend.SetBorder(border)

        self.add_actor(self._legend, reset_camera=False, name=name, pickable=False)
        return self._legend

    def remove_legend(self, render=True):
        """Remove the legend actor.

        Parameters
        ----------
        render : bool, default: True
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

    def add_ruler(
        self,
        pointa,
        pointb,
        flip_range=False,
        number_labels=5,
        show_labels=True,
        font_size_factor=0.6,
        label_size_factor=1.0,
        label_format=None,
        title="Distance",
        number_minor_ticks=0,
        tick_length=5,
        minor_tick_length=3,
        show_ticks=True,
        tick_label_offset=2,
        label_color=None,
        tick_color=None,
    ):
        """Add ruler.

        The ruler is a 2D object that is not occluded by 3D objects.
        To avoid issues with perspective, it is recommended to use
        parallel projection, i.e. :func:`Plotter.enable_parallel_projection`,
        and place the ruler orthogonal to the viewing direction.

        The title and labels are placed to the right of ruler moving from
        ``pointa`` to ``pointb``. Use ``flip_range`` to flip the ``0`` location,
        if needed.

        Since the ruler is placed in an overlay on the viewing scene, the camera
        does not automatically reset to include the ruler in the view.

        Parameters
        ----------
        pointa : sequence[float]
            Starting point for ruler.

        pointb : sequence[float]
            Ending point for ruler.

        flip_range : bool, default: False
            If ``True``, the distance range goes from ``pointb`` to ``pointa``.

        number_labels : int, default: 5
            Number of labels to place on ruler.

        show_labels : bool, default: True
            Whether to show labels.

        font_size_factor : float, default: 0.6
            Factor to scale font size overall.

        label_size_factor : float, default: 1.0
            Factor to scale label size relative to title size.

        label_format : str, optional
            A printf style format for labels, e.g. '%E'.

        title : str, default: "Distance"
            The title to display.

        number_minor_ticks : int, default: 0
            Number of minor ticks between major ticks.

        tick_length : int, default: 5
            Length of ticks in pixels.

        minor_tick_length : int, default: 3
            Length of minor ticks in pixels.

        show_ticks : bool, default: True
            Whether to show the ticks.

        tick_label_offset : int, default: 2
            Offset between tick and label in pixels.

        label_color : ColorLike, optional
            Either a string, rgb list, or hex color string for
            label and title colors.

            .. warning::
                This is either white or black.

        tick_color : ColorLike, optional
            Either a string, rgb list, or hex color string for
            tick line colors.

        Returns
        -------
        vtk.vtkActor
            VTK actor of the ruler.

        Examples
        --------
        >>> import pyvista as pv
        >>> cone = pv.Cone(height=2.0, radius=0.5)
        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(cone)

        Measure x direction of cone and place ruler slightly below.

        >>> _ = plotter.add_ruler(
        ...     pointa=[cone.bounds[0], cone.bounds[2] - 0.1, 0.0],
        ...     pointb=[cone.bounds[1], cone.bounds[2] - 0.1, 0.0],
        ...     title="X Distance",
        ... )

        Measure y direction of cone and place ruler slightly to left.
        The title and labels are placed to the right of the ruler when
        traveling from ``pointa`` to ``pointb``.

        >>> _ = plotter.add_ruler(
        ...     pointa=[cone.bounds[0] - 0.1, cone.bounds[3], 0.0],
        ...     pointb=[cone.bounds[0] - 0.1, cone.bounds[2], 0.0],
        ...     flip_range=True,
        ...     title="Y Distance",
        ... )
        >>> plotter.enable_parallel_projection()
        >>> plotter.view_xy()
        >>> plotter.show()

        """
        label_color = Color(label_color, default_color=self._theme.font.color)
        tick_color = Color(tick_color, default_color=self._theme.font.color)

        ruler = _vtk.vtkAxisActor2D()

        ruler.GetPositionCoordinate().SetCoordinateSystemToWorld()
        ruler.GetPosition2Coordinate().SetCoordinateSystemToWorld()
        ruler.GetPositionCoordinate().SetReferenceCoordinate(None)
        ruler.GetPositionCoordinate().SetValue(pointa[0], pointa[1], pointa[2])
        ruler.GetPosition2Coordinate().SetValue(pointb[0], pointb[1], pointb[2])

        distance = np.linalg.norm(np.asarray(pointa) - np.asarray(pointb))
        if flip_range:
            ruler.SetRange(distance, 0)
        else:
            ruler.SetRange(0, distance)

        ruler.SetTitle(title)
        ruler.SetFontFactor(font_size_factor)
        ruler.SetLabelFactor(label_size_factor)
        ruler.SetNumberOfLabels(number_labels)
        ruler.SetLabelVisibility(show_labels)
        if label_format:
            ruler.SetLabelFormat(label_format)
        ruler.GetProperty().SetColor(*tick_color.int_rgb)
        if label_color != Color('white'):
            # This property turns black if set
            ruler.GetLabelTextProperty().SetColor(*label_color.int_rgb)
            ruler.GetTitleTextProperty().SetColor(*label_color.int_rgb)
        ruler.SetNumberOfMinorTicks(number_minor_ticks)
        ruler.SetTickVisibility(show_ticks)
        ruler.SetTickLength(tick_length)
        ruler.SetMinorTickLength(minor_tick_length)
        ruler.SetTickOffset(tick_label_offset)

        self.add_actor(ruler, reset_camera=True, pickable=False)
        return ruler

    def add_legend_scale(
        self,
        corner_offset_factor=2.0,
        bottom_border_offset=30,
        top_border_offset=30,
        left_border_offset=30,
        right_border_offset=30,
        bottom_axis_visibility=True,
        top_axis_visibility=True,
        left_axis_visibility=True,
        right_axis_visibility=True,
        legend_visibility=True,
        xy_label_mode=False,
        render=True,
        color=None,
        font_size_factor=0.6,
        label_size_factor=1.0,
        label_format=None,
        number_minor_ticks=0,
        tick_length=5,
        minor_tick_length=3,
        show_ticks=True,
        tick_label_offset=2,
    ):
        """Annotate the render window with scale and distance information.

        Its basic goal is to provide an indication of the scale of the scene.
        Four axes surrounding the render window indicate (in a variety of ways)
        the scale of what the camera is viewing. An option also exists for
        displaying a scale legend.

        Parameters
        ----------
        corner_offset_factor : float, default: 2.0
            The corner offset value.

        bottom_border_offset : int, default: 30
            Bottom border offset. Recommended value ``50``.

        top_border_offset : int, default: 30
            Top border offset. Recommended value ``50``.

        left_border_offset : int, default: 30
            Left border offset. Recommended value ``100``.

        right_border_offset : int, default: 30
            Right border offset. Recommended value ``100``.

        bottom_axis_visibility : bool, default: True
            Whether the bottom axis is visible.

        top_axis_visibility : bool, default: True
            Whether the top axis is visible.

        left_axis_visibility : bool, default: True
            Whether the left axis is visible.

        right_axis_visibility : bool, default: True
            Whether the right axis is visible.

        legend_visibility : bool, default: True
            Whether the legend scale is visible.

        xy_label_mode : bool, default: False
            The axes can be programmed either to display distance scales
            or x-y coordinate values. By default,
            the scales display a distance. However, if you know that the
            view is down the z-axis, the scales can be programmed to display
            x-y coordinate values.

        render : bool, default: True
            Whether to render when the actor is added.

        color : ColorLike, optional
            Either a string, rgb list, or hex color string for tick text
            and tick line colors.

            .. warning::
                The axis labels tend to be either white or black.

        font_size_factor : float, default: 0.6
            Factor to scale font size overall.

        label_size_factor : float, default: 1.0
            Factor to scale label size relative to title size.

        label_format : str, optional
            A printf style format for labels, e.g. ``'%E'``.
            See :ref:`old-string-formatting`.

        number_minor_ticks : int, default: 0
            Number of minor ticks between major ticks.

        tick_length : int, default: 5
            Length of ticks in pixels.

        minor_tick_length : int, default: 3
            Length of minor ticks in pixels.

        show_ticks : bool, default: True
            Whether to show the ticks.

        tick_label_offset : int, default: 2
            Offset between tick and label in pixels.

        Returns
        -------
        vtk.vtkActor
            The actor for the added ``vtkLegendScaleActor``.

        Warnings
        --------
        Please be aware that the axes and scale values are subject to perspective
        effects. The distances are computed in the focal plane of the camera. When
        there are large view angles (i.e., perspective projection), the computed
        distances may provide users the wrong sense of scale. These effects are not
        present when parallel projection is enabled.

        Examples
        --------
        >>> import pyvista as pv
        >>> cone = pv.Cone(height=2.0, radius=0.5)
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(cone)
        >>> _ = pl.add_legend_scale()
        >>> pl.show()

        """
        color = Color(color, default_color=self._theme.font.color)

        legend_scale = _vtk.vtkLegendScaleActor()
        legend_scale.SetCornerOffsetFactor(corner_offset_factor)
        legend_scale.SetLegendVisibility(legend_visibility)
        if xy_label_mode:
            legend_scale.SetLabelModeToXYCoordinates()
        else:
            legend_scale.SetLabelModeToDistance()
        legend_scale.SetBottomAxisVisibility(bottom_axis_visibility)
        legend_scale.SetBottomBorderOffset(bottom_border_offset)
        legend_scale.SetLeftAxisVisibility(left_axis_visibility)
        legend_scale.SetLeftBorderOffset(left_border_offset)
        legend_scale.SetRightAxisVisibility(right_axis_visibility)
        legend_scale.SetRightBorderOffset(right_border_offset)
        legend_scale.SetTopAxisVisibility(top_axis_visibility)
        legend_scale.SetTopBorderOffset(top_border_offset)

        for text in ['Label', 'Title']:
            prop = getattr(legend_scale, f'GetLegend{text}Property')()
            if color != Color('white'):
                # This property turns black if set
                prop.SetColor(*color.int_rgb)
            prop.SetFontSize(
                int(font_size_factor * 20)
            )  # hack to avoid multiple font size arguments

        for ax in ['Bottom', 'Left', 'Right', 'Top']:
            axis = getattr(legend_scale, f'Get{ax}Axis')()
            axis.GetProperty().SetColor(*color.int_rgb)
            if color != Color('white'):
                # This label property turns black if set
                axis.GetLabelTextProperty().SetColor(*color.int_rgb)
            axis.SetFontFactor(font_size_factor)
            axis.SetLabelFactor(label_size_factor)
            if label_format:
                axis.SetLabelFormat(label_format)
            axis.SetNumberOfMinorTicks(number_minor_ticks)
            axis.SetTickLength(tick_length)
            axis.SetMinorTickLength(minor_tick_length)
            axis.SetTickVisibility(show_ticks)
            axis.SetTickOffset(tick_label_offset)

        return self.add_actor(
            legend_scale,
            reset_camera=False,
            name='_vtkLegendScaleActor',
            culling=False,
            pickable=False,
            render=render,
        )


def _line_for_legend():
    """Create a simple line-like rectangle for the legend."""
    points = [
        [0, 0, 0],
        [0.4, 0, 0],
        [0.4, 0.07, 0],
        [0, 0.07, 0],
        [
            0.5,
            0,
            0,
        ],  # last point needed to expand the bounds of the PolyData to be rendered smaller
    ]
    legendface = pyvista.PolyData()
    legendface.points = np.array(points)
    legendface.faces = [4, 0, 1, 2, 3]
    return legendface
