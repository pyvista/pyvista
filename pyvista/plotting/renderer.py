"""Module containing pyvista implementation of :vtk:`vtkRenderer`."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
import contextlib
from functools import partial
from functools import wraps
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import cast
import warnings

import numpy as np

import pyvista
from pyvista import MAX_N_COLOR_BARS
from pyvista import vtk_version_info
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core._typing_core import BoundsTuple
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.helpers import wrap
from pyvista.core.utilities.misc import _BoundsSizeMixin
from pyvista.core.utilities.misc import _NoNewAttrMixin
from pyvista.core.utilities.misc import assert_empty_kwargs
from pyvista.core.utilities.misc import try_callback

from . import _vtk
from .actor import Actor
from .camera import Camera
from .charts import Charts
from .colors import Color
from .colors import get_cycler
from .errors import InvalidCameraError
from .helpers import view_vectors
from .mapper import DataSetMapper
from .prop_collection import _PropCollection
from .render_passes import RenderPasses
from .tools import create_axes_marker
from .tools import create_axes_orientation_box
from .tools import create_north_arrow
from .tools import parse_font_family
from .utilities.gl_checks import check_depth_peeling
from .utilities.gl_checks import uses_egl

if TYPE_CHECKING:
    from pyvista.core.pointset import PolyData

    from .cube_axes_actor import CubeAxesActor
    from .lights import Light

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

    Parameters
    ----------
    loc : str
        Location of the actor. Can be a string with values such as 'right',
        'left', 'upper', or 'lower'.
    size : Sequence of length 2
        Size of the actor. It must be a list of length 2.
    border : float, default: 0.05
        Size of the border around the actor.

    Returns
    -------
    tuple
        The VTK position and position2 coordinates. Tuple of the form (x, y, size).

    Raises
    ------
    ValueError
        If the ``size`` parameter is not a list of length 2.

    """
    if not isinstance(size, Sequence) or len(size) != 2:
        msg = f'`size` must be a list of length 2. Passed value is {size}'
        raise ValueError(msg)

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


def make_legend_face(face) -> PolyData:
    """Create the legend face based on the given face.

    Parameters
    ----------
    face : str | pyvista.PolyData | NoneType
        The shape of the legend face. Valid strings are:
        '-', 'line', '^', 'triangle', 'o', 'circle', 'r', 'rectangle', 'none'.
        Also accepts ``None`` or instances of ``pyvista.PolyData``.

    Returns
    -------
    pyvista.PolyData
        The legend face as a PolyData object.

    Raises
    ------
    ValueError
        If the provided face value is invalid.

    """

    def normalize(poly):
        norm_poly = poly.copy()  # Avoid mutating input

        # Center data
        norm_poly.points -= np.array(norm_poly.center)

        # Scale so max bounds are [-0.5, 0.5] along x and y axes
        size = np.array(norm_poly.bounds_size)
        size[size < 1e-8] = 1  # Avoid division by zero
        max_xy_size = max(size[0:2])
        norm_poly.scale(1 / max_xy_size, inplace=True)

        # Add final offset to align the symbol with the adjacent text
        y_offset = 0.6  # determined experimentally
        norm_poly.points += (0, y_offset, 0)
        return norm_poly

    if face is None or face == 'none':
        legendface = pyvista.PolyData([0.0, 0.0, 0.0], faces=np.empty(0, dtype=int))  # type: ignore[arg-type]
    elif face in ['-', 'line']:
        legendface = pyvista.Rectangle().scale((1, 0.2, 1))
    elif face in ['^', 'triangle']:
        legendface = pyvista.Triangle()
    elif face in ['o', 'circle']:
        legendface = pyvista.Circle()
    elif face in ['r', 'rectangle']:
        legendface = pyvista.Rectangle()
    elif isinstance(face, pyvista.PolyData):
        legendface = face
    else:
        msg = (
            f'Invalid face "{face}".  Must be one of the following:\n'
            '\t"triangle"\n'
            '\t"circle"\n'
            '\t"rectangle"\n'
            '\t"none"\n'
            '\tpyvista.PolyData'
        )
        raise ValueError(msg)

    # Normalize the geometry
    legendface = normalize(legendface)

    # Add points to each corner of the normalized geom to define the full extent of the geometry.
    # This is needed for asymmetric shapes (like a line) because otherwise the legend actor
    # will do its own scaling and skew the shape
    rect = normalize(pyvista.Rectangle())
    legendface.points = np.append(legendface.points, rect.points, axis=0)
    return legendface


@_deprecate_positional_args(allowed=['camera', 'point'])
def scale_point(camera, point, invert=False):  # noqa: FBT002
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


class CameraPosition(_NoNewAttrMixin):
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

    def __init__(self, position, focal_point, viewup) -> None:
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
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.camera_position.to_list()
        [(0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

        """
        return [self._position, self._focal_point, self._viewup]

    def __repr__(self) -> str:
        """List representation method."""
        return '[{},\n {},\n {}]'.format(*self.to_list())

    def __getitem__(self, index):
        """Fetch a component by index location like a list."""
        return self.to_list()[index]

    def __eq__(self, other):
        """Comparison operator to act on list version of CameraPosition object."""
        if isinstance(other, CameraPosition):
            return self.to_list() == other.to_list()
        return self.to_list() == other

    __hash__ = None  # type: ignore[assignment]  # https://github.com/pyvista/pyvista/pull/7671

    @property
    def position(self):  # numpydoc ignore=RT01
        """Location of the camera in world coordinates."""
        return self._position

    @position.setter
    def position(self, value) -> None:
        self._position = value

    @property
    def focal_point(self):  # numpydoc ignore=RT01
        """Location of the camera's focus in world coordinates."""
        return self._focal_point

    @focal_point.setter
    def focal_point(self, value) -> None:
        self._focal_point = value

    @property
    def viewup(self):  # numpydoc ignore=RT01
        """Viewup vector of the camera."""
        return self._viewup

    @viewup.setter
    def viewup(self, value) -> None:
        self._viewup = value


class Renderer(
    _NoNewAttrMixin, _BoundsSizeMixin, _vtk.DisableVtkSnakeCase, _vtk.vtkOpenGLRenderer
):
    """Renderer class."""

    # map camera_position string to an attribute
    CAMERA_STR_ATTR_MAP: ClassVar[dict[str, str]] = {
        'xy': 'view_xy',
        'xz': 'view_xz',
        'yz': 'view_yz',
        'yx': 'view_yx',
        'zx': 'view_zx',
        'zy': 'view_zy',
        'iso': 'view_isometric',
    }

    @_deprecate_positional_args(allowed=['parent'])
    def __init__(  # noqa: PLR0917
        self,
        parent,
        border=True,  # noqa: FBT002
        border_color='w',
        border_width=2.0,
    ) -> None:  # numpydoc ignore=PR01,RT01
        """Initialize the renderer."""
        super().__init__()
        self._actors = _PropCollection(self.GetViewProps())
        self.parent = parent  # weakref.proxy to the plotter from Renderers
        self._theme = parent.theme
        self.bounding_box_actor: Actor | None = None
        self.axes_actor: _vtk.vtkAxesActor | None = None
        self.axes_widget: _vtk.vtkOrientationMarkerWidget | None = None
        self.scale = [1.0, 1.0, 1.0]
        self.AutomaticLightCreationOff()
        self._labels: dict[
            str, tuple[_vtk.vtkPolyData | _vtk.vtkImageData, str, Color]
        ] = {}  # tracks labeled actors
        self._legend: _vtk.vtkLegendBoxActor | None = None
        self._floor: PolyData | None = None
        self._floors: list[Actor] = []
        self._floor_kwargs: list[dict[str, Any]] = []
        # this keeps track of lights added manually to prevent garbage collection
        self._lights: list[Light] = []
        self._camera: Camera | None = Camera(self)
        self.SetActiveCamera(self._camera)
        self._empty_str: _vtk.vtkStringArray | None = (
            None  # used to track reference to a vtkStringArray
        )
        self._shadow_pass = None
        self._render_passes = RenderPasses(self)
        self.cube_axes_actor: CubeAxesActor | None = None

        # This is a private variable to keep track of how many colorbars exist
        # This allows us to keep adding colorbars without overlapping
        self._scalar_bar_slots = set(range(MAX_N_COLOR_BARS))
        self._scalar_bar_slot_lookup: dict[str, int] = {}
        self._charts: Charts | None = None

        self._border_actor: _vtk.vtkActor2D | None = None
        if border:
            self.add_border(border_color, border_width)

        self.set_color_cycler(self._theme.color_cycler)
        self._closed = False
        self._bounding_box: _vtk.vtkOutlineCornerSource | _vtk.vtkCubeSource | None = None
        self._box_object: PolyData | None = None
        self._marker_actor: _vtk.vtkAxesActor | None = None

    @property
    def camera_set(self) -> bool:  # numpydoc ignore=RT01
        """Get or set whether this camera has been configured."""
        if self.camera is None:  # pragma: no cover
            return False
        return self.camera.is_set

    @camera_set.setter
    def camera_set(self, is_set: bool) -> None:
        self.camera.is_set = is_set

    def set_color_cycler(self, color_cycler) -> None:
        """Set or reset this renderer's color cycler.

        This color cycler is iterated over by each sequential :meth:`~pyvista.Plotter.add_mesh`
        call to set the default color of the dataset being plotted.

        When setting, the value must be either a list of color-like objects,
        or a cycler of color-like objects. If the value passed is a single
        string, it must be one of:

            * ``'default'`` - Use the default color cycler (matches matplotlib's default)
            * ``'matplotlib`` - Dynamically get matplotlib's current theme's color cycler.
            * ``'all'`` - Cycle through all of the available colors in
              ``pyvista.plotting.colors.hexcolors``

        Setting to ``None`` will disable the use of the color cycler on this
        renderer.

        .. note::
            If a mesh has scalar data, set ``color=True`` in the call to
            :meth:`~pyvista.Plotter.add_mesh` to color the mesh with the
            next color in the cycler. Otherwise, the mesh's scalars are used
            to color the mesh by default.

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

        Load a mesh with active scalars and split it into two separate meshes.

        >>> mesh = pv.Wavelet()
        >>> mesh.active_scalars_name
        'RTData'

        >>> a = mesh.clip(invert=True)
        >>> b = mesh.clip(invert=False)

        Enable color cycling and set ``color=True`` to force the meshes to be colored with the
        cycler's colors.

        >>> pv.global_theme.color_cycler = 'default'
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(a, color=True)
        >>> _ = pl.add_mesh(b, color=True)
        >>> pl.show()

        """
        cycler = get_cycler(color_cycler)
        if cycler is not None:
            # Color cycler - call object to generate `cycle` instance
            self._color_cycle = cycler()
        else:
            self._color_cycle = None

    @property
    def next_color(self):  # numpydoc ignore=RT01
        """Return next color from this renderer's color cycler."""
        if self._color_cycle is None:
            return self._theme.color
        return next(self._color_cycle)['color']

    @property
    def camera_position(self):  # numpydoc ignore=RT01
        """Return or set the camera position of active render window.

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
        if camera_location is None:
            return
        elif isinstance(camera_location, str):
            camera_location = camera_location.lower()
            if camera_location not in self.CAMERA_STR_ATTR_MAP:
                msg = (
                    'Invalid view direction.  '
                    'Use one of the following:\n   '
                    f'{", ".join(self.CAMERA_STR_ATTR_MAP)}'
                )
                raise InvalidCameraError(msg)

            getattr(self, self.CAMERA_STR_ATTR_MAP[camera_location])()

        elif isinstance(camera_location[0], (int, float)):
            if len(camera_location) != 3:
                raise InvalidCameraError
            self.view_vector(camera_location)
        else:
            # check if a valid camera position
            if not isinstance(camera_location, CameraPosition) and (
                not len(camera_location) == 3 or any(len(item) != 3 for item in camera_location)
            ):
                raise InvalidCameraError

            # everything is set explicitly
            self.camera.position = scale_point(self.camera, camera_location[0], invert=False)
            self.camera.focal_point = scale_point(self.camera, camera_location[1], invert=False)
            self.camera.up = camera_location[2]

        # reset clipping range
        self.reset_camera_clipping_range()
        self.camera_set = True
        self.Modified()

    def reset_camera_clipping_range(self) -> None:
        """Reset the camera clipping range based on the bounds of the visible actors.

        This ensures that no props are cut off
        """
        self.ResetCameraClippingRange()

    @property
    def camera(self):  # numpydoc ignore=RT01
        """Return the active camera for the rendering scene."""
        return self._camera

    @camera.setter
    def camera(self, source) -> None:
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
    def bounds(self) -> BoundsTuple:  # numpydoc ignore=RT01
        """Return the bounds of all VISIBLE actors present in the rendering window.

        Actors with :attr:`~pyvista.Actor.visibility` or :attr:`~pyvista.Actor.use_bounds`
        disabled are `not` included in the bounds.

        .. versionchanged:: 0.45

            Only the bounds of visible actors are now returned. Previously, the bounds
            of all actors was returned, regardless of visibility.

        Returns
        -------
        BoundsTuple
            Bounds of all visible actors in the active renderer.

        See Also
        --------
        compute_bounds
            Compute the bounds with options to enable or disable actor visibility.

        """
        bounds = self.ComputeVisiblePropBounds()
        return _fixup_bounds(bounds)

    def compute_bounds(
        self,
        *,
        force_visibility: bool = False,
        force_use_bounds: bool = False,
        ignore_actors: Sequence[str | _vtk.vtkProp | type[_vtk.vtkProp]] | None = None,
    ) -> BoundsTuple:
        """Return the bounds of actors present in the renderer.

        By default, only visible actors are included in the bounds computation.
        Optionally, the bounds of all actors may be computed, regardless if they
        have their :attr:`~pyvista.Actor.visibility` or :attr:`~pyvista.Actor.use_bounds`
        disabled. Specific actors may also be removed from the computation.

        .. versionadded:: 0.45

        Parameters
        ----------
        force_visibility : bool, default: False
            Include actors with :attr:`~pyvista.Actor.visibility` disabled in the
            computation. By default, invisible actors are excluded.

        force_use_bounds : bool, default: False
            Include actors with :attr:`~pyvista.Actor.use_bounds` disabled in the
            computation. By default, actors with use bounds disabled are excluded.

        ignore_actors : sequence[str | :vtk:`vtkProp` | type[:vtk:`vtkProp`]]
            List of actors to ignore. The bounds of any actors included will be ignored.
            Specify actors by name, type, or by instance.

        Returns
        -------
        BoundsTuple
            Bounds of selected actors in the active renderer.

        See Also
        --------
        bounds
            Bounds of all specified actors.

        """
        the_bounds = np.array([np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf])
        if ignore_actors is None:
            ignore_actors = []

        ignored_types = [
            actor_type for actor_type in ignore_actors if isinstance(actor_type, type)
        ]

        def _update_bounds(bounds) -> None:
            def update_axis(ax) -> None:
                the_bounds[ax * 2] = min(bounds[ax * 2], the_bounds[ax * 2])
                the_bounds[ax * 2 + 1] = max(bounds[ax * 2 + 1], the_bounds[ax * 2 + 1])

            for ax in range(3):
                update_axis(ax)

        for name, actor in self._actors.items():
            if not actor.GetUseBounds() and not force_use_bounds:
                continue
            if not actor.GetVisibility() and not force_visibility:
                continue
            if (  # Check if the actor should be ignored
                name in ignore_actors
                or actor in ignore_actors
                or any(isinstance(actor, actor_type) for actor_type in ignored_types)
            ):
                continue
            if hasattr(actor, 'GetBounds') and (actor_bounds := actor.GetBounds()) is not None:
                _update_bounds(actor_bounds)

        return _fixup_bounds(the_bounds)

    @property
    def length(self):  # numpydoc ignore=RT01
        """Return the length of the diagonal of the bounding box of the scene.

        Returns
        -------
        float
            Length of the diagonal of the bounding box.

        """
        return pyvista.Box(self.bounds).length

    @property
    def center(self) -> tuple[float, float, float]:
        """Return the center of the bounding box around all data present in the scene.

        Returns
        -------
        tuple[float, float, float]
            Cartesian coordinates of the center.

        """
        bnds = self.bounds
        x = (bnds.x_max + bnds.x_min) / 2
        y = (bnds.y_max + bnds.y_min) / 2
        z = (bnds.z_max + bnds.z_min) / 2
        return x, y, z

    @property
    def background_color(self):  # numpydoc ignore=RT01
        """Return the background color of this renderer."""
        return Color(self.GetBackground())

    @background_color.setter
    def background_color(self, color) -> None:
        self.set_background(color)
        self.Modified()

    def _before_render_event(self, *args, **kwargs) -> None:
        """Notify all charts about render event."""
        if self._charts is not None:
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

    def disable_depth_peeling(self) -> None:
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
            msg = f'`aa_type` must be a string, not {type(aa_type)}'
            raise TypeError(msg)
        aa_type = aa_type.lower()

        if aa_type == 'fxaa':
            if uses_egl():  # pragma: no cover
                # only display the warning when not building documentation
                if not pyvista.BUILDING_GALLERY:
                    warnings.warn(
                        'VTK compiled with OSMesa/EGL does not properly support '
                        'FXAA anti-aliasing and SSAA will be used instead.',
                    )
                self._render_passes.enable_ssaa_pass()
                return
            self._enable_fxaa()

        elif aa_type == 'ssaa':
            self._render_passes.enable_ssaa_pass()

        else:
            msg = f'Invalid `aa_type` "{aa_type}". Should be either "fxaa" or "ssaa"'
            raise ValueError(msg)

    def disable_anti_aliasing(self) -> None:
        """Disable all anti-aliasing."""
        self._render_passes.disable_ssaa_pass()
        self.SetUseFXAA(False)
        self.Modified()

    def _enable_fxaa(self) -> None:
        """Enable FXAA anti-aliasing."""
        self.SetUseFXAA(True)
        self.Modified()

    def _disable_fxaa(self) -> None:
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
        :vtk:`vtkActor2D`
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
    def has_border(self) -> bool:  # numpydoc ignore=RT01
        """Return if the renderer has a border."""
        return self._border_actor is not None

    @property
    def border_width(self):  # numpydoc ignore=RT01
        """Return the border width."""
        if self.has_border:
            return self._border_actor.GetProperty().GetLineWidth()  # type: ignore[union-attr]
        return 0

    @property
    def border_color(self):  # numpydoc ignore=RT01
        """Return the border color."""
        if self.has_border:
            return Color(self._border_actor.GetProperty().GetColor())  # type: ignore[union-attr]
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
        >>> import pyvista as pv
        >>> chart = pv.Chart2D()
        >>> _ = chart.plot(range(10), range(10))
        >>> pl = pv.Plotter()
        >>> pl.add_chart(chart)
        >>> pl.show()

        """
        # lazy instantiation here to avoid creating the charts object unless needed.
        if self._charts is None:
            self._charts = Charts(self)
            self.AddObserver('StartEvent', partial(try_callback, self._before_render_event))  # type: ignore[arg-type]
        self._charts.add_chart(chart, *charts)

    @property
    def has_charts(self):  # numpydoc ignore=RT01
        """Return whether this renderer has charts."""
        return self._charts is not None and len(self._charts) > 0

    def get_charts(self):  # numpydoc ignore=RT01
        """Return a list of all charts in this renderer.

        Examples
        --------
        .. pyvista-plot::
           :force_static:

           >>> import pyvista as pv
           >>> chart = pv.Chart2D()
           >>> _ = chart.line([1, 2, 3], [0, 1, 0])
           >>> pl = pv.Plotter()
           >>> pl.add_chart(chart)
           >>> chart is pl.renderer.get_charts()[0]
           True

        """
        return [*self._charts] if self.has_charts else []  # type: ignore[misc]

    @wraps(Charts.set_interaction)
    @_deprecate_positional_args(allowed=['interactive'])
    def set_chart_interaction(  # numpydoc ignore=PR01,RT01
        self,
        interactive,
        toggle=False,  # noqa: FBT002
    ):
        """Wrap ``Charts.set_interaction``."""
        return self._charts.set_interaction(interactive, toggle=toggle) if self.has_charts else []  # type: ignore[union-attr]

    @wraps(Charts.get_charts_by_pos)
    def _get_charts_by_pos(self, pos):
        """Wrap ``Charts.get_charts_by_pos``."""
        return self._charts.get_charts_by_pos(pos) if self.has_charts else []  # type: ignore[union-attr]

    def remove_chart(self, chart_or_index) -> None:
        """Remove a chart from this renderer.

        Parameters
        ----------
        chart_or_index : Chart or int
            Either the chart to remove from this renderer or its index in the collection of charts.

        Examples
        --------
        First define a function to add two charts to a renderer.

        >>> import pyvista as pv
        >>> def plotter_with_charts():
        ...     pl = pv.Plotter()
        ...     pl.background_color = 'w'
        ...     chart_left = pv.Chart2D(size=(0.5, 1))
        ...     _ = chart_left.line([0, 1, 2], [2, 1, 3])
        ...     pl.add_chart(chart_left)
        ...     chart_right = pv.Chart2D(size=(0.5, 1), loc=(0.5, 0))
        ...     _ = chart_right.line([0, 1, 2], [3, 1, 2])
        ...     pl.add_chart(chart_right)
        ...     return pl, chart_left, chart_right
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
        if self.has_charts:
            cast('Charts', self._charts).remove_chart(chart_or_index)

    @property
    def actors(self) -> dict[str, _vtk.vtkProp]:  # numpydoc ignore=RT01
        """Return a dictionary of actors assigned to this renderer.

        .. note::

            This may include 2D actors such as :class:`~pyvista.Text`, 3D actors such
            as :class:`~pyvista.Actor`, and assemblies such as :class:`~pyvista.AxesAssembly`.
            The actors may also be unwrapped VTK objects.

        """
        return dict(self._actors.items())

    @_deprecate_positional_args(allowed=['actor'])
    def add_actor(  # noqa: PLR0917
        self,
        actor,
        reset_camera=False,  # noqa: FBT002
        name=None,
        culling=False,  # noqa: FBT002
        pickable=True,  # noqa: FBT002
        render=True,  # noqa: FBT002
        remove_existing_actor=True,  # noqa: FBT002
    ):
        """Add an actor to render window.

        Creates an actor if input is a mapper.

        Parameters
        ----------
        actor : :vtk:`vtkActor` | :vtk:`vtkMapper` | Actor
            The actor to be added. Can be either :vtk:`vtkActor` or :vtk:`vtkMapper`.

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
        actor : :vtk:`vtkActor` | Actor
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

        if name is None:
            name = (
                actor.name
                if (hasattr(actor, 'name') and actor.name)
                else f'{type(actor).__name__}({actor.GetAddressAsString("")})'
            )
        actor.name = name
        actor.SetPickable(pickable)
        # Apply this renderer's scale to the actor (which can be further scaled)
        if hasattr(actor, 'SetScale'):
            actor.SetScale(np.array(actor.GetScale()) * np.array(self.scale))
        self.AddActor(actor)  # must add actor before resetting camera

        if reset_camera or (not self.camera_set and reset_camera is None and not rv):
            self.reset_camera(render=render)
        elif render:
            self.parent.render()

        self.update_bounds_axes()

        if isinstance(culling, str):
            culling = culling.lower()

        if culling:
            if culling in [True, 'back', 'backface', 'b']:
                with contextlib.suppress(AttributeError):
                    actor.GetProperty().BackfaceCullingOn()
            elif culling in ['front', 'frontface', 'f']:
                with contextlib.suppress(AttributeError):
                    actor.GetProperty().FrontfaceCullingOn()
            else:
                msg = f'Culling option ({culling}) not understood.'
                raise ValueError(msg)

        self.Modified()

        prop = None
        if hasattr(actor, 'GetProperty'):
            prop = actor.GetProperty()

        return actor, prop

    @_deprecate_positional_args
    def add_axes_at_origin(  # noqa: PLR0917
        self,
        x_color=None,
        y_color=None,
        z_color=None,
        xlabel='X',
        ylabel='Y',
        zlabel='Z',
        line_width=2,
        labels_off=False,  # noqa: FBT002
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
        :vtk:`vtkAxesActor`
            Actor of the axes.

        See Also
        --------
        add_axes

        :ref:`axes_objects_example`
            Example showing different axes objects.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Sphere(center=(2, 0, 0)), color='r')
        >>> _ = pl.add_mesh(pv.Sphere(center=(0, 2, 0)), color='g')
        >>> _ = pl.add_mesh(pv.Sphere(center=(0, 0, 2)), color='b')
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        """
        marker = create_axes_marker(
            line_width=line_width,
            x_color=x_color,
            y_color=y_color,
            z_color=z_color,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            labels_off=labels_off,
        )
        self.AddActor(marker)
        self.Modified()
        self._marker_actor = marker
        return marker

    def _remove_axes_widget(self):
        """Remove and delete the current axes widget."""
        if self.axes_widget is not None:
            self.axes_actor = None
            # HACK: set the viewport to a tiny value to hide the widget first
            # This is due to an issue with a blue box appearing after removal
            # Tracked in https://gitlab.kitware.com/vtk/vtk/-/issues/19592
            self.axes_widget.SetViewport(0.0, 0.0, 0.0001, 0.0001)
            if self.axes_widget.GetEnabled():
                self.axes_widget.EnabledOff()
            self.Modified()
            self.axes_widget = None

    @_deprecate_positional_args(allowed=['actor'])
    def add_orientation_widget(  # noqa: PLR0917
        self,
        actor,
        interactive=None,
        color=None,
        opacity=1.0,
        viewport=None,
    ):
        """Use the given actor in an orientation marker widget.

        Color and opacity are only valid arguments if a mesh is passed.

        Parameters
        ----------
        actor : :vtk:`vtkActor` | DataSet
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
        :vtk:`vtkOrientationMarkerWidget`
            Orientation marker widget.

        See Also
        --------
        add_axes
            Add arrow-style axes as an orientation widget.

        add_box_axes
            Add an axes box as an orientation widget.

        add_north_arrow_widget
            Add north arrow as an orientation widget.

        :ref:`axes_objects_example`
            Example showing different axes objects.

        Examples
        --------
        Use an Arrow as the orientation widget.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Cube(), show_edges=True)
        >>> widget = pl.add_orientation_widget(pv.Arrow(), color='r')
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
        self._remove_axes_widget()
        if interactive is None:
            interactive = self._theme.interactive
        axes_widget = _vtk.vtkOrientationMarkerWidget()
        self.axes_widget = axes_widget
        axes_widget.SetOrientationMarker(actor)
        if self.parent.iren is not None:
            axes_widget.SetInteractor(self.parent.iren.interactor)
            axes_widget.SetEnabled(1)
            axes_widget.SetInteractive(interactive)
        axes_widget.SetCurrentRenderer(self)
        if viewport is not None:
            axes_widget.SetViewport(viewport)
        self.Modified()
        return axes_widget

    @_deprecate_positional_args
    def add_axes(  # noqa: PLR0917
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
        labels_off=False,  # noqa: FBT002
        box=None,
        box_args=None,
        viewport=(0, 0, 0.2, 0.2),
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
            Color used for the x-axis arrow.  Defaults to theme axes parameters.

        y_color : ColorLike, optional
            Color used for the y-axis arrow.  Defaults to theme axes parameters.

        z_color : ColorLike, optional
            Color used for the z-axis arrow.  Defaults to theme axes parameters.

        xlabel : str, default: "X"
            Text used for the x-axis.

        ylabel : str, default: "Y"
            Text used for the y-axis.

        zlabel : str, default: "Z"
            Text used for the z-axis.

        labels_off : bool, default: False
            Enable or disable the text labels for the axes.

        box : bool, optional
            Show a box orientation marker. Use ``box_args`` to adjust.
            See :func:`pyvista.create_axes_orientation_box` for details.

            .. deprecated:: 0.43.0
                The is deprecated. Use `add_box_axes` method instead.

        box_args : dict, optional
            Parameters for the orientation box widget when
            ``box=True``. See the parameters of
            :func:`pyvista.create_axes_orientation_box`.

        viewport : sequence[float], default: (0, 0, 0.2, 0.2)
            Viewport ``(xstart, ystart, xend, yend)`` of the widget.

        **kwargs : dict, optional
            Used for passing parameters for the orientation marker
            widget. See the parameters of :func:`pyvista.create_axes_marker`.

        Returns
        -------
        AxesActor
            Axes actor of the added widget.

        See Also
        --------
        show_axes
            Similar method which calls :func:`add_axes` without any parameters.

        add_axes_at_origin
            Add an :class:`pyvista.AxesActor` to the origin of a scene.

        add_box_axes
            Add an axes box as an orientation widget.

        add_north_arrow_widget
            Add north arrow as an orientation widget.

        add_orientation_widget
            Add any actor as an orientation widget.

        :ref:`axes_objects_example`
            Example showing different axes objects.

        Examples
        --------
        Show axes without labels and with thick lines.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Box(), show_edges=True)
        >>> _ = pl.add_axes(line_width=5, labels_off=True)
        >>> pl.show()

        Specify more parameters for the axes marker.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Box(), show_edges=True)
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
        if interactive is None:
            interactive = self._theme.interactive
        self._remove_axes_widget()
        if box is None:
            box = self._theme.axes.box
        if box:
            warnings.warn(
                '`box` is deprecated. Use `add_box_axes` or `add_color_box_axes` method instead.',
                PyVistaDeprecationWarning,
            )
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
            self.axes_actor,
            interactive=interactive,
            color=None,
        )
        axes_widget.SetViewport(viewport)
        return self.axes_actor

    @_deprecate_positional_args
    def add_north_arrow_widget(  # noqa: PLR0917
        self,
        interactive=None,
        color='#4169E1',
        opacity=1.0,
        line_width=2,
        edge_color=None,
        lighting=False,  # noqa: FBT002
        viewport=(0, 0, 0.1, 0.1),
    ):
        """Add a geographic north arrow to the scene.

        .. versionadded:: 0.44.0

        Parameters
        ----------
        interactive : bool, optional
            Control if the orientation widget is interactive.  By
            default uses the value from
            :attr:`pyvista.global_theme.interactive
            <pyvista.plotting.themes.Theme.interactive>`.

        color : ColorLike, optional
            Color of the north arrow.

        opacity : float, optional
            Opacity of the north arrow.

        line_width : float, optional
            Width of the north edge arrow lines.

        edge_color : ColorLike, optional
            Color of the edges.

        lighting : bool, optional
            Enable or disable lighting on north arrow.

        viewport : sequence[float], default: (0, 0, 0.1, 0.1)
            Viewport ``(xstart, ystart, xend, yend)`` of the widget.

        Returns
        -------
        :vtk:`vtkOrientationMarkerWidget`
            Orientation marker widget.

        See Also
        --------
        add_axes
            Add arrow-style axes as an orientation widget.

        add_box_axes
            Add an axes box as an orientation widget.

        add_north_arrow_widget
            Add north arrow as an orientation widget.

        :ref:`axes_objects_example`
            Example showing different axes objects.

        Examples
        --------
        Use an north arrow as the orientation widget.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> terrain = examples.download_st_helens().warp_by_scalar()
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(terrain)
        >>> widget = pl.add_north_arrow_widget()
        >>> pl.enable_terrain_style(mouse_wheel_zooms=True)
        >>> pl.show()

        """
        marker = create_north_arrow()
        mapper = pyvista.DataSetMapper(marker)
        actor = pyvista.Actor(mapper)
        actor.prop.show_edges = True
        if edge_color is not None:
            actor.prop.edge_color = edge_color
        actor.prop.line_width = line_width
        actor.prop.color = color
        actor.prop.opacity = opacity
        actor.prop.lighting = lighting
        return self.add_orientation_widget(
            actor,
            interactive=interactive,
            viewport=viewport,
        )

    def add_box_axes(
        self,
        *,
        interactive=None,
        line_width=2,
        text_scale=0.366667,
        edge_color='black',
        x_color=None,
        y_color=None,
        z_color=None,
        xlabel='X',
        ylabel='Y',
        zlabel='Z',
        x_face_color='white',
        y_face_color='white',
        z_face_color='white',
        label_color=None,
        labels_off=False,
        opacity=0.5,
        show_text_edges=False,
        viewport=(0, 0, 0.2, 0.2),
    ):
        """Add an interactive color box axes widget in the bottom left corner.

        Parameters
        ----------
        interactive : bool, optional
            Enable this orientation widget to be moved by the user.

        line_width : float, optional
            The width of the marker lines.

        text_scale : float, optional
            Size of the text relative to the faces.

        edge_color : ColorLike, optional
            Color of the edges.

        x_color : ColorLike, optional
            Color of the x-axis text.

        y_color : ColorLike, optional
            Color of the y-axis text.

        z_color : ColorLike, optional
            Color of the z-axis text.

        xlabel : str, optional
            Text used for the x-axis.

        ylabel : str, optional
            Text used for the y-axis.

        zlabel : str, optional
            Text used for the z-axis.

        x_face_color : ColorLike, optional
            Color used for the x-axis arrow.  Defaults to theme axes
            parameters.

        y_face_color : ColorLike, optional
            Color used for the y-axis arrow.  Defaults to theme axes
            parameters.

        z_face_color : ColorLike, optional
            Color used for the z-axis arrow.  Defaults to theme axes
            parameters.

        label_color : ColorLike, optional
            Color of the labels.

        labels_off : bool, optional
            Enable or disable the text labels for the axes.

        opacity : float, optional
            Opacity in the range of ``[0, 1]`` of the orientation box.

        show_text_edges : bool, optional
            Enable or disable drawing the vector text edges.

        viewport : sequence[float], default: (0, 0, 0.2, 0.2)
            Viewport ``(xstart, ystart, xend, yend)`` of the widget.

        Returns
        -------
        :vtk:`vtkAnnotatedCubeActor`
            Axes actor.

        See Also
        --------
        add_axes
            Add arrow-style axes as an orientation widget.

        add_north_arrow_widget
            Add north arrow as an orientation widget.

        add_orientation_widget
            Add any actor as an orientation widget.

        :ref:`axes_objects_example`
            Example showing different axes objects.

        Examples
        --------
        Use the axes orientation widget instead of the default arrows.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Sphere())
        >>> _ = pl.add_box_axes()
        >>> pl.show()

        """
        if interactive is None:
            interactive = self._theme.interactive
        self._remove_axes_widget()
        self.axes_actor = create_axes_orientation_box(
            line_width=line_width,
            text_scale=text_scale,
            edge_color=edge_color,
            x_color=x_color,
            y_color=y_color,
            z_color=z_color,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            x_face_color=x_face_color,
            y_face_color=y_face_color,
            z_face_color=z_face_color,
            color_box=True,
            label_color=label_color,
            labels_off=labels_off,
            opacity=opacity,
            show_text_edges=show_text_edges,
        )
        axes_widget = self.add_orientation_widget(
            self.axes_actor,
            interactive=interactive,
            color=None,
        )
        axes_widget.SetViewport(viewport)
        return self.axes_actor

    def hide_axes(self) -> None:
        """Hide the axes orientation widget.

        See Also
        --------
        show_axes
            Show the axes orientation widget.

        axes_enabled
            Check if the axes orientation widget is enabled.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.hide_axes()

        """
        if self.axes_widget is not None and self.axes_widget.GetEnabled():
            self.axes_widget.EnabledOff()
            self.Modified()

    def show_axes(self) -> None:
        """Show the axes orientation widget.

        See Also
        --------
        add_axes
            Similar method with additional options.

        hide_axes
            Hide the axes orientation widget.

        axes_enabled
            Check if the axes orientation widget is enabled.

        add_axes_at_origin
            Add a :class:`pyvista.AxesActor` to the origin of a scene.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.show_axes()

        """
        if self.axes_widget is not None:
            self.axes_widget.EnabledOn()
            self.axes_widget.SetCurrentRenderer(self)
        else:
            self.add_axes()
        self.Modified()

    @property
    def axes_enabled(self):
        """Return ``True`` when the axes widget is enabled.

        See Also
        --------
        show_axes
            Show the axes orientation widget.

        hide_axes
            Hide the axes orientation widget.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.hide_axes()
        >>> pl.renderer.axes_enabled
        False

        Returns
        -------
        bool
            Return ``True`` when the axes widget is enabled.

        """
        if self.axes_widget is not None:
            return bool(self.axes_widget.GetEnabled())
        return False

    @_deprecate_positional_args
    def show_bounds(  # noqa: PLR0917
        self,
        mesh=None,
        bounds=None,
        axes_ranges=None,
        show_xaxis=True,  # noqa: FBT002
        show_yaxis=True,  # noqa: FBT002
        show_zaxis=True,  # noqa: FBT002
        show_xlabels=True,  # noqa: FBT002
        show_ylabels=True,  # noqa: FBT002
        show_zlabels=True,  # noqa: FBT002
        bold=True,  # noqa: FBT002
        font_size=None,
        font_family=None,
        color=None,
        xtitle='X Axis',
        ytitle='Y Axis',
        ztitle='Z Axis',
        n_xlabels=5,
        n_ylabels=5,
        n_zlabels=5,
        use_2d=False,  # noqa: FBT002
        grid=None,
        location='closest',
        ticks=None,
        all_edges=False,  # noqa: FBT002
        corner_factor=0.5,
        fmt=None,
        minor_ticks=False,  # noqa: FBT002
        padding=0.0,
        use_3d_text=True,  # noqa: FBT002
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

            ``(x_min, x_max, y_min, y_max, z_min, z_max)``.

        show_xaxis : bool, default: True
            Makes x-axis visible.

        show_yaxis : bool, default: True
            Makes y-axis visible.

        show_zaxis : bool, default: True
            Makes z-axis visible.

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
            Title of the x-axis.  Default ``"X Axis"``.

        ytitle : str, default: "Y Axis"
            Title of the y-axis.  Default ``"Y Axis"``.

        ztitle : str, default: "Z Axis"
            Title of the z-axis.  Default ``"Z Axis"``.

        n_xlabels : int, default: 5
            Number of labels for the x-axis.

        n_ylabels : int, default: 5
            Number of labels for the y-axis.

        n_zlabels : int, default: 5
            Number of labels for the z-axis.

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
            Use :vtk:`vtkTextActor3D` for titles and labels.

        render : bool, optional
            If the render window is being shown, trigger a render
            after showing bounds.

        **kwargs : dict, optional
            Deprecated keyword arguments.

        Returns
        -------
        pyvista.CubeAxesActor
            Bounds actor.

        See Also
        --------
        show_grid
        remove_bounds_axes
        update_bounds_axes

        :ref:`axes_objects_example`
            Example showing different axes objects.
        :ref:`bounds_example`
            Additional examples using this method.

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
        >>> actor = plotter.add_mesh(mesh, cmap='terrain', show_scalar_bar=False)
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
        >>> actor = plotter.add_mesh(mesh, cmap='terrain', show_scalar_bar=False)
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
            # TODO: Change this to (9, 6, 0) when VTK 9.6 is released
            fmt = '%.1f' if pyvista.vtk_version_info < (9, 5, 99) else '{0:.1f}'  # fallback

        if 'xlabel' in kwargs:  # pragma: no cover
            xtitle = kwargs.pop('xlabel')
            warnings.warn(
                '`xlabel` is deprecated. Use `xtitle` instead.',
                PyVistaDeprecationWarning,
            )
        if 'ylabel' in kwargs:  # pragma: no cover
            ytitle = kwargs.pop('ylabel')
            warnings.warn(
                '`ylabel` is deprecated. Use `ytitle` instead.',
                PyVistaDeprecationWarning,
            )
        if 'zlabel' in kwargs:  # pragma: no cover
            ztitle = kwargs.pop('zlabel')
            warnings.warn(
                '`zlabel` is deprecated. Use `ztitle` instead.',
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
                msg = f'`grid` must be a str, not {type(grid)}'
                raise TypeError(msg)
            grid = grid.lower()
            if grid in ('front', 'frontface'):
                cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_CLOSEST)
            elif grid in ('both', 'all'):
                cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_ALL)
            elif grid in ('back', True):
                cube_axes_actor.SetGridLineLocation(cube_axes_actor.VTK_GRID_LINES_FURTHEST)
            else:
                msg = f'`grid` must be either "front", "back, or, "all", not {grid}'
                raise ValueError(msg)
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
                msg = (
                    f'Value of location ("{location}") should be either "all", "origin",'
                    ' "outer", "default", "closest", "front", "furthest", or "back".'
                )
                raise ValueError(msg)
        elif location is not None:
            msg = 'location must be a string'
            raise TypeError(msg)

        if isinstance(padding, (int, float)) and 0.0 <= padding < 1.0:
            if not np.any(np.abs(bounds) == np.inf):
                cushion = (
                    np.array(
                        [
                            np.abs(bounds[1] - bounds[0]),
                            np.abs(bounds[3] - bounds[2]),
                            np.abs(bounds[5] - bounds[4]),
                        ],
                    )
                    * padding
                )
                bounds[::2] -= cushion
                bounds[1::2] += cushion
        else:
            msg = f'padding ({padding}) not understood. Must be float between 0 and 1'
            raise ValueError(msg)
        cube_axes_actor.bounds = bounds

        # set axes ranges if input
        if axes_ranges is not None:
            if isinstance(axes_ranges, (Sequence, np.ndarray)):
                axes_ranges = np.asanyarray(axes_ranges)
            else:
                msg = 'Input axes_ranges must be a numeric sequence.'
                raise TypeError(msg)

            if not np.issubdtype(axes_ranges.dtype, np.number):
                msg = 'All of the elements of axes_ranges must be numbers.'
                raise TypeError(msg)

            # set the axes ranges
            if axes_ranges.shape != (6,):
                msg = (
                    '`axes_ranges` must be passed as a '
                    '(x_min, x_max, y_min, y_max, z_min, z_max) sequence.'
                )
                raise ValueError(msg)

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

        # For 3D text, use `SetFontSize` to a relatively high value and use `SetScreenSize` to
        # shrink it back down. This creates a higher-resolution font and makes it appear sharper.
        # In VTK 9.6+, the 3D font size is also tied to the value set by SetFontSize, so we need
        # an additional scaling factor.
        default_screen_size = 10.0
        default_font_size = 12
        scaled_font_size = 50

        font_size_factor = (
            scaled_font_size / default_font_size if pyvista.vtk_version_info > (9, 5, 99) else 1.0
        )
        for prop in props:
            prop.SetColor(color.float_rgb)
            prop.SetFontFamily(font_family)
            prop.SetBold(bold)

            # this merely makes the font sharper
            if use_3d_text:
                prop.SetFontSize(scaled_font_size)

        cube_axes_actor.SetScreenSize(
            font_size / default_font_size / font_size_factor * default_screen_size
        )

        if all_edges:
            self.add_bounding_box(color=color, corner_factor=corner_factor)

        self.add_actor(cube_axes_actor, reset_camera=False, pickable=False, render=render)
        self.cube_axes_actor = cube_axes_actor

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
        pyvista.CubeAxesActor
            Bounds actor.

        See Also
        --------
        show_bounds
        remove_bounds_axes
        update_bounds_axes

        :ref:`axes_objects_example`
            Example showing different axes objects.
        :ref:`bounds_example`
            Additional examples using this method.

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

    @_deprecate_positional_args
    def remove_bounding_box(self, render=True) -> None:  # noqa: FBT002
        """Remove bounding box.

        Parameters
        ----------
        render : bool, default: True
            Trigger a render once the bounding box is removed.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = pl.add_bounding_box()
        >>> pl.remove_bounding_box()

        """
        if self._box_object is not None:
            actor = self.bounding_box_actor
            self.bounding_box_actor = None
            self._box_object = None
            self.remove_actor(actor, reset_camera=False, render=render)
            self.Modified()

    @_deprecate_positional_args
    def add_bounding_box(  # noqa: PLR0917
        self,
        color='grey',
        corner_factor=0.5,
        line_width=None,
        opacity=1.0,
        render_lines_as_tubes=False,  # noqa: FBT002
        lighting=None,
        reset_camera=None,
        outline=True,  # noqa: FBT002
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
        :vtk:`vtkActor`
            VTK actor of the bounding box.

        See Also
        --------
        pyvista.DataSetFilters.bounding_box
            Create a bounding box or oriented bounding box for a dataset.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Sphere())
        >>> _ = pl.add_bounding_box(line_width=5, color='black')
        >>> pl.show()

        """
        if lighting is None:
            lighting = self._theme.lighting

        self.remove_bounding_box()
        box: _vtk.vtkOutlineCornerSource | _vtk.vtkCubeSource
        if outline:
            source = _vtk.vtkOutlineCornerSource()
            source.SetCornerFactor(corner_factor)
            box = source
        else:
            box = _vtk.vtkCubeSource()
        box.SetBounds(self.bounds)
        box.Update()
        box_object = wrap(box.GetOutput())
        self._bounding_box = box
        self._box_object = box_object
        name = f'BoundingBox({hex(id(box_object))})'

        mapper = _vtk.vtkDataSetMapper()
        mapper.SetInputData(box_object)
        self.bounding_box_actor, prop = self.add_actor(
            mapper,
            reset_camera=reset_camera,
            name=name,
            culling=culling,
            pickable=False,
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

    @_deprecate_positional_args(allowed=['face'])
    def add_floor(  # noqa: PLR0917
        self,
        face='-z',
        i_resolution=10,
        j_resolution=10,
        color=None,
        line_width=None,
        opacity=1.0,
        show_edges=False,  # noqa: FBT002
        lighting=False,  # noqa: FBT002
        edge_color=None,
        reset_camera=None,
        pad=0.0,
        offset=0.0,
        pickable=False,  # noqa: FBT002
        store_floor_kwargs=True,  # noqa: FBT002
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

        pickable : bool, default: False
            Make this floor actor pickable in the renderer.

        store_floor_kwargs : bool, default: True
            Stores the keyword arguments used when adding this floor.
            Useful when updating the bounds and regenerating the
            floor.

        Returns
        -------
        :vtk:`vtkActor`
            VTK actor of the floor.

        Examples
        --------
        Add a floor below a sphere and plot it.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Sphere())
        >>> actor = pl.add_floor()
        >>> pl.show()

        """
        if store_floor_kwargs:
            kwargs = locals()
            kwargs.pop('self')
            self._floor_kwargs.append(kwargs)
        ranges = np.ptp(np.array(self.bounds).reshape(-1, 2), axis=1)
        ranges += ranges * pad
        center = np.array(self.center)
        if face.lower() in '-z':
            center[2] = self.bounds.z_min - (ranges[2] * offset)
            normal = (0, 0, 1)
            i_size = ranges[0]
            j_size = ranges[1]
        elif face.lower() in '-y':
            center[1] = self.bounds.y_min - (ranges[1] * offset)
            normal = (0, 1, 0)
            i_size = ranges[2]
            j_size = ranges[0]
        elif face.lower() in '-x':
            center[0] = self.bounds.x_min - (ranges[0] * offset)
            normal = (1, 0, 0)
            i_size = ranges[2]
            j_size = ranges[1]
        elif face.lower() in '+z':
            center[2] = self.bounds.z_max + (ranges[2] * offset)
            normal = (0, 0, -1)
            i_size = ranges[0]
            j_size = ranges[1]
        elif face.lower() in '+y':
            center[1] = self.bounds.y_max + (ranges[1] * offset)
            normal = (0, -1, 0)
            i_size = ranges[2]
            j_size = ranges[0]
        elif face.lower() in '+x':
            center[0] = self.bounds.x_max + (ranges[0] * offset)
            normal = (-1, 0, 0)
            i_size = ranges[2]
            j_size = ranges[1]
        else:
            msg = f'Face ({face}) not implemented'
            raise NotImplementedError(msg)
        floor = pyvista.Plane(
            center=center,
            direction=normal,
            i_size=i_size,
            j_size=j_size,
            i_resolution=i_resolution,
            j_resolution=j_resolution,
        )
        floor.clear_data()
        self._floor = floor

        if lighting is None:
            lighting = self._theme.lighting

        self.remove_bounding_box()
        mapper = DataSetMapper()
        mapper.SetInputData(self._floor)
        actor, prop = self.add_actor(
            mapper,
            reset_camera=reset_camera,
            name=f'Floor({face})',
            pickable=pickable,
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

    @_deprecate_positional_args
    def remove_floors(self, clear_kwargs=True, render=True) -> None:  # noqa: FBT002
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

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Sphere())
        >>> actor = pl.add_floor()
        >>> pl.remove_floors()
        >>> pl.show()

        """
        if hasattr(self, '_floor') and self._floor is not None:
            self._floor.ReleaseData()
            self._floor = None
        for actor in self._floors:
            self.remove_actor(actor, reset_camera=False, render=render)
        self._floors.clear()
        if clear_kwargs:
            self._floor_kwargs.clear()

    def remove_bounds_axes(self) -> None:
        """Remove bounds axes.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> actor = pl.add_mesh(pv.Sphere())
        >>> actor = pl.show_bounds(grid='front', location='outer')
        >>> pl.subplot(0, 1)
        >>> actor = pl.add_mesh(pv.Sphere())
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
        light : :vtk:`vtkLight` | Light
            Light to add.

        """
        # convert from a vtk type if applicable
        if isinstance(light, _vtk.vtkLight) and not isinstance(light, pyvista.Light):
            light = pyvista.Light.from_vtk(light)

        if not isinstance(light, pyvista.Light):
            msg = f'Expected Light instance, got {type(light).__name__} instead.'
            raise TypeError(msg)
        self._lights.append(light)
        self.AddLight(light)
        self.Modified()

        # we add the renderer to add/remove the light actor if
        # positional or cone angle is modified
        light.add_renderer(self)

    @property
    def lights(self):  # numpydoc ignore=RT01
        """Return a list of all lights in the renderer.

        Returns
        -------
        list
            Lights in the renderer.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.renderer.lights  # doctest:+SKIP
        [<Light (Headlight) at ...>,
         <Light (Camera Light) at 0x7f1dd8155760>,
         <Light (Camera Light) at 0x7f1dd8155340>,
         <Light (Camera Light) at 0x7f1dd8155460>,
         <Light (Camera Light) at 0x7f1dd8155f40>]

        """
        return list(self.GetLights())  # type: ignore[call-overload]

    def remove_all_lights(self) -> None:
        """Remove all lights from the renderer."""
        self.RemoveAllLights()
        self._lights.clear()

    def clear_actors(self) -> None:
        """Remove all actors (keep lights and properties)."""
        if self._actors:
            for actor in list(self._actors):
                with contextlib.suppress(KeyError):
                    self.remove_actor(actor, reset_camera=False, render=False)
            self.Modified()

    def clear(self) -> None:
        """Remove all actors and properties.

        See Also
        --------
        :ref:`clear_example`

        """
        self.clear_actors()
        if self._charts is not None:
            self._charts.deep_clean()
        self.remove_all_lights()
        self.RemoveAllViewProps()
        self.Modified()

        self._scalar_bar_slots = set(range(MAX_N_COLOR_BARS))
        self._scalar_bar_slot_lookup = {}

    def set_focus(self, point) -> None:
        """Set focus to a point.

        Parameters
        ----------
        point : sequence[float]
            Cartesian point to focus on in the form of ``[x, y, z]``.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, show_edges=True)
        >>> _ = pl.add_point_labels([mesh.points[1]], ['Focus'])
        >>> _ = pl.camera  # this initializes the camera
        >>> pl.set_focus(mesh.points[1])
        >>> pl.show()

        """
        if isinstance(point, np.ndarray) and point.ndim != 1:
            point = point.ravel()
        self.camera.focal_point = scale_point(self.camera, point, invert=False)
        self.camera_set = True
        self.Modified()

    @_deprecate_positional_args(allowed=['point'])
    def set_position(self, point, reset=False, render=True) -> None:  # noqa: FBT002
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

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, show_edges=True)
        >>> pl.set_position([7, 7, 7])
        >>> pl.show()

        """
        if isinstance(point, np.ndarray) and point.ndim != 1:
            point = point.ravel()
        self.camera.position = scale_point(self.camera, point, invert=False)
        if reset:
            self.reset_camera(render=render)
        self.camera_set = True
        self.Modified()

    @_deprecate_positional_args(allowed=['vector'])
    def set_viewup(self, vector, reset=True, render=True) -> None:  # noqa: FBT002
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
        Notice how the y-axis appears vertical.

        >>> from pyvista import demos
        >>> pl = demos.orientation_plotter()
        >>> pl.set_viewup([0, 1, 0])
        >>> pl.show()

        """
        if isinstance(vector, np.ndarray) and vector.ndim != 1:
            vector = vector.ravel()

        self.camera.up = vector
        if reset:
            self.reset_camera(render=render)

        self.camera_set = True
        self.Modified()

    def enable_parallel_projection(self) -> None:
        """Enable parallel projection.

        The camera will have a parallel projection. Parallel projection is
        often useful when viewing images or 2D datasets.

        See Also
        --------
        pyvista.Plotter.enable_2d_style

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import demos
        >>> pl = pv.demos.orientation_plotter()
        >>> pl.enable_parallel_projection()
        >>> pl.show()

        """
        # Fix the 'reset camera' effect produced by the VTK when parallel
        # projection is enabled.
        angle = np.radians(self.camera.view_angle)
        self.camera.parallel_scale = self.camera.distance * np.sin(0.5 * angle)

        self.camera.enable_parallel_projection()
        self.Modified()

    def disable_parallel_projection(self) -> None:
        """Reset the camera to use perspective projection.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista import demos
        >>> pl = pv.demos.orientation_plotter()
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
    def parallel_projection(self):  # numpydoc ignore=RT01
        """Return parallel projection state of active render window.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.parallel_projection = False
        >>> pl.parallel_projection
        False

        """
        return self.camera.parallel_projection

    @parallel_projection.setter
    def parallel_projection(self, state) -> None:
        self.camera.parallel_projection = state
        self.Modified()

    @property
    def parallel_scale(self):  # numpydoc ignore=RT01
        """Return parallel scale of active render window.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.parallel_scale = 2

        """
        return self.camera.parallel_scale

    @parallel_scale.setter
    def parallel_scale(self, value) -> None:
        self.camera.parallel_scale = value
        self.Modified()

    @_deprecate_positional_args(allowed=['actor'])
    def remove_actor(self, actor, reset_camera=False, render=True):  # noqa: FBT002
        """Remove an actor from the Renderer.

        Parameters
        ----------
        actor : str | :vtk:`vtkActor` | list | tuple
            If the type is ``str``, removes the previously added actor
            with the given name. If the type is :vtk:`vtkActor`,
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

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> pl = pv.Plotter()
        >>> cube_actor = pl.add_mesh(pv.Cube(), show_edges=True)
        >>> sphere_actor = pl.add_mesh(pv.Sphere(), show_edges=True)
        >>> _ = pl.remove_actor(cube_actor)
        >>> pl.show()

        """
        if isinstance(actor, str):
            name = actor
            keys = list(self._actors.keys())
            names = [k for k in keys if k.startswith(f'{name}-')]
            if len(names) > 0:
                self.remove_actor(names, reset_camera=reset_camera, render=render)
            try:
                actor = self._actors[name]
            except KeyError:
                # If actor of that name is not present then return success
                return False
        if isinstance(actor, Iterable):
            success = False
            for a in actor:
                rv = self.remove_actor(a, reset_camera=reset_camera, render=render)
                if rv or success:
                    success = True
            return success
        if actor is None:
            return False

        # remove any labels associated with the actor
        self._labels.pop(actor.GetAddressAsString(''), None)

        # ensure any scalar bars associated with this actor are removed
        with contextlib.suppress(AttributeError, ReferenceError):
            self.parent.scalar_bars._remove_mapper_from_plotter(actor)
        self.RemoveActor(actor)
        self.update_bounds_axes()
        if reset_camera or (not self.camera_set and reset_camera is None):
            self.reset_camera(render=render)
        elif render:
            self.parent.render()

        self.Modified()
        return True

    @_deprecate_positional_args(allowed=['xscale', 'yscale', 'zscale'])
    def set_scale(  # noqa: PLR0917
        self,
        xscale=None,
        yscale=None,
        zscale=None,
        reset_camera=True,  # noqa: FBT002
        render=True,  # noqa: FBT002
    ) -> None:
        """Scale all the actors in the scene.

        Scaling in performed independently on the X, Y and z-axis.
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

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.set_scale(zscale=2)
        >>> _ = pl.add_mesh(pv.Sphere())  # perfect sphere
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

    @_deprecate_positional_args
    def get_default_cam_pos(self, negative=False):  # noqa: FBT002
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
        position = np.array(self._theme.camera.position).astype(float)
        if negative:
            position *= -1
        position = position / np.array(self.scale).astype(float)
        return [position + np.array(focal_pt), focal_pt, self._theme.camera.viewup]

    def update_bounds_axes(self) -> None:
        """Update the bounds axes of the render window."""
        if self._box_object is not None and self.bounding_box_actor is not None:
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

    @_deprecate_positional_args
    def reset_camera(self, render=True, bounds=None) -> None:  # noqa: FBT002
        """Reset the camera of the active render window.

        The camera slides along the vector defined from camera
        position to focal point until all of the actors can be seen.

        Parameters
        ----------
        render : bool, default: True
            Trigger a render after resetting the camera.

        bounds : iterable(int), optional
            Automatically set up the camera based on a specified bounding box
            ``(x_min, x_max, y_min, y_max, z_min, z_max)``.

        Examples
        --------
        Add a mesh and place the camera position too close to the
        mesh.  Then reset the camera and show the mesh.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Sphere(), show_edges=True)
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

    def isometric_view(self) -> None:
        """Reset the camera to a default isometric view.

        DEPRECATED: Please use ``view_isometric``.

        """
        self.view_isometric()

    @_deprecate_positional_args
    def view_isometric(self, negative=False, render=True, bounds=None) -> None:  # noqa: FBT002
        """Reset the camera to a default isometric view.

        The view will show all the actors in the scene.

        Parameters
        ----------
        negative : bool, default: False
            View from the other isometric direction.

        render : bool, default: True
            If the render window is being shown, trigger a render
            after setting the camera position.

        bounds : iterable(int), optional
            Automatically set up the camera based on a specified bounding box
            ``(x_min, x_max, y_min, y_max, z_min, z_max)``.

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
        self.reset_camera(render=render, bounds=bounds)

    @_deprecate_positional_args(allowed=['vector', 'viewup'])
    def view_vector(  # noqa: PLR0917
        self,
        vector,
        viewup=None,
        render=True,  # noqa: FBT002
        bounds=None,
    ) -> None:
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

        bounds : iterable(int), optional
            Automatically set up the camera based on a specified bounding box
            ``(x_min, x_max, y_min, y_max, z_min, z_max)``.

        """
        focal_pt = self.center
        if viewup is None:
            viewup = self._theme.camera.viewup
        cpos = CameraPosition(vector + np.array(focal_pt), focal_pt, viewup)
        self.camera_position = cpos
        self.reset_camera(render=render, bounds=bounds)

    @_deprecate_positional_args
    def view_xy(self, negative=False, render=True, bounds=None) -> None:  # noqa: FBT002
        """View the XY plane.

        Parameters
        ----------
        negative : bool, default: False
            View from the opposite direction.

        render : bool, default: True
            If the render window is being shown, trigger a render
            after setting the camera position.

        bounds : iterable(int), optional
            Automatically set up the camera based on a specified bounding box
            ``(x_min, x_max, y_min, y_max, z_min, z_max)``.

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
        self.view_vector(*view_vectors('xy', negative=negative), render=render, bounds=bounds)

    @_deprecate_positional_args
    def view_yx(self, negative=False, render=True, bounds=None) -> None:  # noqa: FBT002
        """View the YX plane.

        Parameters
        ----------
        negative : bool, default: False
            View from the opposite direction.

        render : bool, default: True
            If the render window is being shown, trigger a render
            after setting the camera position.

        bounds : iterable(int), optional
            Automatically set up the camera based on a specified bounding box
            ``(x_min, x_max, y_min, y_max, z_min, z_max)``.

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
        self.view_vector(*view_vectors('yx', negative=negative), render=render, bounds=bounds)

    @_deprecate_positional_args
    def view_xz(self, negative=False, render=True, bounds=None) -> None:  # noqa: FBT002
        """View the XZ plane.

        Parameters
        ----------
        negative : bool, default: False
            View from the opposite direction.

        render : bool, default: True
            If the render window is being shown, trigger a render
            after setting the camera position.

        bounds : iterable(int), optional
            Automatically set up the camera based on a specified bounding box
            ``(x_min, x_max, y_min, y_max, z_min, z_max)``.

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
        self.view_vector(*view_vectors('xz', negative=negative), render=render, bounds=bounds)

    @_deprecate_positional_args
    def view_zx(self, negative=False, render=True, bounds=None) -> None:  # noqa: FBT002
        """View the ZX plane.

        Parameters
        ----------
        negative : bool, default: False
            View from the opposite direction.

        render : bool, default: True
            If the render window is being shown, trigger a render
            after setting the camera position.

        bounds : iterable(int), optional
            Automatically set up the camera based on a specified bounding box
            ``(x_min, x_max, y_min, y_max, z_min, z_max)``.

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
        self.view_vector(*view_vectors('zx', negative=negative), render=render, bounds=bounds)

    @_deprecate_positional_args
    def view_yz(self, negative=False, render=True, bounds=None) -> None:  # noqa: FBT002
        """View the YZ plane.

        Parameters
        ----------
        negative : bool, default: False
            View from the opposite direction.

        render : bool, default: True
            If the render window is being shown, trigger a render
            after setting the camera position.

        bounds : iterable(int), optional
            Automatically set up the camera based on a specified bounding box
            ``(x_min, x_max, y_min, y_max, z_min, z_max)``.

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
        self.view_vector(*view_vectors('yz', negative=negative), render=render, bounds=bounds)

    @_deprecate_positional_args
    def view_zy(self, negative=False, render=True, bounds=None) -> None:  # noqa: FBT002
        """View the ZY plane.

        Parameters
        ----------
        negative : bool, default: False
            View from the opposite direction.

        render : bool, default: True
            If the render window is being shown, trigger a render
            after setting the camera position.

        bounds : iterable(int), optional
            Automatically set up the camera based on a specified bounding box
            ``(x_min, x_max, y_min, y_max, z_min, z_max)``.

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
        self.view_vector(*view_vectors('zy', negative=negative), render=render, bounds=bounds)

    def disable(self) -> None:
        """Disable this renderer's camera from being interactive."""
        self.SetInteractive(0)

    def enable(self) -> None:
        """Enable this renderer's camera to be interactive."""
        self.SetInteractive(1)

    def add_blurring(self) -> None:
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

        See :ref:`blurring_example` for a full example using this method.

        """
        self._render_passes.add_blur_pass()

    def remove_blurring(self) -> None:
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

    @_deprecate_positional_args
    def enable_depth_of_field(self, automatic_focal_distance=True) -> None:  # noqa: FBT002
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
        >>> pl = pv.Plotter(lighting='three lights')
        >>> pl.background_color = 'w'
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
        self._render_passes.enable_depth_of_field_pass(
            automatic_focal_distance=automatic_focal_distance
        )

    def disable_depth_of_field(self) -> None:
        """Disable depth of field plotting.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter(lighting='three lights')
        >>> pl.enable_depth_of_field()
        >>> pl.disable_depth_of_field()

        """
        self._render_passes.disable_depth_of_field_pass()

    def enable_eye_dome_lighting(self) -> None:
        """Enable eye dome lighting (EDL).

        Returns
        -------
        :vtk:`vtkOpenGLRenderer`
            VTK renderer with eye dome lighting pass.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = pl.enable_eye_dome_lighting()

        """
        self._render_passes.enable_edl_pass()

    def disable_eye_dome_lighting(self) -> None:
        """Disable eye dome lighting (EDL).

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.disable_eye_dome_lighting()

        """
        self._render_passes.disable_edl_pass()

    def enable_shadows(self) -> None:
        """Enable shadows.

        Examples
        --------
        First, plot without shadows enabled (default)

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> pl = pv.Plotter(lighting='none', window_size=(1000, 1000))
        >>> light = pv.Light()
        >>> light.set_direction_angle(20, -20)
        >>> pl.add_light(light)
        >>> _ = pl.add_mesh(mesh, color='white', smooth_shading=True)
        >>> _ = pl.add_mesh(pv.Box((-1.2, -1, -1, 1, -1, 1)))
        >>> pl.show()

        Now, enable shadows.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> pl = pv.Plotter(lighting='none', window_size=(1000, 1000))
        >>> light = pv.Light()
        >>> light.set_direction_angle(20, -20)
        >>> pl.add_light(light)
        >>> _ = pl.add_mesh(mesh, color='white', smooth_shading=True)
        >>> _ = pl.add_mesh(pv.Box((-1.2, -1, -1, 1, -1, 1)))
        >>> pl.enable_shadows()
        >>> pl.show()

        """
        self._render_passes.enable_shadow_pass()

    def disable_shadows(self) -> None:
        """Disable shadows.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> pl.disable_shadows()

        """
        self._render_passes.disable_shadow_pass()

    @_deprecate_positional_args
    def enable_ssao(  # noqa: PLR0917
        self,
        radius=0.5,
        bias=0.005,
        kernel_size=256,
        blur=True,  # noqa: FBT002
    ) -> None:
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

        See Also
        --------
        :ref:`ssao_example`

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
        self._render_passes.enable_ssao_pass(
            radius=radius, bias=bias, kernel_size=kernel_size, blur=blur
        )

    def disable_ssao(self) -> None:
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

    @_deprecate_positional_args(allowed=['color'])
    def set_background(  # noqa: PLR0917
        self, color, top=None, right=None, side=None, corner=None
    ):
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

        right : ColorLike, optional
            If given, this will enable a gradient background where the
            ``color`` argument is at the left and the color given in
            ``right`` will be the color at the right of the renderer.

        side : ColorLike, optional
            If given, this will enable a gradient background where the
            ``color`` argument is at the center and the color given in
            ``side`` will be the color at the side of the renderer.

        corner : ColorLike, optional
            If given, this will enable a gradient background where the
            ``color`` argument is at the center and the color given in
            ``corner`` will be the color at the corner of the renderer.

        Examples
        --------
        Set the background color to black with a gradient to white at
        the top of the plot.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Cone())
        >>> pl.set_background('black', top='white')
        >>> pl.show()

        """
        self.SetBackground(Color(color, default_color=self._theme.background).float_rgb)
        if not (right is side is corner is None) and vtk_version_info < (
            9,
            3,
        ):  # pragma: no cover
            from pyvista.core.errors import VTKVersionError

            msg = (
                '`right` or `side` or `corner` cannot be used under VTK v9.3.0. '
                'Try installing VTK v9.3.0 or newer.'
            )
            raise VTKVersionError(msg)
        if not (
            (top is right is side is corner is None)
            or (top is not None and right is side is corner is None)
            or (right is not None and top is side is corner is None)
            or (side is not None and top is right is corner is None)
            or (corner is not None and top is right is side is None)
        ):  # pragma: no cover
            msg = 'You can only set one argument in top, right, side, corner.'
            raise ValueError(msg)
        if top is not None:
            self.SetGradientBackground(True)
            self.SetBackground2(Color(top).float_rgb)
        elif right is not None:  # pragma: no cover
            self.SetGradientBackground(True)
            self.SetGradientMode(_vtk.vtkViewport.GradientModes.VTK_GRADIENT_HORIZONTAL)
            self.SetBackground2(Color(right).float_rgb)
        elif side is not None:  # pragma: no cover
            self.SetGradientBackground(True)
            self.SetGradientMode(
                _vtk.vtkViewport.GradientModes.VTK_GRADIENT_RADIAL_VIEWPORT_FARTHEST_SIDE,
            )
            self.SetBackground2(Color(side).float_rgb)
        elif corner is not None:  # pragma: no cover
            self.SetGradientBackground(True)
            self.SetGradientMode(
                _vtk.vtkViewport.GradientModes.VTK_GRADIENT_RADIAL_VIEWPORT_FARTHEST_CORNER,
            )
            self.SetBackground2(Color(corner).float_rgb)
        else:
            self.SetGradientBackground(False)
        self.Modified()

    @_deprecate_positional_args(allowed=['texture'])
    def set_environment_texture(
        self,
        texture,
        is_srgb=False,  # noqa: FBT002
        resample: bool | float | None = None,  # noqa: FBT001
    ) -> None:
        """Set the environment texture used for image based lighting.

        This texture is supposed to represent the scene background. If
        it is not a cubemap, the texture is supposed to represent an
        equirectangular projection. If used with raytracing backends,
        the texture must be an equirectangular projection and must be
        constructed with a valid :vtk:`vtkImageData`.

        Parameters
        ----------
        texture : pyvista.Texture
            Texture.

        is_srgb : bool, default: False
            If the texture is in sRGB color space, set the color flag on the
            texture or set this parameter to ``True``. Textures are assumed
            to be in linear color space by default.

        resample : bool | float, optional
            Resample the environment texture. Set this to a float to set the
            sampling rate explicitly or set to ``True`` to downsample the
            texture to 1/16th of its original resolution. By default, the
            theme value for ``resample_environment_texture`` is used, which
            is ``False`` for the standard theme.

            Downsampling the texture can substantially improve performance for
            some environments, e.g. headless setups or if GPU support is limited.

            .. note::

                This will resample the texture used for image-based lighting only,
                e.g. the texture used for rendering reflective surfaces. It
                does `not` resample the background texture.

            .. versionadded:: 0.45

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
        >>> _ = pl.add_mesh(pv.Sphere(), pbr=True, metallic=0.9, roughness=0.4)
        >>> pl.set_environment_texture(cubemap)
        >>> pl.camera_position = 'xy'
        >>> pl.show()

        """
        # cube_map textures cannot use spherical harmonics
        if texture.cube_map:
            self.AutomaticLightCreationOff()
            self.UseSphericalHarmonicsOff()

        self.UseImageBasedLightingOn()

        if resample is None:
            resample = pyvista.global_theme.resample_environment_texture

        if resample:
            resample = 1 / 16 if resample is True else resample

            # Copy the texture
            # TODO: use Texture.copy() once support for cubemaps is added, see https://github.com/pyvista/pyvista/issues/7300
            texture_copy = pyvista.Texture()  # type: ignore[abstract]
            texture_copy.cube_map = texture.cube_map
            texture_copy.mipmap = texture.mipmap
            texture_copy.interpolate = texture.interpolate
            texture_copy.color_mode = texture.color_mode

            # Resample the texture's images
            for i in range(6 if texture_copy.cube_map else 1):
                texture_copy.SetInputDataObject(
                    i, pyvista.wrap(texture.GetInputDataObject(i, 0)).resample(resample)
                )
            self.SetEnvironmentTexture(texture_copy, is_srgb)
        else:
            self.SetEnvironmentTexture(texture, is_srgb)

        self.SetBackgroundTexture(texture)
        self.Modified()

    def remove_environment_texture(self) -> None:
        """Remove the environment texture.

        Examples
        --------
        >>> from pyvista import examples
        >>> import pyvista as pv
        >>> pl = pv.Plotter(lighting=None)
        >>> cubemap = examples.download_sky_box_cube_map()
        >>> _ = pl.add_mesh(pv.Sphere(), pbr=True, metallic=0.9, roughness=0.4)
        >>> pl.set_environment_texture(cubemap)
        >>> pl.remove_environment_texture()
        >>> pl.camera_position = 'xy'
        >>> pl.show()

        """
        self.UseImageBasedLightingOff()
        self.SetEnvironmentTexture(None)
        self.SetBackgroundTexture(None)  # type: ignore[arg-type]
        self.Modified()

    def close(self) -> None:
        """Close out widgets and sensitive elements."""
        self.RemoveAllObservers()
        self._remove_axes_widget()

        self._bounding_box = None
        self._box_object = None
        self._marker_actor = None

        if self._empty_str is not None:
            self._empty_str.SetReferenceCount(0)
            self._empty_str = None

        # Remove ref to `vtkPropCollection` held by vtkRenderer
        if hasattr(self, '_actors'):
            del self._actors

        self._closed = True

    def on_plotter_render(self) -> None:
        """Notify renderer components of explicit plotter render call."""
        if self._charts is not None:
            for chart in self._charts:
                # Notify Charts that plotter.render() is called
                chart._render_event(plotter_render=True)

    @_deprecate_positional_args
    def deep_clean(self, render=False) -> None:  # noqa: FBT002
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
        if self._box_object is not None:
            self.remove_bounding_box(render=render)
        if self._shadow_pass is not None:
            self.disable_shadows()  # type: ignore[unreachable]
        try:
            if self._charts is not None:
                self._charts.deep_clean()
                self._charts = None
        except AttributeError:  # pragma: no cover
            pass

        self._render_passes.deep_clean()
        self.remove_floors(render=render)
        self.remove_legend(render=render)
        self.RemoveAllViewProps()
        self._camera = None
        self._bounding_box = None
        self._marker_actor = None
        self._border_actor = None
        self._box_object = None
        # remove reference to parent last
        self.parent = None

    def __del__(self) -> None:
        """Delete the renderer."""
        self.deep_clean()

    def enable_hidden_line_removal(self) -> None:
        """Enable hidden line removal."""
        self.UseHiddenLineRemovalOn()

    def disable_hidden_line_removal(self) -> None:
        """Disable hidden line removal."""
        self.UseHiddenLineRemovalOff()

    @property
    def layer(self):  # numpydoc ignore=RT01
        """Return or set the current layer of this renderer."""
        return self.GetLayer()

    @layer.setter
    def layer(self, layer) -> None:
        self.SetLayer(layer)

    @property
    def viewport(self):  # numpydoc ignore=RT01
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

        >>> import pyvista as pv
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> _ = pl.add_mesh(pv.Sphere())
        >>> pl.renderers[0].viewport
        (0.0, 0.0, 0.5, 1.0)

        Change viewport to half size.

        >>> pl.renderers[0].viewport = (0.125, 0.25, 0.375, 0.75)
        >>> pl.show()

        """
        return self.GetViewport()

    @viewport.setter
    def viewport(self, viewport) -> None:
        self.SetViewport(viewport)

    @property
    def width(self):  # numpydoc ignore=RT01
        """Width of the renderer."""
        xmin, _, xmax, _ = self.viewport
        return self.parent.window_size[0] * (xmax - xmin)

    @property
    def height(self):  # numpydoc ignore=RT01
        """Height of the renderer."""
        _, ymin, _, ymax = self.viewport
        return self.parent.window_size[1] * (ymax - ymin)

    @_deprecate_positional_args(allowed=['labels'])
    def add_legend(  # noqa: PLR0917
        self,
        labels=None,
        bcolor=None,
        border=False,  # noqa: FBT002
        size=(0.2, 0.2),
        name=None,
        loc='upper right',
        face=None,
        font_family=None,
        background_opacity=1.0,
    ):
        """Add a legend to render window.

        Entries must be a list containing one string and color entry for each
        item.

        Parameters
        ----------
        labels : list | dict, optional
            When set to ``None``, uses existing labels as specified by

            - :func:`add_mesh <Plotter.add_mesh>`
            - :func:`add_lines <Plotter.add_lines>`
            - :func:`add_points <Plotter.add_points>`

            For dict inputs, the keys are used as labels and the values are used
            as the colors. Labels must be strings, and colors can be any
            :class:`~pyvista.ColorLike`.

            For list inputs, the list must contain one entry for each item to
            be added to the legend. Each entry can contain one of the following:

            * Two strings ([label, color]), where ``label`` is the name of the
              item to add, and ``color`` is the color of the label to add.
            * Three strings ([label, color, face]) where ``label`` is the name
              of the item to add, ``color`` is the color of the label to add,
              and ``face`` is a string which defines the face (i.e. ``circle``,
              ``triangle``, ``box``, etc.).
              ``face`` could be also ``"none"`` (no face shown for the entry),
              or a :class:`pyvista.PolyData`.
            * A dict with the key ``label``. Optionally you can add the
              keys ``color`` and ``face``. The values of these keys can be
              strings. For the ``face`` key, it can be also a
              :class:`pyvista.PolyData`.

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

        face : str | pyvista.PolyData, optional
            Face shape of legend face. Defaults to a triangle for most meshes,
            with the exception of glyphs where the glyph is shown
            (e.g. arrows).

            You may set it to one of the following:

            * None: ``"none"``
            * Line: ``"-"`` or ``"line"``
            * Triangle: ``"^"`` or ``'triangle'``
            * Circle: ``"o"`` or ``'circle'``
            * Rectangle: ``"r"`` or ``'rectangle'``
            * Custom: :class:`pyvista.PolyData`

            Passing ``"none"`` removes the legend face.  A custom face can be
            created using :class:`pyvista.PolyData`.  This will be rendered
            from the XY plane.

        font_family : str, optional
            Font family.  Must be either ``'courier'``, ``'times'``,
            or ``'arial'``. Defaults to :attr:`pyvista.global_theme.font.family
            <pyvista.plotting.themes._Font.family>`.

        background_opacity : float, default: 1.0
            Set background opacity.

        Returns
        -------
        :vtk:`vtkLegendBoxActor`
            Actor for the legend.

        See Also
        --------
        :ref:`legend_example`

        Examples
        --------
        Create a legend by labeling the meshes when using ``add_mesh``

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> sphere = pv.Sphere(center=(0, 0, 1))
        >>> cube = pv.Cube()
        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(
        ...     sphere, color='grey', smooth_shading=True, label='Sphere'
        ... )
        >>> _ = plotter.add_mesh(cube, color='r', label='Cube')
        >>> _ = plotter.add_legend(bcolor='w', face=None)
        >>> plotter.show()

        Alternatively provide labels in the plotter as a list.

        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(sphere, color='grey', smooth_shading=True)
        >>> _ = plotter.add_mesh(cube, color='r')
        >>> legend_entries = []
        >>> legend_entries.append(['My Mesh', 'w'])
        >>> legend_entries.append(['My Other Mesh', 'k'])
        >>> _ = plotter.add_legend(legend_entries)
        >>> plotter.show()

        Or use a dictionary to define them.

        >>> labels = {'Grey Stuff': 'grey', 'Red Stuff': 'red'}
        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(sphere, color='grey', smooth_shading=True)
        >>> _ = plotter.add_mesh(cube, color='red')
        >>> _ = plotter.add_legend(labels, face='rectangle')
        >>> plotter.show()

        """
        if self.legend is not None:
            self.remove_legend()
        self._legend = _vtk.vtkLegendBoxActor()

        if labels is None:
            # use existing labels
            if not self._labels:
                msg = (
                    'No labels input.\n\n'
                    'Add labels to individual items when adding them to'
                    'the plotting object with the "label=" parameter.  '
                    'or enter them as the "labels" parameter.'
                )
                raise ValueError(msg)

            self._legend.SetNumberOfEntries(len(self._labels))
            for i, (vtk_object, text, color) in enumerate(self._labels.values()):
                vtk_object_input = make_legend_face(face) if face is not None else vtk_object
                self._legend.SetEntry(i, vtk_object_input, text, list(color.float_rgb))

        else:
            self._legend.SetNumberOfEntries(len(labels))

            if isinstance(labels, dict):
                face = 'triangle' if face is None else face
                labels = list(labels.items())

            for i, args in enumerate(labels):
                face_ = None
                if isinstance(args, (list, tuple)):
                    if len(args) == 2:
                        # format labels =  [[ text1, color1], [ text2, color2], etc]
                        text, color = args
                    else:
                        # format labels =  [[ text1, color1, face1], [ text2, color2, face2], etc]
                        # Pikcing only the first 3 elements
                        text, color, face_ = args[:3]
                elif isinstance(args, dict):
                    # it is using a dict
                    text = args.pop('label')
                    color = args.pop('color', None)
                    face_ = args.pop('face', None)

                    if args:
                        warnings.warn(
                            f'Some of the arguments given to legend are not used.\n{args}',
                        )
                elif isinstance(args, str):
                    # Only passing label
                    text = args
                    # taking the currents (if any)
                    try:
                        face_, _, color = list(self._labels.values())[i]
                    except (AttributeError, IndexError):
                        # There are no values
                        face_ = None
                        color = None

                else:
                    msg = f'The object passed to the legend ({type(args)}) is not valid.'
                    raise TypeError(msg)

                legend_face = make_legend_face(face_ or face)
                self._legend.SetEntry(i, legend_face, str(text), list(Color(color).float_rgb))

        if loc is not None:
            if loc not in ACTOR_LOC_MAP:
                allowed = '\n'.join([f'\t * "{item}"' for item in ACTOR_LOC_MAP])
                msg = f'Invalid loc "{loc}".  Expected one of the following:\n{allowed}'
                raise ValueError(msg)
            x, y, size = map_loc_to_pos(loc, size, border=0.05)
            self._legend.SetPosition(x, y)
            self._legend.SetPosition2(size[0], size[1])

        if bcolor is None:
            self._legend.SetUseBackground(False)
        else:
            self._legend.SetUseBackground(True)
            self._legend.SetBackgroundColor(Color(bcolor).float_rgb)

        self._legend.SetBorder(border)

        if font_family is None:
            font_family = self._theme.font.family

        font_family = parse_font_family(font_family)
        self._legend.GetEntryTextProperty().SetFontFamily(font_family)

        self._legend.SetBackgroundOpacity(background_opacity)

        self.add_actor(self._legend, reset_camera=False, name=name, pickable=False)
        return self._legend

    @_deprecate_positional_args
    def remove_legend(self, render=True) -> None:  # noqa: FBT002
        """Remove the legend actor.

        Parameters
        ----------
        render : bool, default: True
            Render upon actor removal.  Set this to ``False`` to stop
            the render window from rendering when a the legend is removed.

        Examples
        --------
        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, label='sphere')
        >>> _ = pl.add_legend()
        >>> pl.remove_legend()

        """
        if self.legend is not None:
            self.remove_actor(self.legend, reset_camera=False, render=render)
            self._legend = None

    @property
    def legend(self):  # numpydoc ignore=RT01
        """Legend actor."""
        return self._legend

    @_deprecate_positional_args(allowed=['pointa', 'pointb'])
    def add_ruler(  # noqa: PLR0917
        self,
        pointa,
        pointb,
        flip_range=False,  # noqa: FBT002
        number_labels=None,
        show_labels=True,  # noqa: FBT002
        font_size_factor=0.6,
        label_size_factor=1.0,
        label_format=None,
        title='Distance',
        number_minor_ticks=0,
        tick_length=5,
        minor_tick_length=3,
        show_ticks=True,  # noqa: FBT002
        tick_label_offset=2,
        label_color=None,
        tick_color=None,
        scale=1.0,
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

        number_labels : int, optional
            Number of labels to place on ruler.
            If not supplied, the number will be adjusted for "nice" values.

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

        scale : float, default: 1.0
            Scale factor for the ruler.

            .. versionadded:: 0.44.0

        Returns
        -------
        :vtk:`vtkActor`
            VTK actor of the ruler.

        Examples
        --------
        >>> import pyvista as pv
        >>> cone = pv.Cone(height=2.0, radius=0.5)
        >>> plotter = pv.Plotter()
        >>> _ = plotter.add_mesh(cone)

        Measure x direction of cone and place ruler slightly below.

        >>> _ = plotter.add_ruler(
        ...     pointa=[cone.bounds.x_min, cone.bounds.y_min - 0.1, 0.0],
        ...     pointb=[cone.bounds.x_max, cone.bounds.y_min - 0.1, 0.0],
        ...     title='X Distance',
        ... )

        Measure y direction of cone and place ruler slightly to left.
        The title and labels are placed to the right of the ruler when
        traveling from ``pointa`` to ``pointb``.

        >>> _ = plotter.add_ruler(
        ...     pointa=[cone.bounds.x_min - 0.1, cone.bounds.y_max, 0.0],
        ...     pointb=[cone.bounds.x_min - 0.1, cone.bounds.y_min, 0.0],
        ...     flip_range=True,
        ...     title='Y Distance',
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
        ruler.GetPositionCoordinate().SetReferenceCoordinate(None)  # type: ignore[arg-type]
        ruler.GetPositionCoordinate().SetValue(pointa[0], pointa[1], pointa[2])
        ruler.GetPosition2Coordinate().SetValue(pointb[0], pointb[1], pointb[2])

        distance = np.linalg.norm(np.asarray(pointa) - np.asarray(pointb))
        if flip_range:
            ruler.SetRange(distance * scale, 0)
        else:
            ruler.SetRange(0, distance * scale)

        ruler.SetTitle(title)
        ruler.SetFontFactor(font_size_factor)
        ruler.SetLabelFactor(label_size_factor)
        if number_labels is not None:
            ruler.AdjustLabelsOff()
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

    @_deprecate_positional_args
    def add_legend_scale(  # noqa: PLR0917
        self,
        corner_offset_factor=2.0,
        bottom_border_offset=30,
        top_border_offset=30,
        left_border_offset=30,
        right_border_offset=30,
        bottom_axis_visibility=True,  # noqa: FBT002
        top_axis_visibility=True,  # noqa: FBT002
        left_axis_visibility=True,  # noqa: FBT002
        right_axis_visibility=True,  # noqa: FBT002
        legend_visibility=True,  # noqa: FBT002
        xy_label_mode=False,  # noqa: FBT002
        render=True,  # noqa: FBT002
        color=None,
        font_size_factor=0.6,
        label_size_factor=1.0,
        label_format=None,
        number_minor_ticks=0,
        tick_length=5,
        minor_tick_length=3,
        show_ticks=True,  # noqa: FBT002
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
        :vtk:`vtkActor`
            The actor for the added :vtk:`vtkLegendScaleActor`.

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
            if pyvista.vtk_version_info >= (9, 4):
                legend_scale.SetLabelModeToCoordinates()
            else:
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
                int(font_size_factor * 20),
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


def _fixup_bounds(bounds) -> BoundsTuple:
    the_bounds = np.asarray(bounds)
    if np.any(the_bounds[::2] > the_bounds[1::2]):
        the_bounds[:] = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    return BoundsTuple(*the_bounds.tolist())
