"""Module containing useful plotting tools."""

from __future__ import annotations

from enum import Enum
import os
import platform
import subprocess
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import NoReturn
from typing import overload

import numpy as np
import pyvista_validation as _validation

import pyvista as pv
from pyvista import _vtk
from pyvista.core.errors import DeprecationError

from .colors import Color

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pyvista.core._typing_core import VectorLike

    from ._typing import ColorLike
    from ._typing import OpacityOptions


class FONTS(Enum):
    """Font families available to PyVista."""

    arial = _vtk.VTK_ARIAL
    courier = _vtk.VTK_COURIER
    times = _vtk.VTK_TIMES


# Track render window support and plotting
SUPPORTS_OPENGL: bool | None = None
SUPPORTS_PLOTTING: bool | None = None


def _validate_vector(vector: VectorLike[float], *, name: str) -> tuple[float, float, float]:
    """Return a three-component vector as a tuple of floats."""
    return _validation.validate_array3(vector, dtype_out=float, to_tuple=True, name=name)


def _validate_viewup(vector: VectorLike[float]) -> tuple[float, float, float]:
    """Return a view-up vector, which is normalized and so cannot be zero."""
    viewup = _validate_vector(vector, name='viewup')
    if np.allclose(viewup, 0.0):
        msg = 'Camera up vector cannot be zero.'
        raise ValueError(msg)
    return viewup


def _prepare_offscreen_macos_render_window(  # pragma: no cover
    render_window: _vtk.vtkRenderWindow | None,
) -> None:
    """Configure ``render_window`` for quiet, off-screen use on macOS.

    Two independent fixes for ``vtkCocoaRenderWindow`` behavior, both
    needed because VTK's off-screen path doesn't fully suppress its
    on-screen side effects:

    1. Merely instantiating ``NSApplication``, which VTK does internally
       in ``CreateAWindow()`` unconditionally, even for off-screen use,
       is enough for an unbundled Python process to get a Dock icon. VTK
       never reverses this, so we demote the activation policy via PyObjC.
       ``Accessory`` hides the Dock icon while still allowing the process
       to be activated later; ``Prohibited`` also forbids activation, which
       leaves any later on-screen window stuck behind other applications.
       An application already running on the ``Regular`` policy is left
       alone: the activation policy is process-global, so demoting it
       would strip the Dock icon and menu bar from a host GUI toolkit,
       such as a Qt application embedding a plotter.
    2. ``SetConnectContextToNSView(False)`` stops this particular render
       window from creating a real NSWindow.

    Safe to call unconditionally on any platform or render window type;
    each step no-ops where it doesn't apply (non-macOS, missing PyObjC,
    non-Cocoa render windows, a visible application).
    """

    def _suppress_dock_icon() -> None:
        """Demote the activation policy so an off-screen process gets no Dock icon."""
        if sys.platform != 'darwin':
            return
        try:  # type:ignore[unreachable]
            from AppKit import NSApp  # noqa: PLC0415
            from AppKit import NSApplication  # noqa: PLC0415
            from AppKit import NSApplicationActivationPolicyAccessory  # noqa: PLC0415
            from AppKit import NSApplicationActivationPolicyRegular  # noqa: PLC0415
        except ImportError:
            return

        # NSApp() reads the shared application without creating one, so a
        # process that has none still gets its Dock icon suppressed below
        app = NSApp()
        if app is not None and app.activationPolicy() == NSApplicationActivationPolicyRegular:
            return
        NSApplication.sharedApplication().setActivationPolicy_(
            NSApplicationActivationPolicyAccessory,
        )

    def _disable_cocoa_nsview_context() -> None:
        """Stop a Cocoa render window from creating a real window."""
        if hasattr(render_window, 'SetConnectContextToNSView'):
            render_window.SetConnectContextToNSView(False)  # type:ignore[union-attr]

    if render_window is None:
        return
    _suppress_dock_icon()
    _disable_cocoa_nsview_context()


def supports_open_gl() -> bool:
    """Return if the system supports OpenGL.

    This function checks if the system supports OpenGL by creating a VTK render
    window and querying its OpenGL support.

    Returns
    -------
    bool
        ``True`` if the system supports OpenGL, ``False`` otherwise.

    """
    global SUPPORTS_OPENGL  # noqa: PLW0603
    if SUPPORTS_OPENGL is None:
        ren_win = _vtk.vtkRenderWindow()
        ren_win.SetOffScreenRendering(True)
        _prepare_offscreen_macos_render_window(ren_win)
        SUPPORTS_OPENGL = bool(ren_win.SupportsOpenGL())
    return SUPPORTS_OPENGL


def _system_supports_plotting() -> bool:  # noqa: PLR0911
    """Check if the environment supports plotting on Windows, Linux, or Mac OS.

    Returns
    -------
    system_supports_plotting : bool
        ``True`` when system supports plotting.

    """
    if os.environ.get('ALLOW_PLOTTING', '').lower() == 'true':
        return True

    # Windows case
    if os.name == 'nt':
        # actually have to check here.  Somewhat expensive.
        return supports_open_gl()

    # mac case
    if platform.system() == 'Darwin':
        # check if finder available
        proc = subprocess.Popen(
            ['pgrep', '-qx', 'Finder'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf8',
        )
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            return False
        if proc.returncode == 0:
            return True

        # display variable set, likely available
        return 'DISPLAY' in os.environ

    # Linux case
    if os.environ.get('WAYLAND_DISPLAY'):  # pragma: no cover
        return True

    try:
        proc = subprocess.Popen(
            ['xset', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8'
        )
        proc.communicate(timeout=10)
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover
        # possible we have EGL support
        return supports_open_gl()
    else:  # pragma: no cover
        return proc.returncode == 0


def system_supports_plotting() -> bool:
    """Check if the environment supports plotting.

    Returns
    -------
    bool
        ``True`` when system supports plotting.

    """
    global SUPPORTS_PLOTTING  # noqa: PLW0603
    if SUPPORTS_PLOTTING is None:
        SUPPORTS_PLOTTING = _system_supports_plotting()

    # always use the cached response
    return SUPPORTS_PLOTTING


def _update_axes_label_color(
    axes_actor: _vtk.vtkAxesActor | _vtk.vtkAnnotatedCubeActor | _vtk.vtkPropAssembly,
    color: ColorLike | None = None,
) -> None:
    """Set the axes label color (internal helper)."""
    label_color = Color(color, default_color=pv.global_theme.font.color)
    if isinstance(axes_actor, _vtk.vtkPropAssembly):
        parts = axes_actor.GetParts()
        actors = [parts.GetItemAsObject(i) for i in range(parts.GetNumberOfItems())]
    else:
        actors = [axes_actor]
    for actor in actors:
        if isinstance(actor, _vtk.vtkAxesActor):
            prop_x = actor.GetXAxisCaptionActor2D().GetCaptionTextProperty()
            prop_y = actor.GetYAxisCaptionActor2D().GetCaptionTextProperty()
            prop_z = actor.GetZAxisCaptionActor2D().GetCaptionTextProperty()
            for prop in [prop_x, prop_y, prop_z]:
                prop.SetColor(label_color.float_rgb)
                prop.SetShadow(False)
        elif isinstance(actor, _vtk.vtkAnnotatedCubeActor):
            actor.GetTextEdgesProperty().SetColor(label_color.float_rgb)


def create_axes_marker(
    *,
    label_color: ColorLike | None = None,
    x_color: ColorLike | None = None,
    y_color: ColorLike | None = None,
    z_color: ColorLike | None = None,
    xlabel: str = 'X',
    ylabel: str = 'Y',
    zlabel: str = 'Z',
    labels_off: bool = False,
    line_width: float = 2,
    cone_radius: float = 0.4,
    shaft_length: float = 0.8,
    tip_length: float = 0.2,
    ambient: float = 0.5,
    label_size: VectorLike[float] = (0.25, 0.1),
) -> _vtk.vtkAxesActor:
    """Create an axis actor.

    Parameters
    ----------
    label_color : ColorLike, optional
        Color of the label text.

    x_color : ColorLike, optional
        Color of the x-axis shaft and tip.

    y_color : ColorLike, optional
        Color of the y-axis shaft and tip.

    z_color : ColorLike, optional
        Color of the z-axis shaft and tip.

    xlabel : str, default: "X"
        Text used for the x-axis.

    ylabel : str, default: "Y"
        Text used for the y-axis.

    zlabel : str, default: "Z"
        Text used for the z-axis.

    labels_off : bool, default: False
        Enable or disable the text labels for the axes.

    line_width : float, default: 2
        The width of the marker lines.

    cone_radius : float, default: 0.4
        The radius of the axes arrow tips.

    shaft_length : float, default: 0.8
        The length of the axes arrow shafts.

    tip_length : float, default: 0.2
        Length of the tip.

    ambient : float, default: 0.5
        The ambient of the axes arrows. Value should be between 0 and 1.

    label_size : sequence[float], default: (0.25, 0.1)
        The width and height of the axes label actors. Values should be between
        0 and 1. For example ``(0.2, 0.1)``.

    Returns
    -------
    :vtk:`vtkAxesActor`
        Axes actor.

    Examples
    --------
    Create the default axes marker.

    >>> import pyvista as pv
    >>> marker = pv.create_axes_marker()
    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(marker)
    >>> pl.show()

    Create an axes marker at the origin with custom colors and axis labels.

    >>> import pyvista as pv
    >>> marker = pv.create_axes_marker(
    ...     line_width=4,
    ...     ambient=0.0,
    ...     x_color='#378df0',
    ...     y_color='#ab2e5d',
    ...     z_color='#f7fb9a',
    ...     xlabel='X Axis',
    ...     ylabel='Y Axis',
    ...     zlabel='Z Axis',
    ...     label_size=(0.1, 0.1),
    ... )
    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(marker)
    >>> pl.show()

    """
    color_x = Color(x_color, default_color=pv.global_theme.axes.x_color)
    color_y = Color(y_color, default_color=pv.global_theme.axes.y_color)
    color_z = Color(z_color, default_color=pv.global_theme.axes.z_color)
    axes_actor = _vtk.vtkAxesActor()
    axes_actor.GetXAxisShaftProperty().SetColor(color_x.float_rgb)
    axes_actor.GetXAxisTipProperty().SetColor(color_x.float_rgb)
    axes_actor.GetYAxisShaftProperty().SetColor(color_y.float_rgb)
    axes_actor.GetYAxisTipProperty().SetColor(color_y.float_rgb)
    axes_actor.GetZAxisShaftProperty().SetColor(color_z.float_rgb)
    axes_actor.GetZAxisTipProperty().SetColor(color_z.float_rgb)
    # Set labels
    axes_actor.SetXAxisLabelText(xlabel)
    axes_actor.SetYAxisLabelText(ylabel)
    axes_actor.SetZAxisLabelText(zlabel)
    if labels_off:
        axes_actor.AxisLabelsOff()
    # Set Line width
    axes_actor.GetXAxisShaftProperty().SetLineWidth(line_width)
    axes_actor.GetYAxisShaftProperty().SetLineWidth(line_width)
    axes_actor.GetZAxisShaftProperty().SetLineWidth(line_width)

    axes_actor.SetConeRadius(cone_radius)
    axes_actor.SetNormalizedShaftLength([shaft_length] * 3)
    axes_actor.SetNormalizedTipLength([tip_length] * 3)
    axes_actor.GetXAxisShaftProperty().SetAmbient(ambient)
    axes_actor.GetYAxisShaftProperty().SetAmbient(ambient)
    axes_actor.GetZAxisShaftProperty().SetAmbient(ambient)
    axes_actor.GetXAxisTipProperty().SetAmbient(ambient)
    axes_actor.GetYAxisTipProperty().SetAmbient(ambient)
    axes_actor.GetZAxisTipProperty().SetAmbient(ambient)

    for label_actor in [
        axes_actor.GetXAxisCaptionActor2D(),
        axes_actor.GetYAxisCaptionActor2D(),
        axes_actor.GetZAxisCaptionActor2D(),
    ]:
        label_actor.SetWidth(float(label_size[0]))
        label_actor.SetHeight(float(label_size[1]))

    _update_axes_label_color(axes_actor, label_color)

    return axes_actor


# fmt: off
# ruff: disable[E501]
@overload
def create_axes_orientation_box(*, line_width: float = ..., text_scale: float = ..., edge_color: ColorLike = ..., x_color: ColorLike | None = ..., y_color: ColorLike | None = ..., z_color: ColorLike | None = ..., xlabel: str | None = ..., ylabel: str | None = ..., zlabel: str | None = ..., x_face_color: ColorLike = ..., y_face_color: ColorLike = ..., z_face_color: ColorLike = ..., color_box: Literal[True], label_color: ColorLike | None = ..., labels_off: bool = ..., opacity: float = ..., show_text_edges: bool = ...) -> _vtk.vtkPropAssembly: ...
@overload
def create_axes_orientation_box(*, line_width: float = ..., text_scale: float = ..., edge_color: ColorLike = ..., x_color: ColorLike | None = ..., y_color: ColorLike | None = ..., z_color: ColorLike | None = ..., xlabel: str | None = ..., ylabel: str | None = ..., zlabel: str | None = ..., x_face_color: ColorLike = ..., y_face_color: ColorLike = ..., z_face_color: ColorLike = ..., color_box: Literal[False] = False, label_color: ColorLike | None = ..., labels_off: bool = ..., opacity: float = ..., show_text_edges: bool = ...) -> _vtk.vtkAnnotatedCubeActor: ...
@overload
def create_axes_orientation_box(*, line_width: float = ..., text_scale: float = ..., edge_color: ColorLike = ..., x_color: ColorLike | None = ..., y_color: ColorLike | None = ..., z_color: ColorLike | None = ..., xlabel: str | None = ..., ylabel: str | None = ..., zlabel: str | None = ..., x_face_color: ColorLike = ..., y_face_color: ColorLike = ..., z_face_color: ColorLike = ..., color_box: bool = ..., label_color: ColorLike | None = ..., labels_off: bool = ..., opacity: float = ..., show_text_edges: bool = ...) -> _vtk.vtkAnnotatedCubeActor | _vtk.vtkPropAssembly: ...
# ruff: enable[E501]
# fmt: on
def create_axes_orientation_box(
    *,
    line_width: float = 1,
    text_scale: float = 0.366667,
    edge_color: ColorLike = 'black',
    x_color: ColorLike | None = None,
    y_color: ColorLike | None = None,
    z_color: ColorLike | None = None,
    xlabel: str | None = 'X',
    ylabel: str | None = 'Y',
    zlabel: str | None = 'Z',
    x_face_color: ColorLike = 'red',
    y_face_color: ColorLike = 'green',
    z_face_color: ColorLike = 'blue',
    color_box: bool = False,
    label_color: ColorLike | None = None,
    labels_off: bool = False,
    opacity: float = 0.5,
    show_text_edges: bool = False,
) -> _vtk.vtkAnnotatedCubeActor | _vtk.vtkPropAssembly:
    """Create a Box axes orientation widget with labels.

    Parameters
    ----------
    line_width : float, default: 1
        The width of the text edge lines.

    text_scale : float, default: 0.366667
        Size of the text relative to the faces.

    edge_color : ColorLike, default: 'black'
        Color of the cube edges.

    x_color : ColorLike, optional
        Color of the x-axis text.

    y_color : ColorLike, optional
        Color of the y-axis text.

    z_color : ColorLike, optional
        Color of the z-axis text.

    xlabel : str, default: "X"
        Text used for the x-axis.

    ylabel : str, default: "Y"
        Text used for the y-axis.

    zlabel : str, default: "Z"
        Text used for the z-axis.

    x_face_color : ColorLike, default: 'red'
        Color of the two faces perpendicular to the x-axis. Only used when
        ``color_box`` is ``True``.

    y_face_color : ColorLike, default: 'green'
        Color of the two faces perpendicular to the y-axis. Only used when
        ``color_box`` is ``True``.

    z_face_color : ColorLike, default: 'blue'
        Color of the two faces perpendicular to the z-axis. Only used when
        ``color_box`` is ``True``.

    color_box : bool, default: False
        Enable or disable the face colors.  Otherwise, box is white.

    label_color : ColorLike, optional
        Color of the text edges.

    labels_off : bool, default: False
        Enable or disable the text labels for the axes.

    opacity : float, default: 0.5
        Opacity in the range of ``[0, 1]`` of the orientation box.

    show_text_edges : bool, default: False
        Enable or disable drawing the vector text edges.

    Returns
    -------
    :vtk:`vtkAnnotatedCubeActor` | :vtk:`vtkPropAssembly`
        Annotated cube actor, or a prop assembly of that actor and a colored
        cube when ``color_box`` is ``True``.

    Examples
    --------
    .. pyvista-plot::
        :force_static:

        Create and plot an orientation box

        >>> import pyvista as pv
        >>> actor = pv.create_axes_orientation_box(
        ...     line_width=1,
        ...     text_scale=0.53,
        ...     edge_color='black',
        ...     x_color='k',
        ...     y_color=None,
        ...     z_color=None,
        ...     xlabel='X',
        ...     ylabel='Y',
        ...     zlabel='Z',
        ...     color_box=False,
        ...     labels_off=False,
        ...     opacity=1.0,
        ... )
        >>> pl = pv.Plotter()
        >>> _ = pl.add_actor(actor)
        >>> pl.show()

    """
    color_x = Color(x_color, default_color=pv.global_theme.axes.x_color)
    color_y = Color(y_color, default_color=pv.global_theme.axes.y_color)
    color_z = Color(z_color, default_color=pv.global_theme.axes.z_color)
    color_edge = Color(edge_color, default_color=pv.global_theme.edge_color)
    face_color_x = Color(x_face_color)
    face_color_y = Color(y_face_color)
    face_color_z = Color(z_face_color)
    axes_actor = _vtk.vtkAnnotatedCubeActor()
    axes_actor.SetFaceTextScale(text_scale)
    if xlabel is not None:
        axes_actor.SetXPlusFaceText(f'+{xlabel}')
        axes_actor.SetXMinusFaceText(f'-{xlabel}')
    if ylabel is not None:
        axes_actor.SetYPlusFaceText(f'+{ylabel}')
        axes_actor.SetYMinusFaceText(f'-{ylabel}')
    if zlabel is not None:
        axes_actor.SetZPlusFaceText(f'+{zlabel}')
        axes_actor.SetZMinusFaceText(f'-{zlabel}')
    axes_actor.SetFaceTextVisibility(not labels_off)
    axes_actor.SetTextEdgesVisibility(show_text_edges)
    # https://github.com/pyvista/pyvista/pull/5382
    # axes_actor.GetTextEdgesProperty().SetColor(edge_color.float_rgb)
    axes_actor.GetTextEdgesProperty().SetLineWidth(line_width)
    axes_actor.GetXPlusFaceProperty().SetColor(color_x.float_rgb)
    axes_actor.GetXMinusFaceProperty().SetColor(color_x.float_rgb)
    axes_actor.GetYPlusFaceProperty().SetColor(color_y.float_rgb)
    axes_actor.GetYMinusFaceProperty().SetColor(color_y.float_rgb)
    axes_actor.GetZPlusFaceProperty().SetColor(color_z.float_rgb)
    axes_actor.GetZMinusFaceProperty().SetColor(color_z.float_rgb)

    axes_actor.GetCubeProperty().SetOpacity(opacity)
    axes_actor.GetCubeProperty().SetEdgeColor(color_edge.float_rgb)
    axes_actor.GetCubeProperty().SetEdgeVisibility(True)
    axes_actor.GetCubeProperty().BackfaceCullingOn()
    if opacity < 1.0:
        # Hide the text edges
        axes_actor.GetTextEdgesProperty().SetOpacity(0)

    actor: _vtk.vtkAnnotatedCubeActor | _vtk.vtkPropAssembly
    if color_box:
        # Hide the cube so we can color each face
        axes_actor.GetCubeProperty().SetOpacity(0)
        axes_actor.GetCubeProperty().SetEdgeVisibility(False)

        cube = pv.Cube()
        cube.clear_data()  # remove normals
        face_colors = np.array(
            [
                face_color_x.int_rgb,
                face_color_x.int_rgb,
                face_color_y.int_rgb,
                face_color_y.int_rgb,
                face_color_z.int_rgb,
                face_color_z.int_rgb,
            ],
            np.uint8,
        )
        cube.cell_data['face_colors'] = face_colors

        cube_mapper = _vtk.vtkPolyDataMapper()
        cube_mapper.SetInputData(cube)
        cube_mapper.SetColorModeToDirectScalars()
        cube_mapper.Update()

        cube_actor = pv.Actor(mapper=cube_mapper)
        cube_actor.prop.culling = 'back'
        cube_actor.prop.opacity = opacity

        prop_assembly = _vtk.vtkPropAssembly()
        prop_assembly.AddPart(axes_actor)
        prop_assembly.AddPart(cube_actor)
        actor = prop_assembly
    else:
        actor = axes_actor

    _update_axes_label_color(actor, label_color)

    return actor


def create_north_arrow() -> pv.PolyData:
    """Create a north arrow mesh.

    .. versionadded:: 0.44.0

    Returns
    -------
    pyvista.PolyData
        North arrow mesh.

    """
    points = np.array(
        [
            [0.0, 5.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 5.0, 1.0],
            [-2.0, 0.0, 1.0],
            [0.0, 1.5, 1.0],
            [2.0, 0.0, 1.0],
        ],
    )
    faces = np.array(
        [
            4,
            3,
            7,
            4,
            0,
            4,
            2,
            6,
            7,
            3,
            4,
            1,
            5,
            6,
            2,
            4,
            0,
            4,
            5,
            1,
            4,
            0,
            1,
            2,
            3,
            4,
            4,
            7,
            6,
            5,
        ],
    )
    return pv.PolyData(points, faces)


def normalize(
    x: NDArray[Any],
    minimum: float | None = None,
    maximum: float | None = None,
) -> NDArray[float]:
    """Normalize the given values to the range ``[0, 1]``.

    Parameters
    ----------
    x : numpy.ndarray
        The array of values to normalize.
    minimum : float, optional
        The value which is normalized to ``0``. If not specified, the minimum
        value in ``x`` will be used.
    maximum : float, optional
        The value which is normalized to ``1``. If not specified, the maximum
        value in ``x`` will be used.

    Returns
    -------
    numpy.ndarray
        The normalized array of values.

    """
    low = np.nanmin(x) if minimum is None else minimum
    high = np.nanmax(x) if maximum is None else maximum
    return (x - low) / (high - low)


def _opacity_transfer_functions(n_colors: int) -> dict[str, NDArray[np.uint8]]:
    """Return every named opacity mapping, each ``n_colors`` values long."""

    def sigmoid(x: NDArray[float]) -> NDArray[np.uint8]:  # numpydoc ignore=PR01,RT01
        """Map ``x`` onto the [0, 255] opacity range with a logistic curve."""
        return np.array(1 / (1 + np.exp(-x)) * 255, dtype=np.uint8)

    transfer_func: dict[str, NDArray[np.uint8]] = {
        'linear': np.linspace(0, 255, n_colors, dtype=np.uint8),
        'geom': np.geomspace(1e-6, 255, n_colors, dtype=np.uint8),
        'geom_r': np.geomspace(255, 1e-6, n_colors, dtype=np.uint8),
        'sigmoid': sigmoid(np.linspace(-10.0, 10.0, n_colors)),
        'sigmoid_1': sigmoid(np.linspace(-1.0, 1.0, n_colors)),
        'sigmoid_2': sigmoid(np.linspace(-2.0, 2.0, n_colors)),
        'sigmoid_3': sigmoid(np.linspace(-3.0, 3.0, n_colors)),
        'sigmoid_4': sigmoid(np.linspace(-4.0, 4.0, n_colors)),
        'sigmoid_5': sigmoid(np.linspace(-5.0, 5.0, n_colors)),
        'sigmoid_6': sigmoid(np.linspace(-6.0, 6.0, n_colors)),
        'sigmoid_7': sigmoid(np.linspace(-7.0, 7.0, n_colors)),
        'sigmoid_8': sigmoid(np.linspace(-8.0, 8.0, n_colors)),
        'sigmoid_9': sigmoid(np.linspace(-9.0, 9.0, n_colors)),
        'sigmoid_10': sigmoid(np.linspace(-10.0, 10.0, n_colors)),
        'sigmoid_15': sigmoid(np.linspace(-15.0, 15.0, n_colors)),
        'sigmoid_20': sigmoid(np.linspace(-20.0, 20.0, n_colors)),
        'foreground': np.hstack((0, [255] * (n_colors - 1))).astype(np.uint8),
    }
    reversible = [
        name for name in transfer_func if name != 'foreground' and not name.endswith('_r')
    ]
    for name in reversible:
        transfer_func.setdefault(f'{name}_r', transfer_func[name][::-1])
    return transfer_func


def opacity_transfer_function(
    mapping: OpacityOptions | str | VectorLike[float],
    n_colors: int,
    *,
    interpolate: bool = True,
    kind: str = 'linear',
) -> NDArray[np.uint8]:
    """Get the opacity transfer function for a mapping.

    These values will map on to a scalar bar range and thus the number of
    colors (``n_colors``) must correspond to the number of colors in the color
    mapping that these opacities are associated to.

    If interpolating, ``scipy.interpolate.interp1d`` is used if available,
    otherwise ``np.interp`` is used. The ``kind`` argument controls the kind of
    interpolation for ``interp1d``.

    This returns the opacity range from 0 to 255, where 0 is totally
    transparent and 255 is totally opaque.

    The equation to create the sigmoid mapping is: ``1 / (1 + exp(-x))`` where
    ``x`` is the range from ``-a`` to ``+a`` and ``a`` is the value given in
    the ``mapping`` string. Default is ``a=10`` for 'sigmoid' mapping.

    Parameters
    ----------
    mapping : sequence[float] | str
        The opacity mapping to use. Can be a ``str`` name of a predefined
        mapping including ``'linear'``, ``'geom'``, ``'sigmoid'``,
        ``'sigmoid_1'`` through ``'sigmoid_10'``, ``'sigmoid_15'``,
        ``'sigmoid_20'``, and ``'foreground'``. Append an ``'_r'`` to any of
        those names (except ``'foreground'``) to reverse that mapping.
        The mapping can also be a custom user-defined array/list of values
        that will be interpolated across the ``n_color`` range.

    n_colors : int
        The number of colors that the opacities must be mapped to.

    interpolate : bool, default: True
        Flag on whether or not to interpolate the opacity mapping for all
        colors.

    kind : str, default: 'linear'
        The interpolation kind if ``interpolate`` is ``True`` and ``scipy``
        is available. If ``scipy`` is not available, linear interpolation
        is always used. Options are:

        - ``'linear'``
        - ``'nearest'``
        - ``'zero'``
        - ``'slinear'``
        - ``'quadratic'``
        - ``'cubic'``
        - ``'previous'``
        - ``'next'``

        .. versionchanged:: 0.46

            Linear interpolation is now always used by default. Previously,
            quadratic interpolation was used if ``scipy`` was installed.

    Returns
    -------
    numpy.ndarray
        Array of ``numpy.uint8`` values ``n_colors`` long containing the
        [0-255] opacity mapping values.

    Examples
    --------
    >>> import pyvista as pv
    >>> # Fetch the `sigmoid` mapping between 0 and 255
    >>> tf = pv.opacity_transfer_function('sigmoid', 256)
    >>> # Fetch the `geom_r` mapping between 0 and 1
    >>> tf = pv.opacity_transfer_function('geom_r', 256).astype(float) / 255.0
    >>> # Interpolate a user defined opacity mapping
    >>> opacity = [0, 0.2, 0.9, 0.6, 0.3]
    >>> tf = pv.opacity_transfer_function(opacity, 256)

    """
    transfer_func = _opacity_transfer_functions(n_colors)
    if isinstance(mapping, str):
        try:
            return transfer_func[mapping]
        except KeyError:
            msg = (
                f'Opacity transfer function ({mapping}) unknown. '
                f'Valid options: {list(transfer_func.keys())}'
            )
            raise ValueError(msg) from None
    elif isinstance(mapping, (np.ndarray, list, tuple)):
        values = np.array(mapping)
        if values.size == n_colors:
            # User could pass transfer function ready for lookup table
            pass
        elif values.size < n_colors:
            # User pass custom transfer function to be linearly interpolated
            if np.max(values) > 1.0 or np.min(values) < 0.0:
                values = normalize(values)
            # Interpolate transfer function to match lookup table
            xo = np.linspace(0, n_colors, len(values), dtype=np.int_)
            xx = np.linspace(0, n_colors, n_colors, dtype=np.int_)
            try:
                if not interpolate:
                    msg = 'No interpolation.'
                    raise ValueError(msg)
                from scipy.interpolate import interp1d  # noqa: PLC0415

                f = interp1d(xo, values, kind=kind)
                vals = f(xx)
                vals[vals < 0] = 0.0
                vals[vals > 1.0] = 1.0
                values = (vals * 255.0).astype(np.uint8)

            except (ImportError, ValueError):
                # Otherwise use simple linear interp
                values = (np.interp(xx, xo, values) * 255).astype(np.uint8)
        else:
            msg = (
                f'Transfer function cannot have more values than `n_colors`. '
                f'This has {values.size} elements'
            )
            raise RuntimeError(msg)
        return values
    msg = f'Transfer function type ({type(mapping)}) not understood'
    raise TypeError(msg)


def parse_font_family(font_family: str) -> int:
    """Check and validate the given font family name.

    Parameters
    ----------
    font_family : str
        Font family name to validate. Must be one of the font names defined in
        the ``FONTS`` enum class.

    Returns
    -------
    int
        Corresponding integer value of the valid font family name in the
        ``FONTS`` enum class.

    Raises
    ------
    ValueError
        If the ``font_family`` is not one of the defined font names in the ``FONTS``
        enum class.

    """
    font_family = font_family.lower()
    fonts = [font.name for font in FONTS]
    if font_family not in fonts:
        msg = f'Font must one of the following:\n{", ".join(fonts)}'
        raise ValueError(msg)
    return FONTS[font_family].value


def check_math_text_support() -> NoReturn:
    """Raise a DeprecationError as this has been moved."""
    # Deprecated on v0.47.0, estimated removal on v0.50.0
    msg = (
        '`pyvista.plotting.check_math_text_support` is deprecated. '
        'Use `pyvista.check_math_text_support` instead.'
    )
    raise DeprecationError(msg)


def check_matplotlib_vtk_compatibility() -> NoReturn:
    """Raise a DeprecationError as this has been moved."""
    # Deprecated on v0.47.0, estimated removal on v0.50.0
    msg = (
        '`pyvista.plotting.check_matplotlib_vtk_compatibility` is deprecated. '
        'Use `pyvista.check_matplotlib_vtk_compatibility` instead.'
    )
    raise DeprecationError(msg)
