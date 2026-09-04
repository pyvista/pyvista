"""Module containing the wrapping of CubeAxesActor."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import cast

import numpy as np
import pyvista_validation as _validation

import pyvista as pv
from pyvista import _vtk
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core._typing_core import BoundsTuple
from pyvista.core._vtk_utilities import DisableVtkSnakeCase
from pyvista.core.utilities.arrays import convert_string_array
from pyvista.core.utilities.misc import _BoundsSizeMixin
from pyvista.core.utilities.misc import _NameMixin
from pyvista.core.utilities.misc import _NoNewAttrMixin
from pyvista.plotting.colors import Color
from pyvista.plotting.tools import parse_font_family

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyvista.core._typing_core import VectorLike
    from pyvista.plotting._typing import ColorLike


def _pad_bounds(bounds: VectorLike[float], *, padding: float) -> np.ndarray:
    """Cushion bounds by a percentage of their size along each axial direction."""
    if not (isinstance(padding, (int, float)) and 0.0 <= padding < 1.0):  # type: ignore[redundant-expr]
        msg = f'padding ({padding}) not understood. Must be float between 0 and 1'
        raise ValueError(msg)
    padded = np.asanyarray(bounds, dtype=float).copy()
    if not np.any(np.abs(padded) == np.inf):
        cushion = np.abs(padded[1::2] - padded[::2]) * padding
        padded[::2] -= cushion
        padded[1::2] += cushion
    return padded


_FLT_EPSILON = 1.1920928955078125e-07


def _fit_n_labels(vmin: float, vmax: float, n: int) -> int:
    """Reduce a label count to the most :vtk:`vtkCubeAxesActor` spaces evenly over a range.

    :vtk:`vtkCubeAxesActor` replaces its own major tick spacing with
    ``(vmax - vmin) / (n - 1)`` only when that spacing is finer than the one it
    picks itself. Ask for more labels than that and the ticks stay where VTK put
    them while the labels keep the values of an evenly spaced axis, so every
    label after the first is drawn at the wrong coordinate.
    """
    span = abs(vmax - vmin)
    if n < 2 or span == 0.0 or not math.isfinite(span):
        return n
    # Mirrors the tick spacing of ``vtkCubeAxesActor::AdjustTicksComputeRange``
    power = math.log10(span)
    if power != 0.0:
        power = math.copysign(abs(power) + 10.0e-10, power)
    if power < 0.0:
        power -= 1.0
    decade = 10.0 ** float(int(power))
    ticks = int(span / decade)
    ticks = ticks + 1 if ticks else 0
    divisor = 5.0 if ticks <= 2 else 2.0 if ticks < 5 else 1.0
    major = decade / divisor
    intervals = span / major
    # Mirrors the label count of ``vtkCubeAxesActor::BuildLabels``
    labelled = math.floor(intervals + 2 * _FLT_EPSILON) + 1
    # VTK only overrides ``major`` while there are fewer labels than its own ticks. Past that
    # it keeps its spacing, which suits evenly spaced labels only if its last tick is ``vmax``
    lands_on_vmax = abs(intervals - labelled + 1) <= 1e-9 * (intervals or 1.0)
    maximum = labelled if lands_on_vmax else int(intervals)
    return maximum if n >= maximum else min(n, int(intervals))


@_deprecate_positional_args
def make_axis_labels(vmin, vmax, n, fmt):  # noqa: PLR0917
    """Create axis labels as a :vtk:`vtkStringArray`.

    Parameters
    ----------
    vmin : float
        The minimum value for the axis labels.
    vmax : float
        The maximum value for the axis labels.
    n : int
        The number of labels to create. Fewer labels are created if
        :vtk:`vtkCubeAxesActor` cannot space that many evenly between ``vmin``
        and ``vmax``.
    fmt : str
        A format string for the labels. If the string starts with '%', the label will be formatted
        using the old-style string formatting method.
        Otherwise, the label will be formatted using the new-style string formatting method.

    Returns
    -------
    :vtk:`vtkStringArray`
        The created labels as a :vtk:`vtkStringArray` object.

    """
    labels = _vtk.vtkStringArray()
    for v in np.linspace(vmin, vmax, _fit_n_labels(vmin, vmax, n)):
        label = (fmt % v if fmt.startswith('%') else fmt.format(v)) if fmt else f'{v}'
        labels.InsertNextValue(label)
    return labels


class CubeAxesActor(
    _NoNewAttrMixin, _NameMixin, _BoundsSizeMixin, DisableVtkSnakeCase, _vtk.vtkCubeAxesActor
):
    """Wrap :vtk:`vtkCubeAxesActor`.

    This class is created to wrap :vtk:`vtkCubeAxesActor`, which is used to draw axes
    and labels for the input data bounds. This wrapping aims to provide a
    user-friendly interface to use :vtk:`vtkCubeAxesActor`.

    .. versionchanged:: 0.49

        The bounds, colors, fonts, grid lines, and axis placement are now set by the
        constructor, using the same defaults as :meth:`~pyvista.Plotter.show_bounds`.
        Previously these were only applied to actors created by that method, and an
        actor created directly used VTK's defaults instead.

    Parameters
    ----------
    camera : pyvista.Camera
        Camera to link to the axes actor.

    minor_ticks : bool, default: False
        If ``True``, also plot minor ticks on all axes.

    tick_location : str, optional
        Set how the ticks are drawn on the axes grid. Options include:
        ``'inside', 'outside', 'both'``.

    x_title : str, default: "X Axis"
        Title of the x-axis.

    y_title : str, default: "Y Axis"
        Title of the y-axis.

    z_title : str, default: "Z Axis"
        Title of the z-axis.

    x_axis_visibility : bool, default: True
        Visibility of the x-axis.

    y_axis_visibility : bool, default: True
        Visibility of the y-axis.

    z_axis_visibility : bool, default: True
        Visibility of the z-axis.

    x_label_format : str, optional
        A format string defining how tick labels are generated from tick
        positions for the x-axis. Defaults to the theme format if set,
        otherwise ``'{0:.1f}'``.

    y_label_format : str, optional
        A format string defining how tick labels are generated from tick
        positions for the y-axis. Defaults to the theme format if set,
        otherwise ``'{0:.1f}'``.

    z_label_format : str, optional
        A format string defining how tick labels are generated from tick
        positions for the z-axis. Defaults to the theme format if set,
        otherwise ``'{0:.1f}'``.

    x_label_visibility : bool, default: True
        The visibility of the x-axis labels.

    y_label_visibility : bool, default: True
        The visibility of the y-axis labels.

    z_label_visibility : bool, default: True
        The visibility of the z-axis labels.

    n_xlabels : int, default: 5
        Number of labels along the x-axis. Fewer labels are shown when VTK
        cannot space that many evenly between the axis bounds.

    n_ylabels : int, default: 5
        Number of labels along the y-axis. Fewer labels are shown when VTK
        cannot space that many evenly between the axis bounds.

    n_zlabels : int, default: 5
        Number of labels along the z-axis. Fewer labels are shown when VTK
        cannot space that many evenly between the axis bounds.

    color : ColorLike, optional
        Color of all labels, axis titles, axis lines, and grid lines. Defaults to
        :attr:`pyvista.global_theme.font.color
        <pyvista.plotting.themes._Font.color>`.

        .. versionadded:: 0.49

    grid : bool | str, optional
        Add grid lines to the backface (``True``, ``'back'``, or ``'backface'``) or to
        the frontface (``'front'``, ``'frontface'``) of the axes actor.

        .. versionadded:: 0.49

    location : str, default: "closest"
        Set how the axes are drawn: either static (``'all'``), closest triad
        (``'front'``, ``'closest'``, ``'default'``), furthest triad (``'back'``,
        ``'furthest'``), static closest to the origin (``'origin'``), or outer edges
        (``'outer'``) in relation to the camera position.

        .. versionadded:: 0.49

    font_size : float, optional
        Size of the label font. Defaults to :attr:`pyvista.global_theme.font.size
        <pyvista.plotting.themes._Font.size>`.

        .. versionadded:: 0.49

    font_family : str, optional
        Font family. Must be either ``'courier'``, ``'times'``, or ``'arial'``.
        Defaults to :attr:`pyvista.global_theme.font.family
        <pyvista.plotting.themes._Font.family>`.

        .. versionadded:: 0.49

    bold : bool, default: True
        Bold the axis labels and titles.

        .. versionadded:: 0.49

    use_3d_text : bool, optional
        Use :vtk:`vtkTextActor3D` for titles and labels. Defaults to ``False`` for
        VTK 9.6 and later, and ``True`` for older versions of VTK.

        .. warning::
            Setting ``use_3d_text=True`` is not recommended with VTK 9.6.0 or later since
            the 3D labels may not render at all in some cases. This is a known VTK bug:
            https://gitlab.kitware.com/vtk/vtk/-/issues/19729.

        .. versionadded:: 0.49

    use_2d_mode : bool, default: False
        Use the 2D render mode. This can be enabled for smoother plotting.

        .. versionadded:: 0.49

    bounds : sequence[float], optional
        Bounds of the axes in the form ``(x_min, x_max, y_min, y_max, z_min, z_max)``.
        Defaults to the unit cube.

        .. versionadded:: 0.49

    axes_ranges : sequence[float], optional
        Values shown on the axes in the form
        ``(x_min, x_max, y_min, y_max, z_min, z_max)``. These override the values
        derived from ``bounds``.

        .. versionadded:: 0.49

    padding : float, default: 0.0
        Percent padding applied to ``bounds`` along each axial direction to cushion
        the datasets in the scene from the axes annotations.

        .. versionadded:: 0.49

    See Also
    --------
    :meth:`~pyvista.Plotter.show_bounds`
    :meth:`~pyvista.Plotter.show_grid`

    Examples
    --------
    Create a 3D plotter and add a CubeAxesActor to it.

    >>> import pyvista as pv
    >>> mesh = pv.Cube()
    >>> pl = pv.Plotter()
    >>> actor = pl.add_mesh(mesh)
    >>> cube_axes_actor = pv.CubeAxesActor(pl.camera)
    >>> cube_axes_actor.bounds = mesh.bounds
    >>> actor, property = pl.add_actor(cube_axes_actor)
    >>> pl.show()

    """

    @_deprecate_positional_args(allowed=['camera'])
    def __init__(  # noqa: PLR0917
        self,
        camera,
        minor_ticks: bool = False,  # noqa: FBT001, FBT002
        tick_location=None,
        x_title='X Axis',
        y_title='Y Axis',
        z_title='Z Axis',
        x_axis_visibility: bool = True,  # noqa: FBT001, FBT002
        y_axis_visibility: bool = True,  # noqa: FBT001, FBT002
        z_axis_visibility: bool = True,  # noqa: FBT001, FBT002
        x_label_format=None,
        y_label_format=None,
        z_label_format=None,
        x_label_visibility: bool = True,  # noqa: FBT001, FBT002
        y_label_visibility: bool = True,  # noqa: FBT001, FBT002
        z_label_visibility: bool = True,  # noqa: FBT001, FBT002
        n_xlabels=5,
        n_ylabels=5,
        n_zlabels=5,
        color: ColorLike | None = None,
        grid: bool | str | None = None,  # noqa: FBT001
        location: str | None = 'closest',
        font_size: float | None = None,
        font_family: str | None = None,
        bold: bool = True,  # noqa: FBT001, FBT002
        use_3d_text: bool | None = None,  # noqa: FBT001
        use_2d_mode: bool = False,  # noqa: FBT001, FBT002
        bounds: VectorLike[float] | None = None,
        axes_ranges: VectorLike[float] | None = None,
        padding: float = 0.0,
    ):
        """Initialize CubeAxesActor."""
        super().__init__()
        self.camera = camera

        # empty string used for clearing axis labels
        self._empty_str = _vtk.vtkStringArray()
        self._empty_str.InsertNextValue('')

        # stop labels from being generated several times during init
        self.x_axis_visibility = False
        self.y_axis_visibility = False
        self.z_axis_visibility = False

        if not minor_ticks:
            self.x_axis_minor_tick_visibility = minor_ticks
            self.y_axis_minor_tick_visibility = minor_ticks
            self.z_axis_minor_tick_visibility = minor_ticks

        if tick_location:
            self.tick_location = tick_location
        self.x_title = x_title
        self.y_title = y_title
        self.z_title = z_title

        self._x_label_visibility = x_label_visibility
        self._y_label_visibility = y_label_visibility
        self._z_label_visibility = z_label_visibility

        default_fmt = '%.1f' if pv.vtk_version_info < (9, 6, 0) else '{0:.1f}'
        if x_label_format is None:
            x_label_format = pv.global_theme.font.fmt
            if x_label_format is None:
                x_label_format = default_fmt
        if y_label_format is None:
            y_label_format = pv.global_theme.font.fmt
            if y_label_format is None:
                y_label_format = default_fmt
        if z_label_format is None:
            z_label_format = pv.global_theme.font.fmt
            if z_label_format is None:
                z_label_format = default_fmt

        self.x_label_format = x_label_format
        self.y_label_format = y_label_format
        self.z_label_format = z_label_format

        self.n_xlabels = n_xlabels
        self.n_ylabels = n_ylabels
        self.n_zlabels = n_zlabels

        self.x_axis_visibility = x_axis_visibility
        self.y_axis_visibility = y_axis_visibility
        self.z_axis_visibility = z_axis_visibility

        # 2D mode and the label arrays rebuild the text actors, so text properties come last
        self.use_2d_mode = use_2d_mode

        color_ = Color(color, default_color=pv.global_theme.font.color)
        self._configure_grid_lines(
            grid=grid,
            color=color_,
            visibility=(x_axis_visibility, y_axis_visibility, z_axis_visibility),
        )
        self._configure_fly_mode(location=location)

        if bounds is not None:
            self.bounds = _pad_bounds(bounds, padding=padding)
        if axes_ranges is not None:
            ranges = _validation.validate_array(
                axes_ranges, must_have_shape=(6,), name='axes_ranges'
            )
            self.x_axis_range = ranges[0], ranges[1]
            self.y_axis_range = ranges[2], ranges[3]
            self.z_axis_range = ranges[4], ranges[5]

        self.GetXAxesLinesProperty().SetColor(color_.float_rgb)
        self.GetYAxesLinesProperty().SetColor(color_.float_rgb)
        self.GetZAxesLinesProperty().SetColor(color_.float_rgb)

        self._configure_text(
            color=color_,
            font_size=font_size,
            font_family=font_family,
            bold=bold,
            use_3d_text=use_3d_text,
        )

    def _configure_grid_lines(
        self, *, grid: bool | str | None, color: Color, visibility: tuple[bool, bool, bool]
    ) -> None:
        """Set the grid line location, visibility, and color."""
        if not grid:
            return
        grid = 'back' if grid is True else grid
        if not isinstance(grid, str):
            msg = f'`grid` must be a str, not {type(grid)}'  # type: ignore[unreachable]
            raise TypeError(msg)
        grid = grid.lower()
        if grid in ('front', 'frontface'):
            self.SetGridLineLocation(self.VTK_GRID_LINES_CLOSEST)
        elif grid in ('both', 'all'):
            self.SetGridLineLocation(self.VTK_GRID_LINES_ALL)
        elif grid in ('back', 'backface'):
            self.SetGridLineLocation(self.VTK_GRID_LINES_FURTHEST)
        else:
            msg = f'`grid` must be either "front", "back, or, "all", not {grid}'
            raise ValueError(msg)

        self.SetDrawXGridlines(visibility[0])
        self.SetDrawYGridlines(visibility[1])
        self.SetDrawZGridlines(visibility[2])
        self.GetXAxesGridlinesProperty().SetColor(color.float_rgb)
        self.GetYAxesGridlinesProperty().SetColor(color.float_rgb)
        self.GetZAxesGridlinesProperty().SetColor(color.float_rgb)

    def _configure_text(
        self,
        *,
        color: Color,
        font_size: float | None,
        font_family: str | None,
        bold: bool,
        use_3d_text: bool | None,
    ) -> None:
        """Set the color, font, and rendering mode of the titles and labels."""
        vtk_less_than_96 = pv.vtk_version_info < (9, 6, 0)
        if use_3d_text is None:
            # 3D text does not render at all with VTK 9.6
            # https://gitlab.kitware.com/vtk/vtk/-/issues/19729
            use_3d_text = vtk_less_than_96
        if font_size is None:
            font_size = pv.global_theme.font.size
        if font_family is None:
            font_family = pv.global_theme.font.family

        self.SetUseTextActor3D(use_3d_text)

        # For 3D text, use `SetFontSize` to a relatively high value and use `SetScreenSize` to
        # shrink it back down. This creates a higher-resolution font and makes it appear sharper.
        # In VTK 9.6+, the 3D font size is also tied to the value set by SetFontSize, so we need
        # an additional scaling factor.
        default_screen_size = 10.0
        default_font_size = 12
        scaled_font_size = 50

        for axis in range(3):
            for prop in (self.GetTitleTextProperty(axis), self.GetLabelTextProperty(axis)):
                prop.SetColor(color.float_rgb)
                prop.SetFontFamily(parse_font_family(font_family))
                prop.SetBold(bold)
                prop.SetFontSize(scaled_font_size if use_3d_text else font_size)

        if use_3d_text:
            font_size_factor = 1.0 if vtk_less_than_96 else scaled_font_size / default_font_size
            self.SetScreenSize(
                font_size / default_font_size / font_size_factor * default_screen_size
            )
        elif vtk_less_than_96:
            self.SetScreenSize(font_size / default_font_size * default_screen_size)

    def _configure_fly_mode(self, *, location: str | None) -> None:
        """Set how the axes are drawn in relation to the camera position."""
        if location is None:
            return
        if not isinstance(location, str):
            msg = 'location must be a string'  # type: ignore[unreachable]
            raise TypeError(msg)
        location = location.lower()
        if location == 'all':
            self.SetFlyModeToStaticEdges()
        elif location == 'origin':
            self.SetFlyModeToStaticTriad()
        elif location == 'outer':
            self.SetFlyModeToOuterEdges()
        elif location in ('default', 'closest', 'front'):
            self.SetFlyModeToClosestTriad()
        elif location in ('furthest', 'back'):
            self.SetFlyModeToFurthestTriad()
        else:
            msg = (
                f'Value of location ("{location}") should be either "all", "origin",'
                ' "outer", "default", "closest", "front", "furthest", or "back".'
            )
            raise ValueError(msg)

    @property
    def tick_location(self) -> str:  # numpydoc ignore=RT01
        """Return or set how the ticks are drawn on the axes grid.

        Options include: ``'inside', 'outside', 'both'``.
        """
        tloc = self.GetTickLocation()
        if tloc == 0:
            return 'inside'
        if tloc == 1:
            return 'outside'
        return 'both'

    @tick_location.setter
    def tick_location(self, value: str):
        if not isinstance(value, str):
            msg = f'`tick_location` must be a string, not {type(value)}'  # type: ignore[unreachable]
            raise TypeError(msg)
        value = value.lower()
        if value in ('inside'):
            self.SetTickLocationToInside()
        elif value in ('outside'):
            self.SetTickLocationToOutside()
        elif value in ('both'):
            self.SetTickLocationToBoth()
        else:
            msg = (
                f'Value of tick_location ("{value}") should be either "inside", "outside", '
                'or "both".'
            )
            raise ValueError(msg)

    @property
    def bounds(self) -> BoundsTuple:  # numpydoc ignore=RT01
        """Return or set the bounding box."""
        return BoundsTuple(*self.GetBounds())

    @bounds.setter
    def bounds(self, bounds: VectorLike[float]):
        self.SetBounds(bounds)  # type: ignore[arg-type]
        self._update_labels()
        bnds = self.bounds
        self.x_axis_range = bnds.x_min, bnds.x_max
        self.y_axis_range = bnds.y_min, bnds.y_max
        self.z_axis_range = bnds.z_min, bnds.z_max

    @property
    def center(self) -> tuple[float, float, float]:
        """Return the center.

        Returns
        -------
        tuple[float, float, float]
            Center of axes actor.

        """
        return self.GetCenter()

    @property
    def x_axis_range(self) -> tuple[float, float]:  # numpydoc ignore=RT01
        """Return or set the x-axis range."""
        return self.GetXAxisRange()

    @x_axis_range.setter
    def x_axis_range(self, value: tuple[float, float]):
        self.SetXAxisRange(value)
        self._update_x_labels()

    @property
    def y_axis_range(self) -> tuple[float, float]:  # numpydoc ignore=RT01
        """Return or set the y-axis range."""
        return self.GetYAxisRange()

    @y_axis_range.setter
    def y_axis_range(self, value: tuple[float, float]):
        self.SetYAxisRange(value)
        self._update_y_labels()

    @property
    def z_axis_range(self) -> tuple[float, float]:  # numpydoc ignore=RT01
        """Return or set the z-axis range."""
        return self.GetZAxisRange()

    @z_axis_range.setter
    def z_axis_range(self, value: tuple[float, float]):
        self.SetZAxisRange(value)
        self._update_z_labels()

    @property
    def label_offset(self) -> float:  # numpydoc ignore=RT01
        """Return or set the distance between labels and the axis."""
        return self.GetLabelOffset()

    @label_offset.setter
    def label_offset(self, offset: float):
        self.SetLabelOffset(offset)

    @property
    def title_offset(self) -> float | tuple[float, float]:  # numpydoc ignore=RT01
        """Return or set the distance between title and labels."""
        if pv.vtk_version_info < (9, 5, 0):
            offx, offy = (_vtk.reference(0.0), _vtk.reference(0.0))
            self.GetTitleOffset(offx, offy)  # type: ignore[call-arg]
            return offx, offy  # type: ignore[return-value]

        return self.GetTitleOffset()

    @title_offset.setter
    def title_offset(self, offset: Sequence[float]):
        self.SetTitleOffset(list(offset))

    @property
    def camera(self) -> pv.Camera:  # numpydoc ignore=RT01
        """Return or set the camera that performs scaling and translation."""
        return self.GetCamera()

    @camera.setter
    def camera(self, camera: pv.Camera):
        self.SetCamera(camera)

    @property
    def x_axis_minor_tick_visibility(self) -> bool:  # numpydoc ignore=RT01
        """Return or set visibility of the x-axis minor tick."""
        return bool(self.GetXAxisMinorTickVisibility())

    @x_axis_minor_tick_visibility.setter
    def x_axis_minor_tick_visibility(self, value: bool):
        self.SetXAxisMinorTickVisibility(value)

    @property
    def y_axis_minor_tick_visibility(self) -> bool:  # numpydoc ignore=RT01
        """Return or set visibility of the y-axis minor tick."""
        return bool(self.GetYAxisMinorTickVisibility())

    @y_axis_minor_tick_visibility.setter
    def y_axis_minor_tick_visibility(self, value: bool):
        self.SetYAxisMinorTickVisibility(value)

    @property
    def z_axis_minor_tick_visibility(self) -> bool:  # numpydoc ignore=RT01
        """Return or set visibility of the z-axis minor tick."""
        return bool(self.GetZAxisMinorTickVisibility())

    @z_axis_minor_tick_visibility.setter
    def z_axis_minor_tick_visibility(self, value: bool):
        self.SetZAxisMinorTickVisibility(value)

    @property
    def x_label_visibility(self) -> bool:  # numpydoc ignore=RT01
        """Return or set the visibility of the x-axis labels."""
        return self._x_label_visibility

    @x_label_visibility.setter
    def x_label_visibility(self, value: bool):
        self._x_label_visibility = bool(value)
        self._update_x_labels()

    @property
    def y_label_visibility(self) -> bool:  # numpydoc ignore=RT01
        """Return or set the visibility of the y-axis labels."""
        return self._y_label_visibility

    @y_label_visibility.setter
    def y_label_visibility(self, value: bool):
        self._y_label_visibility = bool(value)
        self._update_y_labels()

    @property
    def z_label_visibility(self) -> bool:  # numpydoc ignore=RT01
        """Return or set the visibility of the z-axis labels."""
        return self._z_label_visibility

    @z_label_visibility.setter
    def z_label_visibility(self, value: bool):
        self._z_label_visibility = bool(value)
        self._update_z_labels()

    @property
    def x_axis_visibility(self) -> bool:  # numpydoc ignore=RT01
        """Return or set the visibility of the x-axis."""
        return bool(self.GetXAxisVisibility())

    @x_axis_visibility.setter
    def x_axis_visibility(self, value: bool):
        self.SetXAxisVisibility(value)

    @property
    def y_axis_visibility(self) -> bool:  # numpydoc ignore=RT01
        """Return or set the visibility of the y-axis."""
        return bool(self.GetYAxisVisibility())

    @y_axis_visibility.setter
    def y_axis_visibility(self, value: bool):
        self.SetYAxisVisibility(value)

    @property
    def z_axis_visibility(self) -> bool:  # numpydoc ignore=RT01
        """Return or set the visibility of the y-axis."""
        return bool(self.GetZAxisVisibility())

    @z_axis_visibility.setter
    def z_axis_visibility(self, value: bool):
        self.SetZAxisVisibility(value)

    @property
    def x_label_format(self) -> str:  # numpydoc ignore=RT01
        """Return or set the label of the x-axis."""
        return self.GetXLabelFormat()

    @x_label_format.setter
    def x_label_format(self, value: str):
        self.SetXLabelFormat(value)
        self._update_x_labels()

    @property
    def y_label_format(self) -> str:  # numpydoc ignore=RT01
        """Return or set the label of the y-axis."""
        return self.GetYLabelFormat()

    @y_label_format.setter
    def y_label_format(self, value: str):
        self.SetYLabelFormat(value)
        self._update_y_labels()

    @property
    def z_label_format(self) -> str:  # numpydoc ignore=RT01
        """Return or set the label of the z-axis."""
        return self.GetZLabelFormat()

    @z_label_format.setter
    def z_label_format(self, value: str):
        self.SetZLabelFormat(value)
        self._update_z_labels()

    @property
    def x_title(self) -> str:  # numpydoc ignore=RT01
        """Return or set the title of the x-axis."""
        return self._x_title

    @x_title.setter
    def x_title(self, value: str):
        _validation.check_string(value, name='x_title')
        self._x_title = value
        self._update_x_labels()

    @property
    def y_title(self) -> str:  # numpydoc ignore=RT01
        """Return or set the title of the y-axis."""
        return self._y_title

    @y_title.setter
    def y_title(self, value: str):
        _validation.check_string(value, name='y_title')
        self._y_title = value
        self._update_y_labels()

    @property
    def z_title(self) -> str:  # numpydoc ignore=RT01
        """Return or set the title of the z-axis."""
        return self._z_title

    @z_title.setter
    def z_title(self, value: str):
        _validation.check_string(value, name='z_title')
        self._z_title = value
        self._update_z_labels()

    @property
    def use_2d_mode(self) -> bool:  # numpydoc ignore=RT01
        """Use the 2d render mode.

        This can be enabled for smoother plotting.
        """
        return bool(self.GetUse2DMode())

    @use_2d_mode.setter
    def use_2d_mode(self, value: bool):
        self.SetUse2DMode(value)

    @property
    def n_xlabels(self):  # numpydoc ignore=RT01
        """Number of labels on the x-axis."""
        return self._n_xlabels

    @n_xlabels.setter
    def n_xlabels(self, value: int):
        self._n_xlabels = value
        self._update_x_labels()

    @property
    def n_ylabels(self):  # numpydoc ignore=RT01
        """Number of labels on the y-axis."""
        return self._n_ylabels

    @n_ylabels.setter
    def n_ylabels(self, value: int):
        self._n_ylabels = value
        self._update_y_labels()

    @property
    def n_zlabels(self):  # numpydoc ignore=RT01
        """Number of labels on the z-axis."""
        return self._n_zlabels

    @n_zlabels.setter
    def n_zlabels(self, value: int):
        self._n_zlabels = value
        self._update_z_labels()

    def _update_labels(self):
        """Update all labels."""
        self._update_x_labels()
        self._update_y_labels()
        self._update_z_labels()

    def _update_x_labels(self):
        """Regenerate x-axis labels."""
        if self.x_axis_visibility:
            self.SetXTitle(self._x_title)
            if self._x_label_visibility:
                vmin, vmax = self.x_axis_range
                self.SetAxisLabels(
                    0,
                    make_axis_labels(
                        vmin=vmin, vmax=vmax, n=self.n_xlabels, fmt=self.x_label_format
                    ),
                )
            else:
                self.SetAxisLabels(0, self._empty_str)
        else:
            self.SetXTitle(' ')
            self.SetAxisLabels(0, self._empty_str)

    def _update_y_labels(self):
        """Regenerate y-axis labels."""
        if self.y_axis_visibility:
            self.SetYTitle(self._y_title)
            if self._y_label_visibility:
                vmin, vmax = self.y_axis_range
                self.SetAxisLabels(
                    1,
                    make_axis_labels(
                        vmin=vmin, vmax=vmax, n=self.n_ylabels, fmt=self.y_label_format
                    ),
                )
            else:
                self.SetAxisLabels(1, self._empty_str)
        else:
            self.SetYTitle(' ')
            self.SetAxisLabels(1, self._empty_str)

    def _update_z_labels(self):
        """Regenerate z-axis labels."""
        if self.z_axis_visibility:
            self.SetZTitle(self._z_title)
            if self._z_label_visibility:
                vmin, vmax = self.z_axis_range
                self.SetAxisLabels(
                    2,
                    make_axis_labels(
                        vmin=vmin, vmax=vmax, n=self.n_zlabels, fmt=self.z_label_format
                    ),
                )
            else:
                self.SetAxisLabels(2, self._empty_str)
        else:
            self.SetZTitle(' ')
            self.SetAxisLabels(2, self._empty_str)

    @property
    def x_labels(self) -> list[str]:  # numpydoc ignore=RT01
        """Return the x-axis labels."""
        labels_vtk = cast('_vtk.vtkStringArray', self.GetAxisLabels(0))
        return convert_string_array(labels_vtk).tolist()

    @property
    def y_labels(self) -> list[str]:  # numpydoc ignore=RT01
        """Return the y-axis labels."""
        labels_vtk = cast('_vtk.vtkStringArray', self.GetAxisLabels(1))
        return convert_string_array(labels_vtk).tolist()

    @property
    def z_labels(self) -> list[str]:  # numpydoc ignore=RT01
        """Return the z-axis labels."""
        labels_vtk = cast('_vtk.vtkStringArray', self.GetAxisLabels(2))
        return convert_string_array(labels_vtk).tolist()

    def update_bounds(self, bounds):
        """Update the bounds of this actor.

        Unlike the :attr:`CubeAxesActor.bounds` attribute, updating the bounds
        also updates the axis labels.

        Parameters
        ----------
        bounds : sequence[float]
            Bounds in the form of ``(x_min, x_max, y_min, y_max, z_min, z_max)``.

        """
        self.bounds = bounds
