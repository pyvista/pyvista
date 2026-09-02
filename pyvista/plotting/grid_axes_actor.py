"""Module containing the wrapping of GridAxesActor."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import cast

import numpy as np

import pyvista as pv
from pyvista import _vtk
from pyvista.core import _validation
from pyvista.core._typing_core import BoundsTuple
from pyvista.core._vtk_utilities import DisableVtkSnakeCase
from pyvista.core.utilities.misc import _BoundsSizeMixin
from pyvista.core.utilities.misc import _NameMixin
from pyvista.core.utilities.misc import _NoNewAttrMixin
from pyvista.plotting.colors import Color
from pyvista.plotting.tools import parse_font_family

if TYPE_CHECKING:
    from pyvista.core._typing_core import VectorLike
    from pyvista.plotting._typing import ColorLike

# Minimum VTK version providing vtkGridAxesActor3D
GRID_AXES_MIN_VTK_VERSION = (9, 5, 0)

if TYPE_CHECKING:
    _GridAxesActor3DBase = _vtk.vtkGridAxesActor3D
else:  # The class is unavailable before VTK 9.5; __init__ raises with a clear message
    try:
        _GridAxesActor3DBase = _vtk.vtkGridAxesActor3D
    except ImportError:  # pragma: no cover
        _GridAxesActor3DBase = object

_ALL_FACES = 0b111111
_AUTOMATIC_NOTATION = _vtk.vtkAxis.STANDARD_NOTATION
_FIXED_NOTATION = _vtk.vtkAxis.FIXED_NOTATION
_LABEL_FORMAT_PATTERN = re.compile(r'^(?:%\.(\d+)f|\{0?:\.(\d+)f\})$')


def _parse_label_format(fmt: str, name: str) -> int:
    """Return the precision of a fixed-point format string."""
    match = _LABEL_FORMAT_PATTERN.match(fmt)
    if match is None:
        msg = (
            f'{name} {fmt!r} is not supported by GridAxesActor, which formats labels with a\n'
            f'fixed number of decimal places. Use a plain fixed-point format such as '
            f"'%.2f' or '{{0:.2f}}'."
        )
        raise ValueError(msg)
    return int(match.group(1) or match.group(2))


class GridAxesActor(
    _NoNewAttrMixin, _NameMixin, _BoundsSizeMixin, DisableVtkSnakeCase, _GridAxesActor3DBase
):
    r"""Wrap :vtk:`vtkGridAxesActor3D`.

    Draw a labelled grid around the input data bounds. Grid lines are drawn on the
    faces of the box furthest from the camera, and each axis is labelled on the
    silhouette edges, in the manner of :func:`matplotlib.pyplot.grid`.

    Unlike :class:`~pyvista.CubeAxesActor` this actor renders titles with
    :vtk:`vtkBillboardTextActor3D`, so titles containing math text such as
    ``r'$\rho$'`` are rendered as symbols.

    .. versionadded:: 0.49

    .. note::

        Requires VTK 9.5 or later.

    .. warning::

        The ``'trame'`` Jupyter backend draws this actor's box but not its labels or
        titles. Use :class:`~pyvista.CubeAxesActor` for interactive scenes.

    Parameters
    ----------
    bounds : sequence[float], optional
        Bounds of the axes in the form ``(x_min, x_max, y_min, y_max, z_min, z_max)``.
        Defaults to ``(-1, 1, -1, 1, -1, 1)``.

    axes_ranges : sequence[float], optional
        Values shown on the axes in the form
        ``(x_min, x_max, y_min, y_max, z_min, z_max)``. These override the values
        derived from ``bounds``.

    padding : float, default: 0.0
        Percent padding applied to ``bounds`` along each axial direction to cushion
        the datasets in the scene from the axes annotations.

    x_title : str, default: "X Axis"
        Title of the x-axis.

    y_title : str, default: "Y Axis"
        Title of the y-axis.

    z_title : str, default: "Z Axis"
        Title of the z-axis.

    x_axis_visibility : bool, default: True
        Visibility of the x-axis, its labels, and its grid lines.

    y_axis_visibility : bool, default: True
        Visibility of the y-axis, its labels, and its grid lines.

    z_axis_visibility : bool, default: True
        Visibility of the z-axis, its labels, and its grid lines.

    x_label_visibility : bool, default: True
        The visibility of the x-axis labels. Hiding them also hides that axis's
        ticks and grid lines.

    y_label_visibility : bool, default: True
        The visibility of the y-axis labels.

    z_label_visibility : bool, default: True
        The visibility of the z-axis labels.

    n_xlabels : int, optional
        Number of evenly spaced labels along the x-axis. By default the axis chooses
        its own labels at rounded values.

    n_ylabels : int, optional
        Number of evenly spaced labels along the y-axis. By default the axis chooses
        its own labels at rounded values.

    n_zlabels : int, optional
        Number of evenly spaced labels along the z-axis. By default the axis chooses
        its own labels at rounded values.

    x_label_format : str, optional
        A fixed-point format string such as ``'%.2f'`` or ``'{0:.2f}'`` defining how
        x-axis tick labels are generated. Defaults to the theme format if set,
        otherwise the axis chooses its own formatting.

    y_label_format : str, optional
        A fixed-point format string such as ``'%.2f'`` or ``'{0:.2f}'`` defining how
        y-axis tick labels are generated. Defaults to the theme format if set,
        otherwise the axis chooses its own formatting.

    z_label_format : str, optional
        A fixed-point format string such as ``'%.2f'`` or ``'{0:.2f}'`` defining how
        z-axis tick labels are generated. Defaults to the theme format if set,
        otherwise the axis chooses its own formatting.

    color : ColorLike, optional
        Color of all labels, axis titles, and grid lines. Defaults to
        :attr:`pyvista.global_theme.font.color
        <pyvista.plotting.themes._Font.color>`.

    font_size : float, optional
        Size of the label font. Defaults to :attr:`pyvista.global_theme.font.size
        <pyvista.plotting.themes._Font.size>`.

    font_family : str, optional
        Font family. Must be either ``'courier'``, ``'times'``, or ``'arial'``.
        Defaults to :attr:`pyvista.global_theme.font.family
        <pyvista.plotting.themes._Font.family>`.

    bold : bool, default: True
        Bold the axis labels and titles.

    grid : bool, default: True
        Draw grid lines across the faces of the box.

    show_ticks : bool, default: True
        Draw tick marks alongside the labels.

    unique_edges_only : bool, default: True
        Label only the edges that belong to a single visible face. When ``False`` an
        edge shared by two faces is labelled twice.

    label_display_offset : sequence[int], optional
        Offset of the labels from their edge, in display coordinates, as
        ``(x_offset, y_offset)``, by default ``(0, 0)``. Useful when labels overlap
        at the corners.

    See Also
    --------
    :class:`~pyvista.CubeAxesActor`
    :meth:`~pyvista.Plotter.show_bounds`
    :meth:`~pyvista.Plotter.show_grid`

    Examples
    --------
    Add grid axes to a plot. Note that no camera is required.

    >>> import pyvista as pv
    >>> mesh = pv.Sphere()
    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(mesh)
    >>> grid_axes_actor = pv.GridAxesActor(bounds=mesh.bounds)
    >>> _ = pl.add_actor(grid_axes_actor)
    >>> pl.show()

    Titles may contain math text.

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(mesh)
    >>> grid_axes_actor = pv.GridAxesActor(
    ...     bounds=mesh.bounds,
    ...     x_title=r'$\rho$',
    ...     y_title=r'$\eta$',
    ...     z_title=r'$\mu$',
    ... )
    >>> _ = pl.add_actor(grid_axes_actor)
    >>> pl.show()

    """

    def __init__(
        self,
        *,
        bounds: VectorLike[float] | None = None,
        axes_ranges: VectorLike[float] | None = None,
        padding: float = 0.0,
        x_title: str = 'X Axis',
        y_title: str = 'Y Axis',
        z_title: str = 'Z Axis',
        x_axis_visibility: bool = True,
        y_axis_visibility: bool = True,
        z_axis_visibility: bool = True,
        x_label_visibility: bool = True,
        y_label_visibility: bool = True,
        z_label_visibility: bool = True,
        n_xlabels: int | None = None,
        n_ylabels: int | None = None,
        n_zlabels: int | None = None,
        x_label_format: str | None = None,
        y_label_format: str | None = None,
        z_label_format: str | None = None,
        color: ColorLike | None = None,
        font_size: float | None = None,
        font_family: str | None = None,
        bold: bool = True,
        grid: bool = True,
        show_ticks: bool = True,
        unique_edges_only: bool = True,
        label_display_offset: VectorLike[int] | None = None,
    ) -> None:
        """Initialize GridAxesActor."""
        if pv.vtk_version_info < GRID_AXES_MIN_VTK_VERSION:
            msg = (
                f'GridAxesActor requires VTK '
                f'{".".join(str(v) for v in GRID_AXES_MIN_VTK_VERSION)} or later, '
                f'but VTK {pv.vtk_version_info} is installed.\n'
                f'Use CubeAxesActor instead.'
            )
            raise pv.VTKVersionError(msg)
        super().__init__()

        # GetProperty returns face 0 only; SetProperty ignores a property it already holds
        self._property = _vtk.vtkProperty()
        self.SetProperty(self._property)
        self.SetFaceMask(_ALL_FACES)

        color_ = Color(color, default_color=pv.global_theme.font.color)
        self._property.SetColor(*color_.float_rgb)
        self._property.SetFrontfaceCulling(True)

        self.grid = grid
        self.show_ticks = show_ticks
        self.unique_edges_only = unique_edges_only
        if label_display_offset is not None:
            self.label_display_offset = label_display_offset

        for name, title in zip('xyz', (x_title, y_title, z_title), strict=True):
            _validation.check_string(title, name=f'{name}_title')

        self._titles = [x_title, y_title, z_title]
        self._axis_visibility = [x_axis_visibility, y_axis_visibility, z_axis_visibility]
        self._label_visibility = [x_label_visibility, y_label_visibility, z_label_visibility]
        self._n_labels = [n_xlabels, n_ylabels, n_zlabels]
        self._label_formats = [x_label_format, y_label_format, z_label_format]
        self._ranges = np.zeros(6)

        self._configure_text(color=color_, font_size=font_size, font_family=font_family, bold=bold)
        self._configure_label_format()

        if not (isinstance(padding, (int, float)) and 0.0 <= padding < 1.0):  # type: ignore[redundant-expr]
            msg = f'padding ({padding}) not understood. Must be float between 0 and 1'
            raise ValueError(msg)
        self._padding = padding
        self._axes_ranges = axes_ranges
        self.bounds = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0) if bounds is None else bounds

    def _configure_text(
        self, *, color: Color, font_size: float | None, font_family: str | None, bold: bool
    ) -> None:
        """Set the color, font, and weight of the titles and labels."""
        font_size = pv.global_theme.font.size if font_size is None else font_size
        font_family = pv.global_theme.font.family if font_family is None else font_family
        family = parse_font_family(font_family)
        for axis in range(3):
            for prop in (self.GetTitleTextProperty(axis), self.GetLabelTextProperty(axis)):
                prop.SetColor(*color.float_rgb)
                prop.SetFontFamily(family)
                prop.SetFontSize(font_size)
                prop.SetBold(bold)

    def _configure_label_format(self) -> None:
        """Set the notation and precision used to format tick labels."""
        names = ('x_label_format', 'y_label_format', 'z_label_format')
        for axis, (fmt, name) in enumerate(zip(self._label_formats, names, strict=True)):
            resolved = pv.global_theme.font.fmt if fmt is None else fmt
            if resolved is None:
                self.SetNotation(axis, _AUTOMATIC_NOTATION)  # type: ignore[unreachable]
                continue
            self.SetNotation(axis, _FIXED_NOTATION)
            self.SetPrecision(axis, _parse_label_format(resolved, name))

    def _refresh(self) -> None:
        """Re-apply the labels and titles for the current ranges."""
        self._apply_labels(self._ranges)
        for axis, title in enumerate(self._titles):
            self.SetTitle(axis, title if self._axis_visibility[axis] else '')

    def _apply_labels(self, ranges: np.ndarray) -> None:
        """Set the tick positions for each axis."""
        for axis in range(3):
            visible = self._axis_visibility[axis] and self._label_visibility[axis]
            n_labels = self._n_labels[axis]
            if visible and n_labels is None:
                self.SetUseCustomLabels(axis, False)
                continue
            # An axis with no tick positions draws no labels, ticks, or grid lines
            count = 0 if not visible else cast('int', n_labels)
            self.SetUseCustomLabels(axis, True)
            self.SetNumberOfLabels(axis, count)
            if count:
                lo, hi = ranges[2 * axis], ranges[2 * axis + 1]
                for index, value in enumerate(np.linspace(lo, hi, count)):
                    self.SetLabel(axis, index, float(value))

    @property
    def bounds(self) -> BoundsTuple:  # numpydoc ignore=RT01
        """Return or set the bounding box."""
        matrix = self.GetMatrix()
        ranges = self.GetGridBounds()
        corners = [
            matrix.MultiplyPoint((ranges[0], ranges[2], ranges[4], 1.0)),
            matrix.MultiplyPoint((ranges[1], ranges[3], ranges[5], 1.0)),
        ]
        return BoundsTuple(*[c[axis] for axis in range(3) for c in corners])

    @bounds.setter
    def bounds(self, bounds: VectorLike[float]) -> None:
        bounds = np.asarray(
            _validation.validate_array(
                bounds, must_have_shape=(6,), name='bounds', dtype_out=float
            ),
            dtype=float,
        )
        if self._padding and not np.any(np.abs(bounds) == np.inf):
            cushion = np.abs(bounds[1::2] - bounds[::2]) * self._padding
            bounds = bounds.copy()
            bounds[::2] -= cushion
            bounds[1::2] += cushion

        if self._axes_ranges is None:
            ranges = bounds
        else:
            ranges = np.asarray(
                _validation.validate_array(
                    self._axes_ranges, must_have_shape=(6,), name='axes_ranges', dtype_out=float
                ),
                dtype=float,
            )
        self.SetGridBounds(*(float(value) for value in ranges))

        # The grid is built in range coordinates; map that box onto the bounds box
        matrix = _vtk.vtkMatrix4x4()
        for axis in range(3):
            r_min, r_max = ranges[2 * axis], ranges[2 * axis + 1]
            b_min, b_max = bounds[2 * axis], bounds[2 * axis + 1]
            scale = 1.0 if r_max == r_min else (b_max - b_min) / (r_max - r_min)
            matrix.SetElement(axis, axis, float(scale))
            matrix.SetElement(axis, 3, float(b_min - scale * r_min))
        self.SetUserMatrix(matrix)

        self._ranges = np.asarray(ranges, dtype=float)
        self._refresh()

    @property
    def center(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return the center of the axes."""
        bnds = self.bounds
        return (
            (bnds.x_min + bnds.x_max) / 2,
            (bnds.y_min + bnds.y_max) / 2,
            (bnds.z_min + bnds.z_max) / 2,
        )

    @property
    def grid(self) -> bool:  # numpydoc ignore=RT01
        """Return or set whether grid lines are drawn across the faces."""
        return bool(self.GetGenerateGrid())

    @grid.setter
    def grid(self, value: bool) -> None:
        self.SetGenerateGrid(bool(value))

    @property
    def show_ticks(self) -> bool:  # numpydoc ignore=RT01
        """Return or set whether tick marks are drawn alongside the labels."""
        return bool(self.GetGenerateTicks())

    @show_ticks.setter
    def show_ticks(self, value: bool) -> None:
        self.SetGenerateTicks(bool(value))

    @property
    def unique_edges_only(self) -> bool:  # numpydoc ignore=RT01
        """Return or set whether only edges belonging to one visible face are labelled."""
        return bool(self.GetLabelUniqueEdgesOnly())

    @unique_edges_only.setter
    def unique_edges_only(self, value: bool) -> None:
        self.SetLabelUniqueEdgesOnly(bool(value))

    @property
    def label_display_offset(self) -> tuple[int, int]:  # numpydoc ignore=RT01
        """Return or set the display-space offset of the labels from their edge."""
        offset = self.GetLabelDisplayOffset()
        return int(offset[0]), int(offset[1])

    @label_display_offset.setter
    def label_display_offset(self, value: VectorLike[int]) -> None:
        offset = _validation.validate_array(
            value, must_have_shape=(2,), name='label_display_offset', dtype_out=int
        )
        self.SetLabelDisplayOffset(int(offset[0]), int(offset[1]))

    @property
    def x_title(self) -> str:  # numpydoc ignore=RT01
        """Return or set the title of the x-axis."""
        return self._titles[0]

    @x_title.setter
    def x_title(self, value: str) -> None:
        _validation.check_string(value, name='x_title')
        self._titles[0] = value
        self._refresh()

    @property
    def x_axis_visibility(self) -> bool:  # numpydoc ignore=RT01
        """Return or set the visibility of the x-axis."""
        return self._axis_visibility[0]

    @x_axis_visibility.setter
    def x_axis_visibility(self, value: bool) -> None:
        self._axis_visibility[0] = bool(value)
        self._refresh()

    @property
    def x_label_visibility(self) -> bool:  # numpydoc ignore=RT01
        """Return or set the visibility of the x-axis labels."""
        return self._label_visibility[0]

    @x_label_visibility.setter
    def x_label_visibility(self, value: bool) -> None:
        self._label_visibility[0] = bool(value)
        self._refresh()

    @property
    def n_xlabels(self) -> int | None:  # numpydoc ignore=RT01
        """Return or set the number of labels on the x-axis."""
        return self._n_labels[0]

    @n_xlabels.setter
    def n_xlabels(self, value: int | None) -> None:
        self._n_labels[0] = value
        self._refresh()

    @property
    def x_label_format(self) -> str | None:  # numpydoc ignore=RT01
        """Return or set the fixed-point format string of the x-axis labels."""
        return self._label_formats[0]

    @x_label_format.setter
    def x_label_format(self, value: str | None) -> None:
        self._label_formats[0] = value
        self._configure_label_format()

    @property
    def y_title(self) -> str:  # numpydoc ignore=RT01
        """Return or set the title of the y-axis."""
        return self._titles[1]

    @y_title.setter
    def y_title(self, value: str) -> None:
        _validation.check_string(value, name='y_title')
        self._titles[1] = value
        self._refresh()

    @property
    def y_axis_visibility(self) -> bool:  # numpydoc ignore=RT01
        """Return or set the visibility of the y-axis."""
        return self._axis_visibility[1]

    @y_axis_visibility.setter
    def y_axis_visibility(self, value: bool) -> None:
        self._axis_visibility[1] = bool(value)
        self._refresh()

    @property
    def y_label_visibility(self) -> bool:  # numpydoc ignore=RT01
        """Return or set the visibility of the y-axis labels."""
        return self._label_visibility[1]

    @y_label_visibility.setter
    def y_label_visibility(self, value: bool) -> None:
        self._label_visibility[1] = bool(value)
        self._refresh()

    @property
    def n_ylabels(self) -> int | None:  # numpydoc ignore=RT01
        """Return or set the number of labels on the y-axis."""
        return self._n_labels[1]

    @n_ylabels.setter
    def n_ylabels(self, value: int | None) -> None:
        self._n_labels[1] = value
        self._refresh()

    @property
    def y_label_format(self) -> str | None:  # numpydoc ignore=RT01
        """Return or set the fixed-point format string of the y-axis labels."""
        return self._label_formats[1]

    @y_label_format.setter
    def y_label_format(self, value: str | None) -> None:
        self._label_formats[1] = value
        self._configure_label_format()

    @property
    def z_title(self) -> str:  # numpydoc ignore=RT01
        """Return or set the title of the z-axis."""
        return self._titles[2]

    @z_title.setter
    def z_title(self, value: str) -> None:
        _validation.check_string(value, name='z_title')
        self._titles[2] = value
        self._refresh()

    @property
    def z_axis_visibility(self) -> bool:  # numpydoc ignore=RT01
        """Return or set the visibility of the z-axis."""
        return self._axis_visibility[2]

    @z_axis_visibility.setter
    def z_axis_visibility(self, value: bool) -> None:
        self._axis_visibility[2] = bool(value)
        self._refresh()

    @property
    def z_label_visibility(self) -> bool:  # numpydoc ignore=RT01
        """Return or set the visibility of the z-axis labels."""
        return self._label_visibility[2]

    @z_label_visibility.setter
    def z_label_visibility(self, value: bool) -> None:
        self._label_visibility[2] = bool(value)
        self._refresh()

    @property
    def n_zlabels(self) -> int | None:  # numpydoc ignore=RT01
        """Return or set the number of labels on the z-axis."""
        return self._n_labels[2]

    @n_zlabels.setter
    def n_zlabels(self, value: int | None) -> None:
        self._n_labels[2] = value
        self._refresh()

    @property
    def z_label_format(self) -> str | None:  # numpydoc ignore=RT01
        """Return or set the fixed-point format string of the z-axis labels."""
        return self._label_formats[2]

    @z_label_format.setter
    def z_label_format(self, value: str | None) -> None:
        self._label_formats[2] = value
        self._configure_label_format()

    def update_bounds(self, bounds: VectorLike[float]) -> None:
        """Update the bounds of this actor.

        Parameters
        ----------
        bounds : sequence[float]
            Bounds in the form of ``(x_min, x_max, y_min, y_max, z_min, z_max)``.

        """
        self.bounds = bounds
