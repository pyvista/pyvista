"""Module containing the wrapping of CubeAxesActor."""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import TYPE_CHECKING
from typing import cast
import warnings

import numpy as np

import pyvista as pv
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core._typing_core import BoundsTuple
from pyvista.core.utilities.arrays import convert_string_array
from pyvista.core.utilities.misc import _BoundsSizeMixin
from pyvista.core.utilities.misc import _NameMixin
from pyvista.core.utilities.misc import _NoNewAttrMixin

from . import _vtk

if TYPE_CHECKING:
    from pyvista.core._typing_core import VectorLike


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
        The number of labels to create.
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
    for v in np.linspace(vmin, vmax, n):
        label = (fmt % v if fmt.startswith('%') else fmt.format(v)) if fmt else f'{v}'
        labels.InsertNextValue(label)
    return labels


class CubeAxesActor(
    _NoNewAttrMixin, _NameMixin, _BoundsSizeMixin, _vtk.DisableVtkSnakeCase, _vtk.vtkCubeAxesActor
):
    """Wrap :vtk:`vtkCubeAxesActor`.

    This class is created to wrap :vtk:`vtkCubeAxesActor`, which is used to draw axes
    and labels for the input data bounds. This wrapping aims to provide a
    user-friendly interface to use :vtk:`vtkCubeAxesActor`.

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
        Number of labels along the x-axis.

    n_ylabels : int, default: 5
        Number of labels along the y-axis.

    n_zlabels : int, default: 5
        Number of labels along the z-axis.

    See Also
    --------
    :meth:`~pyvista.Plotter.show_bounds`
    :meth:`~pyvista.Plotter.show_grid`
    :ref:`axes_objects_example`
        Example showing different axes objects.

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

        # TODO: Change this to (9, 6, 0) when VTK 9.6 is released
        default_fmt = '%.1f' if pv.vtk_version_info < (9, 5, 99) else '{0:.1f}'
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
        if (9, 3, 0) <= pv.vtk_version_info < (9, 5, 0):
            offx, offy = (_vtk.reference(0.0), _vtk.reference(0.0))
            self.GetTitleOffset(offx, offy)  # type: ignore[call-arg]
            return offx, offy  # type: ignore[return-value]

        return self.GetTitleOffset()

    @title_offset.setter
    def title_offset(self, offset: float | MutableSequence[float]):
        vtk_geq_9_3 = pv.vtk_version_info >= (9, 3)

        if vtk_geq_9_3:
            if isinstance(offset, float):
                msg = (
                    f'Setting title_offset with a float is deprecated from vtk >= 9.3. '
                    f'Accepts now a sequence of (x,y) offsets. '
                    f'Setting the x offset to {(x := 0.0)}'
                )
                warnings.warn(msg, UserWarning, stacklevel=2)
                self.SetTitleOffset([x, offset])
            else:
                self.SetTitleOffset(offset)
            return

        if isinstance(offset, MutableSequence):
            msg = (
                f'Setting title_offset with a sequence is only supported from vtk >= 9.3. '
                f'Considering only the second value (ie. y-offset) of {(y := offset[1])}'
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            self.SetTitleOffset(y)  # type: ignore[arg-type]
            return

        self.SetTitleOffset(offset)  # type: ignore[arg-type]

    @property
    def camera(self) -> pv.Camera:  # numpydoc ignore=RT01
        """Return or set the camera that performs scaling and translation."""
        return self.GetCamera()

    @camera.setter
    def camera(self, camera: pv.Camera):
        self.SetCamera(camera)

    @property
    def x_axis_minor_tick_visibility(self) -> bool:  # numpydoc ignore=RT01
        """Return or set visibility of the x-axis minior tick."""
        return bool(self.GetXAxisMinorTickVisibility())

    @x_axis_minor_tick_visibility.setter
    def x_axis_minor_tick_visibility(self, value: bool):
        self.SetXAxisMinorTickVisibility(value)

    @property
    def y_axis_minor_tick_visibility(self) -> bool:  # numpydoc ignore=RT01
        """Return or set visibility of the y-axis minior tick."""
        return bool(self.GetYAxisMinorTickVisibility())

    @y_axis_minor_tick_visibility.setter
    def y_axis_minor_tick_visibility(self, value: bool):
        self.SetYAxisMinorTickVisibility(value)

    @property
    def z_axis_minor_tick_visibility(self) -> bool:  # numpydoc ignore=RT01
        """Return or set visibility of the z-axis minior tick."""
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
        self._x_title = value
        self._update_x_labels()

    @property
    def y_title(self) -> str:  # numpydoc ignore=RT01
        """Return or set the title of the y-axis."""
        return self._y_title

    @y_title.setter
    def y_title(self, value: str):
        self._y_title = value
        self._update_y_labels()

    @property
    def z_title(self) -> str:  # numpydoc ignore=RT01
        """Return or set the title of the z-axis."""
        return self._z_title

    @z_title.setter
    def z_title(self, value: str):
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
