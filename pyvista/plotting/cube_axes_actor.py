"""Module containing the wrapping of CubeAxesActor."""
from typing import List, Tuple

import numpy as np

import pyvista as pv
from pyvista.core._typing_core import BoundsLike
from pyvista.core.utilities.arrays import convert_string_array

from . import _vtk


def make_axis_labels(vmin, vmax, n, fmt):
    """Create a axis labels as a vtkStringArray."""
    labels = _vtk.vtkStringArray()
    for v in np.linspace(vmin, vmax, n):
        if fmt:
            if fmt.startswith('%'):
                label = fmt % v
            else:
                label = fmt.format(v)
        else:
            label = f'{v}'
        labels.InsertNextValue(label)
    return labels


class CubeAxesActor(_vtk.vtkCubeAxesActor):
    """Wrap vtkCubeAxesActor.

    This class is created to wrap vtkCubeAxesActor, which is used to draw axes
    and labels for the input data bounds. This wrapping aims to provide a
    user-friendly interface to use `vtkCubeAxesActor
    <https://vtk.org/doc/nightly/html/classvtkCubeAxesActor.html>`_.

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
        Title of the X axis.

    y_title : str, default: "Y Axis"
        Title of the Y axis.

    z_title : str, default: "Z Axis"
        Title of the Z axis.

    x_axis_visibility : bool, default: True
        Visibility of the X axis.

    y_axis_visibility : bool, default: True
        Visibility of the Y axis.

    z_axis_visibility : bool, default: True
        Visibility of the Z axis.

    x_label_format : str, optional
        A format string defining how tick labels are generated from tick
        positions for the X axis. Defaults to the theme format if set,
        otherwise ``'%.1f'``.

    y_label_format : str, optional
        A format string defining how tick labels are generated from tick
        positions for the Y axis. Defaults to the theme format if set,
        otherwise ``'%.1f'``.

    z_label_format : str, optional
        A format string defining how tick labels are generated from tick
        positions for the Z axis. Defaults to the theme format if set,
        otherwise ``'%.1f'``.

    x_label_visibility : bool, default: True
        The visibility of the X axis labels.

    y_label_visibility : bool, default: True
        The visibility of the Y axis labels.

    z_label_visibility : bool, default: True
        The visibility of the Z axis labels.

    n_xlabels : int, default: 5
        Number of labels along the X axis.

    n_ylabels : int, default: 5
        Number of labels along the Y axis.

    n_zlabels : int, default: 5
        Number of labels along the Z axis.

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

    def __init__(
        self,
        camera,
        minor_ticks=False,
        tick_location=None,
        x_title='X Axis',
        y_title='Y Axis',
        z_title='Z Axis',
        x_axis_visibility=True,
        y_axis_visibility=True,
        z_axis_visibility=True,
        x_label_format=None,
        y_label_format=None,
        z_label_format=None,
        x_label_visibility=True,
        y_label_visibility=True,
        z_label_visibility=True,
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

        if x_label_format is None:
            x_label_format = pv.global_theme.font.fmt
            if x_label_format is None:
                x_label_format = '%.1f'
        if y_label_format is None:
            y_label_format = pv.global_theme.font.fmt
            if y_label_format is None:
                y_label_format = '%.1f'
        if z_label_format is None:
            z_label_format = pv.global_theme.font.fmt
            if z_label_format is None:
                z_label_format = '%.1f'

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
    def tick_location(self) -> str:
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
            raise TypeError(f'`tick_location` must be a string, not {type(value)}')
        value = value.lower()
        if value in ('inside'):
            self.SetTickLocationToInside()
        elif value in ('outside'):
            self.SetTickLocationToOutside()
        elif value in ('both'):
            self.SetTickLocationToBoth()
        else:
            raise ValueError(
                f'Value of tick_location ("{value}") should be either "inside", "outside", '
                'or "both".'
            )

    @property
    def bounds(self) -> BoundsLike:
        """Return or set the bounding box."""
        return self.GetBounds()

    @bounds.setter
    def bounds(self, bounds: BoundsLike):
        self.SetBounds(bounds)
        self._update_labels()
        self.x_axis_range = float(bounds[0]), float(bounds[1])
        self.y_axis_range = float(bounds[2]), float(bounds[3])
        self.z_axis_range = float(bounds[4]), float(bounds[5])

    @property
    def x_axis_range(self) -> Tuple[float, float]:
        """Return or set the X axis range."""
        return self.GetXAxisRange()

    @x_axis_range.setter
    def x_axis_range(self, value: Tuple[float, float]):
        self.SetXAxisRange(value)

    @property
    def y_axis_range(self) -> Tuple[float, float]:
        """Return or set the Y axis range."""
        return self.GetYAxisRange()

    @y_axis_range.setter
    def y_axis_range(self, value: Tuple[float, float]):
        self.SetYAxisRange(value)

    @property
    def z_axis_range(self) -> Tuple[float, float]:
        """Return or set the Z axis range."""
        return self.GetZAxisRange()

    @z_axis_range.setter
    def z_axis_range(self, value: Tuple[float, float]):
        self.SetZAxisRange(value)

    @property
    def label_offset(self) -> float:
        """Return or set the distance between labels and the axis."""
        return self.GetLabelOffset()

    @label_offset.setter
    def label_offset(self, offset: float):
        self.SetLabelOffset(offset)

    @property
    def title_offset(self) -> float:
        """Return or set the distance between title and labels."""
        return self.GetTitleOffset()

    @title_offset.setter
    def title_offset(self, offset: float):
        self.SetTitleOffset(offset)

    @property
    def camera(self) -> 'pv.Camera':
        """Return or set the camera that performs scaling and translation."""
        return self.GetCamera()

    @camera.setter
    def camera(self, camera: 'pv.Camera'):
        self.SetCamera(camera)

    @property
    def x_axis_minor_tick_visibility(self) -> bool:
        """Return or set visibility of the X axis minior tick."""
        return bool(self.GetXAxisMinorTickVisibility())

    @x_axis_minor_tick_visibility.setter
    def x_axis_minor_tick_visibility(self, value: bool):
        self.SetXAxisMinorTickVisibility(value)

    @property
    def y_axis_minor_tick_visibility(self) -> bool:
        """Return or set visibility of the Y axis minior tick."""
        return bool(self.GetYAxisMinorTickVisibility())

    @y_axis_minor_tick_visibility.setter
    def y_axis_minor_tick_visibility(self, value: bool):
        self.SetYAxisMinorTickVisibility(value)

    @property
    def z_axis_minor_tick_visibility(self) -> bool:
        """Return or set visibility of the Z axis minior tick."""
        return bool(self.GetZAxisMinorTickVisibility())

    @z_axis_minor_tick_visibility.setter
    def z_axis_minor_tick_visibility(self, value: bool):
        self.SetZAxisMinorTickVisibility(value)

    @property
    def x_label_visibility(self) -> bool:
        """Return or set the visibility of the X axis labels."""
        return self._x_label_visibility

    @x_label_visibility.setter
    def x_label_visibility(self, value: bool):
        self._x_label_visibility = bool(value)
        self._update_x_labels()

    @property
    def y_label_visibility(self) -> bool:
        """Return or set the visibility of the Y axis labels."""
        return self._y_label_visibility

    @y_label_visibility.setter
    def y_label_visibility(self, value: bool):
        self._y_label_visibility = bool(value)
        self._update_y_labels()

    @property
    def z_label_visibility(self) -> bool:
        """Return or set the visibility of the Z axis labels."""
        return self._z_label_visibility

    @z_label_visibility.setter
    def z_label_visibility(self, value: bool):
        self._z_label_visibility = bool(value)
        self._update_z_labels()

    @property
    def x_axis_visibility(self) -> bool:
        """Return or set the visibility of the X axis."""
        return bool(self.GetXAxisVisibility())

    @x_axis_visibility.setter
    def x_axis_visibility(self, value: bool):
        self.SetXAxisVisibility(value)

    @property
    def y_axis_visibility(self) -> bool:
        """Return or set the visibility of the Y axis."""
        return bool(self.GetYAxisVisibility())

    @y_axis_visibility.setter
    def y_axis_visibility(self, value: bool):
        self.SetYAxisVisibility(value)

    @property
    def z_axis_visibility(self) -> bool:
        """Return or set the visibility of the Y axis."""
        return bool(self.GetZAxisVisibility())

    @z_axis_visibility.setter
    def z_axis_visibility(self, value: bool):
        self.SetZAxisVisibility(value)

    @property
    def x_label_format(self) -> str:
        """Return or set the label of the X axis."""
        return self.GetXLabelFormat()

    @x_label_format.setter
    def x_label_format(self, value: str):
        self.SetXLabelFormat(value)
        self._update_x_labels()

    @property
    def y_label_format(self) -> str:
        """Return or set the label of the Y axis."""
        return self.GetYLabelFormat()

    @y_label_format.setter
    def y_label_format(self, value: str):
        self.SetYLabelFormat(value)
        self._update_y_labels()

    @property
    def z_label_format(self) -> str:
        """Return or set the label of the Z axis."""
        return self.GetZLabelFormat()

    @z_label_format.setter
    def z_label_format(self, value: str):
        self.SetZLabelFormat(value)
        self._update_z_labels()

    @property
    def x_title(self) -> str:
        """Return or set the title of the X axis."""
        return self._x_title

    @x_title.setter
    def x_title(self, value: str):
        self._x_title = value
        self._update_x_labels()

    @property
    def y_title(self) -> str:
        """Return or set the title of the Y axis."""
        return self._y_title

    @y_title.setter
    def y_title(self, value: str):
        self._y_title = value
        self._update_y_labels()

    @property
    def z_title(self) -> str:
        """Return or set the title of the Z axis."""
        return self._z_title

    @z_title.setter
    def z_title(self, value: str):
        self._z_title = value
        self._update_z_labels()

    @property
    def use_2d_mode(self) -> bool:
        """Use the 2d render mode.

        This can be enabled for smoother plotting.
        """
        return bool(self.GetUse2DMode())

    @use_2d_mode.setter
    def use_2d_mode(self, value: bool):
        self.SetUse2DMode(value)

    @property
    def n_xlabels(self):
        """Number of labels on the X axis."""
        return self._n_xlabels

    @n_xlabels.setter
    def n_xlabels(self, value: int):
        self._n_xlabels = value
        self._update_x_labels()

    @property
    def n_ylabels(self):
        """Number of labels on the Y axis."""
        return self._n_ylabels

    @n_ylabels.setter
    def n_ylabels(self, value: int):
        self._n_ylabels = value
        self._update_y_labels()

    @property
    def n_zlabels(self):
        """Number of labels on the Z axis."""
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
        """Regenerate X axis labels."""
        if self.x_axis_visibility:
            self.SetXTitle(self._x_title)
            if self._x_label_visibility:
                self.SetAxisLabels(
                    0, make_axis_labels(*self.bounds[0:2], self.n_xlabels, self.x_label_format)
                )
            else:
                self.SetAxisLabels(0, self._empty_str)
        else:
            self.SetXTitle('')
            self.SetAxisLabels(0, self._empty_str)

    def _update_y_labels(self):
        """Regenerate Y axis labels."""
        if self.y_axis_visibility:
            self.SetYTitle(self._y_title)
            if self._y_label_visibility:
                self.SetAxisLabels(
                    1, make_axis_labels(*self.bounds[2:4], self.n_ylabels, self.y_label_format)
                )
            else:
                self.SetAxisLabels(1, self._empty_str)
        else:
            self.SetYTitle('')
            self.SetAxisLabels(1, self._empty_str)

    def _update_z_labels(self):
        """Regenerate Z axis labels."""
        if self.z_axis_visibility:
            self.SetZTitle(self._z_title)
            if self._z_label_visibility:
                self.SetAxisLabels(
                    2, make_axis_labels(*self.bounds[4:6], self.n_zlabels, self.z_label_format)
                )
            else:
                self.SetAxisLabels(2, self._empty_str)
        else:
            self.SetZTitle('')
            self.SetAxisLabels(2, self._empty_str)

    @property
    def x_labels(self) -> List[str]:
        """Return the x axis labels."""
        return convert_string_array(self.GetAxisLabels(0))

    @property
    def y_labels(self) -> List[str]:
        """Return the y axis labels."""
        return convert_string_array(self.GetAxisLabels(1))

    @property
    def z_labels(self) -> List[str]:
        """Return the z axis labels."""
        return convert_string_array(self.GetAxisLabels(2))

    def update_bounds(self, bounds):
        """Update the bounds of this actor.

        Unlike the :attr:`CubeAxesActor.bounds` attribute, updating the bounds
        also updates the axis labels.

        Parameters
        ----------
        bounds : sequence[float]
            Bounds in the form of ``[xmin, xmax, ymin, ymax, zmin, zmax]``.

        """
        self.bounds = bounds
