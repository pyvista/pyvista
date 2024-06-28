"""Axes actor module."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
from typing import Sequence
from typing import TypedDict
from typing import Union

import pyvista as pv
from pyvista.core.utilities.geometric_sources import AxesGeometrySource
from pyvista.core.utilities.geometric_sources import _AxisEnum
from pyvista.plotting import _vtk
from pyvista.plotting.actor import Actor
from pyvista.plotting.colors import Color

if TYPE_CHECKING:
    from pyvista.core._typing_core import VectorLike
    from pyvista.core.dataset import DataSet
    from pyvista.plotting._typing import ColorLike

    try:
        from typing import Unpack  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        from typing_extensions import Unpack


class _AxesGeometryKwargs(TypedDict):
    shaft_type: AxesGeometrySource.GeometryTypes | DataSet
    shaft_radius: float
    shaft_length: float
    tip_type: AxesGeometrySource.GeometryTypes | DataSet
    tip_radius: float
    tip_length: float | VectorLike[float]
    symmetric: bool


class AxesAssembly(_vtk.vtkPropAssembly):
    def __init__(
        self,
        x_color=None,
        y_color=None,
        z_color=None,
        **kwargs: Unpack[_AxesGeometryKwargs],
    ):
        super().__init__()

        # Add dummy prop3d for calculation transformations
        self._prop3d = Actor()

        # Init shaft and tip actors
        self._shaft_actors = (Actor(), Actor(), Actor())
        self._tip_actors = (Actor(), Actor(), Actor())
        self._shaft_and_tip_actors = (*self._shaft_actors, *self._tip_actors)

        # Init shaft and tip datasets
        self._shaft_and_tip_geometry_source = AxesGeometrySource(**kwargs)
        shaft_tip_datasets = self._shaft_and_tip_geometry_source.output
        for actor, dataset in zip(self._shaft_and_tip_actors, shaft_tip_datasets):
            actor.mapper = pv.DataSetMapper(dataset=dataset)

        # Add actors to assembly
        [self.AddPart(actor) for actor in self._shaft_and_tip_actors]

        # Set colors
        if x_color is None:
            x_color = pv.global_theme.axes.x_color
        if y_color is None:
            y_color = pv.global_theme.axes.y_color
        if z_color is None:
            z_color = pv.global_theme.axes.z_color

        self.x_color = x_color
        self.y_color = y_color
        self.z_color = z_color

        self._update()

    def __repr__(self):
        """Representation of the axes actor."""
        geometry_repr = repr(self._shaft_and_tip_geometry_source).splitlines()[1:]

        attr = [
            f"{type(self).__name__} ({hex(id(self))})",
            *geometry_repr,
        ]
        return '\n'.join(attr)

    def _set_axis_color(self, axis: _AxisEnum, color: ColorLike | tuple[ColorLike, ColorLike]):
        shaft_color, tip_color = _validate_color_sequence(color, n_colors=2)
        self._shaft_actors[axis].prop.color = shaft_color
        self._tip_actors[axis].prop.color = tip_color

    def _get_axis_color(self, axis: _AxisEnum) -> tuple[Color, Color]:
        return (
            self._shaft_actors[axis].prop.color,
            self._tip_actors[axis].prop.color,
        )

    @property
    def x_color(self) -> tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the x-axis shaft and tip."""
        return self._get_axis_color(_AxisEnum.x)

    @x_color.setter
    def x_color(self, color: Union[ColorLike, Sequence[ColorLike]]):  # numpydoc ignore=GL08
        shaft_color, tip_color = _validate_color_sequence(color, n_colors=2)
        self._shaft_actors[_AxisEnum.x].prop.color = shaft_color
        self._tip_actors[_AxisEnum.x].prop.color = tip_color

    @property
    def y_color(self) -> tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the y-axis shaft and tip."""
        return self._get_axis_color(_AxisEnum.y)

    @y_color.setter
    def y_color(self, color: Union[ColorLike, Sequence[ColorLike]]):  # numpydoc ignore=GL08
        shaft_color, tip_color = _validate_color_sequence(color, n_colors=2)
        self._shaft_actors[_AxisEnum.y].prop.color = shaft_color
        self._tip_actors[_AxisEnum.y].prop.color = tip_color

    @property
    def z_color(self) -> tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the z-axis shaft and tip."""
        return self._get_axis_color(_AxisEnum.z)

    @z_color.setter
    def z_color(self, color: Union[ColorLike, Sequence[ColorLike]]):  # numpydoc ignore=GL08
        shaft_color, tip_color = _validate_color_sequence(color, n_colors=2)
        self._shaft_actors[_AxisEnum.z].prop.color = shaft_color
        self._tip_actors[_AxisEnum.z].prop.color = tip_color

    def _update(self):
        self._shaft_and_tip_geometry_source.update()


def _validate_color_sequence(
    color: ColorLike | Sequence[ColorLike],
    n_colors: int | None = None,
) -> tuple[Color, ...]:
    valid_color = None
    try:
        valid_color = [Color(color)]
        n_colors = 1 if n_colors is None else n_colors
        valid_color *= n_colors
    except ValueError:
        if isinstance(color, Sequence) and (n_colors is None or len(color) == n_colors):
            with contextlib.suppress(ValueError):
                valid_color = [Color(c) for c in color]
    if valid_color:
        return tuple(valid_color)
    else:
        raise ValueError(
            f"Invalid color(s):\n"
            f"\t{color}\n"
            f"Input must be a single ColorLike color \n"
            f"or a sequence of {n_colors} ColorLike colors.",
        )
