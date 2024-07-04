"""Axes assembly module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import TypedDict

import numpy as np

import pyvista as pv
from pyvista.core import _validation
from pyvista.core.utilities.geometric_sources import AxesGeometrySource
from pyvista.core.utilities.geometric_sources import _AxisEnum
from pyvista.core.utilities.geometric_sources import _PartEnum
from pyvista.core.utilities.transformations import apply_transformation_to_points
from pyvista.plotting import _vtk
from pyvista.plotting.actor import Actor
from pyvista.plotting.colors import Color
from pyvista.plotting.text import Label

if TYPE_CHECKING:  # pragma: no cover
    import sys
    from typing import Sequence

    from pyvista.core._typing_core import VectorLike
    from pyvista.core.dataset import DataSet
    from pyvista.plotting._property import Property
    from pyvista.plotting._typing import ColorLike

    if sys.version_info >= (3, 11):
        from typing import Unpack
    else:
        from typing_extensions import Unpack


class _AxesPropTuple(NamedTuple):
    x_shaft: Property
    y_shaft: Property
    z_shaft: Property
    x_tip: Property
    y_tip: Property
    z_tip: Property


class _AxesGeometryKwargs(TypedDict):
    shaft_type: AxesGeometrySource.GeometryTypes | DataSet
    shaft_radius: float
    shaft_length: float
    tip_type: AxesGeometrySource.GeometryTypes | DataSet
    tip_radius: float
    tip_length: float | VectorLike[float]
    symmetric: bool


class AxesAssembly(_vtk.vtkPropAssembly):
    """Assembly of arrow-style axes parts.

    The axes may be used as a widget or added to a scene.

    Parameters
    ----------
    x_label : str, default: 'X'
        Text label for the x-axis. Alternatively, set the label with :attr:`labels`.

    y_label : str, default: 'Y'
        Text label for the y-axis. Alternatively, set the label with :attr:`labels`.

    z_label : str, default: 'Z'
        Text label for the z-axis. Alternatively, set the label with :attr:`labels`.

    labels : Sequence[str], optional,
        Text labels for the axes. This is an alternative parameter to using
        :attr:`x_label`, :attr:`y_label`, and :attr:`z_label` separately.

    label_color : ColorLike, default: 'black'
        Color of the text labels.

    show_labels : bool, default: True
        Show or hide the text labels.

    label_position : float | VectorLike[float], optional
        Position of the text labels along each axis. By default, the labels are
        positioned at the ends of the shafts.

    label_size : int, default: 50
        Size of the text labels.

    x_color : ColorLike | Sequence[ColorLike], optional
        Color of the x-axis shaft and tip.

    y_color : ColorLike | Sequence[ColorLike], optional
        Color of the y-axis shaft and tip.

    z_color : ColorLike | Sequence[ColorLike], optional
        Color of the z-axis shaft and tip.

    **kwargs
        Keyword arguments passed to :class:`pyvista.AxesGeometrySource`.

    Examples
    --------
    Add axes to a plot.

    >>> import pyvista as pv
    >>> axes = pv.AxesAssembly()
    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(axes)
    >>> pl.show()

    Customize the axes colors. Set each axis to a single color, or set the colors of
    each shaft and tip separately with two colors.

    >>> axes.x_color = ['cyan', 'blue']
    >>> axes.y_color = ['magenta', 'red']
    >>> axes.z_color = 'yellow'

    Customize the label color too.

    >>> axes.label_color = 'brown'

    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(axes)
    >>> pl.show()
    """

    def __init__(
        self,
        *,
        x_label: str | None = None,
        y_label: str | None = None,
        z_label: str | None = None,
        labels: Sequence[str] | None = None,
        label_color: ColorLike = 'black',
        show_labels: bool = True,
        label_position: float | VectorLike[float] | None = None,
        label_size: int = 50,
        x_color: ColorLike | Sequence[ColorLike] | None = None,
        y_color: ColorLike | Sequence[ColorLike] | None = None,
        z_color: ColorLike | Sequence[ColorLike] | None = None,
        **kwargs: Unpack[_AxesGeometryKwargs],
    ):
        super().__init__()

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

        # Init label actors and add to assembly
        self._label_actors = (Label(), Label(), Label())
        [self.AddPart(actor) for actor in self._label_actors]

        # Set colors
        if x_color is None:
            x_color = pv.global_theme.axes.x_color
        if y_color is None:
            y_color = pv.global_theme.axes.y_color
        if z_color is None:
            z_color = pv.global_theme.axes.z_color

        self.x_color = x_color  # type: ignore[assignment]
        self.y_color = y_color  # type: ignore[assignment]
        self.z_color = z_color  # type: ignore[assignment]

        # Set text labels
        if labels is None:
            self.x_label = 'X' if x_label is None else x_label
            self.y_label = 'Y' if y_label is None else y_label
            self.z_label = 'Z' if z_label is None else z_label
        else:
            msg = "Cannot initialize '{}' and 'labels' properties together. Specify one or the other, not both."
            if x_label is not None:
                raise ValueError(msg.format('x_label'))
            if y_label is not None:
                raise ValueError(msg.format('y_label'))
            if z_label is not None:
                raise ValueError(msg.format('z_label'))
            self.labels = labels  # type: ignore[assignment]
        self.show_labels = show_labels
        self.label_color = label_color  # type: ignore[assignment]
        self.label_size = label_size
        self.label_position = label_position  # type: ignore[assignment]

        # Set default text properties
        for actor in self._label_actors:
            prop = actor.prop
            prop.bold = True
            prop.italic = True

        self._update()

    def __repr__(self):
        """Representation of the axes actor."""
        geometry_repr = repr(self._shaft_and_tip_geometry_source).splitlines()[1:]

        attr = [
            f"{type(self).__name__} ({hex(id(self))})",
            *geometry_repr,
            f"  X label:                    '{self.x_label}'",
            f"  Y label:                    '{self.y_label}'",
            f"  Z label:                    '{self.z_label}'",
            f"  Label color:                {self.label_color}",
            f"  Show labels:                {self.show_labels}",
            f"  Label position:             {self.label_position}",
            "  X Color:                                     ",
            f"      Shaft                   {self.x_color[0]}",
            f"      Tip                     {self.x_color[1]}",
            "  Y Color:                                     ",
            f"      Shaft                   {self.y_color[0]}",
            f"      Tip                     {self.y_color[1]}",
            "  Z Color:                                     ",
            f"      Shaft                   {self.z_color[0]}",
            f"      Tip                     {self.z_color[1]}",
        ]
        return '\n'.join(attr)

    @property
    def labels(self) -> tuple[str, str, str]:  # numpydoc ignore=RT01
        """Return or set the axes labels.

        This property may be used as an alternative to using :attr:`x_label`,
        :attr:`y_label`, and :attr:`z_label` separately.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesActor()
        >>> axes_actor.labels = ['X Axis', 'Y Axis', 'Z Axis']
        >>> axes_actor.labels
        ('X Axis', 'Y Axis', 'Z Axis')
        """
        return self.x_label, self.y_label, self.z_label

    @labels.setter
    def labels(self, labels: list[str] | tuple[str, str, str]):  # numpydoc ignore=GL08
        _validation.check_instance(labels, (list, tuple))
        _validation.check_iterable_items(labels, str, name='labels')
        _validation.check_length(labels, exact_length=3, name='labels')
        self.x_label = labels[0]
        self.y_label = labels[1]
        self.z_label = labels[2]

    @property
    def x_label(self) -> str:  # numpydoc ignore=RT01
        """Text label for the x-axis.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesAssembly()
        >>> axes_actor.x_label = 'This axis'
        >>> axes_actor.x_label
        'This axis'

        """
        return self._label_actors[0].input

    @x_label.setter
    def x_label(self, label: str):  # numpydoc ignore=GL08
        self._label_actors[0].input = label

    @property
    def y_label(self) -> str:  # numpydoc ignore=RT01
        """Text label for the y-axis.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesAssembly()
        >>> axes_actor.y_label = 'This axis'
        >>> axes_actor.y_label
        'This axis'

        """
        return self._label_actors[1].input

    @y_label.setter
    def y_label(self, label: str):  # numpydoc ignore=GL08
        self._label_actors[1].input = label

    @property
    def z_label(self) -> str:  # numpydoc ignore=RT01
        """Text label for the z-axis.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesAssembly()
        >>> axes_actor.z_label = 'This axis'
        >>> axes_actor.z_label
        'This axis'

        """
        return self._label_actors[2].input

    @z_label.setter
    def z_label(self, label: str):  # numpydoc ignore=GL08
        self._label_actors[2].input = label

    @property
    def show_labels(self) -> bool:  # numpydoc ignore=RT01
        """Show or hide the text labels for the axes."""
        return self._show_labels

    @show_labels.setter
    def show_labels(self, value: bool):  # numpydoc ignore=GL08
        self._show_labels = value
        self._label_actors[0].SetVisibility(value)
        self._label_actors[1].SetVisibility(value)
        self._label_actors[2].SetVisibility(value)

    @property
    def label_size(self) -> int:  # numpydoc ignore=RT01
        """Size of the text labels.

        Must be a positive integer.
        """
        return self._label_size

    @label_size.setter
    def label_size(self, size: int):  # numpydoc ignore=GL08
        self._label_size = size
        self._label_actors[0].size = size
        self._label_actors[1].size = size
        self._label_actors[2].size = size

    @property
    def label_position(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Position of the text label along each axis.

        By default, the labels are positioned at the ends of the shafts.

        Values must be non-negative.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesAssembly()
        >>> axes_actor.label_position
        (0.8, 0.8, 0.8)
        >>> axes_actor.label_position = 0.3
        >>> axes_actor.label_position
        (0.3, 0.3, 0.3)
        >>> axes_actor.label_position = (0.1, 0.4, 0.2)
        >>> axes_actor.label_position
        (0.1, 0.4, 0.2)

        """
        position = self._label_position
        return self._shaft_and_tip_geometry_source.shaft_length if position is None else position

    @label_position.setter
    def label_position(self, position: float | VectorLike[float] | None):  # numpydoc ignore=GL08
        self._label_position = (
            None
            if position is None
            else _validation.validate_array3(
                position,
                broadcast=True,
                must_be_in_range=[0, np.inf],
                name='Label position',
                dtype_out=float,
                to_tuple=True,
            )
        )
        self._apply_transformation_to_labels()

    @property
    def label_color(self) -> Color:  # numpydoc ignore=RT01
        """Color of the text labels."""
        return self._label_color

    @label_color.setter
    def label_color(self, color: ColorLike):  # numpydoc ignore=GL08
        valid_color = Color(color)
        self._label_color = valid_color
        self._label_actors[0].prop.color = valid_color
        self._label_actors[1].prop.color = valid_color
        self._label_actors[2].prop.color = valid_color

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
    def x_color(self, color: ColorLike | Sequence[ColorLike]):  # numpydoc ignore=GL08
        self._set_axis_color(_AxisEnum.x, color)

    @property
    def y_color(self) -> tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the y-axis shaft and tip."""
        return self._get_axis_color(_AxisEnum.y)

    @y_color.setter
    def y_color(self, color: ColorLike | Sequence[ColorLike]):  # numpydoc ignore=GL08
        self._set_axis_color(_AxisEnum.y, color)

    @property
    def z_color(self) -> tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the z-axis shaft and tip."""
        return self._get_axis_color(_AxisEnum.z)

    @z_color.setter
    def z_color(self, color: ColorLike | Sequence[ColorLike]):  # numpydoc ignore=GL08
        self._set_axis_color(_AxisEnum.z, color)

    def set_part_prop(
        self,
        name: str,
        value: Any,
        axis: Literal['x', 'y', 'z', 0, 1, 2, 'all'] = 'all',
        part: Literal['shaft', 'tip', 0, 1, 'all'] = 'all',
    ):
        """Set :class:`~pyvista.Property` attributes for the axes shafts and/or tips.

        This is a generalized setter method which sets the value of a specific
        :class:`~pyvista.Property` attribute for any combination of axis shaft or tip
        parts.

        Parameters
        ----------
        name : str
            Name of the :class:`~pyvista.Property` attribute to set.

        value : Any
            Value to set the attribute to.

        axis : str | int, default: 'all'
            Set :class:`~pyvista.Property` attributes for a specific part of the axes.
            Specify one of:

            - ``'x'`` or ``0``: only set the property for the x-axis.
            - ``'y'`` or ``1``: only set the property for the y-axis.
            - ``'z'`` or ``2``: only set the property for the z-axis.
            - ``'all'``: set the property for all three axes.

        part : str | int, default: 'all'
            Set the property for a specific part of the axes. Specify one of:

            - ``'shaft'`` or ``0``: only set the property for the axes shafts.
            - ``'tip'`` or ``1``: only set the property for the axes tips.
            - ``'all'``: set the property for axes shafts and tips.

        Examples
        --------
        Set the ambient property for all axes shafts and tips.

        >>> import pyvista as pv
        >>> axes_actor = pv.AxesAssembly()
        >>> axes_actor.set_part_prop('ambient', 0.7)
        >>> axes_actor.get_part_prop('ambient')
        _AxesPropTuple(x_shaft=0.7, y_shaft=0.7, z_shaft=0.7, x_tip=0.7, y_tip=0.7, z_tip=0.7)

        Set a property for the x-axis only. The property is set for
        both the axis shaft and tip by default.

        >>> axes_actor.set_part_prop('ambient', 0.3, axis='x')
        >>> axes_actor.get_part_prop('ambient')
        _AxesPropTuple(x_shaft=0.3, y_shaft=0.7, z_shaft=0.7, x_tip=0.3, y_tip=0.7, z_tip=0.7)

        Set a property for the axes tips only. The property is set for
        all axes by default.

        >>> axes_actor.set_part_prop('ambient', 0.1, part='tip')
        >>> axes_actor.get_part_prop('ambient')
        _AxesPropTuple(x_shaft=0.3, y_shaft=0.7, z_shaft=0.7, x_tip=0.1, y_tip=0.1, z_tip=0.1)

        Set a property for a single axis and specific part.

        >>> axes_actor.set_part_prop(
        ...     'ambient', 0.9, axis='z', part='shaft'
        ... )
        >>> axes_actor.get_part_prop('ambient')
        _AxesPropTuple(x_shaft=0.3, y_shaft=0.7, z_shaft=0.9, x_tip=0.1, y_tip=0.1, z_tip=0.1)
        """
        actors = self._filter_part_actors(axis=axis, part=part)
        for actor in actors:
            setattr(actor.prop, name, value)

    def get_part_prop(self, name: str):
        """Get :class:`~pyvista.Property` attributes for the axes shafts and/or tips.

        This is a generalized getter method which returns the value of
        a specific :class:`pyvista.Property`attribute for all shafts and tips;

        Parameters
        ----------
        name : str
            Name of the :class:`~pyvista.Property` attribute to get.

        Returns
        -------
        tuple
            Named tuple with the requested attribute value for the axes shafts and tips.
            The values are ordered ``(x_shaft, y_shaft, z_shaft, x_tip, y_tip, z_tip)``.

        Examples
        --------
        Get the ambient property of the axes shafts and tips.

        >>> import pyvista as pv
        >>> axes_assembly = pv.AxesAssembly()
        >>> axes_assembly.get_part_prop('ambient')
        _AxesPropTuple(x_shaft=0.0, y_shaft=0.0, z_shaft=0.0, x_tip=0.0, y_tip=0.0, z_tip=0.0)

        """
        prop_values = [getattr(actor.prop, name) for actor in self._shaft_and_tip_actors]
        return _AxesPropTuple(*prop_values)

    def _filter_part_actors(
        self,
        axis: Literal['x', 'y', 'z', 0, 1, 2, 'all'] = 'all',
        part: Literal['shaft', 'tip', 0, 1, 'all'] = 'all',
    ):
        valid_axis = [0, 1, 2, 'x', 'y', 'z', 'all']
        if axis not in valid_axis:
            raise ValueError(f"Axis must be one of {valid_axis}.")
        valid_part = [0, 1, 'shaft', 'tip', 'all']
        if part not in valid_part:
            raise ValueError(f"Part must be one of {valid_part}.")

        actors: list[Actor] = []
        for axis_num, axis_name in enumerate(['x', 'y', 'z']):
            if axis in [axis_num, axis_name, 'all']:
                if part in [_PartEnum.shaft, _PartEnum.shaft.name, 'all']:
                    actors.append(self._shaft_actors[axis_num])
                if part in [_PartEnum.tip, _PartEnum.tip.name, 'all']:
                    actors.append(self._tip_actors[axis_num])
        return actors

    def _get_transformed_label_positions(self):
        # Create position vectors
        position_vectors = np.diag(self.label_position)

        # Offset label positions radially by the tip radius
        tip_radius = self._shaft_and_tip_geometry_source.tip_radius
        offset_array = np.diag([tip_radius] * 3)
        radial_offset1 = np.roll(offset_array, shift=1, axis=1)
        radial_offset2 = np.roll(offset_array, shift=-1, axis=1)

        position_vectors += radial_offset1 + radial_offset2

        # Transform positions
        matrix = np.eye(4)  # TODO: use Prop3D transformation
        return apply_transformation_to_points(matrix, position_vectors)

    def _apply_transformation_to_labels(self):
        x_pos, y_pos, z_pos = self._get_transformed_label_positions()
        self._label_actors[0].position = x_pos
        self._label_actors[1].position = y_pos
        self._label_actors[2].position = z_pos

    def _update(self):
        self._shaft_and_tip_geometry_source.update()
        self._apply_transformation_to_labels()


def _validate_color_sequence(
    color: ColorLike | Sequence[ColorLike],
    n_colors: int | None = None,
) -> tuple[Color, ...]:
    """Validate a color sequence.

    If `n_colors` is specified, the output will have `n` colors. For single-color
    inputs, the color is copied and a sequence of `n` identical colors is returned.
    For inputs with multiple colors, the number of colors in the input must
    match `n_colors`

    If `n_colors` is None, no broadcasting or length-checking is performed.
    """
    try:
        # Assume we have one color
        color_list = [Color(color)]
        n_colors = 1 if n_colors is None else n_colors
        return tuple(color_list * n_colors)
    except ValueError:
        if isinstance(color, (tuple, list)):
            try:
                color_list = [_validate_color_sequence(c, n_colors=1)[0] for c in color]
                if len(color_list) == 1:
                    n_colors = 1 if n_colors is None else n_colors
                    color_list = color_list * n_colors

                # Only return if we have the correct number of colors
                if n_colors is not None and len(color_list) == n_colors:
                    return tuple(color_list)
            except ValueError:
                pass
    raise ValueError(
        f"Invalid color(s):\n"
        f"\t{color}\n"
        f"Input must be a single ColorLike color "
        f"or a sequence of {n_colors} ColorLike colors.",
    )
