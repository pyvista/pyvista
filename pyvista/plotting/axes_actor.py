"""Axes actor module."""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import TYPE_CHECKING
from typing import Sequence

import pyvista

from . import _vtk
from ._property import _check_range
from .actor_properties import ActorProperties
from .colors import Color
from .colors import _validate_color_sequence

if TYPE_CHECKING:
    from .colors import ColorLike


class AxesActor(_vtk.vtkAxesActor):
    """Axes actor wrapper for vtkAxesActor.

    Hybrid 2D/3D actor used to represent 3D axes in a scene. The user
    can define the geometry to use for the shaft or the tip, and the
    user can set the text for the three axes. To see full customization
    options, refer to `vtkAxesActor Details
    <https://vtk.org/doc/nightly/html/classvtkAxesActor.html#details>`_.

    Examples
    --------
    Customize the axis shaft color and shape.

    >>> import pyvista as pv

    >>> axes = pv.Axes()
    >>> axes.axes_actor.z_axis_shaft_properties.color = (0.0, 1.0, 1.0)
    >>> axes.axes_actor.shaft_type = axes.axes_actor.ShaftType.CYLINDER
    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(axes.axes_actor)
    >>> _ = pl.add_mesh(pv.Sphere())
    >>> pl.show()

    Or you can use this as a custom orientation widget with
    :func:`add_orientation_widget() <pyvista.Renderer.add_orientation_widget>`:

    >>> import pyvista as pv

    >>> axes = pv.Axes()
    >>> axes_actor = axes.axes_actor
    >>> axes.axes_actor.shaft_type = 0

    >>> axes_actor.x_axis_shaft_properties.color = (1.0, 1.0, 1.0)
    >>> axes_actor.y_axis_shaft_properties.color = (1.0, 1.0, 1.0)
    >>> axes_actor.z_axis_shaft_properties.color = (1.0, 1.0, 1.0)

    >>> axes_actor.x_axis_label = 'U'
    >>> axes_actor.y_axis_label = 'V'
    >>> axes_actor.z_axis_label = 'W'

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(pv.Cone())
    >>> _ = pl.add_orientation_widget(
    ...     axes_actor,
    ...     viewport=(0, 0, 0.5, 0.5),
    ... )
    >>> pl.show()

    """

    class ShaftType(Enum):
        """Types of shaft shapes available."""

        CYLINDER = 0
        LINE = 1

    class TipType(Enum):
        """Types of tip shapes available."""

        CONE = 0
        SPHERE = 1

    def __init__(self):
        """Initialize actor."""
        super().__init__()

        self.x_axis_shaft_properties.color = pyvista.global_theme.axes.x_color.float_rgb
        self.x_axis_tip_properties.color = pyvista.global_theme.axes.x_color.float_rgb
        self.x_axis_shaft_properties.opacity = pyvista.global_theme.axes.x_color.float_rgba[3]
        self.x_axis_tip_properties.opacity = pyvista.global_theme.axes.x_color.float_rgba[3]
        self.x_axis_shaft_properties.lighting = pyvista.global_theme.lighting

        self.y_axis_shaft_properties.color = pyvista.global_theme.axes.y_color.float_rgb
        self.y_axis_tip_properties.color = pyvista.global_theme.axes.y_color.float_rgb
        self.y_axis_shaft_properties.opacity = pyvista.global_theme.axes.y_color.float_rgba[3]
        self.y_axis_tip_properties.opacity = pyvista.global_theme.axes.y_color.float_rgba[3]
        self.y_axis_shaft_properties.lighting = pyvista.global_theme.lighting

        self.z_axis_shaft_properties.color = pyvista.global_theme.axes.z_color.float_rgb
        self.z_axis_tip_properties.color = pyvista.global_theme.axes.z_color.float_rgb
        self.z_axis_shaft_properties.opacity = pyvista.global_theme.axes.z_color.float_rgba[3]
        self.z_axis_tip_properties.opacity = pyvista.global_theme.axes.z_color.float_rgba[3]
        self.z_axis_shaft_properties.lighting = pyvista.global_theme.lighting

    @property
    def _actors(
        self,
    ) -> tuple[
        _vtk.vtkActor,
        _vtk.vtkActor,
        _vtk.vtkActor,
        _vtk.vtkActor,
        _vtk.vtkActor,
        _vtk.vtkActor,
    ]:
        collection = _vtk.vtkPropCollection()
        self.GetActors(collection)
        return tuple([collection.GetItemAsObject(i) for i in range(6)])

    @property
    def _label_actors(
        self,
    ) -> tuple[_vtk.vtkCaptionActor2D, _vtk.vtkCaptionActor2D, _vtk.vtkCaptionActor2D]:
        return (
            self.GetXAxisCaptionActor2D(),
            self.GetYAxisCaptionActor2D(),
            self.GetZAxisCaptionActor2D(),
        )

    @property
    def visibility(self) -> bool:  # numpydoc ignore=RT01
        """Return or set AxesActor visibility.

        Examples
        --------
        Create an Axes object and then access the
        visibility attribute of its AxesActor.

        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.visibility
        True

        """
        return bool(self.GetVisibility())

    @visibility.setter
    def visibility(self, value: bool):  # numpydoc ignore=GL08
        self.SetVisibility(value)

    @property
    def total_length(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the length of all axes.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.total_length
        (1.0, 1.0, 1.0)
        >>> axes.axes_actor.total_length = 1.2
        >>> axes.axes_actor.total_length
        (1.2, 1.2, 1.2)
        >>> axes.axes_actor.total_length = (1.0, 0.9, 0.5)
        >>> axes.axes_actor.total_length
        (1.0, 0.9, 0.5)

        """
        return self.GetTotalLength()

    @total_length.setter
    def total_length(self, length):  # numpydoc ignore=GL08
        if isinstance(length, Iterable):
            self.SetTotalLength(length[0], length[1], length[2])
        else:
            self.SetTotalLength(length, length, length)

    @property
    def shaft_length(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the length of the axes shaft.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.shaft_length
        (0.8, 0.8, 0.8)
        >>> axes.axes_actor.shaft_length = 0.7
        >>> axes.axes_actor.shaft_length
        (0.7, 0.7, 0.7)
        >>> axes.axes_actor.shaft_length = (1.0, 0.9, 0.5)
        >>> axes.axes_actor.shaft_length
        (1.0, 0.9, 0.5)

        """
        return self.GetNormalizedShaftLength()

    @shaft_length.setter
    def shaft_length(self, length):  # numpydoc ignore=GL08
        if isinstance(length, Iterable):
            self.SetNormalizedShaftLength(length[0], length[1], length[2])
        else:
            self.SetNormalizedShaftLength(length, length, length)

    @property
    def tip_length(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the length of the tip.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.tip_length
        (0.2, 0.2, 0.2)
        >>> axes.axes_actor.tip_length = 0.3
        >>> axes.axes_actor.tip_length
        (0.3, 0.3, 0.3)
        >>> axes.axes_actor.tip_length = (0.1, 0.4, 0.2)
        >>> axes.axes_actor.tip_length
        (0.1, 0.4, 0.2)

        """
        return self.GetNormalizedTipLength()

    @tip_length.setter
    def tip_length(self, length):  # numpydoc ignore=GL08
        if isinstance(length, Iterable):
            self.SetNormalizedTipLength(length[0], length[1], length[2])
        else:
            self.SetNormalizedTipLength(length, length, length)

    @property
    def label_position(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Position of the label along the axes.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.label_position
        (1.0, 1.0, 1.0)
        >>> axes.axes_actor.label_position = 0.3
        >>> axes.axes_actor.label_position
        (0.3, 0.3, 0.3)
        >>> axes.axes_actor.label_position = (0.1, 0.4, 0.2)
        >>> axes.axes_actor.label_position
        (0.1, 0.4, 0.2)

        """
        return self.GetNormalizedLabelPosition()

    @label_position.setter
    def label_position(self, length):  # numpydoc ignore=GL08
        if isinstance(length, Iterable):
            self.SetNormalizedLabelPosition(length[0], length[1], length[2])
        else:
            self.SetNormalizedLabelPosition(length, length, length)

    @property
    def cone_resolution(self) -> int:  # numpydoc ignore=RT01
        """Return or set the resolution of the cone tip.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.cone_resolution
        16
        >>> axes.axes_actor.cone_resolution = 24
        >>> axes.axes_actor.cone_resolution
        24

        """
        return self.GetConeResolution()

    @cone_resolution.setter
    def cone_resolution(self, res: int):  # numpydoc ignore=GL08
        self.SetConeResolution(res)

    @property
    def sphere_resolution(self) -> int:  # numpydoc ignore=RT01
        """Return or set the resolution of the spherical tip.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.sphere_resolution
        16
        >>> axes.axes_actor.sphere_resolution = 24
        >>> axes.axes_actor.sphere_resolution
        24

        """
        return self.GetSphereResolution()

    @sphere_resolution.setter
    def sphere_resolution(self, res: int):  # numpydoc ignore=GL08
        self.SetSphereResolution(res)

    @property
    def cylinder_resolution(self) -> int:  # numpydoc ignore=RT01
        """Return or set the resolution of the shaft cylinder.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.cylinder_resolution
        16
        >>> axes.axes_actor.cylinder_resolution = 24
        >>> axes.axes_actor.cylinder_resolution
        24

        """
        return self.GetCylinderResolution()

    @cylinder_resolution.setter
    def cylinder_resolution(self, res: int):  # numpydoc ignore=GL08
        self.SetCylinderResolution(res)

    @property
    def cone_radius(self) -> float:  # numpydoc ignore=RT01
        """Return or set the radius of the cone tip.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.cone_radius
        0.4
        >>> axes.axes_actor.cone_radius = 0.8
        >>> axes.axes_actor.cone_radius
        0.8

        """
        return self.GetConeRadius()

    @cone_radius.setter
    def cone_radius(self, rad: float):  # numpydoc ignore=GL08
        self.SetConeRadius(rad)

    @property
    def sphere_radius(self) -> float:  # numpydoc ignore=RT01
        """Return or set the radius of the spherical tip.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.sphere_radius
        0.4
        >>> axes.axes_actor.sphere_radius = 0.8
        >>> axes.axes_actor.sphere_radius
        0.8

        """
        return self.GetSphereRadius()

    @sphere_radius.setter
    def sphere_radius(self, rad: float):  # numpydoc ignore=GL08
        self.SetSphereRadius(rad)

    @property
    def cylinder_radius(self) -> float:  # numpydoc ignore=RT01
        """Return or set the radius of the shaft cylinder.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.cylinder_radius
        0.05
        >>> axes.axes_actor.cylinder_radius = 0.03
        >>> axes.axes_actor.cylinder_radius
        0.03

        """
        return self.GetCylinderRadius()

    @cylinder_radius.setter
    def cylinder_radius(self, rad: float):  # numpydoc ignore=GL08
        self.SetCylinderRadius(rad)

    @property
    def shaft_type(self) -> ShaftType:  # numpydoc ignore=RT01
        """Return or set the shaft type.

        Can be either a cylinder(0) or a line(1).

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.shaft_type = axes.axes_actor.ShaftType.LINE
        >>> axes.axes_actor.shaft_type
        <ShaftType.LINE: 1>

        """
        return AxesActor.ShaftType(self.GetShaftType())

    @shaft_type.setter
    def shaft_type(self, shaft_type: ShaftType | int):  # numpydoc ignore=GL08
        shaft_type = AxesActor.ShaftType(shaft_type)
        if shaft_type == AxesActor.ShaftType.CYLINDER:
            self.SetShaftTypeToCylinder()
        elif shaft_type == AxesActor.ShaftType.LINE:
            self.SetShaftTypeToLine()

    @property
    def tip_type(self) -> TipType:  # numpydoc ignore=RT01
        """Return or set the shaft type.

        Can be either a cone(0) or a sphere(1).

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.tip_type = axes.axes_actor.TipType.SPHERE
        >>> axes.axes_actor.tip_type
        <TipType.SPHERE: 1>

        """
        return AxesActor.TipType(self.GetTipType())

    @tip_type.setter
    def tip_type(self, tip_type: TipType | int):  # numpydoc ignore=GL08
        tip_type = AxesActor.TipType(tip_type)
        if tip_type == AxesActor.TipType.CONE:
            self.SetTipTypeToCone()
        elif tip_type == AxesActor.TipType.SPHERE:
            self.SetTipTypeToSphere()

    @property
    def labels(self) -> tuple[str, str, str]:  # numpydoc ignore=RT01
        """Return or set the axes labels.

        This property may be used as an alternative to using :attr:`~x_axis_label`,
        :attr:`~y_axis_label`, and :attr:`~z_axis_label` separately.

        .. versionadded:: 0.44.0

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesActor()
        >>> axes_actor.labels = ['X Axis', 'Y Axis', 'Z Axis']
        >>> axes_actor.labels
        ('X Axis', 'Y Axis', 'Z Axis')
        """
        return self.x_axis_label, self.y_axis_label, self.z_axis_label

    @labels.setter
    def labels(self, labels: list[str] | tuple[str]):  # numpydoc ignore=GL08
        if not (isinstance(labels, (list, tuple)) and len(labels) == 3):
            raise ValueError(
                f'Labels must be a list or tuple with three items. Got {labels} instead.',
            )
        self.x_axis_label = labels[0]
        self.y_axis_label = labels[1]
        self.z_axis_label = labels[2]

    @property
    def x_axis_label(self) -> str:  # numpydoc ignore=RT01
        """Return or set the label for the x-axis.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.x_axis_label = 'This axis'
        >>> axes.axes_actor.x_axis_label
        'This axis'

        """
        return self.GetXAxisLabelText()

    @x_axis_label.setter
    def x_axis_label(self, label: str):  # numpydoc ignore=GL08
        self.SetXAxisLabelText(label)

    @property
    def y_axis_label(self) -> str:  # numpydoc ignore=RT01
        """Return or set the label for the y-axis.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.y_axis_label = 'This axis'
        >>> axes.axes_actor.y_axis_label
        'This axis'

        """
        return self.GetYAxisLabelText()

    @y_axis_label.setter
    def y_axis_label(self, label: str):  # numpydoc ignore=GL08
        self.SetYAxisLabelText(label)

    @property
    def z_axis_label(self) -> str:  # numpydoc ignore=RT01
        """Return or set the label for the z-axis.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.z_axis_label = 'This axis'
        >>> axes.axes_actor.z_axis_label
        'This axis'

        """
        return self.GetZAxisLabelText()

    @z_axis_label.setter
    def z_axis_label(self, label: str):  # numpydoc ignore=GL08
        self.SetZAxisLabelText(label)

    @property
    def x_axis_shaft_properties(self):  # numpydoc ignore=RT01
        """Return or set the properties of the x-axis shaft."""
        return ActorProperties(self.GetXAxisShaftProperty())

    @property
    def y_axis_shaft_properties(self):  # numpydoc ignore=RT01
        """Return or set the properties of the y-axis shaft."""
        return ActorProperties(self.GetYAxisShaftProperty())

    @property
    def z_axis_shaft_properties(self):  # numpydoc ignore=RT01
        """Return or set the properties of the z-axis shaft."""
        return ActorProperties(self.GetZAxisShaftProperty())

    @property
    def x_axis_tip_properties(self):  # numpydoc ignore=RT01
        """Return or set the properties of the x-axis tip."""
        return ActorProperties(self.GetXAxisTipProperty())

    @x_axis_tip_properties.setter
    def x_axis_tip_properties(self, properties: ActorProperties):  # numpydoc ignore=GL08
        self.x_axis_tip_properties = properties

    @property
    def y_axis_tip_properties(self):  # numpydoc ignore=RT01
        """Return or set the properties of the y-axis tip."""
        return ActorProperties(self.GetYAxisTipProperty())

    @y_axis_tip_properties.setter
    def y_axis_tip_properties(self, properties: ActorProperties):  # numpydoc ignore=GL08
        self.y_axis_tip_properties = properties

    @property
    def z_axis_tip_properties(self):  # numpydoc ignore=RT01
        """Return or set the properties of the z-axis tip."""
        return ActorProperties(self.GetZAxisTipProperty())

    @z_axis_tip_properties.setter
    def z_axis_tip_properties(self, properties: ActorProperties):  # numpydoc ignore=GL08
        self.z_axis_tip_properties = properties

    @property
    def x_label_color(self) -> Color:
        return Color(self.GetXAxisCaptionActor2D().GetCaptionTextProperty().GetColor())

    @x_label_color.setter
    def x_label_color(self, val: Color):
        color = Color(val).float_rgb
        self.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(color)

    @property
    def y_label_color(self) -> Color:
        return Color(self.GetYAxisCaptionActor2D().GetCaptionTextProperty().GetColor())

    @y_label_color.setter
    def y_label_color(self, val: Color):
        color = Color(val).float_rgb
        self.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(color)

    @property
    def z_label_color(self) -> Color:
        return Color(self.GetZAxisCaptionActor2D().GetCaptionTextProperty().GetColor())

    @z_label_color.setter
    def z_label_color(self, val: Color):
        color = Color(val).float_rgb
        self.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(color)

    @property
    def label_color(self) -> tuple[Color, Color, Color]:  # numpydoc ignore=RT01
        """Color of the label text for all axes."""

        return (self.x_label_color, self.y_label_color, self.z_label_color)

    @label_color.setter
    def label_color(self, color: ColorLike | Sequence[ColorLike]):  # numpydoc ignore=GL08
        if color is None:
            color = pyvista.global_theme.font.color
        colors = _validate_color_sequence(color, n_colors=3)
        self.x_label_color = colors[0]
        self.y_label_color = colors[1]
        self.z_label_color = colors[2]

    @property
    def label_size(self) -> tuple[float, float]:  # numpydoc ignore=RT01
        """The width and height of the axes labels.

        The width and height are expressed as a fraction of the viewport.
        Values must be in range ``[0, 1]``.
        """
        # Get size from x actor
        width, height = self._label_actors[0].GetWidth(), self._label_actors[0].GetHeight()
        # Make sure y and z have the same size
        self._label_actors[1].SetWidth(width), self._label_actors[1].SetHeight(height)
        self._label_actors[2].SetWidth(width), self._label_actors[2].SetHeight(height)
        return width, height

    @label_size.setter
    def label_size(self, size: Sequence[float]):  # numpydoc ignore=GL08
        valid_range = [0, 1]
        _check_range(size[0], valid_range, 'label width')
        _check_range(size[1], valid_range, 'label height')
        for actor in self._label_actors:
            actor.SetWidth(size[0])
            actor.SetHeight(size[1])

    @property
    def show_labels(self) -> bool:  # numpydoc ignore=RT01
        """Enable or disable the text labels for the axes."""
        return bool(self.GetAxisLabels())

    @show_labels.setter
    def show_labels(self, value: bool):  # numpydoc ignore=GL08
        self.SetAxisLabels(value)

    def plot(self):
        pl = pyvista.Plotter()
        pl.add_actor(self)
        pl.show()

    # @property
    # def label_position(self) -> Tuple[float, float, float]:  # numpydoc ignore=RT01
    #     """Normalized position of the text label along each axis.
    #
    #     Values must be non-negative.
    #
    #     Examples
    #     --------
    #     >>> import pyvista as pv
    #     >>> axes_actor = pv.AxesAssembly()
    #     >>> axes_actor.label_position
    #     (1.0, 1.0, 1.0)
    #     >>> axes_actor.label_position = 0.3
    #     >>> axes_actor.label_position
    #     (0.3, 0.3, 0.3)
    #     >>> axes_actor.label_position = (0.1, 0.4, 0.2)
    #     >>> axes_actor.label_position
    #     (0.1, 0.4, 0.2)
    #
    #     """
    #     return self._label_position
    #
    # @label_position.setter
    # def label_position(self, position: Union[float, VectorLike[float]]):  # numpydoc ignore=GL08
    #     if isinstance(position, (int, float)):
    #         _check_range(position, (0, float('inf')), 'label_position')
    #         position = [position, position, position]
    #     _check_range(position[0], (0, float('inf')), 'x-axis label_position')
    #     _check_range(position[1], (0, float('inf')), 'y-axis label_position')
    #     _check_range(position[2], (0, float('inf')), 'z-axis label_position')
    #     self._label_position = float(position[0]), float(position[1]), float(position[2])
