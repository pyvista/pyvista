"""Axes actor module."""
from collections.abc import Iterable
from enum import Enum
from typing import Union

import pyvista

from . import _vtk
from .actor_properties import ActorProperties


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
    >>> axes.axes_actor.z_axis_shaft_properties.color = (0, 1, 1)
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

    >>> axes_actor.x_axis_shaft_properties.color = (1, 1, 1)
    >>> axes_actor.y_axis_shaft_properties.color = (1, 1, 1)
    >>> axes_actor.z_axis_shaft_properties.color = (1, 1, 1)

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

        self.x_axis_shaft_properties.color = pyvista.global_theme.axes.x_color.int_rgb
        self.x_axis_tip_properties.color = pyvista.global_theme.axes.x_color.int_rgb
        self.x_axis_shaft_properties.opacity = pyvista.global_theme.axes.x_color.int_rgba[3]
        self.x_axis_tip_properties.opacity = pyvista.global_theme.axes.x_color.int_rgba[3]
        self.x_axis_shaft_properties.lighting = pyvista.global_theme.lighting

        self.y_axis_shaft_properties.color = pyvista.global_theme.axes.y_color.int_rgb
        self.y_axis_tip_properties.color = pyvista.global_theme.axes.y_color.int_rgb
        self.y_axis_shaft_properties.opacity = pyvista.global_theme.axes.y_color.int_rgba[3]
        self.y_axis_tip_properties.opacity = pyvista.global_theme.axes.y_color.int_rgba[3]
        self.y_axis_shaft_properties.lighting = pyvista.global_theme.lighting

        self.z_axis_shaft_properties.color = pyvista.global_theme.axes.z_color.int_rgb
        self.z_axis_tip_properties.color = pyvista.global_theme.axes.z_color.int_rgb
        self.z_axis_shaft_properties.opacity = pyvista.global_theme.axes.z_color.int_rgba[3]
        self.z_axis_tip_properties.opacity = pyvista.global_theme.axes.z_color.int_rgba[3]
        self.z_axis_shaft_properties.lighting = pyvista.global_theme.lighting

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
        return self.SetVisibility(value)

    @property
    def total_length(self) -> tuple:  # numpydoc ignore=RT01
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
    def shaft_length(self) -> tuple:  # numpydoc ignore=RT01
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
    def tip_length(self) -> tuple:  # numpydoc ignore=RT01
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
    def label_position(self) -> tuple:  # numpydoc ignore=RT01
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
    def shaft_type(self, shaft_type: Union[ShaftType, int]):  # numpydoc ignore=GL08
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
    def tip_type(self, tip_type: Union[TipType, int]):  # numpydoc ignore=GL08
        tip_type = AxesActor.TipType(tip_type)
        if tip_type == AxesActor.TipType.CONE:
            self.SetTipTypeToCone()
        elif tip_type == AxesActor.TipType.SPHERE:
            self.SetTipTypeToSphere()

    @property
    def x_axis_label(self) -> str:  # numpydoc ignore=RT01
        """Return or set the label for the X axis.

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
        """Return or set the label for the Y axis.

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
        """Return or set the label for the Z axis.

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
        """Return or set the properties of the X axis shaft."""
        return ActorProperties(self.GetXAxisShaftProperty())

    @property
    def y_axis_shaft_properties(self):  # numpydoc ignore=RT01
        """Return or set the properties of the Y axis shaft."""
        return ActorProperties(self.GetYAxisShaftProperty())

    @property
    def z_axis_shaft_properties(self):  # numpydoc ignore=RT01
        """Return or set the properties of the Z axis shaft."""
        return ActorProperties(self.GetZAxisShaftProperty())

    @property
    def x_axis_tip_properties(self):  # numpydoc ignore=RT01
        """Return or set the properties of the X axis tip."""
        return ActorProperties(self.GetXAxisTipProperty())

    @x_axis_tip_properties.setter
    def x_axis_tip_properties(self, properties: ActorProperties):  # numpydoc ignore=GL08
        self.x_axis_tip_properties = properties

    @property
    def y_axis_tip_properties(self):  # numpydoc ignore=RT01
        """Return or set the properties of the Y axis tip."""
        return ActorProperties(self.GetYAxisTipProperty())

    @y_axis_tip_properties.setter
    def y_axis_tip_properties(self, properties: ActorProperties):  # numpydoc ignore=GL08
        self.y_axis_tip_properties = properties

    @property
    def z_axis_tip_properties(self):  # numpydoc ignore=RT01
        """Return or set the properties of the Z axis tip."""
        return ActorProperties(self.GetZAxisTipProperty())

    @z_axis_tip_properties.setter
    def z_axis_tip_properties(self, properties: ActorProperties):  # numpydoc ignore=GL08
        self.z_axis_tip_properties = properties
