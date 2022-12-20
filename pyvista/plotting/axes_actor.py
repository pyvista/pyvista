"""Axes actor module."""
from enum import Enum
from typing import Union

import pyvista as pv


class ShaftType(Enum):
    CYLINDER = 0
    LINE = 1


class TipType(Enum):
    CONE = 0
    SPHERE = 1


class AxesActor(pv._vtk.vtkAxesActor):
    """Axes actor wrapper for vtkAxesActor.

    Hybrid 2D/3D actor used to represent 3D axes in a scene. The user
    can define the geometry to use for the shaft or the tip, and the
    user can set the text for the three axes. To see full customization
    options, refer to `vtkAxesActor Details
    <https://vtk.org/doc/nightly/html/classvtkAxesActor.html#details>`

    Examples
    --------
    Customize the axis shaft color and shape.

    >>> import pyvista as pv
    >>> axes = pv.Axes()
    >>> axes.axes_actor.GetZAxisShaftProperty().SetColor(0, 1, 1)
    >>> axes.axes_actor.SetShaftTypeToCylinder()
    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(axes.axes_actor)
    >>> _ = pl.add_mesh(pv.Sphere())

    """

    def __init__(self):
        """Initialize actor."""
        super().__init__()

    @property
    def visibility(self) -> bool:
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
    def visibility(self, value: bool):
        return self.SetVisibility(value)

    @property
    def total_length(self) -> tuple:
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
    def total_length(self, length: Union[tuple, float]):
        if hasattr(length, '__iter__'):
            return self.SetTotalLength(length[0], length[1], length[2])
        else:
            return self.SetTotalLength(length, length, length)

    @property
    def shaft_length(self) -> tuple:
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
    def shaft_length(self, length: Union[tuple, float]):
        if hasattr(length, '__iter__'):
            return self.SetNormalizedShaftLength(length[0], length[1], length[2])
        else:
            return self.SetNormalizedShaftLength(length, length, length)

    @property
    def tip_length(self) -> tuple:
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
    def tip_length(self, length: Union[tuple, float]):
        if hasattr(length, '__iter__'):
            return self.SetNormalizedTipLength(length[0], length[1], length[2])
        else:
            return self.SetNormalizedTipLength(length, length, length)

    @property
    def label_position(self) -> tuple:
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
    def label_position(self, length: Union[tuple, float]):
        if hasattr(length, '__iter__'):
            return self.SetNormalizedLabelPosition(length[0], length[1], length[2])
        else:
            return self.SetNormalizedLabelPosition(length, length, length)

    @property
    def cone_resolution(self) -> int:
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
    def cone_resolution(self, res: int):
        return self.SetConeResolution(res)

    @property
    def sphere_resolution(self) -> int:
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
    def sphere_resolution(self, res: int):
        return self.SetSphereResolution(res)

    @property
    def cylinder_resolution(self) -> int:
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
    def cylinder_resolution(self, res: int):
        return self.SetCylinderResolution(res)

    @property
    def cone_radius(self) -> float:
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
    def cone_radius(self, rad: float):
        return self.SetConeRadius(rad)

    @property
    def sphere_radius(self) -> float:
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
    def sphere_radius(self, rad: float):
        return self.SetSphereRadius(rad)

    @property
    def cylinder_radius(self) -> float:
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
    def cylinder_radius(self, rad: float):
        return self.SetCylinderRadius(rad)

    @property
    def shaft_type(self) -> ShaftType:
        """Return or set the shaft type. Can be either
        a cylinder(0) or a line(1).

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista.plotting.axes_actor import ShaftType
        >>> axes = pv.Axes()
        >>> axes.axes_actor.shaft_type = ShaftType.LINE
        >>> axes.axes_actor.shaft_type
        <ShaftType.LINE: 1>

        """
        return ShaftType(self.GetShaftType())

    @shaft_type.setter
    def shaft_type(self, shaft_type: int):
        if shaft_type == ShaftType.CYLINDER:
            return self.SetShaftTypeToCylinder()
        elif shaft_type == ShaftType.LINE:
            return self.SetShaftTypeToLine()

    @property
    def tip_type(self) -> TipType:
        """Return or set the shaft type. Can be either
        a cone(0) or a sphere(1).

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista.plotting.axes_actor import TipType
        >>> axes = pv.Axes()
        >>> axes.axes_actor.tip_type = TipType.SPHERE
        >>> axes.axes_actor.tip_type
        <TipType.SPHERE: 1>
        """
        return TipType(self.GetTipType())

    @tip_type.setter
    def tip_type(self, tip_type: int):
        if tip_type == TipType.CONE:
            return self.SetTipTypeToCone()
        elif tip_type == TipType.SPHERE:
            return self.SetTipTypeToSphere()

    @property
    def x_axis_label(self) -> str:
        """Return or set the label for the X axis.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista.plotting.axes_actor import TipType
        >>> axes = pv.Axes()
        >>> axes.axes_actor.x_axis_label = 'This axis'
        >>> axes.axes_actor.x_axis_label
        'This axis'
        """
        return self.GetXAxisLabelText()

    @x_axis_label.setter
    def x_axis_label(self, label: str):
        return self.SetXAxisLabelText(label)

    @property
    def y_axis_label(self) -> str:
        """Return or set the label for the Y axis.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista.plotting.axes_actor import TipType
        >>> axes = pv.Axes()
        >>> axes.axes_actor.y_axis_label = 'This axis'
        >>> axes.axes_actor.y_axis_label
        'This axis'
        """
        return self.GetYAxisLabelText()

    @y_axis_label.setter
    def y_axis_label(self, label: str):
        return self.SetYAxisLabelText(label)

    @property
    def z_axis_label(self) -> str:
        """Return or set the label for the Z axis.

        Examples
        --------
        >>> import pyvista as pv
        >>> from pyvista.plotting.axes_actor import TipType
        >>> axes = pv.Axes()
        >>> axes.axes_actor.z_axis_label = 'This axis'
        >>> axes.axes_actor.z_axis_label
        'This axis'
        """
        return self.GetZAxisLabelText()

    @z_axis_label.setter
    def z_axis_label(self, label: str):
        return self.SetZAxisLabelText(label)
