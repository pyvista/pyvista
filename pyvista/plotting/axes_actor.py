"""Axes actor module."""
from functools import wraps
from typing import Sequence, Tuple, Union
import warnings

import numpy as np

import pyvista as pv
from pyvista.core._typing_core import BoundsLike
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.arrays import array_from_vtkmatrix, vtkmatrix_from_array
from pyvista.core.utilities.misc import AnnotatedIntEnum
from pyvista.core.utilities.transformations import apply_transformation_to_points

from ._vtk import vtkAxesActor, vtkMatrix4x4, vtkTransform
from .actor_properties import ActorProperties
from .colors import Color, ColorLike
from .prop3d import Prop3D


class AxesActor(Prop3D, vtkAxesActor):  # numpydoc ignore=PR01
    """Wrapper for vtkAxesActor.

    Hybrid 2D/3D actor used to represent 3D axes in a scene.

    """

    class ShaftType(AnnotatedIntEnum):
        """Types of shaft shapes available."""

        CYLINDER = (0, "cylinder")
        LINE = (1, "line")

    class TipType(AnnotatedIntEnum):
        """Types of tip shapes available."""

        CONE = (0, "cone")
        SPHERE = (1, "sphere")

    def __init__(
        self,
        x_label='X',
        y_label='Y',
        z_label='Z',
        label_color=None,
        labels_off=False,
        label_size=(0.25, 0.1),
        label_position=1,
        x_color=None,
        y_color=None,
        z_color=None,
        shaft_type="cylinder",
        shaft_length=0.8,
        shaft_radius=0.01,
        shaft_width=2,
        shaft_resolution=16,
        tip_radius=0.4,
        tip_length=0.2,
        tip_type="cone",
        tip_resolution=16,
        total_length=1,
        scale=1,
        position=(0, 0, 0),
        origin=(0, 0, 0),
        orientation=(0, 0, 0),
        user_matrix=np.eye(4),
        visibility=True,
        properties=None,
        **kwargs,
    ):
        """Create a hybrid 2D/3D actor to represent 3D axes in a scene.

        The axes colors, labels, and shaft/tip geometry can all be customized.
        The axes can also be arbitrarily positioned and oriented in a scene.

        Parameters
        ----------
        x_label : str, default: "X"
            Text label for the x-axis.

        y_label : str, default: "Y"
            Text label for the y-axis.

        z_label : str, default: "Z"
            Text label for the z-axis.

        label_color : ColorLike, optional
            Color of the label text for all axes.

        labels_off : bool, default: False
            Enable or disable the text labels for the axes.

        label_size : Sequence[float], default: (0.25, 0.1)
            The width and height of the axes labels. Values should
            be in range ``[0, 1]``.

        label_position : float | Sequence[float], default: 1
            Normalized label position along the axes shafts. Values should
            be in range ``[0, 1]``.

        x_color : ColorLike, optional
            Color of the x-axis shaft and tip.

        y_color : ColorLike, optional
            Color of the y-axis shaft and tip.

        z_color : ColorLike, optional
            Color of the z-axis shaft and tip.

        shaft_type : str | AxesActor.ShaftType, default: 'cylinder'
            Shaft type of the axes, either ``'cylinder'`` or ``'line'``.

        shaft_length : float | Sequence[float], default: 0.8
            Normalized length of the shaft for each axis. Values should be
            in range ``[0, 1]``.

        shaft_radius : float, default: 0.1
            Cylinder radius of the axes shafts. Only has an effect if ``shaft_type``
            is ``'cylinder'``.

        shaft_width : float, default: 2
            Line width of the axes shafts in screen units. Only has
            an effect if ``shaft_type`` is ``'line'``.

        shaft_resolution : int, default: 16
            Resolution of the axes shafts.

        tip_type : str | AxesActor.TipType, default: 'cone'
            Tip type of the axes, either ``'cone'`` or ``'sphere'``.

        tip_radius : float, default: 0.4
            Radius of the axes tips.

        tip_length : float | Sequence[float], default: 0.2
            Normalized length of the axes tips in range ``[0, 1]``.

        tip_resolution : int , default: 16
            Resolution of the axes tips.

        total_length float | Sequence[float], default: 1
            Total length of each axis (shaft plus tip).

        scale : float | Sequence[float], default: (1, 1, 1)
            Scaling factor for the axes.

        position : Sequence[float], default: (0, 0, 0)
            Position of the axes.

        origin : Sequence[float], default: (0, 0, 0)
            Origin of the axes. This is the point about which all
            rotations take place.

        orientation : Sequence[float], default: (0, 0, 0)
            Orientation angles of the axes which define rotations about
            the world's x-y-z axes. The angles are specified in degrees
            and in x-y-z order. However, the actual rotations are
            applied in the following order: rotate_z first,
            then rotate_x, and finally rotate_y.

        user_matrix : TransformLike
            Transformation to apply to the axes. Can be a vtkTransformation,
            3x3 transformation matrix, or 4x4 transformation matrix.

        properties: dict, optional
            Apply``:class:~pyvista.ActorProperties`` to all axes shafts and tips.

        **kwargs : dict, optional
            Used for handling deprecated parameters.

        Returns
        -------
        pvista.AxesActor

        Examples
        --------
        Create the default axes marker.

        >>> import pyvista as pv
        >>> axes_actor = pv.AxesActor()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_actor(axes_actor)
        >>> pl.show()

        Create an axes actor with custom colors and axis labels.

        >>> axes_actor = pv.AxesActor(
        ...     shaft_type='line',
        ...     shaft_width=4,
        ...     x_color="#378df0",
        ...     y_color="#ab2e5d",
        ...     z_color="#f7fb9a",
        ...     xlabel="X Axis",
        ...     ylabel="Y Axis",
        ...     zlabel="Z Axis",
        ...     label_size=(0.1, 0.1),
        ... )
        >>> pl = pv.Plotter()
        >>> _ = pl.add_actor(axes_actor)
        >>> pl.show()


        The actor can also be used as a custom orientation widget with
        :func:`~pyvista.Renderer.add_orientation_widget`

        >>> axes_actor = pv.AxesActor(
        ...     x_label="U", y_label="V", z_label="W"
        ... )

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Cone())
        >>> _ = pl.add_orientation_widget(
        ...     axes_actor,
        ...     viewport=(0, 0, 0.5, 0.5),
        ... )
        >>> pl.show()

        """
        super().__init__()
        self.__enable_orientation_workaround = True

        # Supported aliases (inherited from `create_axes_actor`)
        x_label = kwargs.pop('xlabel', x_label)
        y_label = kwargs.pop('ylabel', y_label)
        z_label = kwargs.pop('zlabel', z_label)

        # Deprecated on v0.43.0
        cone_radius = kwargs.pop('cone_radius', False)
        if cone_radius:  # pragma: no cover
            warnings.warn(
                "Use of `cone_radius` is deprecated. Use `tip_radius` instead.",
                PyVistaDeprecationWarning,
            )
            tip_radius = cone_radius

        # Deprecated on v0.43.0
        line_width = kwargs.pop('line_width', False)
        if line_width:  # pragma: no cover
            warnings.warn(
                "Use of `line_width` is deprecated. Use `shaft_width` instead.",
                PyVistaDeprecationWarning,
            )
            shaft_width = line_width

        if properties is None:
            properties = dict()
        if not isinstance(properties, dict):
            raise TypeError("Properties must be a dictionary.")

        # Deprecated on v0.43.0
        ambient = kwargs.pop('ambient', False)
        if ambient:  # pragma: no cover
            warnings.warn(
                f"Use of `ambient` is deprecated. Use `properties={{'ambient':{ambient}}}` instead.",
                PyVistaDeprecationWarning,
            )
            properties.setdefault('ambient', ambient)

        # Set axis shaft and tip properties
        properties.setdefault('lighting', pv.global_theme.lighting)
        if len(properties) > 0:
            x_shaft = self.x_axis_shaft_properties
            y_shaft = self.y_axis_shaft_properties
            z_shaft = self.z_axis_shaft_properties
            x_tip = self.x_axis_tip_properties
            y_tip = self.y_axis_tip_properties
            z_tip = self.z_axis_tip_properties
            for name, value in properties.items():
                setattr(x_shaft, name, value)
                setattr(y_shaft, name, value)
                setattr(z_shaft, name, value)
                setattr(x_tip, name, value)
                setattr(y_tip, name, value)
                setattr(z_tip, name, value)

        if len(kwargs) > 0:
            raise TypeError(
                f"AxesActor() got an unexpected keyword argument '{list(kwargs.keys())[0]}'"
            )

        self.visibility = visibility
        self.total_length = total_length

        # Set shaft and tip color
        if x_color is None:
            x_color = Color(x_color, default_color=pv.global_theme.axes.x_color)
        self.x_color = x_color
        if y_color is None:
            y_color = Color(y_color, default_color=pv.global_theme.axes.y_color)
        self.y_color = y_color
        if z_color is None:
            z_color = Color(z_color, default_color=pv.global_theme.axes.z_color)
        self.z_color = z_color

        # Set shaft properties
        self.shaft_type = shaft_type
        self.shaft_radius = shaft_radius
        self.shaft_width = shaft_width
        self.shaft_length = shaft_length
        self.shaft_resolution = shaft_resolution

        # Set tip properties
        self.tip_type = tip_type
        self.tip_radius = tip_radius
        self.tip_length = tip_length
        self.tip_resolution = tip_resolution

        # Set text label properties
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label
        self.labels_off = labels_off
        self.label_size = label_size
        self.label_position = label_position

        if label_color is None:
            label_color = Color(label_color, default_color=pv.global_theme.font.color)
        self.label_color = label_color

        # Set Prop3D properties
        self._user_matrix = user_matrix
        self.position = position
        self.origin = origin
        self.orientation = orientation
        self.scale = scale

    @property
    def visibility(self) -> bool:  # numpydoc ignore=RT01
        """Enable or disable the visibility of the axes.

        Examples
        --------
        Create an AxesActor and check its visibility

        >>> import pyvista as pv
        >>> axes_actor = pv.AxesActor()
        >>> axes_actor.visibility
        True

        """
        return bool(self.GetVisibility())

    @visibility.setter
    def visibility(self, value: bool):  # numpydoc ignore=GL08
        self.SetVisibility(value)

    @property
    def total_length(self) -> tuple:  # numpydoc ignore=RT01
        """Total length of each axis (shaft plus tip).

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesActor()
        >>> axes_actor.total_length
        (1.0, 1.0, 1.0)
        >>> axes_actor.total_length = 1.2
        >>> axes_actor.total_length
        (1.2, 1.2, 1.2)
        >>> axes_actor.total_length = (1.0, 0.9, 0.5)
        >>> axes_actor.total_length
        (1.0, 0.9, 0.5)

        """
        return self.GetTotalLength()

    @total_length.setter
    def total_length(self, length):  # numpydoc ignore=GL08
        if isinstance(length, Sequence):
            self.SetTotalLength(length[0], length[1], length[2])
        else:
            self.SetTotalLength(length, length, length)

    @property
    def shaft_length(self) -> tuple:  # numpydoc ignore=RT01
        """Normalized length of the shaft for each axis.

        Values should be in range ``[0, 1]``.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesActor()
        >>> axes_actor.shaft_length
        (0.8, 0.8, 0.8)
        >>> axes_actor.shaft_length = 0.7
        >>> axes_actor.shaft_length
        (0.7, 0.7, 0.7)
        >>> axes_actor.shaft_length = (1.0, 0.9, 0.5)
        >>> axes_actor.shaft_length
        (1.0, 0.9, 0.5)

        """
        return self.GetNormalizedShaftLength()

    @shaft_length.setter
    def shaft_length(self, length: Union[float, Sequence[float]]):  # numpydoc ignore=GL08
        if isinstance(length, Sequence):
            self.SetNormalizedShaftLength(length[0], length[1], length[2])
        else:
            self.SetNormalizedShaftLength(length, length, length)

    @property
    def tip_length(self) -> Tuple[float, float, float]:  # numpydoc ignore=RT01
        """Normalized length of the tip for each axis.

        Values should be in range ``[0, 1]``.

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
    def tip_length(self, length: Union[float, Sequence[float]]):  # numpydoc ignore=GL08
        if isinstance(length, Sequence):
            self.SetNormalizedTipLength(length[0], length[1], length[2])
        else:
            self.SetNormalizedTipLength(length, length, length)

    @property
    def label_position(self) -> tuple:  # numpydoc ignore=RT01
        """Normalized position of the text label along each axis.

        Values should be in range ``[0, 1]``.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesActor()
        >>> axes_actor.label_position
        (1.0, 1.0, 1.0)
        >>> axes_actor.label_position = 0.3
        >>> axes_actor.label_position
        (0.3, 0.3, 0.3)
        >>> axes_actor.label_position = (0.1, 0.4, 0.2)
        >>> axes_actor.label_position
        (0.1, 0.4, 0.2)

        """
        return self.GetNormalizedLabelPosition()

    @label_position.setter
    def label_position(self, length: Union[float, Sequence[float]]):  # numpydoc ignore=GL08
        if isinstance(length, Sequence):
            self.SetNormalizedLabelPosition(length[0], length[1], length[2])
        else:
            self.SetNormalizedLabelPosition(length, length, length)

    @property
    def tip_resolution(self) -> int:  # numpydoc ignore=RT01
        """Resolution of the axes tips.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesActor()
        >>> axes_actor.tip_resolution
        16
        >>> axes_actor.tip_resolution = 24
        >>> axes_actor.tip_resolution
        24

        """
        # Get cone value and assume value is the same for sphere
        resolution = self.GetConeResolution()

        # Make sure our assumption is true and reset value for cone and sphere
        self.tip_resolution = resolution

        return resolution

    @tip_resolution.setter
    def tip_resolution(self, resolution: int):  # numpydoc ignore=GL08
        self.SetConeResolution(resolution)
        self.SetSphereResolution(resolution)

    @property
    def cone_resolution(self) -> int:  # numpydoc ignore=RT01
        """Resolution of the axes cone tips.

        This parameter is deprecated. Use `tip_resolution` instead.
        """
        warnings.warn(
            "Use of `cone_resolution` is deprecated. Use `tip_resolution` instead.",
            PyVistaDeprecationWarning,
        )
        return self.tip_resolution

    @cone_resolution.setter
    def cone_resolution(self, resolution: int):  # numpydoc ignore=GL08
        warnings.warn(
            "Use of `cone_resolution` is deprecated. Use `tip_resolution` instead.",
            PyVistaDeprecationWarning,
        )
        self.tip_resolution = resolution

    @property
    def sphere_resolution(self) -> int:  # numpydoc ignore=RT01
        """Resolution of the axes sphere tips.

        This parameter is deprecated. Use `tip_resolution` instead.
        """
        warnings.warn(
            "Use of `sphere_resolution` is deprecated. Use `tip_resolution` instead.",
            PyVistaDeprecationWarning,
        )
        return self.tip_resolution

    @sphere_resolution.setter
    def sphere_resolution(self, resolution: int):  # numpydoc ignore=GL08
        warnings.warn(
            "Use of `sphere_resolution` is deprecated. Use `tip_resolution` instead.",
            PyVistaDeprecationWarning,
        )
        self.tip_resolution = resolution

    @property
    def shaft_resolution(self) -> int:  # numpydoc ignore=RT01
        """Resolution of the axes shafts.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesActor()
        >>> axes_actor.shaft_resolution
        16
        >>> axes_actor.shaft_resolution = 24
        >>> axes_actor.shaft_resolution
        24

        """
        return self.GetCylinderResolution()

    @shaft_resolution.setter
    def shaft_resolution(self, resolution: int):  # numpydoc ignore=GL08
        self.SetCylinderResolution(resolution)

    @property
    def cylinder_resolution(self) -> int:  # numpydoc ignore=RT01
        """Cylinder resolution of the axes shafts.

        This parameter is deprecated. Use `shaft_resolution` instead.

        """
        warnings.warn(
            "Use of `cylinder_resolution` is deprecated. Use `shaft_resolution` instead.",
            PyVistaDeprecationWarning,
        )
        return self.shaft_resolution

    @cylinder_resolution.setter
    def cylinder_resolution(self, res: int):  # numpydoc ignore=GL08
        warnings.warn(
            "Use of `cylinder_resolution` is deprecated. Use `shaft_resolution` instead.",
            PyVistaDeprecationWarning,
        )
        self.SetCylinderResolution(res)

    @property
    def tip_radius(self) -> float:  # numpydoc ignore=RT01
        """Radius of the axes tips.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.tip_radius
        0.4
        >>> axes.axes_actor.tip_radius = 0.8
        >>> axes.axes_actor.tip_radius
        0.8

        """
        # Get cone value and assume value is the same for sphere
        radius = self.GetConeRadius()

        # Make sure our assumption is true and reset value for cone and sphere
        self.tip_radius = radius
        return radius

    @tip_radius.setter
    def tip_radius(self, radius: float):  # numpydoc ignore=GL08
        self.SetConeRadius(radius)
        self.SetSphereRadius(radius)

    @property
    def cone_radius(self) -> float:  # numpydoc ignore=RT01
        """Radius of the axes cone tips.

        This parameter is deprecated. Use `tip_radius` instead.

        """
        warnings.warn(
            "Use of `cone_radius` is deprecated. Use `tip_radius` instead.",
            PyVistaDeprecationWarning,
        )
        return self.tip_radius

    @cone_radius.setter
    def cone_radius(self, radius: float):  # numpydoc ignore=GL08
        warnings.warn(
            "Use of `cone_radius` is deprecated. Use `tip_radius` instead.",
            PyVistaDeprecationWarning,
        )
        self.tip_radius = radius

    @property
    def sphere_radius(self) -> float:  # numpydoc ignore=RT01
        """Radius of the axes sphere tips.

        This parameter is deprecated. Use `tip_radius` instead.

        """
        warnings.warn(
            "Use of `sphere_radius` is deprecated. Use `tip_radius` instead.",
            PyVistaDeprecationWarning,
        )
        return self.tip_radius

    @sphere_radius.setter
    def sphere_radius(self, radius: float):  # numpydoc ignore=GL08
        warnings.warn(
            "Use of `sphere_radius` is deprecated. Use `tip_radius` instead.",
            PyVistaDeprecationWarning,
        )
        self.tip_radius = radius

    @property
    def cylinder_radius(self) -> float:  # numpydoc ignore=RT01
        """Cylinder radius of the axes shafts.

        This parameter is deprecated. Use `shaft_radius` instead.

        """
        warnings.warn(
            "Use of `cylinder_radius` is deprecated. Use `shaft_radius` instead.",
            PyVistaDeprecationWarning,
        )
        return self.shaft_radius

    @cylinder_radius.setter
    def cylinder_radius(self, radius: float):  # numpydoc ignore=GL08
        warnings.warn(
            "Use of `cylinder_radius` is deprecated. Use `shaft_radius` instead.",
            PyVistaDeprecationWarning,
        )
        self.shaft_radius = radius

    @property
    def shaft_radius(self):  # numpydoc ignore=RT01
        """Cylinder radius of the axes shafts.

        This property only has an effect if ``shaft_type`` is ``'cylinder'``.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesActor()
        >>> axes_actor.shaft_radius
        0.01
        >>> axes_actor.shaft_radius = 0.03
        >>> axes_actor.shaft_radius
        0.03

        """
        return self.GetCylinderRadius()

    @shaft_radius.setter
    def shaft_radius(self, radius):  # numpydoc ignore=GL08
        self.SetCylinderRadius(radius)

    @property
    def shaft_width(self):  # numpydoc ignore=RT01
        """Line width of the axes shafts in screen units.

        This property only has an effect if ``shaft_type`` is ``'line'``.

        """
        # Get x width and assume width is the same for each x, y, and z
        width_x = self.GetXAxisShaftProperty().GetLineWidth()

        # Make sure our assumption is true and reset all xyz widths
        self.shaft_width = width_x

        return width_x

    @shaft_width.setter
    def shaft_width(self, width):  # numpydoc ignore=GL08
        self.GetXAxisShaftProperty().SetLineWidth(width)
        self.GetYAxisShaftProperty().SetLineWidth(width)
        self.GetZAxisShaftProperty().SetLineWidth(width)

    @property
    def shaft_type(self) -> ShaftType:  # numpydoc ignore=RT01
        """Tip type for all axes.

        Can be a cylinder (``0`` or ``'cylinder'``) or a line (``1`` or ``'line'``).

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.shaft_type = "line"
        >>> axes.axes_actor.shaft_type
        <ShaftType.LINE: 1>

        """
        return AxesActor.ShaftType.from_any(self.GetShaftType())

    @shaft_type.setter
    def shaft_type(self, shaft_type: Union[ShaftType, int, str]):  # numpydoc ignore=GL08
        shaft_type = AxesActor.ShaftType.from_any(shaft_type)
        if shaft_type == AxesActor.ShaftType.CYLINDER:
            self.SetShaftTypeToCylinder()
        elif shaft_type == AxesActor.ShaftType.LINE:
            self.SetShaftTypeToLine()

    @property
    def tip_type(self) -> TipType:  # numpydoc ignore=RT01
        """Tip type for all axes.

        Can be a cone (``0`` or ``'cone'``) or a sphere (``1`` or ``'sphere'``).

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.tip_type = axes.axes_actor.TipType.SPHERE
        >>> axes.axes_actor.tip_type
        <TipType.SPHERE: 1>

        """
        return AxesActor.TipType.from_any(self.GetTipType())

    @tip_type.setter
    def tip_type(self, tip_type: Union[TipType, int, str]):  # numpydoc ignore=GL08
        tip_type = AxesActor.TipType.from_any(tip_type)
        if tip_type == AxesActor.TipType.CONE:
            self.SetTipTypeToCone()
        elif tip_type == AxesActor.TipType.SPHERE:
            self.SetTipTypeToSphere()

    @property
    def x_label(self) -> str:  # numpydoc ignore=RT01
        """Text label for the x-axis.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.x_label = 'This axis'
        >>> axes.axes_actor.x_label
        'This axis'

        """
        return self.GetXAxisLabelText()

    @x_label.setter
    def x_label(self, label: str):  # numpydoc ignore=GL08
        self.SetXAxisLabelText(label)

    @property
    def x_axis_label(self) -> str:  # numpydoc ignore=RT01
        """Text label for the x-axis.

        This parameter is deprecated. Use `x_label` instead.

        """
        warnings.warn(
            "Use of `x_axis_label` is deprecated. Use `x_label` instead.",
            PyVistaDeprecationWarning,
        )
        return self.x_label

    @x_axis_label.setter
    def x_axis_label(self, label: str):  # numpydoc ignore=GL08
        warnings.warn(
            "Use of `x_axis_label` is deprecated. Use `x_label` instead.",
            PyVistaDeprecationWarning,
        )
        self.x_label = label

    @property
    def y_label(self) -> str:  # numpydoc ignore=RT01
        """Text label for the y-axis.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.y_label = 'This axis'
        >>> axes.axes_actor.y_label
        'This axis'

        """
        return self.GetYAxisLabelText()

    @y_label.setter
    def y_label(self, label: str):  # numpydoc ignore=GL08
        self.SetYAxisLabelText(label)

    @property
    def y_axis_label(self) -> str:  # numpydoc ignore=RT01
        """Text label for the y-axis.

        This parameter is deprecated. Use `y_label` instead.

        """
        warnings.warn(
            "Use of `y_axis_label` is deprecated. Use `y_label` instead.",
            PyVistaDeprecationWarning,
        )
        return self.y_label

    @y_axis_label.setter
    def y_axis_label(self, label: str):  # numpydoc ignore=GL08
        warnings.warn(
            "Use of `y_axis_label` is deprecated. Use `y_label` instead.",
            PyVistaDeprecationWarning,
        )
        self.y_label = label

    @property
    def z_label(self) -> str:  # numpydoc ignore=RT01
        """Text label for the z-axis.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.z_label = 'This axis'
        >>> axes.axes_actor.z_label
        'This axis'

        """
        return self.GetZAxisLabelText()

    @z_label.setter
    def z_label(self, label: str):  # numpydoc ignore=GL08
        self.SetZAxisLabelText(label)

    @property
    def z_axis_label(self) -> str:  # numpydoc ignore=RT01
        """Text label for the z-axis.

        This parameter is deprecated. Use `z_label` instead.

        """
        warnings.warn(
            "Use of `z_axis_label` is deprecated. Use `z_label` instead.",
            PyVistaDeprecationWarning,
        )
        return self.z_label

    @z_axis_label.setter
    def z_axis_label(self, label: str):  # numpydoc ignore=GL08
        warnings.warn(
            "Use of `z_axis_label` is deprecated. Use `z_label` instead.",
            PyVistaDeprecationWarning,
        )
        self.z_label = label

    @property
    def x_axis_shaft_properties(self):  # numpydoc ignore=RT01
        """:class:`~pyvista.ActorProperties` of the x-axis shaft."""
        return ActorProperties(self.GetXAxisShaftProperty())

    @property
    def y_axis_shaft_properties(self):  # numpydoc ignore=RT01
        """:class:`~pyvista.ActorProperties` of the y-axis shaft."""
        return ActorProperties(self.GetYAxisShaftProperty())

    @property
    def z_axis_shaft_properties(self):  # numpydoc ignore=RT01
        """:class:`~pyvista.ActorProperties` of the z-axis shaft."""
        return ActorProperties(self.GetZAxisShaftProperty())

    @property
    def x_axis_tip_properties(self):  # numpydoc ignore=RT01
        """:class:`~pyvista.ActorProperties` of the x-axis tip."""
        return ActorProperties(self.GetXAxisTipProperty())

    @property
    def y_axis_tip_properties(self):  # numpydoc ignore=RT01
        """:class:`~pyvista.ActorProperties` of the y-axis tip."""
        return ActorProperties(self.GetYAxisTipProperty())

    @property
    def z_axis_tip_properties(self):  # numpydoc ignore=RT01
        """:class:`~pyvista.ActorProperties` of the z-axis tip."""
        return ActorProperties(self.GetZAxisTipProperty())

    @property
    def label_color(self) -> Color:  # numpydoc ignore=RT01
        """Color of the label text for all axes."""
        # Get x color and assume label color is the same for each x, y, and z
        prop_x = self.GetXAxisCaptionActor2D().GetCaptionTextProperty()
        color = Color(prop_x.GetColor())

        # Make sure our assumption is true and reset all xyz colors
        self.label_color = color
        return color

    @label_color.setter
    def label_color(self, color: ColorLike):  # numpydoc ignore=GL08
        color = Color(color)
        prop_x = self.GetXAxisCaptionActor2D().GetCaptionTextProperty()
        prop_y = self.GetYAxisCaptionActor2D().GetCaptionTextProperty()
        prop_z = self.GetZAxisCaptionActor2D().GetCaptionTextProperty()
        for prop in [prop_x, prop_y, prop_z]:
            prop.SetColor(color.float_rgb)
            prop.SetShadow(False)

    @property
    def x_color(self):  # numpydoc ignore=RT01
        """Color of the x-axis shaft and tip."""
        color = self.x_axis_tip_properties.color
        opacity = self.x_axis_tip_properties.opacity
        return Color(color=color, opacity=opacity)

    @x_color.setter
    def x_color(self, color: ColorLike):  # numpydoc ignore=GL08
        color = Color(color)
        self.x_axis_shaft_properties.color = color.float_rgb
        self.x_axis_tip_properties.color = color.float_rgb
        self.x_axis_shaft_properties.opacity = color.float_rgba[3]
        self.x_axis_tip_properties.opacity = color.float_rgba[3]

    @property
    def y_color(self):  # numpydoc ignore=RT01
        """Color of the y-axis shaft and tip."""
        color = self.y_axis_tip_properties.color
        opacity = self.y_axis_tip_properties.opacity
        return Color(color=color, opacity=opacity)

    @y_color.setter
    def y_color(self, color: ColorLike):  # numpydoc ignore=GL08
        color = Color(color)
        self.y_axis_shaft_properties.color = color.float_rgb
        self.y_axis_tip_properties.color = color.float_rgb
        self.y_axis_shaft_properties.opacity = color.float_rgba[3]
        self.y_axis_tip_properties.opacity = color.float_rgba[3]

    @property
    def z_color(self):  # numpydoc ignore=RT01
        """Color of the z-axis shaft and tip."""
        color = self.z_axis_tip_properties.color
        opacity = self.z_axis_tip_properties.opacity
        return Color(color=color, opacity=opacity)

    @z_color.setter
    def z_color(self, color: ColorLike):  # numpydoc ignore=GL08
        color = Color(color)
        self.z_axis_shaft_properties.color = color.float_rgb
        self.z_axis_tip_properties.color = color.float_rgb
        self.z_axis_shaft_properties.opacity = color.float_rgba[3]
        self.z_axis_tip_properties.opacity = color.float_rgba[3]

    @property
    def labels_off(self) -> bool:  # numpydoc ignore=RT01
        """Enable or disable the text labels for the axes."""
        return not bool(self.GetAxisLabels())

    @labels_off.setter
    def labels_off(self, value: bool):  # numpydoc ignore=GL08
        self.SetAxisLabels(not value)

    @property
    def label_size(self) -> Tuple[float, float]:  # numpydoc ignore=RT01
        """The width and height of the axes labels.

        Values should be in range ``[0, 1]``.
        """
        # Assume label size for x is same as y and z
        label_actor = self.GetXAxisCaptionActor2D()
        size = (label_actor.GetWidth(), label_actor.GetHeight())

        # Make sure our assumption is correct and reset values
        self.label_size = size
        return size

    @label_size.setter
    def label_size(self, size: Sequence[float]):  # numpydoc ignore=GL08
        for label_actor in [
            self.GetXAxisCaptionActor2D(),
            self.GetYAxisCaptionActor2D(),
            self.GetZAxisCaptionActor2D(),
        ]:
            label_actor.SetWidth(size[0])
            label_actor.SetHeight(size[1])

    @wraps(vtkAxesActor.GetBounds)
    def GetBounds(self) -> BoundsLike:  # numpydoc ignore=RT01,GL08
        """Wrap method for orientation workaround."""
        if self._enable_orientation_workaround:
            return self._compute_transformed_bounds()
        return super().GetBounds()

    @wraps(vtkAxesActor.RotateX)
    def RotateX(self, *args):  # numpydoc ignore=RT01,PR01
        """Wrap method for orientation workaround."""
        super().RotateX(*args)
        self._update_UserMatrix() if self._enable_orientation_workaround else None

    @wraps(vtkAxesActor.RotateY)
    def RotateY(self, *args):  # numpydoc ignore=RT01,PR01
        """Wrap method for orientation workaround."""
        super().RotateY(*args)
        self._update_UserMatrix() if self._enable_orientation_workaround else None

    @wraps(vtkAxesActor.RotateZ)
    def RotateZ(self, *args):  # numpydoc ignore=RT01,PR01
        """Wrap method for orientation workaround."""
        super().RotateZ(*args)
        self._update_UserMatrix() if self._enable_orientation_workaround else None

    @wraps(vtkAxesActor.SetScale)
    def SetScale(self, *args):  # numpydoc ignore=RT01,PR01
        """Wrap method for orientation workaround."""
        super().SetScale(*args)
        self._update_UserMatrix() if self._enable_orientation_workaround else None

    @wraps(vtkAxesActor.SetOrientation)
    def SetOrientation(self, *args):  # numpydoc ignore=RT01,PR01
        """Wrap method for orientation workaround."""
        super().SetOrientation(*args)
        self._update_UserMatrix() if self._enable_orientation_workaround else None

    @wraps(vtkAxesActor.SetOrigin)
    def SetOrigin(self, *args):  # numpydoc ignore=RT01,PR01
        """Wrap method for orientation workaround."""
        super().SetOrigin(*args)
        self._update_UserMatrix() if self._enable_orientation_workaround else None

    @wraps(vtkAxesActor.SetPosition)
    def SetPosition(self, *args):  # numpydoc ignore=RT01,PR01
        """Wrap method for orientation workaround."""
        super().SetPosition(*args)
        self._update_UserMatrix() if self._enable_orientation_workaround else None

    @wraps(vtkAxesActor.GetUserTransform)
    def GetUserTransform(self):  # numpydoc ignore=RT01,PR01
        """Wrap method for orientation workaround."""
        transform = super().GetUserTransform()
        if self._enable_orientation_workaround:
            transform_out = vtkTransform()
            transform_out.SetMatrix(transform.GetMatrix())
        return transform

    @property
    def user_matrix(self) -> np.ndarray:  # numpydoc ignore=RT01
        """User-specified transformation matrix.

        This is the last transformation applied to the actor before
        rendering. It can be used with, or in place of, the implicit
        transformation that is created through the use of ``scale``,
        ``position``, ``origin``, and ``orientation``.

        """
        return self._user_matrix

    @user_matrix.setter
    def user_matrix(self, value):  # numpydoc ignore=GL08
        self._update_UserMatrix() if self._enable_orientation_workaround else None
        if isinstance(value, np.ndarray):
            if value.shape != (4, 4):
                raise ValueError('User matrix array must be 4x4.')
        elif isinstance(value, vtkMatrix4x4):
            value = array_from_vtkmatrix(value)
        else:
            raise TypeError(
                'Input user matrix must be either:\n' '\tvtk.vtkMatrix4x4\n' '\t4x4 np.ndarray\n'
            )

        self._user_matrix = value
        self._update_UserMatrix()

    @property
    def _implicit_matrix(self) -> np.ndarray:  # numpydoc ignore=GL08
        # Compute the transformation matrix implicitly defined by 3D parameters
        temp_actor = vtkAxesActor()
        temp_actor.SetOrigin(*self.origin)
        temp_actor.SetScale(*self.scale)
        temp_actor.SetPosition(*self.position)
        temp_actor.SetOrientation(*self.orientation)
        return array_from_vtkmatrix(temp_actor.GetMatrix())

    def _update_UserMatrix(self):
        if self._enable_orientation_workaround:
            matrix = self._concatenate_implicit_matrix_and_user_matrix()
        else:
            matrix = self._user_matrix
        # transform = vtkTransform()
        # transform.SetMatrix(vtkmatrix_from_array(matrix))
        # self.SetUserTransform(transform)
        self.SetUserMatrix(vtkmatrix_from_array(matrix))

    def _concatenate_implicit_matrix_and_user_matrix(self):
        return self._user_matrix @ self._implicit_matrix

    def _compute_transformed_bounds(self) -> BoundsLike:
        # The default method from vtkProp3D does not compute the actual
        # bounds of the rendered actor (see pyvista issue #5019).
        # Therefore, we redefine the bounds using the actor's matrix

        # Define origin-centered, axis-aligned, symmetric extents of the
        # axes using points at each tip of the axes
        x, y, z = self.GetTotalLength()
        points = np.array([[x, 0, 0], [-x, 0, 0], [0, y, 0], [0, -y, 0], [0, 0, z], [0, 0, -z]])

        # Transform points
        matrix = self._concatenate_implicit_matrix_and_user_matrix()
        apply_transformation_to_points(matrix, points, inplace=True)

        # Compute bounds
        xmin, xmax = np.min(points[:, 0]), np.max(points[:, 0])
        ymin, ymax = np.min(points[:, 1]), np.max(points[:, 1])
        zmin, zmax = np.min(points[:, 2]), np.max(points[:, 2])
        return xmin, xmax, ymin, ymax, zmin, zmax

    @property
    def _enable_orientation_workaround(self) -> bool:
        return self.__enable_orientation_workaround

    @_enable_orientation_workaround.setter
    def _enable_orientation_workaround(self, value: bool):
        self.__enable_orientation_workaround = value
        self._update_UserMatrix()
