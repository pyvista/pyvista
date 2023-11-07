"""Axes actor module."""
from functools import wraps
from typing import Sequence, Tuple, Union
import warnings

import numpy as np

import pyvista
from pyvista.core._typing_core import BoundsLike, Vector
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.arrays import array_from_vtkmatrix, vtkmatrix_from_array
from pyvista.core.utilities.misc import AnnotatedIntEnum, assert_empty_kwargs

from . import _vtk
from .actor_properties import ActorProperties
from .colors import Color, ColorLike
from .prop3d import Prop3D


class AxesActor(Prop3D, _vtk.vtkAxesActor):
    """Create a hybrid 2D/3D actor to represent 3D axes in a scene.

    The axes colors, labels, and shaft/tip geometry can all be customized.
    The axes can also be arbitrarily positioned and oriented in a scene.

    .. versionadded:: 0.43.0

        - ``AxesActor`` can now be initialized with any/all properties specified.
        - The shaft and tip type can now be set using strings. Previously, the use
          of a ``ShaftType`` or ``TipType`` Enum was required.
        - Added ability to position and orient the axes in space.
        - Added spatial properties ``orientation``, ``scale``, ``position``, ``origin``,
          and ``user_matrix``.
        - Added spatial methods ``rotate_x``, ``rotate_y``, ``rotate_z``.
        - Added color properties ``label_color``, ``x_color``, ``y_color``, and ``z_color``.
        - Added ``label_size`` property.
        - Added ``properties`` keyword to initialize any ``ActorProperty`` properties
          (e.g. ``ambient``, ``specular``, etc.).

    .. versionchanged:: 0.43.0

        - The default shaft type has been changed from 'line' to 'cylinder'.
        - The axes shaft and tip properties have been abstracted, e.g. use
          ``tip_radius`` to set the radius of the axes tips regardless of the ``tip_type``
          used. Previously, it was necessary to use ``cone_radius`` or ``sphere_radius``
          separately. See the list of deprecated properties below for details.

    .. deprecated:: 0.43.0

        The following properties have been deprecated:

        - ``x_axis_label`` -> use ``x_label`` instead.
        - ``y_axis_label`` -> use ``y_label`` instead.
        - ``z_axis_label`` -> use ``z_label`` instead.
        - ``cone_radius`` ->  use ``tip_radius`` instead.
        - ``sphere_radius`` -> use ``tip_radius`` instead.
        - ``cone_resolution`` -> use ``tip_resolution`` instead.
        - ``sphere_resolution`` -> use ``tip_resolution`` instead.
        - ``cylinder_resolution`` -> use ``shaft_resolution`` instead.
        - ``cylinder_radius`` -> use ``shaft_radius`` instead.
        - ``line_width`` -> use ``shaft_width`` instead.

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

    label_position : float | Vector, default: 1
        Normalized label position along the axes shafts. If a number,
        the label position for all axes is set to this value. Values
        should be in range ``[0, 1]``.

    x_color : ColorLike, optional
        Color of the x-axis shaft and tip. By default, the color is
        set to ``pyvista.global_theme.axes.x_color``.

    y_color : ColorLike, optional
        Color of the y-axis shaft and tip. By default, the color is
        set to ``pyvista.global_theme.axes.y_color``.

    z_color : ColorLike, optional
        Color of the z-axis shaft and tip. By default, the color is
        set to ``pyvista.global_theme.axes.z_color``.

    shaft_type : str | AxesActor.ShaftType, default: 'cylinder'
        Shaft type of the axes, either ``'cylinder'`` or ``'line'``.

    shaft_radius : float, default: 0.1
        Cylinder radius of the axes shafts. Only has an effect if ``shaft_type``
        is ``'cylinder'``.

    shaft_width : float, default: 2
        Line width of the axes shafts in screen units. Only has
        an effect if ``shaft_type`` is ``'line'``.

    shaft_length : float | Vector, default: 0.8
        Normalized length of the shaft for each axis. If a number, the shaft
        length for all axes is set to this value. Values should be in range
        ``[0, 1]``.

    shaft_resolution : int, default: 16
        Resolution of the axes shafts. Only has an effect if ``shaft_type``
        is ``'cylinder'``.

    tip_type : str | AxesActor.TipType, default: 'cone'
        Tip type of the axes, either ``'cone'`` or ``'sphere'``.

    tip_radius : float, default: 0.4
        Radius of the axes tips.

    tip_length : float | Vector, default: 0.2
        Normalized length of the axes tips. If a number, the shaft
        length for all axes is set to this value. Values should be in range
        ``[0, 1]``.

    tip_resolution : int , default: 16
        Resolution of the axes tips.

    total_length : float | Vector, default: (1, 1, 1)
        Total length of each axis (shaft plus tip).

    position : Vector, default: (0, 0, 0)
        Position of the axes.

    orientation : Vector, default: (0, 0, 0)
        Orientation angles of the axes which define rotations about
        the world's x-y-z axes. The angles are specified in degrees
        and in x-y-z order. However, the actual rotations are
        applied in the following order: rotate_z first,
        then rotate_x, and finally rotate_y.

    origin : Vector, default: (0, 0, 0)
        Origin of the axes. This is the point about which all
        rotations take place.

    scale : float | Vector, default: (1, 1, 1)
        Scaling factor for the axes.

    user_matrix : vtkMatrix3x3 | vtkMatrix4x4 | vtkTransform | np.ndarray
        Transformation to apply to the axes. Can be a vtkTransform,
        3x3 transformation matrix, or 4x4 transformation matrix.
        Defaults to the identity matrix.

    visibility : bool, default: True
        Visibility of the axes. If ``False``, the axes are not visible.

    properties : dict, optional
        Apply any :class:`~pyvista.ActorProperties` to all axes shafts and tips.

    **kwargs : dict, optional
        Used for handling deprecated parameters.

    Examples
    --------
    Create the default axes actor.

    >>> import pyvista as pv
    >>> axes_actor = pv.AxesActor()
    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(axes_actor)
    >>> pl.show()

    Create an axes actor with a specific position and orientation in space.

    >>> axes_actor = pv.AxesActor(
    ...     position=(1, 2, 3), orientation=(10, 20, 30)
    ... )
    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(axes_actor, reset_camera=True)
    >>> _ = pl.show_grid()
    >>> pl.show()

    Create an axes actor with custom colors and axis labels.

    >>> axes_actor = pv.AxesActor(
    ...     shaft_type='line',
    ...     shaft_width=4,
    ...     x_color="#378df0",
    ...     y_color="#ab2e5d",
    ...     z_color="#f7fb9a",
    ...     x_label="X Axis",
    ...     y_label="Y Axis",
    ...     z_label="Z Axis",
    ...     label_size=(0.1, 0.1),
    ... )
    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(axes_actor)
    >>> pl.show()


    The actor can also be used as a custom orientation widget with
    :func:`~pyvista.Renderer.add_orientation_widget`

    >>> axes_actor = pv.AxesActor(x_label="U", y_label="V", z_label="W")

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(pv.Cone())
    >>> _ = pl.add_orientation_widget(
    ...     axes_actor,
    ...     viewport=(0, 0, 0.5, 0.5),
    ... )
    >>> pl.show()

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
        shaft_radius=0.01,
        shaft_width=2,
        shaft_length=0.8,
        shaft_resolution=16,
        tip_type="cone",
        tip_radius=0.4,
        tip_length=0.2,
        tip_resolution=16,
        total_length=(1, 1, 1),
        position=(0, 0, 0),
        orientation=(0, 0, 0),
        origin=(0, 0, 0),
        scale=(1, 1, 1),
        user_matrix=None,
        visibility=True,
        properties=None,
        **kwargs,
    ):
        """Initialize AxesActor."""
        super().__init__()
        # Enable workaround to make axes orientable in space
        self._init_make_orientable(kwargs)

        # Supported aliases (inherited from `create_axes_actor`)
        x_label = kwargs.pop('xlabel', x_label)
        y_label = kwargs.pop('ylabel', y_label)
        z_label = kwargs.pop('zlabel', z_label)

        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.47.0
        cone_radius = kwargs.pop('cone_radius', False)
        if cone_radius:  # pragma: no cover
            warnings.warn(
                "Use of `cone_radius` is deprecated. Use `tip_radius` instead.",
                PyVistaDeprecationWarning,
            )
            tip_radius = cone_radius

        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.47.0
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

        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.47.0
        ambient = kwargs.pop('ambient', False)
        if ambient:  # pragma: no cover
            warnings.warn(
                f"Use of `ambient` is deprecated. Use `properties={{'ambient':{ambient}}}` instead.",
                PyVistaDeprecationWarning,
            )
            properties.setdefault('ambient', ambient)

        # Set axis shaft and tip properties
        properties.setdefault('lighting', pyvista.global_theme.lighting)
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

        assert_empty_kwargs(**kwargs)

        self.visibility = visibility
        self.total_length = total_length

        # Set shaft and tip color
        if x_color is None:
            x_color = Color(x_color, default_color=pyvista.global_theme.axes.x_color)
        self.x_color = x_color
        if y_color is None:
            y_color = Color(y_color, default_color=pyvista.global_theme.axes.y_color)
        self.y_color = y_color
        if z_color is None:
            z_color = Color(z_color, default_color=pyvista.global_theme.axes.z_color)
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
            label_color = Color(label_color, default_color=pyvista.global_theme.font.color)
        self.label_color = label_color

        # Set Prop3D properties
        self.user_matrix = user_matrix
        self.position = position
        self.origin = origin
        self.orientation = orientation
        self.scale = scale

    def __repr__(self):
        """Representation of the actor."""
        if self.user_matrix is None:
            mat_info = 'Unset'
        elif np.array_equal(self.user_matrix, np.eye(4)):
            mat_info = 'Identity'
        else:
            mat_info = 'Set'

        if self.shaft_type.annotation == 'cylinder':
            shaft_param = 'Shaft radius:'
            shaft_value = self.shaft_radius
        else:
            shaft_param = 'Shaft width: '
            shaft_value = self.shaft_width

        bnds = self.bounds

        attr = [
            f"{type(self).__name__} ({hex(id(self))})",
            f"  X label:                    '{self.x_label}'",
            f"  Y label:                    '{self.y_label}'",
            f"  Z label:                    '{self.z_label}'",
            f"  Show labels:                {not self.labels_off}",
            f"  Label position:             {self.label_position}",
            f"  Label size:                 {self.label_size}",
            f"  Shaft type:                 '{self.shaft_type.annotation}'",
            f"  {shaft_param}               {shaft_value}",
            f"  Shaft length:               {self.shaft_length}",
            f"  Tip type:                   '{self.tip_type.annotation}'",
            f"  Tip radius:                 {self.tip_radius}",
            f"  Tip length:                 {self.tip_length}",
            f"  Total length:               {self.total_length}",
            f"  Position:                   {self.position}",
            f"  Orientation:                {self.orientation}",
            f"  Origin:                     {self.origin}",
            f"  Scale:                      {self.scale}",
            f"  User matrix:                {mat_info}",
            f"  Visible:                    {self.visibility}",
            f"  X Bounds                    {bnds[0]:.3E}, {bnds[1]:.3E}",
            f"  Y Bounds                    {bnds[2]:.3E}, {bnds[3]:.3E}",
            f"  Z Bounds                    {bnds[4]:.3E}, {bnds[5]:.3E}",
        ]
        return '\n'.join(attr)

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
    def total_length(self) -> Tuple[float, float, float]:  # numpydoc ignore=RT01
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
    def total_length(self, length: Union[float, Vector]):  # numpydoc ignore=GL08
        if isinstance(length, (int, float)):
            self.SetTotalLength(length, length, length)
        else:
            self.SetTotalLength(length[0], length[1], length[2])

    @property
    def shaft_length(self) -> Tuple[float, float, float]:  # numpydoc ignore=RT01
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
    def shaft_length(self, length: Union[float, Vector]):  # numpydoc ignore=GL08
        if isinstance(length, (int, float)):
            self.SetNormalizedShaftLength(length, length, length)
        else:
            self.SetNormalizedShaftLength(length[0], length[1], length[2])

    @property
    def tip_length(self) -> Tuple[float, float, float]:  # numpydoc ignore=RT01
        """Normalized length of the tip for each axis.

        Values should be in range ``[0, 1]``.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesActor()
        >>> axes_actor.tip_length
        (0.2, 0.2, 0.2)
        >>> axes_actor.tip_length = 0.3
        >>> axes_actor.tip_length
        (0.3, 0.3, 0.3)
        >>> axes_actor.tip_length = (0.1, 0.4, 0.2)
        >>> axes_actor.tip_length
        (0.1, 0.4, 0.2)

        """
        return self.GetNormalizedTipLength()

    @tip_length.setter
    def tip_length(self, length: Union[float, Vector]):  # numpydoc ignore=GL08
        if isinstance(length, (int, float)):
            self.SetNormalizedTipLength(length, length, length)
        else:
            self.SetNormalizedTipLength(length[0], length[1], length[2])

    @property
    def label_position(self) -> Tuple[float, float, float]:  # numpydoc ignore=RT01
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
    def label_position(self, length: Union[float, Vector]):  # numpydoc ignore=GL08
        if isinstance(length, (int, float)):
            self.SetNormalizedLabelPosition(length, length, length)
        else:
            self.SetNormalizedLabelPosition(length[0], length[1], length[2])

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

        .. deprecated:: 0.43.0

            This parameter is deprecated. Use `tip_resolution` instead.

        """
        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.47.0
        warnings.warn(
            "Use of `cone_resolution` is deprecated. Use `tip_resolution` instead.",
            PyVistaDeprecationWarning,
        )
        return self.tip_resolution

    @cone_resolution.setter
    def cone_resolution(self, resolution: int):  # numpydoc ignore=GL08
        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.47.0
        warnings.warn(
            "Use of `cone_resolution` is deprecated. Use `tip_resolution` instead.",
            PyVistaDeprecationWarning,
        )
        self.tip_resolution = resolution

    @property
    def sphere_resolution(self) -> int:  # numpydoc ignore=RT01
        """Resolution of the axes sphere tips.

        .. deprecated:: 0.43.0

            This parameter is deprecated. Use `tip_resolution` instead.

        """
        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.47.0
        warnings.warn(
            "Use of `sphere_resolution` is deprecated. Use `tip_resolution` instead.",
            PyVistaDeprecationWarning,
        )
        return self.tip_resolution

    @sphere_resolution.setter
    def sphere_resolution(self, resolution: int):  # numpydoc ignore=GL08
        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.47.0
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

        .. deprecated:: 0.43.0

            This parameter is deprecated. Use `shaft_resolution` instead.

        """
        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.47.0
        warnings.warn(
            "Use of `cylinder_resolution` is deprecated. Use `shaft_resolution` instead.",
            PyVistaDeprecationWarning,
        )
        return self.shaft_resolution

    @cylinder_resolution.setter
    def cylinder_resolution(self, res: int):  # numpydoc ignore=GL08
        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.47.0
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
        >>> axes_actor = pv.AxesActor()
        >>> axes_actor.tip_radius
        0.4
        >>> axes_actor.tip_radius = 0.8
        >>> axes_actor.tip_radius
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

        .. deprecated:: 0.43.0

            This parameter is deprecated. Use `tip_radius` instead.

        """
        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.47.0
        warnings.warn(
            "Use of `cone_radius` is deprecated. Use `tip_radius` instead.",
            PyVistaDeprecationWarning,
        )
        return self.tip_radius

    @cone_radius.setter
    def cone_radius(self, radius: float):  # numpydoc ignore=GL08
        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.47.0
        warnings.warn(
            "Use of `cone_radius` is deprecated. Use `tip_radius` instead.",
            PyVistaDeprecationWarning,
        )
        self.tip_radius = radius

    @property
    def sphere_radius(self) -> float:  # numpydoc ignore=RT01
        """Radius of the axes sphere tips.

        .. deprecated:: 0.43.0

            This parameter is deprecated. Use `tip_radius` instead.

        """
        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.47.0
        warnings.warn(
            "Use of `sphere_radius` is deprecated. Use `tip_radius` instead.",
            PyVistaDeprecationWarning,
        )
        return self.tip_radius

    @sphere_radius.setter
    def sphere_radius(self, radius: float):  # numpydoc ignore=GL08
        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.47.0
        warnings.warn(
            "Use of `sphere_radius` is deprecated. Use `tip_radius` instead.",
            PyVistaDeprecationWarning,
        )
        self.tip_radius = radius

    @property
    def cylinder_radius(self) -> float:  # numpydoc ignore=RT01
        """Cylinder radius of the axes shafts.

        .. deprecated:: 0.43.0

            This parameter is deprecated. Use `shaft_radius` instead.

        """
        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.47.0
        warnings.warn(
            "Use of `cylinder_radius` is deprecated. Use `shaft_radius` instead.",
            PyVistaDeprecationWarning,
        )
        return self.shaft_radius

    @cylinder_radius.setter
    def cylinder_radius(self, radius: float):  # numpydoc ignore=GL08
        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.47.0
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
        >>> axes_actor = pv.AxesActor()
        >>> axes_actor.shaft_type = "line"
        >>> axes_actor.shaft_type
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
        >>> axes_actor = pv.AxesActor()
        >>> axes_actor.tip_type = axes_actor.TipType.SPHERE
        >>> axes_actor.tip_type
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
        >>> axes_actor = pv.AxesActor()
        >>> axes_actor.x_label = 'This axis'
        >>> axes_actor.x_label
        'This axis'

        """
        return self.GetXAxisLabelText()

    @x_label.setter
    def x_label(self, label: str):  # numpydoc ignore=GL08
        self.SetXAxisLabelText(label)

    @property
    def x_axis_label(self) -> str:  # numpydoc ignore=RT01
        """Text label for the x-axis.

        .. deprecated:: 0.43.0

            This parameter is deprecated. Use `x_label` instead.

        """
        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.47.0
        warnings.warn(
            "Use of `x_axis_label` is deprecated. Use `x_label` instead.",
            PyVistaDeprecationWarning,
        )
        return self.x_label

    @x_axis_label.setter
    def x_axis_label(self, label: str):  # numpydoc ignore=GL08
        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.47.0
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
        >>> axes_actor = pv.AxesActor()
        >>> axes_actor.y_label = 'This axis'
        >>> axes_actor.y_label
        'This axis'

        """
        return self.GetYAxisLabelText()

    @y_label.setter
    def y_label(self, label: str):  # numpydoc ignore=GL08
        self.SetYAxisLabelText(label)

    @property
    def y_axis_label(self) -> str:  # numpydoc ignore=RT01
        """Text label for the y-axis.

        .. deprecated:: 0.43.0

            This parameter is deprecated. Use `y_label` instead.

        """
        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.47.0
        warnings.warn(
            "Use of `y_axis_label` is deprecated. Use `y_label` instead.",
            PyVistaDeprecationWarning,
        )
        return self.y_label

    @y_axis_label.setter
    def y_axis_label(self, label: str):  # numpydoc ignore=GL08
        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.47.0
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
        >>> axes_actor = pv.AxesActor()
        >>> axes_actor.z_label = 'This axis'
        >>> axes_actor.z_label
        'This axis'

        """
        return self.GetZAxisLabelText()

    @z_label.setter
    def z_label(self, label: str):  # numpydoc ignore=GL08
        self.SetZAxisLabelText(label)

    @property
    def z_axis_label(self) -> str:  # numpydoc ignore=RT01
        """Text label for the z-axis.

        .. deprecated:: 0.43.0

            This parameter is deprecated. Use `z_label` instead.

        """
        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.47.0
        warnings.warn(
            "Use of `z_axis_label` is deprecated. Use `z_label` instead.",
            PyVistaDeprecationWarning,
        )
        return self.z_label

    @z_axis_label.setter
    def z_axis_label(self, label: str):  # numpydoc ignore=GL08
        # deprecated 0.43.0, convert to error in 0.46.0, remove 0.47.0
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

    @wraps(_vtk.vtkAxesActor.GetBounds)
    def GetBounds(self) -> BoundsLike:  # numpydoc ignore=RT01,GL08
        """Wrap method to make axes orientable in space."""
        if self._make_orientable:
            # GetBounds() defined by vtkAxesActor accesses a protected self.Bounds property
            # which is hard-coded to be centered and symmetric about the origin.
            # However, since the axes are made orientable, we instead compute the bounds from the
            # actual shaft and tip actor bounds.

            # NOTE: Overriding GetBounds() only works for python methods that call GetBounds().
            # It has no effect on compiled vtk code which directly calls vtkAxesActor.GetBounds().
            # This can result in camera issues when rendering a scene since vtkRenderer.ComputeVisiblePropBounds()
            # uses GetBounds() of all actors when resetting the camera, which will not execute this override.
            return self._compute_actor_bounds()
        return super().GetBounds()

    @wraps(_vtk.vtkProp3D.RotateX)
    def RotateX(self, *args):  # numpydoc ignore=RT01,PR01
        """Wrap method to make axes orientable in space."""
        super().RotateX(*args)
        self._update_actor_transformations() if self._make_orientable else None

    @wraps(_vtk.vtkProp3D.RotateY)
    def RotateY(self, *args):  # numpydoc ignore=RT01,PR01
        """Wrap method to make axes orientable in space."""
        super().RotateY(*args)
        self._update_actor_transformations() if self._make_orientable else None

    @wraps(_vtk.vtkProp3D.RotateZ)
    def RotateZ(self, *args):  # numpydoc ignore=RT01,PR01
        """Wrap method to make axes orientable in space."""
        super().RotateZ(*args)
        self._update_actor_transformations() if self._make_orientable else None

    @wraps(_vtk.vtkProp3D.SetScale)
    def SetScale(self, *args):  # numpydoc ignore=RT01,PR01
        """Wrap method to make axes orientable in space."""
        super().SetScale(*args)
        self._update_actor_transformations() if self._make_orientable else None

    @wraps(_vtk.vtkProp3D.SetOrientation)
    def SetOrientation(self, *args):  # numpydoc ignore=RT01,PR01
        """Wrap method to make axes orientable in space."""
        super().SetOrientation(*args)
        self._update_actor_transformations() if self._make_orientable else None

    @wraps(_vtk.vtkProp3D.SetOrigin)
    def SetOrigin(self, *args):  # numpydoc ignore=RT01,PR01
        """Wrap method to make axes orientable in space."""
        super().SetOrigin(*args)
        self._update_actor_transformations() if self._make_orientable else None

    @wraps(_vtk.vtkProp3D.SetPosition)
    def SetPosition(self, *args):  # numpydoc ignore=RT01,PR01
        """Wrap method to make axes orientable in space."""
        super().SetPosition(*args)
        self._update_actor_transformations() if self._make_orientable else None

    @wraps(_vtk.vtkAxesActor.SetNormalizedShaftLength)
    def SetNormalizedShaftLength(self, *args):  # numpydoc ignore=RT01,PR01
        """Wrap method to make axes orientable in space."""
        super().SetNormalizedShaftLength(*args)
        self._update_actor_transformations() if self._make_orientable else None

    @wraps(_vtk.vtkAxesActor.SetNormalizedTipLength)
    def SetNormalizedTipLength(self, *args):  # numpydoc ignore=RT01,PR01
        """Wrap method to make axes orientable in space."""
        super().SetNormalizedTipLength(*args)
        self._update_actor_transformations() if self._make_orientable else None

    @wraps(_vtk.vtkAxesActor.SetTotalLength)
    def SetTotalLength(self, *args):  # numpydoc ignore=RT01,PR01
        """Wrap method to make axes orientable in space."""
        super().SetTotalLength(*args)
        self._update_actor_transformations() if self._make_orientable else None

    @wraps(_vtk.vtkProp3D.GetUserMatrix)
    def GetUserMatrix(self):  # numpydoc ignore=RT01,PR01
        """Wrap method to make axes orientable in space."""
        if self._make_orientable:
            self._user_matrix = np.eye(4) if self._user_matrix is None else self._user_matrix
            return vtkmatrix_from_array(self._user_matrix)
        else:
            matrix = super().GetUserMatrix()
            self._user_matrix = array_from_vtkmatrix(matrix)
            return matrix

    @wraps(_vtk.vtkProp3D.SetUserMatrix)
    def SetUserMatrix(self, *args):  # numpydoc ignore=RT01,PR01
        """Wrap method to make axes orientable in space."""
        super().SetUserMatrix(*args)
        self._user_matrix = array_from_vtkmatrix(*args)
        self._update_actor_transformations() if self._make_orientable else None

    @property
    def _implicit_matrix(
        self, as_ndarray=True
    ) -> Union[np.ndarray, _vtk.vtkMatrix4x4]:  # numpydoc ignore=GL08
        """Compute the transformation matrix implicitly defined by 3D parameters."""
        temp_actor = _vtk.vtkActor()
        temp_actor.SetOrigin(self.GetOrigin())
        temp_actor.SetScale(self.GetScale())
        temp_actor.SetPosition(self.GetPosition())
        temp_actor.SetOrientation(self.GetOrientation())
        if as_ndarray:
            return array_from_vtkmatrix(temp_actor.GetMatrix())
        return temp_actor.GetMatrix()

    def _update_actor_transformations(self, reset=False):  # numpydoc ignore=RT01,PR01
        if reset:
            matrix = np.eye(4)
        else:
            matrix = self._concatenate_implicit_matrix_and_user_matrix()
        if not np.array_equal(matrix, self._cached_matrix) or reset:
            if np.array_equal(matrix, np.eye(4)):
                self.UseBoundsOn()
            else:
                # Calls to vtkRenderer.ResetCamera() will use the incorrect axes bounds
                # when axes have a transformation applied. As a workaround, set
                # UseBoundsOff(). In some cases, this will require manual adjustment
                # of the camera from the user
                self.UseBoundsOff()
            self._cached_matrix = matrix
            matrix = vtkmatrix_from_array(matrix)
            super().SetUserMatrix(matrix)
            [actor.SetUserMatrix(matrix) for actor in self._actors]
        # # Update text caption positions
        # t = _vtk.vtkTransform()
        # t.SetMatrix(matrix)
        # label_scale = np.array(self.label_position) * np.array(self.total_length)
        # label_position = array_from_vtkmatrix(matrix) @ (np.eye(4) * np.diag((*label_scale,1)))
        # print(label_position)
        # x = self.GetXAxisCaptionActor2D()
        # x.LeaderOn()
        # x.SetAttachmentPoint(0,0,0)
        # x.GetTextActor()
        # y=self.GetYAxisCaptionActor2D()
        # y.SetAttachmentPoint(label_position[1,:3] *5)
        # z = self.GetZAxisCaptionActor2D()
        # z.SetAttachmentPoint(label_position[2,:3] *5)

    @property
    def _actors(self):
        props = _vtk.vtkPropCollection()
        self.GetActors(props)
        actors = []
        for num in range(6):
            actors.append(props.GetItemAsObject(num))
        return tuple(actors)

    def _concatenate_implicit_matrix_and_user_matrix(self) -> np.ndarray:
        return self._user_matrix @ self._implicit_matrix

    def _update_props(self):
        # indirectly trigger a call to protected function self.UpdateProps()
        # by toggling the tip type
        if self.tip_type == 'cone':
            self.tip_type = 'sphere'
            self.tip_type = 'cone'
        else:
            self.tip_type = 'cone'
            self.tip_type = 'sphere'

    def _compute_actor_bounds(self) -> BoundsLike:
        """Compute symmetric axes bounds from the shaft and tip actors."""
        all_bounds = np.zeros((12, 3))
        for i, a in enumerate(self._actors):
            bnds = a.GetBounds()
            all_bounds[i] = bnds[0::2]
            all_bounds[i + 6] = bnds[1::2]

        # Append the same bounds, but inverted around the axes position to mimic having symmetric axes
        # The position may be moved with `user_matrix`, so apply the transform to position
        position = (self._user_matrix @ np.append(self.GetPosition(), 1))[:3]
        all_bounds = np.vstack((all_bounds, 2 * position - all_bounds))

        xmin, xmax = np.min(all_bounds[:, 0]), np.max(all_bounds[:, 0])
        ymin, ymax = np.min(all_bounds[:, 1]), np.max(all_bounds[:, 1])
        zmin, zmax = np.min(all_bounds[:, 2]), np.max(all_bounds[:, 2])
        return xmin, xmax, ymin, ymax, zmin, zmax

    @property
    def _make_orientable(self) -> bool:
        return self.__make_orientable

    @_make_orientable.setter
    def _make_orientable(self, value: bool):
        self.__make_orientable = value
        self._update_actor_transformations(reset=not value)

    def _init_make_orientable(self, kwargs):
        """Initialize workaround to make axes orientable in space."""
        self._user_matrix = np.eye(4)
        self._cached_matrix = np.eye(4)
        self._update_actor_transformations(reset=True)
        self.__make_orientable = kwargs.pop('_make_orientable', True)
        self._make_orientable = self.__make_orientable

    # def _re_init(self):
    #     """Re-initialize AxesActor."""
    #     # Create dict of current object properties and their values
    #     # using
    #     import inspect

    #     old_params = dict()
    #     param_keys = inspect.signature(AxesActor).parameters.keys()
    #     for key in param_keys:
    #         try:
    #             old_params[key] = getattr(self, key)
    #         except AttributeError:
    #             pass
    #     # Create new instances of axes actors and copy their default properties
    #     [actor.ShallowCopy(type(actor)()) for actor in self._actors]
    #     # Re-assign property values
    #     [setattr(self, key, val) for key, val in old_params.items()]


# def _orientation_to_direction(orientation, origin=(0,0,0), as_ndarray=True):
#     return _compute_implicit_matrix(orientation=orientation, origin=origin, as_ndarray=as_ndarray)
