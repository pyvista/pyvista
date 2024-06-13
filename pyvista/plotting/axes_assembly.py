"""Axes actor module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import NamedTuple
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np

import pyvista as pv
from pyvista.core import _validation
from pyvista.plotting import _vtk
from pyvista.plotting._property import Property
from pyvista.plotting._property import _check_range
from pyvista.plotting.colors import Color
from pyvista.plotting.prop3d import Prop3D

from .colors import _validate_color_sequence

if TYPE_CHECKING:
    from pyvista.core._typing_core import VectorLike
    from pyvista.plotting._typing import ColorLike
    from pyvista.plotting.actor import Actor


class _AxesTuple(NamedTuple):
    x_shaft: Any
    y_shaft: Any
    z_shaft: Any
    x_tip: Any
    y_tip: Any
    z_tip: Any


class _AxisPartTuple(NamedTuple):
    shaft: Actor
    tip: Actor


class _Tuple3D(NamedTuple):
    x: Any
    y: Any
    z: Any


def _as_nested(obj: Sequence[Any]) -> _Tuple3D:
    """Reshape length-6 shaft and tip sequence as a 3D tuple with nested shaft and tip items."""
    return _Tuple3D(
        x=_AxisPartTuple(shaft=obj[0], tip=obj[3]),
        y=_AxisPartTuple(shaft=obj[1], tip=obj[4]),
        z=_AxisPartTuple(shaft=obj[2], tip=obj[5]),
    )


class AxesAssembly(_vtk.vtkAssembly, Prop3D):
    """Abstract base class for axes-like scene props.

    This class defines a common interface for manipulating the
    geometry and properties of six Prop3D Actors representing
    the axes (three for the shafts, three for the tips) and three
    Caption Actors for the text labels (one for each axis).

    This class is designed to be a superclass for vtkAxesActor
    but is abstracted to interface with similar axes-like
    representations.
    """

    GEOMETRY_OPTIONS: ClassVar[list[str]] = ['cylinder', 'sphere', 'cone', 'pyramid', 'cuboid']

    def __init__(
        self,
        x_color=None,
        y_color=None,
        z_color=None,
        shaft_type='cylinder',
        shaft_radius=0.05,
        shaft_length=None,
        shaft_resolution=None,
        tip_type=None,
        tip_radius=0.2,
        tip_length=None,
        tip_resolution=None,
        total_length=(1, 1, 1),
        position=(0, 0, 0),
        orientation=(0, 0, 0),
        origin=(0, 0, 0),
        scale=(1, 1, 1),
        user_matrix=None,
        visibility=True,
        symmetric_bounds=True,
        auto_length=True,
        properties=None,
    ):
        super().__init__()

        actors = [pv.Actor(mapper=pv.DataSetMapper()) for _ in range(6)]
        self._actors = _AxesTuple(*actors)

        # Add actors to assembly
        [self.AddPart(actor) for actor in self._actors]

        # Init actor properties
        properties = {} if properties is None else properties
        if isinstance(properties, dict):

            def _new_property():
                return Property(**properties)

        elif isinstance(properties, Property):

            def _new_property():
                return properties.copy()

        else:
            raise TypeError('`properties` must be a property object or a dictionary.')
        [actor.SetProperty(_new_property()) for actor in self._actors]

        self.visibility = _set_default(visibility, True)

        # Set shaft and tip color. The setters will auto-set theme vals
        self.x_color = x_color
        self.y_color = y_color
        self.z_color = z_color

        # # Set text labels
        # if labels is None:
        #     self.x_label = _set_default(x_label, 'X')
        #     self.y_label = _set_default(y_label, 'Y')
        #     self.z_label = _set_default(z_label, 'Z')
        # else:
        #     msg = "Cannot initialize '{}' and 'labels' properties together. Specify one or the other, not both."
        #     if x_label is not None:
        #         raise ValueError(msg.format('x_label'))
        #     if y_label is not None:
        #         raise ValueError(msg.format('y_label'))
        #     if z_label is not None:
        #         raise ValueError(msg.format('z_label'))
        #     self.labels = labels
        # self.show_labels = show_labels
        # self.label_color = label_color  # Setter will auto-set theme val
        # self.label_size = label_size

        # Set misc flag params
        self._symmetric_bounds = _set_default(symmetric_bounds, True)
        self._auto_length = _set_default(auto_length, True)

        # Set geometry-dependent params
        # self.label_position = _set_default(label_position, 1.0)
        self.shaft_type = shaft_type
        self.shaft_radius = _set_default(shaft_radius, 0.01)
        self.shaft_resolution = _set_default(shaft_resolution, 24)
        self.tip_type = tip_type
        self.tip_radius = _set_default(tip_radius, 0.4)
        self.tip_resolution = _set_default(tip_resolution, 24)
        self.total_length = _set_default(total_length, 1.0)

        self.position = _set_default(position, (0.0, 0.0, 0.0))
        self.orientation = _set_default(orientation, (0.0, 0.0, 0.0))
        self.origin = _set_default(origin, (0.0, 0.0, 0.0))
        self.scale = _set_default(scale, 1.0)
        self.user_matrix = _set_default(user_matrix, np.eye(4))

        # Check auto-length
        # Disable flag temporarily and restore later
        auto_length_set = self.auto_length
        self.auto_length = False

        shaft_length_set = shaft_length is not None
        tip_length_set = tip_length is not None

        self.shaft_length = _set_default(shaft_length, 0.8)
        self.tip_length = _set_default(tip_length, 0.2)

        if auto_length_set:
            lengths_sum_to_one = all(
                map(lambda x, y: (x + y) == 1.0, self.shaft_length, self.tip_length),
            )
            if shaft_length_set and tip_length_set and not lengths_sum_to_one:
                raise ValueError(
                    "Cannot set both `shaft_length` and `tip_length` when `auto_length` is `True`.\n"
                    "Set either `shaft_length` or `tip_length`, but not both.",
                )
            # Set property again, this time with auto-length enabled
            self.auto_length = True
            if shaft_length_set and not tip_length_set:
                self.shaft_length = shaft_length
            elif tip_length_set and not shaft_length_set:
                self.tip_length = tip_length
        else:
            self.auto_length = False

        self._update_geometry()

    def __repr__(self):
        """Representation of the axes actor."""
        matrix_not_set = self.user_matrix is None or np.array_equal(self.user_matrix, np.eye(4))
        mat_info = 'Identity' if matrix_not_set else 'Set'
        bnds = self.bounds

        attr = [
            f"{type(self).__name__} ({hex(id(self))})",
            # f"  X label:                    '{self.x_label}'",
            # f"  Y label:                    '{self.y_label}'",
            # f"  Z label:                    '{self.z_label}'",
            # f"  Show labels:                {self.show_labels}",
            # f"  Label position:             {self.label_position}",
            f"  Shaft type:                 '{self.shaft_type}'",
            f"  Shaft radius:               {self.shaft_radius}",
            f"  Shaft length:               {self.shaft_length}",
            f"  Tip type:                   '{self.tip_type}'",
            f"  Tip radius:                 {self.tip_radius}",
            f"  Tip length:                 {self.tip_length}",
            f"  Total length:               {self.total_length}",
            f"  Position:                   {self.position}",
            f"  Scale:                      {self.scale}",
            f"  User matrix:                {mat_info}",
            f"  Visible:                    {self.visibility}",
            f"  X Bounds                    {bnds[0]:.3E}, {bnds[1]:.3E}",
            f"  Y Bounds                    {bnds[2]:.3E}, {bnds[3]:.3E}",
            f"  Z Bounds                    {bnds[4]:.3E}, {bnds[5]:.3E}",
        ]
        return '\n'.join(attr)

    @property
    def symmetric_bounds(self) -> bool:  # numpydoc ignore=RT01
        """Enable or disable symmetry in the axes bounds calculation.

        Calculate the axes bounds as though the axes were symmetric,
        i.e. extended along -X, -Y, and -Z directions. Setting this
        parameter primarily affects camera positioning in a scene.

        - If ``True``, the axes :attr:`bounds` are symmetric about
          its :attr:`position`. Symmetric bounds allow for the axes to rotate
          about its origin, which is useful in cases where the actor
          is used as an orientation widget.

        - If ``False``, the axes :attr:`bounds` are calculated as-is.
          Asymmetric bounds are useful in cases where the axes are
          placed in a scene with other actors, since the symmetry
          could otherwise lead to undesirable camera positioning
          (e.g. camera may be positioned further away than necessary).

        Examples
        --------
        Get the symmetric bounds of the axes.

        >>> import pyvista as pv
        >>> axes_actor = pv.AxesActor(symmetric_bounds=True)
        >>> axes_actor.bounds
        (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        >>> axes_actor.center
        (0.0, 0.0, 0.0)

        Get the asymmetric bounds.

        >>> axes_actor.symmetric_bounds = False
        >>> axes_actor.bounds  # doctest:+SKIP
        (-0.08, 1.0, -0.08, 1.0, -0.08, 1.0)
        >>> axes_actor.center  # doctest:+SKIP
        (0.46, 0.46, 0.46)

        Show the difference in camera positioning with and without
        symmetric bounds. Orientation is added for visualization.

        >>> # Create actors
        >>> axes_actor_sym = pv.AxesActor(
        ...     orientation=(90, 0, 0), symmetric_bounds=True
        ... )
        >>> axes_actor_asym = pv.AxesActor(
        ...     orientation=(90, 0, 0), symmetric_bounds=False
        ... )
        >>>
        >>> # Show multi-window plot
        >>> pl = pv.Plotter(shape=(1, 2))
        >>> pl.subplot(0, 0)
        >>> _ = pl.add_text("Symmetric axes")
        >>> _ = pl.add_actor(axes_actor_sym)
        >>> pl.subplot(0, 1)
        >>> _ = pl.add_text("Asymmetric axes")
        >>> _ = pl.add_actor(axes_actor_asym)
        >>> pl.show()

        """
        return self._symmetric_bounds

    @symmetric_bounds.setter
    def symmetric_bounds(self, value: bool):  # numpydoc ignore=GL08
        self._symmetric_bounds = bool(value)

    @property
    def visibility(self) -> bool:  # numpydoc ignore=RT01
        """Enable or disable the visibility of the axes.

        Examples
        --------
        Create an AxesAssembly and check its visibility

        >>> import pyvista as pv
        >>> axes_actor = pv.AxesAssembly()
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

        Values must be non-negative.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesAssembly()
        >>> axes_actor.total_length
        (1.0, 1.0, 1.0)
        >>> axes_actor.total_length = 1.2
        >>> axes_actor.total_length
        (1.2, 1.2, 1.2)
        >>> axes_actor.total_length = (1.0, 0.9, 0.5)
        >>> axes_actor.total_length
        (1.0, 0.9, 0.5)

        """
        return self._total_length

    @total_length.setter
    def total_length(self, length: Union[float, VectorLike[float]]):  # numpydoc ignore=GL08
        if isinstance(length, (int, float)):
            _check_range(length, (0, float('inf')), 'total_length')
            length = (length, length, length)
        _check_range(length[0], (0, float('inf')), 'x-axis total_length')
        _check_range(length[1], (0, float('inf')), 'y-axis total_length')
        _check_range(length[2], (0, float('inf')), 'z-axis total_length')
        self._total_length = float(length[0]), float(length[1]), float(length[2])

    @property
    def shaft_length(self) -> Tuple[float, float, float]:  # numpydoc ignore=RT01
        """Normalized length of the shaft for each axis.

        Values must be in range ``[0, 1]``.

        Notes
        -----
        Setting this property will automatically change the :attr:`tip_length` to
        ``1 - shaft_length`` if :attr:`auto_shaft_type` is ``True``.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesAssembly()
        >>> axes_actor.shaft_length
        (0.8, 0.8, 0.8)
        >>> axes_actor.shaft_length = 0.7
        >>> axes_actor.shaft_length
        (0.7, 0.7, 0.7)
        >>> axes_actor.shaft_length = (1.0, 0.9, 0.5)
        >>> axes_actor.shaft_length
        (1.0, 0.9, 0.5)

        """
        return self._shaft_length

    @shaft_length.setter
    def shaft_length(self, length: Union[float, VectorLike[float]]):  # numpydoc ignore=GL08
        if isinstance(length, (int, float)):
            _check_range(length, (0, 1), 'shaft_length')
            length = (length, length, length)
        _check_range(length[0], (0, 1), 'x-axis shaft_length')
        _check_range(length[1], (0, 1), 'y-axis shaft_length')
        _check_range(length[2], (0, 1), 'z-axis shaft_length')
        self._shaft_length = float(length[0]), float(length[1]), float(length[2])

        if self.auto_length:
            # Calc 1-length and round to nearest 1e-8
            def calc(x):
                return round(1.0 - x, 8)

            self._tip_length = calc(length[0]), calc(length[1]), calc(length[2])

    @property
    def _true_shaft_length(self):
        shaft = self._shaft_length
        total = self._total_length
        return shaft[0] * total[0], shaft[1] * total[1], shaft[2] * total[2]

    @property
    def tip_length(self) -> Tuple[float, float, float]:  # numpydoc ignore=RT01
        """Normalized length of the tip for each axis.

        Values must be in range ``[0, 1]``.

        Notes
        -----
        Setting this property will automatically change the :attr:`shaft_length` to
        ``1 - tip_length`` if :attr:`auto_shaft_type` is ``True``.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesAssembly()
        >>> axes_actor.tip_length
        (0.2, 0.2, 0.2)
        >>> axes_actor.tip_length = 0.3
        >>> axes_actor.tip_length
        (0.3, 0.3, 0.3)
        >>> axes_actor.tip_length = (0.1, 0.4, 0.2)
        >>> axes_actor.tip_length
        (0.1, 0.4, 0.2)

        """
        return self._tip_length

    @tip_length.setter
    def tip_length(self, length: Union[float, VectorLike[float]]):  # numpydoc ignore=GL08
        if isinstance(length, (int, float)):
            _check_range(length, (0, 1), 'tip_length')
            length = (length, length, length)
        _check_range(length[0], (0, 1), 'x-axis tip_length')
        _check_range(length[1], (0, 1), 'y-axis tip_length')
        _check_range(length[2], (0, 1), 'z-axis tip_length')
        self._tip_length = float(length[0]), float(length[1]), float(length[2])

        if self.auto_length:
            # Calc 1 minus length and round to nearest 1e-8
            def calc(x):
                return round(1.0 - x, 8)

            self._shaft_length = calc(length[0]), calc(length[1]), calc(length[2])

    @property
    def _true_tip_length(self):
        tip = self._tip_length
        total = self._total_length
        return tip[0] * total[0], tip[1] * total[1], tip[2] * total[2]

    @property
    def auto_length(self) -> bool:  # numpydoc ignore=RT01
        """Automatically set shaft length when setting tip length and vice-versa.

        If ``True``:

        - Setting :attr:`shaft_length` will also set :attr:`tip_length`
          to ``1 - shaft_length``.
        - Setting :attr:`tip_length` will also set :attr:`shaft_length`
          to ``1 - tip_length``.

        Examples
        --------
        Create an axes actor with a specific shaft length.

        >>> import pyvista as pv
        >>> axes_actor = pv.AxesAssembly(
        ...     shaft_length=0.7, auto_length=True
        ... )
        >>> axes_actor.shaft_length
        (0.7, 0.7, 0.7)
        >>> axes_actor.tip_length
        (0.3, 0.3, 0.3)

        The tip lengths are adjusted dynamically.

        >>> axes_actor.tip_length = (0.1, 0.2, 0.4)
        >>> axes_actor.tip_length
        (0.1, 0.2, 0.4)
        >>> axes_actor.shaft_length
        (0.9, 0.8, 0.6)

        """
        return self._auto_length

    @auto_length.setter
    def auto_length(self, value: bool):  # numpydoc ignore=GL08
        self._auto_length = bool(value)

    @property
    def tip_radius(self) -> float:  # numpydoc ignore=RT01
        """Radius of the axes tips.

        Value must be non-negative.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesAssembly()
        >>> axes_actor.tip_radius
        0.4
        >>> axes_actor.tip_radius = 0.8
        >>> axes_actor.tip_radius
        0.8

        """
        return self._tip_radius

    @tip_radius.setter
    def tip_radius(self, radius: float):  # numpydoc ignore=GL08
        _check_range(radius, (0, float('inf')), 'tip_radius')
        self._tip_radius = radius

    @property
    def shaft_radius(self):  # numpydoc ignore=RT01
        """Cylinder radius of the axes shafts.

        Value must be non-negative.

        Notes
        -----
        Setting this property will automatically change the ``shaft_type`` to
        ``'cylinder'`` if :attr:`auto_shaft_type` is ``True``.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesAssembly()
        >>> axes_actor.shaft_radius
        0.01
        >>> axes_actor.shaft_radius = 0.03
        >>> axes_actor.shaft_radius
        0.03

        """
        return self._shaft_radius

    @shaft_radius.setter
    def shaft_radius(self, radius):  # numpydoc ignore=GL08
        _check_range(radius, (0, float('inf')), 'shaft_radius')
        self._shaft_radius = radius

    @property
    def shaft_type(self) -> str:  # numpydoc ignore=RT01
        """Tip type for all axes.

        Can be a cylinder (``0`` or ``'cylinder'``) or a line (``1`` or ``'line'``).

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesAssembly()
        >>> axes_actor.shaft_type = "line"
        >>> axes_actor.shaft_type
        'line'

        """
        return self._shaft_type

    @shaft_type.setter
    def shaft_type(self, shaft_type: str):
        self._shaft_type = self._set_geometry(part=0, geometry=shaft_type)

    @property
    def tip_type(self) -> str:  # numpydoc ignore=RT01
        """Tip type for all axes.

        Can be a cone (``0`` or ``'cone'``) or a sphere (``1`` or ``'sphere'``).

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesAssembly()
        >>> axes_actor.tip_type = 'sphere'
        >>> axes_actor.tip_type
        'sphere'

        """
        return self._tip_type

    @tip_type.setter
    def tip_type(self, tip_type: Union[str, pv.DataSet]):
        if tip_type is None:
            tip_type = pv.global_theme.axes.tip_type
        self._tip_type = self._set_geometry(part=1, geometry=tip_type)

    # @property
    # def labels(self) -> Tuple[str, str, str]:  # numpydoc ignore=RT01
    #     """Axes text labels.
    #
    #     This property can be used as an alternative to using :attr:`~x_label`,
    #     :attr:`~y_label`, and :attr:`~z_label` separately for setting or
    #     getting the axes text labels.
    #
    #     A single string with exactly three characters can be used to set the labels
    #     of the x, y, and z axes (respectively) to a single character. Alternatively.
    #     a sequence of three strings can be used.
    #
    #     Examples
    #     --------
    #     >>> import pyvista as pv
    #     >>> axes_actor = pv.AxesAssembly()
    #     >>> axes_actor.labels = 'UVW'
    #     >>> axes_actor.labels
    #     ('U', 'V', 'W')
    #     >>> axes_actor.labels = ['X Axis', 'Y Axis', 'Z Axis']
    #     >>> axes_actor.labels
    #     ('X Axis', 'Y Axis', 'Z Axis')
    #
    #     """
    #     return self.x_label, self.y_label, self.z_label
    #
    # @labels.setter
    # def labels(self, labels: Union[str, Sequence[str]]):  # numpydoc ignore=GL08
    #     self.x_label = labels[0]
    #     self.y_label = labels[1]
    #     self.z_label = labels[2]
    #     if len(labels) > 3:
    #         raise ValueError('Labels sequence must have exactly 3 items.')
    #
    # @property
    # def x_label(self) -> str:  # numpydoc ignore=RT01
    #     """Text label for the x-axis.
    #
    #     Examples
    #     --------
    #     >>> import pyvista as pv
    #     >>> axes_actor = pv.AxesAssembly()
    #     >>> axes_actor.x_label = 'This axis'
    #     >>> axes_actor.x_label
    #     'This axis'
    #
    #     """
    #     return self._label_text_getters.x()
    #
    # @x_label.setter
    # def x_label(self, label: str):  # numpydoc ignore=GL08
    #     self._label_text_setters.x(label)
    #
    # @property
    # def y_label(self) -> str:  # numpydoc ignore=RT01
    #     """Text label for the y-axis.
    #
    #     Examples
    #     --------
    #     >>> import pyvista as pv
    #     >>> axes_actor = pv.AxesAssembly()
    #     >>> axes_actor.y_label = 'This axis'
    #     >>> axes_actor.y_label
    #     'This axis'
    #
    #     """
    #     return self._label_text_getters.y()
    #
    # @y_label.setter
    # def y_label(self, label: str):  # numpydoc ignore=GL08
    #     self._label_text_setters.y(label)
    #
    # @property
    # def z_label(self) -> str:  # numpydoc ignore=RT01
    #     """Text label for the z-axis.
    #
    #     Examples
    #     --------
    #     >>> import pyvista as pv
    #     >>> axes_actor = pv.AxesAssembly()
    #     >>> axes_actor.z_label = 'This axis'
    #     >>> axes_actor.z_label
    #     'This axis'
    #
    #     """
    #     return self._label_text_getters.z()
    #
    # @z_label.setter
    # def z_label(self, label: str):  # numpydoc ignore=GL08
    #     self._label_text_setters.z(label)

    @property
    def x_shaft_prop(self) -> Property:  # numpydoc ignore=RT01
        """Return or set the property object of the x-axis shaft."""
        return self._props.x_shaft

    @x_shaft_prop.setter
    def x_shaft_prop(self, prop: Property):  # numpydoc ignore=RT01,GL08
        self._set_prop_obj(prop, axis=0, part=0)

    @property
    def y_shaft_prop(self) -> Property:  # numpydoc ignore=RT01
        """Return or set the property object of the y-axis shaft."""
        return self._props.y_shaft

    @y_shaft_prop.setter
    def y_shaft_prop(self, prop: Property):  # numpydoc ignore=RT01,GL08
        self._set_prop_obj(prop, axis=1, part=0)

    @property
    def z_shaft_prop(self) -> Property:  # numpydoc ignore=RT01
        """Return or set the property object of the z-axis shaft."""
        return self._props.z_shaft

    @z_shaft_prop.setter
    def z_shaft_prop(self, prop: Property):  # numpydoc ignore=RT01,GL08
        self._set_prop_obj(prop, axis=2, part=0)

    @property
    def x_tip_prop(self) -> Property:  # numpydoc ignore=RT01
        """Return or set the property object of the x-axis tip."""
        return self._props.x_tip

    @x_tip_prop.setter
    def x_tip_prop(self, prop: Property):  # numpydoc ignore=RT01,GL08
        self._set_prop_obj(prop, axis=0, part=1)

    @property
    def y_tip_prop(self) -> Property:  # numpydoc ignore=RT01
        """Return or set the property object of the y-axis tip."""
        return self._props.y_tip

    @y_tip_prop.setter
    def y_tip_prop(self, prop: Property):  # numpydoc ignore=RT01,GL08
        self._set_prop_obj(prop, axis=1, part=1)

    @property
    def z_tip_prop(self) -> Property:  # numpydoc ignore=RT01
        """Return or set the property object of the z-axis tip."""
        return self._props.z_tip

    @z_tip_prop.setter
    def z_tip_prop(self, prop: Property):  # numpydoc ignore=RT01,GL08
        self._set_prop_obj(prop, axis=2, part=1)

    def _set_prop_obj(self, prop: Property, axis: int, part: int):
        """Set actor property objects."""
        if not isinstance(prop, Property):
            raise TypeError(f'Prop must have type {Property}, got {type(prop)} instead.')
        _as_nested(self._actors)[axis][part].SetProperty(prop)

    def set_prop_values(self, name, value, axis='all', part='all'):
        """Set the axes shaft and tip properties.

        This is a generalized setter method which sets the value of
        a specific property for any combination of axis shaft or tip
        :class:`pyvista.Property` objects.

        Parameters
        ----------
        name : str
            Name of the property to set.

        value : Any
            Value to set the property to.

        axis : str | int, default: 'all'
            Set the property for a specific part of the axes. Specify one of:

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
        >>> axes_actor.set_prop_values('ambient', 0.7)
        >>> axes_actor.get_prop_values('ambient')
        _AxesPropTuple(x_shaft=0.7, y_shaft=0.7, z_shaft=0.7, x_tip=0.7, y_tip=0.7, z_tip=0.7)

        Set a property for the x-axis only. The property is set for
        both the axis shaft and tip by default.

        >>> axes_actor.set_prop_values('ambient', 0.3, axis='x')
        >>> axes_actor.get_prop_values('ambient')
        _AxesPropTuple(x_shaft=0.3, y_shaft=0.7, z_shaft=0.7, x_tip=0.3, y_tip=0.7, z_tip=0.7)

        Set a property for the axes tips only. The property is set for
        all axes by default.

        >>> axes_actor.set_prop_values('ambient', 0.1, part='tip')
        >>> axes_actor.get_prop_values('ambient')
        _AxesPropTuple(x_shaft=0.3, y_shaft=0.7, z_shaft=0.7, x_tip=0.1, y_tip=0.1, z_tip=0.1)

        Set a property for a single axis and specific part.

        >>> axes_actor.set_prop_values(
        ...     'ambient', 0.9, axis='z', part='shaft'
        ... )
        >>> axes_actor.get_prop_values('ambient')
        _AxesPropTuple(x_shaft=0.3, y_shaft=0.7, z_shaft=0.9, x_tip=0.1, y_tip=0.1, z_tip=0.1)

        The last example is equivalent to setting the property directly.

        >>> axes_actor.z_shaft_prop.ambient = 0.9
        >>> axes_actor.get_prop_values('ambient')
        _AxesPropTuple(x_shaft=0.3, y_shaft=0.7, z_shaft=0.9, x_tip=0.1, y_tip=0.1, z_tip=0.1)

        """
        props_dict = self._filter_prop_objects(axis=axis, part=part)
        for prop in props_dict.values():
            setattr(prop, name, value)

    def get_prop_values(self, name):
        """Get the values of a Property attribute for all axes shafts and tips.

        This is a generalized getter method which returns the value of
        a specific property for all shaft and tip :class:`pyvista.Property` objects.

        Parameters
        ----------
        name : str
            Name of the property to set.

        Returns
        -------
        _AxesTuple
            Named tuple with the requested property value for the axes shafts and tips.

        Examples
        --------
        Get the ambient property of the axes shafts and tips.

        >>> import pyvista as pv
        >>> axes_actor = pv.AxesAssembly()
        >>> axes_actor.get_prop_values('ambient')
        _AxesPropTuple(x_shaft=0.0, y_shaft=0.0, z_shaft=0.0, x_tip=0.0, y_tip=0.0, z_tip=0.0)

        """
        values = [getattr(prop, name) for prop in self._props]
        return _AxesTuple(*values)

    def _filter_prop_objects(self, axis: Union[str, int] = 'all', part: Union[str, int] = 'all'):
        valid_axis = [0, 1, 2, 'x', 'y', 'z', 'all']
        if axis not in valid_axis:
            raise ValueError(f"Axis must be one of {valid_axis}.")
        valid_part = [0, 1, 'shaft', 'tip', 'all']
        if part not in valid_part:
            raise ValueError(f"Part must be one of {valid_part}.")

        props = {}
        for num, char in enumerate(['x', 'y', 'z']):
            if axis in [num, char, 'all']:
                if part in [0, 'shaft', 'all']:
                    key = char + '_shaft'
                    props[key] = getattr(self._props, key)
                if part in [1, 'tip', 'all']:
                    key = char + '_tip'
                    props[key] = getattr(self._props, key)

        return props

    @property
    def x_color(self) -> Tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the x-axis shaft and tip."""
        return self._get_axis_color(axis=0)

    @x_color.setter
    def x_color(self, color: Union[ColorLike, Sequence[ColorLike]]):  # numpydoc ignore=GL08
        self._set_axis_color(axis=0, color=color)

    @property
    def y_color(self) -> Tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the y-axis shaft and tip."""
        return self._get_axis_color(axis=1)

    @y_color.setter
    def y_color(self, color: Union[ColorLike, Sequence[ColorLike]]):  # numpydoc ignore=GL08
        self._set_axis_color(axis=1, color=color)

    @property
    def z_color(self) -> Tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the z-axis shaft and tip."""
        return self._get_axis_color(axis=2)

    @z_color.setter
    def z_color(self, color: Union[ColorLike, Sequence[ColorLike]]):  # numpydoc ignore=GL08
        self._set_axis_color(axis=2, color=color)

    def _get_axis_color(self, axis):
        return _AxisPartTuple(self._shaft_color_getters[axis](), self._tip_color_getters[axis]())

    def _set_axis_color(self, axis, color):
        if color is None:
            if axis == 0:
                color = pv.global_theme.axes.x_color
            elif axis == 1:
                color = pv.global_theme.axes.y_color
            else:
                color = pv.global_theme.axes.z_color
            colors = [Color(color), Color(color)]
        else:
            colors = _validate_color_sequence(color, n_colors=2)
        self._shaft_color_setters[axis](colors[0])
        self._tip_color_setters[axis](colors[1])

    def plot(self):
        pl = pv.Plotter()
        pl.add_actor(self)
        pl.show()

    # @property
    # def _label_text_getters(self) -> _Tuple3D:
    #     return _Tuple3D(
    #         x=lambda: self._labels_actor.x_axis_label,
    #         y=lambda: self._labels_actor.y_axis_label,
    #         z=lambda: self._labels_actor.z_axis_label,
    #     )
    #
    # @property
    # def _label_text_setters(self) -> _Tuple3D:
    #     return _Tuple3D(
    #         x=lambda val: setattr(self._labels_actor, 'x_axis_label', val),
    #         y=lambda val: setattr(self._labels_actor, 'y_axis_label', val),
    #         z=lambda val: setattr(self._labels_actor, 'z_axis_label', val),
    #     )

    @property
    def _shaft_color_getters(self) -> _Tuple3D:
        return _Tuple3D(
            x=lambda: self._actors.x_shaft.prop.color,
            y=lambda: self._actors.y_shaft.prop.color,
            z=lambda: self._actors.z_shaft.prop.color,
        )

    @property
    def _shaft_color_setters(self) -> _Tuple3D:
        return _Tuple3D(
            x=lambda c: setattr(self._actors.x_shaft.prop, 'color', c),
            y=lambda c: setattr(self._actors.y_shaft.prop, 'color', c),
            z=lambda c: setattr(self._actors.z_shaft.prop, 'color', c),
        )

    @property
    def _tip_color_getters(self) -> _Tuple3D:
        return _Tuple3D(
            x=lambda: self._actors.x_tip.prop.color,
            y=lambda: self._actors.y_tip.prop.color,
            z=lambda: self._actors.z_tip.prop.color,
        )

    @property
    def _tip_color_setters(self) -> _Tuple3D:
        return _Tuple3D(
            x=lambda c: setattr(self._actors.x_tip.prop, 'color', c),
            y=lambda c: setattr(self._actors.y_tip.prop, 'color', c),
            z=lambda c: setattr(self._actors.z_tip.prop, 'color', c),
        )

    @property
    def show_labels(self) -> bool:  # numpydoc ignore=RT01
        """Enable or disable the text labels for the axes."""
        return self._labels_actor.show_labels

    @show_labels.setter
    def show_labels(self, value: bool):  # numpydoc ignore=GL08
        self._labels_actor.show_labels = value

    @property
    def _props(self):
        props = [actor.prop for actor in self._actors]
        return _AxesTuple(*props)

    @property
    def _datasets(self):
        datasets = [actor.mapper.dataset for actor in self._actors]
        return _AxesTuple(*datasets)

    @property
    def label_size(self):
        return self._labels_actor.label_size

    @label_size.setter
    def label_size(self, size: tuple[float, float]):
        self._labels_actor.label_size = size

    def _set_geometry(self, part: int, geometry: Union[str, pv.DataSet]):
        # resolution = self._shaft_resolution if part == 0 else self._tip_resolution
        geometry_name, datasets = AxesAssembly._make_axes_parts(geometry)
        if part == 0:
            self._actors.x_shaft.mapper.dataset = datasets.x
            self._actors.y_shaft.mapper.dataset = datasets.y
            self._actors.z_shaft.mapper.dataset = datasets.z
        elif part == 1:
            self._actors.x_tip.mapper.dataset = datasets.x
            self._actors.y_tip.mapper.dataset = datasets.y
            self._actors.z_tip.mapper.dataset = datasets.z
        else:
            raise ValueError
        return geometry_name

    def _update_geometry(self):
        shaft_radius, shaft_length = self.shaft_radius, self._true_shaft_length
        tip_radius, tip_length = (
            self.tip_radius,
            self._true_tip_length,
        )

        parts = _as_nested(self._datasets)
        for axis in range(3):
            for part_num in range(2):
                # Reset geometry
                part = AxesAssembly._normalize_part(parts[axis][part_num])

                # Offset so axis bounds are [0, 1]
                part.points[:, axis] += 0.5

                # Scale by length along axis, scale by radius off-axis
                if part_num == 0:  # shaft
                    scale = [shaft_radius] * 3
                    scale[axis] = shaft_length[axis]
                    part.scale(scale, inplace=True)
                else:  # tip
                    scale = [tip_radius] * 3
                    scale[axis] = tip_length[axis]
                    part.scale(scale, inplace=True)

                    # Move tip to end of shaft
                    part.points[:, axis] += shaft_length[axis]

    @staticmethod
    def _make_default_part(geometry: str) -> pv.PolyData:
        """Create part geometry with its length axis pointing in the +z direction."""
        resolution = 50
        if geometry == 'cylinder':
            return pv.Cylinder(direction=(0, 0, 1), resolution=resolution)
        elif geometry == 'sphere':
            return pv.Sphere(phi_resolution=resolution, theta_resolution=resolution)
        elif geometry == 'cone':
            return pv.Cone(direction=(0, 0, 1), resolution=resolution)
        elif geometry == 'pyramid':
            return pv.Pyramid().extract_surface()
        elif geometry == 'cuboid':
            return pv.Cube()
        else:
            _validation.check_contains(
                item=geometry,
                container=AxesAssembly.GEOMETRY_OPTIONS,
                name='Geometry',
            )
            raise NotImplementedError(f"Geometry '{geometry}' is not implemented")

    @staticmethod
    def _make_any_part(geometry: Union[str, pv.DataSet]):
        if isinstance(geometry, str):
            name = geometry
            part = AxesAssembly._make_default_part(
                geometry,
            )
        elif isinstance(geometry, pv.DataSet):
            name = 'custom'
            if not isinstance(geometry, pv.PolyData):
                part = geometry.cast_to_unstructured_grid().extract_surface()

        else:
            raise TypeError(
                f"Geometry must be a string, PolyData, or castable as UnstructuredGrid. Got {type(geometry)}.",
            )
        part = AxesAssembly._normalize_part(part)
        return name, part

    @staticmethod
    def _normalize_part(part: pv.PolyData) -> pv.PolyData:
        """Scale and translate part to have origin-centered bounding box with edge length one."""
        # Center points at origin
        # mypy ignore since pyvista_ndarray is not compatible with np.ndarray, see GH#5434
        part.points -= part.center  # type: ignore[misc]

        # Scale so bounding box edges have length one
        bnds = part.bounds
        axis_length = np.array((bnds[1] - bnds[0], bnds[3] - bnds[2], bnds[5] - bnds[4]))
        if np.any(axis_length < 1e-8):
            raise ValueError("Part must be 3D.")
        part.scale(np.reciprocal(axis_length), inplace=True)
        return part

    @staticmethod
    def _make_axes_parts(
        geometry: str | pv.DataSet,
        right_handed: bool = True,
    ) -> Tuple[str, _Tuple3D]:
        """Return three axis-aligned normalized parts centered at the origin."""
        name, part_z = AxesAssembly._make_any_part(geometry)
        part_x = part_z.copy().rotate_y(90)
        part_y = part_z.copy().rotate_x(-90)
        if not right_handed:
            part_z.points *= -1
        return name, _Tuple3D(part_x, part_y, part_z)


def _set_default(val, default):
    return default if val is None else val
