"""Axes actor module."""

from __future__ import annotations
from weakref import proxy

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np

import pyvista
import pyvista as pv
from pyvista.core import _validation
from pyvista.core.utilities.transformations import apply_transformation_to_points
from pyvista.plotting import _vtk
from pyvista.plotting._property import _check_range
from pyvista.plotting.colors import Color

from .colors import _validate_color_sequence
from .text3d_follower import Text3DFollower

if TYPE_CHECKING:
    from pyvista.core._typing_core import VectorLike
    from pyvista.plotting._typing import ColorLike

X_AXIS, Y_AXIS, Z_AXIS = 0, 1, 2


class AxesGeometry:
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
        tip_type='cone',
        tip_radius=0.2,
        tip_length=None,
        total_length=(1, 1, 1),
        position=(0, 0, 0),
        direction_vectors=None,
        scale=(1, 1, 1),
        user_matrix=None,
        symmetric_bounds=True,
        auto_length=True,
    ):
        super().__init__()
        # Init datasets
        self._shaft_datasets = (pv.PolyData(), pv.PolyData(), pv.PolyData())
        self._tip_datasets = (pv.PolyData(), pv.PolyData(), pv.PolyData())

        # Set shaft and tip color
        self.x_color = Color(x_color, default_color=pv.global_theme.axes.x_color)
        self.y_color = Color(y_color, default_color=pv.global_theme.axes.y_color)
        self.z_color = Color(z_color, default_color=pv.global_theme.axes.z_color)

        # Set misc flag params
        self._symmetric_bounds = symmetric_bounds
        self._auto_length = auto_length

        # Set geometry-dependent params
        self.shaft_type = shaft_type
        self.shaft_radius = shaft_radius
        self.tip_type = tip_type
        self.tip_radius = tip_radius
        self.total_length = total_length

        self.position = position
        self.direction_vectors = np.eye(3) if direction_vectors is None else direction_vectors
        self.scale = scale
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

        self._camera = None

    # def __repr__(self):
    #     """Representation of the axes actor."""
    #     matrix_not_set = self.user_matrix is None or np.array_equal(self.user_matrix, np.eye(4))
    #     mat_info = 'Identity' if matrix_not_set else 'Set'
    #     bnds = self.bounds
    #
    #     attr = [
    #         f"{type(self).__name__} ({hex(id(self))})",
    #         # f"  X label:                    '{self.x_label}'",
    #         # f"  Y label:                    '{self.y_label}'",
    #         # f"  Z label:                    '{self.z_label}'",
    #         # f"  Show labels:                {self.show_labels}",
    #         # f"  Label position:             {self.label_position}",
    #         f"  Shaft type:                 '{self.shaft_type}'",
    #         f"  Shaft radius:               {self.shaft_radius}",
    #         f"  Shaft length:               {self.shaft_length}",
    #         f"  Tip type:                   '{self.tip_type}'",
    #         f"  Tip radius:                 {self.tip_radius}",
    #         f"  Tip length:                 {self.tip_length}",
    #         f"  Total length:               {self.total_length}",
    #         f"  Position:                   {self.position}",
    #         f"  Scale:                      {self.scale}",
    #         f"  User matrix:                {mat_info}",
    #         f"  Visible:                    {self.visibility}",
    #         f"  X Bounds                    {bnds[0]:.3E}, {bnds[1]:.3E}",
    #         f"  Y Bounds                    {bnds[2]:.3E}, {bnds[3]:.3E}",
    #         f"  Z Bounds                    {bnds[4]:.3E}, {bnds[5]:.3E}",
    #     ]
    #     return '\n'.join(attr)

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

    # @property
    # def visibility(self) -> bool:  # numpydoc ignore=RT01
    #     """Enable or disable the visibility of the axes.
    #
    #     Examples
    #     --------
    #     Create an AxesAssembly and check its visibility
    #
    #     >>> import pyvista as pv
    #     >>> axes_actor = pv.AxesAssembly()
    #     >>> axes_actor.visibility
    #     True
    #
    #     """
    #     return bool(self.GetVisibility())
    #
    # @visibility.setter
    # def visibility(self, value: bool):  # numpydoc ignore=GL08
    #     self.SetVisibility(value)

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
        return tuple(self._total_length)

    @total_length.setter
    def total_length(self, length: float | VectorLike[float]):  # numpydoc ignore=GL08
        self._total_length = _validation.validate_array3(
            length,
            broadcast=True,
            must_be_in_range=[0, np.inf],
            name='Total length',
        )

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
    # def x_shaft_prop(self) -> Property:  # numpydoc ignore=RT01
    #     """Return or set the property object of the x-axis shaft."""
    #     return self._props.x_shaft
    #
    # @x_shaft_prop.setter
    # def x_shaft_prop(self, prop: Property):  # numpydoc ignore=RT01,GL08
    #     self._set_prop_obj(prop, axis=0, part=0)
    #
    # @property
    # def y_shaft_prop(self) -> Property:  # numpydoc ignore=RT01
    #     """Return or set the property object of the y-axis shaft."""
    #     return self._props.y_shaft
    #
    # @y_shaft_prop.setter
    # def y_shaft_prop(self, prop: Property):  # numpydoc ignore=RT01,GL08
    #     self._set_prop_obj(prop, axis=1, part=0)
    #
    # @property
    # def z_shaft_prop(self) -> Property:  # numpydoc ignore=RT01
    #     """Return or set the property object of the z-axis shaft."""
    #     return self._props.z_shaft
    #
    # @z_shaft_prop.setter
    # def z_shaft_prop(self, prop: Property):  # numpydoc ignore=RT01,GL08
    #     self._set_prop_obj(prop, axis=2, part=0)
    #
    # @property
    # def x_tip_prop(self) -> Property:  # numpydoc ignore=RT01
    #     """Return or set the property object of the x-axis tip."""
    #     return self._props.x_tip
    #
    # @x_tip_prop.setter
    # def x_tip_prop(self, prop: Property):  # numpydoc ignore=RT01,GL08
    #     self._set_prop_obj(prop, axis=0, part=1)
    #
    # @property
    # def y_tip_prop(self) -> Property:  # numpydoc ignore=RT01
    #     """Return or set the property object of the y-axis tip."""
    #     return self._props.y_tip
    #
    # @y_tip_prop.setter
    # def y_tip_prop(self, prop: Property):  # numpydoc ignore=RT01,GL08
    #     self._set_prop_obj(prop, axis=1, part=1)
    #
    # @property
    # def z_tip_prop(self) -> Property:  # numpydoc ignore=RT01
    #     """Return or set the property object of the z-axis tip."""
    #     return self._props.z_tip
    #
    # @z_tip_prop.setter
    # def z_tip_prop(self, prop: Property):  # numpydoc ignore=RT01,GL08
    #     self._set_prop_obj(prop, axis=2, part=1)
    #
    # def _set_prop_obj(self, prop: Property, axis: int, part: int):
    #     """Set actor property objects."""
    #     if not isinstance(prop, Property):
    #         raise TypeError(f'Prop must have type {Property}, got {type(prop)} instead.')
    #     _as_nested(self._actors)[axis][part].SetProperty(prop)
    #
    # def set_prop_values(self, name, value, axis='all', part='all'):
    #     """Set the axes shaft and tip properties.
    #
    #     This is a generalized setter method which sets the value of
    #     a specific property for any combination of axis shaft or tip
    #     :class:`pyvista.Property` objects.
    #
    #     Parameters
    #     ----------
    #     name : str
    #         Name of the property to set.
    #
    #     value : Any
    #         Value to set the property to.
    #
    #     axis : str | int, default: 'all'
    #         Set the property for a specific part of the axes. Specify one of:
    #
    #         - ``'x'`` or ``0``: only set the property for the x-axis.
    #         - ``'y'`` or ``1``: only set the property for the y-axis.
    #         - ``'z'`` or ``2``: only set the property for the z-axis.
    #         - ``'all'``: set the property for all three axes.
    #
    #     part : str | int, default: 'all'
    #         Set the property for a specific part of the axes. Specify one of:
    #
    #         - ``'shaft'`` or ``0``: only set the property for the axes shafts.
    #         - ``'tip'`` or ``1``: only set the property for the axes tips.
    #         - ``'all'``: set the property for axes shafts and tips.
    #
    #     Examples
    #     --------
    #     Set the ambient property for all axes shafts and tips.
    #
    #     >>> import pyvista as pv
    #     >>> axes_actor = pv.AxesAssembly()
    #     >>> axes_actor.set_prop_values('ambient', 0.7)
    #     >>> axes_actor.get_prop_values('ambient')
    #     _AxesPropTuple(x_shaft=0.7, y_shaft=0.7, z_shaft=0.7, x_tip=0.7, y_tip=0.7, z_tip=0.7)
    #
    #     Set a property for the x-axis only. The property is set for
    #     both the axis shaft and tip by default.
    #
    #     >>> axes_actor.set_prop_values('ambient', 0.3, axis='x')
    #     >>> axes_actor.get_prop_values('ambient')
    #     _AxesPropTuple(x_shaft=0.3, y_shaft=0.7, z_shaft=0.7, x_tip=0.3, y_tip=0.7, z_tip=0.7)
    #
    #     Set a property for the axes tips only. The property is set for
    #     all axes by default.
    #
    #     >>> axes_actor.set_prop_values('ambient', 0.1, part='tip')
    #     >>> axes_actor.get_prop_values('ambient')
    #     _AxesPropTuple(x_shaft=0.3, y_shaft=0.7, z_shaft=0.7, x_tip=0.1, y_tip=0.1, z_tip=0.1)
    #
    #     Set a property for a single axis and specific part.
    #
    #     >>> axes_actor.set_prop_values(
    #     ...     'ambient', 0.9, axis='z', part='shaft'
    #     ... )
    #     >>> axes_actor.get_prop_values('ambient')
    #     _AxesPropTuple(x_shaft=0.3, y_shaft=0.7, z_shaft=0.9, x_tip=0.1, y_tip=0.1, z_tip=0.1)
    #
    #     The last example is equivalent to setting the property directly.
    #
    #     >>> axes_actor.z_shaft_prop.ambient = 0.9
    #     >>> axes_actor.get_prop_values('ambient')
    #     _AxesPropTuple(x_shaft=0.3, y_shaft=0.7, z_shaft=0.9, x_tip=0.1, y_tip=0.1, z_tip=0.1)
    #
    #     """
    #     props_dict = self._filter_prop_objects(axis=axis, part=part)
    #     for prop in props_dict.values():
    #         setattr(prop, name, value)
    #
    # def get_prop_values(self, name):
    #     """Get the values of a Property attribute for all axes shafts and tips.
    #
    #     This is a generalized getter method which returns the value of
    #     a specific property for all shaft and tip :class:`pyvista.Property` objects.
    #
    #     Parameters
    #     ----------
    #     name : str
    #         Name of the property to set.
    #
    #     Returns
    #     -------
    #     _AxesTuple
    #         Named tuple with the requested property value for the axes shafts and tips.
    #
    #     Examples
    #     --------
    #     Get the ambient property of the axes shafts and tips.
    #
    #     >>> import pyvista as pv
    #     >>> axes_actor = pv.AxesAssembly()
    #     >>> axes_actor.get_prop_values('ambient')
    #     _AxesPropTuple(x_shaft=0.0, y_shaft=0.0, z_shaft=0.0, x_tip=0.0, y_tip=0.0, z_tip=0.0)
    #
    #     """
    #     values = [getattr(prop, name) for prop in self._props]
    #     return _AxesTuple(*values)
    #
    # def _filter_prop_objects(self, axis: Union[str, int] = 'all', part: Union[str, int] = 'all'):
    #     valid_axis = [0, 1, 2, 'x', 'y', 'z', 'all']
    #     if axis not in valid_axis:
    #         raise ValueError(f"Axis must be one of {valid_axis}.")
    #     valid_part = [0, 1, 'shaft', 'tip', 'all']
    #     if part not in valid_part:
    #         raise ValueError(f"Part must be one of {valid_part}.")
    #
    #     props = {}
    #     for num, char in enumerate(['x', 'y', 'z']):
    #         if axis in [num, char, 'all']:
    #             if part in [0, 'shaft', 'all']:
    #                 key = char + '_shaft'
    #                 props[key] = getattr(self._props, key)
    #             if part in [1, 'tip', 'all']:
    #                 key = char + '_tip'
    #                 props[key] = getattr(self._props, key)
    #
    #     return props

    @property
    def x_color(self) -> tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the x-axis shaft and tip."""
        return self._x_color  # type: ignore[return-value]

    @x_color.setter
    def x_color(self, color: ColorLike | Sequence[ColorLike]):  # numpydoc ignore=GL08
        self._x_color = _validate_color_sequence(color, n_colors=2)
        # self._update_x_color() if self._is_init else None

    def _update_x_color(self):
        shaft_color, tip_color = self.x_color
        AxesAssembly._set_rgb_array(self._shaft_datasets[X_AXIS], shaft_color)
        AxesAssembly._set_rgb_array(self._tip_datasets[X_AXIS], tip_color)

    @property
    def y_color(self) -> tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the y-axis shaft and tip."""
        return self._y_color  # type: ignore[return-value]

    @y_color.setter
    def y_color(self, color: ColorLike | Sequence[ColorLike]):  # numpydoc ignore=GL08
        self._y_color = _validate_color_sequence(color, n_colors=2)
        # self._update_x_color()

    def _update_y_color(self):
        shaft_color, tip_color = self.y_color
        AxesAssembly._set_rgb_array(self._shaft_datasets[Y_AXIS], shaft_color)
        AxesAssembly._set_rgb_array(self._tip_datasets[Y_AXIS], tip_color)

    @property
    def z_color(self) -> tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the z-axis shaft and tip."""
        return self._z_color  # type: ignore[return-value]

    @z_color.setter
    def z_color(self, color: ColorLike | Sequence[ColorLike]):  # numpydoc ignore=GL08
        self._z_color = _validate_color_sequence(color, n_colors=2)
        # self._update_z_color()

    def _update_z_color(self):
        shaft_color, tip_color = self.z_color
        AxesAssembly._set_rgb_array(self._shaft_datasets[Z_AXIS], shaft_color)
        AxesAssembly._set_rgb_array(self._tip_datasets[Z_AXIS], tip_color)

    @property
    def position(self) -> tuple[float, float, float]:
        return tuple(self._position)

    @position.setter
    def position(self, xyz):
        self._position = _validation.validate_array3(xyz)

    @property
    def scale(self) -> tuple[float, float, float]:
        return tuple(self._scale)

    @scale.setter
    def scale(self, xyz):
        self._scale = _validation.validate_array3(xyz, broadcast=True, name='Scale')

    @property
    def direction_vectors(self):
        return self._direction_vectors

    @direction_vectors.setter
    def direction_vectors(self, vectors):
        self._direction_vectors = vectors

    @property
    def transformation_matrix(self):
        # Scale proportional to axis length
        scale_matrix = np.eye(4)
        scale_matrix[:3, :3] = np.diag(self._total_length)

        position_matrix = np.eye(4)
        position_matrix[:3, 3] = self.position

        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = self.direction_vectors

        # Scale first, then rotate, then move
        return position_matrix @ rotation_matrix @ scale_matrix

    def plot(self):
        pl = pv.Plotter()
        pl.add_mesh(self)
        pl.show()

    def _set_geometry(self, part: int, geometry: Union[str, pv.DataSet]):
        # resolution = self._shaft_resolution if part == 0 else self._tip_resolution
        geometry_name, new_datasets = AxesAssembly._make_axes_parts(geometry)
        assert part in [0, 1]
        datasets = self._shaft_datasets if part == 0 else self._tip_datasets
        datasets[X_AXIS].copy_from(new_datasets[X_AXIS])
        datasets[Y_AXIS].copy_from(new_datasets[Y_AXIS])
        datasets[Z_AXIS].copy_from(new_datasets[Z_AXIS])
        return geometry_name

    # def _apply_axes_colors(self):
    #     x_shaft, y_shaft, z_shaft = self._datasets['shafts']
    #     x_tip, y_tip, z_tip = self._datasets['tips']
    #     x_color, y_color, z_color = self.x_color, self.y_color, self.z_color
    #
    #     AxesAssembly._set_rgb_array(x_shaft, x_color[0])
    #     AxesAssembly._set_rgb_array(x_tip, x_color[1])
    #     AxesAssembly._set_rgb_array(y_shaft, y_color[0])
    #     AxesAssembly._set_rgb_array(y_tip, y_color[1])
    #     AxesAssembly._set_rgb_array(z_shaft, z_color[0])
    #     AxesAssembly._set_rgb_array(z_tip, z_color[1])

    # TODO: refactor as _reset_geometry and make child class override method
    def _reset_shaft_and_tip_geometry(self):
        shaft_radius, shaft_length = self.shaft_radius, self._true_shaft_length
        tip_radius, tip_length = (
            self.tip_radius,
            self._true_tip_length,
        )

        nested_datasets = [self._shaft_datasets, self._tip_datasets]
        SHAFT, TIP = 0, 1
        for part_type in [SHAFT, TIP]:
            for axis in X_AXIS, Y_AXIS, Z_AXIS:
                # Reset geometry
                part = AxesAssembly._normalize_part(nested_datasets[part_type][axis])

                # Offset so axis bounds are [0, 1]
                part.points[:, axis] += 0.5

                # Scale by length along axis, scale by radius off-axis
                if part_type == SHAFT:
                    scale = [shaft_radius] * 3
                    scale[axis] = shaft_length[axis]
                    part.scale(scale, inplace=True)
                else:  # tip
                    scale = [tip_radius] * 3
                    scale[axis] = tip_length[axis]
                    part.scale(scale, inplace=True)

                    # Move tip to end of shaft
                    part.points[:, axis] += shaft_length[axis]

    def _transform_shafts_and_tips(self):
        for dataset in [*self._shaft_datasets, *self._tip_datasets]:
            dataset.transform(self.transformation_matrix, inplace=True)

    def _update_axes_shaft_and_tip_geometry(self):
        self._reset_shaft_and_tip_geometry()
        self._transform_shafts_and_tips()

    def _update_colors(self):
        self._update_x_color()
        self._update_y_color()
        self._update_z_color()

    @property
    def output(self):
        self._update_axes_shaft_and_tip_geometry()
        self._update_colors()
        # TODO: Return MultiBlock of meshes without merging once RGB MultiBlock
        #  plotting is fixed (pyvista #6012)
        return pv.merge((*self._shaft_datasets, *self._tip_datasets))

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
    def _make_any_part(geometry: Union[str, pv.DataSet]) -> tuple[str, pv.PolyData]:
        if isinstance(geometry, str):
            name = geometry
            part = AxesAssembly._make_default_part(
                geometry,
            )
        elif isinstance(geometry, pv.DataSet):
            name = 'custom'
            if not isinstance(geometry, pv.PolyData):
                part = geometry.extract_geometry()

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
    ) -> tuple[str, tuple[pv.PolyData, pv.PolyData, pv.PolyData]]:
        """Return three axis-aligned normalized parts centered at the origin."""
        name, part_z = AxesAssembly._make_any_part(geometry)
        part_x = part_z.copy().rotate_y(90)
        part_y = part_z.copy().rotate_x(-90)
        if not right_handed:
            part_z.points *= -1  # type: ignore[misc]
        return name, (part_x, part_y, part_z)


class AxesAssembly(AxesGeometry, _vtk.vtkPropAssembly):
    def __init__(
        self,
        x_label=None,
        y_label=None,
        z_label=None,
        labels=None,
        label_color='white',
        include_labels=True,
        label_position=1.1,
        label_size=0.1,
        label_border=True,
        x_color=None,
        y_color=None,
        z_color=None,
        **kwargs,
    ):
        # Init

        self._actors_shafts = [pv.Actor(), pv.Actor(), pv.Actor()]
        self._actors_tips = [pv.Actor(), pv.Actor(), pv.Actor()]
        actors = (*self._actors_shafts, *self._actors_tips)
        [self.AddPart(actor) for actor in actors]

        super().__init__(x_color, y_color, z_color, **kwargs)

        datasets = (*self._shaft_datasets, *self._tip_datasets)
        [
            setattr(actor, 'mapper', pv.DataSetMapper(dataset=dataset))
            for actor, dataset in zip(actors, datasets)
        ]

        self._is_init = False

        # Init label datasets and actors

        self._x_follower = (
            Text3DFollower()
        )  # AxesAssembly._create_label_follower(dataset=self._label_sources[0].output)
        self._y_follower = (
            Text3DFollower()
        )  # AxesAssembly._create_label_follower(dataset=self._label_sources[1].output)
        self._z_follower = Text3DFollower()
        # NOTE: Adding the followers to the assembly does *not* work
        # Instead, the followers must be added to a plot separately
        # self.AddPart(self._x_follower)
        # self.AddPart(self._y_follower)
        # self.AddPart(self._z_follower)

        # dataset = self.output
        # mapper = AxesAssembly._create_rgb_mapper(dataset)
        # return Actor(mapper=mapper)

        # Set text labels
        if labels is None:
            self.x_label = _set_default(x_label, 'X')
            self.y_label = _set_default(y_label, 'Y')
            self.z_label = _set_default(z_label, 'Z')
        else:
            msg = "Cannot initialize '{}' and 'labels' properties together. Specify one or the other, not both."
            if x_label is not None:
                raise ValueError(msg.format('x_label'))
            if y_label is not None:
                raise ValueError(msg.format('y_label'))
            if z_label is not None:
                raise ValueError(msg.format('z_label'))
            self.labels = labels
        self.include_labels = include_labels
        self.label_color = label_color
        self.label_size = label_size
        self.label_position = label_position
        self._is_init = True
        self._update_axes_shaft_and_tip_geometry()
        self._update_label_positions()

    @property
    def labels(self) -> Tuple[str, str, str]:  # numpydoc ignore=RT01
        """Axes text labels.

        This property can be used as an alternative to using :attr:`~x_label`,
        :attr:`~y_label`, and :attr:`~z_label` separately for setting or
        getting the axes text labels.

        A single string with exactly three characters can be used to set the labels
        of the x, y, and z axes (respectively) to a single character. Alternatively.
        a sequence of three strings can be used.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesAssembly()
        >>> axes_actor.labels = 'UVW'
        >>> axes_actor.labels
        ('U', 'V', 'W')
        >>> axes_actor.labels = ['X Axis', 'Y Axis', 'Z Axis']
        >>> axes_actor.labels
        ('X Axis', 'Y Axis', 'Z Axis')

        """
        return self.x_label, self.y_label, self.z_label

    @labels.setter
    def labels(self, labels: Sequence[str]):  # numpydoc ignore=GL08
        _validation.check_iterable_items(labels, str)
        _validation.check_length(labels, exact_length=3)
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
        return self._x_follower.string

    @x_label.setter
    def x_label(self, label: str):  # numpydoc ignore=GL08
        self._x_follower.string = label

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
        return self._y_follower.string

    @y_label.setter
    def y_label(self, label: str):  # numpydoc ignore=GL08
        self._y_follower.string = label

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
        return self._z_follower.string

    @z_label.setter
    def z_label(self, label: str):  # numpydoc ignore=GL08
        self._z_follower.string = label

    @property
    def include_labels(self) -> bool:  # numpydoc ignore=RT01
        """Enable or disable the text labels for the axes."""
        return self._include_labels

    @include_labels.setter
    def include_labels(self, value: bool):  # numpydoc ignore=GL08
        self._include_labels = value

    @property
    def label_size(self):
        return self._label_size

    @label_size.setter
    def label_size(self, size: float):
        self._label_size = _validation.validate_number(size)
        # Scale text height proportional to norm of axes lengths
        height = np.linalg.norm(self.total_length) * size
        self._x_follower.height = height
        self._y_follower.height = height
        self._z_follower.height = height
        # x_label = pyvista.Text3D(self.x_label, height=true_size, depth=0.0)
        # y_label = pyvista.Text3D(self.y_label, height=true_size, depth=0.0)
        # z_label = pyvista.Text3D(self.z_label, height=true_size, depth=0.0)

    @property
    def label_position(self) -> Tuple[float, float, float]:  # numpydoc ignore=RT01
        """Normalized position of the text label along each axis.

        Values must be non-negative.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesAssembly()
        >>> axes_actor.label_position
        (1.0, 1.0, 1.0)
        >>> axes_actor.label_position = 0.3
        >>> axes_actor.label_position
        (0.3, 0.3, 0.3)
        >>> axes_actor.label_position = (0.1, 0.4, 0.2)
        >>> axes_actor.label_position
        (0.1, 0.4, 0.2)

        """
        return tuple(self._label_position)

    @label_position.setter
    def label_position(self, position: Union[float, VectorLike[float]]):  # numpydoc ignore=GL08
        self._label_position = _validation.validate_array3(
            position,
            broadcast=True,
            must_be_in_range=[0, np.inf],
            name='Label position',
        )
        self._update_label_positions() if self._is_init else None

    #
    # @property
    # def tip_position(self):
    #     tips = np.diag(np.array(self.total_length) * self.scale)
    #     return apply_transformation_to_points(self.transformation_matrix, tips)

    @property
    def label_color(self):
        return self._label_color

    @label_color.setter
    def label_color(self, color: ColorLike | Sequence[ColorLike]):
        self._label_color = _validate_color_sequence(color, n_colors=3)

    @property
    def x_color(self) -> Tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the x-axis shaft and tip."""
        return (self._actors_shafts[X_AXIS].prop.color, self._actors_tips[X_AXIS].prop.color)

    @x_color.setter
    def x_color(self, color: Union[ColorLike, Sequence[ColorLike]]):  # numpydoc ignore=GL08
        shaft_color, tip_color = _validate_color_sequence(color, n_colors=2)
        self._actors_shafts[X_AXIS].prop.color = shaft_color
        self._actors_tips[X_AXIS].prop.color = tip_color

    @property
    def y_color(self) -> Tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the y-axis shaft and tip."""
        return (self._actors_shafts[Y_AXIS].prop.color, self._actors_tips[Y_AXIS].prop.color)

    @y_color.setter
    def y_color(self, color: Union[ColorLike, Sequence[ColorLike]]):  # numpydoc ignore=GL08
        shaft_color, tip_color = _validate_color_sequence(color, n_colors=2)
        self._actors_shafts[Y_AXIS].prop.color = shaft_color
        self._actors_tips[Y_AXIS].prop.color = tip_color

    @property
    def z_color(self) -> Tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the z-axis shaft and tip."""
        return (self._actors_shafts[Z_AXIS].prop.color, self._actors_tips[Z_AXIS].prop.color)

    @z_color.setter
    def z_color(self, color: Union[ColorLike, Sequence[ColorLike]]):  # numpydoc ignore=GL08
        shaft_color, tip_color = _validate_color_sequence(color, n_colors=2)
        self._actors_shafts[Z_AXIS].prop.color = shaft_color
        self._actors_tips[Z_AXIS].prop.color = tip_color

    def set_prop(self, name, value, axis='all', part='all'):
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
        >>> axes_actor = pv.AxesActor()
        >>> axes_actor.set_prop('ambient', 0.7)
        >>> axes_actor.get_prop('ambient')
        _AxesPropTuple(x_shaft=0.7, y_shaft=0.7, z_shaft=0.7, x_tip=0.7, y_tip=0.7, z_tip=0.7)
        Set a property for the x-axis only. The property is set for
        both the axis shaft and tip by default.
        >>> axes_actor.set_prop('ambient', 0.3, axis='x')
        >>> axes_actor.get_prop('ambient')
        _AxesPropTuple(x_shaft=0.3, y_shaft=0.7, z_shaft=0.7, x_tip=0.3, y_tip=0.7, z_tip=0.7)
        Set a property for the axes tips only. The property is set for
        all axes by default.
        >>> axes_actor.set_prop('ambient', 0.1, part='tip')
        >>> axes_actor.get_prop('ambient')
        _AxesPropTuple(x_shaft=0.3, y_shaft=0.7, z_shaft=0.7, x_tip=0.1, y_tip=0.1, z_tip=0.1)
        Set a property for a single axis and specific part.
        >>> axes_actor.set_prop('ambient', 0.9, axis='z', part='shaft')
        >>> axes_actor.get_prop('ambient')
        _AxesPropTuple(x_shaft=0.3, y_shaft=0.7, z_shaft=0.9, x_tip=0.1, y_tip=0.1, z_tip=0.1)
        The last example is equivalent to setting the property directly.
        >>> axes_actor.z_shaft_prop.ambient = 0.9
        >>> axes_actor.get_prop('ambient')
        _AxesPropTuple(x_shaft=0.3, y_shaft=0.7, z_shaft=0.9, x_tip=0.1, y_tip=0.1, z_tip=0.1)
        """
        props_dict = self._filter_prop_objects(axis=axis, part=part)
        for prop in props_dict.values():
            setattr(prop, name, value)

    def get_prop(self, name):
        """Get the axes shaft and tip properties.
        This is a generalized getter method which returns the value of
        a specific property for all shaft and tip :class:`pyvista.Property` objects.
        Parameters
        ----------
        name : str
            Name of the property to set.
        Returns
        -------
        _AxesPropTuple
            Named tuple with the requested property value for the axes shafts and tips.
        Examples
        --------
        Get the ambient property of the axes shafts and tips.
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesActor()
        >>> axes_actor.get_prop('ambient')
        _AxesPropTuple(x_shaft=0.0, y_shaft=0.0, z_shaft=0.0, x_tip=0.0, y_tip=0.0, z_tip=0.0)
        """
        props = [getattr(prop, name) for prop in self._actor_properties]
        return tuple(props)

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
                    props[key] = getattr(self._actor_properties, key)
                if part in [1, 'tip', 'all']:
                    key = char + '_tip'
                    props[key] = getattr(self._actor_properties, key)

        return props

    # def _get_axis_color(self, axis):
    #     return (self._shaft_color_getters[axis](),
    #                           self._tip_color_getters[axis]())

    # def _set_axis_color(self, axis, color):
    #     if color is None:
    #         if axis == 0:
    #             color = pv.global_theme.axes.x_color
    #         elif axis == 1:
    #             color = pv.global_theme.axes.y_color
    #         else:
    #             color = pv.global_theme.axes.z_color
    #         colors = [Color(color)] * 2
    #     else:
    #         colors = _validate_color_sequence(color, num_colors=2)
    #     self._shaft_color_setters[axis](colors[0])
    #     self._tip_color_setters[axis](colors[1])

    # @property
    # def camera(self):
    #     return self._camera
    # @camera.setter
    # def camera(self, camera):
    #     self._camera = camera
    #     self._x_follower.camera=camera
    #     self._y_follower.camera=camera
    #     self._z_follower.camera=camera

    def _get_transformed_label_positions(self):
        # Initial position vectors
        points = np.diag(self.label_position)
        matrix = self.transformation_matrix
        return apply_transformation_to_points(matrix, points)

    # def _transform_labels(self):
    #     x_pos, y_pos, z_pos = self._get_transformed_label_positions()
    #     x_label, y_label, z_label = self._datasets['labels']
    #
    #     # Face +x, with +z up
    #     x_label.rotate_y(90, inplace=True)
    #     x_label.rotate_x(90, inplace=True)
    #     x_label.translate(x_pos, inplace=True)
    #
    #     # Face +y, with +z up
    #     y_label.rotate_z(180, inplace=True)
    #     y_label.rotate_x(-90, inplace=True)
    #     y_label.translate(y_pos, inplace=True)
    #
    #     # Face +z, with (-0.5, -0.5, 0) up
    #     z_label.rotate_z(135, inplace=True)
    #     z_label.translate(z_pos, inplace=True)
    #
    #     for block in self._datasets['labels']:
    #         if block is not None:
    #             block.transform(self.transformation_matrix, inplace=True)

    def _apply_label_colors(self):
        ...
        # x_label, y_label, z_label = self._datasets['labels']
        # x_color, y_color, z_color = self.label_color
        # AxesAssembly._set_rgb_array(x_label, x_color)
        # AxesAssembly._set_rgb_array(y_label, y_color)
        # AxesAssembly._set_rgb_array(z_label, z_color)

    # def _reset_label_geometry(self):
    #     # Scale label size proportional to norm of axes lengths
    #     size = np.linalg.norm(self.total_length) * self.label_size
    #
    #     x_label = pyvista.Text3D(self.x_label, height=size, depth=0.0)
    #     y_label = pyvista.Text3D(self.y_label, height=size, depth=0.0)
    #     z_label = pyvista.Text3D(self.z_label, height=size, depth=0.0)
    #
    #     blocks = self._datasets['labels']
    #     blocks['x'] = x_label
    #     blocks['y'] = y_label
    #     blocks['z'] = z_label

    def _update_label_positions(self):
        # self._apply_label_colors()
        x_pos, y_pos, z_pos = self._get_transformed_label_positions()
        self._x_follower.position = x_pos
        self._y_follower.position = y_pos
        self._z_follower.position = z_pos

    # @property
    # def output_label_dataset(self):
    #     self._reset_label_geometry()
    #     self._apply_label_colors()
    #     self._transform_labels()
    #
    #     labels = self._datasets['labels']
    #     return labels['x'], labels['y'], labels['z']

    @staticmethod
    def _set_rgb_array(dataset: pv.PolyData, color: Color):
        array = np.broadcast_to(color.int_rgb, (dataset.n_cells, 3)).copy()
        dataset.cell_data['_rgb'] = array

    @staticmethod
    def _create_rgb_mapper(dataset: pv.DataSet):
        mapper = pyvista.DataSetMapper(dataset=dataset)
        mapper.set_scalars(
            scalars=dataset.active_scalars,
            scalars_name=dataset.active_scalars_name,
            rgb=True,
        )
        return mapper

    @staticmethod
    def _create_label_follower(dataset, position=(0, 0, 0)):
        mapper = AxesAssembly._create_rgb_mapper(dataset)
        follower = Text3DFollower(mapper=mapper)
        follower.position = position
        follower.prop.lighting = False
        return follower


def _set_default(val, default):
    return default if val is None else val
