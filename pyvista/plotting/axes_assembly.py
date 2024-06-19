"""Axes actor module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np

import pyvista
import pyvista as pv
from pyvista.core import _validation
from pyvista.core.utilities.geometric_sources import AxesGeometrySource
from pyvista.core.utilities.geometric_sources import _AxisEnum
from pyvista.core.utilities.transformations import apply_transformation_to_points
from pyvista.plotting import _vtk

from .colors import _validate_color_sequence
from .text3d_follower import Text3DFollower

if TYPE_CHECKING:
    from pyvista.core._typing_core import VectorLike
    from pyvista.plotting._typing import ColorLike
    from pyvista.plotting.colors import Color


class AxesAssembly(_vtk.vtkPropAssembly):
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
        symmetric_bounds=False,
        **kwargs,
    ):
        super().__init__()

        self._actors_shafts = [pv.Actor(), pv.Actor(), pv.Actor()]
        self._actors_tips = [pv.Actor(), pv.Actor(), pv.Actor()]
        actors = (*self._actors_shafts, *self._actors_tips)
        [self.AddPart(actor) for actor in actors]

        if x_color is None:
            x_color = pv.global_theme.axes.x_color
        if y_color is None:
            y_color = pv.global_theme.axes.y_color
        if z_color is None:
            z_color = pv.global_theme.axes.z_color

        self.x_color = x_color
        self.y_color = y_color
        self.z_color = z_color

        axes_geometry = AxesGeometrySource(**kwargs, rgb_scalars=False)
        self._axes_geometry = axes_geometry

        datasets = (*axes_geometry._shaft_datasets, *axes_geometry._tip_datasets)
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
        self._update()

    def __getattr__(self, item: str):
        try:
            return self.__getattribute__(item)
        except AttributeError as e:
            if not item.startswith('_'):
                try:
                    return getattr(self._axes_geometry, item)
                except AttributeError:
                    raise AttributeError(str(e)) from e

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
        return (
            self._actors_shafts[_AxisEnum.x].prop.color,
            self._actors_tips[_AxisEnum.x].prop.color,
        )

    @x_color.setter
    def x_color(self, color: Union[ColorLike, Sequence[ColorLike]]):  # numpydoc ignore=GL08
        shaft_color, tip_color = _validate_color_sequence(color, n_colors=2)
        self._actors_shafts[_AxisEnum.x].prop.color = shaft_color
        self._actors_tips[_AxisEnum.x].prop.color = tip_color

    @property
    def y_color(self) -> Tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the y-axis shaft and tip."""
        return (
            self._actors_shafts[_AxisEnum.y].prop.color,
            self._actors_tips[_AxisEnum.y].prop.color,
        )

    @y_color.setter
    def y_color(self, color: Union[ColorLike, Sequence[ColorLike]]):  # numpydoc ignore=GL08
        shaft_color, tip_color = _validate_color_sequence(color, n_colors=2)
        self._actors_shafts[_AxisEnum.y].prop.color = shaft_color
        self._actors_tips[_AxisEnum.y].prop.color = tip_color

    @property
    def z_color(self) -> Tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the z-axis shaft and tip."""
        return (
            self._actors_shafts[_AxisEnum.z].prop.color,
            self._actors_tips[_AxisEnum.z].prop.color,
        )

    @z_color.setter
    def z_color(self, color: Union[ColorLike, Sequence[ColorLike]]):  # numpydoc ignore=GL08
        shaft_color, tip_color = _validate_color_sequence(color, n_colors=2)
        self._actors_shafts[_AxisEnum.z].prop.color = shaft_color
        self._actors_tips[_AxisEnum.z].prop.color = tip_color

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
        matrix = self._axes_geometry.transformation_matrix
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

    # def _update_axes_geometry(self):
    #     self._axes_geometry._update_axes_shaft_and_tip_geometry()

    def _update(self):
        self._axes_geometry.update()
        self._update_label_positions()

    # @property
    # def output_label_dataset(self):
    #     self._reset_label_geometry()
    #     self._apply_label_colors()
    #     self._transform_labels()
    #
    #     labels = self._datasets['labels']
    #     return labels['x'], labels['y'], labels['z']

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
