"""Axes actor module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Sequence
from typing import Tuple
from typing import TypedDict
from typing import Union

import numpy as np

import pyvista as pv
from pyvista.core import _validation
from pyvista.core.utilities.geometric_sources import AxesGeometrySource
from pyvista.core.utilities.geometric_sources import _AxisEnum
from pyvista.core.utilities.transformations import apply_transformation_to_points
from pyvista.plotting.actor import Actor
from pyvista.plotting.assembly import Assembly
from pyvista.plotting.colors import Color
from pyvista.plotting.colors import _validate_color_sequence
from pyvista.plotting.text import TextLabel

if TYPE_CHECKING:
    from pyvista.core._typing_core import BoundsLike
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core import TransformLike
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


class AxesAssembly(Assembly):
    def __init__(
        self,
        x_label=None,
        y_label=None,
        z_label=None,
        labels=None,
        label_color='black',
        show_labels=True,
        label_position=None,
        label_size=50,
        x_color=None,
        y_color=None,
        z_color=None,
        axes_vectors=None,
        position=(0, 0, 0),
        orientation=(0, 0, 0),
        rotation=None,
        scale=(1, 1, 1),
        origin=(0, 0, 0),
        user_matrix=None,
        symmetric_bounds=False,
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
        self.add_parts(self._shaft_and_tip_actors)

        # Create label actors
        self._label_actors = (TextLabel(), TextLabel(), TextLabel())
        self.add_parts(self._label_actors)

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

        self._is_init = False

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
            self.labels = labels
        self.show_labels = show_labels
        self.label_color = label_color
        self.label_size = label_size
        self.label_position = label_position

        # Set default text properties
        for actor in self._label_actors:
            prop = actor.prop
            prop.bold = True
            prop.italic = True
            # prop.justification_horizontal = 'center'
            # prop.justification_vertical = 'center'

        self.position = position
        self.axes_vectors = np.eye(3) if axes_vectors is None else axes_vectors
        if orientation is not None and rotation is not None:
            raise ValueError(
                "Cannot set both orientation and rotation. Set either orientation or rotation, not both."
            )
        self.orientation = orientation
        if rotation is not None:
            self.rotation = rotation
        self.scale = scale
        self.origin = origin
        self.user_matrix = user_matrix

        self._is_init = True
        self._update()

    # def __getattr__(self, item: str):
    #     try:
    #         return self.__getattribute__(item)
    #     except AttributeError as e:
    #         if not item.startswith('_'):
    #             try:
    #                 return getattr(self._axes_geometry, item)
    #             except AttributeError:
    #                 raise AttributeError(str(e)) from e

    def __repr__(self):
        """Representation of the axes actor."""
        matrix_not_set = self.user_matrix is None or np.array_equal(self.user_matrix, np.eye(4))
        mat_info = 'Identity' if matrix_not_set else 'Set'
        bnds = self.bounds

        geometry_repr = repr(self._shaft_and_tip_geometry_source).splitlines()[1:]

        attr = [
            f"{type(self).__name__} ({hex(id(self))})",
            *geometry_repr,
            f"  X label:                    '{self.x_label}'",
            f"  Y label:                    '{self.y_label}'",
            f"  Z label:                    '{self.z_label}'",
            f"  Show labels:                {self.show_labels}",
            f"  Label position:             {self.label_position}",
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
        """Enable or disable the text labels for the axes."""
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
        position = self._label_position
        return self._shaft_and_tip_geometry_source.shaft_length if position is None else position

    @label_position.setter
    def label_position(self, position: Union[float, VectorLike[float]]):  # numpydoc ignore=GL08
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
        self._apply_transformation_to_labels() if self._is_init else None

    # @property
    # def label_offset(self) -> Tuple[float, float, float]:  # numpydoc ignore=RT01
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
    #     return tuple(self._label_position)
    #
    # @label_offset.setter
    # def label_offset(self, offset: float | VectorLike[float]):  # numpydoc ignore=GL08
    #     self._label_offset = _validation.validate_array3(
    #         position,
    #         broadcast=True,
    #         must_be_in_range=[0, np.inf],
    #         name='Label offset',
    #     )
    #     self._update_label_positions() if self._is_init else None

    #
    # @property
    # def tip_position(self):
    #     tips = np.diag(np.array(self.total_length) * self.scale)
    #     return apply_transformation_to_points(self.transformation_matrix, tips)

    @property
    def label_color(self):  # numpydoc ignore=RT01
        """Color of the text labels."""
        return tuple([actor.prop.color for actor in self._label_actors])

    @label_color.setter
    def label_color(self, color: ColorLike | Sequence[ColorLike]):  # numpydoc ignore=GL08
        colors = _validate_color_sequence(color, n_colors=3)
        self._label_actors[0].prop.color = colors[0]
        self._label_actors[1].prop.color = colors[1]
        self._label_actors[2].prop.color = colors[2]

    @property
    def x_color(self) -> Tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the x-axis shaft and tip."""
        return (
            self._shaft_actors[_AxisEnum.x].prop.color,
            self._tip_actors[_AxisEnum.x].prop.color,
        )

    @x_color.setter
    def x_color(self, color: Union[ColorLike, Sequence[ColorLike]]):  # numpydoc ignore=GL08
        shaft_color, tip_color = _validate_color_sequence(color, n_colors=2)
        self._shaft_actors[_AxisEnum.x].prop.color = shaft_color
        self._tip_actors[_AxisEnum.x].prop.color = tip_color

    @property
    def y_color(self) -> Tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the y-axis shaft and tip."""
        return (
            self._shaft_actors[_AxisEnum.y].prop.color,
            self._tip_actors[_AxisEnum.y].prop.color,
        )

    @y_color.setter
    def y_color(self, color: Union[ColorLike, Sequence[ColorLike]]):  # numpydoc ignore=GL08
        shaft_color, tip_color = _validate_color_sequence(color, n_colors=2)
        self._shaft_actors[_AxisEnum.y].prop.color = shaft_color
        self._tip_actors[_AxisEnum.y].prop.color = tip_color

    @property
    def z_color(self) -> Tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the z-axis shaft and tip."""
        return (
            self._shaft_actors[_AxisEnum.z].prop.color,
            self._tip_actors[_AxisEnum.z].prop.color,
        )

    @z_color.setter
    def z_color(self, color: Union[ColorLike, Sequence[ColorLike]]):  # numpydoc ignore=GL08
        shaft_color, tip_color = _validate_color_sequence(color, n_colors=2)
        self._shaft_actors[_AxisEnum.z].prop.color = shaft_color
        self._tip_actors[_AxisEnum.z].prop.color = tip_color

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

    def _get_transformed_label_positions(self):
        # Create position vectors
        shaft_length = self._shaft_and_tip_geometry_source.shaft_length
        label_position = self.label_position if self.label_position is not None else shaft_length
        position_vectors = np.diag(label_position)

        # Offset label positions radially by the tip radius
        tip_radius = self._shaft_and_tip_geometry_source.tip_radius
        offset_array = np.diag([tip_radius] * 3)
        radial_offset1 = np.roll(offset_array, shift=1, axis=1)
        radial_offset2 = np.roll(offset_array, shift=-1, axis=1)

        position_vectors += radial_offset1 + radial_offset2

        # Transform positions
        matrix = self._prop3d.transformation_matrix
        return apply_transformation_to_points(matrix, position_vectors)

    def _apply_transformation_to_labels(self):
        x_pos, y_pos, z_pos = self._get_transformed_label_positions()
        self._label_actors[0].position = x_pos
        self._label_actors[1].position = y_pos
        self._label_actors[2].position = z_pos

    def _update(self):
        self._shaft_and_tip_geometry_source.update()
        self._apply_transformation_to_labels()

    @property
    def scale(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Scaling factor applied to the axes.

        Examples
        --------
        Create an actor using the :class:`pyvista.Plotter` and then change the
        scale of the actor.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Sphere())
        >>> actor.scale = (2.0, 2.0, 2.0)
        >>> actor.scale
        (2.0, 2.0, 2.0)

        """
        return self._prop3d.scale

    @scale.setter
    def scale(self, scale: VectorLike[float]):  # numpydoc ignore=GL08
        self._set_prop3d_attr('scale', scale)

    @property
    def position(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the entity position.

        Examples
        --------
        Change the position of an actor. Note how this does not change the
        position of the underlying dataset, just the relative location of the
        actor in the :class:`pyvista.Plotter`.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, color='b')
        >>> actor = pl.add_mesh(mesh, color='r')
        >>> actor.position = (0, 0, 1)  # shifts the red sphere up
        >>> pl.show()

        """
        return self._prop3d.position

    @position.setter
    def position(self, position: VectorLike[float]):  # numpydoc ignore=GL08
        self._set_prop3d_attr('position', position)

    def rotate_x(self, angle: float):
        """Rotate the entity about the x-axis.

        Parameters
        ----------
        angle : float
            Angle to rotate the entity about the x-axis in degrees.

        Examples
        --------
        Rotate the actor about the x-axis 45 degrees. Note how this does not
        change the location of the underlying dataset.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, color='b')
        >>> actor = pl.add_mesh(
        ...     mesh,
        ...     color='r',
        ...     style='wireframe',
        ...     line_width=5,
        ...     lighting=False,
        ... )
        >>> actor.rotate_x(45)
        >>> pl.show_axes()
        >>> pl.show()

        """
        self.RotateX(angle)

    def rotate_y(self, angle: float):
        """Rotate the entity about the y-axis.

        Parameters
        ----------
        angle : float
            Angle to rotate the entity about the y-axis in degrees.

        Examples
        --------
        Rotate the actor about the y-axis 45 degrees. Note how this does not
        change the location of the underlying dataset.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, color='b')
        >>> actor = pl.add_mesh(
        ...     mesh,
        ...     color='r',
        ...     style='wireframe',
        ...     line_width=5,
        ...     lighting=False,
        ... )
        >>> actor.rotate_y(45)
        >>> pl.show_axes()
        >>> pl.show()

        """
        self.RotateY(angle)

    def rotate_z(self, angle: float):
        """Rotate the entity about the z-axis.

        Parameters
        ----------
        angle : float
            Angle to rotate the entity about the z-axis in degrees.

        Examples
        --------
        Rotate the actor about the z-axis 45 degrees. Note how this does not
        change the location of the underlying dataset.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, color='b')
        >>> actor = pl.add_mesh(
        ...     mesh,
        ...     color='r',
        ...     style='wireframe',
        ...     line_width=5,
        ...     lighting=False,
        ... )
        >>> actor.rotate_z(45)
        >>> pl.show_axes()
        >>> pl.show()

        """
        self.RotateZ(angle)

    @property
    def orientation(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the entity orientation angles.

        Orientation angles of the axes which define rotations about the
        world's x-y-z axes. The angles are specified in degrees and in
        x-y-z order. However, the actual rotations are applied in the
        following order: :func:`~rotate_y` first, then :func:`~rotate_x`
        and finally :func:`~rotate_z`.

        Rotations are applied about the specified :attr:`~origin`.

        Examples
        --------
        Reorient just the actor and plot it. Note how the actor is rotated
        about the origin ``(0, 0, 0)`` by default.

        >>> import pyvista as pv
        >>> mesh = pv.Cube(center=(0, 0, 3))
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, color='b')
        >>> actor = pl.add_mesh(
        ...     mesh,
        ...     color='r',
        ...     style='wireframe',
        ...     line_width=5,
        ...     lighting=False,
        ... )
        >>> actor.orientation = (45, 0, 0)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        Repeat the last example, but this time reorient the actor about
        its center by specifying its :attr:`~origin`.

        >>> import pyvista as pv
        >>> mesh = pv.Cube(center=(0, 0, 3))
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, color='b')
        >>> actor = pl.add_mesh(
        ...     mesh,
        ...     color='r',
        ...     style='wireframe',
        ...     line_width=5,
        ...     lighting=False,
        ... )
        >>> actor.origin = actor.center
        >>> actor.orientation = (45, 0, 0)
        >>> _ = pl.add_axes_at_origin()
        >>> pl.show()

        Show that the orientation changes with rotation.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(mesh)
        >>> actor.rotate_x(90)
        >>> actor.orientation  # doctest:+SKIP
        (90, 0, 0)

        Set the orientation directly.

        >>> actor.orientation = (0, 45, 45)
        >>> actor.orientation  # doctest:+SKIP
        (0, 45, 45)

        """
        return self._prop3d.orientation

    @orientation.setter
    def orientation(self, orientation: tuple[float, float, float]):  # numpydoc ignore=GL08
        self._set_prop3d_attr('orientation', orientation)

    @property
    def rotation(self) -> NumpyArray[float]:  # numpydoc ignore=GL08
        return self._prop3d.rotation

    @rotation.setter
    def rotation(self, array: NumpyArray[float]):  # numpydoc ignore=GL08
        self._set_prop3d_attr('rotation', array)

    def _set_prop3d_attr(self, name, value):
        # Set props for shaft and tip actors
        setattr(self._prop3d, name, value)
        valid_value = getattr(self._prop3d, name)
        [setattr(actor, name, valid_value) for actor in self._shaft_and_tip_actors]

        # Update labels
        self._apply_transformation_to_labels()

    @property
    def origin(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the entity origin.

        This is the point about which all rotations take place.

        See :attr:`~orientation` for examples.

        """
        return self._prop3d.origin

    @origin.setter
    def origin(self, origin: tuple[float, float, float]):  # numpydoc ignore=GL08
        self._prop3d.origin = origin

    @property
    def bounds(self) -> BoundsLike:  # numpydoc ignore=RT01
        """Return the bounds of the entity.

        Bounds are ``(-X, +X, -Y, +Y, -Z, +Z)``

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> mesh = pv.Cube(x_length=0.1, y_length=0.2, z_length=0.3)
        >>> actor = pl.add_mesh(mesh)
        >>> actor.bounds
        (-0.05, 0.05, -0.1, 0.1, -0.15, 0.15)

        """
        return self.GetBounds()

    @property
    def center(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return the center of the entity.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Sphere(center=(0.5, 0.5, 1)))
        >>> actor.center  # doctest:+SKIP
        (0.5, 0.5, 1)
        """
        bnds = self.bounds
        return (bnds[0] + bnds[1]) / 2, (bnds[1] + bnds[2]) / 2, (bnds[4] + bnds[5]) / 2

    @property
    def user_matrix(self) -> NumpyArray[float]:  # numpydoc ignore=RT01
        """Return or set the user matrix.

        In addition to the instance variables such as position and orientation, the user
        can add an additional transformation to the actor.

        This matrix is concatenated with the actor's internal transformation that is
        implicitly created when the actor is created. This affects the actor/rendering
        only, not the input data itself.

        The user matrix is the last transformation applied to the actor before
        rendering.

        Returns
        -------
        np.ndarray
            A 4x4 transformation matrix.

        Examples
        --------
        Apply a 4x4 translation to a wireframe actor. This 4x4 transformation
        effectively translates the actor by one unit in the Z direction,
        rotates the actor about the z-axis by approximately 45 degrees, and
        shrinks the actor by a factor of 0.5.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(mesh, color="b")
        >>> actor = pl.add_mesh(
        ...     mesh,
        ...     color="r",
        ...     style="wireframe",
        ...     line_width=5,
        ...     lighting=False,
        ... )
        >>> arr = np.array(
        ...     [
        ...         [0.707, -0.707, 0, 0],
        ...         [0.707, 0.707, 0, 0],
        ...         [0, 0, 1, 1.500001],
        ...         [0, 0, 0, 2],
        ...     ]
        ... )
        >>> actor.user_matrix = arr
        >>> pl.show_axes()
        >>> pl.show()

        """
        return self._user_matrix

    @user_matrix.setter
    def user_matrix(self, matrix: TransformLike):  # numpydoc ignore=GL08
        array = np.eye(4) if matrix is None else _validation.validate_transform4x4(matrix)
        self._user_matrix = array

    @property
    def length(self) -> float:  # numpydoc ignore=RT01
        """Return the length of the entity.

        Examples
        --------
        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Sphere())
        >>> actor.length
        1.7272069317100354
        """
        bnds = self.bounds
        min_bnds = np.array((bnds[0], bnds[2], bnds[4]))
        max_bnds = np.array((bnds[1], bnds[3], bnds[5]))
        return np.linalg.norm(max_bnds - min_bnds).tolist()

    @property
    def axes_vectors(self):  # numpydoc ignore=RT01
        """Direction vectors of the axes.

        The direction vectors are used as a 3x3 rotation matrix to orient the axes in
        space. By default, the direction vectors align with the XYZ axes of the world
        coordinates.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> axes_geometry_source = pv.AxesGeometrySource()
        >>> axes_geometry_source.axes_vectors
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])

        Orient the axes in space.

        >>> vectors = np.array(
        ...     [
        ...         [0.36, 0.48, -0.80],
        ...         [-0.80, 0.60, 0.00],
        ...         [0.48, 0.64, 0.60],
        ...     ]
        ... )

        >>> axes_geometry_source.axes_vectors = vectors
        >>> axes_geometry_source.axes_vectors
        array([[ 0.36,  0.48, -0.8 ],
               [-0.8 ,  0.6 ,  0.  ],
               [ 0.48,  0.64,  0.6 ]])
        """
        return self._axes_vectors

    @axes_vectors.setter
    def axes_vectors(self, vectors):  # numpydoc ignore=GL08
        self._axes_vectors = _validation.validate_axes(
            vectors, name='axes vectors', must_be_orthogonal=False, must_have_orientation=None
        )
