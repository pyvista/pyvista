"""Axes assembly module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import TypedDict

import numpy as np

import pyvista as pv
from pyvista.core import _validation
from pyvista.core.utilities.arrays import array_from_vtkmatrix
from pyvista.core.utilities.geometric_sources import AxesGeometrySource
from pyvista.core.utilities.geometric_sources import _AxisEnum
from pyvista.core.utilities.transformations import apply_transformation_to_points
from pyvista.plotting import _vtk
from pyvista.plotting.actor import Actor
from pyvista.plotting.colors import Color
from pyvista.plotting.text import Label

if TYPE_CHECKING:  # pragma: no cover
    import sys
    from typing import Iterator
    from typing import Sequence

    from pyvista.core._typing_core import BoundsLike
    from pyvista.core._typing_core import MatrixLike
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core import TransformLike
    from pyvista.core._typing_core import VectorLike
    from pyvista.core.dataset import DataSet
    from pyvista.plotting._typing import ColorLike

    if sys.version_info >= (3, 11):
        from typing import Unpack
    else:
        from typing_extensions import Unpack


class _AxesGeometryKwargs(TypedDict):
    shaft_type: AxesGeometrySource.GeometryTypes | DataSet
    shaft_radius: float
    shaft_length: float | VectorLike[float]
    tip_type: AxesGeometrySource.GeometryTypes | DataSet
    tip_radius: float
    tip_length: float | VectorLike[float]
    symmetric_bounds: bool


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

    position : VectorLike[float], default: (0.0, 0.0, 0.0)
        Position of the axes in space.

    orientation : VectorLike[float], default: (0, 0, 0)
        Orientation angles of the axes which define rotations about the
        world's x-y-z axes. The angles are specified in degrees and in
        x-y-z order. However, the actual rotations are applied in the
        around the y-axis first, then the x-axis, and finally the z-axis.

    origin : VectorLike[float], default: (0.0, 0.0, 0.0)
        Origin of the axes. This is the point about which all rotations take place. The
        rotations are defined by the :attr:`orientation`.

    scale : VectorLike[float], default: (1.0, 1.0, 1.0)
        Scaling factor applied to the axes.

    user_matrix : MatrixLike[float], optional
        A 4x4 transformation matrix applied to the axes. Defaults to the identity matrix.
        The user matrix is the last transformation applied to the actor.

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

    Create axes with custom geometry. Use pyramid shafts and hemisphere tips and
    modify the lengths.

    >>> axes = pv.AxesAssembly(
    ...     shaft_type='pyramid',
    ...     tip_type='hemisphere',
    ...     tip_length=0.1,
    ...     shaft_length=(0.5, 1.0, 1.5),
    ... )
    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(axes)
    >>> pl.show()

    Position and orient the axes in space.

    >>> axes = pv.AxesAssembly(
    ...     position=(1.0, 2.0, 3.0), orientation=(10, 20, 30)
    ... )
    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(axes)
    >>> pl.show()

    Add the axes as a custom orientation widget with
    :func:`~pyvista.Renderer.add_orientation_widget`:

    >>> import pyvista as pv

    >>> axes = pv.AxesAssembly(symmetric_bounds=True)

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(pv.Cone())
    >>> _ = pl.add_orientation_widget(
    ...     axes,
    ...     viewport=(0, 0, 0.5, 0.5),
    ... )
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
        position: VectorLike[float] = (0.0, 0.0, 0.0),
        orientation: VectorLike[float] = (0.0, 0.0, 0.0),
        origin: VectorLike[float] = (0.0, 0.0, 0.0),
        scale: VectorLike[float] = (1.0, 1.0, 1.0),
        user_matrix: MatrixLike[float] | None = None,
        **kwargs: Unpack[_AxesGeometryKwargs],
    ):
        super().__init__()

        # Add dummy prop3d for calculating transformations
        self._prop3d = Actor()

        # Init shaft and tip actors
        self._shaft_actors = (Actor(), Actor(), Actor())
        self._tip_actors = (Actor(), Actor(), Actor())
        self._shaft_and_tip_actors = (*self._shaft_actors, *self._tip_actors)

        # Init shaft and tip datasets
        self._shaft_and_tip_geometry_source = AxesGeometrySource(symmetric=False, **kwargs)
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
        for label in self._label_actor_iterator:
            prop = label.prop
            prop.bold = True
            prop.italic = True

        self.position = position  # type: ignore[assignment]
        self.orientation = orientation  # type: ignore[assignment]
        self.scale = scale  # type: ignore[assignment]
        self.origin = origin  # type: ignore[assignment]
        self.user_matrix = user_matrix  # type: ignore[assignment]

    def __repr__(self):
        """Representation of the axes assembly."""
        if self.user_matrix is None or np.array_equal(self.user_matrix, np.eye(4)):
            mat_info = 'Identity'
        else:
            mat_info = 'Set'
        bnds = self.bounds

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
            f"  Position:                   {self.position}",
            f"  Orientation:                {self.orientation}",
            f"  Origin:                     {self.origin}",
            f"  Scale:                      {self.scale}",
            f"  User matrix:                {mat_info}",
            f"  X Bounds                    {bnds[0]:.3E}, {bnds[1]:.3E}",
            f"  Y Bounds                    {bnds[2]:.3E}, {bnds[3]:.3E}",
            f"  Z Bounds                    {bnds[4]:.3E}, {bnds[5]:.3E}",
        ]
        return '\n'.join(attr)

    @property
    def _label_actor_iterator(self) -> Iterator[Label]:
        collection = self.GetParts()
        parts = [collection.GetItemAsObject(i) for i in range(collection.GetNumberOfItems())]
        return (part for part in parts if isinstance(part, Label))

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
        labels = _validate_label_sequence(labels, n_labels=3, name='labels')
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
        for label in self._label_actor_iterator:
            label.SetVisibility(value)

    @property
    def label_size(self) -> int:  # numpydoc ignore=RT01
        """Size of the text labels.

        Must be a positive integer.
        """
        return self._label_size

    @label_size.setter
    def label_size(self, size: int):  # numpydoc ignore=GL08
        self._label_size = size
        for label in self._label_actor_iterator:
            label.size = size

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
        self._update_label_positions()

    @property
    def label_color(self) -> Color:  # numpydoc ignore=RT01
        """Color of the text labels."""
        return self._label_color

    @label_color.setter
    def label_color(self, color: ColorLike):  # numpydoc ignore=GL08
        valid_color = Color(color)
        self._label_color = valid_color
        for label in self._label_actor_iterator:
            label.prop.color = valid_color

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

    def _transform_label_position(self, position_scalars: tuple[float, float, float]):
        # Create position vectors
        position_vectors = np.diag(position_scalars)

        # Offset label positions radially by the tip radius
        tip_radius = self._shaft_and_tip_geometry_source.tip_radius
        offset_array = np.diag([tip_radius] * 3)
        radial_offset1 = np.roll(offset_array, shift=1, axis=1)
        radial_offset2 = np.roll(offset_array, shift=-1, axis=1)

        position_vectors += radial_offset1 + radial_offset2

        # Transform positions
        matrix = array_from_vtkmatrix(self._prop3d.GetMatrix())
        return apply_transformation_to_points(matrix, position_vectors)

    def _apply_transformation_to_labels(
        self, position_scalars: tuple[float, float, float], labels: tuple[Label, Label, Label]
    ):
        vectors = self._transform_label_position(position_scalars)
        for label, vector in zip(labels, vectors):
            label.position = vector

    def _update_label_positions(self):
        self._apply_transformation_to_labels(self.label_position, self._label_actors)

    def _set_prop3d_attr(self, name, value):
        # Set props for shaft and tip actors
        # Validate input by setting then getting from prop3d
        setattr(self._prop3d, name, value)
        valid_value = getattr(self._prop3d, name)
        [setattr(actor, name, valid_value) for actor in self._shaft_and_tip_actors]

        # Update labels
        self._update_label_positions()

    @property
    def scale(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the scaling factor applied to the axes.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.AxesAssembly()
        >>> axes.scale = (2.0, 2.0, 2.0)
        >>> axes.scale
        (2.0, 2.0, 2.0)
        """
        return self._prop3d.scale

    @scale.setter
    def scale(self, scale: VectorLike[float]):  # numpydoc ignore=GL08
        self._set_prop3d_attr('scale', scale)

    @property
    def position(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the position of the axes.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.AxesAssembly()
        >>> axes.position = (1.0, 2.0, 3.0)
        >>> axes.position
        (1.0, 2.0, 3.0)
        """
        return self._prop3d.position

    @position.setter
    def position(self, position: VectorLike[float]):  # numpydoc ignore=GL08
        self._set_prop3d_attr('position', position)

    @property
    def orientation(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the axes orientation angles.

        Orientation angles of the axes which define rotations about the
        world's x-y-z axes. The angles are specified in degrees and in
        x-y-z order. However, the actual rotations are applied in the
        following order: :func:`~rotate_y` first, then :func:`~rotate_x`
        and finally :func:`~rotate_z`.

        Rotations are applied about the specified :attr:`~origin`.

        Examples
        --------
        Create axes positioned above the origin and set its orientation.

        >>> import pyvista as pv
        >>> axes = pv.AxesAssembly(
        ...     position=(0, 0, 2), orientation=(45, 0, 0)
        ... )

        Create default non-oriented axes as well for reference.

        >>> reference_axes = pv.AxesAssembly(
        ...     x_color='black', y_color='black', z_color='black'
        ... )

        Plot the axes. Note how the axes are rotated about the origin ``(0, 0, 0)`` by
        default, such that the rotated axes appear directly above the reference axes.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_actor(axes)
        >>> _ = pl.add_actor(reference_axes)
        >>> pl.show()

        Now change the origin of the axes and plot the result. Since the rotation
        is performed about a different point, the final position of the axes changes.

        >>> axes.origin = (2, 2, 2)
        >>> pl = pv.Plotter()
        >>> _ = pl.add_actor(axes)
        >>> _ = pl.add_actor(reference_axes)
        >>> pl.show()
        """
        return self._prop3d.orientation

    @orientation.setter
    def orientation(self, orientation: tuple[float, float, float]):  # numpydoc ignore=GL08
        self._set_prop3d_attr('orientation', orientation)

    @property
    def origin(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the origin of the axes.

        This is the point about which all rotations take place.

        See :attr:`~orientation` for examples.

        """
        return self._prop3d.origin

    @origin.setter
    def origin(self, origin: tuple[float, float, float]):  # numpydoc ignore=GL08
        self._set_prop3d_attr('origin', origin)

    @property
    def user_matrix(self) -> NumpyArray[float]:  # numpydoc ignore=RT01
        """Return or set the user matrix.

        In addition to the instance variables such as position and orientation, the user
        can add a transformation to the actor.

        This matrix is concatenated with the actor's internal transformation that is
        implicitly created when the actor is created. The user matrix is the last
        transformation applied to the actor before rendering.

        Returns
        -------
        np.ndarray
            A 4x4 transformation matrix.

        Examples
        --------
        Apply a 4x4 transformation to the axes. This effectively translates the actor
        by one unit in the Z direction, rotates the actor about the z-axis by
        approximately 45 degrees, and shrinks the actor by a factor of 0.5.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> axes = pv.AxesAssembly()
        >>> array = np.array(
        ...     [
        ...         [0.35355339, -0.35355339, 0.0, 0.0],
        ...         [0.35355339, 0.35355339, 0.0, 0.0],
        ...         [0.0, 0.0, 0.5, 1.0],
        ...         [0.0, 0.0, 0.0, 1.0],
        ...     ]
        ... )
        >>> axes.user_matrix = array

        >>> pl = pv.Plotter()
        >>> _ = pl.add_actor(axes)
        >>> pl.show()

        """
        return self._prop3d.user_matrix

    @user_matrix.setter
    def user_matrix(self, matrix: TransformLike):  # numpydoc ignore=GL08
        self._set_prop3d_attr('user_matrix', matrix)

    @property
    def bounds(self) -> BoundsLike:  # numpydoc ignore=RT01
        """Return the bounds of the axes.

        Bounds are ``(-X, +X, -Y, +Y, -Z, +Z)``

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.AxesAssembly()
        >>> axes.bounds
        (-0.10000000149011612, 1.0, -0.10000000149011612, 1.0, -0.10000000149011612, 1.0)
        """
        return self.GetBounds()

    @property
    def center(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return the center of the axes.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.AxesAssembly()
        >>> axes.center
        (0.44999999925494194, 0.44999999925494194, 0.44999999925494194)
        """
        bnds = self.bounds
        return (bnds[0] + bnds[1]) / 2, (bnds[1] + bnds[2]) / 2, (bnds[4] + bnds[5]) / 2

    @property
    def length(self) -> float:  # numpydoc ignore=RT01
        """Return the length of the axes.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.AxesAssembly()
        >>> axes.length
        1.9052558909067219
        """
        bnds = self.bounds
        min_bnds = np.array((bnds[0], bnds[2], bnds[4]))
        max_bnds = np.array((bnds[1], bnds[3], bnds[5]))
        return np.linalg.norm(max_bnds - min_bnds).tolist()


def _validate_label_sequence(labels: Sequence[str], n_labels: int | Sequence[int], name: str):
    _validation.check_instance(labels, (list, tuple), name=name)
    _validation.check_iterable_items(labels, str, name=name)
    _validation.check_length(labels, exact_length=n_labels, name=name)
    return labels


def _validate_color_sequence(
    color: ColorLike | Sequence[ColorLike],
    n_colors: int | None = None,
) -> tuple[Color, ...]:
    """Validate a color sequence.

    If `n_colors` is specified, the output will have `n` colors. For single-color
    inputs, the color is copied and a sequence of `n` identical colors is returned.
    For inputs with multiple colors, the number of colors in the input must
    match `n_colors`.

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


class AxesAssemblySymmetric(AxesAssembly):
    """Symmetric assembly of arrow-style axes parts.

    This class is similar to :class:`~pyvista.AxesAssembly` but the axes are
    symmetric.

    The axes may be used as a widget or added to a scene.

    Parameters
    ----------
    x_label : str, default: ('+X', '-X')
        Text labels for the positive and negative x-axis. Specify two strings or a
        single string. If a single string, plus ``'+'`` and minus ``'-'`` characters
        are added. Alternatively, set the labels with :attr:`labels`.

    y_label : str, default: ('+Y', '-Y')
        Text labels for the positive and negative y-axis. Specify two strings or a
        single string. If a single string, plus ``'+'`` and minus ``'-'`` characters
        are added. Alternatively, set the labels with :attr:`labels`.

    z_label : str, default: ('+Z', '-Z')
        Text labels for the positive and negative z-axis. Specify two strings or a
        single string. If a single string, plus ``'+'`` and minus ``'-'`` characters
        are added. Alternatively, set the labels with :attr:`labels`.

    labels : Sequence[str], optional
        Text labels for the axes. Specify three strings, one for each axis, or
        six strings, one for each +/- axis. If three strings plus ``'+'`` and minus
        ``'-'`` characters are added. This is an alternative parameter to using
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

    position : VectorLike[float], default: (0.0, 0.0, 0.0)
        Position of the axes in space.

    orientation : VectorLike[float], default: (0, 0, 0)
        Orientation angles of the axes which define rotations about the
        world's x-y-z axes. The angles are specified in degrees and in
        x-y-z order. However, the actual rotations are applied in the
        around the y-axis first, then the x-axis, and finally the z-axis.

    origin : VectorLike[float], default: (0.0, 0.0, 0.0)
        Origin of the axes. This is the point about which all rotations take place. The
        rotations are defined by the :attr:`orientation`.

    scale : VectorLike[float], default: (1.0, 1.0, 1.0)
        Scaling factor applied to the axes.

    user_matrix : MatrixLike[float], optional
        A 4x4 transformation matrix applied to the axes. Defaults to the identity matrix.
        The user matrix is the last transformation applied to the actor.

    **kwargs
        Keyword arguments passed to :class:`pyvista.AxesGeometrySource`.

    Examples
    --------
    Add symmetric axes to a plot.

    >>> import pyvista as pv
    >>> axes_assembly = pv.AxesAssemblySymmetric()
    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(axes_assembly)
    >>> pl.show()

    Customize the axes labels.

    >>> axes_assembly.labels = [
    ...     'east',
    ...     'west',
    ...     'north',
    ...     'south',
    ...     'up',
    ...     'down',
    ... ]
    >>> axes_assembly.label_color = 'darkgoldenrod'

    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(axes_assembly)
    >>> pl.show()

    Add the axes as a custom orientation widget with
    :func:`~pyvista.Renderer.add_orientation_widget`. We also configure the labels to
    only show text for the positive axes.

    >>> axes_assembly = pv.AxesAssemblySymmetric(
    ...     x_label=('X', ""), y_label=('Y', ""), z_label=('Z', "")
    ... )
    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(pv.Cone())
    >>> _ = pl.add_orientation_widget(
    ...     axes_assembly,
    ...     viewport=(0, 0, 0.5, 0.5),
    ... )
    >>> pl.show()
    """

    def __init__(
        self,
        *,
        x_label: str | Sequence[str] | None = None,
        y_label: str | Sequence[str] | None = None,
        z_label: str | Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
        label_color: ColorLike = 'black',
        show_labels: bool = True,
        label_position: float | VectorLike[float] | None = None,
        label_size: int = 50,
        x_color: ColorLike | Sequence[ColorLike] | None = None,
        y_color: ColorLike | Sequence[ColorLike] | None = None,
        z_color: ColorLike | Sequence[ColorLike] | None = None,
        position: VectorLike[float] = (0.0, 0.0, 0.0),
        orientation: VectorLike[float] = (0.0, 0.0, 0.0),
        origin: VectorLike[float] = (0.0, 0.0, 0.0),
        scale: VectorLike[float] = (1.0, 1.0, 1.0),
        user_matrix: MatrixLike[float] | None = None,
        **kwargs: Unpack[_AxesGeometryKwargs],
    ):
        # Init symmetric label actors and add to assembly
        self._label_actors_symmetric = (Label(), Label(), Label())
        [self.AddPart(actor) for actor in self._label_actors_symmetric]

        super().__init__(
            x_label=x_label,  # type: ignore[arg-type]
            y_label=y_label,  # type: ignore[arg-type]
            z_label=z_label,  # type: ignore[arg-type]
            labels=labels,
            label_color=label_color,
            show_labels=show_labels,
            label_position=label_position,
            label_size=label_size,
            x_color=x_color,
            y_color=y_color,
            z_color=z_color,
            position=position,
            orientation=orientation,
            origin=origin,
            scale=scale,
            user_matrix=user_matrix,
            **kwargs,
        )

        # Make the geometry symmetric
        self._shaft_and_tip_geometry_source.symmetric = True
        self._shaft_and_tip_geometry_source.update()

    @property  # type: ignore[override]
    def labels(self) -> tuple[str, str, str, str, str, str]:  # numpydoc ignore=RT01
        """Return or set the axes labels.

        Specify three strings, one for each axis, or six strings, one for each +/- axis.
        If three strings, plus ``'+'`` and minus ``'-'`` characters are added.
        This property may be used as an alternative to using :attr:`x_label`,
        :attr:`y_label`, and :attr:`z_label` separately.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_assembly = pv.AxesAssemblySymmetric()

        Use three strings to set the labels. Plus ``'+'`` and minus ``'-'``
        characters are added automatically.

        >>> axes_assembly.labels = ['U', 'V', 'W']
        >>> axes_assembly.labels
        ('+U', '-U', '+V', '-V', '+W', '-W')

        Alternatively, use six strings to set the labels explicitly.

        >>> axes_assembly.labels = [
        ...     'east',
        ...     'west',
        ...     'north',
        ...     'south',
        ...     'up',
        ...     'down',
        ... ]
        >>> axes_assembly.labels
        ('east', 'west', 'north', 'south', 'up', 'down')
        """
        return *self.x_label, *self.y_label, *self.z_label

    @labels.setter
    def labels(
        self, labels: list[str] | tuple[str, str, str] | tuple[str, str, str, str, str, str]
    ):  # numpydoc ignore=GL08
        valid_labels = _validate_label_sequence(labels, n_labels=[3, 6], name='labels')
        if len(valid_labels) == 3:
            self.x_label = valid_labels[0]
            self.y_label = valid_labels[1]
            self.z_label = valid_labels[2]
        else:
            self.x_label = valid_labels[0:2]
            self.y_label = valid_labels[2:4]
            self.z_label = valid_labels[4:6]

    def _get_axis_label(self, axis: _AxisEnum) -> tuple[str, str]:
        label_plus = self._label_actors[axis].input
        label_minus = self._label_actors_symmetric[axis].input
        return label_plus, label_minus

    def _set_axis_label(self, axis: _AxisEnum, label: str | list[str] | tuple[str, str]):
        if isinstance(label, str):
            label_plus, label_minus = '+' + label, '-' + label
        else:
            label_plus, label_minus = _validate_label_sequence(label, n_labels=2, name='label')
        self._label_actors[axis].input = label_plus
        self._label_actors_symmetric[axis].input = label_minus

    @property  # type: ignore[override]
    def x_label(self) -> tuple[str, str]:  # numpydoc ignore=RT01
        """Return or set the labels for the positive and negative x-axis.

        The labels may be set with a single string or two strings. If a single string,
        plus ``'+'`` and minus ``'-'`` characters are added. Alternatively, set the
        labels with :attr:`labels`.

        Examples
        --------
        Set the labels with a single string. Plus ``'+'`` and minus ``'-'``
        characters are added automatically.

        >>> import pyvista as pv
        >>> axes_assembly = pv.AxesAssemblySymmetric()
        >>> axes_assembly.x_label = 'Axis'
        >>> axes_assembly.x_label
        ('+Axis', '-Axis')

        Set the labels explicitly with two strings.

        >>> axes_assembly.x_label = 'anterior', 'posterior'
        >>> axes_assembly.x_label
        ('anterior', 'posterior')
        """
        return self._get_axis_label(_AxisEnum.x)

    @x_label.setter
    def x_label(self, label: str | list[str] | tuple[str, str]):  # numpydoc ignore=GL08
        self._set_axis_label(_AxisEnum.x, label)

    @property  # type: ignore[override]
    def y_label(self) -> tuple[str, str]:  # numpydoc ignore=RT01
        """Return or set the labels for the positive and negative y-axis.

        The labels may be set with a single string or two strings. If a single string,
        plus ``'+'`` and minus ``'-'`` characters are added. Alternatively, set the
        labels with :attr:`labels`.

        Examples
        --------
        Set the labels with a single string. Plus ``'+'`` and minus ``'-'``
        characters are added automatically.

        >>> import pyvista as pv
        >>> axes_assembly = pv.AxesAssemblySymmetric()
        >>> axes_assembly.y_label = 'Axis'
        >>> axes_assembly.y_label
        ('+Axis', '-Axis')

        Set the labels explicitly with two strings.

        >>> axes_assembly.y_label = 'left', 'right'
        >>> axes_assembly.y_label
        ('left', 'right')
        """
        return self._get_axis_label(_AxisEnum.y)

    @y_label.setter
    def y_label(self, label: str | list[str] | tuple[str, str]):  # numpydoc ignore=GL08
        self._set_axis_label(_AxisEnum.y, label)

    @property  # type: ignore[override]
    def z_label(self) -> tuple[str, str]:  # numpydoc ignore=RT01
        """Return or set the labels for the positive and negative z-axis.

        The labels may be set with a single string or two strings. If a single string,
        plus ``'+'`` and minus ``'-'`` characters are added. Alternatively, set the
        labels with :attr:`labels`.

        Examples
        --------
        Set the labels with a single string. Plus ``'+'`` and minus ``'-'``
        characters are added automatically.

        >>> import pyvista as pv
        >>> axes_assembly = pv.AxesAssemblySymmetric()
        >>> axes_assembly.z_label = 'Axis'
        >>> axes_assembly.z_label
        ('+Axis', '-Axis')

        Set the labels explicitly with two strings.

        >>> axes_assembly.z_label = 'superior', 'inferior'
        >>> axes_assembly.z_label
        ('superior', 'inferior')
        """
        return self._get_axis_label(_AxisEnum.z)

    @z_label.setter
    def z_label(self, label: str | list[str] | tuple[str, str]):  # numpydoc ignore=GL08
        self._set_axis_label(_AxisEnum.z, label)

    def _update_label_positions(self):
        position_plus = self.label_position
        labels_plus = self._label_actors
        self._apply_transformation_to_labels(position_plus, labels_plus)

        position_minus = (-position_plus[0], -position_plus[1], -position_plus[2])
        labels_minus = self._label_actors_symmetric
        self._apply_transformation_to_labels(position_minus, labels_minus)
