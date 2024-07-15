"""Planes assembly module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal
from typing import NamedTuple
from typing import Sequence
from typing import TypedDict

import numpy as np

import pyvista as pv
from pyvista.core import _validation
from pyvista.core.utilities.geometric_sources import OrthogonalPlanesSource
from pyvista.plotting import _vtk
from pyvista.plotting.actor import Actor
from pyvista.plotting.axes_assembly import _validate_label_sequence
from pyvista.plotting.colors import Color
from pyvista.plotting.text import Label

if TYPE_CHECKING:  # pragma: no cover
    import sys
    from typing import Iterator

    from pyvista.core._typing_core import BoundsLike
    from pyvista.core._typing_core import MatrixLike
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core import TransformLike
    from pyvista.core._typing_core import VectorLike
    from pyvista.plotting._typing import ColorLike

    if sys.version_info >= (3, 11):
        from typing import Unpack
    else:
        from typing_extensions import Unpack


class _BoundsTuple(NamedTuple):
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


class _OrthogonalPlanesKwargs(TypedDict):
    bounds: VectorLike[float]
    resolution: int | VectorLike[int]
    normal_sign: Literal['+', '-'] | Sequence[str]
    # names: Sequence[str] = ('xy', 'yz', 'zx')


class PlanesAssembly(_vtk.vtkPropAssembly):
    """Assembly of orthogonal planes.

    The assembly may be used as a widget or added to a scene.

    Parameters
    ----------
    x_label : str, default: 'XY'
        Text label for the xy-plane. Alternatively, set the label with :attr:`labels`.

    y_label : str, default: 'YZ'
        Text label for the yz-plane. Alternatively, set the label with :attr:`labels`.

    z_label : str, default: 'ZX'
        Text label for the zx-plane. Alternatively, set the label with :attr:`labels`.

    labels : Sequence[str], optional,
        Text labels for the planes. This is an alternative parameter to using
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

    label_mode : '2D' | '3D', default: '2D'
        Mode to use for text labels. In '2D' mode, the label actors are 2D
        sprites and are always visible. In '3D' mode, they are polygons
        and may be occluded. The two modes have minor differences in
        appearance as well as behavior in terms of how they follow the camera.

    x_color : ColorLike, optional
        Color of the xy-plane.

    y_color : ColorLike, optional
        Color of the yz-plane.

    z_color : ColorLike, optional
        Color of the zx-plane.

    opacity : float, default: 0.3
        Opacity of the planes.

    position : VectorLike[float], default: (0.0, 0.0, 0.0)
        Position of the planes in space.

    orientation : VectorLike[float], default: (0, 0, 0)
        Orientation angles of the assembly which define rotations about the
        world's x-y-z axes. The angles are specified in degrees and in
        x-y-z order. However, the actual rotations are applied in the
        around the y-axis first, then the x-axis, and finally the z-axis.

    origin : VectorLike[float], default: (0.0, 0.0, 0.0)
        Origin of the assembly. This is the point about which all rotations take place.
        The rotations are defined by the :attr:`orientation`.

    scale : VectorLike[float], default: (1.0, 1.0, 1.0)
        Scaling factor applied to the assembly.

    user_matrix : MatrixLike[float], optional
        A 4x4 transformation matrix applied to the assembly. Defaults to the identity
        matrix. The user matrix is the last transformation applied to the actor.

    **kwargs
        Keyword arguments passed to :class:`pyvista.AxesGeometrySource`.

    Examples
    --------
    Add planes to a plot.

    >>> import pyvista as pv
    >>> planes = pv.PlanesAssembly()
    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(planes)
    >>> pl.show()

    Customize the plane colors.

    >>> planes.x_color = ['cyan', 'blue']
    >>> planes.y_color = ['magenta', 'red']
    >>> planes.z_color = 'yellow'

    Customize the label color too.

    >>> planes.label_color = 'brown'

    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(planes)
    >>> pl.show()

    Position and orient the axes in space.

    >>> axes = pv.PlanesAssembly(
    ...     position=(1.0, 2.0, 3.0), orientation=(10, 20, 30)
    ... )
    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(axes)
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
        label_position: int | VectorLike[int] = 0,
        label_size: int = 50,
        label_mode: Literal['2D', '3D'] = '2D',
        x_color: ColorLike | Sequence[ColorLike] | None = None,
        y_color: ColorLike | Sequence[ColorLike] | None = None,
        z_color: ColorLike | Sequence[ColorLike] | None = None,
        opacity: float | VectorLike[float] = 0.3,
        position: VectorLike[float] = (0.0, 0.0, 0.0),
        orientation: VectorLike[float] = (0.0, 0.0, 0.0),
        origin: VectorLike[float] = (0.0, 0.0, 0.0),
        scale: VectorLike[float] = (1.0, 1.0, 1.0),
        user_matrix: MatrixLike[float] | None = None,
        **kwargs: Unpack[_OrthogonalPlanesKwargs],
    ):
        super().__init__()

        # Add dummy prop3d for calculating transformations
        self._prop3d = Actor()

        # Init plane actors
        self._plane_actors = (Actor(), Actor(), Actor())

        # Init planes from source
        self._geometry_source = OrthogonalPlanesSource(**kwargs)
        output = self._geometry_source.output
        # Change order of planes and rename
        # This is to match the standard 'x-y-z' API used by assemblies
        self.planes = pv.MultiBlock(dict(x=output['yz'], y=output['zx'], z=output['xy']))

        # Repeat for individual plane sources
        plane_sources = self._geometry_source._plane_sources
        self._plane_sources = plane_sources[1], plane_sources[2], plane_sources[0]

        for actor, dataset in zip(self._plane_actors, self.planes):
            actor.mapper = pv.DataSetMapper(dataset=dataset)

        # Add actors to assembly
        [self.AddPart(actor) for actor in self._plane_actors]

        # Init label actors and add to assembly
        self._axis_actors = (_AxisActor(), _AxisActor(), _AxisActor())
        self._label_properties = tuple(axis.GetTitleTextProperty() for axis in self._axis_actors)
        [self.AddPart(actor) for actor in self._axis_actors]

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
        self.opacity = opacity  # type: ignore[assignment]
        # Set default properties
        for actor in self._plane_actors:
            prop = actor.prop
            prop.show_edges = True
            prop.line_width = 3

        # Set text labels
        if labels is None:
            self.x_label = 'YZ' if x_label is None else x_label
            self.y_label = 'ZX' if y_label is None else y_label
            self.z_label = 'XY' if z_label is None else z_label
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
        self.label_mode = label_mode

        # Set default text properties
        for label in self._label_actor_iterator:
            prop = label.prop
            prop.bold = True
            prop.italic = True
            prop.justification_vertical = 'center'

        self.position = position  # type: ignore[assignment]
        self.orientation = orientation  # type: ignore[assignment]
        self.scale = scale  # type: ignore[assignment]
        self.origin = origin  # type: ignore[assignment]
        self.user_matrix = user_matrix  # type: ignore[assignment]

    def __repr__(self):
        """Representation of the planes assembly."""
        if self.user_matrix is None or np.array_equal(self.user_matrix, np.eye(4)):
            mat_info = 'Identity'
        else:
            mat_info = 'Set'
        bnds = self.bounds

        geometry_repr = repr(self._geometry_source).splitlines()[1:]

        attr = [
            f"{type(self).__name__} ({hex(id(self))})",
            *geometry_repr,
            f"  X label:                    '{self.x_label}'",
            f"  Y label:                    '{self.y_label}'",
            f"  Z label:                    '{self.z_label}'",
            f"  Label color:                {self.label_color}",
            f"  Show labels:                {self.show_labels}",
            f"  Label position:             {self.label_position}",
            f"  XY Color:                   {self.x_color}",
            f"  YZ Color:                   {self.y_color}",
            f"  ZX Color:                   {self.z_color}",
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
        """Return or set the assembly labels.

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
        """Text label for the xy-plane.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesAssembly()
        >>> axes_actor.x_label = 'This axis'
        >>> axes_actor.x_label
        'This axis'

        """
        return self._axis_actors[0].GetTitle()

    @x_label.setter
    def x_label(self, label: str):  # numpydoc ignore=GL08
        self._axis_actors[0].SetTitle(label)

    @property
    def y_label(self) -> str:  # numpydoc ignore=RT01
        """Text label for the yz-plane.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesAssembly()
        >>> axes_actor.y_label = 'This axis'
        >>> axes_actor.y_label
        'This axis'

        """
        return self._axis_actors[1].GetTitle()

    @y_label.setter
    def y_label(self, label: str):  # numpydoc ignore=GL08
        self._axis_actors[1].SetTitle(label)

    @property
    def z_label(self) -> str:  # numpydoc ignore=RT01
        """Text label for the zx-plane.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_actor = pv.AxesAssembly()
        >>> axes_actor.z_label = 'This axis'
        >>> axes_actor.z_label
        'This axis'

        """
        return self._axis_actors[2].GetTitle()

    @z_label.setter
    def z_label(self, label: str):  # numpydoc ignore=GL08
        self._axis_actors[2].SetTitle(label)

    @property
    def show_labels(self) -> bool:  # numpydoc ignore=RT01
        """Show or hide the text labels for the assembly."""
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
        for axis in self._axis_actors:
            axis.SetTitleScale(size)

    @property
    def label_position(self) -> tuple[int, int, int]:  # numpydoc ignore=RT01
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
        return self._label_position

    @label_position.setter
    def label_position(self, position: int | VectorLike[int] | None):  # numpydoc ignore=GL08
        self._label_position = _validation.validate_array3(
            position,
            broadcast=True,
            must_be_integer=[0, np.inf],
            name='Label position',
            dtype_out=int,
            to_tuple=True,
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

    @property
    def label_mode(self) -> Literal['2D', '3D']:  # numpydoc ignore=RT01
        """Mode to use for text labels.

        Mode must be either '2D' or '3D'. In '2D' mode, the label actors are 2D
        sprites and are always visible. In '3D' mode, the label actors polygons
        and may be occluded. The two modes also have minor differences in
        appearance as well as behavior in terms of how they follow the camera.
        """
        return self._label_mode

    @label_mode.setter
    def label_mode(self, mode: Literal['2D', '3D']):  # numpydoc ignore=GL08
        _validation.check_contains(item=mode, container=['2D', '3D'])
        self._label_mode = mode
        use_2D = mode == '2D'
        for axis in self._axis_actors:
            axis.SetUse2DMode(use_2D)

    @property
    def x_color(self) -> tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the xy-plane."""
        return self._plane_actors[0].prop.color

    @x_color.setter
    def x_color(self, color: ColorLike | Sequence[ColorLike]):  # numpydoc ignore=GL08
        self._plane_actors[0].prop.color = color

    @property
    def y_color(self) -> tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the yz-plane."""
        return self._plane_actors[1].prop.color

    @y_color.setter
    def y_color(self, color: ColorLike | Sequence[ColorLike]):  # numpydoc ignore=GL08
        self._plane_actors[1].prop.color = color

    @property
    def z_color(self) -> tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the zx-plane."""
        return self._plane_actors[2].prop.color

    @z_color.setter
    def z_color(self, color: ColorLike | Sequence[ColorLike]):  # numpydoc ignore=GL08
        self._plane_actors[2].prop.color = color

    @property
    def opacity(self) -> float:  # numpydoc ignore=RT01
        """Color of the zx-plane."""
        return self._plane_actors[2].prop.opacity

    @opacity.setter
    def opacity(self, opacity: float):  # numpydoc ignore=GL08
        valid_opacity = _validation.validate_array3(opacity, broadcast=True, dtype_out=float)
        self._opacity = valid_opacity
        for actor, opacity in zip(self._plane_actors, valid_opacity):
            actor.prop.opacity = opacity

    @property
    def camera(self):  # numpydoc ignore=RT01
        """Camera to use for displaying the labels."""
        if not hasattr(self, '_camera'):
            raise ValueError('Camera has not been')
        return self._camera

    @camera.setter
    def camera(self, camera):  # numpydoc ignore=GL08
        self._camera = camera
        for axis in self._axis_actors:
            axis.SetCamera(camera)

    def _update_label_positions(self):
        axis_actors = self._axis_actors
        plane_sources = self._plane_sources

        def set_axis_location(plane_id, location: int):
            this_plane_source = plane_sources[plane_id]
            this_axis_actor = axis_actors[plane_id]

            origin, point1, point2 = (
                np.array(this_plane_source.GetOrigin()),
                np.array(this_plane_source.GetPoint1()),
                np.array(this_plane_source.GetPoint2()),
            )

            vector1 = point1 - origin
            vector2 = point2 - origin

            corner_bottom_left = origin
            corner_bottom_right = origin + vector1
            corner_top_left = origin + vector2
            corner_top_right = corner_bottom_right + vector2

            midpoint_left = (corner_top_left + corner_bottom_left) / 2
            midpoint_right = (corner_bottom_right + corner_top_right) / 2
            midpoint_top = (corner_top_left + corner_top_right) / 2
            midpoint_bottom = (corner_bottom_left + corner_bottom_right) / 2

            # Order points counter-clockwise right side
            ordered_points = [
                midpoint_right,
                corner_top_right,
                midpoint_top,
                corner_top_left,
                midpoint_left,
                corner_bottom_left,
                midpoint_bottom,
                corner_bottom_right,
            ]
            # Duplicate first point as last point
            ordered_points.append(ordered_points[0])

            # Set axis points to specified location
            axis_point1, axis_point2 = ordered_points[location : location + 2]
            this_axis_actor.SetPoint1(axis_point1)
            this_axis_actor.SetPoint2(axis_point2)

            # Align axis type to its direction
            # NOTE: Using SetAxisType() doesn't really seem to have any effect
            axis_dir = np.abs(axis_point1 - axis_point2) / np.linalg.norm(axis_point1 - axis_point2)
            if np.allclose(axis_dir, [1, 0, 0]):
                this_axis_actor.SetAxisTypeToX()
            elif np.allclose(axis_dir, [0, 1, 0]):
                this_axis_actor.SetAxisTypeToY()
            elif np.allclose(axis_dir, [0, 0, 1]):
                this_axis_actor.SetAxisTypeToZ()
            else:
                raise RuntimeError(f"Unexpected axis direction! Got {axis_dir}.")

            # Re-scale the title text proportional to planes assembly
            # Values on the order of 0.01-0.05 seem to work best. Use a normalization
            # factor so that values for `label_size` are on the order of 10-50
            NORM_FACTOR = 1000
            scale = self.planes.length * self.label_size / NORM_FACTOR
            this_axis_actor.GetTitleActor().SetScale(scale)

        #          2          1
        #    +----------+----------+
        #  3 | (-i, +j) | (+i, +j) | 0
        #    +----------+----------+
        #  4 | (-i, -j) | (+i, -j) | 7
        #    +----------+----------+
        #          5          6

        positions = self.label_position
        set_axis_location(0, positions[0])
        set_axis_location(1, positions[1])
        set_axis_location(2, positions[2])

    def _set_prop3d_attr(self, name, value):
        # Set props for plane actors
        # Validate input by setting then getting from prop3d
        setattr(self._prop3d, name, value)
        valid_value = getattr(self._prop3d, name)
        [setattr(actor, name, valid_value) for actor in self._plane_actors]

        # Update labels
        self._update_label_positions()

    @property
    def scale(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the scaling factor applied to the assembly.

        Examples
        --------
        >>> import pyvista as pv
        >>> assembly = pv.AxesAssembly()
        >>> assembly.scale = (2.0, 2.0, 2.0)
        >>> assembly.scale
        (2.0, 2.0, 2.0)
        """
        return self._prop3d.scale

    @scale.setter
    def scale(self, scale: VectorLike[float]):  # numpydoc ignore=GL08
        self._set_prop3d_attr('scale', scale)

    @property
    def position(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the position of the assembly.

        Examples
        --------
        >>> import pyvista as pv
        >>> assembly = pv.AxesAssembly()
        >>> assembly.position = (1.0, 2.0, 3.0)
        >>> assembly.position
        (1.0, 2.0, 3.0)
        """
        return self._prop3d.position

    @position.setter
    def position(self, position: VectorLike[float]):  # numpydoc ignore=GL08
        self._set_prop3d_attr('position', position)

    @property
    def orientation(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Return or set the assembly's orientation angles.

        Orientation angles of the assembly which define rotations about the
        world's x-y-z axes. The angles are specified in degrees and in
        x-y-z order. However, the actual rotations are applied in the
        following order: :func:`~rotate_y` first, then :func:`~rotate_x`
        and finally :func:`~rotate_z`.

        Rotations are applied about the specified :attr:`~origin`.

        Examples
        --------
        Create assembly positioned above the origin and set its orientation.

        >>> import pyvista as pv
        >>> axes = pv.AxesAssembly(
        ...     position=(0, 0, 2), orientation=(45, 0, 0)
        ... )

        Create default non-oriented assembly as well for reference.

        >>> reference_axes = pv.AxesAssembly(
        ...     x_color='black', y_color='black', z_color='black'
        ... )

        Plot the assembly. Note how the axes are rotated about the origin ``(0, 0, 0)`` by
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
        """Return or set the origin of the assembly.

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
        Apply a 4x4 transformation to the assembly. This effectively translates the actor
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
        """Return the bounds of the assembly.

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
        """Return the center of the assembly.

        Examples
        --------
        >>> import pyvista as pv
        >>> assembly = pv.AxesAssembly()
        >>> assembly.center
        (0.44999999925494194, 0.44999999925494194, 0.44999999925494194)
        """
        bnds = self.bounds
        return (bnds[0] + bnds[1]) / 2, (bnds[1] + bnds[2]) / 2, (bnds[4] + bnds[5]) / 2

    @property
    def length(self) -> float:  # numpydoc ignore=RT01
        """Return the length of the assembly.

        Examples
        --------
        >>> import pyvista as pv
        >>> assembly = pv.AxesAssembly()
        >>> assembly.length
        1.9052558909067219
        """
        bnds = self.bounds
        min_bnds = np.array((bnds[0], bnds[2], bnds[4]))
        max_bnds = np.array((bnds[1], bnds[3], bnds[5]))
        return np.linalg.norm(max_bnds - min_bnds).tolist()


def _AxisActor():
    actor = _vtk.vtkAxisActor()

    # Only show the title
    actor.TitleVisibilityOn()
    actor.MinorTicksVisibleOff()
    actor.TickVisibilityOff()
    actor.DrawGridlinesOff()

    # Set empty tick labels
    labels = _vtk.vtkStringArray()
    labels.SetNumberOfTuples(0)
    # labels.SetValue(0, "")
    actor.SetLabels(labels)

    # Format title positioning
    actor.SetTitleOffset(0, 0)
    actor.SetLabelOffset(0)

    # For 2D mode only
    actor.SetVerticalOffsetXTitle2D(0)
    actor.SetHorizontalOffsetYTitle2D(0)
    actor.GetTitleTextProperty().SetVerticalJustificationToCentered()
    return actor
