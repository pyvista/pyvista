"""Axes assembly module."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import itertools
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import Sequence
from typing import TypedDict

import numpy as np

import pyvista as pv
from pyvista import BoundsLike
from pyvista.core import _validation
from pyvista.core.utilities.geometric_sources import AxesGeometrySource
from pyvista.core.utilities.geometric_sources import _AxisEnum
from pyvista.core.utilities.geometric_sources import _PartEnum
from pyvista.plotting import _vtk
from pyvista.plotting.actor import Actor
from pyvista.plotting.colors import Color
from pyvista.plotting.prop3d import _Prop3DMixin
from pyvista.plotting.text import Label

if TYPE_CHECKING:  # pragma: no cover
    import sys
    from typing import Iterator

    from pyvista.core._typing_core import MatrixLike
    from pyvista.core._typing_core import VectorLike
    from pyvista.core.dataset import DataSet
    from pyvista.plotting._typing import ColorLike

    if sys.version_info >= (3, 11):
        from typing import Unpack
    else:
        from typing_extensions import Unpack


class _AxesPropTuple(NamedTuple):
    x_shaft: float | str | ColorLike
    y_shaft: float | str | ColorLike
    z_shaft: float | str | ColorLike
    x_tip: float | str | ColorLike
    y_tip: float | str | ColorLike
    z_tip: float | str | ColorLike


class _AxesGeometryKwargs(TypedDict):
    shaft_type: AxesGeometrySource.GeometryTypes | DataSet
    shaft_radius: float
    shaft_length: float | VectorLike[float]
    tip_type: AxesGeometrySource.GeometryTypes | DataSet
    tip_radius: float
    tip_length: float | VectorLike[float]
    symmetric_bounds: bool


class _CubeFacesSource(TypedDict):
    center: VectorLike[float]
    x_length: float
    y_length: float
    z_length: float
    bounds: VectorLike[float] | None
    shrink: float | None
    explode: float | None
    names: Sequence[str]
    point_dtype: str


class _XYZTuple(NamedTuple):
    x: Any
    y: Any
    z: Any


class _XYZAssembly(_Prop3DMixin, _vtk.vtkPropAssembly, ABC):
    DEFAULT_LABELS = _XYZTuple('X', 'Y', 'Z')

    def __init__(
        self,
        *,
        xyz_actors: tuple[Any, Any, Any],
        xyz_label_actors: tuple[Any, Any, Any],
        x_label,
        y_label,
        z_label,
        labels,
        label_color,
        show_labels,
        label_position,
        label_size,
        x_color,
        y_color,
        z_color,
        position: VectorLike[float],
        orientation: VectorLike[float],
        origin: VectorLike[float],
        scale: float | VectorLike[float],
        user_matrix: MatrixLike[float] | None,
    ):
        super().__init__()

        def _make_xyz_tuple(xyz):
            def _get_tuple(actor_or_actors):
                return actor_or_actors if isinstance(actor_or_actors, tuple) else (actor_or_actors,)

            actor_tuples = [_get_tuple(actors) for actors in xyz]
            return _XYZTuple(*actor_tuples)

        self._assembly_actors = _make_xyz_tuple(xyz_actors)
        self._assembly_label_actors = _make_xyz_tuple(xyz_label_actors)

        # Add all actors to assembly
        for parts in (*self._assembly_actors, *self._assembly_label_actors):
            for part in parts:
                self.AddPart(part)

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

        # Set text labels
        if labels is None:
            self.x_label = self.DEFAULT_LABELS.x if x_label is None else x_label
            self.y_label = self.DEFAULT_LABELS.y if y_label is None else y_label
            self.z_label = self.DEFAULT_LABELS.z if z_label is None else z_label
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

        self.position = position  # type: ignore[assignment]
        self.orientation = orientation  # type: ignore[assignment]
        self.scale = scale  # type: ignore[assignment]
        self.origin = origin  # type: ignore[assignment]
        self.user_matrix = user_matrix  # type: ignore[assignment]

    def __new__(cls, *args, **kwargs):
        # Check subclasses have implemented abstract methods
        if hasattr(cls, '__abstractmethods__') and len(cls.__abstractmethods__) > 0:
            raise TypeError(f'Class {cls.__name__} must implement abstract methods {tuple(cls.__abstractmethods__)}')
        return super().__new__(cls, *args, **kwargs)

    @property
    def parts(self):
        collection = self.GetParts()
        return tuple([collection.GetItemAsObject(i) for i in range(collection.GetNumberOfItems())])

    @property
    def _label_actor_iterator(self) -> Iterator[Label]:
        return itertools.chain.from_iterable(self._assembly_label_actors)

    def _post_set_update(self):
        # Update prop3D attributes for shaft, tip, and label actors
        parts = self.parts
        for name in ['position', 'orientation', 'scale', 'origin', 'user_matrix']:
            # Only update values if modified
            value = getattr(self._prop3d, name)
            [
                setattr(part, name, value)
                for part in parts
                if not np.array_equal(getattr(part, name), value)
            ]

    def _get_bounds(self) -> BoundsLike:  # numpydoc ignore=RT01
        return self.GetBounds()

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
    @abstractmethod
    def labels(self):  # numpydoc ignore=RT01
        """XYZ labels."""

    @labels.setter
    @abstractmethod
    def labels(self, labels):  # numpydoc ignore=GL08
        """XYZ labels."""

    @property
    @abstractmethod
    def x_label(self):  # numpydoc ignore=RT01
        """Text label for the x-axis."""

    @x_label.setter
    @abstractmethod
    def x_label(self, label):  # numpydoc ignore=GL08
        """Text label for the x-axis."""

    @property
    @abstractmethod
    def y_label(self):  # numpydoc ignore=RT01
        """Text label for the y-axis."""

    @y_label.setter
    @abstractmethod
    def y_label(self, label):  # numpydoc ignore=GL08
        """Text label for the y-axis."""

    @property
    @abstractmethod
    def z_label(self):  # numpydoc ignore=RT01
        """Text label for the z-axis."""

    @z_label.setter
    @abstractmethod
    def z_label(self, label):  # numpydoc ignore=GL08
        """Text label for the z-axis."""

    @property
    @abstractmethod
    def label_size(self):  # numpydoc ignore=RT01
        """Size of the text labels."""

    @label_size.setter
    @abstractmethod
    def label_size(self, size):  # numpydoc ignore=GL08
        """Size of the text labels."""

    @property
    @abstractmethod
    def label_position(self):  # numpydoc ignore=RT01
        """Position of the text labels."""

    @label_position.setter
    @abstractmethod
    def label_position(self, position):  # numpydoc ignore=GL08
        """Position of the text labels."""

    @property
    @abstractmethod
    def label_color(self):  # numpydoc ignore=RT01
        """Color of the text labels."""

    @label_color.setter
    @abstractmethod
    def label_color(self, color):  # numpydoc ignore=GL08
        """Color of the text labels."""

    @property
    @abstractmethod
    def x_color(self):  # numpydoc ignore=RT01
        """Color of the x-axis actors."""

    @x_color.setter
    @abstractmethod
    def x_color(self, color):  # numpydoc ignore=GL08
        """Color of the x-axis actors."""

    @property
    @abstractmethod
    def y_color(self):  # numpydoc ignore=RT01
        """Color of the y-axis actors."""

    @y_color.setter
    @abstractmethod
    def y_color(self, color):  # numpydoc ignore=GL08
        """Color of the y-axis actors."""

    @property
    @abstractmethod
    def z_color(self):  # numpydoc ignore=RT01
        """Color of the z-axis actors."""

    @z_color.setter
    @abstractmethod
    def z_color(self, color):  # numpydoc ignore=GL08
        """Color of the z-axis actors."""


class AxesAssembly(_XYZAssembly):
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

    def _init_actors_from_source(self, geometry_source: AxesGeometrySource):
        # Init shaft and tip actors
        self._shaft_actors: tuple[Actor, Actor, Actor] = (Actor(), Actor(), Actor())
        self._tip_actors: tuple[Actor, Actor, Actor] = (Actor(), Actor(), Actor())
        self._shaft_and_tip_actors = (*self._shaft_actors, *self._tip_actors)

        # Init shaft and tip datasets
        self._shaft_and_tip_geometry_source = geometry_source
        shaft_tip_datasets = self._shaft_and_tip_geometry_source.output
        for actor, dataset in zip(self._shaft_and_tip_actors, shaft_tip_datasets):
            actor.mapper = pv.DataSetMapper(dataset=dataset)

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
        scale: float | VectorLike[float] = (1.0, 1.0, 1.0),
        user_matrix: MatrixLike[float] | None = None,
        **kwargs: Unpack[_AxesGeometryKwargs],
    ):
        # Init shaft and tip actors
        self._init_actors_from_source(AxesGeometrySource(symmetric=False, **kwargs))
        # Init label actors
        self._label_actors = (Label(), Label(), Label())

        _XYZAssembly.__init__(
            self,
            xyz_actors=tuple(zip(self._shaft_actors, self._tip_actors)),  # type: ignore[arg-type]
            xyz_label_actors=self._label_actors,
            x_label=x_label,
            y_label=y_label,
            z_label=z_label,
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
        )
        self._set_default_label_props()

    def _set_default_label_props(self):
        # TODO: implement set_text_prop() and use that instead
        for label in self._label_actor_iterator:
            prop = label.prop
            prop.bold = True
            prop.italic = True

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
    def labels(self) -> tuple[str, str, str]:  # numpydoc ignore=RT01
        """Return or set the axes labels.

        This property may be used as an alternative to using :attr:`x_label`,
        :attr:`y_label`, and :attr:`z_label` separately.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes_assembly = pv.AxesAssembly()
        >>> axes_assembly.labels = ['X Axis', 'Y Axis', 'Z Axis']
        >>> axes_assembly.labels
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
        >>> axes_assembly = pv.AxesAssembly()
        >>> axes_assembly.x_label = 'This axis'
        >>> axes_assembly.x_label
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
        >>> axes_assembly = pv.AxesAssembly()
        >>> axes_assembly.y_label = 'This axis'
        >>> axes_assembly.y_label
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
        >>> axes_assembly = pv.AxesAssembly()
        >>> axes_assembly.z_label = 'This axis'
        >>> axes_assembly.z_label
        'This axis'

        """
        return self._label_actors[2].input

    @z_label.setter
    def z_label(self, label: str):  # numpydoc ignore=GL08
        self._label_actors[2].input = label

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
        >>> axes_assembly = pv.AxesAssembly()
        >>> axes_assembly.label_position
        (0.8, 0.8, 0.8)
        >>> axes_assembly.label_position = 0.3
        >>> axes_assembly.label_position
        (0.3, 0.3, 0.3)
        >>> axes_assembly.label_position = (0.1, 0.4, 0.2)
        >>> axes_assembly.label_position
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

    @property
    def x_color(self) -> tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the x-axis shaft and tip."""
        return self.get_actor_prop('color')[_AxisEnum.x :: 3]

    @x_color.setter
    def x_color(self, color: ColorLike | Sequence[ColorLike]):  # numpydoc ignore=GL08
        self.set_actor_prop('color', color, axis=_AxisEnum.x.value)  # type: ignore[arg-type]

    @property
    def y_color(self) -> tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the y-axis shaft and tip."""
        return self.get_actor_prop('color')[_AxisEnum.y :: 3]

    @y_color.setter
    def y_color(self, color: ColorLike | Sequence[ColorLike]):  # numpydoc ignore=GL08
        self.set_actor_prop('color', color, axis=_AxisEnum.y.value)  # type: ignore[arg-type]

    @property
    def z_color(self) -> tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the z-axis shaft and tip."""
        return self.get_actor_prop('color')[_AxisEnum.z.value :: 3]

    @z_color.setter
    def z_color(self, color: ColorLike | Sequence[ColorLike]):  # numpydoc ignore=GL08
        self.set_actor_prop('color', color, axis=_AxisEnum.z.value)  # type: ignore[arg-type]

    def set_actor_prop(
        self,
        name: str,
        value: float | str | ColorLike | Sequence[float | str | ColorLike],
        axis: Literal['x', 'y', 'z', 'all'] = 'all',
        part: Literal['shaft', 'tip', 'all'] = 'all',
    ):
        """Set :class:`~pyvista.Property` attributes for the axes shaft and/or tip actors.

        This is a generalized setter method which sets the value of a specific
        :class:`~pyvista.Property` attribute for any combination of axis shaft or tip
        parts.

        Parameters
        ----------
        name : str
            Name of the :class:`~pyvista.Property` attribute to set.

        value : float | str | ColorLike | Sequence[float | str | ColorLike]
            Value to set the attribute to. If a single value, set all specified axes
            shaft(s) or tip(s) :class:`~pyvista.Property` attributes to this value.
            If a sequence of values, set the specified parts to these values.

        axis : str | int, default: 'all'
            Set :class:`~pyvista.Property` attributes for a specific part of the axes.
            Specify one of:

            - ``'x'``: only set the property for the x-axis.
            - ``'y'``: only set the property for the y-axis.
            - ``'z'``: only set the property for the z-axis.
            - ``'all'``: set the property for all three axes.

        part : str | int, default: 'all'
            Set the property for a specific part of the axes. Specify one of:

            - ``'shaft'``: only set the property for the axes shafts.
            - ``'tip'``: only set the property for the axes tips.
            - ``'all'``: set the property for axes shafts and tips.

        Examples
        --------
        Set :attr:`~pyvista.Property.ambient` for all axes shafts and tips to a
        single value.

        >>> import pyvista as pv
        >>> axes_assembly = pv.AxesAssembly()
        >>> axes_assembly.set_actor_prop('ambient', 0.7)
        >>> axes_assembly.get_actor_prop('ambient')
        _AxesPropTuple(x_shaft=0.7, y_shaft=0.7, z_shaft=0.7, x_tip=0.7, y_tip=0.7, z_tip=0.7)

        Set the property again, but this time set separate values for each part.

        >>> values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        >>> axes_assembly.set_actor_prop('ambient', values)
        >>> axes_assembly.get_actor_prop('ambient')
        _AxesPropTuple(x_shaft=0.1, y_shaft=0.2, z_shaft=0.3, x_tip=0.4, y_tip=0.5, z_tip=0.6)

        Set :attr:`~pyvista.Property.opacity` for the x-axis only. The property is set
        for both the axis shaft and tip by default.

        >>> axes_assembly.set_actor_prop('opacity', 0.5, axis='x')
        >>> axes_assembly.get_actor_prop('opacity')
        _AxesPropTuple(x_shaft=0.5, y_shaft=1.0, z_shaft=1.0, x_tip=0.5, y_tip=1.0, z_tip=1.0)

        Set the property again, but this time set separate values for the shaft and tip.

        >>> axes_assembly.set_actor_prop('opacity', [0.3, 0.7], axis='x')
        >>> axes_assembly.get_actor_prop('opacity')
        _AxesPropTuple(x_shaft=0.3, y_shaft=1.0, z_shaft=1.0, x_tip=0.7, y_tip=1.0, z_tip=1.0)

        Set :attr:`~pyvista.Property.show_edges` for the axes shafts only. The property
        is set for all axes by default.

        >>> axes_assembly.set_actor_prop('show_edges', True, part='shaft')
        >>> axes_assembly.get_actor_prop('show_edges')
        _AxesPropTuple(x_shaft=True, y_shaft=True, z_shaft=True, x_tip=False, y_tip=False, z_tip=False)

        Set the property again, but this time set separate values for each shaft.

        >>> axes_assembly.set_actor_prop(
        ...     'show_edges', [True, False, True], part='shaft'
        ... )
        >>> axes_assembly.get_actor_prop('show_edges')
        _AxesPropTuple(x_shaft=True, y_shaft=False, z_shaft=True, x_tip=False, y_tip=False, z_tip=False)

        Set :attr:`~pyvista.Property.style` for a single axis and specific part.

        >>> axes_assembly.set_actor_prop(
        ...     'style', 'wireframe', axis='x', part='shaft'
        ... )
        >>> axes_assembly.get_actor_prop('style')
        _AxesPropTuple(x_shaft='Wireframe', y_shaft='Surface', z_shaft='Surface', x_tip='Surface', y_tip='Surface', z_tip='Surface')
        """
        actors = self._filter_part_actors(axis=axis, part=part)
        values: Sequence[float | str | ColorLike]

        # Validate input as a sequence of values
        if 'color' in name:
            # Special case for color inputs
            if axis == 'all' and part == 'all':
                n_values = 6
            elif part == 'all':
                n_values = 2
            elif axis == 'all':
                n_values = 3
            else:
                n_values = 1
            values = _validate_color_sequence(value, n_values)
        elif isinstance(value, Sequence) and not isinstance(value, str):
            # Number sequence
            values = value
        else:
            # Scalar number or string
            values = [value] * len(actors)

        if len(values) != len(actors):
            raise ValueError(
                f"Number of values ({len(values)}) in {value} must match the number of actors ({len(actors)}) for axis '{axis}' and part '{part}'"
            )

        # Sequence is valid, now set values
        for actor, val in zip(actors, values):
            setattr(actor.prop, name, val)

    def get_actor_prop(self, name: str):
        """Get :class:`~pyvista.Property` attributes for the axes shaft and/or tip actors.

        This is a generalized getter method which returns the value of
        a specific :class:`pyvista.Property` attribute for all shafts and tips.

        Parameters
        ----------
        name : str
            Name of the :class:`~pyvista.Property` attribute to get.

        Returns
        -------
        tuple
            Named tuple with attribute values for the axes shafts and tips.
            The values are ordered ``(x_shaft, y_shaft, z_shaft, x_tip, y_tip, z_tip)``.

        Examples
        --------
        Get the ambient property of the axes shafts and tips.

        >>> import pyvista as pv
        >>> axes_assembly = pv.AxesAssembly()
        >>> axes_assembly.get_actor_prop('ambient')
        _AxesPropTuple(x_shaft=0.0, y_shaft=0.0, z_shaft=0.0, x_tip=0.0, y_tip=0.0, z_tip=0.0)

        """
        prop_values = [getattr(actor.prop, name) for actor in self._shaft_and_tip_actors]
        return _AxesPropTuple(*prop_values)

    def _filter_part_actors(
        self,
        axis: Literal['x', 'y', 'z', 'all'] = 'all',
        part: Literal['shaft', 'tip', 'all'] = 'all',
    ):
        valid_axis = [0, 1, 2, 'x', 'y', 'z', 'all']
        valid_axis_official = valid_axis[3:]
        if axis not in valid_axis:
            raise ValueError(f"Axis must be one of {valid_axis_official}.")
        valid_part = [0, 1, 'shaft', 'tip', 'all']
        valid_part_official = valid_part[2:]
        if part not in valid_part:
            raise ValueError(f"Part must be one of {valid_part_official}.")

        # Create ordered list of filtered actors
        # Iterate over parts in <shaft-xyz> then <tip-xyz> order
        actors: list[Actor] = []
        for part_type, axis_num in itertools.product(_PartEnum, _AxisEnum):
            if part in [part_type.name, part_type.value, 'all']:
                if axis in [axis_num.name, axis_num.value, 'all']:
                    # Add actor to list
                    if part_type == _PartEnum.shaft:
                        actors.append(self._shaft_actors[axis_num])
                    else:
                        actors.append(self._tip_actors[axis_num])

        return actors

    def _get_offset_label_position_vectors(self, position_scalars: tuple[float, float, float]):
        # Create position vectors
        position_vectors = np.diag(position_scalars)

        # Offset label positions radially by the tip radius
        tip_radius = self._shaft_and_tip_geometry_source.tip_radius
        offset_array = np.diag([tip_radius] * 3)
        radial_offset1 = np.roll(offset_array, shift=1, axis=1)
        radial_offset2 = np.roll(offset_array, shift=-1, axis=1)

        position_vectors += radial_offset1 + radial_offset2
        return position_vectors

    def _update_label_positions(self):
        labels = self._label_actors
        position_vectors = self._get_offset_label_position_vectors(self.label_position)
        for label, position in zip(labels, position_vectors):
            label.relative_position = position


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
        scale: float | VectorLike[float] = (1.0, 1.0, 1.0),
        user_matrix: MatrixLike[float] | None = None,
        **kwargs: Unpack[_AxesGeometryKwargs],
    ):
        # Init shaft and tip actors
        self._init_actors_from_source(AxesGeometrySource(symmetric=True, **kwargs))
        # Init label actors
        self._label_actors = (Label(), Label(), Label())
        self._label_actors_symmetric = (Label(), Label(), Label())

        _XYZAssembly.__init__(
            self,
            xyz_actors=tuple(zip(self._shaft_actors, self._tip_actors)),  # type: ignore[arg-type]
            xyz_label_actors=tuple(zip(self._label_actors, self._label_actors_symmetric)),  # type: ignore[arg-type]
            x_label=x_label,
            y_label=y_label,
            z_label=z_label,
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
        )
        self._set_default_label_props()

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
        # Update plus labels using parent method
        AxesAssembly._update_label_positions(self)

        # Update minus labels
        label_position = self.label_position
        label_position_minus = (-label_position[0], -label_position[1], -label_position[2])
        labels_minus = self._label_actors_symmetric
        vector_position_minus = self._get_offset_label_position_vectors(label_position_minus)
        for label, position in zip(labels_minus, vector_position_minus):
            label.relative_position = position


class LabeledCubeAssembly(_XYZAssembly):
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

    Shrink the faces so they appear as a single point and render them as spheres.

    >>> cube_faces_source.shrink = 1e-8
    >>> cube_faces_source.update()

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(mesh, color='tomato')
    >>> _ = pl.add_mesh(
    ...     output,
    ...     style='points',
    ...     render_points_as_spheres=True,
    ...     point_size=20,
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
        **kwargs: Unpack[_CubeFacesSource],
    ):
        # Init geometry
        self._geometry_source = pv.CubeFacesSource(**kwargs)
        # Init face actors
        self._face_actors = tuple(
            [
                Actor(mapper=pv.DataSetMapper(dataset=dataset))
                for dataset in self._geometry_source.output
            ]
        )
        self._face_actors_plus = self._face_actors[0::2]
        self._face_actors_minus = self._face_actors[1::2]

        # Init label actors
        self._label_actors = tuple(Label() for _ in range(6))
        self._label_actors_plus = self._label_actors[0::2]
        self._label_actors_minus = self._label_actors[1::2]

        _XYZAssembly.__init__(
            self,
            xyz_actors=tuple(zip(self._face_actors_plus, self._face_actors_minus)),  # type: ignore[arg-type]
            xyz_label_actors=tuple(zip(self._label_actors_plus, self._label_actors_minus)),  # type: ignore[arg-type]
            x_label=x_label,
            y_label=y_label,
            z_label=z_label,
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
        )
        for label in self._label_actor_iterator:
            prop = label.prop
            prop.bold = True
