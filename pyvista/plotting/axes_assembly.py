"""Axes assembly module."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from functools import wraps
import itertools
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import TypedDict

import numpy as np

import pyvista as pv
from pyvista import BoundsTuple
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core import _validation
from pyvista.core._validation.validate import _validate_color_sequence
from pyvista.core.utilities.geometric_sources import AxesGeometrySource
from pyvista.core.utilities.geometric_sources import OrthogonalPlanesSource
from pyvista.core.utilities.geometric_sources import _AxisEnum
from pyvista.core.utilities.geometric_sources import _PartEnum
from pyvista.core.utilities.misc import _NameMixin
from pyvista.core.utilities.misc import _NoNewAttrMixin
from pyvista.core.utilities.misc import abstract_class
from pyvista.plotting import _vtk
from pyvista.plotting.actor import Actor
from pyvista.plotting.colors import Color
from pyvista.plotting.prop3d import Prop3D
from pyvista.plotting.prop3d import _Prop3DMixin
from pyvista.plotting.text import Label
from pyvista.plotting.text import TextProperty

if TYPE_CHECKING:
    from collections.abc import Iterator
    import sys

    from pyvista.core._typing_core import MatrixLike
    from pyvista.core._typing_core import VectorLike
    from pyvista.core.dataset import DataSet
    from pyvista.plotting._typing import ColorLike

    if sys.version_info >= (3, 11):
        from typing import Unpack
    else:
        from typing_extensions import Unpack

ScaleModeOptions = Literal['uniform', 'normalized_shape']


class _AxesPropTuple(NamedTuple):
    x_shaft: float | str | ColorLike
    y_shaft: float | str | ColorLike
    z_shaft: float | str | ColorLike
    x_tip: float | str | ColorLike
    y_tip: float | str | ColorLike
    z_tip: float | str | ColorLike


class _OrthogonalPlanesKwargs(TypedDict):
    bounds: VectorLike[float]
    resolution: int | VectorLike[int]
    normal_sign: Literal['+', '-'] | Sequence[str]


class _XYZTuple(NamedTuple):
    x: Any
    y: Any
    z: Any


@abstract_class
class _XYZAssembly(
    _NoNewAttrMixin,
    _vtk.DisableVtkSnakeCase,
    _Prop3DMixin,
    _NameMixin,
    _vtk.vtkPropAssembly,
):
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
        name: str | None = None,
    ):
        super().__init__()

        def _make_xyz_tuple(xyz):
            def _get_tuple(actor_or_actors):
                return (
                    actor_or_actors if isinstance(actor_or_actors, tuple) else (actor_or_actors,)
                )

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
            msg = (
                "Cannot initialize '{}' and 'labels' properties together. "
                'Specify one or the other, not both.'
            )
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

        self.position = position
        self.orientation = orientation
        self.scale = scale  # type: ignore[assignment]
        self.origin = origin
        self.user_matrix = user_matrix

        self._name = name

    @property
    def parts(self):
        collection = self.GetParts()
        return tuple(collection.GetItemAsObject(i) for i in range(collection.GetNumberOfItems()))

    @property
    def _label_actor_iterator(self) -> Iterator[Label]:
        return itertools.chain.from_iterable(self._assembly_label_actors)

    def _post_set_update(self):
        # Update prop3D attributes for all assembly parts
        parts = self.parts
        new_matrix = pv.array_from_vtkmatrix(self._prop3d.GetMatrix())
        for part in parts:
            if isinstance(part, (Prop3D, _Prop3DMixin)) and not np.array_equal(
                part.user_matrix, new_matrix
            ):
                part.user_matrix = new_matrix

    def _get_bounds(self) -> BoundsTuple:  # numpydoc ignore=RT01
        return BoundsTuple(*self.GetBounds())

    @property
    def show_labels(self) -> bool:  # numpydoc ignore=RT01
        """Show or hide the text labels for the axes."""
        return self._show_labels

    @show_labels.setter
    def show_labels(self, value: bool):
        self._show_labels = value
        for label in self._label_actor_iterator:
            label.SetVisibility(value)

    @property
    @abstractmethod
    def labels(self):  # numpydoc ignore=RT01
        """XYZ labels."""

    @labels.setter
    @abstractmethod
    def labels(self, labels):
        """XYZ labels."""

    @property
    @abstractmethod
    def x_label(self):  # numpydoc ignore=RT01
        """Text label for the x-axis."""

    @x_label.setter
    @abstractmethod
    def x_label(self, label):
        """Text label for the x-axis."""

    @property
    @abstractmethod
    def y_label(self):  # numpydoc ignore=RT01
        """Text label for the y-axis."""

    @y_label.setter
    @abstractmethod
    def y_label(self, label):
        """Text label for the y-axis."""

    @property
    @abstractmethod
    def z_label(self):  # numpydoc ignore=RT01
        """Text label for the z-axis."""

    @z_label.setter
    @abstractmethod
    def z_label(self, label):
        """Text label for the z-axis."""

    @property
    @abstractmethod
    def label_size(self):  # numpydoc ignore=RT01
        """Size of the text labels."""

    @label_size.setter
    @abstractmethod
    def label_size(self, size):
        """Size of the text labels."""

    @property
    @abstractmethod
    def label_position(self):  # numpydoc ignore=RT01
        """Position of the text labels."""

    @label_position.setter
    @abstractmethod
    def label_position(self, position):
        """Position of the text labels."""

    @property
    def label_color(self) -> Color:  # numpydoc ignore=RT01
        """Color of the text labels."""
        return self._label_color

    @label_color.setter
    def label_color(self, color: ColorLike):
        valid_color = Color(color)
        self._label_color = valid_color
        for label in self._label_actor_iterator:
            label.prop.color = valid_color

    @property
    @abstractmethod
    def x_color(self):  # numpydoc ignore=RT01
        """Color of the x-axis actors."""

    @x_color.setter
    @abstractmethod
    def x_color(self, color):
        """Color of the x-axis actors."""

    @property
    @abstractmethod
    def y_color(self):  # numpydoc ignore=RT01
        """Color of the y-axis actors."""

    @y_color.setter
    @abstractmethod
    def y_color(self, color):
        """Color of the y-axis actors."""

    @property
    @abstractmethod
    def z_color(self):  # numpydoc ignore=RT01
        """Color of the z-axis actors."""

    @z_color.setter
    @abstractmethod
    def z_color(self, color):
        """Color of the z-axis actors."""


class AxesAssembly(_XYZAssembly):
    """Assembly of arrow-style axes parts.

    The axes may be used as a widget or added to a scene.

    Parameters
    ----------
    shaft_type : str | DataSet, default: 'cylinder'
        Shaft type for all axes. Can be any of the following:

        - ``'cylinder'``
        - ``'sphere'``
        - ``'hemisphere'``
        - ``'cone'``
        - ``'pyramid'``
        - ``'cube'``
        - ``'octahedron'``

        Alternatively, any arbitrary 3-dimensional :class:`pyvista.DataSet` may be
        specified. In this case, the dataset must be oriented such that it "points" in
        the positive z direction.

        .. versionadded:: 0.47

    shaft_radius : float | VectorLike[float], default: 0.025
        Radius of the axes shafts.

        .. versionadded:: 0.47

    shaft_length : float | VectorLike[float], default: 0.8
        Length of the shaft for each axis.

        .. versionadded:: 0.47

    tip_type : str | DataSet, default: 'cone'
        Tip type for all axes. Can be any of the following:

        - ``'cylinder'``
        - ``'sphere'``
        - ``'hemisphere'``
        - ``'cone'``
        - ``'pyramid'``
        - ``'cube'``
        - ``'octahedron'``

        Alternatively, any arbitrary 3-dimensional :class:`pyvista.DataSet` may be
        specified. In this case, the dataset must be oriented such that it "points" in
        the positive z direction.

        .. versionadded:: 0.47

    tip_radius : float | VectorLike[float], default: 0.1
        Radius of the axes tips.

        .. versionadded:: 0.47

    tip_length : float | VectorLike[float], default: 0.2
        Length of the tip for each axis.

        .. versionadded:: 0.47

    symmetric_bounds : bool, default: False
        Make the bounds of the axes symmetric. This option is similar to
        :attr:`symmetric`, except only the bounds are made to be symmetric,
        not the actual geometry. Has no effect if :attr:`symmetric` is ``True``.

        .. versionadded:: 0.47

    scale_mode
        Blarg

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
        rotations are defined by the :attr:`~pyvista.Prop3D.orientation`.

    scale : VectorLike[float], default: (1.0, 1.0, 1.0)
        Scaling factor applied to the axes.

    user_matrix : MatrixLike[float], optional
        A 4x4 transformation matrix applied to the axes. Defaults to the identity matrix.
        The user matrix is the last transformation applied to the actor.

    name : str, optional
        The name of this assembly used when tracking on a plotter.

        .. versionadded:: 0.45

    See Also
    --------
    AxesAssemblySymmetric

    :ref:`axes_objects_example`
        Example showing different axes objects.

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

    >>> axes = pv.AxesAssembly(position=(1.0, 2.0, 3.0), orientation=(10, 20, 30))
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
        # Get output without updating source, since source will be updated when setting actor scale
        shaft_tip_datasets = self._shaft_and_tip_geometry_source._output
        for actor, dataset in zip(self._shaft_and_tip_actors, shaft_tip_datasets, strict=True):
            actor.mapper = pv.DataSetMapper(dataset=dataset)

        # Length and radii set on this object may differ from the actual values set on the source
        self._shaft_length = geometry_source.shaft_length
        self._tip_length = geometry_source.tip_length
        self._shaft_radius = geometry_source.shaft_radius
        self._tip_radius = geometry_source.tip_radius

    def __init__(
        self,
        *,
        shaft_type: AxesGeometrySource.GeometryTypes | DataSet = 'cylinder',
        shaft_radius: float | VectorLike[float] = 0.025,
        shaft_length: float | VectorLike[float] = 0.8,
        tip_type: AxesGeometrySource.GeometryTypes | DataSet = 'cone',
        tip_radius: float | VectorLike[float] = 0.1,
        tip_length: float | VectorLike[float] = 0.2,
        symmetric_bounds: bool = False,
        scale_mode: ScaleModeOptions = 'uniform',
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
        name: str | None = None,
    ):
        self._scale_mode = scale_mode
        # Init shaft and tip actors
        source = AxesGeometrySource(
            shaft_type=shaft_type,
            shaft_radius=shaft_radius,
            shaft_length=shaft_length,
            tip_type=tip_type,
            tip_radius=tip_radius,
            tip_length=tip_length,
            symmetric_bounds=symmetric_bounds,
            symmetric=False,
        )
        self._init_actors_from_source(source)
        # Init label actors
        self._label_actors = (Label(), Label(), Label())

        _XYZAssembly.__init__(
            self,
            xyz_actors=tuple(zip(self._shaft_actors, self._tip_actors, strict=True)),  # type: ignore[arg-type]
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
            name=name,
        )
        self._set_default_label_props()

    def _set_default_label_props(self):
        # TODO: implement set_text_prop() and use that instead
        for label in self._label_actor_iterator:
            prop = label.prop
            prop.bold = True
            prop.italic = True
            prop.enable_shadow()
            prop.SetShadowOffset(-1, 1)

    def __repr__(self):
        """Representation of the axes assembly."""
        mat_info = 'Identity' if np.array_equal(self.user_matrix, np.eye(4)) else 'Set'
        bnds = self.bounds

        geometry_repr = repr(self._shaft_and_tip_geometry_source).splitlines()[1:]

        attr = [
            f'{type(self).__name__} ({hex(id(self))})',
            *geometry_repr,
            f"  X label:                    '{self.x_label}'",
            f"  Y label:                    '{self.y_label}'",
            f"  Z label:                    '{self.z_label}'",
            f'  Label color:                {self.label_color}',
            f'  Show labels:                {self.show_labels}',
            f'  Label position:             {self.label_position}',
            '  X Color:                                     ',
            f'      Shaft                   {self.x_color[0]}',
            f'      Tip                     {self.x_color[1]}',
            '  Y Color:                                     ',
            f'      Shaft                   {self.y_color[0]}',
            f'      Tip                     {self.y_color[1]}',
            '  Z Color:                                     ',
            f'      Shaft                   {self.z_color[0]}',
            f'      Tip                     {self.z_color[1]}',
            f'  Position:                   {self.position}',
            f'  Orientation:                {self.orientation}',
            f'  Origin:                     {self.origin}',
            f'  Scale:                      {self.scale}',
            f'  User matrix:                {mat_info}',
            f'  X Bounds                    {bnds.x_min:.3E}, {bnds.x_max:.3E}',
            f'  Y Bounds                    {bnds.y_min:.3E}, {bnds.y_max:.3E}',
            f'  Z Bounds                    {bnds.z_min:.3E}, {bnds.z_max:.3E}',
        ]
        return '\n'.join(attr)

    @property
    @wraps(AxesGeometrySource.shaft_length.fget)  # type: ignore[attr-defined]
    def shaft_length(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Wrap AxesGeometrySource."""
        return self._shaft_length

    @shaft_length.setter
    @wraps(AxesGeometrySource.shaft_length.fset)  # type: ignore[attr-defined]
    def shaft_length(self, length: float | VectorLike[float]) -> None:
        """Wrap AxesGeometrySource."""
        # Set value on source to validate
        self._shaft_and_tip_geometry_source.shaft_length = length
        # Store value on this object
        self._shaft_length = self._shaft_and_tip_geometry_source.shaft_length
        # Update geometry. The geometry may modify its value internally.
        self._shaft_and_tip_geometry_source.update()

    @property
    @wraps(AxesGeometrySource.tip_length.fget)  # type: ignore[attr-defined]
    def tip_length(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Wrap AxesGeometrySource."""
        return self._shaft_and_tip_geometry_source.tip_length

    @tip_length.setter
    @wraps(AxesGeometrySource.tip_length.fset)  # type: ignore[attr-defined]
    def tip_length(self, length: float | VectorLike[float]) -> None:
        """Wrap AxesGeometrySource."""
        # Set value on source to validate
        self._shaft_and_tip_geometry_source.tip_length = length
        # Store value on this object
        self._tip_length = self._shaft_and_tip_geometry_source.tip_length
        # Update geometry. The geometry may modify its value internally.
        self._shaft_and_tip_geometry_source.update()

    @property
    @wraps(AxesGeometrySource.shaft_radius.fget)  # type: ignore[attr-defined]
    def shaft_radius(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Wrap AxesGeometrySource."""
        return self._shaft_and_tip_geometry_source.shaft_radius

    @shaft_radius.setter
    @wraps(AxesGeometrySource.shaft_radius.fset)  # type: ignore[attr-defined]
    def shaft_radius(self, radius: float | VectorLike[float]) -> None:
        """Wrap AxesGeometrySource."""
        # Set value on source to validate
        self._shaft_and_tip_geometry_source.shaft_radius = radius
        # Store value on this object
        self._shaft_radius = self._shaft_and_tip_geometry_source.shaft_radius
        # Update geometry. The geometry may modify its value internally.
        self._shaft_and_tip_geometry_source.update()

    @property
    @wraps(AxesGeometrySource.tip_radius.fget)  # type: ignore[attr-defined]
    def tip_radius(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Wrap AxesGeometrySource."""
        return self._shaft_and_tip_geometry_source.tip_radius

    @tip_radius.setter
    @wraps(AxesGeometrySource.tip_radius.fset)  # type: ignore[attr-defined]
    def tip_radius(self, radius: float | VectorLike[float]) -> None:
        """Wrap AxesGeometrySource."""
        # Set value on source to validate
        self._shaft_and_tip_geometry_source.tip_radius = radius
        # Store value on this object
        self._tip_radius = self._shaft_and_tip_geometry_source.tip_radius
        # Update geometry. The geometry may modify its value internally.
        self._shaft_and_tip_geometry_source.update()

    @property
    @wraps(AxesGeometrySource.shaft_type.fget)  # type: ignore[attr-defined]
    def shaft_type(self) -> str:  # numpydoc ignore=RT01
        """Wrap AxesGeometrySource."""
        return self._shaft_and_tip_geometry_source.shaft_type

    @shaft_type.setter
    @wraps(AxesGeometrySource.shaft_type.fset)  # type: ignore[attr-defined]
    def shaft_type(self, shaft_type: AxesGeometrySource.GeometryTypes | DataSet) -> None:
        """Wrap AxesGeometrySource."""
        self._shaft_and_tip_geometry_source.shaft_type = shaft_type

    @property
    @wraps(AxesGeometrySource.tip_type.fget)  # type: ignore[attr-defined]
    def tip_type(self) -> str:  # numpydoc ignore=RT01
        """Wrap AxesGeometrySource."""
        return self._shaft_and_tip_geometry_source.tip_type

    @tip_type.setter
    @wraps(AxesGeometrySource.tip_type.fset)  # type: ignore[attr-defined]
    def tip_type(self, tip_type: AxesGeometrySource.GeometryTypes | DataSet) -> None:
        """Wrap AxesGeometrySource."""
        self._shaft_and_tip_geometry_source.tip_type = tip_type

    @property
    @wraps(Prop3D.scale.fget)  # type: ignore[attr-defined]
    def scale(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Wrap Prop3D.scale."""
        return _Prop3DMixin.scale.fget(self)

    @scale.setter
    @wraps(Prop3D.scale.fset)  # type: ignore[attr-defined]
    def scale(self, scale: float | VectorLike[float]):
        """Wrap Prop3D.scale."""
        _Prop3DMixin.scale.fset(self, scale)
        source = self._shaft_and_tip_geometry_source
        source._actor_scale = (1.0, 1.0, 1.0) if self.scale_mode == 'uniform' else self.scale
        source.update()
        self._update_label_positions()

    @property
    def scale_mode(self) -> ScaleModeOptions:  # numpydoc ignore=RT01
        """Set or return the scaling mode."""
        return self._scale_mode

    @scale_mode.setter
    def scale_mode(self, mode: ScaleModeOptions) -> None:
        self._scale_mode = mode

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
    def labels(self, labels: list[str] | tuple[str, str, str]):
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
    def x_label(self, label: str):
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
    def y_label(self, label: str):
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
    def z_label(self, label: str):
        self._label_actors[2].input = label

    @property
    def label_size(self) -> int:  # numpydoc ignore=RT01
        """Size of the text labels.

        Must be a positive integer.
        """
        return self._label_size

    @label_size.setter
    def label_size(self, size: int):
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
    def label_position(self, position: float | VectorLike[float] | None):
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
    def x_color(self) -> tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the x-axis shaft and tip."""
        return self.get_actor_prop('color')[_AxisEnum.x :: 3]

    @x_color.setter
    def x_color(self, color: ColorLike | Sequence[ColorLike]):
        self.set_actor_prop('color', color, axis=_AxisEnum.x.value)  # type: ignore[arg-type]

    @property
    def y_color(self) -> tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the y-axis shaft and tip."""
        return self.get_actor_prop('color')[_AxisEnum.y :: 3]

    @y_color.setter
    def y_color(self, color: ColorLike | Sequence[ColorLike]):
        self.set_actor_prop('color', color, axis=_AxisEnum.y.value)  # type: ignore[arg-type]

    @property
    def z_color(self) -> tuple[Color, Color]:  # numpydoc ignore=RT01
        """Color of the z-axis shaft and tip."""
        return self.get_actor_prop('color')[_AxisEnum.z.value :: 3]

    @z_color.setter
    def z_color(self, color: ColorLike | Sequence[ColorLike]):
        self.set_actor_prop('color', color, axis=_AxisEnum.z.value)  # type: ignore[arg-type]

    @_deprecate_positional_args(allowed=['name', 'value'])
    def set_actor_prop(  # noqa: PLR0917
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
        >>> axes_assembly.get_actor_prop('ambient')  # doctest: +NORMALIZE_WHITESPACE
        _AxesPropTuple(x_shaft=0.7,
                       y_shaft=0.7,
                       z_shaft=0.7,
                       x_tip=0.7,
                       y_tip=0.7,
                       z_tip=0.7)

        Set the property again, but this time set separate values for each part.

        >>> values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        >>> axes_assembly.set_actor_prop('ambient', values)
        >>> axes_assembly.get_actor_prop('ambient')  # doctest: +NORMALIZE_WHITESPACE
        _AxesPropTuple(x_shaft=0.1,
                       y_shaft=0.2,
                       z_shaft=0.3,
                       x_tip=0.4,
                       y_tip=0.5,
                       z_tip=0.6)

        Set :attr:`~pyvista.Property.opacity` for the x-axis only. The property is set
        for both the axis shaft and tip by default.

        >>> axes_assembly.set_actor_prop('opacity', 0.5, axis='x')
        >>> axes_assembly.get_actor_prop('opacity')  # doctest: +NORMALIZE_WHITESPACE
        _AxesPropTuple(x_shaft=0.5,
                       y_shaft=1.0,
                       z_shaft=1.0,
                       x_tip=0.5,
                       y_tip=1.0,
                       z_tip=1.0)

        Set the property again, but this time set separate values for the shaft and tip.

        >>> axes_assembly.set_actor_prop('opacity', [0.3, 0.7], axis='x')
        >>> axes_assembly.get_actor_prop('opacity')  # doctest: +NORMALIZE_WHITESPACE
        _AxesPropTuple(x_shaft=0.3,
                       y_shaft=1.0,
                       z_shaft=1.0,
                       x_tip=0.7,
                       y_tip=1.0,
                       z_tip=1.0)

        Set :attr:`~pyvista.Property.show_edges` for the axes shafts only. The property
        is set for all axes by default.

        >>> axes_assembly.set_actor_prop('show_edges', True, part='shaft')
        >>> axes_assembly.get_actor_prop(
        ...     'show_edges'
        ... )  # doctest: +NORMALIZE_WHITESPACE
        _AxesPropTuple(x_shaft=True,
                       y_shaft=True,
                       z_shaft=True,
                       x_tip=False,
                       y_tip=False,
                       z_tip=False)

        Set the property again, but this time set separate values for each shaft.

        >>> axes_assembly.set_actor_prop(
        ...     'show_edges', [True, False, True], part='shaft'
        ... )
        >>> axes_assembly.get_actor_prop(
        ...     'show_edges'
        ... )  # doctest: +NORMALIZE_WHITESPACE
        _AxesPropTuple(x_shaft=True,
                       y_shaft=False,
                       z_shaft=True,
                       x_tip=False,
                       y_tip=False,
                       z_tip=False)

        Set :attr:`~pyvista.Property.style` for a single axis and specific part.

        >>> axes_assembly.set_actor_prop('style', 'wireframe', axis='x', part='shaft')
        >>> axes_assembly.get_actor_prop('style')  # doctest: +NORMALIZE_WHITESPACE
        _AxesPropTuple(x_shaft='Wireframe',
                       y_shaft='Surface',
                       z_shaft='Surface',
                       x_tip='Surface',
                       y_tip='Surface',
                       z_tip='Surface')

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
            values = _validate_color_sequence(value, n_values)  # type: ignore[arg-type]
        elif isinstance(value, Sequence) and not isinstance(value, str):
            # Number sequence
            values = value
        else:
            # Scalar number or string
            values = [value] * len(actors)

        if len(values) != len(actors):
            msg = (
                f'Number of values ({len(values)}) in {value} must match the number of '
                f"actors ({len(actors)}) for axis '{axis}' and part '{part}'"
            )
            raise ValueError(msg)

        # Sequence is valid, now set values
        for actor, val in zip(actors, values, strict=True):
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
            msg = f'Axis must be one of {valid_axis_official}.'
            raise ValueError(msg)
        valid_part = [0, 1, 'shaft', 'tip', 'all']
        valid_part_official = valid_part[2:]
        if part not in valid_part:
            msg = f'Part must be one of {valid_part_official}.'
            raise ValueError(msg)

        # Create ordered list of filtered actors
        # Iterate over parts in <shaft-xyz> then <tip-xyz> order
        actors: list[Actor] = []
        for part_type, axis_num in itertools.product(_PartEnum, _AxisEnum):
            if part in [part_type.name, part_type.value, 'all'] and axis in [
                axis_num.name,
                axis_num.value,
                'all',
            ]:
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
        scale_factor = self._shaft_and_tip_geometry_source._actor_scale
        tip_radius = self._shaft_and_tip_geometry_source.tip_radius * np.array(scale_factor)
        offset_array = np.diag(tip_radius)
        radial_offset1 = np.roll(offset_array, shift=1, axis=1)
        radial_offset2 = np.roll(offset_array, shift=-1, axis=1)

        position_vectors += radial_offset1 + radial_offset2
        return position_vectors

    def _update_label_positions(self):
        labels = self._label_actors
        position_vectors = self._get_offset_label_position_vectors(self.label_position)
        for label, position in zip(labels, position_vectors, strict=True):
            label.relative_position = position


def _validate_label_sequence(labels: Sequence[str], n_labels: int | Sequence[int], name: str):
    _validation.check_instance(labels, (list, tuple), name=name)
    _validation.check_iterable_items(labels, str, name=name)
    _validation.check_length(labels, exact_length=n_labels, name=name)
    return labels


class AxesAssemblySymmetric(AxesAssembly):
    """Symmetric assembly of arrow-style axes parts.

    This class is similar to :class:`~pyvista.AxesAssembly` but the axes are
    symmetric.

    The axes may be used as a widget or added to a scene.

    Parameters
    ----------
    shaft_type : str | DataSet, default: 'cylinder'
        Shaft type for all axes. Can be any of the following:

        - ``'cylinder'``
        - ``'sphere'``
        - ``'hemisphere'``
        - ``'cone'``
        - ``'pyramid'``
        - ``'cube'``
        - ``'octahedron'``

        Alternatively, any arbitrary 3-dimensional :class:`pyvista.DataSet` may be
        specified. In this case, the dataset must be oriented such that it "points" in
        the positive z direction.

        .. versionadded:: 0.47

    shaft_radius : float | VectorLike[float], default: 0.025
        Radius of the axes shafts.

        .. versionadded:: 0.47

    shaft_length : float | VectorLike[float], default: 0.8
        Length of the shaft for each axis.

        .. versionadded:: 0.47

    tip_type : str | DataSet, default: 'cone'
        Tip type for all axes. Can be any of the following:

        - ``'cylinder'``
        - ``'sphere'``
        - ``'hemisphere'``
        - ``'cone'``
        - ``'pyramid'``
        - ``'cube'``
        - ``'octahedron'``

        Alternatively, any arbitrary 3-dimensional :class:`pyvista.DataSet` may be
        specified. In this case, the dataset must be oriented such that it "points" in
        the positive z direction.

        .. versionadded:: 0.47

    tip_radius : float | VectorLike[float], default: 0.1
        Radius of the axes tips.

        .. versionadded:: 0.47

    tip_length : float | VectorLike[float], default: 0.2
        Length of the tip for each axis.

        .. versionadded:: 0.47

    symmetric_bounds : bool, default: False
        Make the bounds of the axes symmetric. This option is similar to
        :attr:`symmetric`, except only the bounds are made to be symmetric,
        not the actual geometry. Has no effect if :attr:`symmetric` is ``True``.

        .. versionadded:: 0.47

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
        rotations are defined by the :attr:`~pyvista.Prop3D.orientation`.

    scale : VectorLike[float], default: (1.0, 1.0, 1.0)
        Scaling factor applied to the axes.

    user_matrix : MatrixLike[float], optional
        A 4x4 transformation matrix applied to the axes. Defaults to the identity matrix.
        The user matrix is the last transformation applied to the actor.

    name : str, optional
        The name of this assembly used when tracking on a plotter.

        .. versionadded:: 0.45

    **kwargs
        Keyword arguments passed to :class:`pyvista.AxesGeometrySource`.

    See Also
    --------
    AxesAssembly

    :ref:`axes_objects_example`
        Example showing different axes objects.

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
    ...     x_label=('X', ''), y_label=('Y', ''), z_label=('Z', '')
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
        shaft_type: AxesGeometrySource.GeometryTypes | DataSet = 'cylinder',
        shaft_radius: float | VectorLike[float] = 0.025,
        shaft_length: float | VectorLike[float] = 0.8,
        tip_type: AxesGeometrySource.GeometryTypes | DataSet = 'cone',
        tip_radius: float | VectorLike[float] = 0.1,
        tip_length: float | VectorLike[float] = 0.2,
        symmetric_bounds: bool = False,
        scale_mode: str = 'uniform',
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
        name: str | None = None,
    ):
        self._scale_mode = scale_mode
        # Init shaft and tip actors
        source = AxesGeometrySource(
            shaft_type=shaft_type,
            shaft_radius=shaft_radius,
            shaft_length=shaft_length,
            tip_type=tip_type,
            tip_radius=tip_radius,
            tip_length=tip_length,
            symmetric_bounds=symmetric_bounds,
            symmetric=True,
        )
        self._init_actors_from_source(source)
        # Init label actors
        self._label_actors = (Label(), Label(), Label())
        self._label_actors_symmetric = (Label(), Label(), Label())

        _XYZAssembly.__init__(
            self,
            xyz_actors=tuple(zip(self._shaft_actors, self._tip_actors, strict=True)),  # type: ignore[arg-type]
            xyz_label_actors=tuple(
                zip(self._label_actors, self._label_actors_symmetric, strict=True)
            ),  # type: ignore[arg-type]
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
            name=name,
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
        self,
        labels: list[str] | tuple[str, str, str] | tuple[str, str, str, str, str, str],
    ):
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
    def x_label(self, label: str | list[str] | tuple[str, str]):
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
    def y_label(self, label: str | list[str] | tuple[str, str]):
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
    def z_label(self, label: str | list[str] | tuple[str, str]):
        self._set_axis_label(_AxisEnum.z, label)

    def _update_label_positions(self):
        # Update plus labels using parent method
        AxesAssembly._update_label_positions(self)

        # Update minus labels
        label_position = self.label_position
        label_position_minus = (
            -label_position[0],
            -label_position[1],
            -label_position[2],
        )
        labels_minus = self._label_actors_symmetric
        vector_position_minus = self._get_offset_label_position_vectors(label_position_minus)
        for label, position in zip(labels_minus, vector_position_minus, strict=True):
            label.relative_position = position


class PlanesAssembly(_XYZAssembly):
    """Assembly of orthogonal planes.

    Assembly of three orthogonal plane meshes with labels.

    The labels can be 2D or 3D, and will follow the camera such that they have the
    correct orientation and remain parallel to the edges of the planes.

    The positioning of the labels may be customized using the :attr:`label_edge`,
    :attr:`label_position`, and :attr:`label_offset` attributes.

    .. warning::

        The :attr:`camera` must be set before rendering the assembly. Otherwise,
        attempting to render it will cause python to crash.

    .. versionadded:: 0.45

    Parameters
    ----------
    x_label : str, default: 'YZ'
        Text label for the yz-plane. Alternatively, set the label with :attr:`labels`.

    y_label : str, default: 'ZX'
        Text label for the zx-plane. Alternatively, set the label with :attr:`labels`.

    z_label : str, default: 'XY'
        Text label for the xy-plane. Alternatively, set the label with :attr:`labels`.

    labels : Sequence[str], optional,
        Text labels for the planes. This is an alternative parameter to using
        :attr:`x_label`, :attr:`y_label`, and :attr:`z_label` separately.

    label_color : ColorLike, default: 'black'
        Color of the text labels.

    show_labels : bool, default: True
        Show or hide the text labels.

    label_position : float | VectorLike[float], default: 0.5
        Normalized relative position of the text labels along each plane's respective
        :attr:`label_edge`. The positions are normalized to have a range of
        ``[-1.0, 1.0]`` such that ``0.0`` is at the center of the edge and ``-1.0`` and
        ``1.0`` are at the corners.

        .. note::

            The label text is centered horizontally at the specified positions.

    label_edge : str, default: 'right'
        Edge on which to position each plane's label. Can be ``'top'``, ``'bottom'``,
        ``'right'``, or ``'left'``. Use a single value to set the edge for all labels
        or set each edge independently.

    label_offset : float | VectorLike[float], optional
        Vertical offset of the text labels. The offset is proportional to
        the :attr:`~pyvista.Prop3D.length` of the assembly. Positive values move the labels away
        from the center; negative values move them towards it.

    label_size : int, default: 50
        Size of the text labels. If :attr:`label_mode` is ``'2D'``, this is the
        font size. If :attr:`label_mode` is ``'3D'``, the labels are scaled
        proportional to the :attr:`~pyvista.Prop3D.length` of the assembly.

    label_mode : '2D' | '3D', default: '2D'
        Mode to use for text labels. In 2D mode, the label actors are always visible
        and have a constant size regardless of window size. In 3D mode, the label actors
        may be occluded by other geometry and will scale with changes to the window
        size. The two modes also have minor differences in appearance and behavior in
        terms of how they follow the camera.

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
        The rotations are defined by the :attr:`~pyvista.Prop3D.orientation`.

    scale : VectorLike[float], default: (1.0, 1.0, 1.0)
        Scaling factor applied to the assembly.

    user_matrix : MatrixLike[float], optional
        A 4x4 transformation matrix applied to the assembly. Defaults to the identity
        matrix. The user matrix is the last transformation applied to the actor.

    name : str, optional
        The name of this assembly used when tracking on a plotter.

        .. versionadded:: 0.45

    **kwargs
        Keyword arguments passed to :class:`pyvista.OrthogonalPlanesSource`.

    Examples
    --------
    Fit planes to a model of a human.

    >>> import numpy as np
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> human = examples.download_human()
    >>> labels = 'Sagittal', 'Coronal', 'Transverse'
    >>> planes = pv.PlanesAssembly(
    ...     bounds=human.bounds, labels=labels, label_size=25
    ... )

    Plot the planes and the model.

    >>> pl = pv.Plotter()
    >>> human_actor = pl.add_mesh(human, scalars='Color', rgb=True)
    >>> _ = pl.add_actor(planes)
    >>> planes.camera = pl.camera
    >>> pl.show()

    Apply a transformation to the planes and the model.

    >>> transform = np.array(
    ...     [
    ...         [0.70645893, 0.69636424, 0.12646197, 1.0],
    ...         [-0.62246712, 0.69636424, -0.35722756, 2.0],
    ...         [-0.33682409, 0.17364818, 0.92541658, 3.0],
    ...         [0.0, 0.0, 0.0, 1.0],
    ...     ]
    ... )
    >>> planes.user_matrix = transform
    >>> human_actor.user_matrix = transform

    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(human_actor)
    >>> _ = pl.add_actor(planes)
    >>> planes.camera = pl.camera
    >>> pl.show()

    Create a new assembly and customize the colors and opacity.

    >>> color = pv.global_theme.color
    >>> planes = pv.PlanesAssembly(
    ...     bounds=human.bounds,
    ...     x_color=color,
    ...     y_color=color,
    ...     z_color=color,
    ...     opacity=1.0,
    ... )

    Since the planes are opaque, the 3D labels may be occluded. Use 2D labels instead
    so the labels are always visible.

    >>> planes.label_mode = '2D'

    Offset the labels to position them inside the bounds of the planes.

    >>> planes.label_offset = -0.05

    Move the labels for the two larger planes closer to the corners.

    >>> planes.label_position = (0.8, 0.8, 0.5)

    Visualize the result.

    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(human, scalars='Color', rgb=True)
    >>> _ = pl.add_actor(planes)
    >>> planes.camera = pl.camera
    >>> pl.show()

    """

    DEFAULT_LABELS = _XYZTuple('YZ', 'ZX', 'XY')

    def __init__(
        self,
        *,
        x_label: str | None = None,
        y_label: str | None = None,
        z_label: str | None = None,
        labels: Sequence[str] | None = None,
        label_color: ColorLike = 'black',
        show_labels: bool = True,
        label_position: float | VectorLike[float] = 0.5,
        label_edge: Literal['top', 'bottom', 'right', 'left'] | Sequence[str] = 'right',
        label_offset: float = 0.05,
        label_size: int = 50,
        label_mode: Literal['2D', '3D'] = '3D',
        x_color: ColorLike | None = None,
        y_color: ColorLike | None = None,
        z_color: ColorLike | None = None,
        opacity: float | VectorLike[float] = 0.3,
        position: VectorLike[float] = (0.0, 0.0, 0.0),
        orientation: VectorLike[float] = (0.0, 0.0, 0.0),
        origin: VectorLike[float] = (0.0, 0.0, 0.0),
        scale: float | VectorLike[float] = (1.0, 1.0, 1.0),
        user_matrix: MatrixLike[float] | None = None,
        name: str | None = None,
        **kwargs: Unpack[_OrthogonalPlanesKwargs],
    ):
        self._camera = None

        # Init plane actors
        self._plane_actors = (Actor(), Actor(), Actor())
        # Init planes from source
        self._geometry_source = OrthogonalPlanesSource(**kwargs)
        self._planes = self._geometry_source.output
        self._plane_sources = self._geometry_source.sources

        for actor, dataset in zip(self._plane_actors, self.planes, strict=True):
            actor.mapper = pv.DataSetMapper(dataset=dataset)

        # Init label actors
        self._axis_actors = (_AxisActor(), _AxisActor(), _AxisActor())

        # Tempt init values for call to super class, will validate inputs later
        self._label_offset = 0.05
        self._label_edge = ('right', 'right', 'right')
        self._label_position = 0.5, 0.5, 0.5

        _XYZAssembly.__init__(
            self,
            xyz_actors=self._plane_actors,
            xyz_label_actors=self._axis_actors,
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
            name=name,
        )

        self.opacity = opacity  # type: ignore[assignment]
        self.label_mode = label_mode
        self.label_offset = label_offset
        self.label_edge = label_edge

        # Set default properties
        for actor in self._plane_actors:
            prop = actor.prop
            prop.show_edges = True
            prop.line_width = 3

        # Set default text properties
        # TODO: implement set_text_prop() and use that instead
        for label in self._label_actor_iterator:
            prop = label.prop
            prop.justification_vertical = 'center'
            prop.justification_horizontal = 'center'

    def __repr__(self):
        """Representation of the planes assembly."""
        mat_info = 'Identity' if np.array_equal(self.user_matrix, np.eye(4)) else 'Set'
        bnds = self.bounds

        attr = [
            f'{type(self).__name__} ({hex(id(self))})',
            f'  Resolution:                 {self._geometry_source.resolution}',
            f'  Normal sign:                {self._geometry_source.normal_sign}',
            f"  X label:                    '{self.x_label}'",
            f"  Y label:                    '{self.y_label}'",
            f"  Z label:                    '{self.z_label}'",
            f'  Label color:                {self.label_color}',
            f'  Show labels:                {self.show_labels}',
            f'  Label position:             {self.label_position}',
            f'  Label edge:                 {self.label_edge}',
            f'  Label offset:               {self.label_offset}',
            f"  Label mode:                 '{self.label_mode}'",
            f'  X Color:                    {self.x_color}',
            f'  Y Color:                    {self.y_color}',
            f'  Z Color:                    {self.z_color}',
            f'  Position:                   {self.position}',
            f'  Orientation:                {self.orientation}',
            f'  Origin:                     {self.origin}',
            f'  Scale:                      {self.scale}',
            f'  User matrix:                {mat_info}',
            f'  X Bounds                    {bnds.x_min:.3E}, {bnds.x_max:.3E}',
            f'  Y Bounds                    {bnds.y_min:.3E}, {bnds.y_max:.3E}',
            f'  Z Bounds                    {bnds.z_min:.3E}, {bnds.z_max:.3E}',
        ]
        return '\n'.join(attr)

    @property
    def labels(self) -> tuple[str, str, str]:  # numpydoc ignore=RT01
        """Return or set the labels for the planes.

        This property may be used as an alternative to using :attr:`x_label`,
        :attr:`y_label`, and :attr:`z_label` separately.

        Examples
        --------
        >>> import pyvista as pv
        >>> planes = pv.PlanesAssembly()
        >>> planes.labels = ['Sagittal', 'Coronal', 'Transverse']
        >>> planes.labels
        ('Sagittal', 'Coronal', 'Transverse')

        """
        return self.x_label, self.y_label, self.z_label

    @labels.setter
    def labels(self, labels: list[str] | tuple[str, str, str]):
        labels = _validate_label_sequence(labels, n_labels=3, name='labels')
        self.x_label = labels[0]
        self.y_label = labels[1]
        self.z_label = labels[2]

    @property
    def x_label(self) -> str:  # numpydoc ignore=RT01
        """Text label for the yz-plane.

        Examples
        --------
        >>> import pyvista as pv
        >>> planes = pv.PlanesAssembly()
        >>> planes.x_label = 'This plane'
        >>> planes.x_label
        'This plane'

        """
        return self._axis_actors[0].GetTitle()

    @x_label.setter
    def x_label(self, label: str):
        self._axis_actors[0].SetTitle(label)
        self.planes.set_block_name(0, label)

    @property
    def y_label(self) -> str:  # numpydoc ignore=RT01
        """Text label for the zx-plane.

        Examples
        --------
        >>> import pyvista as pv
        >>> planes = pv.PlanesAssembly()
        >>> planes.y_label = 'This plane'
        >>> planes.y_label
        'This plane'

        """
        return self._axis_actors[1].GetTitle()

    @y_label.setter
    def y_label(self, label: str):
        self._axis_actors[1].SetTitle(label)
        self.planes.set_block_name(1, label)

    @property
    def z_label(self) -> str:  # numpydoc ignore=RT01
        """Text label for the xy-plane.

        Examples
        --------
        >>> import pyvista as pv
        >>> planes = pv.PlanesAssembly()
        >>> planes.z_label = 'This plane'
        >>> planes.z_label
        'This plane'

        """
        return self._axis_actors[2].GetTitle()

    @z_label.setter
    def z_label(self, label: str):
        self._axis_actors[2].SetTitle(label)
        self.planes.set_block_name(2, label)

    @property
    def label_size(self) -> int:  # numpydoc ignore=RT01
        """Size of the text labels.

        Must be a positive integer.
        """
        return self._label_size

    @label_size.setter
    def label_size(self, size: int):
        valid_size = _validation.validate_number(
            size,
            must_be_in_range=[0, np.inf],
            must_be_integer=True,
            dtype_out=int,
            name='label size',
        )
        self._label_size = valid_size
        # 2D labels use font size (int) but 3D labels use a scaling factor (float)
        # For 3D labels, we re-scale the text proportional to the planes assembly
        # Values on the order of 0.01-0.05 seem to work best. Use a normalization
        # factor so that input values are on the order of 10-50 and roughly match 2D sizes
        NORM_FACTOR = 1000
        scale_3d = self.planes.length * float(valid_size) / NORM_FACTOR

        # In VTK 9.6+, the 3D label size depends on the 2D label size, so in the 3D case
        # we need to reset the 2D font size to match the VTK default value of 12
        font_size_2d = (
            valid_size if hasattr(self, 'label_mode') and self.label_mode == '2D' else 12
        )
        for axis in self._axis_actors:
            axis.GetTitleActor().SetScale(scale_3d)  # 3D labels
            axis.GetTitleTextProperty().SetFontSize(font_size_2d)  # 2D labels

    @property
    def label_position(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Normalized relative position of the text labels along the edge of each plane.

        Labels are positioned relative to each plane's respective :attr:`label_edge`.
        The positions are normalized to have a range of ``[-1.0, 1.0]`` such that ``0.0``
        is at the center of the edge and ``-1.0`` and ``1.0`` are at the corners.

        .. note::

            The label text is centered horizontally at the specified positions.

        Examples
        --------
        Position the labels at the center (along the edges) and plot the assembly.

        >>> import pyvista as pv
        >>> planes = pv.PlanesAssembly(label_position=0)
        >>> planes.label_position
        (0.0, 0.0, 0.0)

        >>> pl = pv.Plotter()
        >>> _ = pl.add_actor(planes)
        >>> planes.camera = pl.camera
        >>> pl.show()

        Position the labels at the corners.

        >>> planes.label_position = 1.0
        >>> pl = pv.Plotter()
        >>> _ = pl.add_actor(planes)
        >>> planes.camera = pl.camera
        >>> pl.show()

        Vary the position of the labels independently for each plane. The values may be
        negative and/or exceed a value of ``1.0``.

        >>> planes.label_position = (-1.3, -1.0, -0.5)
        >>> pl = pv.Plotter()
        >>> _ = pl.add_actor(planes)
        >>> planes.camera = pl.camera
        >>> pl.show()

        """
        return self._label_position

    @label_position.setter
    def label_position(self, position: int | VectorLike[int]):
        self._label_position = _validation.validate_array3(
            position,
            broadcast=True,
            name='Label position',
            dtype_out=float,
            to_tuple=True,
        )
        self._update_label_positions()

    @property
    def label_edge(self) -> tuple[str, str, str]:  # numpydoc ignore=RT01
        """Edge on which to position each plane's label.

        Edge can be ``'top'``,``'bottom'``,``'right'``, or ``'left'``, and can be
        set independently for each plane or to the same edge for all planes.

        The edge is relative to each plane's local ``i`` and ``j`` coordinates.

        Examples
        --------
        Position the labels at the top edge and plot.

        >>> import pyvista as pv
        >>> planes = pv.PlanesAssembly(label_edge='top')
        >>> planes.label_edge
        ('top', 'top', 'top')

        >>> pl = pv.Plotter()
        >>> _ = pl.add_actor(planes)
        >>> planes.camera = pl.camera
        >>> pl.show()

        Position the labels at the bottom.

        >>> planes.label_edge = 'bottom'
        >>> pl = pv.Plotter()
        >>> _ = pl.add_actor(planes)
        >>> planes.camera = pl.camera
        >>> pl.show()

        Vary the edge of the labels independently for each plane.

        >>> planes.label_edge = ('top', 'right', 'left')
        >>> pl = pv.Plotter()
        >>> _ = pl.add_actor(planes)
        >>> planes.camera = pl.camera
        >>> pl.show()

        """
        return self._label_edge

    @label_edge.setter
    def label_edge(self, edge: Literal['top', 'bottom', 'right', 'left'] | Sequence[str]):
        valid_edge = (
            [edge] * 3
            if isinstance(edge, str)
            else _validate_label_sequence(edge, n_labels=3, name='label edge')
        )
        for edge_ in valid_edge:
            _validation.check_contains(
                ['top', 'bottom', 'right', 'left'],
                must_contain=edge_,
                name='label_edge',
            )
        self._label_edge = tuple(valid_edge)
        self._update_label_positions()

    @property
    def label_offset(self) -> float:  # numpydoc ignore=RT01
        """Vertical offset of the text labels.

        The offset is proportional to the :attr:`~pyvista.Prop3D.length` of the assembly. Positive
        values move the labels away from the center; negative values move them
        towards it.
        """
        return self._label_offset

    @label_offset.setter
    def label_offset(self, offset: float):
        self._label_offset = _validation.validate_number(offset, dtype_out=float)
        self._update_label_positions()

    @property
    def label_mode(self) -> Literal['2D', '3D']:  # numpydoc ignore=RT01
        """Mode to use for text labels.

        Mode must be either ``'2D'`` or ``'3D'``. In 2D mode, the label actors are
        always visible and have a constant size regardless of window size. In 3D mode,
        the label actors may be occluded by other geometry and will scale with changes
        to the window size. The two modes also have minor differences in appearance as
        well as behavior in terms of how they follow the camera.
        """
        return self._label_mode

    @label_mode.setter
    def label_mode(self, mode: Literal['2D', '3D']):
        _validation.check_contains(['2D', '3D'], must_contain=mode, name='label_mode')
        self._label_mode = mode
        use_2D = mode == '2D'
        for axis in self._axis_actors:
            axis.SetUse2DMode(use_2D)

        # The 3D label size depends on the 2D label size so we need to reset this property
        self.label_size = self.label_size

    @property
    def x_color(self) -> Color:  # numpydoc ignore=RT01
        """Color of the yz-plane."""
        return self._plane_actors[0].prop.color

    @x_color.setter
    def x_color(self, color: ColorLike):
        self._plane_actors[0].prop.color = color

    @property
    def y_color(self) -> Color:  # numpydoc ignore=RT01
        """Color of the zx-plane."""
        return self._plane_actors[1].prop.color

    @y_color.setter
    def y_color(self, color: ColorLike):
        self._plane_actors[1].prop.color = color

    @property
    def z_color(self) -> Color:  # numpydoc ignore=RT01
        """Color of the xy-plane."""
        return self._plane_actors[2].prop.color

    @z_color.setter
    def z_color(self, color: ColorLike):
        self._plane_actors[2].prop.color = color

    @property
    def opacity(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Opacity of the planes."""
        return self._opacity

    @opacity.setter
    def opacity(self, opacity: float):
        valid_opacity = _validation.validate_array3(
            opacity, broadcast=True, dtype_out=float, to_tuple=True
        )
        self._opacity = valid_opacity
        for actor, opacity_ in zip(self._plane_actors, valid_opacity, strict=True):
            actor.prop.opacity = opacity_

    @property
    def camera(self):  # numpydoc ignore=RT01
        """Camera to use for displaying the labels."""
        return self._camera

    @camera.setter
    def camera(self, camera):
        self._camera = camera
        for axis in self._axis_actors:
            axis.SetCamera(camera)

    @property
    def planes(self):
        """Get the orthogonal plane datasets of the assembly.

        The planes are :class:`pyvista.PolyData` meshes stored as a
        :class:`pyvista.MultiBlock`. The names of the blocks match the names of the
        assembly's :attr:`labels`.

        The planes are initially generated with :class:`pyvista.OrthogonalPlanesSource`.

        Returns
        -------
        pyvista.MultiBlock
            Composite mesh with three planes.

        """
        return self._planes

    def _update_label_positions(self):
        axis_actors = self._axis_actors
        plane_sources = self._plane_sources
        transformation_matrix = self._transformation_matrix

        def transform_point(point):
            return (transformation_matrix @ (*point, 1))[:3]

        def set_axis_location(plane_id, edge: str, position: float):
            this_plane_source = plane_sources[plane_id]
            this_axis_actor = axis_actors[plane_id]

            # Get vectors which define the plane
            origin, point1, point2 = (
                np.array(this_plane_source.GetOrigin()),
                np.array(this_plane_source.GetPoint1()),
                np.array(this_plane_source.GetPoint2()),
            )

            vector1 = point1 - origin
            vector2 = point2 - origin

            # Define corners
            corner_bottom_left = origin
            corner_bottom_right = origin + vector1
            corner_top_left = origin + vector2
            corner_top_right = corner_bottom_right + vector2

            # Define axis points in counter-clockwise order
            if edge == 'top':
                axis_point1, axis_point2 = corner_top_right, corner_top_left
            elif edge == 'left':
                axis_point1, axis_point2 = corner_top_left, corner_bottom_left
            elif edge == 'bottom':
                axis_point1, axis_point2 = corner_bottom_left, corner_bottom_right
            else:  # 'right'
                axis_point1, axis_point2 = corner_bottom_right, corner_top_right

            # Move axis to position along the edge
            axis_vector = axis_point1 - axis_point2
            # Define position relative to center of edge
            position_vector = np.abs(axis_vector) * position * 0.5
            axis_point1 += position_vector
            axis_point2 += position_vector

            # Add offset
            axis_dir = axis_vector / np.linalg.norm(axis_vector)
            offset_dir = np.cross(axis_dir, this_plane_source.GetNormal())
            offset_dir = -1 * offset_dir / np.linalg.norm(offset_dir)
            offset_mag = self.planes.length * self.label_offset
            offset = offset_mag * offset_dir
            axis_point1 += offset
            axis_point2 += offset

            # Set axis points
            this_axis_actor.SetPoint1(transform_point(axis_point1))
            this_axis_actor.SetPoint2(transform_point(axis_point2))

        edge = self.label_edge
        position = self.label_position
        set_axis_location(0, edge[0], position[0])
        set_axis_location(1, edge[1], position[1])
        set_axis_location(2, edge[2], position[2])

    def _post_set_update(self):
        _XYZAssembly._post_set_update(self)
        # Need to manually update axis actors
        self._update_label_positions()


class _AxisActor(_vtk.DisableVtkSnakeCase, _vtk.vtkAxisActor):
    def __init__(self):
        super().__init__()
        # Only show the title
        self.TitleVisibilityOn()
        self.MinorTicksVisibleOff()
        self.TickVisibilityOff()
        self.DrawGridlinesOff()
        self.AxisVisibilityOff()  # Turn this on for debugging

        # Set empty tick labels
        labels = _vtk.vtkStringArray()
        labels.SetNumberOfTuples(0)
        # labels.SetValue(0, "")
        self.SetLabels(labels)

        # Ignore the axis bounds when rendering. Otherwise, the bounds must be
        # set with SetBounds() every time the axis is updated
        self.SetUseBounds(False)

        # Format title positioning
        offset = (0,) if pv.vtk_version_info < (9, 3) else (0, 0)
        self.SetTitleOffset(*offset)
        self.SetLabelOffset(0)

        # For 2D mode only
        self.SetVerticalOffsetXTitle2D(0)
        self.SetHorizontalOffsetYTitle2D(0)
        text_prop = TextProperty()
        text_prop.justification_vertical = 'center'
        self.SetTitleTextProperty(text_prop)
        self.GetTitleActor()

        # For 3D mode only
        self.GetProperty().SetLighting(False)

    @property
    def prop(self) -> TextProperty:
        return self.GetTitleTextProperty()
