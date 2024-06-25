"""Wrap vtkActor module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

import numpy as np

import pyvista
from pyvista.core.utilities.misc import no_new_attr

from . import _vtk
from ._property import Property
from .prop3d import Prop3D

if TYPE_CHECKING:  # pragma: no cover
    from .mapper import _BaseMapper


@no_new_attr
class Actor(Prop3D, _vtk.vtkActor):
    """Wrap vtkActor.

    This class represents the geometry & properties in a rendered
    scene. Normally, a :class:`pyvista.Actor` is constructed from
    :func:`pyvista.Plotter.add_mesh`, but there may be times when it is more
    convenient to construct an actor directly from a
    :class:`pyvista.DataSetMapper`.

    Parameters
    ----------
    mapper : pyvista.DataSetMapper, optional
        DataSetMapper.

    prop : pyvista.Property, optional
        Property of the actor.

    name : str, optional
        The name of this actor used when tracking on a plotter.

    Examples
    --------
    Create an actor without using :class:`pyvista.Plotter`.

    >>> import pyvista as pv
    >>> mesh = pv.Sphere()
    >>> mapper = pv.DataSetMapper(mesh)
    >>> actor = pv.Actor(mapper=mapper)
    >>> actor
    Actor (...)
      Center:                     (0.0, 0.0, 0.0)
      Pickable:                   True
      Position:                   (0.0, 0.0, 0.0)
      Scale:                      (1.0, 1.0, 1.0)
      Visible:                    True
      X Bounds                    -4.993E-01, 4.993E-01
      Y Bounds                    -4.965E-01, 4.965E-01
      Z Bounds                    -5.000E-01, 5.000E-01
      User matrix:                Identity
      Has mapper:                 True
    ...

    Change the actor properties and plot the actor.

    >>> import pyvista as pv
    >>> mesh = pv.Sphere()
    >>> mapper = pv.DataSetMapper(mesh)
    >>> actor = pv.Actor(mapper=mapper)
    >>> actor.prop.color = 'blue'
    >>> actor.plot()

    Create an actor using the :class:`pyvista.Plotter` and then change the
    visibility of the actor.

    >>> import pyvista as pv
    >>> pl = pv.Plotter()
    >>> mesh = pv.Sphere()
    >>> actor = pl.add_mesh(mesh)
    >>> actor.visibility = False
    >>> actor.visibility
    False

    """

    _new_attr_exceptions: ClassVar[list[str]] = ['_name']

    def __init__(self, mapper=None, prop=None, name=None):
        """Initialize actor."""
        super().__init__()
        if mapper is not None:
            self.mapper = mapper
        if prop is None:
            self.prop = Property()
        else:
            self.prop = prop
        self._name = name

    @property
    def name(self) -> str:  # numpydoc ignore=RT01
        """Get or set the unique name identifier used by PyVista."""
        if self._name is None:
            self._name = f'{type(self).__name__}({self.memory_address})'
        return self._name

    @name.setter
    def name(self, value: str):  # numpydoc ignore=GL08
        if not value:
            raise ValueError('Name must be truthy.')
        self._name = value

    @property
    def mapper(self) -> _BaseMapper:  # numpydoc ignore=RT01
        """Return or set the mapper of the actor.

        Examples
        --------
        Create an actor and assign a mapper to it.

        >>> import pyvista as pv
        >>> dataset = pv.Sphere()
        >>> actor = pv.Actor()
        >>> actor.mapper = pv.DataSetMapper(dataset)
        >>> actor.mapper
        DataSetMapper (...)
          Scalar visibility:           True
          Scalar range:                (0.0, 1.0)
          Interpolate before mapping:  True
          Scalar map mode:             default
          Color mode:                  direct
        <BLANKLINE>
        Attached dataset:
        PolyData (...)
          N Cells:    1680
          N Points:   842
          N Strips:   0
          X Bounds:   -4.993e-01, 4.993e-01
          Y Bounds:   -4.965e-01, 4.965e-01
          Z Bounds:   -5.000e-01, 5.000e-01
          N Arrays:   1

        """
        return self.GetMapper()

    @mapper.setter
    def mapper(self, obj):  # numpydoc ignore=GL08
        self.SetMapper(obj)

    @property
    def prop(self):  # numpydoc ignore=RT01
        """Return or set the property of this actor.

        Examples
        --------
        Modify the properties of an actor after adding a dataset to the plotter.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Sphere())
        >>> prop = actor.prop
        >>> prop.diffuse = 0.6
        >>> pl.show()

        """
        return self.GetProperty()

    @prop.setter
    def prop(self, obj: Property):  # numpydoc ignore=GL08
        self.SetProperty(obj)

    @property
    def texture(self):  # numpydoc ignore=RT01
        """Return or set the actor texture.

        Notes
        -----
        The mapper dataset must have texture coordinates for the texture to be
        used.

        Examples
        --------
        Create an actor and add a texture to it. Note how the
        :class:`pyvista.PolyData` has texture coordinates by default.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> plane = pv.Plane()
        >>> plane.active_texture_coordinates is not None
        True
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(plane)
        >>> actor.texture = examples.download_masonry_texture()
        >>> actor.texture
        Texture (...)
          Components:   3
          Cube Map:     False
          Dimensions:   256, 256

        """
        return self.GetTexture()

    @texture.setter
    def texture(self, obj):  # numpydoc ignore=GL08
        self.SetTexture(obj)

    @property
    def memory_address(self):  # numpydoc ignore=RT01
        """Return the memory address of this actor."""
        return self.GetAddressAsString("")

    @property
    def pickable(self) -> bool:  # numpydoc ignore=RT01
        """Return or set actor pickability.

        Examples
        --------
        Create an actor using the :class:`pyvista.Plotter` and then make the
        actor unpickable.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Sphere())
        >>> actor.pickable = False
        >>> actor.pickable
        False

        """
        return bool(self.GetPickable())

    @pickable.setter
    def pickable(self, value):  # numpydoc ignore=GL08
        self.SetPickable(value)

    @property
    def visibility(self) -> bool:  # numpydoc ignore=RT01
        """Return or set actor visibility.

        Examples
        --------
        Create an actor using the :class:`pyvista.Plotter` and then change the
        visibility of the actor.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Sphere())
        >>> actor.visibility = False
        >>> actor.visibility
        False

        """
        return bool(self.GetVisibility())

    @visibility.setter
    def visibility(self, value: bool):  # numpydoc ignore=GL08
        self.SetVisibility(value)

    def plot(self, **kwargs):
        """Plot just the actor.

        This may be useful when interrogating or debugging individual actors.

        Parameters
        ----------
        **kwargs : dict, optional
            Optional keyword arguments passed to :func:`pyvista.Plotter.show`.

        Examples
        --------
        Create an actor without the :class:`pyvista.Plotter`, change its
        properties, and plot it.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mapper = pv.DataSetMapper(mesh)
        >>> actor = pv.Actor(mapper=mapper)
        >>> actor.prop.color = 'red'
        >>> actor.prop.show_edges = True
        >>> actor.plot()

        """
        pl = pyvista.Plotter()
        pl.add_actor(self)
        pl.show(**kwargs)

    def copy(self, deep=True) -> Actor:
        """Create a copy of this actor.

        Parameters
        ----------
        deep : bool, default: True
            Create a shallow or deep copy of the actor. A deep copy will have a
            new property and mapper, while a shallow copy will use the mapper
            and property of this actor.

        Returns
        -------
        pyvista.Actor
            Deep or shallow copy of this actor.

        Examples
        --------
        Create an actor of a cube by adding it to a :class:`pyvista.Plotter`
        and then copy the actor, change the properties, and add it back to the
        :class:`pyvista.Plotter`.

        >>> import pyvista as pv
        >>> mesh = pv.Cube()
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(mesh, color='b')
        >>> new_actor = actor.copy()
        >>> new_actor.prop.style = 'wireframe'
        >>> new_actor.prop.line_width = 5
        >>> new_actor.prop.color = 'r'
        >>> new_actor.prop.lighting = False
        >>> _ = pl.add_actor(new_actor)
        >>> pl.show()

        """
        new_actor = Actor()
        if deep:
            new_actor.mapper = self.mapper.copy()
            new_actor.prop = self.prop.copy()
        else:
            new_actor.ShallowCopy(self)
        return new_actor

    def __repr__(self):
        """Representation of the actor."""
        if self.user_matrix is None or np.array_equal(self.user_matrix, np.eye(4)):
            mat_info = 'Identity'
        else:
            mat_info = 'Set'
        bnd = self.bounds
        attr = [
            f'{type(self).__name__} ({hex(id(self))})',
            f'  Center:                     {self.center}',
            f'  Pickable:                   {self.pickable}',
            f'  Position:                   {self.position}',
            f'  Scale:                      {self.scale}',
            f'  Visible:                    {self.visibility}',
            f'  X Bounds                    {bnd[0]:.3E}, {bnd[1]:.3E}',
            f'  Y Bounds                    {bnd[2]:.3E}, {bnd[3]:.3E}',
            f'  Z Bounds                    {bnd[4]:.3E}, {bnd[5]:.3E}',
            f'  User matrix:                {mat_info}',
            f'  Has mapper:                 {self.mapper is not None}',
            '',
            repr(self.prop),
        ]
        if self.mapper is not None:
            attr.append('')
            attr.append(repr(self.mapper))
        return '\n'.join(attr)

    @property
    def backface_prop(self) -> pyvista.Property | None:  # numpydoc ignore=RT01
        """Return or set the backface property.

        By default this property matches the frontface property
        :attr:`Actor.prop`. Once accessed or modified, this backface
        property becomes independent of the frontface property. In
        order to restore the fallback to frontface property, assign
        ``None`` to the property.

        Returns
        -------
        pyvista.Property
            The object describing backfaces.

        Examples
        --------
        Clip a sphere by a plane and color the inside of the clipped sphere
        light blue using the ``backface_prop``.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> plane = pv.Plane(i_size=1.5, j_size=1.5)
        >>> mesh = pv.Sphere().clip_surface(plane, invert=False)
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(mesh, smooth_shading=True)
        >>> actor.backface_prop.color = 'lightblue'
        >>> _ = pl.add_mesh(
        ...     plane,
        ...     opacity=0.25,
        ...     show_edges=True,
        ...     color='grey',
        ...     lighting=False,
        ... )
        >>> pl.show()

        """
        if self.GetBackfaceProperty() is None:
            self.SetBackfaceProperty(self.prop.copy())
        return self.GetBackfaceProperty()

    @backface_prop.setter
    def backface_prop(self, value: pyvista.Property):  # numpydoc ignore=GL08
        self.SetBackfaceProperty(value)
