"""Wrap vtkActor."""

import weakref

import pyvista as pv

from ._property import Property


class Actor(pv._vtk.vtkActor):
    """Wrap vtkActor.

    Parameters
    ----------
    mapper : pyvista.DataSetMapper, optional
        DataSetMapper.

    prop : pyvista.Property, optional
        Property.

    Examples
    --------
    Create an actor without using :class:`pyvista.Plotter`.

    >>> import pyvista as pv
    >>> mesh = pv.Sphere()
    >>> mapper = pv.DataSetMapper(mesh)
    >>> actor = pv.Actor(mapper=mapper)
    >>> actor

    """

    _renderer = None

    def __init__(self, mapper=None, prop=None):
        """Initialize actor."""
        super().__init__()
        if mapper is not None:
            self.mapper = mapper
        if prop is None:
            self.prop = Property()

    @property
    def mapper(self):
        """Return or set the mapper of the actor."""
        return self.GetMapper()

    @mapper.setter
    def mapper(self, obj):
        return self.SetMapper(obj)

    @property
    def prop(self):
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
    def prop(self, obj: Property):
        self.SetProperty(obj)

    @property
    def texture(self):
        """Return or set the actor texture."""
        return self.GetTexture()

    @texture.setter
    def texture(self, obj):
        self.SetTexture(obj)

    def __setattr__(self, name, value):
        """Do not allow setting attributes."""
        if hasattr(self, name):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f'Attribute {name} does not exist and cannot be added to type '
                f'{self.__class__.__name__}'
            )

    @property
    def renderer(self):
        """Return the renderer associated with this actor."""
        return self._renderer

    @renderer.setter
    def renderer(self, obj):
        if not isinstance(obj, weakref.ProxyType):
            raise TypeError("Only a ProxyType can be assigned to `renderer`")
        self._renderer = obj

    @property
    def address(self):
        """Return the memory address of this actor."""
        return self.GetAddressAsString("")

    @property
    def pickable(self):
        """Return or set actor pickability."""
        return self.GetPickable()

    @pickable.setter
    def pickable(self, value):
        return self.SetPickable(value)

    @property
    def visibility(self) -> bool:
        """Return or set actor visibility."""
        return self.GetVisibility()

    @visibility.setter
    def visibility(self, value: bool):
        return self.SetVisibility(value)

    @property
    def scale(self) -> float:
        """Return or set actor scale."""
        return self.GetScale()

    @scale.setter
    def scale(self, value: float):
        return self.SetScale(value)

    def plot(self):
        """Plot just the actor.

        This may be useful for interrogating individual actors.

        Examples
        --------
        Create an actor, change its properties, and plot it.

        >>> import pyvista as pv
        >>> mesh = pv.Sphere()
        >>> mapper = pv.DataSetMapper(mesh)
        >>> actor = pv.Actor(mapper=mapper)
        >>> actor.prop.color = 'red'
        >>> actor.prop.show_edges = True
        >>> actor.plot()

        """
        pl = pv.Plotter()
        pl.add_actor(self)
        pl.show()

    def __str__(self):
        """Representation of the actor."""
        return '\n'.join(
            [
                f'Actor {self.address} Attributes',
                f'Visible:   {self.visibility}',
                f'Pickable:  {self.pickable}',
                f'Scale:     {self.scale}',
                '',
                str(self.prop),
            ]
        )
