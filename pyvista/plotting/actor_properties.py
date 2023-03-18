"""Module containing pyvista implementation of vtkProperty."""
import pyvista as pv
from pyvista.plotting.opts import InterpolationType, RepresentationType


class ActorProperties:
    """Properties wrapper for ``vtkProperty``.

    Contains the surface properties of the object.

    Parameters
    ----------
    properties : pv._vtk.vtkProperty
        VTK properties of the current object.

    Examples
    --------
    Access the properties of the z axis shaft.

    >>> import pyvista as pv

    >>> axes = pv.Axes()
    >>> z_axes_prop = axes.axes_actor.z_axis_shaft_properties
    >>> z_axes_prop.color = (1, 1, 0)
    >>> z_axes_prop.opacity = 0.5
    >>> axes.axes_actor.shaft_type = axes.axes_actor.ShaftType.CYLINDER

    >>> pl = pv.Plotter()
    >>> pl.add_actor(axes.axes_actor)  # doctest:+SKIP
    >>> pl.add_mesh(pv.Sphere())  # doctest:+SKIP
    >>> pl.show()  # doctest:+SKIP

    """

    def __init__(self, properties: pv._vtk.vtkProperty) -> None:
        super().__init__()
        self.properties = properties

    @property
    def color(self):
        """Return or set the color of the actor."""
        return self.properties.GetColor()

    @color.setter
    def color(self, color: tuple):
        self.properties.SetColor(color[0], color[1], color[2])

    @property
    def metallic(self):
        """Return or set the metallic coefficient of the surface."""
        return self.properties.GetMetallic()

    @metallic.setter
    def metallic(self, value: float):
        self.properties.SetMetallic(value)

    @property
    def roughness(self):
        """Return or set the roughness of the surface."""
        return self.properties.GetRoughness()

    @roughness.setter
    def roughness(self, value: float):
        self.properties.SetRoughness(value)

    @property
    def anisotropy(self):
        """Return or set the anisotropy coefficient."""
        return self.properties.GetAnisotropy()

    @anisotropy.setter
    def anisotropy(self, value: float):
        self.properties.SetAnisotropy(value)

    @property
    def anisotropy_rotation(self):
        """Return or set the anisotropy rotation coefficient."""
        return self.properties.GetAnisotropyRotation()

    @anisotropy_rotation.setter
    def anisotropy_rotation(self, value: float):
        self.properties.SetAnisotropyRotation(value)

    @property
    def lighting(self):
        """Return or set the lighting activation flag."""
        return self.properties.GetLighting()

    @lighting.setter
    def lighting(self, flag: bool):
        self.properties.SetLighting(flag)

    @property
    def interpolation_model(self):
        """Return or set the interpolation model.

        Can be any of the options in :class:`pyvista.plotting.opts.InterpolationType` enum.
        """
        return InterpolationType.from_any(self.properties.GetInterpolation())

    @interpolation_model.setter
    def interpolation_model(self, model: InterpolationType):
        self.properties.SetInterpolation(model.value)

    @property
    def index_of_refraction(self):
        """Return or set the Index Of Refraction of the base layer."""
        return self.properties.GetBaseIOR()

    @index_of_refraction.setter
    def index_of_refraction(self, value: float):
        self.properties.SetBaseIOR(value)

    @property
    def opacity(self):
        """Return or set the opacity of the actor."""
        return self.properties.GetOpacity()

    @opacity.setter
    def opacity(self, value: float):
        self.properties.SetOpacity(value)

    @property
    def shading(self):
        """Return or set the flag to activate the shading."""
        return self.properties.GetShading()

    @shading.setter
    def shading(self, is_active: bool):
        self.properties.SetShading(is_active)

    @property
    def representation(self) -> RepresentationType:
        """Return or set the representation of the actor.

        Can be any of the options in :class:`pyvista.plotting.opts.RepresentationType` enum.
        """
        return RepresentationType.from_any(self.properties.GetRepresentation())

    @representation.setter
    def representation(self, value: RepresentationType):
        self.properties.SetRepresentation(RepresentationType.from_any(value).value)
