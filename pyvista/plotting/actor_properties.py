"""Module containing pyvista implementation of :vtk:`vtkProperty`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyvista._warn_external import warn_external
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.misc import _NoNewAttrMixin

from .opts import InterpolationType
from .opts import RepresentationType

if TYPE_CHECKING:
    from pyvista import _vtk


class ActorProperties(_NoNewAttrMixin):
    """Properties wrapper for :vtk:`vtkProperty`.

    Contains the surface properties of the object.

    .. deprecated:: 0.49
        Use :class:`pyvista.Property` instead. Its
        :attr:`~pyvista.Property.interpolation` replaces ``interpolation_model``,
        and ``shading`` has no equivalent.

    Parameters
    ----------
    properties : :vtk:`vtkProperty`
        VTK properties of the current object.

    """

    def __init__(self, properties: _vtk.vtkProperty) -> None:
        """Initialize the wrapper."""
        # deprecated 0.49, convert to error in 0.52, remove 0.53
        warn_external(
            '`ActorProperties` is deprecated. Use `pyvista.Property` instead.',
            PyVistaDeprecationWarning,
        )
        super().__init__()
        self.properties = properties

    @property
    def color(self):  # numpydoc ignore=RT01
        """Return or set the color of the actor."""
        return self.properties.GetColor()

    @color.setter
    def color(self, color: tuple[float, float, float]):
        self.properties.SetColor(color[0], color[1], color[2])

    @property
    def metallic(self):  # numpydoc ignore=RT01
        """Return or set the metallic coefficient of the surface."""
        return self.properties.GetMetallic()

    @metallic.setter
    def metallic(self, value: float):
        self.properties.SetMetallic(value)

    @property
    def roughness(self):  # numpydoc ignore=RT01
        """Return or set the roughness of the surface."""
        return self.properties.GetRoughness()

    @roughness.setter
    def roughness(self, value: float):
        self.properties.SetRoughness(value)

    @property
    def anisotropy(self):  # numpydoc ignore=RT01
        """Return or set the anisotropy coefficient."""
        return self.properties.GetAnisotropy()

    @anisotropy.setter
    def anisotropy(self, value: float):
        self.properties.SetAnisotropy(value)

    @property
    def anisotropy_rotation(self):  # numpydoc ignore=RT01
        """Return or set the anisotropy rotation coefficient."""
        return self.properties.GetAnisotropyRotation()

    @anisotropy_rotation.setter
    def anisotropy_rotation(self, value: float):
        self.properties.SetAnisotropyRotation(value)

    @property
    def lighting(self):  # numpydoc ignore=RT01
        """Return or set the lighting activation flag."""
        return self.properties.GetLighting()

    @lighting.setter
    def lighting(self, flag: bool):
        self.properties.SetLighting(flag)

    @property
    def interpolation_model(self):  # numpydoc ignore=RT01
        """Return or set the interpolation model.

        Can be any of the options in :class:`pyvista.plotting.opts.InterpolationType` enum.
        """
        return InterpolationType.from_any(self.properties.GetInterpolation())

    @interpolation_model.setter
    def interpolation_model(self, model: InterpolationType):
        self.properties.SetInterpolation(model.value)

    @property
    def index_of_refraction(self):  # numpydoc ignore=RT01
        """Return or set the Index Of Refraction of the base layer."""
        return self.properties.GetBaseIOR()

    @index_of_refraction.setter
    def index_of_refraction(self, value: float):
        self.properties.SetBaseIOR(value)

    @property
    def opacity(self):  # numpydoc ignore=RT01
        """Return or set the opacity of the actor."""
        return self.properties.GetOpacity()

    @opacity.setter
    def opacity(self, value: float):
        self.properties.SetOpacity(value)

    @property
    def shading(self):  # numpydoc ignore=RT01
        """Return or set the flag to activate the shading."""
        return self.properties.GetShading()

    @shading.setter
    def shading(self, is_active: bool):
        self.properties.SetShading(is_active)

    @property
    def representation(self) -> RepresentationType:  # numpydoc ignore=RT01
        """Return or set the representation of the actor.

        Can be any of the options in :class:`pyvista.plotting.opts.RepresentationType` enum.
        """
        return RepresentationType.from_any(self.properties.GetRepresentation())

    @representation.setter
    def representation(self, value: RepresentationType):
        self.properties.SetRepresentation(RepresentationType.from_any(value).value)
