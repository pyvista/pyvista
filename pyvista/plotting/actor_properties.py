"""Module containing pyvista implementation of vtkProperty.

.. deprecated:: 0.43.0

    This class is deprecated. Use :class:`pyvista.Property` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Tuple
import warnings

from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.plotting.opts import InterpolationType
from pyvista.plotting.opts import RepresentationType

if TYPE_CHECKING:
    from pyvista.plotting import _vtk


class ActorProperties:
    """Properties wrapper for ``vtkProperty``.

    Contains the surface properties of the object.

    Parameters
    ----------
    properties : vtk.vtkProperty
        VTK properties of the current object.

    Examples
    --------
    Access the properties of the z-axis shaft.

    >>> import pyvista as pv

    >>> axes = pv.Axes()
    >>> z_axes_prop = axes.axes_actor.z_axis_shaft_properties
    >>> z_axes_prop.color = (1.0, 1.0, 0.0)
    >>> z_axes_prop.opacity = 0.5
    >>> axes.axes_actor.shaft_type = axes.axes_actor.ShaftType.CYLINDER
    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(axes.axes_actor)  # doctest:+SKIP
    >>> _ = pl.add_mesh(pv.Sphere())
    >>> pl.show()

    """

    def __init__(self, properties: _vtk.vtkProperty) -> None:
        super().__init__()
        self.properties = properties

        warnings.warn(
            "Use of `ActorProperties` is deprecated. Use `pyvista.Property` instead.",
            PyVistaDeprecationWarning,
        )

    @property
    def color(self):  # numpydoc ignore=RT01
        """Return or set the color of the actor."""
        return self.properties.GetColor()  # pragma:no cover

    @color.setter
    def color(self, color: Tuple[float, float, float]):  # numpydoc ignore=GL08
        self.properties.SetColor(color[0], color[1], color[2])  # pragma:no cover

    @property
    def metallic(self):  # numpydoc ignore=RT01
        """Return or set the metallic coefficient of the surface."""
        return self.properties.GetMetallic()  # pragma:no cover

    @metallic.setter
    def metallic(self, value: float):  # numpydoc ignore=GL08
        self.properties.SetMetallic(value)  # pragma:no cover

    @property
    def roughness(self):  # numpydoc ignore=RT01
        """Return or set the roughness of the surface."""
        return self.properties.GetRoughness()  # pragma:no cover

    @roughness.setter
    def roughness(self, value: float):  # numpydoc ignore=GL08
        self.properties.SetRoughness(value)  # pragma:no cover

    @property
    def anisotropy(self):  # numpydoc ignore=RT01
        """Return or set the anisotropy coefficient."""
        return self.properties.GetAnisotropy()  # pragma:no cover

    @anisotropy.setter
    def anisotropy(self, value: float):  # numpydoc ignore=GL08
        self.properties.SetAnisotropy(value)  # pragma:no cover

    @property
    def anisotropy_rotation(self):  # numpydoc ignore=RT01
        """Return or set the anisotropy rotation coefficient."""
        return self.properties.GetAnisotropyRotation()  # pragma:no cover

    @anisotropy_rotation.setter
    def anisotropy_rotation(self, value: float):  # numpydoc ignore=GL08
        self.properties.SetAnisotropyRotation(value)  # pragma:no cover

    @property
    def lighting(self):  # numpydoc ignore=RT01
        """Return or set the lighting activation flag."""
        return self.properties.GetLighting()  # pragma:no cover

    @lighting.setter
    def lighting(self, flag: bool):  # numpydoc ignore=GL08
        self.properties.SetLighting(flag)  # pragma:no cover

    @property
    def interpolation_model(self):  # numpydoc ignore=RT01
        """Return or set the interpolation model.

        Can be any of the options in :class:`pyvista.plotting.opts.InterpolationType` enum.
        """
        return InterpolationType.from_any(self.properties.GetInterpolation())  # pragma:no cover

    @interpolation_model.setter
    def interpolation_model(self, model: InterpolationType):  # numpydoc ignore=GL08
        self.properties.SetInterpolation(model.value)  # pragma:no cover

    @property
    def index_of_refraction(self):  # numpydoc ignore=RT01
        """Return or set the Index Of Refraction of the base layer."""
        return self.properties.GetBaseIOR()  # pragma:no cover

    @index_of_refraction.setter
    def index_of_refraction(self, value: float):  # numpydoc ignore=GL08
        self.properties.SetBaseIOR(value)  # pragma:no cover

    @property
    def opacity(self):  # numpydoc ignore=RT01
        """Return or set the opacity of the actor."""
        return self.properties.GetOpacity()  # pragma:no cover

    @opacity.setter
    def opacity(self, value: float):  # numpydoc ignore=GL08
        self.properties.SetOpacity(value)  # pragma:no cover

    @property
    def shading(self):  # numpydoc ignore=RT01
        """Return or set the flag to activate the shading."""
        return self.properties.GetShading()  # pragma:no cover

    @shading.setter
    def shading(self, is_active: bool):  # numpydoc ignore=GL08
        self.properties.SetShading(is_active)  # pragma:no cover

    @property
    def representation(self) -> RepresentationType:  # numpydoc ignore=RT01
        """Return or set the representation of the actor.

        Can be any of the options in :class:`pyvista.plotting.opts.RepresentationType` enum.
        """
        return RepresentationType.from_any(self.properties.GetRepresentation())  # pragma:no cover

    @representation.setter
    def representation(self, value: RepresentationType):  # numpydoc ignore=GL08
        self.properties.SetRepresentation(
            RepresentationType.from_any(value).value,
        )  # pragma:no cover

    @property
    def ambient(self):  # numpydoc ignore=RT01
        """Return or set the ambient lighting coefficient.

        Value should be between 0 and 1.

        """
        return self.properties.GetAmbient()  # pragma:no cover

    @ambient.setter
    def ambient(self, ambient: float):  # numpydoc ignore=GL08
        self.properties.SetAmbient(ambient)  # pragma:no cover
