"""This module contains the Property class."""
from pyvista import _vtk

from .colors import Color


class Property(_vtk.vtkProperty):
    """Wrap vtkProperty."""

    def __init__(
        self,
        theme,
        interpolation=None,
        color=None,
        style=None,
        metallic=None,
        roughness=None,
        point_size=None,
        opacity=None,
        ambient=None,
        diffuse=None,
        specular=None,
        specular_power=None,
        show_edges=None,
        edge_color=None,
        render_points_as_spheres=None,
        render_lines_as_tubes=None,
        lighting=None,
        line_width=None,
    ):
        """Initialize this property."""
        self._theme = theme
        self.interpolation = interpolation
        self.color = color
        self.style = style
        self.metallic = metallic
        self.roughness = roughness
        self.point_size = point_size
        self.opacity = opacity
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.specular_power = specular_power
        self.show_edges = show_edges
        self.edge_color = edge_color
        self.render_points_as_spheres = render_points_as_spheres
        self.render_lines_as_tubes = render_lines_as_tubes
        self.lighting = lighting
        self.line_width = line_width

    @property
    def style(self) -> str:
        """Return or set the representation."""
        return self.GetRepresentationAsString()

    @style.setter
    def style(self, new_style: str):
        if new_style is None:
            new_style = 'surface'
        new_style = new_style.lower()

        if new_style == 'wireframe':
            self.SetRepresentationToWireframe()
            if not self._color_set:
                self.color = self._theme.outline_color
        elif new_style == 'points':
            self.SetRepresentationToPoints()
        elif new_style == 'surface':
            self.SetRepresentationToSurface()
        else:
            raise ValueError(
                f'Invalid style "{new_style}".  Must be one of the following:\n'
                '\t"surface"\n'
                '\t"wireframe"\n'
                '\t"points"\n'
            )

    @property
    def color(self):
        """Return or set the color of this property."""
        return Color(self.GetColor())

    @color.setter
    def color(self, new_color):
        self._color_set = new_color is None
        rgb_color = Color(new_color, default_color=self._theme.color)
        self.SetColor(rgb_color.float_rgb)

    @property
    def edge_color(self):
        """Return or set the edge color of this property."""
        return Color(self.GetEdgeColor())

    @edge_color.setter
    def edge_color(self, new_color):
        rgb_color = Color(new_color, default_color=self._theme.edge_color)
        self.SetEdgeColor(rgb_color.float_rgb)

    @property
    def opacity(self):
        """Return or set the opacity of this property."""
        return self.GetOpacity()

    @opacity.setter
    def opacity(self, value):
        if value is None:
            return
        self.SetOpacity(value)

    @property
    def show_edges(self):
        """Return or set show edges."""
        return self.GetEdgeVisibility()

    @show_edges.setter
    def show_edges(self, value):
        if value is None:
            value = self._theme.show_edges
        self.SetEdgeVisibility(value)

    @property
    def lighting(self) -> bool:
        """Return or set lighting."""
        return self.SetLighting()

    @lighting.setter
    def lighting(self, value):
        if value is None:
            value = self._theme.lighting
        self.SetLighting(value)

    @property
    def ambient(self):
        """Return or set ambient."""
        return self.GetAmbient()

    @ambient.setter
    def ambient(self, new_ambient):
        self.SetAmbient(new_ambient)

    @property
    def diffuse(self):
        """Return or set diffuse."""
        return self.GetDiffuse()

    @diffuse.setter
    def diffuse(self, new_diffuse):
        self.SetDiffuse(new_diffuse)

    @property
    def specular(self):
        """Return or set specular."""
        return self.GetSpecular()

    @specular.setter
    def specular(self, new_specular):
        self.SetSpecular(new_specular)

    @property
    def specular_power(self):
        """Return or set specular power."""
        return self.GetSpecularPower()

    @specular_power.setter
    def specular_power(self, new_specular_power):
        self.SetSpecularPower(new_specular_power)

    @property
    def metallic(self):
        """Return or set metallic."""
        return self.GetMetallic()

    @metallic.setter
    def metallic(self, new_metallic):
        self.SetMetallic(new_metallic)

    @property
    def roughness(self):
        """Return or set roughness."""
        return self.GetRoughness()

    @roughness.setter
    def roughness(self, new_roughness):
        self.SetRoughness(new_roughness)

    @property
    def interpolation(self) -> str:
        """Return or set show edges of this property."""
        return self.GetInterpolationAsString()

    @interpolation.setter
    def interpolation(self, new_interpolation):
        if new_interpolation == 'Physically based rendering':
            if not _vtk.VTK9:  # pragma: no cover
                from pyvista.core.errors import VTKVersionError

                raise VTKVersionError('Physically based rendering requires VTK 9 or newer.')
            self.SetInterpolationToPBR()
        elif new_interpolation == 'Phong':
            self.SetInterpolationToPhong()
        elif new_interpolation == 'Gouraud':
            self.SetInterpolationToGouraud()
        elif new_interpolation == 'Flat':
            self.SetInterpolationToFlat()
        else:
            raise ValueError(f'Invalid interpolation "{new_interpolation}"')

    @property
    def render_points_as_spheres(self):
        """Return or set rendering points as spheres."""
        self.GetRenderPointsAsSpheres()

    @render_points_as_spheres.setter
    def render_points_as_spheres(self, new_render_points_as_spheres):
        if new_render_points_as_spheres is None:
            return
        self.SetRenderPointsAsSpheres(new_render_points_as_spheres)

    @property
    def render_lines_as_tubes(self):
        """Return or set rendering lines as tubes."""
        self.GetRenderLinesAsTubes()

    @render_lines_as_tubes.setter
    def render_lines_as_tubes(self, new_value):
        if new_value is None:
            return
        self.SetRenderLinesAsTubes(new_value)

    @property
    def line_width(self):
        """Return or set the line width."""
        self.GetLineWidth()

    @line_width.setter
    def line_width(self, new_size):
        if new_size is None:
            return
        self.SetLineWidth(new_size)

    @property
    def point_size(self):
        """Return or set the point size."""
        self.GetPointSize()

    @point_size.setter
    def point_size(self, new_size):
        if new_size is None:
            return
        self.SetPointSize(new_size)
