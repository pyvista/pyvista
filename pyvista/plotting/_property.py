"""This module contains the Property class."""
from functools import lru_cache

import pyvista
from pyvista import _vtk

from .colors import Color


@lru_cache(maxsize=None)
def _check_supports_pbr():
    """Check if VTK supports physically based rendering."""
    if not _vtk.VTK9:  # pragma: no cover
        from pyvista.core.errors import VTKVersionError

        raise VTKVersionError('Physically based rendering requires VTK 9 or newer.')


class Property(_vtk.vtkProperty):
    """Wrap vtkProperty and expose it pythonically.

    This class is useful when needing to set the property of actors.

    Parameters
    ----------
    theme : pyvista.themes.DefaultTheme, optional
        Plot-specific theme.

    interpolation : str
        Set the method of shading. One of the following:

        * ``'Physically based rendering'``
        * ``'pbr'` - Alias for Physically based rendering.
        * ``'Phong'`` - Phong shading
        * ``'Gouraud'`` - Gouraud shading
        * ``'Flat'`` - Flat Shading

    color : color_like, optional
        Used to make the entire mesh have a single solid color.
        Either a string, RGB list, or hex color string.  For example:
        ``color='white'``, ``color='w'``, ``color=[1.0, 1.0, 1.0]``, or
        ``color='#FFFFFF'``. Color will be overridden if scalars are
        specified.

    style : str, optional
        Visualization style of the mesh.  One of the following:
        ``style='surface'``, ``style='wireframe'``, ``style='points'``.
        Defaults to ``'surface'``. Note that ``'wireframe'`` only shows a
        wireframe of the outer geometry.

    metallic : float, optional
        Usually this value is either 0 or 1 for a real material
        but any value in between is valid. This parameter is only
        used by PBR interpolation.

    roughness : float, optional
        This value has to be between 0 (glossy) and 1 (rough). A
        glossy material has reflections and a high specular
        part. This parameter is only used by PBR
        interpolation.

    point_size : float, optional
        Size of the points represented by this property.

    opacity : float, optional
        Opacity of the mesh. A single float value that will be applied globally
        opacity of the mesh and uniformly applied everywhere - should be
        between 0 and 1.

    ambient : float, optional
        When lighting is enabled, this is the amount of light in
        the range of 0 to 1 (default 0.0) that reaches the actor
        when not directed at the light source emitted from the
        viewer.

    diffuse : float, optional
        The diffuse lighting coefficient. Default 1.0.

    specular : float, optional
        The specular lighting coefficient. Default 0.0.

    specular_power : float, optional
        The specular power. Between 0.0 and 128.0.

    show_edges : bool, optional
        Shows the edges.  Does not apply to a wireframe
        representation.

    edge_color : color_like, optional
        The solid color to give the edges when ``show_edges=True``.
        Either a string, RGB list, or hex color string.

    render_points_as_spheres : bool, optional
        Render points as spheres rather than dots.

    render_lines_as_tubes : bool, optional
        Show lines as thick tubes rather than flat lines.  Control
        the width with ``line_width``.

    lighting : bool, optional
        Enable or disable view direction lighting.

    culling : str, bool, optional
        Does not render faces that are culled. This can be helpful for
        dense surface meshes, especially when edges are visible, but can
        cause flat meshes to be partially displayed. Defaults to
        ``False``. One of the following:

        * ``True``
        * ``"b"`` - Enable backface culling
        * ``"back"`` - Enable backface culling
        * ``"backface"`` - Enable backface culling
        * ``"f"`` - Enable frontface culling
        * ``"front"`` - Enable frontface culling
        * ``"frontface"`` - Enable frontface culling
        * ``False`` - Disable both backface and frontface culling

    Examples
    --------
    Create a vtk Actor and assign properties to it.

    >>> import vtk
    >>> import pyvista as pv
    >>> prop = pv.Property(
    ...     color='r',
    ...     show_edges=True,
    ...     interpolation='Physically based rendering',
    ...     metallic=0.5,
    ...     roughness=0.1
    ... )
    >>> actor = vtk.vtkActor()
    >>> actor.SetProperty(prop)

    """

    def __init__(
        self,
        theme=None,
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
        culling=None,
    ):
        """Initialize this property."""
        self._theme = pyvista.themes.DefaultTheme()
        if theme is None:
            # copy global theme to ensure local property theme is fixed
            # after creation.
            self._theme.load_theme(pyvista.global_theme)
        else:
            self._theme.load_theme(theme)

        self.interpolation = interpolation
        self.color = color
        self.style = style
        if interpolation in ['Physically based rendering', 'pbr']:
            if metallic is not None:
                self.metallic = metallic
            if roughness is not None:
                self.roughness = roughness
        self.point_size = point_size
        if opacity is not None:
            self.opacity = opacity
        if ambient is not None:
            self.ambient = ambient
        if diffuse is not None:
            self.diffuse = diffuse
        if specular is not None:
            self.specular = specular
        if specular_power is not None:
            self.specular_power = specular_power
        self.show_edges = show_edges
        self.edge_color = edge_color
        self.render_points_as_spheres = render_points_as_spheres
        self.render_lines_as_tubes = render_lines_as_tubes
        self.lighting = lighting
        self.line_width = line_width

        if culling is not None:
            self.culling = culling

    @property
    def style(self) -> str:
        """Return or set Visualization style of the mesh.

        One of the following (case insensitive)

        * ``'surface'``
        * ``'wireframe'``
        * ``'points'``

        Examples
        --------
        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.style = 'Surface'
        >>> prop.style
        'Surface'

        """
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
    def color(self) -> Color:
        """Return or set the color of this property.

        Either a string, RGB list, or hex color string.  For example:
        ``color='white'``, ``color='w'``, ``color=[1.0, 1.0, 1.0]``, or
        ``color='#FFFFFF'``. Color will be overridden if scalars are
        specified.

        Examples
        --------
        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.color = 'b'
        >>> prop.color
        Color(name='blue', hex='#0000ffff')

        """
        return Color(self.GetColor())

    @color.setter
    def color(self, new_color):
        self._color_set = new_color is None
        rgb_color = Color(new_color, default_color=self._theme.color)
        self.SetColor(rgb_color.float_rgb)

    @property
    def edge_color(self) -> Color:
        """Return or set the edge color of this property.

        The solid color to give the edges when ``show_edges=True``.
        Either a string, RGB list, or hex color string.

        Examples
        --------
        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.edge_color = 'brown'
        >>> prop.edge_color
        Color(name='brown', hex='#654321ff')

        """
        return Color(self.GetEdgeColor())

    @edge_color.setter
    def edge_color(self, new_color):
        rgb_color = Color(new_color, default_color=self._theme.edge_color)
        self.SetEdgeColor(rgb_color.float_rgb)

    @property
    def opacity(self) -> float:
        """Return or set the opacity of this property.

        Opacity of the mesh. A single float value that will be applied globally
        opacity of the mesh and uniformly applied everywhere - should be
        between 0 and 1.

        Examples
        --------
        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.opacity = 0.5
        >>> prop.opacity
        0.5

        """
        return self.GetOpacity()

    @opacity.setter
    def opacity(self, value):
        if value is None:
            return
        self.SetOpacity(value)

    @property
    def show_edges(self) -> bool:
        """Return or set the visibility of edges.

        Shows or hides the edges.  Does not apply to a wireframe
        representation.

        Examples
        --------
        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.show_edges = False
        >>> prop.show_edges
        False

        """
        return bool(self.GetEdgeVisibility())

    @show_edges.setter
    def show_edges(self, value):
        if value is None:
            value = self._theme.show_edges
        self.SetEdgeVisibility(value)

    @property
    def lighting(self) -> bool:
        """Return or set view direction lighting.

        Examples
        --------
        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.lighting = True
        >>> prop.lighting
        True

        """
        return self.GetLighting()

    @lighting.setter
    def lighting(self, value):
        if value is None:
            value = self._theme.lighting
        self.SetLighting(value)

    @property
    def ambient(self) -> float:
        """Return or set ambient.

        When lighting is enabled, this is the amount of light in
        the range of 0 to 1 (default 0.0) that reaches the actor
        when not directed at the light source emitted from the
        viewer.

        Examples
        --------
        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.ambient = 0.2
        >>> prop.ambient
        0.2

        """
        return self.GetAmbient()

    @ambient.setter
    def ambient(self, new_ambient: float):
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
        _check_supports_pbr()
        return self.GetMetallic()

    @metallic.setter
    def metallic(self, new_metallic):
        _check_supports_pbr()
        self.SetMetallic(new_metallic)

    @property
    def roughness(self):
        """Return or set roughness."""
        _check_supports_pbr()
        return self.GetRoughness()

    @roughness.setter
    def roughness(self, new_roughness):
        _check_supports_pbr()
        self.SetRoughness(new_roughness)

    @property
    def interpolation(self) -> str:
        """Return or set show edges of this property."""
        return self.GetInterpolationAsString()

    @interpolation.setter
    def interpolation(self, new_interpolation):
        if new_interpolation in ['Physically based rendering', 'pbr']:
            _check_supports_pbr()

            self.SetInterpolationToPBR()
        elif new_interpolation == 'Phong':
            self.SetInterpolationToPhong()
        elif new_interpolation == 'Gouraud':
            self.SetInterpolationToGouraud()
        elif new_interpolation == 'Flat' or new_interpolation is None:
            self.SetInterpolationToFlat()
        else:
            raise ValueError(
                f'Invalid interpolation "{new_interpolation}". Should be one of the '
                'following:\n'
                '    - "Physically based rendering"\n'
                '    - "pbr"\n'
                '    - "Phong"\n'
                '    - "Gouraud"\n'
                '    - "Flat"\n'
                '    - None'
            )

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

    @property
    def culling(self):
        """Return or set face culling."""
        if self.BackfaceCullingOn():
            return 'back'
        elif self.FrontfaceCullingOn():
            return 'front'
        return

    @culling.setter
    def culling(self, value):
        if isinstance(value, str):
            value = value.lower()

        if value in [True, 'back', 'backface', 'b']:
            try:
                self.BackfaceCullingOn()
            except AttributeError:  # pragma: no cover
                pass
        elif value in ['front', 'frontface', 'f']:
            try:
                self.FrontfaceCullingOn()
            except AttributeError:  # pragma: no cover
                pass
        elif value is False:
            self.FrontfaceCullingOff()
            self.BackfaceCullingOff()
        else:
            raise ValueError(
                f'Culling option ({value}) not understood. Should be either:\n'
                'True, "back", "backface", "b", "front", "frontface", or "f"'
            )
