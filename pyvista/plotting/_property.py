"""This module contains the Property class."""
from typing import Union

import pyvista
from pyvista import vtk_version_info
from pyvista.core.utilities.misc import _check_range, no_new_attr

from . import _vtk
from .colors import Color
from .opts import InterpolationType


@no_new_attr
class Property(_vtk.vtkProperty):
    """Wrap vtkProperty and expose it pythonically.

    This class is used to set the property of actors.

    Parameters
    ----------
    theme : pyvista.plotting.themes.Theme, optional
        Plot-specific theme.

    interpolation : str, default: :attr:`pyvista.plotting.themes._LightingConfig.interpolation`
        Set the method of shading. One of the following:

        * ``'Physically based rendering'`` - Physically based rendering.
        * ``'pbr'`` - Alias for Physically based rendering.
        * ``'Phong'`` - Phong shading.
        * ``'Gouraud'`` - Gouraud shading.
        * ``'Flat'`` - Flat Shading.

        This parameter is case insensitive.

    color : ColorLike, default: :attr:`pyvista.plotting.themes.Theme.color`
        Used to make the entire mesh have a single solid color.
        Either a string, RGB list, or hex color string.  For example:
        ``color='white'``, ``color='w'``, ``color=[1.0, 1.0, 1.0]``, or
        ``color='#FFFFFF'``. Color will be overridden if scalars are
        specified.

    style : str, default: 'surface'
        Visualization style of the mesh.  One of the following:
        ``style='surface'``, ``style='wireframe'``, ``style='points'``.
        Note that ``'wireframe'`` only shows a wireframe of the outer
        geometry.

    metallic : float, default: :attr:`pyvista.plotting.themes._LightingConfig.metallic`
        Usually this value is either 0 or 1 for a real material but any
        value in between is valid. This parameter is only used by PBR
        interpolation.

    roughness : float, default: :attr:`pyvista.plotting.themes._LightingConfig.roughness`
        This value has to be between 0 (glossy) and 1 (rough). A glossy
        material has reflections and a high specular part. This parameter
        is only used by PBR interpolation.

    point_size : float, default: :attr:`pyvista.plotting.themes.Theme.point_size`
        Size of the points represented by this property.

    opacity : float, default: :attr:`pyvista.plotting.themes.Theme.opacity`
        Opacity of the mesh. A single float value that will be applied globally
        opacity of the mesh and uniformly applied everywhere - should be
        between 0 and 1.

    ambient : float, default: :attr:`pyvista.plotting.themes._LightingConfig.ambient`
        When lighting is enabled, this is the amount of light in the range
        of 0 to 1 that reaches the actor when not directed at the light
        source emitted from the viewer.

    diffuse : float, default: :attr:`pyvista.plotting.themes._LightingConfig.diffuse`
        The diffuse lighting coefficient.

    specular : float, default: :attr:`pyvista.plotting.themes._LightingConfig.specular`
        The specular lighting coefficient.

    specular_power : float, default: :attr:`pyvista.plotting.themes._LightingConfig.specular_power`
        The specular power. Must be between 0.0 and 128.0.

    show_edges : bool, default: :attr:`pyvista.plotting.themes.Theme.show_edges`
        Shows the edges.  Does not apply to a wireframe representation.

    edge_color : ColorLike, default: :attr:`pyvista.plotting.themes.Theme.edge_color`
        The solid color to give the edges when ``show_edges=True``.
        Either a string, RGB list, or hex color string.

    render_points_as_spheres : bool, default: :attr:`pyvista.plotting.themes.Theme.render_points_as_spheres`
        Render points as spheres rather than dots.

    render_lines_as_tubes : bool, default: :attr:`pyvista.plotting.themes.Theme.render_lines_as_tubes`
        Show lines as thick tubes rather than flat lines.  Control
        the width with ``line_width``.

    lighting : bool, default: :attr:`pyvista.plotting.themes.Theme.lighting`
        Enable or disable view direction lighting.

    line_width : float, default: :attr:`pyvista.plotting.themes.Theme.line_width`
        Thickness of lines.  Only valid for wireframe and surface
        representations.

    culling : str | bool, optional
        Does not render faces that are culled. This can be helpful for
        dense surface meshes, especially when edges are visible, but can
        cause flat meshes to be partially displayed. Defaults to
        ``'none'``. One of the following:

        * ``"back"`` - Enable backface culling
        * ``"front"`` - Enable frontface culling
        * ``'none'`` - Disable both backface and frontface culling

    edge_opacity : float, default: :attr:`pyvista.plotting.themes.Theme.edge_opacity`
        Edge opacity of the mesh. A single float value that will be applied globally
        edge opacity of the mesh and uniformly applied everywhere - should be
        between 0 and 1.

        .. note::
            `edge_opacity` uses ``SetEdgeOpacity`` as the underlying method which
            requires VTK version 9.3 or higher. If ``SetEdgeOpacity`` is not
            available, `edge_opacity` is set to 1.

    Examples
    --------
    Create a :class:`pyvista.Actor` and assign properties to it.

    >>> import pyvista as pv
    >>> actor = pv.Actor()
    >>> actor.prop = pv.Property(
    ...     color='r',
    ...     show_edges=True,
    ...     interpolation='Physically based rendering',
    ...     metallic=0.5,
    ...     roughness=0.1,
    ... )

    Visualize how the property would look when applied to a mesh.

    >>> actor.prop.plot()

    Set custom properties not directly available in
    :func:`pyvista.Plotter.add_mesh`. Here, we set diffuse, ambient, and
    specular power and colors.

    >>> pl = pv.Plotter()
    >>> actor = pl.add_mesh(pv.Sphere())
    >>> prop = actor.prop
    >>> prop.diffuse = 0.6
    >>> prop.diffuse_color = 'w'
    >>> prop.ambient = 0.3
    >>> prop.ambient_color = 'r'
    >>> prop.specular = 0.5
    >>> prop.specular_color = 'b'
    >>> pl.show()

    """

    _theme = None
    _color_set = None

    def __init__(
        self,
        theme=None,
        interpolation=None,
        color=None,
        style='surface',
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
        edge_opacity=None,
    ):
        """Initialize this property."""
        self._theme = pyvista.themes.Theme()
        if theme is None:
            # copy global theme to ensure local property theme is fixed
            # after creation.
            self._theme.load_theme(pyvista.global_theme)
        else:
            self._theme.load_theme(theme)

        if interpolation is None:
            interpolation = self._theme.lighting_params.interpolation
        self.interpolation = interpolation

        self.color = color

        if style is not None:
            self.style = style

        if self.interpolation == InterpolationType.PBR:
            if metallic is None:
                metallic = self._theme.lighting_params.metallic
            self.metallic = metallic
            if roughness is None:
                roughness = self._theme.lighting_params.roughness
            self.roughness = roughness

        if point_size is None:
            point_size = self._theme.point_size
        self.point_size = point_size
        if opacity is None:
            opacity = self._theme.opacity
        self.opacity = opacity
        if ambient is None:
            ambient = self._theme.lighting_params.ambient
        self.ambient = ambient
        if diffuse is None:
            diffuse = self._theme.lighting_params.diffuse
        self.diffuse = diffuse
        if specular is None:
            specular = self._theme.lighting_params.specular
        self.specular = specular
        if specular_power is None:
            specular_power = self._theme.lighting_params.specular_power
        self.specular_power = specular_power

        if show_edges is None:
            self.show_edges = self._theme.show_edges
        else:
            self.show_edges = show_edges

        self.edge_color = edge_color
        if render_points_as_spheres is None:
            render_points_as_spheres = self._theme.render_points_as_spheres
        self.render_points_as_spheres = render_points_as_spheres
        if render_lines_as_tubes is None:
            render_lines_as_tubes = self._theme.render_lines_as_tubes
        self.render_lines_as_tubes = render_lines_as_tubes
        self.lighting = lighting
        if line_width is None:
            line_width = self._theme.line_width
        self.line_width = line_width
        if culling is not None:
            self.culling = culling
        if vtk_version_info < (9, 3) and edge_opacity is not None:  # pragma: no cover
            import warnings

            warnings.warn(
                '`edge_opacity` cannot be used under VTK v9.3.0. Try installing VTK v9.3.0 or newer.',
                UserWarning,
            )
        if edge_opacity is None:
            edge_opacity = self._theme.edge_opacity
        self.edge_opacity = edge_opacity

    @property
    def style(self) -> str:  # numpydoc ignore=RT01
        """Return or set Visualization style of the mesh.

        One of the following (case insensitive):

        * ``'surface'``
        * ``'wireframe'``
        * ``'points'``

        Examples
        --------
        Set the representation style to ``'Wireframe'``

        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.style = 'wireframe'
        >>> prop.style
        'Wireframe'

        Visualize default surface representation style.

        >>> prop.style = 'surface'
        >>> prop.plot()

        Visualize wireframe representation style.

        >>> prop.style = 'wireframe'
        >>> prop.plot()

        Visualize points representation style.

        >>> prop.style = 'points'
        >>> prop.point_size = 5.0
        >>> prop.plot()
        """
        return self.GetRepresentationAsString()

    @style.setter
    def style(self, new_style: str):  # numpydoc ignore=GL08
        new_style = new_style.lower()

        if new_style == 'wireframe':
            self.SetRepresentationToWireframe()
            if not self._color_set:
                self.color = self._theme.outline_color  # type: ignore
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
    def color(self) -> Color:  # numpydoc ignore=RT01
        """Return or set the color of this property.

        Either a string, RGB list, or hex color string.  For example:
        ``color='white'``, ``color='w'``, ``color=[1.0, 1.0, 1.0]``, or
        ``color='#FFFFFF'``. Color will be overridden if scalars are
        specified.

        Examples
        --------
        Set the color to blue.

        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.color = 'b'
        >>> prop.color
        Color(name='blue', hex='#0000ffff', opacity=255)

        Visualize setting the property to blue.

        >>> prop.color = 'b'
        >>> prop.plot()

        Visualize setting the color using an RGB value.

        >>> prop.color = (0.5, 0.5, 0.1)
        >>> prop.plot()

        """
        return Color(self.GetColor())

    @color.setter
    def color(self, value):  # numpydoc ignore=GL08
        self._color_set = value is not None
        rgb_color = Color(value, default_color=self._theme.color)
        self.SetColor(rgb_color.float_rgb)

    @property
    def edge_color(self) -> Color:  # numpydoc ignore=RT01
        """Return or set the edge color of this property.

        The solid color to give the edges when ``show_edges=True``.
        Either a string, RGB list, or hex color string.

        Examples
        --------
        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.edge_color = 'red'
        >>> prop.edge_color
        Color(name='red', hex='#ff0000ff', opacity=255)

        Visualize red edges. Set the edge's visibility to ``True`` so we can see
        them.

        >>> prop.show_edges = True
        >>> prop.edge_color = 'red'
        >>> prop.plot()

        """
        return Color(self.GetEdgeColor())

    @edge_color.setter
    def edge_color(self, value):  # numpydoc ignore=GL08
        rgb_color = Color(value, default_color=self._theme.edge_color)
        self.SetEdgeColor(rgb_color.float_rgb)

    @property
    def opacity(self) -> float:  # numpydoc ignore=RT01
        """Return or set the opacity of this property.

        Opacity of the mesh. A single float value that will be applied globally
        opacity of the mesh and uniformly applied everywhere. Between 0 and 1.

        Examples
        --------
        Set opacity to ``0.5``.

        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.opacity = 0.5
        >>> prop.opacity
        0.5

        Visualize default opacity of ``1.0``.

        >>> prop.opacity = 1.0
        >>> prop.plot()

        Visualize opacity of ``0.75``.

        >>> prop.opacity = 0.75
        >>> prop.plot()

        Visualize opacity of ``0.25``.

        >>> prop.opacity = 0.25
        >>> prop.plot()


        """
        return self.GetOpacity()

    @opacity.setter
    def opacity(self, value: float):  # numpydoc ignore=GL08
        _check_range(value, (0, 1), 'opacity')
        self.SetOpacity(value)

    @property
    def edge_opacity(self) -> float:  # numpydoc ignore=RT01
        """Return or set the edge opacity of this property.

        Edge opacity of the mesh. A single float value that will be applied globally
        edge opacity of the mesh and uniformly applied everywhere. Between 0 and 1.

        .. note::
            `edge_opacity` uses ``SetEdgeOpacity`` as the underlying method which
            requires VTK version 9.3 or higher. If ``SetEdgeOpacity`` is not
            available, `edge_opacity` is set to 1.

        Examples
        --------
        Set edge opacity to ``0.5``.

        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.show_edges = True
        >>> prop.edge_opacity = 0.5
        >>> prop.edge_opacity
        0.5

        Visualize default edge opacity of ``1.0``.

        >>> prop.edge_opacity = 1.0
        >>> prop.plot()

        Visualize edge opacity of ``0.1``.

        >>> prop.edge_opacity = 0.1
        >>> prop.plot()

        """
        if vtk_version_info < (9, 3):
            return 1.0
        else:
            return self.GetEdgeOpacity()

    @edge_opacity.setter
    def edge_opacity(self, value: float):  # numpydoc ignore=GL08
        _check_range(value, (0, 1), 'edge_opacity')
        if vtk_version_info >= (9, 3):
            self.SetEdgeOpacity(value)

    @property
    def show_edges(self) -> bool:  # numpydoc ignore=RT01
        """Return or set the visibility of edges.

        Shows or hides the edges.  Does not apply to a wireframe
        representation.

        Examples
        --------
        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.show_edges = True
        >>> prop.show_edges
        True

        Visualize default edge visibility of ``False``.

        >>> prop.show_edges = False
        >>> prop.plot()

        Visualize edge visibility of ``True``.

        >>> prop.show_edges = True
        >>> prop.plot()

        """
        return bool(self.GetEdgeVisibility())

    @show_edges.setter
    def show_edges(self, value: bool):  # numpydoc ignore=GL08
        self.SetEdgeVisibility(value)

    @property
    def lighting(self) -> bool:  # numpydoc ignore=RT01
        """Return or set view direction lighting.

        Examples
        --------
        Disable lighting.

        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.lighting = False
        >>> prop.lighting
        False

        Visualize it.

        >>> prop.plot()

        """
        return self.GetLighting()

    @lighting.setter
    def lighting(self, value: bool):  # numpydoc ignore=GL08
        if value is None:
            value = self._theme.lighting
        self.SetLighting(value)

    @property
    def ambient(self) -> float:  # numpydoc ignore=RT01
        """Return or set ambient.

        Default :attr:`pyvista.plotting.themes._LightingConfig.ambient`.

        When lighting is enabled, this is the amount of light in the range
        of 0 to 1 that reaches the actor when not directed at the light
        source emitted from the viewer.

        Examples
        --------
        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.ambient = 0.2
        >>> prop.ambient
        0.2

        Visualize default ambient light.

        >>> prop.ambient = 0.0
        >>> prop.plot()

        Visualize ambient at ``0.5``.

        >>> prop.ambient = 0.5
        >>> prop.plot()

        Visualize ambient at ``1.0``.

        >>> prop.ambient = 1.0
        >>> prop.plot()

        """
        return self.GetAmbient()

    @ambient.setter
    def ambient(self, value: float):  # numpydoc ignore=GL08
        _check_range(value, (0, 1), 'ambient')
        self.SetAmbient(value)

    @property
    def diffuse(self) -> float:  # numpydoc ignore=RT01
        """Return or set the diffuse lighting coefficient.

        Default :attr:`pyvista.plotting.themes._LightingConfig.diffuse`.

        This is the scattering of light by reflection or
        transmission. Diffuse reflection results when light strikes an
        irregular surface such as a frosted window or the surface of a
        frosted or coated light bulb. Must be between 0 and 1.

        Examples
        --------
        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.diffuse = 0.2
        >>> prop.diffuse
        0.2

        Visualize default diffuse light.

        >>> prop.diffuse = 1.0
        >>> prop.plot()

        Visualize diffuse at ``0.5``.

        >>> prop.diffuse = 0.5
        >>> prop.plot()

        Visualize diffuse at ``0.0``.

        >>> prop.diffuse = 0.0
        >>> prop.plot()

        """
        return self.GetDiffuse()

    @diffuse.setter
    def diffuse(self, value: float):  # numpydoc ignore=GL08
        _check_range(value, (0, 1), 'diffuse')
        self.SetDiffuse(value)

    @property
    def specular(self) -> float:  # numpydoc ignore=RT01
        """Return or set specular.

        Default :attr:`pyvista.plotting.themes._LightingConfig.specular`.

        Specular lighting simulates the bright spot of a light that appears
        on shiny objects. Must be between 0 and 1.

        Examples
        --------
        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.specular = 0.2
        >>> prop.specular
        0.2

        Visualize default specular light.

        >>> prop.specular = 0.0
        >>> prop.plot()

        Visualize specular at ``0.5``.

        >>> prop.specular = 0.5
        >>> prop.plot()

        Visualize specular at ``1.0``.

        >>> prop.specular = 1.0
        >>> prop.plot()

        """
        return self.GetSpecular()

    @specular.setter
    def specular(self, value: float):  # numpydoc ignore=GL08
        _check_range(value, (0, 1), 'specular')
        self.SetSpecular(value)

    @property
    def specular_power(self) -> float:  # numpydoc ignore=RT01
        """Return or set specular power.

        Default :attr:`pyvista.plotting.themes._LightingConfig.specular_power`.

        The specular power. Must be between 0.0 and 128.0.

        Examples
        --------
        Set specular power to 5.0

        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.specular = 0.1  # enable specular
        >>> prop.specular_power = 5.0
        >>> prop.specular_power
        5.0

        Visualize default specular power.

        >>> prop.specular_power = 1.0
        >>> prop.plot()

        Visualize specular power at ``0.1``.

        >>> prop.specular_power = 0.1
        >>> prop.plot()

        Visualize specular power at ``5.0``.

        >>> prop.specular_power = 5.0
        >>> prop.plot()

        Visualize specular power at ``128.0``.

        >>> prop.specular_power = 128.0
        >>> prop.plot()

        """
        return self.GetSpecularPower()

    @specular_power.setter
    def specular_power(self, value: float):  # numpydoc ignore=GL08
        _check_range(value, (0, 128), 'specular_power')
        self.SetSpecularPower(value)

    @property
    def metallic(self) -> float:  # numpydoc ignore=RT01
        """Return or set metallic.

        Default :attr:`pyvista.plotting.themes._LightingConfig.metallic`.

        This requires that the interpolation be set to ``'Physically based
        rendering'``. Must be between 0 and 1.

        Examples
        --------
        Set metallic to 0.1

        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> # requires physically based rendering
        >>> prop.interpolation = 'pbr'
        >>> prop.metallic = 0.1
        >>> prop.metallic
        0.1

        Visualize default metallic.

        >>> prop.metallic = 0.0
        >>> prop.plot()

        Visualize metallic at ``0.5``.

        >>> prop.metallic = 0.5
        >>> prop.plot()

        Visualize metallic at ``1.0``.

        >>> prop.metallic = 1.0
        >>> prop.plot()

        """
        return self.GetMetallic()

    @metallic.setter
    def metallic(self, value: float):  # numpydoc ignore=GL08
        _check_range(value, (0, 1), 'metallic')
        self.SetMetallic(value)

    @property
    def roughness(self) -> float:  # numpydoc ignore=RT01
        """Return or set roughness.

        Default :attr:`pyvista.plotting.themes._LightingConfig.roughness`.

        This requires that the interpolation be set to ``'Physically based
        rendering'``. Must be between 0 and 1.

        Examples
        --------
        Set roughness to 0.1

        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> # requires physically based rendering
        >>> prop.interpolation = 'pbr'
        >>> prop.metallic = 0.5  # helps to visualize metallic
        >>> prop.roughness = 0.1
        >>> prop.roughness
        0.1

        Visualize default roughness with metallic of ``0.5``.

        >>> prop.roughness = 0.5
        >>> prop.plot()

        Visualize roughness at ``0.0`` with metallic of ``0.5``.

        >>> prop.roughness = 0.0
        >>> prop.plot()

        Visualize roughness at ``1.0`` with metallic of ``0.5``.

        >>> prop.roughness = 1.0
        >>> prop.plot()

        """
        return self.GetRoughness()

    @roughness.setter
    def roughness(self, value: bool):  # numpydoc ignore=GL08
        _check_range(value, (0, 1), 'roughness')
        self.SetRoughness(value)

    @property
    def interpolation(self) -> InterpolationType:  # numpydoc ignore=RT01
        """Return or set the method of shading.

        Defaults to :attr:`pyvista.plotting.themes._LightingConfig.interpolation`.

        One of the following options.

        * ``'Physically based rendering'`` - Physically based rendering.
        * ``'pbr'`` - Alias for Physically based rendering.
        * ``'Phong'`` - Phong shading.
        * ``'Gouraud'`` - Gouraud shading.
        * ``'Flat'`` - Flat Shading.

        This parameter is case insensitive.

        Examples
        --------
        Set interpolation to physically based rendering.

        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.interpolation = 'pbr'
        >>> prop.interpolation
        <InterpolationType.PBR: 3>

        Visualize default flat shading.

        >>> prop.interpolation = 'Flat'
        >>> prop.plot()

        Visualize gouraud shading.

        >>> prop.interpolation = 'Gouraud'
        >>> prop.plot()

        Visualize phong shading.

        >>> prop.interpolation = 'Phong'
        >>> prop.plot()

        Visualize physically based rendering.

        >>> prop.interpolation = 'Physically based rendering'
        >>> prop.plot()

        """
        return InterpolationType.from_any(self.GetInterpolation())

    @interpolation.setter
    def interpolation(self, value: Union[str, int, InterpolationType]):  # numpydoc ignore=GL08
        value = InterpolationType.from_any(value).value
        if value == InterpolationType.PBR:
            self.SetInterpolationToPBR()
        else:
            self.SetInterpolation(value)

    @property
    def render_points_as_spheres(self) -> bool:  # numpydoc ignore=RT01
        """Return or set rendering points as spheres.

        Defaults to :attr:`pyvista.plotting.themes.Theme.render_points_as_spheres`.

        Requires representation style be set to ``'points'``.

        Examples
        --------
        Enable rendering points as spheres.

        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.style = 'points'
        >>> prop.point_size = 20
        >>> prop.render_points_as_spheres = True
        >>> prop.render_points_as_spheres
        True

        Visualize default point rendering.

        >>> prop.render_points_as_spheres = False
        >>> prop.plot()

        Visualize rendering points as spheres.

        >>> prop.render_points_as_spheres = True
        >>> prop.plot()

        """
        return self.GetRenderPointsAsSpheres()

    @render_points_as_spheres.setter
    def render_points_as_spheres(self, value: bool):  # numpydoc ignore=GL08
        self.SetRenderPointsAsSpheres(value)

    @property
    def render_lines_as_tubes(self) -> bool:  # numpydoc ignore=RT01
        """Return or set rendering lines as tubes.

        Defaults to :attr:`pyvista.plotting.themes.Theme.render_lines_as_tubes`.

        Requires representation style be set to ``'wireframe'``.

        Examples
        --------
        Enable rendering lines as tubes.

        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.style = 'wireframe'
        >>> prop.line_width = 10
        >>> prop.render_lines_as_tubes = True
        >>> prop.render_lines_as_tubes
        True

        Visualize default line rendering.

        >>> prop.render_lines_as_tubes = False
        >>> prop.plot()

        Visualize rendering lines as tubes

        >>> prop.render_lines_as_tubes = True
        >>> prop.plot()

        """
        return self.GetRenderLinesAsTubes()

    @render_lines_as_tubes.setter
    def render_lines_as_tubes(self, value: bool):  # numpydoc ignore=GL08
        self.SetRenderLinesAsTubes(value)

    @property
    def line_width(self) -> float:  # numpydoc ignore=RT01
        """Return or set the line width.

        Defaults to :attr:`pyvista.plotting.themes.Theme.line_width`.

        Examples
        --------
        Change the line width to ``10``.

        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.line_width = 10
        >>> prop.line_width
        10.0

        Visualize the default line width.

        >>> prop.line_width = 1.0
        >>> prop.show_edges = True
        >>> prop.plot()

        Visualize with a line width of 5.0

        >>> prop.line_width = 5.0
        >>> prop.plot()

        """
        return self.GetLineWidth()

    @line_width.setter
    def line_width(self, value: float):  # numpydoc ignore=GL08
        self.SetLineWidth(value)

    @property
    def point_size(self):  # numpydoc ignore=RT01
        """Return or set the point size.

        Defaults to :attr:`pyvista.plotting.themes.Theme.point_size`.

        Examples
        --------
        Change the point size to ``10.0``.

        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.point_size = 10
        >>> prop.point_size
        10.0

        Visualize a point size of ``5.0``.

        >>> prop.point_size = 5.0
        >>> prop.style = 'points'
        >>> prop.plot()

        Visualize the a point size of ``10.0``.

        >>> prop.point_size = 10.0
        >>> prop.plot()

        """
        return self.GetPointSize()

    @point_size.setter
    def point_size(self, new_size):  # numpydoc ignore=GL08
        self.SetPointSize(new_size)

    @property
    def culling(self) -> str:  # numpydoc ignore=RT01
        """Return or set face culling.

        Does not render faces that are culled. This can be helpful for dense
        surface meshes, especially when edges are visible, but can cause flat
        meshes to be partially displayed. Defaults to ``'none'``. One of the
        following:

        * ``"back"`` - Enable backface culling
        * ``"front"`` - Enable frontface culling
        * ``'none'`` - Disable both backface and frontface culling

        Examples
        --------
        Enable back face culling

        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.culling = 'back'
        >>> prop.culling
        'back'

        Plot default culling.

        >>> prop.culling = 'none'
        >>> prop.plot()

        Plot backface culling. This looks the same as the default culling
        ``'none'`` because the forward facing faces are already obscuring the
        back faces.

        >>> prop.culling = 'back'
        >>> prop.plot()

        Plot frontface culling. Here, the forward facing faces are hidden
        entirely.

        >>> prop.culling = 'front'
        >>> prop.plot()

        """
        if self.GetBackfaceCulling():
            return 'back'
        elif self.GetFrontfaceCulling():
            return 'front'
        return 'none'

    @culling.setter
    def culling(self, value):  # numpydoc ignore=GL08
        if isinstance(value, str):
            value = value.lower()

        if value == 'back':
            try:
                self.BackfaceCullingOn()
                self.FrontfaceCullingOff()
            except AttributeError:  # pragma: no cover
                pass
        elif value == 'front':
            try:
                self.FrontfaceCullingOn()
                self.BackfaceCullingOff()
            except AttributeError:  # pragma: no cover
                pass
        elif value == 'none':
            self.FrontfaceCullingOff()
            self.BackfaceCullingOff()
        else:
            raise ValueError(
                f'Invalid culling "{value}". Should be either:\n' '"back", "front", or "None"'
            )

    @property
    def ambient_color(self) -> Color:  # numpydoc ignore=RT01
        """Return or set the ambient color of this property.

        Either a string, RGB list, or hex color string.  For example:
        ``color='white'``, ``color='w'``, ``color=[1.0, 1.0, 1.0]``, or
        ``color='#FFFFFF'``. Color will be overridden if scalars are
        specified.

        Examples
        --------
        Set the ambient color to blue.

        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.ambient_color = 'b'
        >>> prop.ambient_color
        Color(name='blue', hex='#0000ffff', opacity=255)

        Visualize setting the ambient color to blue with ``ambient = 0.1``

        >>> prop.ambient = 0.1
        >>> prop.ambient_color = 'b'
        >>> prop.plot()

        """
        return Color(self.GetAmbientColor())

    @ambient_color.setter
    def ambient_color(self, value):  # numpydoc ignore=GL08
        self.SetAmbientColor(Color(value).float_rgb)

    @property
    def specular_color(self) -> Color:  # numpydoc ignore=RT01
        """Return or set the specular color of this property.

        Either a string, RGB list, or hex color string.  For example:
        ``color='white'``, ``color='w'``, ``color=[1.0, 1.0, 1.0]``, or
        ``color='#FFFFFF'``.

        Examples
        --------
        Set the specular color to blue.

        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.specular_color = 'b'
        >>> prop.specular_color
        Color(name='blue', hex='#0000ffff', opacity=255)

        Visualize setting the specular color to blue with ``specular = 0.2``

        >>> prop.specular = 0.2
        >>> prop.specular_color = 'r'
        >>> prop.plot()

        """
        return Color(self.GetSpecularColor())

    @specular_color.setter
    def specular_color(self, value):  # numpydoc ignore=GL08
        self.SetSpecularColor(Color(value).float_rgb)

    @property
    def diffuse_color(self) -> Color:  # numpydoc ignore=RT01
        """Return or set the diffuse color of this property.

        Either a string, RGB list, or hex color string.  For example:
        ``color='white'``, ``color='w'``, ``color=[1.0, 1.0, 1.0]``, or
        ``color='#FFFFFF'``.

        Examples
        --------
        Set the diffuse color to blue.

        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop.diffuse_color = 'b'
        >>> prop.diffuse_color
        Color(name='blue', hex='#0000ffff', opacity=255)

        Visualize setting the diffuse color to white with ``diffuse = 0.5``

        >>> prop.diffuse = 0.5
        >>> prop.diffuse_color = 'w'
        >>> prop.plot()

        """
        return Color(self.GetDiffuseColor())

    @diffuse_color.setter
    def diffuse_color(self, value):  # numpydoc ignore=GL08
        self.SetDiffuseColor(Color(value).float_rgb)

    @property
    def anisotropy(self):  # numpydoc ignore=RT01
        """Return or set the anisotropy coefficient.

        This value controls the anisotropy of the material (0.0 means
        isotropic). This requires physically based rendering.

        For further details see `PBR Journey Part 2 : Anisotropy model with VTK
        <https://www.kitware.com/pbr-journey-part-2-anisotropy-model-with-vtk/>`_

        Notes
        -----
        This attribute requires VTK v9.1.0 or newer.

        Examples
        --------
        Set anisotropy to 0.1

        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> # requires physically based rendering
        >>> prop.interpolation = 'pbr'
        >>> prop.anisotropy
        0.1

        """
        if not hasattr(self, 'GetAnisotropy'):  # pragma: no cover
            from pyvista.core.errors import VTKVersionError

            raise VTKVersionError('Anisotropy requires VTK v9.1.0 or newer.')
        return self.GetAnisotropy()

    @anisotropy.setter
    def anisotropy(self, value: float):  # numpydoc ignore=GL08
        if not hasattr(self, 'SetAnisotropy'):  # pragma: no cover
            from pyvista.core.errors import VTKVersionError

            raise VTKVersionError('Anisotropy requires VTK v9.1.0 or newer.')
        _check_range(value, (0, 1), 'anisotropy')
        self.SetAnisotropy(value)

    def plot(self, **kwargs) -> None:
        """Plot this property on the Stanford Bunny.

        This is useful for visualizing how this property would be applied to an
        actor.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments for :class:`pyvista.Plotter`.

        Examples
        --------
        >>> import pyvista as pv
        >>> prop = pv.Property(
        ...     show_edges=True,
        ...     color='brown',
        ...     edge_color='blue',
        ...     line_width=4,
        ...     specular=1.0,
        ... )
        >>> prop.plot()

        """
        from pyvista import examples  # avoid circular import

        before_close_callback = kwargs.pop('before_close_callback', None)

        pl = pyvista.Plotter(**kwargs)
        actor = pl.add_mesh(examples.download_bunny_coarse())
        actor.SetProperty(self)

        if self.interpolation == 'Physically based rendering':
            cubemap = examples.download_sky_box_cube_map()
            pl.set_environment_texture(cubemap)

        pl.camera_position = 'xy'
        pl.show(before_close_callback=before_close_callback)

    def copy(self) -> 'Property':
        """Create a deep copy of this property.

        Returns
        -------
        pyvista.Property
            Deep copy of this property.

        Examples
        --------
        >>> import pyvista as pv
        >>> prop = pv.Property()
        >>> prop_copy = prop.copy()

        """
        new_prop = Property()
        new_prop.DeepCopy(self)
        return new_prop

    def __repr__(self):
        """Representation of this property."""
        from pyvista.core.errors import VTKVersionError

        props = [
            f'{type(self).__name__} ({hex(id(self))})',
        ]

        for attr in dir(self):
            if not attr.startswith('_') and attr[0].islower():
                name = ' '.join(attr.split('_')).capitalize() + ':'
                try:
                    value = getattr(self, attr)
                except VTKVersionError:  # pragma:no cover
                    continue
                if callable(value):
                    continue
                if isinstance(value, str):
                    value = f'"{value}"'
                props.append(f'  {name:28s} {value}')

        return '\n'.join(props)
