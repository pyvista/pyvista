"""Module managing different plotting theme parameters."""

import os

import pyvista
from pyvista import _vtk
from .colors import string_to_rgb, PARAVIEW_BACKGROUND
from pyvista.jupyter import ALLOWED_BACKENDS

MAX_N_COLOR_BARS = 10
FONT_KEYS = {'arial': _vtk.VTK_ARIAL,
             'courier': _vtk.VTK_COURIER,
             'times': _vtk.VTK_TIMES}


class Theme():
    """PyVista global theme.

    Stores and sets the global theme in ``pyvista``.

    Examples
    --------
    Change the default background color to white.

    >>> import pyvista
    >>> pyvista.theme.color = 'white'

    """

    def __init__(self):
        """Initialize the theme."""
        self._jupyter_backend = None
        self._auto_close = None
        self._background = None
        self._full_screen = None
        self._camera = None
        self._notebook = None
        self._window_size = None
        self._font = None
        self._cmap = None
        self._color = None
        self._nan_color = None
        self._edge_color = None
        self._outline_color = None
        self._floor_color = None
        self._colorbar_orientation = None
        self._colorbar_horizontal = None
        self._colorbar_vertical = None
        self._show_scalar_bar = None
        self._show_edges = None
        self._lighting = None
        self._interactive = None
        self._render_points_as_spheres = None
        self._use_ipyvtk = None
        self._transparent_background = None
        self._title = None
        self._axes = None
        self._multi_samples = None
        self._multi_rendering_splitting_position = None
        self._volume_mapper = None
        self._smooth_shading = None
        self._depth_peeling = None
        self._silhouette = None
        self._slider_style = None
        self.restore_defaults()

    @property
    def background(self):
        """Return or set the default background color of a pyvista plot.

        Examples
        --------
        Set the default global background of all plots to white.

        >>> import pyvista
        >>> pyvista.theme.background = 'white'
        """
        return self._background

    @background.setter
    def background(self, new_background):
        self._background = parse_color(new_background)

    @property
    def jupyter_backend(self):
        """Return or set the jupyter notebook plotting backend.

        Jupyter backend to use when plotting.  Must be one of the
        following:

        * ``'ipyvtklink'`` : Render remotely and stream the
          resulting VTK images back to the client.  Supports all VTK
          methods, but suffers from lag due to remote rendering.
          Requires that a virtual framebuffer be setup when displaying
          on a headless server.  Must have ``ipyvtklink`` installed.

        * ``'panel'`` : Convert the VTK render window to a vtkjs
          object and then visualize that within jupyterlab. Supports
          most VTK objects.  Requires that a virtual framebuffer be
          setup when displaying on a headless server.  Must have
          ``panel`` installed.

        * ``'ipygany'`` : Convert all the meshes into ``ipygany``
          meshes and streams those to be rendered on the client side.
          Supports VTK meshes, but few others.  Aside from ``none``,
          this is the only method that does not require a virtual
          framebuffer.  Must have ``ipygany`` installed.

        * ``'static'`` : Display a single static image within the
          Jupyterlab environment.  Still requires that a virtual
          framebuffer be setup when displaying on a headless server,
          but does not require any additional modules to be installed.

        * ``'none'`` : Do not display any plots within jupyterlab,
          instead display using dedicated VTK render windows.  This
          will generate nothing on headless servers even with a
          virtual framebuffer.

        Examples
        --------
        Enable the ipygany backend.

        >>> import pyvista as pv
        >>> pv.set_jupyter_backend('ipygany')

        Enable the panel backend.

        >>> pv.set_jupyter_backend('panel')

        Enable the ipyvtklink backend.

        >>> pv.set_jupyter_backend('ipyvtklink')

        Just show static images.

        >>> pv.set_jupyter_backend('static')

        Disable all plotting within JupyterLab and display using a
        standard desktop VTK render window.

        >>> pv.set_jupyter_backend(None)  # or 'none'

        """
        return self._jupyter_backend

    @jupyter_backend.setter
    def jupyter_backend(self, value):
        pyvista.set_jupyter_backend(value)

    @property
    def auto_close(self):
        """Automatically close the figures when finished plotting.

        .. DANGER::
           Set to ``False`` with extreme caution.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.theme.auto_close = False

        """
        return self._auto_close

    @auto_close.setter
    def auto_close(self, value):
        self._auto_close = value

    @property
    def full_screen(self):
        """Return if figures are show in full screen.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.theme.full_screen = True
        """
        return self._full_screen

    @full_screen.setter
    def full_screen(self, value):
        self._full_screen = value

    @property
    def camera(self):
        """Return or set the default camera position

        Examples
        --------
        Set both the position and view of the camera.

        >>> import pyvista
        >>> pyvista.theme.camera = {'position': [1, 1, 1],
        ...                         'viewup': [0, 0, 1]}

        Set the default position of the camera

        >>> pyvista.theme.camera['position'] = [1, 1, 1]

        Set the default view of the camera

        >>> pyvista.theme.camera['viewup'] = [0, 0, 1]

        """
        return self._camera

    @camera.setter
    def camera(self, camera):
        if not isinstance(camera, dict):
            raise TypeError(f'Expected ``camera`` to be a dict, not {type(camera)}')

        if 'position' not in camera:
            raise ValueError('Expected the "position" key in the camera dictionary')
        if 'viewup' not in camera:
            raise ValueError('Expected the "viewup" key in the camera dictionary')

        self._camera = camera

    @property
    def notebook(self):
        """Return or set the state of notebook plotting

        Setting this to ``True`` always enables notebook plotting,
        while setting it to ``False`` disables plotting even when
        plotting within a jupyter notebook and plots externally.

        Examples
        --------
        Disable all jupyter notebook plotting

        >>> import pyvista
        >>> pyvista.theme.notebook = False

        """

    @notebook.setter
    def notebook(self, value):
        self._notebook = value

    @property
    def window_size(self):
        """Return or set the default render window size.

        Examples
        --------
        Set window size to ``[400, 400]``

        >>> import pyvista
        >>> pyvista.theme.window_size = [400, 400]

        """
        return self._window_size

    @window_size.setter
    def window_size(self, window_size):
        if not len(window_size) == 2:
            raise ValueError('Expected a length 2 iterable for ``window_size``')

        # ensure positve size
        if window_size[0] < 0 or window_size[1] < 0:
            raise ValueError('Window size must be a positive value')

        self._window_size = window_size

    @property
    def font(self):
        """Return or set the default font size, family, and/or color.

        Examples
        --------
        Set the default font family to 'arial'.  Must be either
        'arial', 'courier', or 'times'.

        >>> import pyvista
        >>> pyvista.theme.font['family'] = 'arial'

        Set the default font size to 20.

        >>> pyvista.theme.font['size'] = 20

        Set the default title size to 40

        >>> pyvista.theme.font['title_size'] = 40

        Set the default label size to 10

        >>> pyvista.theme.font['label_size'] = 10

        Set the default text color to 'grey'

        >>> pyvista.theme.font['color'] = 'grey'

        String formatter used to format numerical data to '%.6e'

        >>> pyvista.theme.font['color'] = '%.6e'

        """
        return self._font

    @font.setter
    def font(self, font):
        self._font = font

    @property
    def cmap(self):
        """Return or set the default global colormap of pyvista.

        See available Matplotlib colormaps.  Only applicable for when
        displaying ``scalars``. Requires Matplotlib to be installed.
        ``colormap`` is also an accepted alias for this. If
        ``colorcet`` or ``cmocean`` are installed, their colormaps can
        be specified by name.

        You can also specify a list of colors to override an existing
        colormap with a custom one.  For example, to create a three
        color colormap you might specify ``['green', 'red', 'blue']``

        Examples
        --------
        Set the default global colormap to 'jet'

        >>> import pyvista
        >>> pyvista.theme.cmap = 'jet'

        """
        return self._cmap

    @cmap.setter
    def cmap(self, cmap):
        self._cmap = cmap

    @property
    def color(self):
        """Return or set the default color of meshes in pyvista.

        Used for meshes without ``scalars``.

        A string or 3 item list, optional, defaults to white
        Either a string, rgb list, or hex color string.  For example:

        * ``color='white'``
        * ``color='w'``
        * ``color=[1, 1, 1]``
        * ``color='#FFFFFF'``

        Examples
        --------
        Set the default mesh color to 'red'

        >>> import pyvista
        >>> pyvista.theme.color = 'red'

        """
        return self._color

    @color.setter
    def color(self, color):
        self._color = parse_color(color)

    @property
    def nan_color(self):
        """Return or set the default global NAN color.

        This color is used to plot all NAN values.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.theme.nan_color = 'darkgray'
        """
        return self._nan_color

    @nan_color.setter
    def nan_color(self, nan_color):
        self._nan_color = parse_color(nan_color)

    @property
    def edge_color(self):
        """Return or set the default global edge color.

        Examples
        --------
        Set the global edge color to 'blue'

        >>> import pyvista
        >>> pyvista.theme.edge_color = 'blue'
        """
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color):
        self._edge_color = parse_color(edge_color)

    @property
    def outline_color(self):
        """Return or set the default outline color.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.theme.outline_color = 'white'
        """
        return self._outline_color

    @outline_color.setter
    def outline_color(self, outline_color):
        self._outline_color = parse_color(outline_color)

    @property
    def floor_color(self):
        """Return or set the default floor color.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.theme.floor_color = 'black'
        """
        return self._floor_color

    @floor_color.setter
    def floor_color(self, floor_color):
        self._floor_color = floor_color

    @property
    def colorbar_orientation(self):
        """Return or set the default global colorbar orientation

        Must be either ``'vertical'`` or ``'horizontal'``.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.theme.colorbar_orientation = 'horizontal'
        """
        return self._colorbar_orientation

    @colorbar_orientation.setter
    def colorbar_orientation(self, colorbar_orientation):
        if colorbar_orientation not in ['vertical', 'horizontal']:
            raise ValueError('Colorbar orientation must be either "vertical" or '
                             '"horizontal"')
        self._colorbar_orientation = colorbar_orientation

    @property
    def colorbar_horizontal(self):
        """Default orientation when the colorbar is set to 'horizontal'

        Examples
        --------
        Set the default colorbar width to 0.6

        >>> import pyvista
        >>> pyvista.theme.colorbar_horizontal['width'] = 0.6

        Set all the parameters of the colorbar

        >>> colorbar_parm = {
        ... 'width': 0.6,
        ... 'height': 0.08,
        ... 'position_x': 0.35,
        ... 'position_y': 0.05}
        >>> pyvista.theme.colorbar_horizontal = colorbar_parm

        """
        return self._colorbar_horizontal

    @colorbar_horizontal.setter
    def colorbar_horizontal(self, colorbar_horizontal):
        for key in colorbar_horizontal:
            if key not in self._colorbar_horizontal:
                raise KeyError(f'Invalid key {key} for colorbar_horizontal')
        self._colorbar_horizontal = colorbar_horizontal

    @property
    def colorbar_vertical(self):
        """Default orientation when the colorbar is set to 'vertical'

        Examples
        --------
        Set the default colorbar width to 0.45

        >>> import pyvista
        >>> pyvista.theme.colorbar_vertical['width'] = 0.45

        Set all the parameters of the colorbar

        >>> colorbar_parm = {
        ... 'width': 0.08,
        ... 'height': 0.45,
        ... 'position_x': 0.9,
        ... 'position_y': 0.02}
        >>> pyvista.theme.colorbar_vertical = colorbar_parm

        """
        return self._colorbar_vertical

    @colorbar_vertical.setter
    def colorbar_vertical(self, colorbar_vertical):
        for key in colorbar_vertical:
            if key not in self._colorbar_vertical:
                raise KeyError(f'Invalid key {key} for colorbar_vertical')
        self._colorbar_vertical = colorbar_vertical

    @property
    def show_scalar_bar(self):
        """Return or set the default color bar visibility.

        Examples
        --------
        Show the scalar bar by default when scalars are available.

        >>> import pyvista
        >>> pyvista.theme.show_scalar_bar = True

        """
        return self._show_scalar_bar

    @show_scalar_bar.setter
    def show_scalar_bar(self, show_scalar_bar):
        self._show_scalar_bar = show_scalar_bar

    @property
    def show_edges(self):
        """Return or set the global default edge visibility.

        Examples
        --------
        Show edges globally by default.

        >>> import pyvista
        >>> pyvista.theme.show_edges = True

        """
        return self._show_edges

    @show_edges.setter
    def show_edges(self, show_edges):
        self._show_edges = show_edges

    @property
    def lighting(self):
        """Return or set the default global ``lighting``.

        Examples
        --------
        Disable lighting globally

        >>> import pyvista
        >>> pyvista.theme.lighting = False
        """
        return self._lighting

    @lighting.setter
    def lighting(self, lighting):
        self._lighting = lighting

    @property
    def interactive(self):
        return self._interactive

    @interactive.setter
    def interactive(self, interactive):
        self._interactive = interactive

    @property
    def render_points_as_spheres(self):
        return self._render_points_as_spheres

    @render_points_as_spheres.setter
    def render_points_as_spheres(self, render_points_as_spheres):
        self._render_points_as_spheres = render_points_as_spheres

    @property
    def use_ipyvtk(self):
        from pyvista.core.errors import DeprecationError
        raise DeprecationError('DEPRECATED: Please use ``jupyter_backend``')

    @use_ipyvtk.setter
    def use_ipyvtk(self, use_ipyvtk):
        from pyvista.core.errors import DeprecationError
        raise DeprecationError('DEPRECATED: Please use ``jupyter_backend``')

    @property
    def transparent_background(self):
        return self._transparent_background

    @transparent_background.setter
    def transparent_background(self, transparent_background):
        self._transparent_background = transparent_background

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title):
        self._title = title

    @property
    def multi_samples(self):
        return self._multi_samples

    @multi_samples.setter
    def multi_samples(self, multi_samples):
        self._multi_samples = multi_samples

    @property
    def multi_rendering_splitting_position(self):
        return self._multi_rendering_splitting_position

    @multi_rendering_splitting_position.setter
    def multi_rendering_splitting_position(self, multi_rendering_splitting_position):
        self._multi_rendering_splitting_position = multi_rendering_splitting_position

    @property
    def volume_mapper(self):
        return self._volume_mapper

    @volume_mapper.setter
    def volume_mapper(self, volume_mapper):
        self._volume_mapper = volume_mapper

    @property
    def smooth_shading(self):
        return self._smooth_shading

    @smooth_shading.setter
    def smooth_shading(self, smooth_shading):
        self._smooth_shading = smooth_shading

    @property
    def depth_peeling(self):
        return self._depth_peeling

    @depth_peeling.setter
    def depth_peeling(self, depth_peeling):
        self._depth_peeling = depth_peeling

    @property
    def silhouette(self):
        return self._silhouette

    @silhouette.setter
    def silhouette(self, silhouette):
        self._silhouette = silhouette

    @property
    def slider_style(self):
        return self._slider_style

    @slider_style.setter
    def slider_style(self, slider_style):
        self._slider_style = slider_style

    @property
    def axes(self):
        return self._axes

    @axes.setter
    def axes(self, axes):
        self._axes = axes

    def restore_defaults(self):
        """Restore the theme defaults."""
        self._jupyter_backend = 'ipyvtklink'
        self._auto_close = True  # DANGER: set to False with extreme caution
        self._background = [0.3, 0.3, 0.3]
        self._full_screen = False
        self._camera = {
            'position': [1, 1, 1],
            'viewup': [0, 0, 1],
        }

        self._notebook = None
        self._window_size = [1024, 768]
        self._font = {
            'family': 'arial',
            'size': 12,
            'title_size': None,
            'label_size': None,
            'color': [1, 1, 1],
            'fmt': None,
        }

        self._cmap = 'viridis'
        self._color = 'white'
        self._nan_color = 'darkgray'
        self._edge_color = 'black'
        self._outline_color = 'white'
        self._floor_color = 'gray'
        self._colorbar_orientation = 'horizontal'
        self._colorbar_horizontal = {
            'width': 0.6,
            'height': 0.08,
            'position_x': 0.35,
            'position_y': 0.05,
        }
        self._colorbar_vertical = {
            'width': 0.08,
            'height': 0.45,
            'position_x': 0.9,
            'position_y': 0.02,
        }
        self._show_scalar_bar = True
        self._show_edges = False
        self._lighting = True
        self._interactive = False
        self._render_points_as_spheres = False
        self._use_ipyvtk = False
        self._transparent_background = False
        self._title = 'PyVista'
        self._axes = {
            'x_color': 'tomato',
            'y_color': 'seagreen',
            'z_color': 'mediumblue',
            'box': False,
            'show': True,
        }
        self._multi_samples = 4
        self._multi_rendering_splitting_position = None
        self._volume_mapper = 'fixed_point' if os.name == 'nt' else 'smart'
        self._smooth_shading = False
        self._depth_peeling = {
            'number_of_peels': 4,
            'occlusion_ratio': 0.0,
            'enabled': False,
        }
        self._silhouette = {
            'color': 'black',
            'line_width': 2,
            'opacity': 1.0,
            'feature_angle': False,
            'decimate': 0.9,
        },
        self._slider_style = {
            'classic': {
                'slider_length': 0.02,
                'slider_width': 0.04,
                'slider_color': (0.5, 0.5, 0.5),
                'tube_width': 0.005,
                'tube_color': (1, 1, 1),
                'cap_opacity': 1,
                'cap_length': 0.01,
                'cap_width': 0.02,
            },
            'modern': {
                'slider_length': 0.02,
                'slider_width': 0.04,
                'slider_color': (0.43137255, 0.44313725, 0.45882353),
                'tube_width': 0.04,
                'tube_color': (0.69803922, 0.70196078, 0.70980392),
                'cap_opacity': 0,
                'cap_length': 0.01,
                'cap_width': 0.02,
            },
        },

    def __repr__(self):
        """User friendly representation of the pyvista theme."""
        txt = ['PyVista Theme']
        txt.append(f'Background Color : {self.background}')

    def set_to_paraview(self):
        """Set the theme to a paraview-like theme.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.theme.set_to_paraview()

        """
        self.restore_defaults()
        self._background = PARAVIEW_BACKGROUND
        self._cmap = 'coolwarm'
        self._font['family'] = 'arial'
        self._font['label_size'] = 16
        self._font['color'] = 'white'
        self._show_edges = False
        self._color = 'white'
        self._outline_color = 'white'
        self._edge_color = 'black'
        self._axes['x_color'] = 'tomato'
        self._axes['y_color'] = 'gold'
        self._axes['z_color'] = 'green'

    def set_to_document(self):
        """Set the global theme to the document theme.

        This theme uses a white background, the "viridis" colormap,
        disables edges and black fonts.  Best used for presentations,
        papers, etc.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.theme.set_to_document()

        """
        self.restore_defaults()
        self._background = 'white'
        self._cmap = 'viridis'
        self._font['size'] = 18
        self._font['title_size'] = 18
        self._font['label_size'] = 18
        self._font['color'] = 'black'
        self._show_edges = False
        self._color = 'tan'
        self._outline_color = 'black'
        self._edge_color = 'black'
        self._axes['x_color'] = 'tomato'
        self._axes['y_color'] = 'seagreen'
        self._axes['z_color'] = 'blue'

    def _set_to_testing(self):
        """Low resolution testing theme.

        Necessary for image regression.  Xvfb doesn't support
        multi-sampling, so we disable it here for consistency between
        desktops and remote testing.
        """
        self.restore_defaults()
        self._off_screen = True
        self._multi_samples = 1
        self._window_size = [400, 400]

    def set_to_dark(self):
        """Set the global theme to dark mode.

        Black background, "viridis" colormap, tan meshes, white (hidden) edges.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.theme.set_to_dark()

        """
        self.restore_defaults()
        self._background = 'black'
        self._cmap = 'viridis'
        self._font['color'] = 'white'
        self._show_edges = False
        self._color = 'tan'
        self._outline_color = 'white'
        self._edge_color = 'white'
        self._axes['x_color'] = 'tomato'
        self._axes['y_color'] = 'seagreen'
        self._axes['z_color'] = 'blue'


def parse_color(color, opacity=None):
    """Parse color into a vtk friendly rgb list.

    Values returned will be between 0 and 1.

    """
    if color is None:
        color = pyvista.theme.color
    if isinstance(color, str):
        color = string_to_rgb(color)
    elif len(color) == 3:
        pass
    elif len(color) == 4:
        color = color[:3]
    else:
        raise ValueError(f"""
    Invalid color input: ({color})
    Must be string, rgb list, or hex color string.  For example:
        color='white'
        color='w'
        color=[1, 1, 1]
        color='#FFFFFF'""")
    if opacity is not None and isinstance(opacity, (float, int)):
        color = [color[0], color[1], color[2], opacity]
    return color


def parse_font_family(font_family):
    """Check font name."""
    # check font name
    font_family = font_family.lower()
    if font_family not in ['courier', 'times', 'arial']:
        raise ValueError('Font must be either "courier", "times" '
                         'or "arial"')

    return FONT_KEYS[font_family]
