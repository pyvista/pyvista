"""Module managing different plotting theme parameters."""

import warnings
from enum import Enum
import os

from .plotting.colors import PARAVIEW_BACKGROUND
from .plotting.tools import parse_color
from .utilities.misc import PyvistaDeprecationWarning


def set_plot_theme(theme):
    """Set the plotting parameters to a predefined theme."""
    import pyvista
    if isinstance(theme, str):
        if theme == 'night':
            warnings.warn('use "dark" instead of "night" theme', PyvistaDeprecationWarning)
        new_theme = ALLOWED_THEMES[theme].value()
        pyvista.global_theme.load_theme(new_theme)
    elif isinstance(theme, DefaultTheme):
        pyvista.global_theme.load_theme(theme)
    else:
        raise TypeError(f'Expected a pyvista.Theme or str, not '
                        f'a {type(theme)}')


class DefaultTheme():
    """PyVista default theme.

    Examples
    --------
    Change the global default background color to white.

    >>> import pyvista
    >>> pyvista.global_theme.color = 'white'

    Show edges by default.

    >>> pyvista.global_theme.show_edges = True

    Create a new theme from the DefaultTheme and apply it globally.

    >>> my_theme = pyvista.themes.DefaultTheme()
    >>> my_theme.color = 'red'
    >>> my_theme.background_color = 'white'
    >>> pyvista.global_theme.load_theme(my_theme)

    """

    def __init__(self):
        """Initialize the theme."""
        self._name = 'default'
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

        # Grab system flag for anti-aliasing
        try:
            self._multi_samples = int(os.environ.get('PYVISTA_MULTI_SAMPLES', 4))
        except ValueError:  # pragma: no cover
            self._multi_samples = 4

        # Grab system flag for auto-closing
        self._auto_close = os.environ.get('PYVISTA_AUTO_CLOSE', '').lower() != 'false'

        self._jupyter_backend = os.environ.get('PYVISTA_JUPYTER_BACKEND', 'ipyvtklink')

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
        }
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
        }

    @property
    def background(self):
        """Return or set the default background color of a pyvista plot.

        Examples
        --------
        Set the default global background of all plots to white.

        >>> import pyvista
        >>> pyvista.global_theme.background = 'white'
        """
        return self._background

    def __eq__(self, other_theme):
        if not isinstance(other_theme, DefaultTheme):
            return False

        for name, value in vars(other_theme).items():
            if not getattr(self, name) == value:
                return False

        return True

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
        import pyvista
        pyvista.set_jupyter_backend(value)

    @property
    def auto_close(self):
        """Automatically close the figures when finished plotting.

        .. DANGER::
           Set to ``False`` with extreme caution.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.auto_close = False

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
        >>> pyvista.global_theme.full_screen = True
        """
        return self._full_screen

    @full_screen.setter
    def full_screen(self, value):
        self._full_screen = value

    @property
    def camera(self):
        """Return or set the default camera position.

        Examples
        --------
        Set both the position and view of the camera.

        >>> import pyvista
        >>> pyvista.global_theme.camera = {'position': [1, 1, 1],
        ...                            'viewup': [0, 0, 1]}

        Set the default position of the camera

        >>> pyvista.global_theme.camera['position'] = [1, 1, 1]

        Set the default view of the camera

        >>> pyvista.global_theme.camera['viewup'] = [0, 0, 1]

        """
        return self._camera

    @camera.setter
    def camera(self, camera):
        if not isinstance(camera, dict):
            raise TypeError(f'Expected ``camera`` to be a dict, not {type(camera)}')

        if 'position' not in camera:
            raise KeyError('Expected the "position" key in the camera dict')
        if 'viewup' not in camera:
            raise KeyError('Expected the "viewup" key in the camera dict')

        self._camera = camera

    @property
    def notebook(self):
        """Return or set the state of notebook plotting.

        Setting this to ``True`` always enables notebook plotting,
        while setting it to ``False`` disables plotting even when
        plotting within a jupyter notebook and plots externally.

        Examples
        --------
        Disable all jupyter notebook plotting

        >>> import pyvista
        >>> pyvista.global_theme.notebook = False

        """
        return self._notebook

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
        >>> pyvista.global_theme.window_size = [400, 400]

        """
        return self._window_size

    @window_size.setter
    def window_size(self, window_size):
        if not len(window_size) == 2:
            raise ValueError('Expected a length 2 iterable for ``window_size``')

        # ensure positive size
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
        >>> pyvista.global_theme.font['family'] = 'arial'

        Set the default font size to 20.

        >>> pyvista.global_theme.font['size'] = 20

        Set the default title size to 40

        >>> pyvista.global_theme.font['title_size'] = 40

        Set the default label size to 10

        >>> pyvista.global_theme.font['label_size'] = 10

        Set the default text color to 'grey'

        >>> pyvista.global_theme.font['color'] = 'grey'

        String formatter used to format numerical data to '%.6e'

        >>> pyvista.global_theme.font['color'] = '%.6e'

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
        >>> pyvista.global_theme.cmap = 'jet'

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
        >>> pyvista.global_theme.color = 'red'

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
        >>> pyvista.global_theme.nan_color = 'darkgray'
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
        >>> pyvista.global_theme.edge_color = 'blue'
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
        >>> pyvista.global_theme.outline_color = 'white'
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
        >>> pyvista.global_theme.floor_color = 'black'
        """
        return self._floor_color

    @floor_color.setter
    def floor_color(self, floor_color):
        self._floor_color = floor_color

    @property
    def colorbar_orientation(self):
        """Return or set the default global colorbar orientation.

        Must be either ``'vertical'`` or ``'horizontal'``.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.colorbar_orientation = 'horizontal'
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
        """Return or set the default parameters of a horizontal colorbar.

        Examples
        --------
        Set the default colorbar width to 0.6

        >>> import pyvista
        >>> pyvista.global_theme.colorbar_horizontal['width'] = 0.6

        Set all the parameters of the colorbar

        >>> colorbar_parm = {
        ... 'width': 0.6,
        ... 'height': 0.08,
        ... 'position_x': 0.35,
        ... 'position_y': 0.05}
        >>> pyvista.global_theme.colorbar_horizontal = colorbar_parm

        """
        return self._colorbar_horizontal

    @colorbar_horizontal.setter
    def colorbar_horizontal(self, colorbar_horizontal):
        for key, value in colorbar_horizontal.items():
            if key not in self._colorbar_horizontal:
                raise KeyError(f'Invalid key {key} for colorbar_horizontal'
                               f'Permitted keys are: {", ".join(self._colorbar_horizontal)}')
            self._colorbar_horizontal[key] = value

    @property
    def colorbar_vertical(self):
        """Return or set the default parameters of a vertical colorbar.

        Examples
        --------
        Set the default colorbar width to 0.45

        >>> import pyvista
        >>> pyvista.global_theme.colorbar_vertical['width'] = 0.45

        Set all the parameters of the colorbar

        >>> colorbar_parm = {
        ... 'width': 0.08,
        ... 'height': 0.45,
        ... 'position_x': 0.9,
        ... 'position_y': 0.02}
        >>> pyvista.global_theme.colorbar_vertical = colorbar_parm

        """
        return self._colorbar_vertical

    @colorbar_vertical.setter
    def colorbar_vertical(self, colorbar_vertical):
        for key, value in colorbar_vertical.items():
            if key not in self._colorbar_vertical:
                raise KeyError(f'Invalid key {key} for colorbar_vertical'
                               f'Permitted keys are: {", ".join(self._colorbar_vertical)}')
            self._colorbar_vertical[key] = value

    @property
    def show_scalar_bar(self):
        """Return or set the default color bar visibility.

        Examples
        --------
        Show the scalar bar by default when scalars are available.

        >>> import pyvista
        >>> pyvista.global_theme.show_scalar_bar = True

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
        >>> pyvista.global_theme.show_edges = True

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
        >>> pyvista.global_theme.lighting = False
        """
        return self._lighting

    @lighting.setter
    def lighting(self, lighting):
        self._lighting = lighting

    @property
    def interactive(self):
        """Return or set the default global ``interactive`` parameter.

        Examples
        --------
        Make all plots non-interactive globally.

        >>> import pyvista
        >>> pyvista.global_theme.interactive = False
        """
        return self._interactive

    @interactive.setter
    def interactive(self, interactive):
        self._interactive = interactive

    @property
    def render_points_as_spheres(self):
        """Return or set the default global ``render_points_as_spheres`` parameter.

        Examples
        --------
        Render points as spheres by default globally to ``True``.

        >>> import pyvista
        >>> pyvista.global_theme.render_points_as_spheres = True
        """
        return self._render_points_as_spheres

    @render_points_as_spheres.setter
    def render_points_as_spheres(self, render_points_as_spheres):
        self._render_points_as_spheres = render_points_as_spheres

    @property
    def use_ipyvtk(self):
        """Return or set the default global ``use_ipyvtk`` parameter.

        This parameter has been deprecated in favor of
        ``jupyter_backend``.
        """
        from pyvista.core.errors import DeprecationError
        raise DeprecationError('DEPRECATED: Please use ``jupyter_backend``')

    @property
    def transparent_background(self):
        """Return or set the default global ``transparent_background`` parameter.

        Examples
        --------
        Set transparent_background globally to ``True``

        >>> import pyvista
        >>> pyvista.global_theme.transparent_background = True
        """
        return self._transparent_background

    @transparent_background.setter
    def transparent_background(self, transparent_background):
        self._transparent_background = transparent_background

    @property
    def title(self):
        """Return or set the default global ``title`` parameter.

        This is the VTK render window title.

        Examples
        --------
        Set title globally to 'plot'

        >>> import pyvista
        >>> pyvista.global_theme.title = 'plot'
        """
        return self._title

    @title.setter
    def title(self, title):
        self._title = title

    @property
    def multi_samples(self):
        """Return or set the default global ``multi_samples`` parameter.

        Set the number of multisamples to enable hardware antialiasing.

        Examples
        --------
        Set the default number of multisamples to 2.

        >>> import pyvista
        >>> pyvista.global_theme.multi_samples = 2
        """
        return self._multi_samples

    @multi_samples.setter
    def multi_samples(self, multi_samples):
        self._multi_samples = multi_samples

    @property
    def multi_rendering_splitting_position(self):
        """Return or set the default global ``multi_rendering_splitting_position`` parameter.

        Examples
        --------
        Set multi_rendering_splitting_position globally to 0.5 (the
        middle of the window).

        >>> import pyvista
        >>> pyvista.global_theme.multi_rendering_splitting_position = 0.5
        """
        return self._multi_rendering_splitting_position

    @multi_rendering_splitting_position.setter
    def multi_rendering_splitting_position(self, multi_rendering_splitting_position):
        self._multi_rendering_splitting_position = multi_rendering_splitting_position

    @property
    def volume_mapper(self):
        """Return or set the default global ``volume_mapper`` parameter.

        Must be one of the following strings, which are mapped to the
        following VTK volume mappers.

        ``'fixed_point'`` : ``vtk.vtkFixedPointVolumeRayCastMapper``
        ``'gpu'`` : ``vtk.vtkGPUVolumeRayCastMapper``
        ``'open_gl'`` : ``vtk.vtkOpenGLGPUVolumeRayCastMapper``
        ``'smart'`` : ``vtk.vtkSmartVolumeMapper``

        Examples
        --------
        Set default volume mapper globally to 'gpu'.

        >>> import pyvista
        >>> pyvista.global_theme.volume_mapper = 'gpu'
        """
        return self._volume_mapper

    @volume_mapper.setter
    def volume_mapper(self, mapper):
        mappers = ['fixed_point', 'gpu', 'open_gl', 'smart']
        if mapper not in mappers:
            raise TypeError(f"Mapper ({mapper}) unknown. Available volume mappers "
                            f"include:\n {', '.join(mappers)}")

        self._volume_mapper = mapper

    @property
    def smooth_shading(self):
        """Return or set the global default ``smooth_shading`` parameter.

        Examples
        --------
        Set the global smooth_shading parameter default to ``True``.

        >>> import pyvista
        >>> pyvista.global_theme.smooth_shading = True
        """
        return self._smooth_shading

    @smooth_shading.setter
    def smooth_shading(self, smooth_shading):
        self._smooth_shading = smooth_shading

    @property
    def depth_peeling(self):
        """Return or set the global default ``depth_peeling`` parameter.

        self._depth_peeling = {
            'number_of_peels': 4,
            'occlusion_ratio': 0.0,
            'enabled': False,
        }

        Examples
        --------
        Set the global depth_peeling parameter default to be enabled
        with 8 peels.

        >>> import pyvista
        >>> pyvista.global_theme.depth_peeling = {
        ...     'number_of_peels': 8,
        ...     'occlusion_ratio': 0.0,
        ...     'enabled': False}
        """
        return self._depth_peeling

    @depth_peeling.setter
    def depth_peeling(self, depth_peeling):
        for key, value in depth_peeling.items():
            if key not in self._depth_peeling:
                raise KeyError(f'Invalid key ``{key}`` for depth_peeling.\n'
                               f'Permitted keys are: {", ".join(self._depth_peeling)}')
            self._depth_peeling[key] = value

    @property
    def silhouette(self):
        """Return or set the global default ``silhouette`` parameter.

        Examples
        --------
        Set the silhouette parameter dictionary

        >>> import pyvista
        >>> pyvista.global_theme.silhouette = {
        ...    'color': 'black',
        ...    'line_width': 2,
        ...    'opacity': 1.0,
        ...    'feature_angle': False,
        ...    'decimate': 0.9}

        Set a single value of the silhouette.

        >>> pyvista.global_theme.silhouette['opacity'] = 0.5

        """
        return self._silhouette

    @silhouette.setter
    def silhouette(self, silhouette):
        for key, value in silhouette.items():
            if key not in self._silhouette:
                raise KeyError(f'Invalid key ``{key}`` for silhouette.\n'
                               f'Permitted keys are: {", ".join(self._silhouette)}')
            self._silhouette[key] = value

    @property
    def slider_style(self):
        """Return or set the global default ``slider_style`` parameter.

        Examples
        --------
        Set the ``slider_style`` dictionary.

        >>> import pyvista
        >>> pyvista.global_theme.slider_style = {
        ...     'classic': {
        ...         'slider_length': 0.02,
        ...         'slider_width': 0.04,
        ...         'slider_color': (0.5, 0.5, 0.5),
        ...         'tube_width': 0.005,
        ...         'tube_color': (1, 1, 1),
        ...         'cap_opacity': 1,
        ...         'cap_length': 0.01,
        ...         'cap_width': 0.02,
        ...     },
        ...     'modern': {
        ...         'slider_length': 0.02,
        ...         'slider_width': 0.04,
        ...         'slider_color': (0.43137255, 0.44313725, 0.45882353),
        ...         'tube_width': 0.04,
        ...         'tube_color': (0.69803922, 0.70196078, 0.70980392),
        ...         'cap_opacity': 0,
        ...         'cap_length': 0.01,
        ...         'cap_width': 0.02,
        ...     },
        ... }

        Set a single slider style parameter

        >>> pyvista.global_theme.slider_style['classic']['slider_length'] = 0.05

        """
        return self._slider_style

    @slider_style.setter
    def slider_style(self, slider_style):
        self._slider_style = slider_style

    @property
    def axes(self):
        """Return or set the global default ``axes`` parameter.

        Examples
        --------
        Set the axes dictionary.

        >>> import pyvista
        >>> pyvista.global_theme.axes = {
        ...     'x_color': 'tomato',
        ...     'y_color': 'seagreen',
        ...     'z_color': 'mediumblue',
        ...     'box': False,
        ...     'show': True,
        ... }

        Set a single axes theme value

        >>> pyvista.global_theme.axes['x_color'] = 'black'

        """
        return self._axes

    @axes.setter
    def axes(self, axes):
        self._axes = axes

    def restore_defaults(self):
        """Restore the theme defaults.

        Examples
        --------
        >>> import pyvista
        >>> pyvista.global_theme.restore_defaults()

        """
        self.__init__()

    def __repr__(self):
        """User friendly representation of the current theme."""
        txt = [f'{self.name.capitalize()} Theme']
        parm = {
            'Background': 'background',
            'Jupyter backend': 'jupyter_backend',
            'Full screen': 'full_screen',
            'Window size': 'window_size',
            'Camera': 'camera',
            'Notebook': 'notebook',
            'Font': 'font',
            'Auto close': 'auto_close',
            'Colormap': 'cmap',
            'Color': 'color',
            'NAN color': 'nan_color',
            'Edge color': 'edge_color',
            'Outline color': 'outline_color',
            'Floor color': 'floor_color',
            'Colorbar orientation': 'colorbar_orientation',
            'Colorbar - horizontal': 'colorbar_horizontal',
            'Colorbar - vertical': 'colorbar_vertical',
            'Show scalar bar': 'show_scalar_bar',
            'Show edges': 'show_edges',
            'Lighting': 'lighting',
            'Interactive': 'interactive',
            'Render points as spheres': 'render_points_as_spheres',
            'Transparent Background': 'transparent_background',
            'Title': 'title',
            'Axes': 'axes',
            'Multi-samples': 'multi_samples',
            'Multi-renderer Split Pos': 'multi_rendering_splitting_position',
            'Volume mapper': 'volume_mapper',
            'Smooth shading': 'smooth_shading',
            'Depth peeling': 'depth_peeling',
            'Silhouette': 'silhouette',
        }
        for name, attr in parm.items():
            setting = getattr(self, attr)
            if isinstance(setting, dict):
                txt.append(f'{name:<25}')
                for key, item in setting.items():
                    txt.append(f'    {key:<21}: {item}')
            else:
                txt.append(f'{name:<25}: {setting}')

        max_len = max([len(entry) for entry in txt])
        txt.insert(1, '-'*max_len)

        return '\n'.join(txt)

    @property
    def name(self):
        """Return or set the name of the theme."""
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def load_theme(self, theme):
        """Overwrite the current them with a theme.

        Examples
        --------
        Create a custom theme from the default theme and load it into
        pyvista.

        >>> import pyvista
        >>> from pyvista.themes import DefaultTheme
        >>> my_theme = DefaultTheme()
        >>> my_theme.font['size'] = 20
        >>> my_theme.font['title_size'] = 40
        >>> my_theme.cmap = 'jet'
        ...
        >>> pyvista.global_theme.load_theme(my_theme)
        >>> pyvista.global_theme.font['size']
        20

        Create a custom theme from the dark theme and load it into
        pyvista.

        >>> from pyvista.themes import DarkTheme
        >>> my_theme = DarkTheme()
        >>> my_theme.show_edges = True
        >>> pyvista.global_theme.load_theme(my_theme)
        >>> pyvista.global_theme.show_edges
        True

        """
        if not isinstance(theme, DefaultTheme):
            raise TypeError('``theme`` must be a pyvista theme like '
                            '``pyvista.themes.DefaultTheme``')

        for name, value in vars(theme).items():
            setattr(self, name, value)


class DarkTheme(DefaultTheme):
    """Dark mode theme.

    Black background, "viridis" colormap, tan meshes, white (hidden) edges.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import themes
    >>> pyvista.set_plot_theme(themes.DarkTheme())

    Alternatively, set via a string.

    >>> pyvista.set_plot_theme('dark')

    """

    def __init__(self):
        """Initialize the theme."""
        super().__init__()
        self._name = 'dark'
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


class ParaViewTheme(DefaultTheme):
    """Set the theme to a paraview-like theme.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import themes
    >>> pyvista.set_plot_theme(themes.ParaViewTheme())

    Alternatively, set via a string.

    >>> pyvista.set_plot_theme('paraview')

    """

    def __init__(self):
        """Initialize theme."""
        super().__init__()
        self._name = 'paraview'
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


class DocumentTheme(DefaultTheme):
    """Set the global theme to the document theme.

    This theme uses a white background, the "viridis" colormap,
    disables edges and black fonts.  Best used for presentations,
    papers, etc.

    Examples
    --------
    >>> import pyvista
    >>> from pyvista import themes
    >>> pyvista.set_plot_theme(themes.DocumentTheme())

    Alternatively, set via a string.

    >>> pyvista.set_plot_theme('document')

    """

    def __init__(self):
        """Initialize the theme."""
        super().__init__()
        self._name = 'document'
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


class _TestingTheme(DefaultTheme):
    """Low resolution testing theme for ``pytest``.

    Necessary for image regression.  Xvfb doesn't support
    multi-sampling, so we disable it here for consistency between
    desktops and remote testing.
    """

    def __init__(self):
        super().__init__()
        self._name = 'testing'
        self._multi_samples = 1
        self._window_size = [400, 400]


class ALLOWED_THEMES(Enum):
    paraview = ParaViewTheme
    document = DocumentTheme
    dark = DarkTheme
    default = DefaultTheme
    testing = _TestingTheme
