"""PyVista plotting module."""

import sys
import pathlib
import collections.abc
from typing import Sequence
import logging
import os
import textwrap
import time
import warnings
import weakref
from functools import wraps
from threading import Thread
from typing import Dict

import numpy as np
import scooby

import pyvista
from pyvista import _vtk
from pyvista.utilities import (assert_empty_kwargs, convert_array,
                               convert_string_array, get_array,
                               is_pyvista_dataset, abstract_class,
                               numpy_to_texture, raise_not_matching,
                               wrap)
from ..utilities.regression import image_from_window
from ..utilities.misc import PyvistaDeprecationWarning
from .colors import get_cmap_safe
from .export_vtkjs import export_plotter_vtkjs
from .mapper import make_mapper
from .picking import PickingHelper
from .renderer import Renderer, Camera
from .tools import (normalize, opacity_transfer_function, parse_color,
                    parse_font_family, FONTS)
from .widgets import WidgetHelper
from .scalar_bars import ScalarBars
from .renderers import Renderers
from .render_window_interactor import RenderWindowInteractor

def _has_matplotlib():
    try:
        import matplotlib
        return True
    except ImportError:  # pragma: no cover
        return False


SUPPORTED_FORMATS = [".png", ".jpeg", ".jpg", ".bmp", ".tif", ".tiff"]
VERY_FIRST_RENDER = True  # windows plotter helper

def close_all():
    """Close all open/active plotters and clean up memory."""
    for key, p in _ALL_PLOTTERS.items():
        if not p._closed:
            p.close()
        p.deep_clean()
    _ALL_PLOTTERS.clear()
    return True


log = logging.getLogger(__name__)
log.setLevel('CRITICAL')
log.addHandler(logging.StreamHandler())


def _warn_xserver():  # pragma: no cover
    """Check if plotting is supported and persist this state.

    Check once and cache this value between calls.  Warn the user if
    plotting is not supported.  Configured to check on Linux and Mac
    OS since the Windows check is not quick.

    """
    # disable windows check until we can get a fast way of verifying
    # if windows has a windows manager (which it generally does)
    if os.name == 'nt':
        return

    if not hasattr(_warn_xserver, 'has_support'):
        _warn_xserver.has_support = pyvista.system_supports_plotting()

    if not _warn_xserver.has_support:
        # check if a display has been set
        if 'DISPLAY' in os.environ:
            return

        # finally, check if using a backend that doesn't require an xserver
        if pyvista.global_theme.jupyter_backend in ['ipygany']:
            return

        # Check if VTK has EGL support
        if 'EGL' in str(type(_vtk.vtkRenderWindow())):
            return

        warnings.warn('\n'
                      'This system does not appear to be running an xserver.\n'
                      'PyVista will likely segfault when rendering.\n\n'
                      'Try starting a virtual frame buffer with xvfb, or using\n '
                      ' ``pyvista.start_xvfb()``\n')


USE_SCALAR_BAR_ARGS = """
"stitle" is a depreciated keyword and will be removed in a future
release.

Use ``scalar_bar_args`` instead.  For example:

scalar_bar_args={'title': 'Scalar Bar Title'}
"""


@abstract_class
class BasePlotter(PickingHelper, WidgetHelper):
    """To be used by the Plotter and pyvistaqt.QtInteractor classes.

    Parameters
    ----------
    shape : list or tuple, optional
        Number of sub-render windows inside of the main window.
        Specify two across with ``shape=(2, 1)`` and a two by two grid
        with ``shape=(2, 2)``.  By default there is only one renderer.
        Can also accept a string descriptor as shape. E.g.:

            * ``shape="3|1"`` means 3 plots on the left and 1 on the right,
            * ``shape="4/2"`` means 4 plots on top and 2 at the bottom.

    border : bool, optional
        Draw a border around each render window.  Default ``False``.

    border_color : string or 3 item list, optional
        Either a string, rgb list, or hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1, 1, 1]``
            * ``color='#FFFFFF'``

    border_width : float, optional
        Width of the border in pixels when enabled.

    title : str, optional
        Window title of the scalar bar

    lighting : str, optional
        What lighting to set up for the plotter.
        Accepted options:

            * ``'light_kit'``: a vtk Light Kit composed of 5 lights.
            * ``'three lights'``: illumination using 3 lights.
            * ``'none'``: no light sources at instantiation.

        The default is a Light Kit (to be precise, 5 separate lights
        that act like a Light Kit).

    theme : pyvista.themes.DefaultTheme, optional
        Plot-specific theme.

    """

    mouse_position = None
    click_position = None

    def __init__(self, shape=(1, 1), border=None, border_color='k',
                 border_width=2.0, title=None, splitting_position=None,
                 groups=None, row_weights=None, col_weights=None,
                 lighting='light kit', theme=None):
        """Initialize base plotter."""
        log.debug('BasePlotter init start')
        self._theme = pyvista.themes.DefaultTheme()
        if theme is None:
            # copy global theme to ensure local plot theme is fixed
            # after creation.
            self._theme.load_theme(pyvista.global_theme)
        else:
            if not isinstance(theme, pyvista.themes.DefaultTheme):
                raise TypeError('Expected ``pyvista.themes.DefaultTheme`` for '
                                f'``theme``, not {type(theme).__name__}.')
            self._theme.load_theme(theme)

        self.image_transparent_background = self._theme.transparent_background

        # optional function to be called prior to closing
        self.__before_close_callback = None
        self._store_image = False
        self.mesh = None
        if title is None:
            title = self._theme.title
        self.title = str(title)

        # add renderers
        self.renderers = Renderers(self, shape, splitting_position, row_weights,
                                   col_weights, groups, border, border_color,
                                   border_width)

        # This keeps track of scalars names already plotted and their ranges
        self._scalar_bars = ScalarBars(self)

        # track if the camera has been setup
        self._first_time = True
        # Keep track of the scale
        self._labels = []

        # track if render window has ever been rendered
        self._rendered = False

        # this helps managing closed plotters
        self._closed = False

        # lighting style; be forgiving with input (accept underscores
        # and ignore case)
        lighting_normalized = str(lighting).replace('_', ' ').lower()
        if lighting_normalized == 'light kit':
            self.enable_lightkit()
        elif lighting_normalized == 'three lights':
            self.enable_3_lights()
        elif lighting_normalized != 'none':
            raise ValueError(f'Invalid lighting option "{lighting}".')

        # Add self to open plotters
        self._id_name = f"{hex(id(self))}-{len(_ALL_PLOTTERS)}"
        _ALL_PLOTTERS[self._id_name] = self

        # Key bindings
        self.reset_key_events()
        log.debug('BasePlotter init stop')

        self._image_depth_null = None
        self.last_image_depth = None
        self.last_image = None

    @property
    def theme(self):
        """Return or set the theme used for this plotter.

        Examples
        --------
        Use the dark theme for a plotter.

        >>> import pyvista
        >>> from pyvista import themes
        >>> pl = pyvista.Plotter()
        >>> pl.theme = themes.DarkTheme()

        """
        return self._theme

    @theme.setter
    def theme(self, theme):
        if not isinstance(theme, pyvista.themes.DefaultTheme):
            raise TypeError('Expected a pyvista theme like '
                            '``pyvista.themes.DefaultTheme``, '
                            f'not {type(theme).__name__}.')
        self._theme = theme

    @property
    def scalar_bar(self):
        """First scalar bar.  Kept for backwards compatibility."""
        return list(self.scalar_bars.values())[0]

    @property
    def scalar_bars(self):
        """Scalar bars.

        Examples
        --------
        >>> import pyvista
        >>> sphere = pyvista.Sphere()
        >>> sphere['Data'] = sphere.points[:, 2]
        >>> plotter = pyvista.Plotter()
        >>> _ = plotter.add_mesh(sphere)
        >>> plotter.scalar_bars
        Scalar Bar Title     Interactive
        "Data"               False

        Select a scalar bar actor based on the title of the bar.

        >>> plotter.scalar_bars['Data']  # doctest:+SKIP
        (vtkmodules.vtkRenderingAnnotation.vtkScalarBarActor)0x7fcd3567ca00

        """
        return self._scalar_bars

    @property
    def _before_close_callback(self):
        """Return the cached function (expecting a reference)."""
        if self.__before_close_callback is not None:
            return self.__before_close_callback()

    @_before_close_callback.setter
    def _before_close_callback(self, func):
        """Store a weakref.ref of the function being called."""
        if func is not None:
            self.__before_close_callback = weakref.ref(func)
        else:
            self.__before_close_callback = None

    @property
    def shape(self):
        """Shape of the plotter.

        Examples
        --------
        Return the plotter shape.

        >>> import pyvista
        >>> plotter = pyvista.Plotter(shape=(2, 2))
        >>> plotter.shape
        (2, 2)
        """
        return self.renderers._shape

    @property
    def renderer(self):
        """Return the active renderer."""
        return self.renderers.active_renderer

    @property
    def store_image(self):
        """Return if an image will be saved on close."""
        return self._store_image

    @store_image.setter
    def store_image(self, value):
        """Store last rendered frame on close."""
        self._store_image = bool(value)

    def subplot(self, index_row, index_column=None):
        """Set the active subplot.

        Parameters
        ----------
        index_row : int
            Index of the subplot to activate along the rows.

        index_column : int
            Index of the subplot to activate along the columns.

        """
        self.renderers.set_active_renderer(index_row, index_column)

    @wraps(Renderer.add_floor)
    def add_floor(self, *args, **kwargs):
        """Wrap ``Renderer.add_floor``."""
        return self.renderer.add_floor(*args, **kwargs)

    @wraps(Renderer.remove_floors)
    def remove_floors(self, *args, **kwargs):
        """Wrap ``Renderer.remove_floors``."""
        return self.renderer.remove_floors(*args, **kwargs)

    def enable_3_lights(self, only_active=False):
        """Enable 3-lights illumination.

        This will replace all pre-existing lights in the scene.

        Parameters
        ----------
        only_active : bool
            If ``True``, only change the active renderer. The default
            is that every renderer is affected.

        """
        def _to_pos(elevation, azimuth):
            theta = azimuth * np.pi / 180.0
            phi = (90.0 - elevation) * np.pi / 180.0
            x = np.sin(theta) * np.sin(phi)
            y = np.cos(phi)
            z = np.cos(theta) * np.sin(phi)
            return x, y, z

        renderers = [self.renderer] if only_active else self.renderers
        for renderer in renderers:
            renderer.remove_all_lights()

        # Inspired from Mayavi's version of Raymond Maple 3-lights illumination
        intensities = [1, 0.6, 0.5]
        all_angles = [(45.0, 45.0), (-30.0, -60.0), (-30.0, 60.0)]
        for intensity, angles in zip(intensities, all_angles):
            light = pyvista.Light(light_type='camera light')
            light.intensity = intensity
            light.position = _to_pos(*angles)
            for renderer in renderers:
                renderer.add_light(light)

    def disable_3_lights(self):
        """Please use ``enable_lightkit``, this method has been depreciated."""
        from pyvista.core.errors import DeprecationError
        raise DeprecationError('DEPRECATED: Please use ``enable_lightkit``')

    def enable_lightkit(self, only_active=False):
        """Enable the default light-kit lighting.

        See:
        https://www.researchgate.net/publication/2926068

        This will replace all pre-existing lights in the renderer.

        Parameters
        ----------
        only_active : bool
            If ``True``, only change the active renderer. The default is that
            every renderer is affected.

        """
        renderers = [self.renderer] if only_active else self.renderers

        light_kit = _vtk.vtkLightKit()
        for renderer in renderers:
            renderer.remove_all_lights()
            # Use the renderer as a vtkLightKit parser.
            # Feed it the LightKit, pop off the vtkLights, put back
            # pyvista Lights. This is the price we must pay for using
            # inheritance rather than composition.
            light_kit.AddLightsToRenderer(renderer)
            vtk_lights = renderer.lights
            renderer.remove_all_lights()
            for vtk_light in vtk_lights:
                light = pyvista.Light.from_vtk(vtk_light)
                renderer.add_light(light)
            renderer.LightFollowCameraOn()

    @wraps(Renderer.enable_anti_aliasing)
    def enable_anti_aliasing(self, *args, **kwargs):
        """Wrap ``Renderer.enable_anti_aliasing``."""
        self.renderer.enable_anti_aliasing(*args, **kwargs)

    @wraps(Renderer.disable_anti_aliasing)
    def disable_anti_aliasing(self, *args, **kwargs):
        """Wrap ``Renderer.disable_anti_aliasing``."""
        self.renderer.disable_anti_aliasing(*args, **kwargs)

    @wraps(Renderer.set_focus)
    def set_focus(self, *args, **kwargs):
        """Wrap ``Renderer.set_focus``."""
        log.debug('set_focus: %s, %s', str(args), str(kwargs))
        self.renderer.set_focus(*args, **kwargs)
        self.render()

    @wraps(Renderer.set_position)
    def set_position(self, *args, **kwargs):
        """Wrap ``Renderer.set_position``."""
        self.renderer.set_position(*args, **kwargs)
        self.render()

    @wraps(Renderer.set_viewup)
    def set_viewup(self, *args, **kwargs):
        """Wrap ``Renderer.set_viewup``."""
        self.renderer.set_viewup(*args, **kwargs)
        self.render()

    @wraps(Renderer.add_orientation_widget)
    def add_orientation_widget(self, *args, **kwargs):
        """Wrap ``Renderer.add_orientation_widget``."""
        return self.renderer.add_orientation_widget(*args, **kwargs)

    @wraps(Renderer.add_axes)
    def add_axes(self, *args, **kwargs):
        """Wrap ``Renderer.add_axes``."""
        return self.renderer.add_axes(*args, **kwargs)

    @wraps(Renderer.hide_axes)
    def hide_axes(self, *args, **kwargs):
        """Wrap ``Renderer.hide_axes``."""
        return self.renderer.hide_axes(*args, **kwargs)

    @wraps(Renderer.show_axes)
    def show_axes(self, *args, **kwargs):
        """Wrap ``Renderer.show_axes``."""
        return self.renderer.show_axes(*args, **kwargs)

    @wraps(Renderer.update_bounds_axes)
    def update_bounds_axes(self, *args, **kwargs):
        """Wrap ``Renderer.update_bounds_axes``."""
        return self.renderer.update_bounds_axes(*args, **kwargs)

    @wraps(Renderer.add_actor)
    def add_actor(self, *args, **kwargs):
        """Wrap ``Renderer.add_actor``."""
        return self.renderer.add_actor(*args, **kwargs)

    @wraps(Renderer.enable_parallel_projection)
    def enable_parallel_projection(self, *args, **kwargs):
        """Wrap ``Renderer.enable_parallel_projection``."""
        return self.renderer.enable_parallel_projection(*args, **kwargs)

    @wraps(Renderer.disable_parallel_projection)
    def disable_parallel_projection(self, *args, **kwargs):
        """Wrap ``Renderer.disable_parallel_projection``."""
        return self.renderer.disable_parallel_projection(*args, **kwargs)

    @wraps(Renderer.enable_shadows)
    def enable_shadows(self, *args, **kwargs):
        """Wrap ``Renderer.enable_shadows``."""
        return self.renderer.enable_shadows(*args, **kwargs)

    @wraps(Renderer.disable_shadows)
    def disable_shadows(self, *args, **kwargs):
        """Wrap ``Renderer.disable_shadows``."""
        return self.renderer.disable_shadows(*args, **kwargs)

    @property
    def parallel_projection(self):
        """Return parallel projection state of active render window."""
        return self.renderer.parallel_projection

    @parallel_projection.setter
    def parallel_projection(self, state):
        """Set parallel projection state of all active render windows."""
        self.renderer.parallel_projection = state

    @property
    def parallel_scale(self):
        """Return parallel scale of active render window."""
        return self.renderer.parallel_scale

    @parallel_scale.setter
    def parallel_scale(self, value):
        """Set parallel scale of all active render windows."""
        self.renderer.parallel_scale = value

    @wraps(Renderer.add_axes_at_origin)
    def add_axes_at_origin(self, *args, **kwargs):
        """Wrap ``Renderer.add_axes_at_origin``."""
        return self.renderer.add_axes_at_origin(*args, **kwargs)

    @wraps(Renderer.show_bounds)
    def show_bounds(self, *args, **kwargs):
        """Wrap ``Renderer.show_bounds``."""
        return self.renderer.show_bounds(*args, **kwargs)

    @wraps(Renderer.add_bounding_box)
    def add_bounding_box(self, *args, **kwargs):
        """Wrap ``Renderer.add_bounding_box``."""
        return self.renderer.add_bounding_box(*args, **kwargs)

    @wraps(Renderer.remove_bounding_box)
    def remove_bounding_box(self, *args, **kwargs):
        """Wrap ``Renderer.remove_bounding_box``."""
        return self.renderer.remove_bounding_box(*args, **kwargs)

    @wraps(Renderer.remove_bounds_axes)
    def remove_bounds_axes(self, *args, **kwargs):
        """Wrap ``Renderer.remove_bounds_axes``."""
        return self.renderer.remove_bounds_axes(*args, **kwargs)

    @wraps(Renderer.show_grid)
    def show_grid(self, *args, **kwargs):
        """Wrap ``Renderer.show_grid``."""
        return self.renderer.show_grid(*args, **kwargs)

    @wraps(Renderer.set_scale)
    def set_scale(self, *args, **kwargs):
        """Wrap ``Renderer.set_scale``."""
        return self.renderer.set_scale(*args, **kwargs)

    @wraps(Renderer.enable_eye_dome_lighting)
    def enable_eye_dome_lighting(self, *args, **kwargs):
        """Wrap ``Renderer.enable_eye_dome_lighting``."""
        return self.renderer.enable_eye_dome_lighting(*args, **kwargs)

    @wraps(Renderer.disable_eye_dome_lighting)
    def disable_eye_dome_lighting(self, *args, **kwargs):
        """Wrap ``Renderer.disable_eye_dome_lighting``."""
        return self.renderer.disable_eye_dome_lighting(*args, **kwargs)

    @wraps(Renderer.reset_camera)
    def reset_camera(self, *args, **kwargs):
        """Wrap ``Renderer.reset_camera``."""
        self.renderer.reset_camera(*args, **kwargs)
        self.render()

    @wraps(Renderer.isometric_view)
    def isometric_view(self, *args, **kwargs):
        """Wrap ``Renderer.isometric_view``."""
        return self.renderer.isometric_view(*args, **kwargs)

    @wraps(Renderer.view_isometric)
    def view_isometric(self, *args, **kwarg):
        """Wrap ``Renderer.view_isometric``."""
        return self.renderer.view_isometric(*args, **kwarg)

    @wraps(Renderer.view_vector)
    def view_vector(self, *args, **kwarg):
        """Wrap ``Renderer.view_vector``."""
        return self.renderer.view_vector(*args, **kwarg)

    @wraps(Renderer.view_xy)
    def view_xy(self, *args, **kwarg):
        """Wrap ``Renderer.view_xy``."""
        return self.renderer.view_xy(*args, **kwarg)

    @wraps(Renderer.view_yx)
    def view_yx(self, *args, **kwarg):
        """Wrap ``Renderer.view_yx``."""
        return self.renderer.view_yx(*args, **kwarg)

    @wraps(Renderer.view_xz)
    def view_xz(self, *args, **kwarg):
        """Wrap ``Renderer.view_xz``."""
        return self.renderer.view_xz(*args, **kwarg)

    @wraps(Renderer.view_zx)
    def view_zx(self, *args, **kwarg):
        """Wrap ``Renderer.view_zx``."""
        return self.renderer.view_zx(*args, **kwarg)

    @wraps(Renderer.view_yz)
    def view_yz(self, *args, **kwarg):
        """Wrap ``Renderer.view_yz``."""
        return self.renderer.view_yz(*args, **kwarg)

    @wraps(Renderer.view_zy)
    def view_zy(self, *args, **kwarg):
        """Wrap ``Renderer.view_zy``."""
        return self.renderer.view_zy(*args, **kwarg)

    @wraps(Renderer.disable)
    def disable(self, *args, **kwarg):
        """Wrap ``Renderer.disable``."""
        return self.renderer.disable(*args, **kwarg)

    @wraps(Renderer.enable)
    def enable(self, *args, **kwarg):
        """Wrap ``Renderer.enable``."""
        return self.renderer.enable(*args, **kwarg)

    @wraps(Renderer.enable_depth_peeling)
    def enable_depth_peeling(self, *args, **kwargs):
        """Wrap ``Renderer.enable_depth_peeling``."""
        if hasattr(self, 'ren_win'):
            result = self.renderer.enable_depth_peeling(*args, **kwargs)
            if result:
                self.ren_win.AlphaBitPlanesOn()

        return result

    @wraps(Renderer.disable_depth_peeling)
    def disable_depth_peeling(self):
        """Wrap ``Renderer.disable_depth_peeling``."""
        if hasattr(self, 'ren_win'):
            self.ren_win.AlphaBitPlanesOff()
            return self.renderer.disable_depth_peeling()

    @wraps(Renderer.get_default_cam_pos)
    def get_default_cam_pos(self, *args, **kwargs):
        """Wrap ``Renderer.get_default_cam_pos``."""
        return self.renderer.get_default_cam_pos(*args, **kwargs)

    @wraps(Renderer.remove_actor)
    def remove_actor(self, *args, **kwargs):
        """Wrap ``Renderer.remove_actor``."""
        for renderer in self.renderers:
            renderer.remove_actor(*args, **kwargs)
        return True

    @wraps(Renderer.set_environment_texture)
    def set_environment_texture(self, *args, **kwargs):
        """Wrap ``Renderer.set_environment_texture``."""
        return self.renderer.set_environment_texture(*args, **kwargs)

    #### Properties from Renderer ####

    @property
    def camera(self):
        """Return the active camera of the active renderer."""
        if not self.camera_set:
            self.camera_position = self.get_default_cam_pos()
            self.reset_camera()
            self.camera_set = True
        return self.renderer.camera

    @camera.setter
    def camera(self, camera):
        """Set the active camera for the rendering scene."""
        self.renderer.camera = camera

    @property
    def camera_set(self):
        """Return if the camera of the active renderer has been set."""
        return self.renderer.camera_set

    @camera_set.setter
    def camera_set(self, is_set):
        """Set if the camera has been set on the active renderer."""
        self.renderer.camera_set = is_set

    @property
    def bounds(self):
        """Return the bounds of the active renderer."""
        return self.renderer.bounds

    @property
    def length(self):
        """Return the length of the diagonal of the bounding box of the scene."""
        return self.renderer.length

    @property
    def center(self):
        """Return the center of the active renderer."""
        return self.renderer.center

    @property
    def _scalar_bar_slots(self):
        """Return the scalar bar slots of the active renderer."""
        return self.renderer._scalar_bar_slots

    @_scalar_bar_slots.setter
    def _scalar_bar_slots(self, value):
        """Set the scalar bar slots of the active renderer."""
        self.renderer._scalar_bar_slots = value

    @property
    def _scalar_bar_slot_lookup(self):
        """Return the scalar bar slot lookup of the active renderer."""
        return self.renderer._scalar_bar_slot_lookup

    @_scalar_bar_slot_lookup.setter
    def _scalar_bar_slot_lookup(self, value):
        """Set the scalar bar slot lookup of the active renderer."""
        self.renderer._scalar_bar_slot_lookup = value

    @property
    def scale(self):
        """Return the scaling of the active renderer."""
        return self.renderer.scale

    @scale.setter
    def scale(self, scale):
        """Set the scaling of the active renderer."""
        self.renderer.set_scale(*scale)

    @property
    def camera_position(self):
        """Return camera position of the active render window."""
        return self.renderer.camera_position

    @camera_position.setter
    def camera_position(self, camera_location):
        """Set camera position of the active render window."""
        self.renderer.camera_position = camera_location

    @property
    def background_color(self):
        """Return the background color of the active render window."""
        return self.renderers.active_renderer.GetBackground()

    @background_color.setter
    def background_color(self, color):
        """Set the background color of all the render windows."""
        self.set_background(color)

    @property
    def window_size(self):
        """Return the render window size."""
        return list(self.ren_win.GetSize())

    @window_size.setter
    def window_size(self, window_size):
        """Set the render window size."""
        self.ren_win.SetSize(window_size[0], window_size[1])

    @property
    def image_depth(self):
        """Return a depth image representing current render window.

        Helper attribute for ``get_image_depth``.

        """
        return self.get_image_depth()

    def _check_rendered(self):
        """Check if the render window has been shown and raise an exception if not."""
        if not self._rendered:
            raise AttributeError('\nThis plotter has not yet been setup and rendered '
                                 'with ``show()``.\n'
                                 'Consider setting ``off_screen=True`` '
                                 'for off screen rendering.\n')

    def _check_has_ren_win(self):
        """Check if render window attribute exists and raise an exception if not."""
        if not hasattr(self, 'ren_win'):
            raise AttributeError('\n\nTo retrieve an image after the render window '
                                 'has been closed, set:\n\n'
                                 ' ``plotter.store_image = True``\n\n'
                                 'before closing the plotter.')

    @property
    def image(self):
        """Return an image array of current render window.

        To retrieve an image after the render window has been closed,
        set: ``plotter.store_image = True`` before closing the plotter.
        """
        if not hasattr(self, 'ren_win') and self.last_image is not None:
            return self.last_image

        self._check_rendered()
        self._check_has_ren_win()

        data = image_from_window(self.ren_win)
        if self.image_transparent_background:
            return data

        # ignore alpha channel
        return data[:, :, :-1]

    def render(self):
        """Render the main window.

        Does nothing until ``show`` has been called.
        """
        if hasattr(self, 'ren_win') and not self._first_time:
            log.debug('Rendering')
            self.ren_win.Render()
            self._rendered = True

    @wraps(RenderWindowInteractor.add_key_event)
    def add_key_event(self, *args, **kwargs):
        """Wrap RenderWindowInteractor.add_key_event."""
        if hasattr(self, 'iren'):
            self.iren.add_key_event(*args, **kwargs)

    def clear_events_for_key(self, key):
        """Remove the callbacks associated to the key."""
        self.iren.clear_events_for_key(key)

    def store_mouse_position(self, *args):
        """Store mouse position."""
        if not hasattr(self, "iren"):
            raise AttributeError("This plotting window is not interactive.")
        self.mouse_position = self.iren.get_event_position()

    def store_click_position(self, *args):
        """Store click position in viewport coordinates."""
        if not hasattr(self, "iren"):
            raise AttributeError("This plotting window is not interactive.")
        self.click_position = self.iren.get_event_position()
        self.mouse_position = self.click_position

    def track_mouse_position(self):
        """Keep track of the mouse position.

        This will potentially slow down the interactor. No callbacks
        supported here - use
        :func:`pyvista.BasePlotter.track_click_position` instead.

        """
        self.iren.track_mouse_position(self.store_mouse_position)

    def untrack_mouse_position(self):
        """Stop tracking the mouse position."""
        self.iren.untrack_mouse_position()

    @wraps(RenderWindowInteractor.track_click_position)
    def track_click_position(self, *args, **kwargs):
        """Wrap RenderWindowInteractor.track_click_position."""
        self.iren.track_click_position(*args, **kwargs)

    def untrack_click_position(self):
        """Stop tracking the click position."""
        self.iren.untrack_click_position()

    def _prep_for_close(self):
        """Make sure a screenshot is acquired before closing.

        This doesn't actually close anything! It just preps the plotter for
        closing.
        """
        # Grab screenshot right before renderer closes
        self.last_image = self.screenshot(True, return_img=True)
        self.last_image_depth = self.get_image_depth()

    def increment_point_size_and_line_width(self, increment):
        """Increment point size and line width of all actors.

        For every actor in the scene, increment both its point size and
        line width by the given value.

        """
        for renderer in self.renderers:
            for actor in renderer._actors.values():
                if hasattr(actor, "GetProperty"):
                    prop = actor.GetProperty()
                    if hasattr(prop, "SetPointSize"):
                        prop.SetPointSize(prop.GetPointSize() + increment)
                    if hasattr(prop, "SetLineWidth"):
                        prop.SetLineWidth(prop.GetLineWidth() + increment)
        self.render()
        return

    def reset_key_events(self):
        """Reset all of the key press events to their defaults."""
        if hasattr(self, 'iren'):
            self.iren.clear_key_event_callbacks()

        self.add_key_event('q', self._prep_for_close) # Add no matter what
        b_left_down_callback = lambda: self.iren.add_observer('LeftButtonPressEvent', self.left_button_down)
        self.add_key_event('b', b_left_down_callback)
        self.add_key_event('v', lambda: self.isometric_view_interactive())
        self.add_key_event('C', lambda: self.enable_cell_picking())
        self.add_key_event('Up', lambda: self.camera.Zoom(1.05))
        self.add_key_event('Down', lambda: self.camera.Zoom(0.95))
        self.add_key_event('plus', lambda: self.increment_point_size_and_line_width(1))
        self.add_key_event('minus', lambda: self.increment_point_size_and_line_width(-1))

    @wraps(RenderWindowInteractor.key_press_event)
    def key_press_event(self, *args, **kwargs):
        """Wrap RenderWindowInteractor.key_press_event."""
        self.iren.key_press_event(*args, **kwargs)

    def left_button_down(self, obj, event_type):
        """Register the event for a left button down click."""
        if hasattr(self.ren_win, 'GetOffScreenFramebuffer'):
            if not self.ren_win.GetOffScreenFramebuffer().GetFBOIndex():
                # must raise a runtime error as this causes a segfault on VTK9
                raise ValueError('Invoking helper with no framebuffer')
        # Get 2D click location on window
        click_pos = self.iren.get_event_position()

        # Get corresponding click location in the 3D plot
        picker = _vtk.vtkWorldPointPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        self.pickpoint = np.asarray(picker.GetPickPosition()).reshape((-1, 3))
        if np.any(np.isnan(self.pickpoint)):
            self.pickpoint[:] = 0

    @wraps(RenderWindowInteractor.enable_trackball_style)
    def enable_trackball_style(self):
        """Wrap RenderWindowInteractor.enable_trackball_style."""
        self.iren.enable_trackball_style()

    @wraps(RenderWindowInteractor.enable_trackball_actor_style)
    def enable_trackball_actor_style(self):
        """Wrap RenderWindowInteractor.enable_trackball_actor_style."""
        self.iren.enable_trackball_actor_style()

    @wraps(RenderWindowInteractor.enable_image_style)
    def enable_image_style(self):
        """Wrap RenderWindowInteractor.enable_image_style."""
        self.iren.enable_image_style()

    @wraps(RenderWindowInteractor.enable_joystick_style)
    def enable_joystick_style(self):
        """Wrap RenderWindowInteractor.enable_joystick_style."""
        self.iren.enable_joystick_style()

    @wraps(RenderWindowInteractor.enable_zoom_style)
    def enable_zoom_style(self):
        """Wrap RenderWindowInteractor.enable_zoom_style."""
        self.iren.enable_zoom_style()

    @wraps(RenderWindowInteractor.enable_terrain_style)
    def enable_terrain_style(self):
        """Wrap RenderWindowInteractor.enable_terrain_style."""
        self.iren.enable_terrain_style()

    @wraps(RenderWindowInteractor.enable_rubber_band_style)
    def enable_rubber_band_style(self):
        """Wrap RenderWindowInteractor.enable_rubber_band_style."""
        self.iren.enable_rubber_band_style()

    @wraps(RenderWindowInteractor.enable_rubber_band_2d_style)
    def enable_rubber_band_2d_style(self):
        """Wrap RenderWindowInteractor.enable_rubber_band_2d_style."""
        self.iren.enable_rubber_band_2d_style()

    def hide_axes_all(self):
        """Hide the axes orientation widget in all renderers."""
        for renderer in self.renderers:
            renderer.hide_axes()

    def show_axes_all(self):
        """Show the axes orientation widget in all renderers."""
        for renderer in self.renderers:
            renderer.show_axes()

    def isometric_view_interactive(self):
        """Set the current interactive render window to isometric view."""
        interactor = self.iren.get_interactor_style()
        renderer = interactor.GetCurrentRenderer()
        if renderer is None:
            renderer = self.renderer
        renderer.view_isometric()

    def update(self, stime=1, force_redraw=True):
        """Update window, redraw, process messages query.

        Parameters
        ----------
        stime : int, optional
            Duration of timer that interrupt vtkRenderWindowInteractor
            in milliseconds.

        force_redraw : bool, optional
            Call ``render`` immediately.

        """
        if stime <= 0:
            stime = 1

        curr_time = time.time()
        if Plotter.last_update_time > curr_time:
            Plotter.last_update_time = curr_time

        update_rate = self.iren.get_desired_update_rate()
        if (curr_time - Plotter.last_update_time) > (1.0/update_rate):
            self.right_timer_id = self.iren.create_repeating_timer(stime)
            self.render()
            Plotter.last_update_time = curr_time
        elif force_redraw:
            self.render()

    def add_mesh(self, mesh, color=None, style=None, scalars=None,
                 clim=None, show_edges=None, edge_color=None,
                 point_size=5.0, line_width=None, opacity=1.0,
                 flip_scalars=False, lighting=None, n_colors=256,
                 interpolate_before_map=True, cmap=None, label=None,
                 reset_camera=None, scalar_bar_args=None, show_scalar_bar=None,
                 multi_colors=False, name=None, texture=None,
                 render_points_as_spheres=None, render_lines_as_tubes=False,
                 smooth_shading=None, ambient=0.0, diffuse=1.0, specular=0.0,
                 specular_power=100.0, nan_color=None, nan_opacity=1.0,
                 culling=None, rgb=False, categories=False, silhouette=False,
                 use_transparency=False, below_color=None, above_color=None,
                 annotations=None, pickable=True, preference="point",
                 log_scale=False, pbr=False, metallic=0.0, roughness=0.5,
                 render=True, component=None, **kwargs):
        """Add any PyVista/VTK mesh or dataset that PyVista can wrap to the scene.

        This method is using a mesh representation to view the surfaces
        and/or geometry of datasets. For volume rendering, see
        :func:`pyvista.BasePlotter.add_volume`.

        Parameters
        ----------
        mesh : pyvista.DataSet or pyvista.MultiBlock
            Any PyVista or VTK mesh is supported. Also, any dataset
            that :func:`pyvista.wrap` can handle including NumPy
            arrays of XYZ points.

        color : string or 3 item list, optional, defaults to white
            Use to make the entire mesh have a single solid color.
            Either a string, RGB list, or hex color string.  For example:
            ``color='white'``, ``color='w'``, ``color=[1, 1, 1]``, or
            ``color='#FFFFFF'``. Color will be overridden if scalars are
            specified.

        style : string, optional
            Visualization style of the mesh.  One of the following:
            ``style='surface'``, ``style='wireframe'``, ``style='points'``.
            Defaults to ``'surface'``. Note that ``'wireframe'`` only shows a
            wireframe of the outer geometry.

        scalars : str or numpy.ndarray, optional
            Scalars used to "color" the mesh.  Accepts a string name
            of an array that is present on the mesh or an array equal
            to the number of cells or the number of points in the
            mesh.  Array should be sized as a single vector. If both
            ``color`` and ``scalars`` are ``None``, then the active
            scalars are used.

        clim : 2 item list, optional
            Color bar range for scalars.  Defaults to minimum and
            maximum of scalars array.  Example: ``[-1, 2]``. ``rng``
            is also an accepted alias for this.

        show_edges : bool, optional
            Shows the edges of a mesh.  Does not apply to a wireframe
            representation.

        edge_color : string or 3 item list, optional, defaults to black
            The solid color to give the edges when ``show_edges=True``.
            Either a string, RGB list, or hex color string.

        point_size : float, optional
            Point size of any nodes in the dataset plotted. Also
            applicable when style='points'. Default ``5.0``.

        line_width : float, optional
            Thickness of lines.  Only valid for wireframe and surface
            representations.  Default None.

        opacity : float, str, array-like
            Opacity of the mesh. If a single float value is given, it
            will be the global opacity of the mesh and uniformly
            applied everywhere - should be between 0 and 1. A string
            can also be specified to map the scalars range to a
            predefined opacity transfer function (options include:
            'linear', 'linear_r', 'geom', 'geom_r').  A string could
            also be used to map a scalars array from the mesh to the
            opacity (must have same number of elements as the
            ``scalars`` argument). Or you can pass a custom made
            transfer function that is an array either ``n_colors`` in
            length or shorter.

        flip_scalars : bool, optional
            Flip direction of cmap. Most colormaps allow ``*_r``
            suffix to do this as well.

        lighting : bool, optional
            Enable or disable view direction lighting. Default ``False``.

        n_colors : int, optional
            Number of colors to use when displaying scalars. Defaults to 256.
            The scalar bar will also have this many colors.

        interpolate_before_map : bool, optional
            Enabling makes for a smoother scalars display.  Default is
            ``True``.  When ``False``, OpenGL will interpolate the
            mapped colors which can result is showing colors that are
            not present in the color map.

        cmap : str, list, optional
            Name of the Matplotlib colormap to use when mapping the
            ``scalars``.  See available Matplotlib colormaps.  Only
            applicable for when displaying ``scalars``. Requires
            Matplotlib to be installed.  ``colormap`` is also an
            accepted alias for this. If ``colorcet`` or ``cmocean``
            are installed, their colormaps can be specified by name.

            You can also specify a list of colors to override an
            existing colormap with a custom one.  For example, to
            create a three color colormap you might specify
            ``['green', 'red', 'blue']``

        label : str, optional
            String label to use when adding a legend to the scene with
            :func:`pyvista.BasePlotter.add_legend`

        reset_camera : bool, optional
            Reset the camera after adding this mesh to the scene

        scalar_bar_args : dict, optional
            Dictionary of keyword arguments to pass when adding the
            scalar bar to the scene. For options, see
            :func:`pyvista.BasePlotter.add_scalar_bar`.

        show_scalar_bar : bool
            If ``False``, a scalar bar will not be added to the
            scene. Defaults to ``True``.

        multi_colors : bool, optional
            If a ``MultiBlock`` dataset is given this will color each
            block by a solid color using matplotlib's color cycler.

        name : str, optional
            The name for the added mesh/actor so that it can be easily
            updated.  If an actor of this name already exists in the
            rendering window, it will be replaced by the new actor.

        texture : vtk.vtkTexture or np.ndarray or boolean, optional
            A texture to apply if the input mesh has texture
            coordinates.  This will not work with MultiBlock
            datasets. If set to ``True``, the first available texture
            on the object will be used. If a string name is given, it
            will pull a texture with that name associated to the input
            mesh.

        render_points_as_spheres : bool, optional
            Render points as spheres rather than dots.

        render_lines_as_tubes : bool, optional
            Show lines as thick tubes rather than flat lines.  Control
            the width with ``line_width``.

        smooth_shading : bool, optional
            Enable smooth shading when ``True`` using either the 
            Gouraud or Phong shading algorithm.  When ``False``, use
            flat shading.
            Automatically enabled when ``pbr=True``.

        ambient : float, optional
            When lighting is enabled, this is the amount of light from
            0 to 1 that reaches the actor when not directed at the
            light source emitted from the viewer.  Default 0.0

        diffuse : float, optional
            The diffuse lighting coefficient. Default 1.0

        specular : float, optional
            The specular lighting coefficient. Default 0.0

        specular_power : float, optional
            The specular power. Between 0.0 and 128.0

        nan_color : string or 3 item list, optional, defaults to gray
            The color to use for all ``NaN`` values in the plotted
            scalar array.

        nan_opacity : float, optional
            Opacity of ``NaN`` values.  Should be between 0 and 1.
            Default 1.0

        culling : str, optional
            Does not render faces that are culled. Options are
            ``'front'`` or ``'back'``. This can be helpful for dense
            surface meshes, especially when edges are visible, but can
            cause flat meshes to be partially displayed.  Defaults to
            ``False``.

        rgb : bool, optional
            If an 2 dimensional array is passed as the scalars, plot
            those values as RGB(A) colors! ``rgba`` is also an accepted
            alias for this.  Opacity (the A) is optional.

        categories : bool, optional
            If set to ``True``, then the number of unique values in
            the scalar array will be used as the ``n_colors``
            argument.

        silhouette : dict, bool, optional
            If set to ``True``, plot a silhouette highlight for the
            mesh. This feature is only available for a triangulated
            ``PolyData``.  As a ``dict``, it contains the properties
            of the silhouette to display:

                * ``color``: ``str`` or 3-item ``list``, color of the silhouette
                * ``line_width``: ``float``, edge width
                * ``opacity``: ``float`` between 0 and 1, edge transparency
                * ``feature_angle``: If a ``float``, display sharp edges
                  exceeding that angle in degrees.
                * ``decimate``: ``float`` between 0 and 1, level of decimation

        use_transparency : bool, optional
            Invert the opacity mappings and make the values correspond
            to transparency.

        below_color : string or 3 item list, optional
            Solid color for values below the scalars range
            (``clim``). This will automatically set the scalar bar
            ``below_label`` to ``'Below'``.

        above_color : string or 3 item list, optional
            Solid color for values below the scalars range
            (``clim``). This will automatically set the scalar bar
            ``above_label`` to ``'Above'``.

        annotations : dict, optional
            Pass a dictionary of annotations. Keys are the float
            values in the scalars range to annotate on the scalar bar
            and the values are the the string annotations.

        pickable : bool, optional
            Set whether this mesh is pickable.

        log_scale : bool, optional
            Use log scale when mapping data to colors. Scalars less
            than zero are mapped to the smallest representable
            positive float. Default: ``True``.

        pbr : bool, optional
            Enable physics based rendering (PBR) if the mesh is
            ``PolyData``.  Use the ``color`` argument to set the base
            color. This is only available in VTK>=9.

        metallic : float, optional
            Usually this value is either 0 or 1 for a real material
            but any value in between is valid. This parameter is only
            used by PBR interpolation. Default value is 0.0.

        roughness : float, optional
            This value has to be between 0 (glossy) and 1 (rough). A
            glossy material has reflections and a high specular
            part. This parameter is only used by PBR
            interpolation. Default value is 0.5.

        render : bool, optional
            Force a render when ``True``.  Default ``True``.

        component :  int, optional
            Set component of vector valued scalars to plot.  Must be
            nonnegative, if supplied. If ``None``, the magnitude of
            the vector is plotted.

        Returns
        -------
        actor: vtk.vtkActor
            VTK actor of the mesh.

        Examples
        --------
        Add a sphere to the plotter and show it with a custom scalar
        bar title.

        >>> import pyvista
        >>> sphere = pyvista.Sphere()
        >>> sphere['Data'] = sphere.points[:, 2]
        >>> plotter = pyvista.Plotter()
        >>> _ = plotter.add_mesh(sphere,
        ...                      scalar_bar_args={'title': 'Z Position'})
        >>> cpos = plotter.show()

        """
        # Convert the VTK data object to a pyvista wrapped object if necessary
        if not is_pyvista_dataset(mesh):
            mesh = wrap(mesh)
            if not is_pyvista_dataset(mesh):
                raise TypeError(f'Object type ({type(mesh)}) not supported for plotting in PyVista.')
        ##### Parse arguments to be used for all meshes #####

        # Avoid mutating input
        if scalar_bar_args is None:
            scalar_bar_args = {'n_colors': n_colors}
        else:
            scalar_bar_args = scalar_bar_args.copy()

        if show_edges is None:
            show_edges = self._theme.show_edges

        if edge_color is None:
            edge_color = self._theme.edge_color

        if show_scalar_bar is None:
            show_scalar_bar = self._theme.show_scalar_bar

        if lighting is None:
            lighting = self._theme.lighting

        if smooth_shading is None:
            if pbr:
                smooth_shading = True
            else:
                smooth_shading = self._theme.smooth_shading

        # supported aliases
        clim = kwargs.pop('rng', clim)
        cmap = kwargs.pop('colormap', cmap)
        culling = kwargs.pop("backface_culling", culling)

        if render_points_as_spheres is None:
            render_points_as_spheres = self._theme.render_points_as_spheres

        if name is None:
            name = f'{type(mesh).__name__}({mesh.memory_address})'

        if nan_color is None:
            nan_color = self._theme.nan_color
        nan_color = list(parse_color(nan_color))
        nan_color.append(nan_opacity)
        if color is True:
            color = self._theme.color

        if texture is False:
            texture = None

        if culling is True:
            culling = 'backface'

        rgb = kwargs.pop('rgba', rgb)

        # account for legacy behavior
        if 'stitle' in kwargs:  # pragma: no cover
            warnings.warn(USE_SCALAR_BAR_ARGS, PyvistaDeprecationWarning)
            scalar_bar_args.setdefault('title', kwargs.pop('stitle'))

        if "scalar" in kwargs:
            raise TypeError("`scalar` is an invalid keyword argument for `add_mesh`. Perhaps you mean `scalars` with an s?")
        assert_empty_kwargs(**kwargs)

        ##### Handle composite datasets #####

        if isinstance(mesh, pyvista.MultiBlock):
            # first check the scalars
            if clim is None and scalars is not None:
                # Get the data range across the array for all blocks
                # if scalars specified
                if isinstance(scalars, str):
                    clim = mesh.get_data_range(scalars)
                else:
                    # TODO: an array was given... how do we deal with
                    #       that? Possibly a 2D arrays or list of
                    #       arrays where first index corresponds to
                    #       the block? This could get complicated real
                    #       quick.
                    raise TypeError('scalars array must be given as a string name for multiblock datasets.')

            the_arguments = locals()
            the_arguments.pop('self')
            the_arguments.pop('mesh')
            the_arguments.pop('kwargs')

            if multi_colors:
                # Compute unique colors for each index of the block
                if _has_matplotlib():
                    import matplotlib
                    from itertools import cycle
                    cycler = matplotlib.rcParams['axes.prop_cycle']
                    colors = cycle(cycler)
                else:
                    multi_colors = False
                    logging.warning('Please install matplotlib for color cycles')

            # Now iteratively plot each element of the multiblock dataset
            actors = []
            for idx in range(mesh.GetNumberOfBlocks()):
                if mesh[idx] is None:
                    continue
                # Get a good name to use
                next_name = f'{name}-{idx}'
                # Get the data object
                if not is_pyvista_dataset(mesh[idx]):
                    data = wrap(mesh.GetBlock(idx))
                    if not is_pyvista_dataset(mesh[idx]):
                        continue # move on if we can't plot it
                else:
                    data = mesh.GetBlock(idx)
                if data is None or (not isinstance(data, pyvista.MultiBlock) and data.n_points < 1):
                    # Note that a block can exist but be None type
                    # or it could have zeros points (be empty) after filtering
                    continue
                # Now check that scalars is available for this dataset
                if isinstance(data, _vtk.vtkMultiBlockDataSet) or get_array(data, scalars) is None:
                    ts = None
                else:
                    ts = scalars
                if multi_colors:
                    color = next(colors)['color']

                ## Add to the scene
                the_arguments['color'] = color
                the_arguments['scalars'] = ts
                the_arguments['name'] = next_name
                the_arguments['texture'] = None
                a = self.add_mesh(data, **the_arguments)
                actors.append(a)

                if (reset_camera is None and not self.camera_set) or reset_camera:
                    cpos = self.get_default_cam_pos()
                    self.camera_position = cpos
                    self.camera_set = False
                    self.reset_camera()
            return actors

        ##### Plot a single PyVista mesh #####

        silhouette_params = self._theme.silhouette.to_dict()
        if isinstance(silhouette, dict):
            silhouette_params.update(silhouette)
            silhouette = True
        if silhouette:
            if not isinstance(mesh, pyvista.PolyData):
                raise TypeError(f"Expected type is `PolyData` but {type(mesh)} was given.")
            if isinstance(silhouette_params["decimate"], float):
                silhouette_mesh = mesh.decimate(silhouette_params["decimate"])
            else:
                silhouette_mesh = mesh
            alg = _vtk.vtkPolyDataSilhouette()
            alg.SetInputData(silhouette_mesh)
            alg.SetCamera(self.renderer.camera)
            if silhouette_params["feature_angle"] is not None:
                alg.SetEnableFeatureAngle(True)
                alg.SetFeatureAngle(silhouette_params["feature_angle"])
            else:
                alg.SetEnableFeatureAngle(False)
            mapper = make_mapper(_vtk.vtkDataSetMapper)
            mapper.SetInputConnection(alg.GetOutputPort())
            _, prop = self.add_actor(mapper)
            prop.SetColor(parse_color(silhouette_params["color"]))
            prop.SetOpacity(silhouette_params["opacity"])
            prop.SetLineWidth(silhouette_params["line_width"])

        # Compute surface normals if using smooth shading
        if smooth_shading:
            # extract surface if mesh is exterior
            if not isinstance(mesh, pyvista.PolyData):
                grid = mesh
                mesh = grid.extract_surface()
                ind = mesh.point_arrays['vtkOriginalPointIds']
                # remap scalars
                if isinstance(scalars, np.ndarray):
                    scalars = scalars[ind]
            if texture:
                _tcoords = mesh.t_coords
            mesh.compute_normals(cell_normals=False, inplace=True)
            if texture:
                mesh.t_coords = _tcoords

        if mesh.n_points < 1:
            raise ValueError('Empty meshes cannot be plotted. Input mesh has zero points.')

        # Try to plot something if no preference given
        if scalars is None and color is None and texture is None:
            # Prefer texture first
            if len(list(mesh.textures.keys())) > 0:
                texture = True
            # If no texture, plot any active scalar
            else:
                # Make sure scalars components are not vectors/tuples
                scalars = mesh.active_scalars_name
                # Don't allow plotting of string arrays by default
                if scalars is not None:# and np.issubdtype(mesh.active_scalars.dtype, np.number):
                    scalar_bar_args.setdefault('title', scalars)
                else:
                    scalars = None

        # set main values
        self.mesh = mesh
        self.mapper = make_mapper(_vtk.vtkDataSetMapper)
        self.mapper.SetInputData(self.mesh)
        self.mapper.GetLookupTable().SetNumberOfTableValues(n_colors)
        if interpolate_before_map:
            self.mapper.InterpolateScalarsBeforeMappingOn()

        actor = _vtk.vtkActor()
        prop = _vtk.vtkProperty()
        actor.SetMapper(self.mapper)
        actor.SetProperty(prop)

        # Make sure scalars is a numpy array after this point
        original_scalar_name = None
        if isinstance(scalars, str):
            self.mapper.SetArrayName(scalars)
            original_scalar_name = scalars
            scalars = get_array(mesh, scalars,
                                preference=preference, err=True)
            scalar_bar_args.setdefault('title', original_scalar_name)

        if texture is True or isinstance(texture, (str, int)):
            texture = mesh._activate_texture(texture)

        if texture:

            if isinstance(texture, np.ndarray):
                texture = numpy_to_texture(texture)
            if not isinstance(texture, (_vtk.vtkTexture, _vtk.vtkOpenGLTexture)):
                raise TypeError(f'Invalid texture type ({type(texture)})')
            if mesh.GetPointData().GetTCoords() is None:
                raise ValueError('Input mesh does not have texture coordinates to support the texture.')
            actor.SetTexture(texture)
            # Set color to white by default when using a texture
            if color is None:
                color = 'white'
            if scalars is None:
                show_scalar_bar = False
            self.mapper.SetScalarModeToUsePointFieldData()

            # see https://github.com/pyvista/pyvista/issues/950
            mesh.set_active_scalars(None)

        # Handle making opacity array =========================================

        _custom_opac = False
        if isinstance(opacity, str):
            try:
                # Get array from mesh
                opacity = get_array(mesh, opacity,
                                    preference=preference, err=True)
                if np.any(opacity > 1):
                    warnings.warn("Opacity scalars contain values over 1")
                if np.any(opacity < 0):
                    warnings.warn("Opacity scalars contain values less than 0")
                _custom_opac = True
            except:
                # Or get opacity transfer function
                opacity = opacity_transfer_function(opacity, n_colors)
            else:
                if scalars.shape[0] != opacity.shape[0]:
                    raise ValueError('Opacity array and scalars array must have the same number of elements.')
        elif isinstance(opacity, (np.ndarray, list, tuple)):
            opacity = np.array(opacity)
            if scalars.shape[0] == opacity.shape[0]:
                # User could pass an array of opacities for every point/cell
                _custom_opac = True
            else:
                opacity = opacity_transfer_function(opacity, n_colors)

        if use_transparency and np.max(opacity) <= 1.0:
            opacity = 1 - opacity
        elif use_transparency and isinstance(opacity, np.ndarray):
            opacity = 255 - opacity

        # Scalars formatting ==================================================
        if cmap is None:  # Set default map if matplotlib is available
            if _has_matplotlib():
                cmap = self._theme.cmap
        # Set the array title for when it is added back to the mesh
        if _custom_opac:
            title = '__custom_rgba'
        else:
            title = scalar_bar_args.get('title', 'Data')
        if scalars is not None:
            # if scalars is a string, then get the first array found with that name

            if not isinstance(scalars, np.ndarray):
                scalars = np.asarray(scalars)

            _using_labels = False
            if not np.issubdtype(scalars.dtype, np.number):
                # raise TypeError('Non-numeric scalars are currently not supported for plotting.')
                # TODO: If str array, digitive and annotate
                cats, scalars = np.unique(scalars.astype('|S'), return_inverse=True)
                values = np.unique(scalars)
                clim = [np.min(values) - 0.5, np.max(values) + 0.5]
                title = f'{title}-digitized'
                n_colors = len(cats)
                scalar_bar_args.setdefault('n_labels', 0)
                _using_labels = True

            if rgb:
                show_scalar_bar = False
                if scalars.ndim != 2 or scalars.shape[1] < 3 or scalars.shape[1] > 4:
                    raise ValueError('RGB array must be n_points/n_cells by 3/4 in shape.')

            if scalars.ndim != 1:
                if rgb:
                    pass
                elif scalars.ndim == 2 and (scalars.shape[0] == mesh.n_points or scalars.shape[0] == mesh.n_cells):
                    if not isinstance(component, (int, type(None))):
                        raise TypeError('component must be either None or an integer')
                    if component is None:
                        scalars = np.linalg.norm(scalars.copy(), axis=1)
                        title = '{}-normed'.format(title)
                    elif component < scalars.shape[1] and component >= 0:
                        scalars = scalars[:, component].copy()
                        title = '{}-{}'.format(title, component)
                    else:
                        raise ValueError(
                            ('component must be nonnegative and less than the '
                             'dimensionality of the scalars array: {}').format(
                                 scalars.shape[1]
                             )
                        )
                else:
                    scalars = scalars.ravel()

            if scalars.dtype == np.bool_:
                scalars = scalars.astype(np.float_)

            def prepare_mapper(scalars):
                # Scalars interpolation approach
                if scalars.shape[0] == mesh.n_points:
                    self.mesh.point_arrays.append(scalars, title, True)
                    self.mesh.active_scalars_name = title
                    self.mapper.SetScalarModeToUsePointData()
                elif scalars.shape[0] == mesh.n_cells:
                    self.mesh.cell_arrays.append(scalars, title, True)
                    self.mesh.active_scalars_name = title
                    self.mapper.SetScalarModeToUseCellData()
                else:
                    raise_not_matching(scalars, mesh)
                # Common tasks
                self.mapper.GetLookupTable().SetNumberOfTableValues(n_colors)
                if interpolate_before_map:
                    self.mapper.InterpolateScalarsBeforeMappingOn()
                if rgb or _custom_opac:
                    self.mapper.SetColorModeToDirectScalars()
                else:
                    self.mapper.SetColorModeToMapScalars()
                return

            prepare_mapper(scalars)
            table = self.mapper.GetLookupTable()

            if _using_labels:
                table.SetAnnotations(convert_array(values), convert_string_array(cats))

            if isinstance(annotations, dict):
                for val, anno in annotations.items():
                    table.SetAnnotation(float(val), str(anno))

            # Set scalars range
            if clim is None:
                clim = [np.nanmin(scalars), np.nanmax(scalars)]
            elif isinstance(clim, (int, float)):
                clim = [-clim, clim]

            if log_scale:
                if clim[0] <= 0:
                    clim = [sys.float_info.min, clim[1]]
                table.SetScaleToLog10()

            if np.any(clim) and not rgb:
                self.mapper.scalar_range = clim[0], clim[1]

            table.SetNanColor(nan_color)
            if above_color:
                table.SetUseAboveRangeColor(True)
                table.SetAboveRangeColor(*parse_color(above_color, opacity=1))
                scalar_bar_args.setdefault('above_label', 'Above')
            if below_color:
                table.SetUseBelowRangeColor(True)
                table.SetBelowRangeColor(*parse_color(below_color, opacity=1))
                scalar_bar_args.setdefault('below_label', 'Below')

            if cmap is not None:
                # have to add the attribute to pass it onward to some classes
                if isinstance(cmap, str):
                    self.mapper.cmap = cmap
                # ipygany uses different colormaps
                if self._theme.jupyter_backend == 'ipygany':
                    from ..jupyter.pv_ipygany import check_colormap
                    check_colormap(cmap)
                else:
                    if not _has_matplotlib():
                        cmap = None
                        logging.warning('Please install matplotlib for color maps.')

                    cmap = get_cmap_safe(cmap)
                    if categories:
                        if categories is True:
                            n_colors = len(np.unique(scalars))
                        elif isinstance(categories, int):
                            n_colors = categories
                    ctable = cmap(np.linspace(0, 1, n_colors))*255
                    ctable = ctable.astype(np.uint8)
                    # Set opactities
                    if isinstance(opacity, np.ndarray) and not _custom_opac:
                        ctable[:, -1] = opacity
                    if flip_scalars:
                        ctable = np.ascontiguousarray(ctable[::-1])
                    table.SetTable(_vtk.numpy_to_vtk(ctable))
                    if _custom_opac:
                        # need to round the colors here since we're
                        # directly displaying the colors
                        hue = normalize(scalars, minimum=clim[0], maximum=clim[1])
                        scalars = np.round(hue*n_colors)/n_colors
                        scalars = cmap(scalars)*255
                        scalars[:, -1] *= opacity
                        scalars = scalars.astype(np.uint8)
                        prepare_mapper(scalars)

            else:  # no cmap specified
                if flip_scalars:
                    table.SetHueRange(0.0, 0.66667)
                else:
                    table.SetHueRange(0.66667, 0.0)
        else:
            self.mapper.SetScalarModeToUseFieldData()

        # Set actor properties ================================================

        # select view style
        if not style:
            style = 'surface'
        style = style.lower()
        if style == 'wireframe':
            prop.SetRepresentationToWireframe()
            if color is None:
                color = self._theme.outline_color
        elif style == 'points':
            prop.SetRepresentationToPoints()
        elif style == 'surface':
            prop.SetRepresentationToSurface()
        else:
            raise ValueError('Invalid style.  Must be one of the following:\n'
                             '\t"surface"\n'
                             '\t"wireframe"\n'
                             '\t"points"\n')

        prop.SetPointSize(point_size)
        prop.SetAmbient(ambient)
        prop.SetDiffuse(diffuse)
        prop.SetSpecular(specular)
        prop.SetSpecularPower(specular_power)

        if pbr:
            if not _vtk.VTK9:  # pragma: no cover
                raise RuntimeError('Physically based rendering requires VTK 9 '
                                   'or newer')
            prop.SetInterpolationToPBR()
            prop.SetMetallic(metallic)
            prop.SetRoughness(roughness)
        elif smooth_shading:
            prop.SetInterpolationToPhong()
        else:
            prop.SetInterpolationToFlat()
        # edge display style
        if show_edges:
            prop.EdgeVisibilityOn()

        rgb_color = parse_color(color, default_color=self._theme.color)
        prop.SetColor(rgb_color)
        if isinstance(opacity, (float, int)):
            prop.SetOpacity(opacity)
        prop.SetEdgeColor(parse_color(edge_color))

        if render_points_as_spheres:
            prop.SetRenderPointsAsSpheres(render_points_as_spheres)
        if render_lines_as_tubes:
            prop.SetRenderLinesAsTubes(render_lines_as_tubes)

        # legend label
        if label:
            if not isinstance(label, str):
                raise TypeError('Label must be a string')
            geom = pyvista.single_triangle()
            if scalars is not None:
                geom = pyvista.Box()
                rgb_color = parse_color('black')
            geom.points -= geom.center
            self._labels.append([geom, label, rgb_color])

        # lighting display style
        if not lighting:
            prop.LightingOff()

        # set line thickness
        if line_width:
            prop.SetLineWidth(line_width)

        self.add_actor(actor, reset_camera=reset_camera, name=name, culling=culling,
                       pickable=pickable, render=render)

        # hide scalar bar if using special scalars
        if scalar_bar_args.get('title') == '__custom_rgba':
            show_scalar_bar = False

        # Only show scalar bar if there are scalars
        if show_scalar_bar and scalars is not None:
            self.add_scalar_bar(**scalar_bar_args)

        self.renderer.Modified()

        return actor

    def add_volume(self, volume, scalars=None, clim=None, resolution=None,
                   opacity='linear', n_colors=256, cmap=None, flip_scalars=False,
                   reset_camera=None, name=None, ambient=0.0, categories=False,
                   culling=False, multi_colors=False,
                   blending='composite', mapper=None,
                   scalar_bar_args=None, show_scalar_bar=None,
                   annotations=None, pickable=True, preference="point",
                   opacity_unit_distance=None, shade=False,
                   diffuse=0.7, specular=0.2, specular_power=10.0,
                   render=True, **kwargs):
        """Add a volume, rendered using a smart mapper by default.

        Requires a 3D :class:`numpy.ndarray` or :class:`pyvista.UniformGrid`.

        Parameters
        ----------
        volume : 3D numpy.ndarray or pyvista.UniformGrid
            The input volume to visualize. 3D numpy arrays are accepted.

        scalars : str or numpy.ndarray, optional
            Scalars used to "color" the mesh.  Accepts a string name of an
            array that is present on the mesh or an array equal
            to the number of cells or the number of points in the
            mesh.  Array should be sized as a single vector. If ``scalars`` is
            ``None``, then the active scalars are used.

        clim : 2 item list, optional
            Color bar range for scalars.  Defaults to minimum and
            maximum of scalars array.  Example: ``[-1, 2]``. ``rng``
            is also an accepted alias for this.

        opacity : string or numpy.ndarray, optional
            Opacity mapping for the scalars array.
            A string can also be specified to map the scalars range to a
            predefined opacity transfer function (options include: 'linear',
            'linear_r', 'geom', 'geom_r'). Or you can pass a custom made
            transfer function that is an array either ``n_colors`` in length or
            shorter.

        n_colors : int, optional
            Number of colors to use when displaying scalars. Defaults to 256.
            The scalar bar will also have this many colors.

        cmap : str, optional
            Name of the Matplotlib colormap to us when mapping the ``scalars``.
            See available Matplotlib colormaps.  Only applicable for when
            displaying ``scalars``. Requires Matplotlib to be installed.
            ``colormap`` is also an accepted alias for this. If ``colorcet`` or
            ``cmocean`` are installed, their colormaps can be specified by name.

        flip_scalars : bool, optional
            Flip direction of cmap. Most colormaps allow ``*_r`` suffix to do
            this as well.

        reset_camera : bool, optional
            Reset the camera after adding this mesh to the scene

        name : str, optional
            The name for the added actor so that it can be easily
            updated.  If an actor of this name already exists in the
            rendering window, it will be replaced by the new actor.

        ambient : float, optional
            When lighting is enabled, this is the amount of light from
            0 to 1 that reaches the actor when not directed at the
            light source emitted from the viewer.  Default 0.0.

        culling : str, optional
            Does not render faces that are culled. Options are ``'front'`` or
            ``'back'``. This can be helpful for dense surface meshes,
            especially when edges are visible, but can cause flat
            meshes to be partially displayed.  Defaults ``False``.

        categories : bool, optional
            If set to ``True``, then the number of unique values in the scalar
            array will be used as the ``n_colors`` argument.

        multi_colors : bool, optional
            Whether or not to use multiple colors when plotting MultiBlock
            object. Blocks will be colored sequentially as 'Reds', 'Greens',
            'Blues', and 'Grays'.

        blending : str, optional
            Blending mode for visualisation of the input object(s). Can be
            one of 'additive', 'maximum', 'minimum', 'composite', or
            'average'. Defaults to 'additive'.

        mapper : str, optional
            Volume mapper to use given by name. Options include:
            ``'fixed_point'``, ``'gpu'``, ``'open_gl'``, and
            ``'smart'``.  If ``None`` the ``"volume_mapper"`` in the
            ``self._theme`` is used.

        scalar_bar_args : dict, optional
            Dictionary of keyword arguments to pass when adding the
            scalar bar to the scene. For options, see
            :func:`pyvista.BasePlotter.add_scalar_bar`.

        show_scalar_bar : bool
            If ``False``, a scalar bar will not be added to the
            scene. Defaults to ``True``.

        annotations : dict, optional
            Pass a dictionary of annotations. Keys are the float
            values in the scalars range to annotate on the scalar bar
            and the values are the the string annotations.

        opacity_unit_distance : float
            Set/Get the unit distance on which the scalar opacity
            transfer function is defined. Meaning that over that
            distance, a given opacity (from the transfer function) is
            accumulated. This is adjusted for the actual sampling
            distance during rendering. By default, this is the length
            of the diagonal of the bounding box of the volume divided
            by the dimensions.

        shade : bool
            Default off. If shading is turned on, the mapper may
            perform shading calculations - in some cases shading does
            not apply (for example, in a maximum intensity projection)
            and therefore shading will not be performed even if this
            flag is on.

        diffuse : float, optional
            The diffuse lighting coefficient. Default 1.0

        specular : float, optional
            The specular lighting coefficient. Default 0.0

        specular_power : float, optional
            The specular power. Between 0.0 and 128.0

        render : bool, optional
            Force a render when True.  Default ``True``.

        Returns
        -------
        actor: vtk.vtkVolume
            VTK volume of the input data.

        """
        # Handle default arguments

        # Supported aliases
        clim = kwargs.pop('rng', clim)
        cmap = kwargs.pop('colormap', cmap)
        culling = kwargs.pop("backface_culling", culling)

        if "scalar" in kwargs:
            raise TypeError("`scalar` is an invalid keyword argument for `add_mesh`. Perhaps you mean `scalars` with an s?")
        assert_empty_kwargs(**kwargs)

        # Avoid mutating input
        if scalar_bar_args is None:
            scalar_bar_args = {}
        else:
            scalar_bar_args = scalar_bar_args.copy()
        # account for legacy behavior
        if 'stitle' in kwargs:  # pragma: no cover
            warnings.warn(USE_SCALAR_BAR_ARGS, PyvistaDeprecationWarning)
            scalar_bar_args.setdefault('title', kwargs.pop('stitle'))

        if show_scalar_bar is None:
            show_scalar_bar = self._theme.show_scalar_bar

        if culling is True:
            culling = 'backface'

        if mapper is None:
            mapper = self._theme.volume_mapper

        # only render when the plotter has already been shown
        if render is None:
            render = not self._first_time

        # Convert the VTK data object to a pyvista wrapped object if necessary
        if not is_pyvista_dataset(volume):
            if isinstance(volume, np.ndarray):
                volume = wrap(volume)
                if resolution is None:
                    resolution = [1,1,1]
                elif len(resolution) != 3:
                    raise ValueError('Invalid resolution dimensions.')
                volume.spacing = resolution
            else:
                volume = wrap(volume)
                if not is_pyvista_dataset(volume):
                    raise TypeError(f'Object type ({type(volume)}) not supported for plotting in PyVista.')
        else:
            # HACK: Make a copy so the original object is not altered.
            #       Also, place all data on the nodes as issues arise when
            #       volume rendering on the cells.
            volume = volume.cell_data_to_point_data()

        if name is None:
            name = f'{type(volume).__name__}({volume.memory_address})'

        if isinstance(volume, pyvista.MultiBlock):
            from itertools import cycle
            cycler = cycle(['Reds', 'Greens', 'Blues', 'Greys', 'Oranges', 'Purples'])
            # Now iteratively plot each element of the multiblock dataset
            actors = []
            for idx in range(volume.GetNumberOfBlocks()):
                if volume[idx] is None:
                    continue
                # Get a good name to use
                next_name = f'{name}-{idx}'
                # Get the data object
                block = wrap(volume.GetBlock(idx))
                if resolution is None:
                    try:
                        block_resolution = block.GetSpacing()
                    except AttributeError:
                        block_resolution = resolution
                else:
                    block_resolution = resolution
                if multi_colors:
                    color = next(cycler)
                else:
                    color = cmap

                a = self.add_volume(block, resolution=block_resolution, opacity=opacity,
                                    n_colors=n_colors, cmap=color, flip_scalars=flip_scalars,
                                    reset_camera=reset_camera, name=next_name,
                                    ambient=ambient, categories=categories,
                                    culling=culling, clim=clim,
                                    mapper=mapper, pickable=pickable,
                                    opacity_unit_distance=opacity_unit_distance,
                                    shade=shade, diffuse=diffuse, specular=specular,
                                    specular_power=specular_power, render=render)

                actors.append(a)
            return actors

        if not isinstance(volume, pyvista.UniformGrid):
            raise TypeError(f'Type {type(volume)} not supported for volume rendering at this time. Use `pyvista.UniformGrid`.')

        if opacity_unit_distance is None:
            opacity_unit_distance = volume.length / (np.mean(volume.dimensions) - 1)

        if scalars is None:
            # Make sure scalars components are not vectors/tuples
            scalars = volume.active_scalars
            # Don't allow plotting of string arrays by default
            if scalars is not None and np.issubdtype(scalars.dtype, np.number):
                scalar_bar_args.setdefault('title', volume.active_scalars_info[1])
            else:
                raise ValueError('No scalars to use for volume rendering.')
        elif isinstance(scalars, str):
            pass

        ##############

        title = 'Data'
        if isinstance(scalars, str):
            title = scalars
            scalars = get_array(volume, scalars,
                                preference=preference, err=True)
            scalar_bar_args.setdefault('title', title)

        if not isinstance(scalars, np.ndarray):
            scalars = np.asarray(scalars)

        if not np.issubdtype(scalars.dtype, np.number):
            raise TypeError('Non-numeric scalars are currently not supported for volume rendering.')

        if scalars.ndim != 1:
            scalars = scalars.ravel()

        if scalars.dtype == np.bool_ or scalars.dtype == np.uint8:
            scalars = scalars.astype(np.float_)

        # Define mapper, volume, and add the correct properties
        mappers = {
            'fixed_point': _vtk.vtkFixedPointVolumeRayCastMapper,
            'gpu': _vtk.vtkGPUVolumeRayCastMapper,
            'open_gl': _vtk.vtkOpenGLGPUVolumeRayCastMapper,
            'smart': _vtk.vtkSmartVolumeMapper,
        }
        if not isinstance(mapper, str) or mapper not in mappers.keys():
            raise TypeError(f"Mapper ({mapper}) unknown. Available volume mappers include: {', '.join(mappers.keys())}")
        self.mapper = make_mapper(mappers[mapper])

        # Scalars interpolation approach
        if scalars.shape[0] == volume.n_points:
            volume.point_arrays.append(scalars, title, True)
            self.mapper.SetScalarModeToUsePointData()
        elif scalars.shape[0] == volume.n_cells:
            volume.cell_arrays.append(scalars, title, True)
            self.mapper.SetScalarModeToUseCellData()
        else:
            raise_not_matching(scalars, volume)

        # Set scalars range
        if clim is None:
            clim = [np.nanmin(scalars), np.nanmax(scalars)]
        elif isinstance(clim, float) or isinstance(clim, int):
            clim = [-clim, clim]

        ###############

        scalars = scalars.astype(np.float_)
        with np.errstate(invalid='ignore'):
            idxs0 = scalars < clim[0]
            idxs1 = scalars > clim[1]
        scalars[idxs0] = clim[0]
        scalars[idxs1] = clim[1]
        scalars = ((scalars - np.nanmin(scalars)) / (np.nanmax(scalars) - np.nanmin(scalars))) * 255
        # scalars = scalars.astype(np.uint8)
        volume[title] = scalars

        self.mapper.scalar_range = clim

        # Set colormap and build lookup table
        table = _vtk.vtkLookupTable()
        # table.SetNanColor(nan_color) # NaN's are chopped out with current implementation
        # above/below colors not supported with volume rendering

        if isinstance(annotations, dict):
            for val, anno in annotations.items():
                table.SetAnnotation(float(val), str(anno))

        if cmap is None:  # Set default map if matplotlib is available
            if _has_matplotlib():
                cmap = self._theme.cmap

        if cmap is not None:
            if not _has_matplotlib():
                raise ImportError('Please install matplotlib for volume rendering.')

            cmap = get_cmap_safe(cmap)
            if categories:
                if categories is True:
                    n_colors = len(np.unique(scalars))
                elif isinstance(categories, int):
                    n_colors = categories
        if flip_scalars:
            cmap = cmap.reversed()

        color_tf = _vtk.vtkColorTransferFunction()
        for ii in range(n_colors):
            color_tf.AddRGBPoint(ii, *cmap(ii)[:-1])

        # Set opacities
        if isinstance(opacity, (float, int)):
            opacity_values = [opacity] * n_colors
        elif isinstance(opacity, str):
            opacity_values = pyvista.opacity_transfer_function(opacity, n_colors)
        elif isinstance(opacity, (np.ndarray, list, tuple)):
            opacity = np.array(opacity)
            opacity_values = opacity_transfer_function(opacity, n_colors)

        opacity_tf = _vtk.vtkPiecewiseFunction()
        for ii in range(n_colors):
            opacity_tf.AddPoint(ii, opacity_values[ii] / n_colors)

        # Now put color tf and opacity tf into a lookup table for the scalar bar
        table.SetNumberOfTableValues(n_colors)
        lut = cmap(np.array(range(n_colors))) * 255
        lut[:,3] = opacity_values
        lut = lut.astype(np.uint8)
        table.SetTable(_vtk.numpy_to_vtk(lut))
        table.SetRange(*clim)
        self.mapper.lookup_table = table

        self.mapper.SetInputData(volume)

        blending = blending.lower()
        if blending in ['additive', 'add', 'sum']:
            self.mapper.SetBlendModeToAdditive()
        elif blending in ['average', 'avg', 'average_intensity']:
            self.mapper.SetBlendModeToAverageIntensity()
        elif blending in ['composite', 'comp']:
            self.mapper.SetBlendModeToComposite()
        elif blending in ['maximum', 'max', 'maximum_intensity']:
            self.mapper.SetBlendModeToMaximumIntensity()
        elif blending in ['minimum', 'min', 'minimum_intensity']:
            self.mapper.SetBlendModeToMinimumIntensity()
        else:
            raise ValueError(f'Blending mode \'{blending}\' invalid. ' +
                             'Please choose one ' + 'of \'additive\', '
                             '\'composite\', \'minimum\' or ' + '\'maximum\'.')
        self.mapper.Update()

        self.volume = _vtk.vtkVolume()
        self.volume.SetMapper(self.mapper)

        prop = _vtk.vtkVolumeProperty()
        prop.SetColor(color_tf)
        prop.SetScalarOpacity(opacity_tf)
        prop.SetAmbient(ambient)
        prop.SetScalarOpacityUnitDistance(opacity_unit_distance)
        prop.SetShade(shade)
        prop.SetDiffuse(diffuse)
        prop.SetSpecular(specular)
        prop.SetSpecularPower(specular_power)
        self.volume.SetProperty(prop)

        actor, prop = self.add_actor(self.volume, reset_camera=reset_camera,
                                     name=name, culling=culling,
                                     pickable=pickable, render=render)

        # Add scalar bar if scalars are available
        if show_scalar_bar and scalars is not None:
            self.add_scalar_bar(**scalar_bar_args)

        self.renderer.Modified()

        return actor

    def update_scalar_bar_range(self, clim, name=None):
        """Update the value range of the active or named scalar bar.

        Parameters
        ----------
        2 item list
            The new range of scalar bar. Example: ``[-1, 2]``.

        name : str, optional
            The title of the scalar bar to update

        """
        if isinstance(clim, float) or isinstance(clim, int):
            clim = [-clim, clim]
        if len(clim) != 2:
            raise TypeError('clim argument must be a length 2 iterable of values: (min, max).')
        if name is None:
            if not hasattr(self, 'mapper'):
                raise AttributeError('This plotter does not have an active mapper.')
            self.mapper.scalar_range = clim
            return

        # Use the name to find the desired actor
        def update_mapper(mapper_helper):
            mapper_helper.scalar_range = clim
            return

        try:
            for mh in self._scalar_bar_mappers[name]:
                update_mapper(mh)
        except KeyError:
            raise KeyError('Name ({}) not valid/not found in this plotter.')
        return

    def clear(self):
        """Clear plot by removing all actors and properties."""
        self.renderers.clear()
        self.scalar_bars.clear()
        self.mesh = None

    def link_views(self, views=0):
        """Link the views' cameras.

        Parameters
        ----------
        views : int | tuple or list
            If ``views`` is int, link the views to the given view
            index or if ``views`` is a tuple or a list, link the given
            views cameras.

        """
        if isinstance(views, (int, np.integer)):
            for renderer in self.renderers:
                renderer.camera = self.renderers[views].camera
            return
        views = np.asarray(views)
        if np.issubdtype(views.dtype, np.integer):
            for view_index in views:
                self.renderers[view_index].camera = \
                    self.renderers[views[0]].camera
        else:
            raise TypeError('Expected type is int, list or tuple:'
                            f'{type(views)} is given')

    def unlink_views(self, views=None):
        """Unlink the views' cameras.

        Parameters
        ----------
        views : None | int | tuple or list
            If ``views`` is None unlink all the views, if ``views``
            is int unlink the selected view's camera or if ``views``
            is a tuple or a list, unlink the given views cameras.

        """
        if views is None:
            for renderer in self.renderers:
                renderer.camera = Camera()
                renderer.reset_camera()
        elif isinstance(views, int):
            self.renderers[views].camera = Camera()
            self.renderers[views].reset_camera()
        elif isinstance(views, collections.abc.Iterable):
            for view_index in views:
                self.renderers[view_index].camera = Camera()
                self.renderers[view_index].reset_camera()
        else:
            raise TypeError('Expected type is None, int, list or tuple:'
                            f'{type(views)} is given')

    @wraps(ScalarBars.add_scalar_bar)
    def add_scalar_bar(self, *args, **kwargs):
        """Wrap for ``ScalarBars.add_scalar_bar``."""
        # only render when the plotter has already been shown
        render = kwargs.get('render', None)
        if render is None:
            kwargs['render'] = not self._first_time

        # check if maper exists
        mapper = kwargs.get('mapper', None)
        if mapper is None:
            if not hasattr(self, 'mapper') or self.mapper is None:
                raise AttributeError('Mapper does not exist.  '
                                     'Add a mesh with scalars first.')
            kwargs['mapper'] = self.mapper

        # title can be the first and only arg
        if len(args):
            title = args[0]
        else:
            title = kwargs.get('title', '')
        if title is None:
            title = ''
        kwargs['title'] = title

        interactive = kwargs.get('interactive', None)
        if interactive is None:
            interactive = self._theme.interactive
            if self.shape != (1, 1):
                interactive = False
        elif interactive and self.shape != (1, 1):
            raise ValueError('Interactive scalar bars disabled for multi-renderer plots')

        return self.scalar_bars.add_scalar_bar(**kwargs)

    def update_scalars(self, scalars, mesh=None, render=True):
        """Update scalars of an object in the plotter.

        Parameters
        ----------
        scalars : np.ndarray
            Scalars to replace existing scalars.

        mesh : vtk.PolyData or vtk.UnstructuredGrid, optional
            Object that has already been added to the Plotter.  If
            None, uses last added mesh.

        render : bool, optional
            Force a render when True.  Default ``True``.
        """
        if mesh is None:
            mesh = self.mesh

        if isinstance(mesh, (collections.abc.Iterable, pyvista.MultiBlock)):
            # Recursive if need to update scalars on many meshes
            for m in mesh:
                self.update_scalars(scalars, mesh=m, render=False)
            if render:
                self.render()
            return

        if isinstance(scalars, str):
            # Grab scalars array if name given
            scalars = get_array(mesh, scalars)

        if scalars is None:
            if render:
                self.render()
            return

        if scalars.shape[0] == mesh.GetNumberOfPoints():
            data = mesh.GetPointData()
        elif scalars.shape[0] == mesh.GetNumberOfCells():
            data = mesh.GetCellData()
        else:
            raise_not_matching(scalars, mesh)

        vtk_scalars = data.GetScalars()
        if vtk_scalars is None:
            raise ValueError('No active scalars')
        s = convert_array(vtk_scalars)
        s[:] = scalars
        data.Modified()
        try:
            # Why are the points updated here? Not all datasets have points
            # and only the scalars array is modified by this function...
            mesh.GetPoints().Modified()
        except:
            pass

        if render:
            self.render()

    def update_coordinates(self, points, mesh=None, render=True):
        """Update the points of an object in the plotter.

        Parameters
        ----------
        points : np.ndarray
            Points to replace existing points.

        mesh : vtk.PolyData or vtk.UnstructuredGrid, optional
            Object that has already been added to the Plotter.  If
            None, uses last added mesh.

        render : bool, optional
            Force a render when True.  Default ``True``.
        """
        if mesh is None:
            mesh = self.mesh

        mesh.points = points

        # only render when the plotter has already been shown
        if render is None:
            render = not self._first_time

        if render:
            self.render()

    def _clear_ren_win(self):
        """Clear the render window."""
        if hasattr(self, 'ren_win'):
            self.ren_win.Finalize()
            del self.ren_win

    def close(self, render=False):
        """Close the render window."""
        # optionally run just prior to exiting the plotter
        if self._before_close_callback is not None:
            self._before_close_callback(self)
            self._before_close_callback = None

        # must close out widgets first
        super().close()
        # Renderer has an axes widget, so close it
        self.renderers.close()
        self.renderers.remove_all_lights()

        # Grab screenshots of last render
        if self._store_image:
            self.last_image = self.screenshot(None, return_img=True)
            self.last_image_depth = self.get_image_depth()

        # reset scalar bars
        self.clear()

        self._clear_ren_win()

        if self.iren is not None:
            self.iren.remove_observers()
            self.iren.terminate_app()
            self.iren = None

        if hasattr(self, 'textActor'):
            del self.textActor

        # end movie
        if hasattr(self, 'mwriter'):
            try:
                self.mwriter.close()
            except BaseException:
                pass

        # this helps managing closed plotters
        self._closed = True

    def deep_clean(self):
        """Clean the plotter of the memory."""
        if hasattr(self, 'renderers'):
            self.renderers.deep_clean()
        if getattr(self, 'mesh', None) is not None:
            self.mesh.point_arrays = None
            self.mesh.cell_arrays = None
        self.mesh = None
        if getattr(self, 'mapper', None) is not None:
            self.mapper.lookup_table = None
        self.mapper = None
        self.volume = None
        self.textactor = None

    def add_text(self, text, position='upper_left', font_size=18, color=None,
                 font=None, shadow=False, name=None, viewport=False):
        """Add text to plot object in the top left corner by default.

        Parameters
        ----------
        text : str
            The text to add the rendering

        position : str, tuple(float)
            Position to place the bottom left corner of the text box.
            If tuple is used, the position of the text uses the pixel
            coordinate system (default). In this case,
            it returns a more general `vtkOpenGLTextActor`.
            If string name is used, it returns a `vtkCornerAnnotation`
            object normally used for fixed labels (like title or xlabel).
            Default is to find the top left corner of the rendering window
            and place text box up there. Available position: ``'lower_left'``,
            ``'lower_right'``, ``'upper_left'``, ``'upper_right'``,
            ``'lower_edge'``, ``'upper_edge'``, ``'right_edge'``, and
            ``'left_edge'``

        font : string, optional
            Font name may be courier, times, or arial

        shadow : bool, optional
            Adds a black shadow to the text.  Defaults to False

        name : str, optional
            The name for the added actor so that it can be easily updated.
            If an actor of this name already exists in the rendering window, it
            will be replaced by the new actor.

        viewport: bool
            If True and position is a tuple of float, uses
            the normalized viewport coordinate system (values between 0.0
            and 1.0 and support for HiDPI).

        Returns
        -------
        textActor : vtk.vtkTextActor
            Text actor added to plot

        """
        if font is None:
            font = self._theme.font.family
        if font_size is None:
            font_size = self._theme.font.size
        if color is None:
            color = self._theme.font.color
        if position is None:
            # Set the position of the text to the top left corner
            window_size = self.window_size
            x = (window_size[0] * 0.02) / self.shape[0]
            y = (window_size[1] * 0.85) / self.shape[0]
            position = [x, y]

        corner_mappings = {
            'lower_left': _vtk.vtkCornerAnnotation.LowerLeft,
            'lower_right': _vtk.vtkCornerAnnotation.LowerRight,
            'upper_left': _vtk.vtkCornerAnnotation.UpperLeft,
            'upper_right': _vtk.vtkCornerAnnotation.UpperRight,
            'lower_edge': _vtk.vtkCornerAnnotation.LowerEdge,
            'upper_edge': _vtk.vtkCornerAnnotation.UpperEdge,
            'left_edge': _vtk.vtkCornerAnnotation.LeftEdge,
            'right_edge': _vtk.vtkCornerAnnotation.RightEdge,

        }
        corner_mappings['ll'] = corner_mappings['lower_left']
        corner_mappings['lr'] = corner_mappings['lower_right']
        corner_mappings['ul'] = corner_mappings['upper_left']
        corner_mappings['ur'] = corner_mappings['upper_right']
        corner_mappings['top'] = corner_mappings['upper_edge']
        corner_mappings['bottom'] = corner_mappings['lower_edge']
        corner_mappings['right'] = corner_mappings['right_edge']
        corner_mappings['r'] = corner_mappings['right_edge']
        corner_mappings['left'] = corner_mappings['left_edge']
        corner_mappings['l'] = corner_mappings['left_edge']

        if isinstance(position, (int, str, bool)):
            if isinstance(position, str):
                position = corner_mappings[position]
            elif position is True:
                position = corner_mappings['upper_left']
            self.textActor = _vtk.vtkCornerAnnotation()
            # This is how you set the font size with this actor
            self.textActor.SetLinearFontScaleFactor(font_size // 2)
            self.textActor.SetText(position, text)
        else:
            self.textActor = _vtk.vtkTextActor()
            self.textActor.SetInput(text)
            self.textActor.SetPosition(position)
            if viewport:
                self.textActor.GetActualPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
                self.textActor.GetActualPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
            self.textActor.GetTextProperty().SetFontSize(int(font_size * 2))

        self.textActor.GetTextProperty().SetColor(parse_color(color))
        self.textActor.GetTextProperty().SetFontFamily(FONTS[font].value)
        self.textActor.GetTextProperty().SetShadow(shadow)

        self.add_actor(self.textActor, reset_camera=False, name=name, pickable=False)
        return self.textActor

    def open_movie(self, filename, framerate=24):
        """Establish a connection to the ffmpeg writer.

        Parameters
        ----------
        filename : str
            Filename of the movie to open.  Filename should end in mp4,
            but other filetypes may be supported.  See "imagio.get_writer"

        framerate : int, optional
            Frames per second.

        """
        from imageio import get_writer
        if isinstance(pyvista.FIGURE_PATH, str) and not os.path.isabs(filename):
            filename = os.path.join(pyvista.FIGURE_PATH, filename)
        self.mwriter = get_writer(filename, fps=framerate)

    def open_gif(self, filename):
        """Open a gif file.

        Parameters
        ----------
        filename : str
            Filename of the gif to open.  Filename must end in gif.

        """
        from imageio import get_writer
        if filename[-3:] != 'gif':
            raise ValueError('Unsupported filetype.  Must end in .gif')
        if isinstance(pyvista.FIGURE_PATH, str) and not os.path.isabs(filename):
            filename = os.path.join(pyvista.FIGURE_PATH, filename)
        self._gif_filename = os.path.abspath(filename)
        self.mwriter = get_writer(filename, mode='I')

    def write_frame(self):
        """Write a single frame to the movie file."""
        # if off screen, show has not been called and we must render
        # before extracting an image
        if self._first_time:
            self._on_first_render_request()
            self.render()

        if not hasattr(self, 'mwriter'):
            raise RuntimeError('This plotter has not opened a movie or GIF file.')
        self.update()
        self.mwriter.append_data(self.image)

    def _run_image_filter(self, ifilter):
        # Update filter and grab pixels
        ifilter.Modified()
        ifilter.Update()
        image = pyvista.wrap(ifilter.GetOutput())
        img_size = image.dimensions
        img_array = pyvista.utilities.point_array(image, 'ImageScalars')

        # Reshape and write
        tgt_size = (img_size[1], img_size[0], -1)
        return img_array.reshape(tgt_size)[::-1]

    def get_image_depth(self,
                        fill_value=np.nan,
                        reset_camera_clipping_range=True):
        """Return a depth image representing current render window.

        Parameters
        ----------
        fill_value : float, optional
            Fill value for points in image that do not include objects
            in scene.  To not use a fill value, pass ``None``.

        reset_camera_clipping_range : bool, optional
            Reset the camera clipping range to include data in view.

        Returns
        -------
        image_depth : numpy.ndarray
            Image of depth values from camera orthogonal to image
            plane.

        Examples
        --------
        >>> import pyvista
        >>> plotter = pyvista.Plotter()
        >>> actor = plotter.add_mesh(pyvista.Sphere())
        >>> plotter.store_image = True
        >>> cpos = plotter.show()
        >>> zval = plotter.get_image_depth()

        Notes
        -----
        Values in image_depth are negative to adhere to a
        right-handed coordinate system.

        """
        # allow no render window
        if not hasattr(self, 'ren_win') and self.last_image_depth is not None:
            zval = self.last_image_depth.copy()
            if fill_value is not None:
                zval[self._image_depth_null] = fill_value
            return zval

        self._check_rendered()
        self._check_has_ren_win()

        # Ensure points in view are within clipping range of renderer?
        if reset_camera_clipping_range:
            self.renderer.ResetCameraClippingRange()

        # Get the z-buffer image
        ifilter = _vtk.vtkWindowToImageFilter()
        ifilter.SetInput(self.ren_win)
        ifilter.ReadFrontBufferOff()
        ifilter.SetInputBufferTypeToZBuffer()
        zbuff = self._run_image_filter(ifilter)[:, :, 0]

        # Convert z-buffer values to depth from camera
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            near, far = self.camera.clipping_range
            if self.camera.is_parallel_projection:
                zval = (zbuff - near) / (far - near)
            else:
                zval = 2 * near * far / ((zbuff - 0.5) * 2 * (far - near) - near - far)

            # Consider image values outside clipping range as nans
            self._image_depth_null = np.logical_or(zval < -far, np.isclose(zval, -far))

        if fill_value is not None:
            zval[self._image_depth_null] = fill_value

        return zval

    def add_lines(self, lines, color=(1, 1, 1), width=5, label=None, name=None):
        """Add lines to the plotting object.

        Parameters
        ----------
        lines : np.ndarray or pyvista.PolyData
            Points representing line segments.  For example, two line segments
            would be represented as:

            np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]])

        color : string or 3 item list, optional, defaults to white
            Either a string, rgb list, or hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1, 1, 1]``
            * ``color='#FFFFFF'``

        width : float, optional
            Thickness of lines

        name : str, optional
            The name for the added actor so that it can be easily updated.
            If an actor of this name already exists in the rendering window, it
            will be replaced by the new actor.

        Returns
        -------
        actor : vtk.vtkActor
            Lines actor.

        """
        if not isinstance(lines, np.ndarray):
            raise TypeError('Input should be an array of point segments')

        lines = pyvista.lines_from_points(lines)

        # Create mapper and add lines
        mapper = _vtk.vtkDataSetMapper()
        mapper.SetInputData(lines)

        rgb_color = parse_color(color)

        # legend label
        if label:
            if not isinstance(label, str):
                raise TypeError('Label must be a string')
            self._labels.append([lines, label, rgb_color])

        # Create actor
        actor = _vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(width)
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(rgb_color)
        actor.GetProperty().SetColor(rgb_color)
        actor.GetProperty().LightingOff()

        # Add to renderer
        self.add_actor(actor, reset_camera=False, name=name, pickable=False)
        return actor

    @wraps(ScalarBars.remove_scalar_bar)
    def remove_scalar_bar(self, *args, **kwargs):
        """Remove the active scalar bar."""
        self.scalar_bars.remove_scalar_bar(*args, **kwargs)

    def add_point_labels(self, points, labels, italic=False, bold=True,
                         font_size=None, text_color=None,
                         font_family=None, shadow=False,
                         show_points=True, point_color=None, point_size=5,
                         name=None, shape_color='grey', shape='rounded_rect',
                         fill_shape=True, margin=3, shape_opacity=1.0,
                         pickable=False, render_points_as_spheres=False,
                         tolerance=0.001, reset_camera=None, always_visible=False,
                         render=True):
        """Create a point actor with one label from list labels assigned to each point.

        Parameters
        ----------
        points : np.ndarray or pyvista.DataSet
            n x 3 numpy array of points or pyvista dataset with points

        labels : list or str
            List of labels.  Must be the same length as points. If a string name
            is given with a pyvista.DataSet input for points, then these are fetched.

        italic : bool, optional
            Italicises title and bar labels.  Default False.

        bold : bool, optional
            Bolds title and bar labels.  Default True

        font_size : float, optional
            Sets the size of the title font.  Defaults to 16.

        text_color : string or 3 item list, optional
            Color of text. Either a string, rgb list, or hex color string.

                text_color='white'
                text_color='w'
                text_color=[1, 1, 1]
                text_color='#FFFFFF'

        font_family : string, optional
            Font family.  Must be either courier, times, or arial.

        shadow : bool, optional
            Adds a black shadow to the text.  Defaults to False

        show_points : bool, optional
            Controls if points are visible.  Default True

        point_color : string or 3 item list, optional. Color of points (if visible).
            Either a string, rgb list, or hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1, 1, 1]``
            * ``color='#FFFFFF'``

        point_size : float, optional
            Size of points (if visible)

        name : str, optional
            The name for the added actor so that it can be easily updated.
            If an actor of this name already exists in the rendering window, it
            will be replaced by the new actor.

        shape_color : string or 3 item list, optional. Color of points (if visible).
            Either a string, rgb list, or hex color string.  For example:

        shape : str, optional
            The string name of the shape to use. Options are ``'rect'`` or
            ``'rounded_rect'``. If you want no shape, pass ``None``

        fill_shape : bool, optional
            Fill the shape with the ``shape_color``. Outlines if ``False``.

        margin : int, optional
            The size of the margin on the label background shape. Default is 3.

        shape_opacity : float
            The opacity of the shape between zero and one.

        tolerance : float
            a tolerance to use to determine whether a point label is visible.
            A tolerance is usually required because the conversion from world
            space to display space during rendering introduces numerical
            round-off.

        reset_camera : bool, optional
            Reset the camera after adding the points to the scene.

        always_visible : bool, optional
            Skip adding the visibility filter. Default False.

        render : bool, optional
            Force a render when ``True`` (default).

        Returns
        -------
        labelActor : vtk.vtkActor2D
            VTK label actor.  Can be used to change properties of the labels.

        """
        if font_family is None:
            font_family = self._theme.font.family
        if font_size is None:
            font_size = self._theme.font.size
        if point_color is None:
            point_color = self._theme.color
        if text_color is None:
            text_color = self._theme.font.color

        if isinstance(points, (list, tuple)):
            points = np.array(points)

        if isinstance(points, np.ndarray):
            vtkpoints = pyvista.PolyData(points) # Cast to poly data
        elif is_pyvista_dataset(points):
            vtkpoints = pyvista.PolyData(points.points)
            if isinstance(labels, str):
                labels = points.point_arrays[labels]
        else:
            raise TypeError(f'Points type not usable: {type(points)}')

        if len(vtkpoints.points) != len(labels):
            raise ValueError('There must be one label for each point')

        if name is None:
            name = f'{type(vtkpoints).__name__}({vtkpoints.memory_address})'

        vtklabels = _vtk.vtkStringArray()
        vtklabels.SetName('labels')
        for item in labels:
            vtklabels.InsertNextValue(str(item))
        vtkpoints.GetPointData().AddArray(vtklabels)

        # Create hierarchy
        hier = _vtk.vtkPointSetToLabelHierarchy()
        hier.SetLabelArrayName('labels')

        if always_visible:
            hier.SetInputData(vtkpoints)
        else:
            # Only show visible points
            vis_points = _vtk.vtkSelectVisiblePoints()
            vis_points.SetInputData(vtkpoints)
            vis_points.SetRenderer(self.renderer)
            vis_points.SetTolerance(tolerance)

            hier.SetInputConnection(vis_points.GetOutputPort())

        # create label mapper
        labelMapper = _vtk.vtkLabelPlacementMapper()
        labelMapper.SetInputConnection(hier.GetOutputPort())
        if not isinstance(shape, str):
            labelMapper.SetShapeToNone()
        elif shape.lower() in 'rect':
            labelMapper.SetShapeToRect()
        elif shape.lower() in 'rounded_rect':
            labelMapper.SetShapeToRoundedRect()
        else:
            raise ValueError(f'Shape ({shape}) not understood')
        if fill_shape:
            labelMapper.SetStyleToFilled()
        else:
            labelMapper.SetStyleToOutline()
        labelMapper.SetBackgroundColor(parse_color(shape_color))
        labelMapper.SetBackgroundOpacity(shape_opacity)
        labelMapper.SetMargin(margin)

        textprop = hier.GetTextProperty()
        textprop.SetItalic(italic)
        textprop.SetBold(bold)
        textprop.SetFontSize(font_size)
        textprop.SetFontFamily(parse_font_family(font_family))
        textprop.SetColor(parse_color(text_color))
        textprop.SetShadow(shadow)

        self.remove_actor(f'{name}-points', reset_camera=False)
        self.remove_actor(f'{name}-labels', reset_camera=False)

        # add points
        if show_points:
            self.add_mesh(vtkpoints, color=point_color, point_size=point_size,
                          name=f'{name}-points', pickable=pickable,
                          render_points_as_spheres=render_points_as_spheres,
                          reset_camera=reset_camera, render=render)

        labelActor = _vtk.vtkActor2D()
        labelActor.SetMapper(labelMapper)
        self.add_actor(labelActor, reset_camera=False,
                       name=f'{name}-labels', pickable=False)

        return labelActor

    def add_point_scalar_labels(self, points, labels, fmt=None, preamble='', **kwargs):
        """Label the points from a dataset with the values of their scalars.

        Wrapper for :func:`pyvista.BasePlotter.add_point_labels`.

        Parameters
        ----------
        points : np.ndarray or pyvista.DataSet
            n x 3 numpy array of points or pyvista dataset with points

        labels : str
            String name of the point data array to use.

        fmt : str
            String formatter used to format numerical data.

        """
        if not is_pyvista_dataset(points):
            raise TypeError(f'input points must be a pyvista dataset, not: {type(points)}')
        if not isinstance(labels, str):
            raise TypeError('labels must be a string name of the scalars array to use')
        if fmt is None:
            fmt = self._theme.font.fmt
        if fmt is None:
            fmt = '%.6e'
        scalars = points.point_arrays[labels]
        phrase = f'{preamble} %.3e'
        labels = [phrase % val for val in scalars]
        return self.add_point_labels(points, labels, **kwargs)

    def add_points(self, points, **kwargs):
        """Add points to a mesh."""
        kwargs['style'] = 'points'
        return self.add_mesh(points, **kwargs)

    def add_arrows(self, cent, direction, mag=1, **kwargs):
        """Add arrows to the plotter.

        Parameters
        ----------
        cent : np.ndarray
            Array of centers.

        direction : np.ndarray
            Array of direction vectors.

        mag : float, optional
            Amount to scale the direction vectors.

        Examples
        --------
        Plot a random field of vectors and save a screenshot of it.

        >>> import numpy as np
        >>> import pyvista
        >>> cent = np.random.random((10, 3))
        >>> direction = np.random.random((10, 3))
        >>> plotter = pyvista.Plotter()
        >>> _ = plotter.add_arrows(cent, direction, mag=2)
        >>> cpos = plotter.show()

        """
        if cent.shape != direction.shape:  # pragma: no cover
            raise ValueError('center and direction arrays must have the same shape')

        direction = direction.copy()
        if cent.ndim != 2:
            cent = cent.reshape((-1, 3))

        if direction.ndim != 2:
            direction = direction.reshape((-1, 3))

        if mag != 1:
            direction = direction*mag

        pdata = pyvista.vector_poly_data(cent, direction)
        # Create arrow object
        arrow = _vtk.vtkArrowSource()
        arrow.Update()
        glyph3D = _vtk.vtkGlyph3D()
        glyph3D.SetSourceData(arrow.GetOutput())
        glyph3D.SetInputData(pdata)
        glyph3D.SetVectorModeToUseVector()
        glyph3D.Update()

        arrows = wrap(glyph3D.GetOutput())
        return self.add_mesh(arrows, **kwargs)

    @staticmethod
    def _save_image(image, filename, return_img):
        """Save to file and/or return a NumPy image array.

        This is an internal helper.

        """
        if not image.size:
            raise ValueError('Empty image. Have you run plot() first?')
        # write screenshot to file if requested
        if isinstance(filename, (str, pathlib.Path)):
            from PIL import Image
            filename = pathlib.Path(filename)
            if isinstance(pyvista.FIGURE_PATH, str) and not filename.is_absolute():
                filename = pathlib.Path(os.path.join(pyvista.FIGURE_PATH, filename))
            if not filename.suffix:
                filename = filename.with_suffix('.png')
            elif filename.suffix not in SUPPORTED_FORMATS:
                raise ValueError(f'Unsupported extension {filename.suffix}\n' +
                                 f'Must be one of the following: {SUPPORTED_FORMATS}')
            image_path = os.path.abspath(os.path.expanduser(str(filename)))
            Image.fromarray(image).save(image_path)
        # return image array if requested
        if return_img:
            return image

    def save_graphic(self, filename, title='PyVista Export', raster=True, painter=True):
        """Save a screenshot of the rendering window as a graphic file.

        The supported formats are: '.svg', '.eps', '.ps', '.pdf', '.tex'

        """
        if not hasattr(self, 'ren_win'):
            raise AttributeError('This plotter is closed and unable to save a screenshot.')
        if isinstance(pyvista.FIGURE_PATH, str) and not os.path.isabs(filename):
            filename = os.path.join(pyvista.FIGURE_PATH, filename)
        filename = os.path.abspath(os.path.expanduser(filename))
        extension = pyvista.fileio.get_ext(filename)
        valid = ['.svg', '.eps', '.ps', '.pdf', '.tex']
        if extension not in valid:
            raise ValueError(f"Extension ({extension}) is an invalid choice. Valid options include: {', '.join(valid)}")
        writer = _vtk.lazy_vtkGL2PSExporter()
        modes = {
            '.svg': writer.SetFileFormatToSVG,
            '.eps': writer.SetFileFormatToEPS,
            '.ps': writer.SetFileFormatToPS,
            '.pdf': writer.SetFileFormatToPDF,
            '.tex': writer.SetFileFormatToTeX,
        }
        writer.CompressOff()
        writer.SetFilePrefix(filename.replace(extension, ''))
        writer.SetInput(self.ren_win)
        modes[extension]()
        writer.SetTitle(title)
        writer.SetWrite3DPropsAsRasterImage(raster)
        if painter:
            writer.UsePainterSettings()
        writer.Update()
        return

    def screenshot(self, filename=None, transparent_background=None,
                   return_img=True, window_size=None):
        """Take screenshot at current camera position.

        Parameters
        ----------
        filename : str, optional
            Location to write image to.  If ``None``, no image is written.

        transparent_background : bool, optional
            Whether to make the background transparent.  The default is
            looked up on the plotter's theme.

        return_img : bool, optional
            If ``True`` (the default), a NumPy array of the image will
            be returned.

        window_size : 2-length tuple, optional
            Set the plotter's size to this ``(width, height)`` before
            taking the screenshot.

        Returns
        -------
        img : numpy.ndarray
            Array containing pixel RGB and alpha.  Sized:

            * [Window height x Window width x 3] if
              ``transparent_background`` is set to ``False``.
            * [Window height x Window width x 4] if
              ``transparent_background`` is set to ``True``.

        Examples
        --------
        >>> import pyvista
        >>> sphere = pyvista.Sphere()
        >>> plotter = pyvista.Plotter(off_screen=True)
        >>> actor = plotter.add_mesh(sphere)
        >>> plotter.screenshot('screenshot.png')  # doctest:+SKIP

        """
        if window_size is not None:
            self.window_size = window_size

        # configure image filter
        if transparent_background is None:
            transparent_background = self._theme.transparent_background
        self.image_transparent_background = transparent_background

        # This if statement allows you to save screenshots of closed plotters
        # This is needed for the sphinx-gallery to work
        if not hasattr(self, 'ren_win'):
            # If plotter has been closed...
            # check if last_image exists
            if self.last_image is not None:
                # Save last image
                return self._save_image(self.last_image, filename, return_img)
            # Plotter hasn't been rendered or was improperly closed
            raise RuntimeError('This plotter is closed and unable to save a screenshot.')

        if self._first_time and not self.off_screen:
            raise RuntimeError("Nothing to screenshot - call .show first or "
                               "use the off_screen argument")

        # if off screen, show has not been called and we must render
        # before extracting an image
        if self._first_time:
            self._on_first_render_request()
            self.render()

        return self._save_image(self.image, filename, return_img)

    def add_legend(self, labels=None, bcolor=(0.5, 0.5, 0.5), border=False,
                   size=None, name=None, origin=None):
        """Add a legend to render window.

        Entries must be a list containing one string and color entry for each
        item.

        Parameters
        ----------
        labels : list, optional
            When set to None, uses existing labels as specified by

            - add_mesh
            - add_lines
            - add_points

            List containing one entry for each item to be added to the
            legend.  Each entry must contain two strings, [label,
            color], where label is the name of the item to add, and
            color is the color of the label to add.

        bcolor : list or string, optional
            Background color, either a three item 0 to 1 RGB color
            list, or a matplotlib color string (e.g. 'w' or 'white'
            for a white color).  If None, legend background is
            disabled.

        border : bool, optional
            Controls if there will be a border around the legend.
            Default False.

        size : list, optional
            Two float list, each float between 0 and 1.  For example
            [0.1, 0.1] would make the legend 10% the size of the
            entire figure window.

        name : str, optional
            The name for the added actor so that it can be easily updated.
            If an actor of this name already exists in the rendering window, it
            will be replaced by the new actor.

        origin : list, optional
            If used, specifies the x and y position of the lower left corner
            of the legend.

        Returns
        -------
        legend : vtk.vtkLegendBoxActor
            Actor for the legend.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> mesh = examples.load_hexbeam()
        >>> othermesh = examples.load_uniform()
        >>> plotter = pyvista.Plotter()
        >>> _ = plotter.add_mesh(mesh, label='My Mesh')
        >>> _ = plotter.add_mesh(othermesh, 'k', label='My Other Mesh')
        >>> _ = plotter.add_legend()
        >>> cpos = plotter.show()

        Alternative manual example

        >>> import pyvista
        >>> from pyvista import examples
        >>> mesh = examples.load_hexbeam()
        >>> othermesh = examples.load_uniform()
        >>> legend_entries = []
        >>> legend_entries.append(['My Mesh', 'w'])
        >>> legend_entries.append(['My Other Mesh', 'k'])
        >>> plotter = pyvista.Plotter()
        >>> _ = plotter.add_mesh(mesh)
        >>> _ = plotter.add_mesh(othermesh, 'k')
        >>> _ = plotter.add_legend(legend_entries)
        >>> cpos = plotter.show()

        """
        self.legend = _vtk.vtkLegendBoxActor()

        if labels is None:
            # use existing labels
            if not self._labels:
                raise ValueError('No labels input.\n\n'
                                 'Add labels to individual items when adding them to'
                                 'the plotting object with the "label=" parameter.  '
                                 'or enter them as the "labels" parameter.')

            self.legend.SetNumberOfEntries(len(self._labels))
            for i, (vtk_object, text, color) in enumerate(self._labels):
                self.legend.SetEntry(i, vtk_object, text, parse_color(color))

        else:
            self.legend.SetNumberOfEntries(len(labels))
            legendface = pyvista.single_triangle()
            for i, (text, color) in enumerate(labels):
                self.legend.SetEntry(i, legendface, text, parse_color(color))

        if origin is not None:
            if not isinstance(origin, Sequence) or len(origin) != 2:
                raise ValueError(
                    '`origin` must be a list of length 2. Passed value is {}'
                    .format(origin)
                )
            self.legend.SetPosition(origin[0], origin[1])

        if size is not None:
            if not isinstance(size, Sequence) or len(size) != 2:
                raise ValueError(
                    '`size` must be a list of length 2. Passed value is {}'
                    .format(size)
                )
            self.legend.SetPosition2(size[0], size[1])

        if bcolor is None:
            self.legend.UseBackgroundOff()
        else:
            self.legend.UseBackgroundOn()
            self.legend.SetBackgroundColor(bcolor)

        if border:
            self.legend.BorderOn()
        else:
            self.legend.BorderOff()

        # Add to renderer
        self.add_actor(self.legend, reset_camera=False, name=name, pickable=False)
        return self.legend

    @wraps(Renderers.set_background)
    def set_background(self, *args, **kwargs):
        """Wrap ``Renderers.set_background``."""
        self.renderers.set_background(*args, **kwargs)

    def remove_legend(self):
        """Remove the legend actor."""
        if hasattr(self, 'legend'):
            self.remove_actor(self.legend, reset_camera=False)
            self.render()

    def generate_orbital_path(self, factor=3., n_points=20, viewup=None, shift=0.0):
        """Generate an orbital path around the data scene.

        Parameters
        ----------
        factor : float
            A scaling factor when building the orbital extent.

        n_points : int
            Number of points on the orbital path.

        viewup : list(float)
            The normal to the orbital plane.

        shift : float, optional
            Shift the plane up/down from the center of the scene by
            this amount.

        """
        if viewup is None:
            viewup = self._theme.camera['viewup']
        center = np.array(self.center)
        bnds = np.array(self.bounds)
        radius = (bnds[1] - bnds[0]) * factor
        y = (bnds[3] - bnds[2]) * factor
        if y > radius:
            radius = y
        center += np.array(viewup) * shift
        return pyvista.Polygon(center=center, radius=radius, normal=viewup, n_sides=n_points)

    def fly_to(self, point):
        """Move the current camera's focal point to a position point.

        The movement is animated over the number of frames specified in
        NumberOfFlyFrames. The LOD desired frame rate is used.

        """
        return self.iren.fly_to(self.renderer, point)

    def orbit_on_path(self, path=None, focus=None, step=0.5, viewup=None,
                      write_frames=False, threaded=False):
        """Orbit on the given path focusing on the focus point.

        Parameters
        ----------
        path : pyvista.PolyData
            Path of orbital points. The order in the points is the order of
            travel

        focus : list(float) of length 3, optional
            The point of focus the camera.

        step : float, optional
            The timestep between flying to each camera position

        viewup : list(float)
            the normal to the orbital plane

        write_frames : bool
            Assume a file is open and write a frame on each camera view during
            the orbit.

        threaded : bool, optional
            Run this as a background thread.  Generally used within a
            GUI (i.e. PyQt).

        """
        if focus is None:
            focus = self.center
        if viewup is None:
            viewup = self._theme.camera['viewup']
        if path is None:
            path = self.generate_orbital_path(viewup=viewup)
        if not is_pyvista_dataset(path):
            path = pyvista.PolyData(path)
        points = path.points

        # Make sure the whole scene is visible
        self.camera.thickness = path.length

        def orbit():
            """Define the internal thread for running the orbit."""
            for point in points:
                self.set_position(point)
                self.set_focus(focus)
                self.set_viewup(viewup)
                self.renderer.ResetCameraClippingRange()
                self.render()
                time.sleep(step)
                if write_frames:
                    self.write_frame()

        if threaded:
            thread = Thread(target=orbit)
            thread.start()
        else:
            orbit()

        return

    def export_vtkjs(self, filename, compress_arrays=False):
        """Export the current rendering scene as a VTKjs scene.

        It can be used for rendering in a web browser.

        """
        if not hasattr(self, 'ren_win'):
            raise RuntimeError('Export must be called before showing/closing the scene.')
        if isinstance(pyvista.FIGURE_PATH, str) and not os.path.isabs(filename):
            filename = os.path.join(pyvista.FIGURE_PATH, filename)
        else:
            filename = os.path.abspath(os.path.expanduser(filename))
        return export_plotter_vtkjs(self, filename, compress_arrays=compress_arrays)

    def export_obj(self, filename):
        """Export scene to OBJ format."""
        # lazy import vtkOBJExporter here as it takes a long time to
        # load and is not always used
        try:
            from vtkmodules.vtkIOExport import vtkOBJExporter
        except:
            from vtk import vtkOBJExporter

        if not hasattr(self, "ren_win"):
            raise RuntimeError("This plotter must still have a render window open.")
        if isinstance(pyvista.FIGURE_PATH, str) and not os.path.isabs(filename):
            filename = os.path.join(pyvista.FIGURE_PATH, filename)
        else:
            filename = os.path.abspath(os.path.expanduser(filename))
        exporter = vtkOBJExporter()
        exporter.SetFilePrefix(filename)
        exporter.SetRenderWindow(self.ren_win)
        return exporter.Write()

    def __del__(self):
        """Delete the plotter."""
        # We have to check here if it has the closed attribute as it
        # may not exist should the plotter have failed to initialize.
        if hasattr(self, '_closed'):
            if not self._closed:
                self.close()
        self.deep_clean()
        if hasattr(self, 'renderers'):
            del self.renderers

    def add_background_image(self, image_path, scale=1, auto_resize=True,
                             as_global=True):
        """Add a background image to a plot.

        Parameters
        ----------
        image_path : str
            Path to an image file.

        scale : float, optional
            Scale the image larger or smaller relative to the size of
            the window.  For example, a scale size of 2 will make the
            largest dimension of the image twice as large as the
            largest dimension of the render window.  Defaults to 1.

        auto_resize : bool, optional
            Resize the background when the render window changes size.

        as_global : bool, optional
            When multiple render windows are present, setting
            ``as_global=False`` will cause the background to only
            appear in one window.

        Examples
        --------
        >>> import pyvista
        >>> from pyvista import examples
        >>> plotter = pyvista.Plotter()
        >>> actor = plotter.add_mesh(pyvista.Sphere())
        >>> plotter.add_background_image(examples.mapfile)
        >>> cpos = plotter.show()

        """
        if self.renderers.has_active_background_renderer:
            raise RuntimeError('A background image already exists.  '
                               'Remove it with ``remove_background_image`` '
                               'before adding one')

        # Need to change the number of layers to support an additional
        # background layer
        self.ren_win.SetNumberOfLayers(3)
        renderer = self.renderers.add_background_renderer(image_path, scale, as_global)
        self.ren_win.AddRenderer(renderer)

        # setup autoscaling of the image
        if auto_resize:  # pragma: no cover
            self.iren.add_observer('ModifiedEvent', renderer.resize)

    def remove_background_image(self):
        """Remove the background image from the current subplot."""
        self.renderers.remove_background_image()

    def _on_first_render_request(self, cpos=None):
        """Once an image or render is officially requested, run this routine.

        For example on the show call or any screenshot producing code.
        """
        # reset unless camera for the first render unless camera is set
        if self._first_time:  # and not self.camera_set:
            for renderer in self.renderers:
                if not renderer.camera_set and cpos is None:
                    renderer.camera_position = renderer.get_default_cam_pos()
                    renderer.ResetCamera()
                elif cpos is not None:
                    renderer.camera_position = cpos
            self._first_time = False

    def reset_camera_clipping_range(self):
        """Reset camera clipping planes."""
        self.renderer.ResetCameraClippingRange()

    def add_light(self, light, only_active=False):
        """Add a Light to the scene.

        Parameters
        ----------
        light : Light or vtkLight
            The light to be added.

        only_active : bool, optional
            If ``True``, only add the light to the active
            renderer. The default is that every renderer adds the
            light. To add the light to an arbitrary renderer, see the
            ``add_light`` method of the Renderer class.

        Examples
        --------
        Create a plotter that we initialize with no lights, and add a
        cube and a single headlight to it.

        >>> import pyvista as pv
        >>> plotter = pv.Plotter(lighting='none')
        >>> _ = plotter.add_mesh(pv.Cube())
        >>> light = pv.Light(color='cyan', light_type='headlight')
        >>> plotter.add_light(light)
        >>> cpos = plotter.show()

        """
        renderers = [self.renderer] if only_active else self.renderers
        for renderer in renderers:
            renderer.add_light(light)

    def remove_all_lights(self, only_active=False):
        """Remove all lights from the scene.

        Parameters
        ----------
        only_active : bool
            If ``True``, only remove lights from the active renderer. The default
            is that lights are stripped from every renderer.

        Examples
        --------
        Create a plotter, forget to initialize it without default lighting,
        correct the mistake after instantiation.

        >>> import pyvista as pv
        >>> plotter = pv.Plotter()
        >>> plotter.remove_all_lights()
        >>> plotter.renderer.lights
        []

        """
        renderers = [self.renderer] if only_active else self.renderers
        for renderer in renderers:
            renderer.remove_all_lights()

    def where_is(self, name):
        """Return the subplot coordinates of a given actor.

        Parameters
        ----------
        name : str
            Actor's name.

        Returns
        -------
        places : list(tuple(int))
            A list with the subplot coordinates of the actor.

        Examples
        --------
        >>> import pyvista as pv
        >>> plotter = pv.Plotter(shape=(2, 2))
        >>> plotter.subplot(0, 0)
        >>> _ = plotter.add_mesh(pv.Box(), name='box')
        >>> plotter.subplot(0, 1)
        >>> _ = plotter.add_mesh(pv.Sphere(), name='sphere')
        >>> plotter.subplot(1, 0)
        >>> _ = plotter.add_mesh(pv.Box(), name='box')
        >>> plotter.subplot(1, 1)
        >>> _ = plotter.add_mesh(pv.Cone(), name='cone')
        >>> plotter.where_is('box')
        [(0, 0), (1, 0)]

        """
        places = []
        for index in range(len(self.renderers)):
            if name in self.renderers[index]._actors:
                places.append(tuple(self.renderers.index_to_loc(index)))
        return places


class Plotter(BasePlotter):
    """Plotting object to display vtk meshes or numpy arrays.

    Example
    -------
    >>> import pyvista
    >>> from pyvista import examples
    >>> mesh = examples.load_hexbeam()
    >>> another_mesh = examples.load_uniform()
    >>> plotter = pyvista.Plotter()
    >>> actor = plotter.add_mesh(mesh, color='red')
    >>> actor = plotter.add_mesh(another_mesh, color='blue')
    >>> cpos = plotter.show()

    Parameters
    ----------
    off_screen : bool, optional
        Renders off screen when ``True``.  Useful for automated
        screenshots.

    notebook : bool, optional
        When ``True``, the resulting plot is placed inline a jupyter
        notebook.  Assumes a jupyter console is active.  Automatically
        enables ``off_screen``.

    shape : list or tuple, optional
        Number of sub-render windows inside of the main window.
        Specify two across with ``shape=(2, 1)`` and a two by two grid
        with ``shape=(2, 2)``.  By default there is only one render
        window.  Can also accept a string descriptor as shape. E.g.:

            * ``shape="3|1"`` means 3 plots on the left and 1 on the right,
            * ``shape="4/2"`` means 4 plots on top and 2 at the bottom.

    border : bool, optional
        Draw a border around each render window.  Default ``False``.

    border_color : string or 3 item list, optional
        Either a string, rgb list, or hex color string.  For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1, 1, 1]``
            * ``color='#FFFFFF'``

    window_size : list, optional
        Window size in pixels.  Defaults to ``[1024, 768]``, unless
        set differently in the relevant theme's ``window_size``
        property.

    multi_samples : int, optional
        The number of multi-samples used to mitigate aliasing. 4 is a
        good default but 8 will have better results with a potential
        impact on performance.

    line_smoothing : bool, optional
        If ``True``, enable line smoothing.

    point_smoothing : bool, optional
        If ``True``, enable point smoothing.

    polygon_smoothing : bool, optional
        If ``True``, enable polygon smoothing.

    lighting : str, optional
        What lighting to set up for the plotter.
        Accepted options:

            * ``'light_kit'``: a vtk Light Kit composed of 5 lights.
            * ``'three lights'``: illumination using 3 lights.
            * ``'none'``: no light sources at instantiation.

        The default is a ``'light_kit'`` (to be precise, 5 separate
        lights that act like a Light Kit).

    theme : pyvista.themes.DefaultTheme, optional
        Plot-specific theme.

    """

    last_update_time = 0.0
    right_timer_id = -1

    def __init__(self, off_screen=None, notebook=None, shape=(1, 1),
                 groups=None, row_weights=None, col_weights=None,
                 border=None, border_color='k', border_width=2.0,
                 window_size=None, multi_samples=None, line_smoothing=False,
                 point_smoothing=False, polygon_smoothing=False,
                 splitting_position=None, title=None, lighting='light kit',
                 theme=None):
        """Initialize a vtk plotting object."""
        super().__init__(shape=shape, border=border,
                         border_color=border_color,
                         border_width=border_width,
                         groups=groups, row_weights=row_weights,
                         col_weights=col_weights,
                         splitting_position=splitting_position,
                         title=title, lighting=lighting, theme=theme)

        log.debug('Plotter init start')

        # check if a plotting backend is enabled
        _warn_xserver()

        def on_timer(iren, event_id):
            """Exit application if interactive renderer stops."""
            if event_id == 'TimerEvent':
                self.iren.terminate_app()

        if off_screen is None:
            off_screen = pyvista.OFF_SCREEN

        if notebook is None:
            if self._theme.notebook is not None:
                notebook = self._theme.notebook
            else:
                notebook = scooby.in_ipykernel()

        self.notebook = notebook
        if self.notebook:
            off_screen = True
        self.off_screen = off_screen

        self._window_size_unset = False
        if window_size is None:
            self._window_size_unset = True
            window_size = self._theme.window_size
        self.__prior_window_size = window_size

        if multi_samples is None:
            multi_samples = self._theme.multi_samples

        # initialize render window
        self.ren_win = _vtk.vtkRenderWindow()
        self.ren_win.SetMultiSamples(multi_samples)
        self.ren_win.SetBorders(True)
        if line_smoothing:
            self.ren_win.LineSmoothingOn()
        if point_smoothing:
            self.ren_win.PointSmoothingOn()
        if polygon_smoothing:
            self.ren_win.PolygonSmoothingOn()

        for renderer in self.renderers:
            self.ren_win.AddRenderer(renderer)

        # Add the shadow renderer to allow us to capture interactions within
        # a given viewport
        # https://vtk.org/pipermail/vtkusers/2018-June/102030.html
        number_or_layers = self.ren_win.GetNumberOfLayers()
        current_layer = self.renderer.GetLayer()
        self.ren_win.SetNumberOfLayers(number_or_layers + 1)
        self.ren_win.AddRenderer(self.renderers.shadow_renderer)
        self.renderers.shadow_renderer.SetLayer(current_layer + 1)
        self.renderers.shadow_renderer.SetInteractive(False)  # never needs to capture

        if self.off_screen:
            self.ren_win.SetOffScreenRendering(1)

        # Add ren win and interactor
        self.iren = RenderWindowInteractor(self, light_follow_camera=False)
        self.iren.set_render_window(self.ren_win)
        self.enable_trackball_style()  # internally calls update_style()
        self.iren.add_observer("KeyPressEvent", self.key_press_event)

        # Set background
        self.set_background(self._theme.background)

        # Set window size
        self.window_size = window_size

        # add timer event if interactive render exists
        self.iren.add_observer(_vtk.vtkCommand.TimerEvent, on_timer)

        if self._theme.depth_peeling.enabled:
            if self.enable_depth_peeling():
                for renderer in self.renderers:
                    renderer.enable_depth_peeling()
        log.debug('Plotter init stop')

    def show(self, title=None, window_size=None, interactive=True,
             auto_close=None, interactive_update=False, full_screen=None,
             screenshot=False, return_img=False, cpos=None, use_ipyvtk=None,
             jupyter_backend=None, return_viewer=False, **kwargs):
        """Display the plotting window.

        Notes
        -----
        Please use the ``q``-key to close the plotter as some
        operating systems (namely Windows) will experience issues
        saving a screenshot if the exit button in the GUI is pressed.

        Parameters
        ----------
        title : string, optional
            Title of plotting window.

        window_size : list, optional
            Window size in pixels.  Defaults to ``[1024, 768]``

        interactive : bool, optional
            Enabled by default.  Allows user to pan and move figure.

        auto_close : bool, optional
            Enabled by default.  Exits plotting session when user
            closes the window when interactive is ``True``.

        interactive_update: bool, optional
            Disabled by default.  Allows user to non-blocking draw,
            user should call ``Update()`` in each iteration.

        full_screen : bool, optional
            Opens window in full screen.  When enabled, ignores
            ``window_size``.  Default ``False``.

        cpos : list(tuple(floats))
            The camera position.  You can also set this with
            ``Plotter.camera_position``.

        screenshot : str or bool, optional
            Take a screenshot of the initial state of the plot.
            If a string, it specifies the path to which the screenshot
            is saved. If ``True``, the screenshot is returned as an
            array. Defaults to ``False``. For interactive screenshots
            it's recommended to first call ``show()`` with
            ``auto_close=False`` to set the scene, then save the
            screenshot in a separate call to ``show()`` or
            ``screenshot()``.

        return_img : bool
            Returns a numpy array representing the last image along
            with the camera position.

        use_ipyvtk : bool, optional
            Deprecated.  Instead, set the backend either globally with
            ``pyvista.set_jupyter_backend('ipyvtklink')`` or with
            ``backend='ipyvtklink'``.

        jupyter_backend : str, optional
            Jupyter notebook plotting backend to use.  One of the
            following:

            * ``'none'`` : Do not display in the notebook.
            * ``'static'`` : Display a static figure.
            * ``'ipygany'`` : Show a ``ipygany`` widget
            * ``'panel'`` : Show a ``panel`` widget.

            This can also be set globally with
            ``pyvista.set_jupyter_backend``

        return_viewer : bool, optional
            Return the jupyterlab viewer, scene, or display object
            when plotting with jupyter notebook.

        Returns
        -------
        cpos : list
            List of camera position, focal point, and view up.

        image : np.ndarray
            Numpy array of the last image when either ``return_img=True``
            or ``screenshot=True`` is set. Optionally contains alpha
            values. Sized:

            * [Window height x Window width x 3] if the theme sets
              ``transparent_background=False``.
            * [Window height x Window width x 4] if the theme sets
              ``transparent_background=True``.

        widget
            IPython widget when ``return_viewer=True``.

        Examples
        --------
        Simply show the plot of a mesh.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Cube())
        >>> pl.show()  # doctest:+SKIP

        Take a screenshot interactively.  Screenshot will be of the
        first image shown, so use the first call with
        ``auto_close=False`` to set the scene before taking the
        screenshot.

        >>> pl = pv.Plotter()
        >>> _ = pl.add_mesh(pv.Cube())
        >>> pl.show(auto_close=False)  # doctest:+SKIP
        >>> pl.show(screenshot='my_image.png')  # doctest:+SKIP

        Display an ``ipygany`` scene within a jupyter notebook

        >>> pl.show(jupyter_backend='ipygany')  # doctest:+SKIP

        Return an ``ipygany`` scene.

        >>> pl.show(jupyter_backend='ipygany', return_viewer=True)  # doctest:+SKIP

        """
        # developer keyword argument: runs a function immediately prior to ``close``
        self._before_close_callback = kwargs.pop('before_close_callback', None)
        jupyter_kwargs = kwargs.pop('jupyter_kwargs', {})
        assert_empty_kwargs(**kwargs)

        if interactive_update and auto_close is None:
            auto_close = False
        elif interactive_update and auto_close:
            warnings.warn(textwrap.dedent("""\
                The plotter will close immediately automatically since ``auto_close=True``.
                Either, do not specify ``auto_close``, or set it to ``False`` if you want to
                interact with the plotter interactively.\
                """)
            )
        elif auto_close is None:
            auto_close = self._theme.auto_close

        if use_ipyvtk:
            txt = textwrap.dedent("""\
            use_ipyvtk is deprecated.  Set the backend
            globally with ``pyvista.set_jupyter_backend("ipyvtklink"
            or with ``backend="ipyvtklink"``)
            """)
            from pyvista.core.errors import DeprecationError
            raise DeprecationError(txt)

        if not hasattr(self, "ren_win"):
            raise RuntimeError("This plotter has been closed and cannot be shown.")

        if full_screen is None:
            full_screen = self._theme.full_screen

        if full_screen:
            self.ren_win.SetFullScreen(True)
            self.ren_win.BordersOn()  # super buggy when disabled
        else:
            if window_size is None:
                window_size = self.window_size
            else:
                self._window_size_unset = False
            self.ren_win.SetSize(window_size[0], window_size[1])

        # reset unless camera for the first render unless camera is set
        self._on_first_render_request(cpos)

        # handle plotter notebook
        if jupyter_backend and not self.notebook:
            warnings.warn('Not within a jupyter notebook environment.\n'
                          'Ignoring ``jupyter_backend``.')

        if self.notebook:
            from ..jupyter.notebook import handle_plotter
            if jupyter_backend is None:
                jupyter_backend = self._theme.jupyter_backend

            if jupyter_backend != 'none':
                disp = handle_plotter(self, backend=jupyter_backend,
                                      return_viewer=return_viewer,
                                      **jupyter_kwargs)
                return disp

        self.render()

        # This has to be after the first render for some reason
        if title is None:
            title = self.title
        if title:
            self.ren_win.SetWindowName(title)
            self.title = title

        # Keep track of image for sphinx-gallery
        if pyvista.BUILDING_GALLERY or screenshot:
            # always save screenshots for sphinx_gallery
            self.last_image = self.screenshot(screenshot, return_img=True)
            self.last_image_depth = self.get_image_depth()

        # See: https://github.com/pyvista/pyvista/issues/186#issuecomment-550993270
        if interactive and not self.off_screen:
            try:  # interrupts will be caught here
                log.debug('Starting iren')
                self.iren.update_style()
                if not interactive_update:

                    # Resolves #1260
                    if os.name == 'nt':
                        if _vtk.VTK9:
                            self.iren.process_events()
                        else:
                            global VERY_FIRST_RENDER
                            if not VERY_FIRST_RENDER:
                                self.iren.start()
                            VERY_FIRST_RENDER = False

                    self.iren.start()
                self.iren.initialize()
            except KeyboardInterrupt:
                log.debug('KeyboardInterrupt')
                self.close()
                raise KeyboardInterrupt
        # In the event that the user hits the exit-button on the GUI  (on
        # Windows OS) then it must be finalized and deleted as accessing it
        # will kill the kernel.
        # Here we check for that and clean it up before moving on to any of
        # the closing routines that might try to still access that
        # render window.
        if not self.ren_win.IsCurrent():
            self._clear_ren_win()  # The ren_win is deleted
            # proper screenshots cannot be saved if this happens
            if not auto_close:
                warnings.warn("`auto_close` ignored: by clicking the exit button, "
                              "you have destroyed the render window and we have to "
                              "close it out.")
                auto_close = True
        # NOTE: after this point, nothing from the render window can be accessed
        #       as if a user presed the close button, then it destroys the
        #       the render view and a stream of errors will kill the Python
        #       kernel if code here tries to access that renderer.
        #       See issues #135 and #186 for insight before editing the
        #       remainder of this function.

        # Get camera position before closing
        cpos = self.camera_position

        # Cleanup
        if auto_close:
            self.close()

        # If user asked for screenshot, return as numpy array after camera
        # position
        if return_img or screenshot is True:
            return cpos, self.last_image

        # default to returning last used camera position
        return cpos

    def add_title(self, title, font_size=18, color=None, font=None,
                  shadow=False):
        """Add text to the top center of the plot.

        This is merely a convenience method that calls ``add_text``
        with ``position='upper_edge'``.

        Parameters
        ----------
        text : str
            The text to add the rendering.

        font : string, optional
            Font name may be courier, times, or arial.

        shadow : bool, optional
            Adds a black shadow to the text.  Defaults to False

        name : str, optional
            The name for the added actor so that it can be easily
            updated.  If an actor of this name already exists in the
            rendering window, it will be replaced by the new actor.

        Returns
        -------
        textActor : vtk.vtkTextActor
            Text actor added to plot.

        """
        # add additional spacing from the top of the figure by default
        title = '\n' + title
        return self.add_text(title, position='upper_edge',
                             font_size=font_size, color=color, font=font,
                             shadow=shadow, name='title', viewport=False)



# Tracks created plotters.  At the end of the file as we need to
# define ``BasePlotter`` before including it in the type definition.
_ALL_PLOTTERS: Dict[str, BasePlotter] = {}
