"""Plotting routines."""

from pyvista import MAX_N_COLOR_BARS
from .charts import Chart2D, ChartMPL, ChartBox, ChartPie
from .colors import (
    Color,
    color_like,
    color_char_to_word,
    get_cmap_safe,
    hexcolors,
    PARAVIEW_BACKGROUND,
)
from .composite_mapper import CompositeAttributes, BlockAttributes, CompositePolyDataMapper
from .export_vtkjs import export_plotter_vtkjs, get_vtkjs_url
from .helpers import plot, plot_arrows, plot_compare_four
from .plotting import BasePlotter, Plotter, close_all
from ._property import Property
from .renderer import CameraPosition, Renderer, scale_point
from .tools import (
    create_axes_marker,
    create_axes_orientation_box,
    opacity_transfer_function,
    FONTS,
    system_supports_plotting,
    parse_font_family,
)
from .widgets import WidgetHelper
from .lights import Light
from .camera import Camera
from .axes import Axes
from .actor import Actor
from .mapper import DataSetMapper
from .lookup_table import LookupTable


class QtDeprecationError(Exception):
    """Deprecation Error for features that moved to `pyvistaqt`."""

    message = """`{}` has moved to pyvistaqt.
    You can install this from PyPI with: `pip install pyvistaqt`
    Then import it via: `from pyvistaqt import {}`
    `{}` is no longer accessible by `pyvista.{}`
    See https://github.com/pyvista/pyvistaqt
"""

    def __init__(self, feature_name):
        """Empty init."""
        Exception.__init__(self, self.message.format(*[feature_name] * 4))


class BackgroundPlotter:
    """This class has been moved to pyvistaqt."""

    def __init__(self, *args, **kwargs):
        """Empty init."""
        raise QtDeprecationError('BackgroundPlotter')


class QtInteractor:
    """This class has been moved to pyvistaqt."""

    def __init__(self, *args, **kwargs):
        """Empty init."""
        raise QtDeprecationError('QtInteractor')


# __all__ only left for mypy --strict to work when pyvista is a dependency
__all__ = [
    'Actor',
    'Axes',
    'BackgroundPlotter',
    'BasePlotter',
    'BlockAttributes',
    'Camera',
    'CameraPosition',
    'Chart2D',
    'ChartBox',
    'ChartMPL',
    'ChartPie',
    'Color',
    'CompositeAttributes',
    'CompositePolyDataMapper',
    'DataSetMapper',
    'FONTS',
    'Light',
    'LookupTable',
    'PARAVIEW_BACKGROUND',
    'Plotter',
    'Property',
    'QtDeprecationError',
    'QtInteractor',
    'Renderer',
    'WidgetHelper',
    'actor',
    'axes',
    'background_renderer',
    'camera',
    'charts',
    'close_all',
    'color_char_to_word',
    'color_like',
    'colors',
    'composite_mapper',
    'create_axes_marker',
    'create_axes_orientation_box',
    'export_plotter_vtkjs',
    'export_vtkjs',
    'get_cmap_safe',
    'get_vtkjs_url',
    'hexcolors',
    'lights',
    'lookup_table',
    'mapper',
    'opacity_transfer_function',
    'parse_font_family',
    'picking',
    'plot',
    'plot_arrows',
    'plot_compare_four',
    'plotting',
    'render_passes',
    'render_window_interactor',
    'renderer',
    'renderers',
    'scalar_bars',
    'scale_point',
    'system_supports_plotting',
    'tools',
    'widgets',
]
