"""Plotting routines."""

from pyvista import MAX_N_COLOR_BARS
from .charts import Chart2D, ChartMPL, ChartBox, ChartPie
from .colors import (
    Color,
    color_like,
    color_char_to_word,
    get_cmap_safe,
    hex_to_rgb,
    hexcolors,
    string_to_rgb,
    PARAVIEW_BACKGROUND,
)
from .composite_mapper import CompositeAttributes, BlockAttributes, CompositePolyDataMapper
from .export_vtkjs import export_plotter_vtkjs, get_vtkjs_url
from .helpers import plot, plot_arrows, plot_compare_four, plot_itk
from .plotting import BasePlotter, Plotter, close_all
from ._property import Property
from .renderer import CameraPosition, Renderer, scale_point
from .tools import (
    create_axes_marker,
    create_axes_orientation_box,
    opacity_transfer_function,
    FONTS,
    system_supports_plotting,
    parse_color,
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
    """Depreciation Error for features that moved to `pyvistaqt`."""

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
