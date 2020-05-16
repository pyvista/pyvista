"""Plotting routines."""

from .colors import (color_char_to_word, get_cmap_safe, hex_to_rgb, hexcolors,
                     string_to_rgb, PARAVIEW_BACKGROUND)
from .export_vtkjs import export_plotter_vtkjs, get_vtkjs_url
from .theme import (DEFAULT_THEME, FONT_KEYS, MAX_N_COLOR_BARS,
                    parse_color, parse_font_family, rcParams, set_plot_theme)
from .tools import (create_axes_marker, create_axes_orientation_box,
                    opacity_transfer_function, system_supports_plotting)
from .widgets import WidgetHelper

from .renderer import CameraPosition, Renderer, scale_point
from .plotting import BasePlotter, Plotter, close_all
from .helpers import plot, plot_arrows, plot_compare_four, plot_itk
from .itkplotter import PlotterITK



class DeprecationError(Exception):
    """Depreciation Error."""

    pass



class BackgroundPlotter():
    """This class has been moved to pyvistaqt."""

    def __init__(self, *args, **kwargs):
        """Empty init."""
        raise DeprecationError('`BackgroundPlotter` has moved to pyvistaqt')


class QtInteractor():
    """This class has been moved to pyvistaqt."""

    def __init__(self, *args, **kwargs):
        """Empty init."""
        raise DeprecationError('`QtInteractor` has moved to pyvistaqt')
