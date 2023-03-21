"""Plotting routines."""

from pyvista import MAX_N_COLOR_BARS

from ._property import Property
from .actor import Actor
from .actor_properties import ActorProperties
from .axes import Axes
from .axes_actor import AxesActor
from .camera import Camera
from .charts import Chart, Chart2D, ChartBox, ChartMPL, ChartPie
from .colors import (
    PARAVIEW_BACKGROUND,
    Color,
    ColorLike,
    color_char_to_word,
    get_cmap_safe,
    hexcolors,
)
from .composite_mapper import (
    BlockAttributes,
    CompositeAttributes,
    CompositePolyDataMapper,
)
from .export_vtkjs import export_plotter_vtkjs, get_vtkjs_url
from .helpers import plot, plot_arrows, plot_compare_four
from .lights import Light
from .lookup_table import LookupTable
from .mapper import DataSetMapper, _BaseMapper
from .plotting import BasePlotter, Plotter, close_all
from .renderer import CameraPosition, Renderer, scale_point
from .tools import (
    FONTS,
    create_axes_marker,
    create_axes_orientation_box,
    opacity_transfer_function,
    parse_font_family,
    system_supports_plotting,
)
from .widgets import WidgetHelper


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
