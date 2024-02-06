"""Plotting routines."""
# flake8: noqa: F401

from pyvista import MAX_N_COLOR_BARS
from pyvista._plot import plot

from . import _vtk
from ._property import Property
from ._typing import Chart, ColorLike
from .actor import Actor
from .actor_properties import ActorProperties
from .axes import Axes
from .axes_actor import AxesActor
from .camera import Camera
from .charts import Chart2D, ChartBox, ChartMPL, ChartPie
from .colors import PARAVIEW_BACKGROUND, Color, color_char_to_word, get_cmap_safe, hexcolors
from .composite_mapper import BlockAttributes, CompositeAttributes, CompositePolyDataMapper
from .cube_axes_actor import CubeAxesActor
from .errors import InvalidCameraError, RenderWindowUnavailable
from .helpers import plot_arrows, plot_compare_four
from .lights import Light
from .lookup_table import LookupTable
from .mapper import (
    DataSetMapper,
    FixedPointVolumeRayCastMapper,
    GPUVolumeRayCastMapper,
    OpenGLGPUVolumeRayCastMapper,
    PointGaussianMapper,
    SmartVolumeMapper,
    UnstructuredGridVolumeRayCastMapper,
)
from .picking import PickingHelper
from .plotter import _ALL_PLOTTERS, BasePlotter, Plotter, close_all
from .prop3d import Prop3D
from .render_window_interactor import RenderWindowInteractor, Timer
from .renderer import CameraPosition, Renderer, scale_point
from .text import CornerAnnotation, Text, TextProperty
from .texture import Texture, image_to_texture, numpy_to_texture
from .themes import (
    DocumentTheme as _GlobalTheme,
    _set_plot_theme_from_env,
    load_theme,
    set_plot_theme,
)
from .tools import (
    FONTS,
    check_math_text_support,
    check_matplotlib_vtk_compatibility,
    create_axes_marker,
    create_axes_orientation_box,
    normalize,
    opacity_transfer_function,
    parse_font_family,
    system_supports_plotting,
)
from .utilities import *
from .utilities.sphinx_gallery import _get_sg_image_scraper
from .volume import Volume
from .volume_property import VolumeProperty
from .widgets import AffineWidget3D, WidgetHelper


class QtDeprecationError(Exception):  # numpydoc ignore=PR01
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


class BackgroundPlotter:  # numpydoc ignore=PR01
    """This class has been moved to pyvistaqt."""

    def __init__(self, *args, **kwargs):
        """Empty init."""
        raise QtDeprecationError('BackgroundPlotter')


class QtInteractor:  # numpydoc ignore=PR01
    """This class has been moved to pyvistaqt."""

    def __init__(self, *args, **kwargs):
        """Empty init."""
        raise QtDeprecationError('QtInteractor')


global_theme: _GlobalTheme = _GlobalTheme()

# Set preferred plot theme
_set_plot_theme_from_env()
