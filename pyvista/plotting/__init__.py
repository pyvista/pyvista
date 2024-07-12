"""Plotting routines."""

# ruff: noqa: F401
from __future__ import annotations

from pyvista import MAX_N_COLOR_BARS
from pyvista._plot import plot

from . import _vtk
from ._property import Property
from ._typing import Chart
from ._typing import ColorLike
from .actor import Actor
from .actor_properties import ActorProperties
from .axes import Axes
from .axes_actor import AxesActor
from .axes_assembly import AxesAssembly
from .axes_assembly import AxesAssemblySymmetric
from .camera import Camera
from .charts import Chart2D
from .charts import ChartBox
from .charts import ChartMPL
from .charts import ChartPie
from .colors import PARAVIEW_BACKGROUND
from .colors import Color
from .colors import color_char_to_word
from .colors import get_cmap_safe
from .colors import hexcolors
from .composite_mapper import BlockAttributes
from .composite_mapper import CompositeAttributes
from .composite_mapper import CompositePolyDataMapper
from .cube_axes_actor import CubeAxesActor
from .errors import InvalidCameraError
from .errors import RenderWindowUnavailable
from .helpers import plot_arrows
from .helpers import plot_compare_four
from .lights import Light
from .lookup_table import LookupTable
from .mapper import DataSetMapper
from .mapper import FixedPointVolumeRayCastMapper
from .mapper import GPUVolumeRayCastMapper
from .mapper import OpenGLGPUVolumeRayCastMapper
from .mapper import PointGaussianMapper
from .mapper import SmartVolumeMapper
from .mapper import UnstructuredGridVolumeRayCastMapper
from .picking import PickingHelper
from .plotter import _ALL_PLOTTERS
from .plotter import BasePlotter
from .plotter import Plotter
from .plotter import close_all
from .prop3d import Prop3D
from .render_window_interactor import RenderWindowInteractor
from .render_window_interactor import Timer
from .renderer import CameraPosition
from .renderer import Renderer
from .renderer import scale_point
from .text import CornerAnnotation
from .text import Label
from .text import Text
from .text import TextProperty
from .texture import Texture
from .texture import image_to_texture
from .texture import numpy_to_texture
from .themes import DocumentTheme as _GlobalTheme
from .themes import _set_plot_theme_from_env
from .themes import load_theme
from .themes import set_plot_theme
from .tools import FONTS
from .tools import check_math_text_support
from .tools import check_matplotlib_vtk_compatibility
from .tools import create_axes_marker
from .tools import create_axes_orientation_box
from .tools import normalize
from .tools import opacity_transfer_function
from .tools import parse_font_family
from .tools import system_supports_plotting
from .utilities import *
from .utilities.sphinx_gallery import _get_sg_image_scraper
from .volume import Volume
from .volume_property import VolumeProperty
from .widgets import AffineWidget3D
from .widgets import WidgetHelper


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
