"""Plotting routines."""

from __future__ import annotations

import warnings

from pyvista import MAX_N_COLOR_BARS as MAX_N_COLOR_BARS
from pyvista._plot import plot as plot
from pyvista.core.errors import PyVistaDeprecationWarning

from . import _vtk as _vtk
from ._property import Property as Property
from ._typing import CameraPositionOptions as CameraPositionOptions
from ._typing import Chart as Chart
from ._typing import ColorLike as ColorLike
from .actor import Actor as Actor
from .actor_properties import ActorProperties as ActorProperties
from .affine_widget import AffineWidget3D as AffineWidget3D
from .axes import Axes as Axes
from .axes_actor import AxesActor as AxesActor
from .axes_assembly import AxesAssembly as AxesAssembly
from .axes_assembly import AxesAssemblySymmetric as AxesAssemblySymmetric
from .axes_assembly import PlanesAssembly as PlanesAssembly
from .camera import Camera as Camera
from .charts import Chart2D as Chart2D
from .charts import ChartBox as ChartBox
from .charts import ChartMPL as ChartMPL
from .charts import ChartPie as ChartPie
from .colors import PARAVIEW_BACKGROUND as PARAVIEW_BACKGROUND
from .colors import Color as Color
from .colors import color_char_to_word as color_char_to_word
from .colors import get_cmap_safe as get_cmap_safe
from .colors import hex_colors as hex_colors
from .composite_mapper import BlockAttributes as BlockAttributes
from .composite_mapper import CompositeAttributes as CompositeAttributes
from .composite_mapper import CompositePolyDataMapper as CompositePolyDataMapper
from .cube_axes_actor import CubeAxesActor as CubeAxesActor
from .errors import InvalidCameraError as InvalidCameraError
from .errors import RenderWindowUnavailable as RenderWindowUnavailable
from .follower import Follower as Follower
from .helpers import plot_arrows as plot_arrows
from .helpers import plot_compare_four as plot_compare_four
from .lights import Light as Light
from .lookup_table import LookupTable as LookupTable
from .mapper import DataSetMapper as DataSetMapper
from .mapper import FixedPointVolumeRayCastMapper as FixedPointVolumeRayCastMapper
from .mapper import GPUVolumeRayCastMapper as GPUVolumeRayCastMapper
from .mapper import OpenGLGPUVolumeRayCastMapper as OpenGLGPUVolumeRayCastMapper
from .mapper import PointGaussianMapper as PointGaussianMapper
from .mapper import SmartVolumeMapper as SmartVolumeMapper
from .mapper import UnstructuredGridVolumeRayCastMapper as UnstructuredGridVolumeRayCastMapper
from .picking import PickingHelper as PickingHelper
from .plotter import _ALL_PLOTTERS as _ALL_PLOTTERS
from .plotter import BasePlotter as BasePlotter
from .plotter import Plotter as Plotter
from .plotter import close_all as close_all
from .prop3d import Prop3D as Prop3D
from .render_window_interactor import RenderWindowInteractor as RenderWindowInteractor
from .render_window_interactor import Timer as Timer
from .renderer import CameraPosition as CameraPosition
from .renderer import Renderer as Renderer
from .renderer import scale_point as scale_point
from .text import CornerAnnotation as CornerAnnotation
from .text import Label as Label
from .text import Text as Text
from .text import TextProperty as TextProperty
from .texture import Texture as Texture
from .texture import image_to_texture as image_to_texture
from .texture import numpy_to_texture as numpy_to_texture
from .themes import DocumentTheme as _GlobalTheme
from .themes import _set_plot_theme_from_env
from .themes import load_theme as load_theme
from .themes import set_plot_theme as set_plot_theme
from .tools import FONTS as FONTS
from .tools import check_math_text_support as check_math_text_support
from .tools import check_matplotlib_vtk_compatibility as check_matplotlib_vtk_compatibility
from .tools import create_axes_marker as create_axes_marker
from .tools import create_axes_orientation_box as create_axes_orientation_box
from .tools import normalize as normalize
from .tools import opacity_transfer_function as opacity_transfer_function
from .tools import parse_font_family as parse_font_family
from .tools import system_supports_plotting as system_supports_plotting
from .utilities import *
from .utilities.sphinx_gallery import _get_sg_image_scraper as _get_sg_image_scraper
from .volume import Volume as Volume
from .volume_property import VolumeProperty as VolumeProperty
from .widgets import WidgetHelper as WidgetHelper

with warnings.catch_warnings():
    warnings.filterwarnings(
        'ignore',
        category=PyVistaDeprecationWarning,
    )
    from .colors import hexcolors as hexcolors


class QtDeprecationError(Exception):  # numpydoc ignore=PR01
    """Deprecation Error for features that moved to `pyvistaqt`."""

    message = """`{}` has moved to pyvistaqt.
    You can install this from PyPI with: `pip install pyvistaqt`
    Then import it via: `from pyvistaqt import {}`
    `{}` is no longer accessible by `pyvista.{}`
    See https://github.com/pyvista/pyvistaqt
"""

    def __init__(self, feature_name: str) -> None:
        """Empty init."""
        Exception.__init__(self, self.message.format(*[feature_name] * 4))


class BackgroundPlotter:  # numpydoc ignore=PR01
    """This class has been moved to pyvistaqt."""  # noqa: D404

    def __init__(self, *args, **kwargs) -> None:  # noqa: ARG002
        """Empty init."""
        msg = 'BackgroundPlotter'
        raise QtDeprecationError(msg)


class QtInteractor:  # numpydoc ignore=PR01
    """This class has been moved to pyvistaqt."""  # noqa: D404

    def __init__(self, *args, **kwargs) -> None:  # noqa: ARG002
        """Empty init."""
        msg = 'QtInteractor'
        raise QtDeprecationError(msg)


global_theme: _GlobalTheme = _GlobalTheme()

# Set preferred plot theme
_set_plot_theme_from_env()
