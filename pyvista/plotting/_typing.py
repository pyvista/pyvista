"""Type aliases for type hints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Literal
from typing import TypedDict
from typing import Union

import matplotlib as mpl

from pyvista.core._typing_core import BoundsTuple as BoundsTuple
from pyvista.core._typing_core import MatrixLike
from pyvista.core._typing_core import Number as Number
from pyvista.core._typing_core import NumpyArray
from pyvista.core._typing_core import VectorLike

from . import _vtk
from .renderer import CameraPosition

if TYPE_CHECKING:
    from pyvista.plotting.themes import Theme

    from .charts import Chart2D as Chart2D
    from .charts import ChartBox as ChartBox
    from .charts import ChartMPL as ChartMPL
    from .charts import ChartPie as ChartPie
    from .colors import _CMCRAMERI_CMAPS_LITERAL
    from .colors import _CMOCEAN_CMAPS_LITERAL
    from .colors import _COLORCET_CMAPS_LITERAL
    from .colors import _MATPLOTLIB_CMAPS_LITERAL
    from .colors import Color as Color

NamedColormaps = Union[
    '_MATPLOTLIB_CMAPS_LITERAL',
    '_CMOCEAN_CMAPS_LITERAL',
    '_COLORCET_CMAPS_LITERAL',
    '_CMCRAMERI_CMAPS_LITERAL',
]

ColormapOptions = NamedColormaps | list[str] | mpl.colors.Colormap

ColorLike = Union[
    tuple[int, int, int],
    tuple[int, int, int, int],
    tuple[float, float, float],
    tuple[float, float, float, float],
    Sequence[int],
    Sequence[float],
    NumpyArray[float],
    dict[str, int | float | str],
    str,
    'Color',
    _vtk.vtkColor3ub,
]
# Overwrite default docstring, as sphinx is not able to capture the docstring
# when it is put beneath the definition somehow?
ColorLike.__doc__ = 'Any object convertible to a :class:`Color`.'
Chart = Union['Chart2D', 'ChartBox', 'ChartPie', 'ChartMPL']
FontFamilyOptions = Literal['courier', 'times', 'arial']
OpacityOptions = Literal[
    'linear',
    'linear_r',
    'geom',
    'geom_r',
    'sigmoid',
    'sigmoid_1',
    'sigmoid_2',
    'sigmoid_3',
    'sigmoid_4',
    'sigmoid_5',
    'sigmoid_6',
    'sigmoid_7',
    'sigmoid_8',
    'sigmoid_9',
    'sigmoid_10',
    'sigmoid_15',
    'sigmoid_20',
    'foreground',
]
CullingOptions = Literal['front', 'back', 'frontface', 'backface', 'f', 'b']
StyleOptions = Literal['surface', 'wireframe', 'points', 'points_gaussian']
LightingOptions = Literal['light kit', 'three lights', 'none']
CameraPositionOptions = Union[
    Literal['xy', 'xz', 'yz', 'yx', 'zx', 'zy', 'iso'],
    VectorLike[float],
    MatrixLike[float],
    CameraPosition,
]


class BackfaceArgs(TypedDict, total=False):
    theme: Theme
    interpolation: Literal['Physically based rendering', 'pbr', 'Phong', 'Gouraud', 'Flat']
    color: ColorLike
    style: StyleOptions
    metallic: float
    roughness: float
    point_size: float
    opacity: float
    ambient: float
    diffuse: float
    specular: float
    specular_power: float
    show_edges: bool
    edge_color: ColorLike
    render_points_as_spheres: bool
    render_lines_as_tubes: bool
    lighting: bool
    line_width: float
    culling: CullingOptions | bool
    edge_opacity: float


class ScalarBarArgs(TypedDict, total=False):
    title: str
    mapper: _vtk.vtkMapper
    n_labels: int
    italic: bool
    bold: bool
    title_font_size: float
    label_font_size: float
    color: ColorLike
    font_family: FontFamilyOptions
    shadow: bool
    width: float
    height: float
    position_x: float
    position_y: float
    vertical: bool
    interactive: bool
    fmt: str
    use_opacity: bool
    outline: bool
    nan_annotation: bool
    below_label: str
    above_label: str
    background_color: ColorLike
    n_colors: int
    fill: bool
    render: bool
    theme: Theme
    unconstrained_font_size: bool


class SilhouetteArgs(TypedDict, total=False):
    color: ColorLike
    line_width: float
    opacity: float
    feature_angle: float
    decimate: float
