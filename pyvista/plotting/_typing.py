"""Type aliases for type hints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Union

from pyvista.core._typing_core import BoundsTuple as BoundsTuple
from pyvista.core._typing_core import Number as Number
from pyvista.core._typing_core import NumpyArray

from . import _vtk

if TYPE_CHECKING:
    from .charts import Chart2D as Chart2D
    from .charts import ChartBox as ChartBox
    from .charts import ChartMPL as ChartMPL
    from .charts import ChartPie as ChartPie
    from .colors import Color as Color

ColorLike = Union[
    tuple[int, int, int],
    tuple[int, int, int, int],
    tuple[float, float, float],
    tuple[float, float, float, float],
    Sequence[int],
    Sequence[float],
    NumpyArray[float],
    dict[str, Union[int, float, str]],
    str,
    'Color',
    _vtk.vtkColor3ub,
]
# Overwrite default docstring, as sphinx is not able to capture the docstring
# when it is put beneath the definition somehow?
ColorLike.__doc__ = 'Any object convertible to a :class:`Color`.'
Chart = Union['Chart2D', 'ChartBox', 'ChartPie', 'ChartMPL']
