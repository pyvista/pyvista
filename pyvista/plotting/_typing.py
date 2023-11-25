"""Type aliases for type hints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Sequence, Tuple, Union

import numpy as np

from pyvista.core._typing_core import BoundsLike, Number  # noqa: F401

from . import _vtk

if TYPE_CHECKING:  # pragma: no cover
    from .plotting.charts import Chart2D, ChartBox, ChartMPL, ChartPie
    from .plotting.colors import Color

ColorLike = Union[
    Tuple[int, int, int],
    Tuple[int, int, int, int],
    Tuple[float, float, float],
    Tuple[float, float, float, float],
    Sequence[int],
    Sequence[float],
    np.ndarray,
    Dict[str, Union[int, float, str]],
    str,
    "Color",
    _vtk.vtkColor3ub,
]
# Overwrite default docstring, as sphinx is not able to capture the docstring
# when it is put beneath the definition somehow?
ColorLike.__doc__ = "Any object convertible to a :class:`Color`."
Chart = Union["Chart2D", "ChartBox", "ChartPie", "ChartMPL"]
