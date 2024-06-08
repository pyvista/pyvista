"""Type aliases for type hints."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import Sequence
from typing import Tuple
from typing import Union

from pyvista.core._typing_core import BoundsLike  # noqa: F401
from pyvista.core._typing_core import Number  # noqa: F401
from pyvista.core._typing_core import NumpyArray

from . import _vtk

if TYPE_CHECKING:  # pragma: no cover
    from .plotting.charts import Chart2D
    from .plotting.charts import ChartBox
    from .plotting.charts import ChartMPL
    from .plotting.charts import ChartPie
    from .plotting.colors import Color

ColorLike = Union[
    Tuple[int, int, int],
    Tuple[int, int, int, int],
    Tuple[float, float, float],
    Tuple[float, float, float, float],
    Sequence[int],
    Sequence[float],
    NumpyArray[float],
    Dict[str, Union[int, float, str]],
    str,
    "Color",
    _vtk.vtkColor3ub,
]
# Overwrite default docstring, as sphinx is not able to capture the docstring
# when it is put beneath the definition somehow?
ColorLike.__doc__ = "Any object convertible to a :class:`Color`."
Chart = Union["Chart2D", "ChartBox", "ChartPie", "ChartMPL"]
