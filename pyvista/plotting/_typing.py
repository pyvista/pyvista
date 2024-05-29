"""Type aliases for type hints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from pyvista.core._typing_core import BoundsLike, Number, NumpyArray  # noqa: F401

    from . import _vtk
    from .plotting.charts import Chart2D, ChartBox, ChartMPL, ChartPie
    from .plotting.colors import Color

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
        "Color",
        _vtk.vtkColor3ub,
    ]
    # Overwrite default docstring, as sphinx is not able to capture the docstring
    # when it is put beneath the definition somehow?
    ColorLike.__doc__ = "Any object convertible to a :class:`Color`."
    Chart = Union["Chart2D", "ChartBox", "ChartPie", "ChartMPL"]
