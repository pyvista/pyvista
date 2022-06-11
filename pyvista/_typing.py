"""Type aliases for type hints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple, Union

import numpy as np

from . import _vtk

if TYPE_CHECKING:  # pragma: no cover
    from .plotting.colors import Color

Vector = Union[List[float], Tuple[float, float, float], np.ndarray]
VectorArray = Union[np.ndarray, Sequence[Vector]]
Number = Union[float, int, np.number]
NumericArray = Union[Sequence[Number], np.ndarray]
color_like = Union[
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
color_like.__doc__ = """Any object convertible to a :class:`Color`."""
