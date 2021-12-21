"""Type aliases for type hints."""

from typing import List, Sequence, Tuple, Union

import numpy as np

Vector = Union[List[float], Tuple[float, float, float], np.ndarray]
VectorArray = Union[np.ndarray, Sequence[Vector]]
Number = Union[float, int]
NumericArray = Union[Sequence[Union[float, int]], np.ndarray]
Color = Union[str, Sequence[Number]]
