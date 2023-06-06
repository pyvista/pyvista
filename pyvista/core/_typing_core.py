"""Type aliases for type hints."""
from typing import List, Sequence, Tuple, Union

import numpy as np

Vector = Union[List[float], Tuple[float, float, float], np.ndarray]
VectorArray = Union[np.ndarray, Sequence[Vector]]
Number = Union[float, int, np.number]
NumericArray = Union[Sequence[Number], np.ndarray]
BoundsLike = Tuple[Number, Number, Number, Number, Number, Number]
