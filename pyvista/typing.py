"""Type aliases for type hints."""

from typing import Union, List, Tuple, Sequence

import numpy as np

Vector = Union[List[float], Tuple[float, float, float]]
NumericArray = Union[Sequence[Union[float, int]], np.ndarray]
