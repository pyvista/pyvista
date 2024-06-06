# flake8: noqa
from typing import TYPE_CHECKING

import numpy as np

from pyvista.core._validation._array_wrapper import _ArrayLikeWrapper

if TYPE_CHECKING:
    # scalars
    reveal_type(_ArrayLikeWrapper(1.0)._array)   # EXPECTED_TYPE: "float"
    reveal_type(_ArrayLikeWrapper(1)._array)     # EXPECTED_TYPE: "int"
    reveal_type(_ArrayLikeWrapper(True)._array)  # EXPECTED_TYPE: "bool"

    # numpy arrays
    reveal_type(_ArrayLikeWrapper(np.array((1.0),dtype=float))._array)  # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(_ArrayLikeWrapper(np.array((1),dtype=int))._array)      # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(_ArrayLikeWrapper(np.array((True),dtype=bool))._array)  # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"

    # lists
    reveal_type(_ArrayLikeWrapper([1.0])._array)         # EXPECTED_TYPE: "Sequence[float]"
    reveal_type(_ArrayLikeWrapper([1])._array)           # EXPECTED_TYPE: "Sequence[int]"
    reveal_type(_ArrayLikeWrapper([True])._array)        # EXPECTED_TYPE: "Sequence[bool]"
    reveal_type(_ArrayLikeWrapper([[1.0]])._array)       # EXPECTED_TYPE: "Sequence[Sequence[float]]"
    reveal_type(_ArrayLikeWrapper([[1]])._array)         # EXPECTED_TYPE: "Sequence[Sequence[int]]"
    reveal_type(_ArrayLikeWrapper([[True]])._array)      # EXPECTED_TYPE: "Sequence[Sequence[bool]]"
    reveal_type(_ArrayLikeWrapper([[[1.0]]])._array)     # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(_ArrayLikeWrapper([[[1]]])._array)       # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(_ArrayLikeWrapper([[[True]]])._array)    # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(_ArrayLikeWrapper([[[[1.0]]]])._array)   # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(_ArrayLikeWrapper([[[[1]]]])._array)     # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(_ArrayLikeWrapper([[[[True]]]])._array)  # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"

    # tuples
    reveal_type(_ArrayLikeWrapper((1.0,))._array)            # EXPECTED_TYPE: "Sequence[float]"
    reveal_type(_ArrayLikeWrapper((1,))._array)              # EXPECTED_TYPE: "Sequence[int]"
    reveal_type(_ArrayLikeWrapper((True,))._array)           # EXPECTED_TYPE: "Sequence[bool]"
    reveal_type(_ArrayLikeWrapper(((1.0,),))._array)         # EXPECTED_TYPE: "Sequence[Sequence[float]]"
    reveal_type(_ArrayLikeWrapper(((1,),))._array)           # EXPECTED_TYPE: "Sequence[Sequence[int]]"
    reveal_type(_ArrayLikeWrapper(((True,),))._array)        # EXPECTED_TYPE: "Sequence[Sequence[bool]]"
    reveal_type(_ArrayLikeWrapper((((1.0,),),))._array)      # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(_ArrayLikeWrapper((((1,),),))._array)        # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(_ArrayLikeWrapper((((True,),),))._array)     # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(_ArrayLikeWrapper(((((1.0,),),),))._array)   # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(_ArrayLikeWrapper(((((1,),),),))._array)     # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(_ArrayLikeWrapper(((((True,),),),))._array)  # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
