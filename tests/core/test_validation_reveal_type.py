from typing import TYPE_CHECKING

import numpy as np

from pyvista.core._validation import validate_array
from pyvista.core._validation._array_wrapper import _ArrayLikeWrapper

# Disable text formatting / linting for this file
# fmt: off
# flake8: noqa

# CASE: Array[T] -> Array[T]
if TYPE_CHECKING:
    # scalars
    reveal_type(_ArrayLikeWrapper(1.0)._array)   # EXPECTED_TYPE: "float"
    reveal_type(_ArrayLikeWrapper(1)._array)     # EXPECTED_TYPE: "int"
    reveal_type(_ArrayLikeWrapper(True)._array)  # EXPECTED_TYPE: "bool"

    reveal_type(validate_array(1.0))   # EXPECTED_TYPE: "float"
    reveal_type(validate_array(1))     # EXPECTED_TYPE: "int"
    reveal_type(validate_array(True))  # EXPECTED_TYPE: "bool"

    # numpy arrays
    reveal_type(_ArrayLikeWrapper(np.array((1.0),dtype=float))._array)  # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(_ArrayLikeWrapper(np.array((1),dtype=int))._array)      # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(_ArrayLikeWrapper(np.array((True),dtype=bool))._array)  # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"

    reveal_type(validate_array(np.array((1.0),dtype=float)))  # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(validate_array(np.array((1),dtype=int)))      # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array(np.array((True),dtype=bool)))  # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"

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

    reveal_type(validate_array([1.0]))         # EXPECTED_TYPE: "list[float]"
    reveal_type(validate_array([1]))           # EXPECTED_TYPE: "list[int]"
    reveal_type(validate_array([True]))        # EXPECTED_TYPE: "list[bool]"
    reveal_type(validate_array([[1.0]]))       # EXPECTED_TYPE: "list[list[float]]"
    reveal_type(validate_array([[1]]))         # EXPECTED_TYPE: "list[list[int]]"
    reveal_type(validate_array([[True]]))      # EXPECTED_TYPE: "list[list[bool]]"
    reveal_type(validate_array([[[1.0]]]))     # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(validate_array([[[1]]]))       # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array([[[True]]]))    # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_array([[[[1.0]]]]))   # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(validate_array([[[[1]]]]))     # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array([[[[True]]]]))  # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"

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

    reveal_type(validate_array((1.0,)))            # EXPECTED_TYPE: "tuple[float]"
    reveal_type(validate_array((1,)))              # EXPECTED_TYPE: "tuple[int]"
    reveal_type(validate_array((True,)))           # EXPECTED_TYPE: "tuple[bool]"
    reveal_type(validate_array(((1.0,),)))         # EXPECTED_TYPE: "tuple[tuple[float]]"
    reveal_type(validate_array(((1,),)))           # EXPECTED_TYPE: "tuple[tuple[int]]"
    reveal_type(validate_array(((True,),)))        # EXPECTED_TYPE: "tuple[tuple[bool]]"
    reveal_type(validate_array((((1.0,),),)))      # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(validate_array((((1,),),)))        # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array((((True,),),)))     # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_array(((((1.0,),),),)))   # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(validate_array(((((1,),),),)))     # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array(((((True,),),),)))  # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"


# CASE: Array[T1] -> Array[T2]
if TYPE_CHECKING:
    # scalars
    reveal_type(validate_array(1.0, dtype_out=int))     # EXPECTED_TYPE: "int"
    reveal_type(validate_array(1, dtype_out=bool))      # EXPECTED_TYPE: "bool"
    reveal_type(validate_array(True, dtype_out=float))  # EXPECTED_TYPE: "float"

    # numpy arrays
    reveal_type(validate_array(np.array(1.0), dtype_out=int))     # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array(np.array(1), dtype_out=bool))      # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_array(np.array(True), dtype_out=float))  # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"

    # lists
    reveal_type(validate_array([1.0], dtype_out=int))           # EXPECTED_TYPE: "list[int]"
    reveal_type(validate_array([1], dtype_out=bool))            # EXPECTED_TYPE: "list[bool]"
    reveal_type(validate_array([True], dtype_out=float))        # EXPECTED_TYPE: "list[float]"
    reveal_type(validate_array([[1.0]], dtype_out=int))         # EXPECTED_TYPE: "list[list[int]]"
    reveal_type(validate_array([[1]], dtype_out=bool))          # EXPECTED_TYPE: "list[list[bool]]"
    reveal_type(validate_array([[True]], dtype_out=float))      # EXPECTED_TYPE: "list[list[float]]"
    reveal_type(validate_array([[[1.0]]], dtype_out=int))       # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array([[[1]]], dtype_out=bool))        # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_array([[[True]]], dtype_out=float))    # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(validate_array([[[[1.0]]]], dtype_out=int))     # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array([[[[1]]]], dtype_out=bool))      # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_array([[[[True]]]], dtype_out=float))  # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"


    # tuples
    reveal_type(validate_array((1.0,), dtype_out=int))              # EXPECTED_TYPE: "tuple[int]"
    reveal_type(validate_array((1,), dtype_out=bool))               # EXPECTED_TYPE: "tuple[bool]"
    reveal_type(validate_array((True,), dtype_out=float))           # EXPECTED_TYPE: "tuple[float]"
    reveal_type(validate_array(((1.0,),), dtype_out=int))           # EXPECTED_TYPE: "tuple[tuple[int]]"
    reveal_type(validate_array(((1,),), dtype_out=bool))            # EXPECTED_TYPE: "tuple[tuple[bool]]"
    reveal_type(validate_array(((True,),), dtype_out=float))        # EXPECTED_TYPE: "tuple[tuple[float]]"
    reveal_type(validate_array((((1.0,),),), dtype_out=int))        # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array((((1,),),), dtype_out=bool))         # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_array((((True,),),), dtype_out=float))     # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(validate_array(((((1.0,),),),), dtype_out=int))     # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array(((((1,),),),), dtype_out=bool))      # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_array(((((True,),),),), dtype_out=float))  # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
