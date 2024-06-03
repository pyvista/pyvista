# flake8: noqa
from typing import TYPE_CHECKING

import numpy as np

from pyvista.core._validation import validate_array

if TYPE_CHECKING:
    # scalars
    reveal_type(validate_array(1.0, dtype_out=int))     # EXPECTED_TYPE: "int"
    reveal_type(validate_array(1, dtype_out=bool))      # EXPECTED_TYPE: "bool"
    reveal_type(validate_array(True, dtype_out=float))  # EXPECTED_TYPE: "float"

    # numpy arrays
    reveal_type(validate_array(np.array(1.0, dtype=float), dtype_out=int))    # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array(np.array(1, dtype=int), dtype_out=bool))       # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_array(np.array(True, dtype=bool), dtype_out=float))  # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"

    # lists
    reveal_type(validate_array([1.0], dtype_out=int))           # EXPECTED_TYPE: "list[int]"
    reveal_type(validate_array([1], dtype_out=bool))            # EXPECTED_TYPE: "list[bool]"
    reveal_type(validate_array([True], dtype_out=float))        # EXPECTED_TYPE: "list[float]"
    reveal_type(validate_array([[1.0]], dtype_out=int))         # EXPECTED_TYPE: "list[list[int]]"
    reveal_type(validate_array([[1]], dtype_out=bool))          # EXPECTED_TYPE: "list[list[bool]]"
    reveal_type(validate_array([[True]], dtype_out=float))      # EXPECTED_TYPE: "list[list[float]]"
    reveal_type(validate_array([[[1.0]]], dtype_out=int))       # EXPECTED_TYPE: "Union[int, list[int], list[list[int]], list[list[list[int]]], list[list[list[list[int]]]]]"
    reveal_type(validate_array([[[1]]], dtype_out=bool))        # EXPECTED_TYPE: "Union[bool, list[bool], list[list[bool]], list[list[list[bool]]], list[list[list[list[bool]]]]]"
    reveal_type(validate_array([[[True]]], dtype_out=float))    # EXPECTED_TYPE: "Union[float, list[float], list[list[float]], list[list[list[float]]], list[list[list[list[float]]]]]"
    reveal_type(validate_array([[[[1.0]]]], dtype_out=int))     # EXPECTED_TYPE: "Union[int, list[int], list[list[int]], list[list[list[int]]], list[list[list[list[int]]]]]"
    reveal_type(validate_array([[[[1]]]], dtype_out=bool))      # EXPECTED_TYPE: "Union[bool, list[bool], list[list[bool]], list[list[list[bool]]], list[list[list[list[bool]]]]]"
    reveal_type(validate_array([[[[True]]]], dtype_out=float))  # EXPECTED_TYPE: "Union[float, list[float], list[list[float]], list[list[list[float]]], list[list[list[list[float]]]]]"

    # tuples
    reveal_type(validate_array((1.0,), dtype_out=int))              # EXPECTED_TYPE: "tuple[int]"
    reveal_type(validate_array((1,), dtype_out=bool))               # EXPECTED_TYPE: "tuple[bool]"
    reveal_type(validate_array((True,), dtype_out=float))           # EXPECTED_TYPE: "tuple[float]"
    reveal_type(validate_array(((1.0,),), dtype_out=int))           # EXPECTED_TYPE: "tuple[tuple[int]]"
    reveal_type(validate_array(((1,),), dtype_out=bool))            # EXPECTED_TYPE: "tuple[tuple[bool]]"
    reveal_type(validate_array(((True,),), dtype_out=float))        # EXPECTED_TYPE: "tuple[tuple[float]]"
    reveal_type(validate_array((((1.0,),),), dtype_out=int))        # EXPECTED_TYPE: "Union[int, tuple[int], tuple[tuple[int]], tuple[tuple[tuple[int]]], tuple[tuple[tuple[tuple[int]]]]]"
    reveal_type(validate_array((((1,),),), dtype_out=bool))         # EXPECTED_TYPE: "Union[bool, tuple[bool], tuple[tuple[bool]], tuple[tuple[tuple[bool]]], tuple[tuple[tuple[tuple[bool]]]]]"
    reveal_type(validate_array((((True,),),), dtype_out=float))     # EXPECTED_TYPE: "Union[float, tuple[float], tuple[tuple[float]], tuple[tuple[tuple[float]]], tuple[tuple[tuple[tuple[float]]]]]"
    reveal_type(validate_array(((((1.0,),),),), dtype_out=int))     # EXPECTED_TYPE: "Union[int, tuple[int], tuple[tuple[int]], tuple[tuple[tuple[int]]], tuple[tuple[tuple[tuple[int]]]]]"
    reveal_type(validate_array(((((1,),),),), dtype_out=bool))      # EXPECTED_TYPE: "Union[bool, tuple[bool], tuple[tuple[bool]], tuple[tuple[tuple[bool]]], tuple[tuple[tuple[tuple[bool]]]]]"
    reveal_type(validate_array(((((True,),),),), dtype_out=float))  # EXPECTED_TYPE: "Union[float, tuple[float], tuple[tuple[float]], tuple[tuple[tuple[float]]], tuple[tuple[tuple[tuple[float]]]]]"
