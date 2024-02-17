from typing import TYPE_CHECKING

import numpy as np

from pyvista.core._validation import validate_array

if TYPE_CHECKING:
    # scalars
    reveal_type(validate_array(1.0, return_type='tuple'))   # EXPECTED_TYPE: "float"
    reveal_type(validate_array(1, return_type='tuple'))     # EXPECTED_TYPE: "int"
    reveal_type(validate_array(True, return_type='tuple'))  # EXPECTED_TYPE: "bool"

    # numpy arrays
    reveal_type(validate_array(np.array(1.0, dtype=float), return_type='tuple'))  # EXPECTED_TYPE: "Union[tuple[float], tuple[tuple[float]], tuple[tuple[tuple[float]]], tuple[tuple[tuple[tuple[float]]]]]"
    reveal_type(validate_array(np.array(1, dtype=int), return_type='tuple'))      # EXPECTED_TYPE: "Union[tuple[int], tuple[tuple[int]], tuple[tuple[tuple[int]]], tuple[tuple[tuple[tuple[int]]]]]"
    reveal_type(validate_array(np.array(True, dtype=bool), return_type='tuple'))  # EXPECTED_TYPE: "Union[tuple[bool], tuple[tuple[bool]], tuple[tuple[tuple[bool]]], tuple[tuple[tuple[tuple[bool]]]]]"

    # lists
    reveal_type(validate_array([1.0], return_type='tuple'))         # EXPECTED_TYPE: "tuple[float]"
    reveal_type(validate_array([1], return_type='tuple'))           # EXPECTED_TYPE: "tuple[int]"
    reveal_type(validate_array([True], return_type='tuple'))        # EXPECTED_TYPE: "tuple[bool]"
    reveal_type(validate_array([[1.0]], return_type='tuple'))       # EXPECTED_TYPE: "tuple[tuple[float]]"
    reveal_type(validate_array([[1]], return_type='tuple'))         # EXPECTED_TYPE: "tuple[tuple[int]]"
    reveal_type(validate_array([[True]], return_type='tuple'))      # EXPECTED_TYPE: "tuple[tuple[bool]]"
    reveal_type(validate_array([[[1.0]]], return_type='tuple'))     # EXPECTED_TYPE: "Union[tuple[float], tuple[tuple[float]], tuple[tuple[tuple[float]]], tuple[tuple[tuple[tuple[float]]]]]"
    reveal_type(validate_array([[[1]]], return_type='tuple'))       # EXPECTED_TYPE: "Union[tuple[int], tuple[tuple[int]], tuple[tuple[tuple[int]]], tuple[tuple[tuple[tuple[int]]]]]"
    reveal_type(validate_array([[[True]]], return_type='tuple'))    # EXPECTED_TYPE: "Union[tuple[bool], tuple[tuple[bool]], tuple[tuple[tuple[bool]]], tuple[tuple[tuple[tuple[bool]]]]]"
    reveal_type(validate_array([[[[1.0]]]], return_type='tuple'))   # EXPECTED_TYPE: "Union[tuple[float], tuple[tuple[float]], tuple[tuple[tuple[float]]], tuple[tuple[tuple[tuple[float]]]]]"
    reveal_type(validate_array([[[[1]]]], return_type='tuple'))     # EXPECTED_TYPE: "Union[tuple[int], tuple[tuple[int]], tuple[tuple[tuple[int]]], tuple[tuple[tuple[tuple[int]]]]]"
    reveal_type(validate_array([[[[True]]]], return_type='tuple'))  # EXPECTED_TYPE: "Union[tuple[bool], tuple[tuple[bool]], tuple[tuple[tuple[bool]]], tuple[tuple[tuple[tuple[bool]]]]]"

    # tuples
    reveal_type(validate_array((1.0,), return_type='tuple'))            # EXPECTED_TYPE: "tuple[float]"
    reveal_type(validate_array((1,), return_type='tuple'))              # EXPECTED_TYPE: "tuple[int]"
    reveal_type(validate_array((True,), return_type='tuple'))           # EXPECTED_TYPE: "tuple[bool]"
    reveal_type(validate_array(((1.0,),), return_type='tuple'))         # EXPECTED_TYPE: "tuple[tuple[float]]"
    reveal_type(validate_array(((1,),), return_type='tuple'))           # EXPECTED_TYPE: "tuple[tuple[int]]"
    reveal_type(validate_array(((True,),), return_type='tuple'))        # EXPECTED_TYPE: "tuple[tuple[bool]]"
    reveal_type(validate_array((((1.0,),),), return_type='tuple'))      # EXPECTED_TYPE: "Union[tuple[float], tuple[tuple[float]], tuple[tuple[tuple[float]]], tuple[tuple[tuple[tuple[float]]]]]"
    reveal_type(validate_array((((1,),),), return_type='tuple'))        # EXPECTED_TYPE: "Union[tuple[int], tuple[tuple[int]], tuple[tuple[tuple[int]]], tuple[tuple[tuple[tuple[int]]]]]"
    reveal_type(validate_array((((True,),),), return_type='tuple'))     # EXPECTED_TYPE: "Union[tuple[bool], tuple[tuple[bool]], tuple[tuple[tuple[bool]]], tuple[tuple[tuple[tuple[bool]]]]]"
    reveal_type(validate_array(((((1.0,),),),), return_type='tuple'))   # EXPECTED_TYPE: "Union[tuple[float], tuple[tuple[float]], tuple[tuple[tuple[float]]], tuple[tuple[tuple[tuple[float]]]]]"
    reveal_type(validate_array(((((1,),),),), return_type='tuple'))     # EXPECTED_TYPE: "Union[tuple[int], tuple[tuple[int]], tuple[tuple[tuple[int]]], tuple[tuple[tuple[tuple[int]]]]]"
    reveal_type(validate_array(((((True,),),),), return_type='tuple'))  # EXPECTED_TYPE: "Union[tuple[bool], tuple[tuple[bool]], tuple[tuple[tuple[bool]]], tuple[tuple[tuple[tuple[bool]]]]]"
