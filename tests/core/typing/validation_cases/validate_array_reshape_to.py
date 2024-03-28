from typing import TYPE_CHECKING

import numpy as np

from pyvista.core._validation import validate_array

if TYPE_CHECKING:
    # scalars
    reveal_type(validate_array(1.0, reshape_to=()))        # EXPECTED_TYPE: "float"
    reveal_type(validate_array(1, reshape_to=(1)))         # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array(True, reshape_to=((1,1))))  # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"

    # numpy arrays
    reveal_type(validate_array(np.array((1.0), dtype=float), reshape_to=()))       # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(validate_array(np.array((1), dtype=int), reshape_to=(1)))          # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array(np.array((True), dtype=bool), reshape_to=((1,1))))  # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"

    # lists
    reveal_type(validate_array([1.0], reshape_to=()))              # EXPECTED_TYPE: "Union[float, list[float], list[list[float]], list[list[list[float]]], list[list[list[list[float]]]]]"
    reveal_type(validate_array([1], reshape_to=(1)))               # EXPECTED_TYPE: "list[int]"
    reveal_type(validate_array([True], reshape_to=((1,1))))        # EXPECTED_TYPE: "Union[bool, list[bool], list[list[bool]], list[list[list[bool]]], list[list[list[list[bool]]]]]"
    reveal_type(validate_array([[1.0]], reshape_to=()))            # EXPECTED_TYPE: "Union[float, list[float], list[list[float]], list[list[list[float]]], list[list[list[list[float]]]]]"
    reveal_type(validate_array([[1]], reshape_to=(1)))             # EXPECTED_TYPE: "Union[int, list[int], list[list[int]], list[list[list[int]]], list[list[list[list[int]]]]]"
    reveal_type(validate_array([[True]], reshape_to=((1,1))))      # EXPECTED_TYPE: "list[list[bool]]"
    reveal_type(validate_array([[[1.0]]], reshape_to=()))          # EXPECTED_TYPE: "Union[float, list[float], list[list[float]], list[list[list[float]]], list[list[list[list[float]]]]]"
    reveal_type(validate_array([[[1]]], reshape_to=(1)))           # EXPECTED_TYPE: "Union[int, list[int], list[list[int]], list[list[list[int]]], list[list[list[list[int]]]]]"
    reveal_type(validate_array([[[True]]], reshape_to=((1,1))))    # EXPECTED_TYPE: "Union[bool, list[bool], list[list[bool]], list[list[list[bool]]], list[list[list[list[bool]]]]]"
    reveal_type(validate_array([[[[1.0]]]], reshape_to=()))        # EXPECTED_TYPE: "Union[float, list[float], list[list[float]], list[list[list[float]]], list[list[list[list[float]]]]]"
    reveal_type(validate_array([[[[1]]]], reshape_to=(1)))         # EXPECTED_TYPE: "Union[int, list[int], list[list[int]], list[list[list[int]]], list[list[list[list[int]]]]]"
    reveal_type(validate_array([[[[True]]]], reshape_to=((1,1))))  # EXPECTED_TYPE: "Union[bool, list[bool], list[list[bool]], list[list[list[bool]]], list[list[list[list[bool]]]]]"

    # tuples
    reveal_type(validate_array((1.0,), reshape_to=()))                 # EXPECTED_TYPE: "Union[float, tuple[float], tuple[tuple[float]], tuple[tuple[tuple[float]]], tuple[tuple[tuple[tuple[float]]]]]"
    reveal_type(validate_array((1,), reshape_to=(1)))                  # EXPECTED_TYPE: "tuple[int]"
    reveal_type(validate_array((True,), reshape_to=((1,1))))           # EXPECTED_TYPE: "Union[bool, tuple[bool], tuple[tuple[bool]], tuple[tuple[tuple[bool]]], tuple[tuple[tuple[tuple[bool]]]]]"
    reveal_type(validate_array(((1.0,),), reshape_to=()))              # EXPECTED_TYPE: "Union[float, tuple[float], tuple[tuple[float]], tuple[tuple[tuple[float]]], tuple[tuple[tuple[tuple[float]]]]]"
    reveal_type(validate_array(((1,),), reshape_to=(1)))               # EXPECTED_TYPE: "Union[int, tuple[int], tuple[tuple[int]], tuple[tuple[tuple[int]]], tuple[tuple[tuple[tuple[int]]]]]"
    reveal_type(validate_array(((True,),), reshape_to=((1,1))))        # EXPECTED_TYPE: "tuple[tuple[bool]]"
    reveal_type(validate_array((((1.0,),),), reshape_to=()))           # EXPECTED_TYPE: "Union[float, tuple[float], tuple[tuple[float]], tuple[tuple[tuple[float]]], tuple[tuple[tuple[tuple[float]]]]]"
    reveal_type(validate_array((((1,),),), reshape_to=(1)))            # EXPECTED_TYPE: "Union[int, tuple[int], tuple[tuple[int]], tuple[tuple[tuple[int]]], tuple[tuple[tuple[tuple[int]]]]]"
    reveal_type(validate_array((((True,),),), reshape_to=((1,1))))     # EXPECTED_TYPE: "Union[bool, tuple[bool], tuple[tuple[bool]], tuple[tuple[tuple[bool]]], tuple[tuple[tuple[tuple[bool]]]]]"
    reveal_type(validate_array(((((1.0,),),),), reshape_to=()))        # EXPECTED_TYPE: "Union[float, tuple[float], tuple[tuple[float]], tuple[tuple[tuple[float]]], tuple[tuple[tuple[tuple[float]]]]]"
    reveal_type(validate_array(((((1,),),),), reshape_to=(1)))         # EXPECTED_TYPE: "Union[int, tuple[int], tuple[tuple[int]], tuple[tuple[tuple[int]]], tuple[tuple[tuple[tuple[int]]]]]"
    reveal_type(validate_array(((((True,),),),), reshape_to=((1,1))))  # EXPECTED_TYPE: "Union[bool, tuple[bool], tuple[tuple[bool]], tuple[tuple[tuple[bool]]], tuple[tuple[tuple[tuple[bool]]]]]"
