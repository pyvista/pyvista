from typing import TYPE_CHECKING

import numpy as np

from pyvista.core._validation import validate_array

if TYPE_CHECKING:
    # scalars
    reveal_type(validate_array(1.0, return_type='list', dtype_out=int))     # EXPECTED_TYPE: "int"
    reveal_type(validate_array(1, return_type='list', dtype_out=bool))      # EXPECTED_TYPE: "bool"
    reveal_type(validate_array(True, return_type='list', dtype_out=float))  # EXPECTED_TYPE: "float"

    # numpy arrays
    reveal_type(validate_array(np.array(1.0, dtype=float), return_type='list', dtype_out=int))    # EXPECTED_TYPE: "Union[int, list[int], list[list[int]], list[list[list[int]]], list[list[list[list[int]]]]]"
    reveal_type(validate_array(np.array(1, dtype=int), return_type='list', dtype_out=bool))       # EXPECTED_TYPE: "Union[bool, list[bool], list[list[bool]], list[list[list[bool]]], list[list[list[list[bool]]]]]"
    reveal_type(validate_array(np.array(True, dtype=bool), return_type='list', dtype_out=float))  # EXPECTED_TYPE: "Union[float, list[float], list[list[float]], list[list[list[float]]], list[list[list[list[float]]]]]"

    # lists
    reveal_type(validate_array([1.0], return_type='list', dtype_out=int))           # EXPECTED_TYPE: "list[int]"
    reveal_type(validate_array([1], return_type='list', dtype_out=bool))            # EXPECTED_TYPE: "list[bool]"
    reveal_type(validate_array([True], return_type='list', dtype_out=float))        # EXPECTED_TYPE: "list[float]"
    reveal_type(validate_array([[1.0]], return_type='list', dtype_out=int))         # EXPECTED_TYPE: "list[list[int]]"
    reveal_type(validate_array([[1]], return_type='list', dtype_out=bool))          # EXPECTED_TYPE: "list[list[bool]]"
    reveal_type(validate_array([[True]], return_type='list', dtype_out=float))      # EXPECTED_TYPE: "list[list[float]]"
    reveal_type(validate_array([[[1.0]]], return_type='list', dtype_out=int))       # EXPECTED_TYPE: "Union[int, list[int], list[list[int]], list[list[list[int]]], list[list[list[list[int]]]]]"
    reveal_type(validate_array([[[1]]], return_type='list', dtype_out=bool))        # EXPECTED_TYPE: "Union[bool, list[bool], list[list[bool]], list[list[list[bool]]], list[list[list[list[bool]]]]]"
    reveal_type(validate_array([[[True]]], return_type='list', dtype_out=float))    # EXPECTED_TYPE: "Union[float, list[float], list[list[float]], list[list[list[float]]], list[list[list[list[float]]]]]"
    reveal_type(validate_array([[[[1.0]]]], return_type='list', dtype_out=int))     # EXPECTED_TYPE: "Union[int, list[int], list[list[int]], list[list[list[int]]], list[list[list[list[int]]]]]"
    reveal_type(validate_array([[[[1]]]], return_type='list', dtype_out=bool))      # EXPECTED_TYPE: "Union[bool, list[bool], list[list[bool]], list[list[list[bool]]], list[list[list[list[bool]]]]]"
    reveal_type(validate_array([[[[True]]]], return_type='list', dtype_out=float))  # EXPECTED_TYPE: "Union[float, list[float], list[list[float]], list[list[list[float]]], list[list[list[list[float]]]]]"

    # tuples
    reveal_type(validate_array((1.0,), return_type='list', dtype_out=int))              # EXPECTED_TYPE: "list[int]"
    reveal_type(validate_array((1,), return_type='list', dtype_out=bool))               # EXPECTED_TYPE: "list[bool]"
    reveal_type(validate_array((True,), return_type='list', dtype_out=float))           # EXPECTED_TYPE: "list[float]"
    reveal_type(validate_array(((1.0,),), return_type='list', dtype_out=int))           # EXPECTED_TYPE: "list[list[int]]"
    reveal_type(validate_array(((1,),), return_type='list', dtype_out=bool))            # EXPECTED_TYPE: "list[list[bool]]"
    reveal_type(validate_array(((True,),), return_type='list', dtype_out=float))        # EXPECTED_TYPE: "list[list[float]]"
    reveal_type(validate_array((((1.0,),),), return_type='list', dtype_out=int))        # EXPECTED_TYPE: "Union[int, list[int], list[list[int]], list[list[list[int]]], list[list[list[list[int]]]]]"
    reveal_type(validate_array((((1,),),), return_type='list', dtype_out=bool))         # EXPECTED_TYPE: "Union[bool, list[bool], list[list[bool]], list[list[list[bool]]], list[list[list[list[bool]]]]]"
    reveal_type(validate_array((((True,),),), return_type='list', dtype_out=float))     # EXPECTED_TYPE: "Union[float, list[float], list[list[float]], list[list[list[float]]], list[list[list[list[float]]]]]"
    reveal_type(validate_array(((((1.0,),),),), return_type='list', dtype_out=int))     # EXPECTED_TYPE: "Union[int, list[int], list[list[int]], list[list[list[int]]], list[list[list[list[int]]]]]"
    reveal_type(validate_array(((((1,),),),), return_type='list', dtype_out=bool))      # EXPECTED_TYPE: "Union[bool, list[bool], list[list[bool]], list[list[list[bool]]], list[list[list[list[bool]]]]]"
    reveal_type(validate_array(((((True,),),),), return_type='list', dtype_out=float))  # EXPECTED_TYPE: "Union[float, list[float], list[list[float]], list[list[list[float]]], list[list[list[list[float]]]]]"
