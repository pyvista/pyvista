from typing import TYPE_CHECKING

import numpy as np

from pyvista.core._validation import validate_array

if TYPE_CHECKING:
    # scalars
    reveal_type(validate_array(1.0, return_type='numpy', dtype_out=int))    # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array(1, return_type='numpy', dtype_out=bool))     # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_array(True, return_type='numpy', dtype_out=float))  # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"

    # numpy arrays
    reveal_type(validate_array(np.array(1.0, dtype=float), return_type='numpy', dtype_out=int))    # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array(np.array(1, dtype=int), return_type='numpy', dtype_out=bool))       # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_array(np.array(True, dtype=bool), return_type='numpy', dtype_out=float))  # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"

    # lists
    reveal_type(validate_array([1.0], return_type='numpy', dtype_out=int))           # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array([1], return_type='numpy', dtype_out=bool))            # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_array([True], return_type='numpy', dtype_out=float))        # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(validate_array([[1.0]], return_type='numpy', dtype_out=int))         # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array([[1]], return_type='numpy', dtype_out=bool))          # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_array([[True]], return_type='numpy', dtype_out=float))      # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(validate_array([[[1.0]]], return_type='numpy', dtype_out=int))       # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array([[[1]]], return_type='numpy', dtype_out=bool))        # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_array([[[True]]], return_type='numpy', dtype_out=float))    # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(validate_array([[[[1.0]]]], return_type='numpy', dtype_out=int))     # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array([[[[1]]]], return_type='numpy', dtype_out=bool))      # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_array([[[[True]]]], return_type='numpy', dtype_out=float))  # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"

    # tuples
    reveal_type(validate_array((1.0,), return_type='numpy', dtype_out=int))              # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array((1,), return_type='numpy', dtype_out=bool))               # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_array((True,), return_type='numpy', dtype_out=float))           # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(validate_array(((1.0,),), return_type='numpy', dtype_out=int))           # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array(((1,),), return_type='numpy', dtype_out=bool))            # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_array(((True,),), return_type='numpy', dtype_out=float))        # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(validate_array((((1.0,),),), return_type='numpy', dtype_out=int))        # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array((((1,),),), return_type='numpy', dtype_out=bool))         # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_array((((True,),),), return_type='numpy', dtype_out=float))     # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(validate_array(((((1.0,),),),), return_type='numpy', dtype_out=int))     # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_array(((((1,),),),), return_type='numpy', dtype_out=bool))      # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_array(((((True,),),),), return_type='numpy', dtype_out=float))  # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
