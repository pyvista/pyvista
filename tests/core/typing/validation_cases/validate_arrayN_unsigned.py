# flake8: noqa
from typing import TYPE_CHECKING

import numpy as np

from pyvista.core._validation import validate_arrayN_unsigned

if TYPE_CHECKING:
    # scalars
    reveal_type(validate_arrayN_unsigned(1.0))   # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_arrayN_unsigned(1))     # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_arrayN_unsigned(True))  # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"

    reveal_type(validate_arrayN_unsigned(1.0, dtype_out=int))          # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_arrayN_unsigned(1, dtype_out=bool))           # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_arrayN_unsigned(True, dtype_out=np.integer))  # EXPECTED_TYPE: "ndarray[Any, dtype[integer[Any]]]"

    # numpy arrays
    reveal_type(validate_arrayN_unsigned(np.array([1.0, 2.0, 3.0], dtype=float)))     # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_arrayN_unsigned(np.array([1, 2, 3], dtype=int)))             # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_arrayN_unsigned(np.array([True, False, True], dtype=bool)))  # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"

    reveal_type(validate_arrayN_unsigned(np.array([1.0, 2.0, 3.0]), dtype_out=int))           # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_arrayN_unsigned(np.array([1, 2, 3]), dtype_out=bool))                # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_arrayN_unsigned(np.array([True, False, True]), dtype_out=np.bool_))  # EXPECTED_TYPE: "ndarray[Any, dtype[bool_]]"

    # 1D inputs
    reveal_type(validate_arrayN_unsigned([1.0, 2.0, 3.0]))      # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_arrayN_unsigned([1, 2, 3]))            # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_arrayN_unsigned([True, False, True]))  # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"

    reveal_type(validate_arrayN_unsigned([1.0, 2.0, 3.0], dtype_out=int))           # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_arrayN_unsigned([1, 2, 3], dtype_out=bool))                # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_arrayN_unsigned([True, False, True], dtype_out=np.bool_))  # EXPECTED_TYPE: "ndarray[Any, dtype[bool_]]"

    # 2D inputs
    reveal_type(validate_arrayN_unsigned([[1.0, 2.0, 3.0]]))      # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_arrayN_unsigned([[1, 2, 3]]))            # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_arrayN_unsigned([[True, False, True]]))  # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"

    reveal_type(validate_arrayN_unsigned(((1.0, 2.0, 3.0),), dtype_out=int))           # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_arrayN_unsigned(((1, 2, 3),), dtype_out=bool))                # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_arrayN_unsigned(((True, False, True),), dtype_out=np.bool_))  # EXPECTED_TYPE: "ndarray[Any, dtype[bool_]]"
