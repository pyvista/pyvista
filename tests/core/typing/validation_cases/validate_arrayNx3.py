from typing import TYPE_CHECKING

import numpy as np

from pyvista.core._validation import validate_arrayNx3

if TYPE_CHECKING:
    # numpy arrays
    reveal_type(validate_arrayNx3(np.array([1.0, 2.0, 3.0], dtype=float)))     # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(validate_arrayNx3(np.array([1, 2, 3], dtype=int)))             # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_arrayNx3(np.array([True, False, True], dtype=bool)))  # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"

    reveal_type(validate_arrayNx3(np.array([1.0, 2.0, 3.0]), dtype_out=int))        # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_arrayNx3(np.array([1, 2, 3]), dtype_out=bool))             # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_arrayNx3(np.array([True, False, True]), dtype_out=float))  # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"

    # 1D inputs
    reveal_type(validate_arrayNx3([1.0, 2.0, 3.0]))      # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(validate_arrayNx3([1, 2, 3]))            # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_arrayNx3([True, False, True]))  # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"

    reveal_type(validate_arrayNx3([1.0, 2.0, 3.0], dtype_out=int))        # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_arrayNx3([1, 2, 3], dtype_out=bool))             # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_arrayNx3([True, False, True], dtype_out=float))  # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"

    # 2D inputs
    reveal_type(validate_arrayNx3([[1.0, 2.0, 3.0]]))      # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
    reveal_type(validate_arrayNx3([[1, 2, 3]]))            # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_arrayNx3([[True, False, True]]))  # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"

    reveal_type(validate_arrayNx3(((1.0, 2.0, 3.0),), dtype_out=int))        # EXPECTED_TYPE: "ndarray[Any, dtype[int]]"
    reveal_type(validate_arrayNx3(((1, 2, 3),), dtype_out=bool))             # EXPECTED_TYPE: "ndarray[Any, dtype[bool]]"
    reveal_type(validate_arrayNx3(((True, False, True),), dtype_out=float))  # EXPECTED_TYPE: "ndarray[Any, dtype[float]]"
