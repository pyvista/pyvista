from __future__ import annotations

import numpy as np
from typing_extensions import reveal_type

from pyvista.core._validation import validate_number

# fmt: off

reveal_type(validate_number(1.0))   # EXPECTED_TYPE: "float"
reveal_type(validate_number(1))     # EXPECTED_TYPE: "int"
# reveal_type(validate_number(True))  # EXPECTED_TYPE: "bool"

reveal_type(validate_number(np.array(1.0, dtype=float)))  # EXPECTED_TYPE: "float"
reveal_type(validate_number(np.array(1, dtype=int)))      # EXPECTED_TYPE: "int"
# reveal_type(validate_number(np.array(True, dtype=bool)))  # EXPECTED_TYPE: "bool"

reveal_type(validate_number([1.0]))   # EXPECTED_TYPE: "float"
reveal_type(validate_number(1))       # EXPECTED_TYPE: "int"
# reveal_type(validate_number([True]))  # EXPECTED_TYPE: "bool"

reveal_type(validate_number(1.0, dtype_out=int))     # EXPECTED_TYPE: "int"
reveal_type(validate_number(1, dtype_out=bool))      # EXPECTED_TYPE: "bool"
# reveal_type(validate_number(True, dtype_out=float))  # EXPECTED_TYPE: "float"

reveal_type(validate_number(np.array(1.0), dtype_out=int))     # EXPECTED_TYPE: "int"
reveal_type(validate_number(np.array(1), dtype_out=bool))      # EXPECTED_TYPE: "bool"
# reveal_type(validate_number(np.array(True), dtype_out=float))  # EXPECTED_TYPE: "float"

reveal_type(validate_number([1.0], dtype_out=int))     # EXPECTED_TYPE: "int"
reveal_type(validate_number((1), dtype_out=bool))      # EXPECTED_TYPE: "bool"
# reveal_type(validate_number([True], dtype_out=float))  # EXPECTED_TYPE: "float"
