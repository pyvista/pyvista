from __future__ import annotations

from typing import TYPE_CHECKING
from typing import reveal_type

if TYPE_CHECKING:
    import numpy as np

    import pyvista as pv

# fmt: off
if TYPE_CHECKING:
    # test transform
    reveal_type(pv.RectilinearGrid().transform(np.eye(4), inplace=False))           # EXPECTED_TYPE: "StructuredGrid"
    reveal_type(pv.ImageData().transform(np.eye(4), inplace=False))                 # EXPECTED_TYPE: "ImageData"
    reveal_type(pv.PointSet().transform(np.eye(4), inplace=False))                  # EXPECTED_TYPE: "PointSet"
    reveal_type(pv.StructuredGrid().transform(np.eye(4), inplace=False))            # EXPECTED_TYPE: "StructuredGrid"
    reveal_type(pv.PolyData().transform(np.eye(4), inplace=False))                  # EXPECTED_TYPE: "PolyData"
    reveal_type(pv.ExplicitStructuredGrid().transform(np.eye(4), inplace=False))    # EXPECTED_TYPE: "ExplicitStructuredGrid"
    reveal_type(pv.UnstructuredGrid().transform(np.eye(4), inplace=False))          # EXPECTED_TYPE: "UnstructuredGrid"
    reveal_type(pv.MultiBlock().transform(np.eye(4), inplace=False))                # EXPECTED_TYPE: "MultiBlock"
