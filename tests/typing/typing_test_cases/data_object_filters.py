from __future__ import annotations

from typing import TYPE_CHECKING
from typing import reveal_type

if TYPE_CHECKING:
    import numpy as np

    import pyvista as pv

# fmt: off
if TYPE_CHECKING:
    # test transform with all mesh types
    reveal_type(pv.RectilinearGrid().transform(np.eye(4), inplace=False))           # EXPECTED_TYPE: "StructuredGrid"
    reveal_type(pv.ImageData().transform(np.eye(4), inplace=False))                 # EXPECTED_TYPE: "ImageData"
    reveal_type(pv.PointSet().transform(np.eye(4), inplace=False))                  # EXPECTED_TYPE: "PointSet"
    reveal_type(pv.StructuredGrid().transform(np.eye(4), inplace=False))            # EXPECTED_TYPE: "StructuredGrid"
    reveal_type(pv.PolyData().transform(np.eye(4), inplace=False))                  # EXPECTED_TYPE: "PolyData"
    reveal_type(pv.ExplicitStructuredGrid().transform(np.eye(4), inplace=False))    # EXPECTED_TYPE: "ExplicitStructuredGrid"
    reveal_type(pv.UnstructuredGrid().transform(np.eye(4), inplace=False))          # EXPECTED_TYPE: "UnstructuredGrid"
    reveal_type(pv.MultiBlock().transform(np.eye(4), inplace=False))                # EXPECTED_TYPE: "MultiBlock"

    # test reflect
    reveal_type(pv.RectilinearGrid().reflect((0,0,1), inplace=False))     # EXPECTED_TYPE: "StructuredGrid"
    reveal_type(pv.ImageData().reflect((0,0,1), inplace=False))           # EXPECTED_TYPE: "ImageData"
    reveal_type(pv.MultiBlock().reflect((0,0,1), inplace=False))          # EXPECTED_TYPE: "MultiBlock"

    # test rotate_x
    reveal_type(pv.RectilinearGrid().rotate_x(0, inplace=False))    # EXPECTED_TYPE: "StructuredGrid"
    reveal_type(pv.ImageData().rotate_x(0, inplace=False))          # EXPECTED_TYPE: "ImageData"
    reveal_type(pv.MultiBlock().rotate_x(0, inplace=False))         # EXPECTED_TYPE: "MultiBlock"

    # test rotate_y
    reveal_type(pv.RectilinearGrid().rotate_y(0, inplace=False))    # EXPECTED_TYPE: "StructuredGrid"
    reveal_type(pv.ImageData().rotate_y(0, inplace=False))          # EXPECTED_TYPE: "ImageData"
    reveal_type(pv.MultiBlock().rotate_y(0, inplace=False))         # EXPECTED_TYPE: "MultiBlock"

    # test rotate_z
    reveal_type(pv.RectilinearGrid().rotate_z(0, inplace=False))    # EXPECTED_TYPE: "StructuredGrid"
    reveal_type(pv.ImageData().rotate_z(0, inplace=False))          # EXPECTED_TYPE: "ImageData"
    reveal_type(pv.MultiBlock().rotate_z(0, inplace=False))         # EXPECTED_TYPE: "MultiBlock"
