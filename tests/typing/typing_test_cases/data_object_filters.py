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
    reveal_type(pv.RectilinearGrid().reflect((0,0,1), inplace=False))   # EXPECTED_TYPE: "StructuredGrid"
    reveal_type(pv.ImageData().reflect((0,0,1), inplace=False))         # EXPECTED_TYPE: "ImageData"
    reveal_type(pv.MultiBlock().reflect((0,0,1), inplace=False))        # EXPECTED_TYPE: "MultiBlock"

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

    # test rotate_vector
    reveal_type(pv.RectilinearGrid().rotate_vector((0,0,1), 0, inplace=False))  # EXPECTED_TYPE: "StructuredGrid"
    reveal_type(pv.ImageData().rotate_vector((0,0,1), 0, inplace=False))        # EXPECTED_TYPE: "ImageData"
    reveal_type(pv.MultiBlock().rotate_vector((0,0,1), 0, inplace=False))       # EXPECTED_TYPE: "MultiBlock"

    # test rotate
    reveal_type(pv.RectilinearGrid().rotate(np.eye(3), inplace=False))  # EXPECTED_TYPE: "StructuredGrid"
    reveal_type(pv.ImageData().rotate(np.eye(3), inplace=False))        # EXPECTED_TYPE: "ImageData"
    reveal_type(pv.MultiBlock().rotate(np.eye(3), inplace=False))       # EXPECTED_TYPE: "MultiBlock"

    # test translate
    reveal_type(pv.RectilinearGrid().translate((0,0,0), inplace=False)) # EXPECTED_TYPE: "StructuredGrid"
    reveal_type(pv.ImageData().translate((0,0,0), inplace=False))       # EXPECTED_TYPE: "ImageData"
    reveal_type(pv.MultiBlock().translate((0,0,0), inplace=False))      # EXPECTED_TYPE: "MultiBlock"

    # test scale
    reveal_type(pv.RectilinearGrid().scale(1, inplace=False))   # EXPECTED_TYPE: "StructuredGrid"
    reveal_type(pv.ImageData().scale(1, inplace=False))         # EXPECTED_TYPE: "ImageData"
    reveal_type(pv.MultiBlock().scale(1, inplace=False))        # EXPECTED_TYPE: "MultiBlock"

    # test flip_x
    reveal_type(pv.RectilinearGrid().flip_x(inplace=False))     # EXPECTED_TYPE: "StructuredGrid"
    reveal_type(pv.ImageData().flip_x(inplace=False))           # EXPECTED_TYPE: "ImageData"
    reveal_type(pv.MultiBlock().flip_x(inplace=False))          # EXPECTED_TYPE: "MultiBlock"

    # test flip_y
    reveal_type(pv.RectilinearGrid().flip_y(inplace=False))     # EXPECTED_TYPE: "StructuredGrid"
    reveal_type(pv.ImageData().flip_y(inplace=False))           # EXPECTED_TYPE: "ImageData"
    reveal_type(pv.MultiBlock().flip_y(inplace=False))          # EXPECTED_TYPE: "MultiBlock"

    # test flip_z
    reveal_type(pv.RectilinearGrid().flip_z(inplace=False))     # EXPECTED_TYPE: "StructuredGrid"
    reveal_type(pv.ImageData().flip_z(inplace=False))           # EXPECTED_TYPE: "ImageData"
    reveal_type(pv.MultiBlock().flip_z(inplace=False))          # EXPECTED_TYPE: "MultiBlock"

    # test rotate_vector
    reveal_type(pv.RectilinearGrid().flip_normal((0,0,1), inplace=False))   # EXPECTED_TYPE: "StructuredGrid"
    reveal_type(pv.ImageData().flip_normal((0,0,1), inplace=False))         # EXPECTED_TYPE: "ImageData"
    reveal_type(pv.MultiBlock().flip_normal((0,0,1), inplace=False))        # EXPECTED_TYPE: "MultiBlock"
