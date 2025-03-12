from __future__ import annotations

import numpy as np
from typing_extensions import reveal_type

from pyvista import ImageData
from pyvista import MultiBlock
from pyvista import PointSet
from pyvista import PolyData
from pyvista import RectilinearGrid
from pyvista import StructuredGrid
from pyvista import UnstructuredGrid

# fmt: off

# test transform with all mesh types
reveal_type(RectilinearGrid().transform(np.eye(4), inplace=False))           # EXPECTED_TYPE: "StructuredGrid"
reveal_type(ImageData().transform(np.eye(4), inplace=False))                 # EXPECTED_TYPE: "ImageData"
reveal_type(PointSet().transform(np.eye(4), inplace=False))                  # EXPECTED_TYPE: "PointSet"
reveal_type(StructuredGrid().transform(np.eye(4), inplace=False))            # EXPECTED_TYPE: "StructuredGrid"
reveal_type(PolyData().transform(np.eye(4), inplace=False))                  # EXPECTED_TYPE: "PolyData"
# reveal_type(ExplicitStructuredGrid().transform(np.eye(4), inplace=False))    # EXPECTED_TYPE: "ExplicitStructuredGrid"
reveal_type(UnstructuredGrid().transform(np.eye(4), inplace=False))          # EXPECTED_TYPE: "UnstructuredGrid"
reveal_type(MultiBlock().transform(np.eye(4), inplace=False))                # EXPECTED_TYPE: "MultiBlock"

# test reflect
reveal_type(RectilinearGrid().reflect((0,0,1), inplace=False))   # EXPECTED_TYPE: "StructuredGrid"
reveal_type(ImageData().reflect((0,0,1), inplace=False))         # EXPECTED_TYPE: "ImageData"
reveal_type(MultiBlock().reflect((0,0,1), inplace=False))        # EXPECTED_TYPE: "MultiBlock"

# test rotate_x
reveal_type(RectilinearGrid().rotate_x(0, inplace=False))    # EXPECTED_TYPE: "StructuredGrid"
reveal_type(ImageData().rotate_x(0, inplace=False))          # EXPECTED_TYPE: "ImageData"
reveal_type(MultiBlock().rotate_x(0, inplace=False))         # EXPECTED_TYPE: "MultiBlock"

# test rotate_y
reveal_type(RectilinearGrid().rotate_y(0, inplace=False))    # EXPECTED_TYPE: "StructuredGrid"
reveal_type(ImageData().rotate_y(0, inplace=False))          # EXPECTED_TYPE: "ImageData"
reveal_type(MultiBlock().rotate_y(0, inplace=False))         # EXPECTED_TYPE: "MultiBlock"

# test rotate_z
reveal_type(RectilinearGrid().rotate_z(0, inplace=False))    # EXPECTED_TYPE: "StructuredGrid"
reveal_type(ImageData().rotate_z(0, inplace=False))          # EXPECTED_TYPE: "ImageData"
reveal_type(MultiBlock().rotate_z(0, inplace=False))         # EXPECTED_TYPE: "MultiBlock"

# test rotate_vector
reveal_type(RectilinearGrid().rotate_vector((0,0,1), 0, inplace=False))  # EXPECTED_TYPE: "StructuredGrid"
reveal_type(ImageData().rotate_vector((0,0,1), 0, inplace=False))        # EXPECTED_TYPE: "ImageData"
reveal_type(MultiBlock().rotate_vector((0,0,1), 0, inplace=False))       # EXPECTED_TYPE: "MultiBlock"

# test rotate
reveal_type(RectilinearGrid().rotate(np.eye(3), inplace=False))  # EXPECTED_TYPE: "StructuredGrid"
reveal_type(ImageData().rotate(np.eye(3), inplace=False))        # EXPECTED_TYPE: "ImageData"
reveal_type(MultiBlock().rotate(np.eye(3), inplace=False))       # EXPECTED_TYPE: "MultiBlock"

# test translate
reveal_type(RectilinearGrid().translate((0,0,0), inplace=False)) # EXPECTED_TYPE: "StructuredGrid"
reveal_type(ImageData().translate((0,0,0), inplace=False))       # EXPECTED_TYPE: "ImageData"
reveal_type(MultiBlock().translate((0,0,0), inplace=False))      # EXPECTED_TYPE: "MultiBlock"

# test scale
reveal_type(RectilinearGrid().scale(1, inplace=False))   # EXPECTED_TYPE: "StructuredGrid"
reveal_type(ImageData().scale(1, inplace=False))         # EXPECTED_TYPE: "ImageData"
reveal_type(MultiBlock().scale(1, inplace=False))        # EXPECTED_TYPE: "MultiBlock"

# test flip_x
reveal_type(RectilinearGrid().flip_x(inplace=False))     # EXPECTED_TYPE: "StructuredGrid"
reveal_type(ImageData().flip_x(inplace=False))           # EXPECTED_TYPE: "ImageData"
reveal_type(MultiBlock().flip_x(inplace=False))          # EXPECTED_TYPE: "MultiBlock"

# test flip_y
reveal_type(RectilinearGrid().flip_y(inplace=False))     # EXPECTED_TYPE: "StructuredGrid"
reveal_type(ImageData().flip_y(inplace=False))           # EXPECTED_TYPE: "ImageData"
reveal_type(MultiBlock().flip_y(inplace=False))          # EXPECTED_TYPE: "MultiBlock"

# test flip_z
reveal_type(RectilinearGrid().flip_z(inplace=False))     # EXPECTED_TYPE: "StructuredGrid"
reveal_type(ImageData().flip_z(inplace=False))           # EXPECTED_TYPE: "ImageData"
reveal_type(MultiBlock().flip_z(inplace=False))          # EXPECTED_TYPE: "MultiBlock"

# test flip_normal
reveal_type(RectilinearGrid().flip_normal((0,0,1), inplace=False))   # EXPECTED_TYPE: "StructuredGrid"
reveal_type(ImageData().flip_normal((0,0,1), inplace=False))         # EXPECTED_TYPE: "ImageData"
reveal_type(MultiBlock().flip_normal((0,0,1), inplace=False))        # EXPECTED_TYPE: "MultiBlock"
