from __future__ import annotations

import numpy as np
from trimesh import Trimesh
from typing_extensions import reveal_type

from pyvista import wrap
from pyvista.core import _vtk_core as _vtk

# fmt: off

reveal_type(wrap(_vtk.vtkPolyData()))               # EXPECTED_TYPE: "PolyData"
reveal_type(wrap(_vtk.vtkStructuredGrid()))         # EXPECTED_TYPE: "StructuredGrid"
# reveal_type(wrap(_vtk.vtkExplicitStructuredGrid())) # EXPECTED_TYPE: "ExplicitStructuredGrid"
reveal_type(wrap(_vtk.vtkUnstructuredGrid()))       # EXPECTED_TYPE: "UnstructuredGrid"
reveal_type(wrap(_vtk.vtkPointSet()))               # EXPECTED_TYPE: "PointSet"
reveal_type(wrap(_vtk.vtkRectilinearGrid()))        # EXPECTED_TYPE: "RectilinearGrid"
reveal_type(wrap(_vtk.vtkStructuredPoints()))       # EXPECTED_TYPE: "ImageData"
reveal_type(wrap(_vtk.vtkImageData()))              # EXPECTED_TYPE: "ImageData"
reveal_type(wrap(_vtk.vtkMultiBlockDataSet()))      # EXPECTED_TYPE: "MultiBlock"
reveal_type(wrap(_vtk.vtkTable()))                  # EXPECTED_TYPE: "Table"
reveal_type(wrap(_vtk.vtkPartitionedDataSet()))     # EXPECTED_TYPE: "PartitionedDataSet"
reveal_type(wrap(np.zeros(shape=(100,3))))          # EXPECTED_TYPE: "Union[PolyData, ImageData]"
reveal_type(wrap(_vtk.vtkFloatArray()))             # EXPECTED_TYPE: "pyvista_ndarray"
reveal_type(wrap(None))                             # EXPECTED_TYPE: "None"
reveal_type(wrap(Trimesh()))                        # EXPECTED_TYPE: "PolyData"
