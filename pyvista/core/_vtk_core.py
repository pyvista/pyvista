"""Limited imports from VTK (excludes any GL-dependent).

These are the modules within VTK that must be loaded across pyvista's
core API. Here, we attempt to import modules using the ``vtkmodules``
package, which lets us only have to import from select modules and not
the entire library.

"""

from __future__ import annotations

import contextlib
import sys
from typing import NamedTuple
import warnings

from vtkmodules.vtkCommonCore import vtkInformation as vtkInformation
from vtkmodules.vtkCommonCore import vtkVersion as vtkVersion
from vtkmodules.vtkImagingSources import vtkImageEllipsoidSource as vtkImageEllipsoidSource
from vtkmodules.vtkImagingSources import vtkImageGaussianSource as vtkImageGaussianSource
from vtkmodules.vtkImagingSources import vtkImageGridSource as vtkImageGridSource
from vtkmodules.vtkImagingSources import vtkImageMandelbrotSource as vtkImageMandelbrotSource
from vtkmodules.vtkImagingSources import vtkImageNoiseSource as vtkImageNoiseSource
from vtkmodules.vtkImagingSources import vtkImageSinusoidSource as vtkImageSinusoidSource

# vtkExtractEdges moved from vtkFiltersExtraction to vtkFiltersCore in
# VTK commit d9981b9aeb93b42d1371c6e295d76bfdc18430bd
try:
    from vtkmodules.vtkFiltersCore import vtkExtractEdges as vtkExtractEdges
except ImportError:
    from vtkmodules.vtkFiltersExtraction import (  # type: ignore[attr-defined, no-redef]
        vtkExtractEdges as vtkExtractEdges,
    )

# vtkCellTreeLocator moved from vtkFiltersGeneral to vtkCommonDataModel in
# VTK commit 4a29e6f7dd9acb460644fe487d2e80aac65f7be9
try:
    from vtkmodules.vtkCommonDataModel import vtkCellTreeLocator as vtkCellTreeLocator
except ImportError:
    from vtkmodules.vtkFiltersGeneral import (  # type: ignore[attr-defined, no-redef]
        vtkCellTreeLocator as vtkCellTreeLocator,
    )

from vtkmodules.numpy_interface.dataset_adapter import VTKArray as VTKArray
from vtkmodules.numpy_interface.dataset_adapter import VTKObjectWrapper as VTKObjectWrapper
from vtkmodules.numpy_interface.dataset_adapter import numpyTovtkDataArray as numpyTovtkDataArray
from vtkmodules.util.numpy_support import get_vtk_array_type as get_vtk_array_type
from vtkmodules.util.numpy_support import numpy_to_vtk as numpy_to_vtk
from vtkmodules.util.numpy_support import numpy_to_vtkIdTypeArray as numpy_to_vtkIdTypeArray
from vtkmodules.util.numpy_support import vtk_to_numpy as vtk_to_numpy

with contextlib.suppress(ImportError):
    from vtkmodules.util.pickle_support import (
        serialize_VTK_data_object as serialize_VTK_data_object,
    )

from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase as VTKPythonAlgorithmBase
from vtkmodules.vtkCommonComputationalGeometry import vtkKochanekSpline as vtkKochanekSpline
from vtkmodules.vtkCommonComputationalGeometry import (
    vtkParametricBohemianDome as vtkParametricBohemianDome,
)
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricBour as vtkParametricBour
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricBoy as vtkParametricBoy
from vtkmodules.vtkCommonComputationalGeometry import (
    vtkParametricCatalanMinimal as vtkParametricCatalanMinimal,
)
from vtkmodules.vtkCommonComputationalGeometry import (
    vtkParametricConicSpiral as vtkParametricConicSpiral,
)
from vtkmodules.vtkCommonComputationalGeometry import (
    vtkParametricCrossCap as vtkParametricCrossCap,
)
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricDini as vtkParametricDini
from vtkmodules.vtkCommonComputationalGeometry import (
    vtkParametricEllipsoid as vtkParametricEllipsoid,
)
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricEnneper as vtkParametricEnneper
from vtkmodules.vtkCommonComputationalGeometry import (
    vtkParametricFigure8Klein as vtkParametricFigure8Klein,
)
from vtkmodules.vtkCommonComputationalGeometry import (
    vtkParametricFunction as vtkParametricFunction,
)
from vtkmodules.vtkCommonComputationalGeometry import (
    vtkParametricHenneberg as vtkParametricHenneberg,
)
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricKlein as vtkParametricKlein
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricKuen as vtkParametricKuen
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricMobius as vtkParametricMobius
from vtkmodules.vtkCommonComputationalGeometry import (
    vtkParametricPluckerConoid as vtkParametricPluckerConoid,
)
from vtkmodules.vtkCommonComputationalGeometry import (
    vtkParametricPseudosphere as vtkParametricPseudosphere,
)
from vtkmodules.vtkCommonComputationalGeometry import (
    vtkParametricRandomHills as vtkParametricRandomHills,
)
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricRoman as vtkParametricRoman
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricSpline as vtkParametricSpline
from vtkmodules.vtkCommonComputationalGeometry import (
    vtkParametricSuperEllipsoid as vtkParametricSuperEllipsoid,
)
from vtkmodules.vtkCommonComputationalGeometry import (
    vtkParametricSuperToroid as vtkParametricSuperToroid,
)
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricTorus as vtkParametricTorus
from vtkmodules.vtkCommonCore import VTK_ARIAL as VTK_ARIAL
from vtkmodules.vtkCommonCore import VTK_BIT as VTK_BIT
from vtkmodules.vtkCommonCore import VTK_CHAR as VTK_CHAR
from vtkmodules.vtkCommonCore import VTK_COURIER as VTK_COURIER
from vtkmodules.vtkCommonCore import VTK_DOUBLE as VTK_DOUBLE
from vtkmodules.vtkCommonCore import VTK_FLOAT as VTK_FLOAT
from vtkmodules.vtkCommonCore import VTK_FONT_FILE as VTK_FONT_FILE
from vtkmodules.vtkCommonCore import VTK_ID_TYPE as VTK_ID_TYPE
from vtkmodules.vtkCommonCore import VTK_INT as VTK_INT
from vtkmodules.vtkCommonCore import VTK_LONG as VTK_LONG
from vtkmodules.vtkCommonCore import VTK_LONG_LONG as VTK_LONG_LONG
from vtkmodules.vtkCommonCore import VTK_SHORT as VTK_SHORT
from vtkmodules.vtkCommonCore import VTK_SIGNED_CHAR as VTK_SIGNED_CHAR
from vtkmodules.vtkCommonCore import VTK_STRING as VTK_STRING
from vtkmodules.vtkCommonCore import VTK_TIMES as VTK_TIMES
from vtkmodules.vtkCommonCore import VTK_UNSIGNED_CHAR as VTK_UNSIGNED_CHAR
from vtkmodules.vtkCommonCore import VTK_UNSIGNED_INT as VTK_UNSIGNED_INT
from vtkmodules.vtkCommonCore import VTK_UNSIGNED_LONG as VTK_UNSIGNED_LONG
from vtkmodules.vtkCommonCore import VTK_UNSIGNED_LONG_LONG as VTK_UNSIGNED_LONG_LONG
from vtkmodules.vtkCommonCore import VTK_UNSIGNED_SHORT as VTK_UNSIGNED_SHORT
from vtkmodules.vtkCommonCore import buffer_shared as buffer_shared  # type: ignore[attr-defined]
from vtkmodules.vtkCommonCore import mutable as mutable
from vtkmodules.vtkCommonCore import reference as reference
from vtkmodules.vtkCommonCore import vtkAbstractArray as vtkAbstractArray
from vtkmodules.vtkCommonCore import vtkBitArray as vtkBitArray
from vtkmodules.vtkCommonCore import vtkCharArray as vtkCharArray
from vtkmodules.vtkCommonCore import vtkCommand as vtkCommand
from vtkmodules.vtkCommonCore import vtkDataArray as vtkDataArray
from vtkmodules.vtkCommonCore import vtkDoubleArray as vtkDoubleArray
from vtkmodules.vtkCommonCore import vtkFileOutputWindow as vtkFileOutputWindow
from vtkmodules.vtkCommonCore import vtkFloatArray as vtkFloatArray
from vtkmodules.vtkCommonCore import vtkIdList as vtkIdList
from vtkmodules.vtkCommonCore import vtkIdTypeArray as vtkIdTypeArray
from vtkmodules.vtkCommonCore import vtkIntArray as vtkIntArray
from vtkmodules.vtkCommonCore import vtkLogger as vtkLogger
from vtkmodules.vtkCommonCore import vtkLongArray as vtkLongArray
from vtkmodules.vtkCommonCore import vtkLongLongArray as vtkLongLongArray
from vtkmodules.vtkCommonCore import vtkLookupTable as vtkLookupTable
from vtkmodules.vtkCommonCore import vtkMath as vtkMath
from vtkmodules.vtkCommonCore import vtkOutputWindow as vtkOutputWindow
from vtkmodules.vtkCommonCore import vtkPoints as vtkPoints
from vtkmodules.vtkCommonCore import vtkShortArray as vtkShortArray
from vtkmodules.vtkCommonCore import vtkSignedCharArray as vtkSignedCharArray
from vtkmodules.vtkCommonCore import vtkStringArray as vtkStringArray
from vtkmodules.vtkCommonCore import vtkStringOutputWindow as vtkStringOutputWindow
from vtkmodules.vtkCommonCore import vtkTypeInt32Array as vtkTypeInt32Array
from vtkmodules.vtkCommonCore import vtkTypeInt64Array as vtkTypeInt64Array
from vtkmodules.vtkCommonCore import vtkTypeUInt32Array as vtkTypeUInt32Array
from vtkmodules.vtkCommonCore import vtkUnsignedCharArray as vtkUnsignedCharArray
from vtkmodules.vtkCommonCore import vtkUnsignedIntArray as vtkUnsignedIntArray
from vtkmodules.vtkCommonCore import vtkUnsignedLongArray as vtkUnsignedLongArray
from vtkmodules.vtkCommonCore import vtkUnsignedLongLongArray as vtkUnsignedLongLongArray
from vtkmodules.vtkCommonCore import vtkUnsignedShortArray as vtkUnsignedShortArray
from vtkmodules.vtkCommonCore import vtkWeakReference as vtkWeakReference
from vtkmodules.vtkCommonDataModel import VTK_BEZIER_CURVE as VTK_BEZIER_CURVE
from vtkmodules.vtkCommonDataModel import VTK_BEZIER_HEXAHEDRON as VTK_BEZIER_HEXAHEDRON
from vtkmodules.vtkCommonDataModel import VTK_BEZIER_PYRAMID as VTK_BEZIER_PYRAMID
from vtkmodules.vtkCommonDataModel import VTK_BEZIER_QUADRILATERAL as VTK_BEZIER_QUADRILATERAL
from vtkmodules.vtkCommonDataModel import VTK_BEZIER_TETRAHEDRON as VTK_BEZIER_TETRAHEDRON
from vtkmodules.vtkCommonDataModel import VTK_BEZIER_TRIANGLE as VTK_BEZIER_TRIANGLE
from vtkmodules.vtkCommonDataModel import VTK_BEZIER_WEDGE as VTK_BEZIER_WEDGE
from vtkmodules.vtkCommonDataModel import VTK_BIQUADRATIC_QUAD as VTK_BIQUADRATIC_QUAD
from vtkmodules.vtkCommonDataModel import (
    VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON as VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON,
)
from vtkmodules.vtkCommonDataModel import (
    VTK_BIQUADRATIC_QUADRATIC_WEDGE as VTK_BIQUADRATIC_QUADRATIC_WEDGE,
)
from vtkmodules.vtkCommonDataModel import VTK_BIQUADRATIC_TRIANGLE as VTK_BIQUADRATIC_TRIANGLE
from vtkmodules.vtkCommonDataModel import VTK_CONVEX_POINT_SET as VTK_CONVEX_POINT_SET
from vtkmodules.vtkCommonDataModel import VTK_CUBIC_LINE as VTK_CUBIC_LINE
from vtkmodules.vtkCommonDataModel import VTK_EMPTY_CELL as VTK_EMPTY_CELL
from vtkmodules.vtkCommonDataModel import VTK_HEXAGONAL_PRISM as VTK_HEXAGONAL_PRISM
from vtkmodules.vtkCommonDataModel import VTK_HEXAHEDRON as VTK_HEXAHEDRON
from vtkmodules.vtkCommonDataModel import VTK_HIGHER_ORDER_EDGE as VTK_HIGHER_ORDER_EDGE
from vtkmodules.vtkCommonDataModel import (
    VTK_HIGHER_ORDER_HEXAHEDRON as VTK_HIGHER_ORDER_HEXAHEDRON,
)
from vtkmodules.vtkCommonDataModel import VTK_HIGHER_ORDER_POLYGON as VTK_HIGHER_ORDER_POLYGON
from vtkmodules.vtkCommonDataModel import VTK_HIGHER_ORDER_PYRAMID as VTK_HIGHER_ORDER_PYRAMID
from vtkmodules.vtkCommonDataModel import VTK_HIGHER_ORDER_QUAD as VTK_HIGHER_ORDER_QUAD
from vtkmodules.vtkCommonDataModel import (
    VTK_HIGHER_ORDER_TETRAHEDRON as VTK_HIGHER_ORDER_TETRAHEDRON,
)
from vtkmodules.vtkCommonDataModel import VTK_HIGHER_ORDER_TRIANGLE as VTK_HIGHER_ORDER_TRIANGLE
from vtkmodules.vtkCommonDataModel import VTK_HIGHER_ORDER_WEDGE as VTK_HIGHER_ORDER_WEDGE
from vtkmodules.vtkCommonDataModel import VTK_LAGRANGE_CURVE as VTK_LAGRANGE_CURVE
from vtkmodules.vtkCommonDataModel import VTK_LAGRANGE_HEXAHEDRON as VTK_LAGRANGE_HEXAHEDRON
from vtkmodules.vtkCommonDataModel import VTK_LAGRANGE_PYRAMID as VTK_LAGRANGE_PYRAMID
from vtkmodules.vtkCommonDataModel import VTK_LAGRANGE_QUADRILATERAL as VTK_LAGRANGE_QUADRILATERAL
from vtkmodules.vtkCommonDataModel import VTK_LAGRANGE_TETRAHEDRON as VTK_LAGRANGE_TETRAHEDRON
from vtkmodules.vtkCommonDataModel import VTK_LAGRANGE_TRIANGLE as VTK_LAGRANGE_TRIANGLE
from vtkmodules.vtkCommonDataModel import VTK_LAGRANGE_WEDGE as VTK_LAGRANGE_WEDGE
from vtkmodules.vtkCommonDataModel import VTK_LINE as VTK_LINE
from vtkmodules.vtkCommonDataModel import VTK_PARAMETRIC_CURVE as VTK_PARAMETRIC_CURVE
from vtkmodules.vtkCommonDataModel import VTK_PARAMETRIC_HEX_REGION as VTK_PARAMETRIC_HEX_REGION
from vtkmodules.vtkCommonDataModel import (
    VTK_PARAMETRIC_QUAD_SURFACE as VTK_PARAMETRIC_QUAD_SURFACE,
)
from vtkmodules.vtkCommonDataModel import VTK_PARAMETRIC_SURFACE as VTK_PARAMETRIC_SURFACE
from vtkmodules.vtkCommonDataModel import (
    VTK_PARAMETRIC_TETRA_REGION as VTK_PARAMETRIC_TETRA_REGION,
)
from vtkmodules.vtkCommonDataModel import VTK_PARAMETRIC_TRI_SURFACE as VTK_PARAMETRIC_TRI_SURFACE
from vtkmodules.vtkCommonDataModel import VTK_PENTAGONAL_PRISM as VTK_PENTAGONAL_PRISM
from vtkmodules.vtkCommonDataModel import VTK_PIXEL as VTK_PIXEL
from vtkmodules.vtkCommonDataModel import VTK_POLY_LINE as VTK_POLY_LINE
from vtkmodules.vtkCommonDataModel import VTK_POLY_VERTEX as VTK_POLY_VERTEX
from vtkmodules.vtkCommonDataModel import VTK_POLYGON as VTK_POLYGON
from vtkmodules.vtkCommonDataModel import VTK_POLYHEDRON as VTK_POLYHEDRON
from vtkmodules.vtkCommonDataModel import VTK_PYRAMID as VTK_PYRAMID
from vtkmodules.vtkCommonDataModel import VTK_QUAD as VTK_QUAD
from vtkmodules.vtkCommonDataModel import VTK_QUADRATIC_EDGE as VTK_QUADRATIC_EDGE
from vtkmodules.vtkCommonDataModel import VTK_QUADRATIC_HEXAHEDRON as VTK_QUADRATIC_HEXAHEDRON
from vtkmodules.vtkCommonDataModel import VTK_QUADRATIC_LINEAR_QUAD as VTK_QUADRATIC_LINEAR_QUAD
from vtkmodules.vtkCommonDataModel import VTK_QUADRATIC_LINEAR_WEDGE as VTK_QUADRATIC_LINEAR_WEDGE
from vtkmodules.vtkCommonDataModel import VTK_QUADRATIC_POLYGON as VTK_QUADRATIC_POLYGON
from vtkmodules.vtkCommonDataModel import VTK_QUADRATIC_PYRAMID as VTK_QUADRATIC_PYRAMID
from vtkmodules.vtkCommonDataModel import VTK_QUADRATIC_QUAD as VTK_QUADRATIC_QUAD
from vtkmodules.vtkCommonDataModel import VTK_QUADRATIC_TETRA as VTK_QUADRATIC_TETRA
from vtkmodules.vtkCommonDataModel import VTK_QUADRATIC_TRIANGLE as VTK_QUADRATIC_TRIANGLE
from vtkmodules.vtkCommonDataModel import VTK_QUADRATIC_WEDGE as VTK_QUADRATIC_WEDGE
from vtkmodules.vtkCommonDataModel import VTK_TETRA as VTK_TETRA
from vtkmodules.vtkCommonDataModel import VTK_TRIANGLE as VTK_TRIANGLE
from vtkmodules.vtkCommonDataModel import VTK_TRIANGLE_STRIP as VTK_TRIANGLE_STRIP
from vtkmodules.vtkCommonDataModel import (
    VTK_TRIQUADRATIC_HEXAHEDRON as VTK_TRIQUADRATIC_HEXAHEDRON,
)
from vtkmodules.vtkCommonDataModel import VTK_VERTEX as VTK_VERTEX
from vtkmodules.vtkCommonDataModel import VTK_VOXEL as VTK_VOXEL
from vtkmodules.vtkCommonDataModel import VTK_WEDGE as VTK_WEDGE
from vtkmodules.vtkCommonDataModel import vtkAbstractCellLocator as vtkAbstractCellLocator
from vtkmodules.vtkCommonDataModel import vtkBezierCurve as vtkBezierCurve
from vtkmodules.vtkCommonDataModel import vtkBezierHexahedron as vtkBezierHexahedron
from vtkmodules.vtkCommonDataModel import vtkBezierQuadrilateral as vtkBezierQuadrilateral
from vtkmodules.vtkCommonDataModel import vtkBezierTetra as vtkBezierTetra
from vtkmodules.vtkCommonDataModel import vtkBezierTriangle as vtkBezierTriangle
from vtkmodules.vtkCommonDataModel import vtkBezierWedge as vtkBezierWedge
from vtkmodules.vtkCommonDataModel import vtkBiQuadraticQuad as vtkBiQuadraticQuad
from vtkmodules.vtkCommonDataModel import (
    vtkBiQuadraticQuadraticHexahedron as vtkBiQuadraticQuadraticHexahedron,
)
from vtkmodules.vtkCommonDataModel import (
    vtkBiQuadraticQuadraticWedge as vtkBiQuadraticQuadraticWedge,
)
from vtkmodules.vtkCommonDataModel import vtkBiQuadraticTriangle as vtkBiQuadraticTriangle
from vtkmodules.vtkCommonDataModel import vtkCell as vtkCell
from vtkmodules.vtkCommonDataModel import vtkCellArray as vtkCellArray
from vtkmodules.vtkCommonDataModel import vtkCellLocator as vtkCellLocator
from vtkmodules.vtkCommonDataModel import vtkColor3ub as vtkColor3ub
from vtkmodules.vtkCommonDataModel import vtkCompositeDataSet as vtkCompositeDataSet
from vtkmodules.vtkCommonDataModel import vtkConvexPointSet as vtkConvexPointSet
from vtkmodules.vtkCommonDataModel import vtkCubicLine as vtkCubicLine
from vtkmodules.vtkCommonDataModel import vtkDataObject as vtkDataObject
from vtkmodules.vtkCommonDataModel import vtkDataSet as vtkDataSet
from vtkmodules.vtkCommonDataModel import vtkDataSetAttributes as vtkDataSetAttributes
from vtkmodules.vtkCommonDataModel import vtkEmptyCell as vtkEmptyCell
from vtkmodules.vtkCommonDataModel import vtkExplicitStructuredGrid as vtkExplicitStructuredGrid
from vtkmodules.vtkCommonDataModel import vtkFieldData as vtkFieldData
from vtkmodules.vtkCommonDataModel import vtkGenericCell as vtkGenericCell
from vtkmodules.vtkCommonDataModel import vtkHexagonalPrism as vtkHexagonalPrism
from vtkmodules.vtkCommonDataModel import vtkHexahedron as vtkHexahedron
from vtkmodules.vtkCommonDataModel import vtkImageData as vtkImageData
from vtkmodules.vtkCommonDataModel import vtkImplicitFunction as vtkImplicitFunction
from vtkmodules.vtkCommonDataModel import (
    vtkIterativeClosestPointTransform as vtkIterativeClosestPointTransform,
)
from vtkmodules.vtkCommonDataModel import vtkLagrangeCurve as vtkLagrangeCurve
from vtkmodules.vtkCommonDataModel import vtkLagrangeHexahedron as vtkLagrangeHexahedron
from vtkmodules.vtkCommonDataModel import vtkLagrangeQuadrilateral as vtkLagrangeQuadrilateral
from vtkmodules.vtkCommonDataModel import vtkLagrangeTriangle as vtkLagrangeTriangle
from vtkmodules.vtkCommonDataModel import vtkLagrangeWedge as vtkLagrangeWedge
from vtkmodules.vtkCommonDataModel import vtkLine as vtkLine
from vtkmodules.vtkCommonDataModel import vtkMultiBlockDataSet as vtkMultiBlockDataSet
from vtkmodules.vtkCommonDataModel import vtkNonMergingPointLocator as vtkNonMergingPointLocator
from vtkmodules.vtkCommonDataModel import vtkPartitionedDataSet as vtkPartitionedDataSet
from vtkmodules.vtkCommonDataModel import vtkPentagonalPrism as vtkPentagonalPrism
from vtkmodules.vtkCommonDataModel import vtkPerlinNoise as vtkPerlinNoise
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction as vtkPiecewiseFunction
from vtkmodules.vtkCommonDataModel import vtkPixel as vtkPixel
from vtkmodules.vtkCommonDataModel import vtkPlane as vtkPlane
from vtkmodules.vtkCommonDataModel import vtkPlaneCollection as vtkPlaneCollection
from vtkmodules.vtkCommonDataModel import vtkPlanes as vtkPlanes
from vtkmodules.vtkCommonDataModel import vtkPointLocator as vtkPointLocator
from vtkmodules.vtkCommonDataModel import vtkPointSet as vtkPointSet
from vtkmodules.vtkCommonDataModel import vtkPolyData as vtkPolyData
from vtkmodules.vtkCommonDataModel import vtkPolygon as vtkPolygon
from vtkmodules.vtkCommonDataModel import vtkPolyhedron as vtkPolyhedron
from vtkmodules.vtkCommonDataModel import vtkPolyLine as vtkPolyLine
from vtkmodules.vtkCommonDataModel import vtkPolyPlane as vtkPolyPlane
from vtkmodules.vtkCommonDataModel import vtkPolyVertex as vtkPolyVertex
from vtkmodules.vtkCommonDataModel import vtkPyramid as vtkPyramid
from vtkmodules.vtkCommonDataModel import vtkQuad as vtkQuad
from vtkmodules.vtkCommonDataModel import vtkQuadraticEdge as vtkQuadraticEdge
from vtkmodules.vtkCommonDataModel import vtkQuadraticHexahedron as vtkQuadraticHexahedron
from vtkmodules.vtkCommonDataModel import vtkQuadraticLinearQuad as vtkQuadraticLinearQuad
from vtkmodules.vtkCommonDataModel import vtkQuadraticLinearWedge as vtkQuadraticLinearWedge
from vtkmodules.vtkCommonDataModel import vtkQuadraticPolygon as vtkQuadraticPolygon
from vtkmodules.vtkCommonDataModel import vtkQuadraticPyramid as vtkQuadraticPyramid
from vtkmodules.vtkCommonDataModel import vtkQuadraticQuad as vtkQuadraticQuad
from vtkmodules.vtkCommonDataModel import vtkQuadraticTetra as vtkQuadraticTetra
from vtkmodules.vtkCommonDataModel import vtkQuadraticTriangle as vtkQuadraticTriangle
from vtkmodules.vtkCommonDataModel import vtkQuadraticWedge as vtkQuadraticWedge
from vtkmodules.vtkCommonDataModel import vtkRectf as vtkRectf
from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid as vtkRectilinearGrid
from vtkmodules.vtkCommonDataModel import vtkSelection as vtkSelection
from vtkmodules.vtkCommonDataModel import vtkSelectionNode as vtkSelectionNode
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator as vtkStaticCellLocator
from vtkmodules.vtkCommonDataModel import vtkStaticPointLocator as vtkStaticPointLocator
from vtkmodules.vtkCommonDataModel import vtkStructuredGrid as vtkStructuredGrid
from vtkmodules.vtkCommonDataModel import vtkStructuredPoints as vtkStructuredPoints
from vtkmodules.vtkCommonDataModel import vtkTable as vtkTable
from vtkmodules.vtkCommonDataModel import vtkTetra as vtkTetra
from vtkmodules.vtkCommonDataModel import vtkTriangle as vtkTriangle
from vtkmodules.vtkCommonDataModel import vtkTriangleStrip as vtkTriangleStrip
from vtkmodules.vtkCommonDataModel import vtkTriQuadraticHexahedron as vtkTriQuadraticHexahedron
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid as vtkUnstructuredGrid
from vtkmodules.vtkCommonDataModel import vtkVertex as vtkVertex
from vtkmodules.vtkCommonDataModel import vtkVoxel as vtkVoxel
from vtkmodules.vtkCommonDataModel import vtkWedge as vtkWedge

with contextlib.suppress(ImportError):  # Introduced prior to VTK 9.2
    from vtkmodules.vtkCommonDataModel import VTK_TRIQUADRATIC_PYRAMID as VTK_TRIQUADRATIC_PYRAMID
    from vtkmodules.vtkCommonDataModel import vtkTriQuadraticPyramid as vtkTriQuadraticPyramid

from vtkmodules.vtkCommonExecutionModel import vtkAlgorithm as vtkAlgorithm
from vtkmodules.vtkCommonExecutionModel import vtkAlgorithmOutput as vtkAlgorithmOutput
from vtkmodules.vtkCommonExecutionModel import vtkCompositeDataPipeline as vtkCompositeDataPipeline
from vtkmodules.vtkCommonExecutionModel import vtkImageToStructuredGrid as vtkImageToStructuredGrid
from vtkmodules.vtkCommonMath import vtkMatrix3x3 as vtkMatrix3x3
from vtkmodules.vtkCommonMath import vtkMatrix4x4 as vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform as vtkTransform
from vtkmodules.vtkDomainsChemistry import vtkProteinRibbonFilter as vtkProteinRibbonFilter
from vtkmodules.vtkFiltersCore import VTK_BEST_FITTING_PLANE as VTK_BEST_FITTING_PLANE
from vtkmodules.vtkFiltersCore import vtkAppendArcLength as vtkAppendArcLength
from vtkmodules.vtkFiltersCore import vtkAppendFilter as vtkAppendFilter
from vtkmodules.vtkFiltersCore import vtkAppendPolyData as vtkAppendPolyData
from vtkmodules.vtkFiltersCore import vtkCellCenters as vtkCellCenters
from vtkmodules.vtkFiltersCore import vtkCellDataToPointData as vtkCellDataToPointData
from vtkmodules.vtkFiltersCore import vtkCenterOfMass as vtkCenterOfMass
from vtkmodules.vtkFiltersCore import vtkCleanPolyData as vtkCleanPolyData
from vtkmodules.vtkFiltersCore import vtkClipPolyData as vtkClipPolyData
from vtkmodules.vtkFiltersCore import vtkConnectivityFilter as vtkConnectivityFilter
from vtkmodules.vtkFiltersCore import vtkContourFilter as vtkContourFilter
from vtkmodules.vtkFiltersCore import vtkCutter as vtkCutter
from vtkmodules.vtkFiltersCore import vtkDecimatePolylineFilter as vtkDecimatePolylineFilter
from vtkmodules.vtkFiltersCore import vtkDecimatePro as vtkDecimatePro
from vtkmodules.vtkFiltersCore import vtkDelaunay2D as vtkDelaunay2D
from vtkmodules.vtkFiltersCore import vtkDelaunay3D as vtkDelaunay3D
from vtkmodules.vtkFiltersCore import vtkElevationFilter as vtkElevationFilter
from vtkmodules.vtkFiltersCore import (
    vtkExplicitStructuredGridToUnstructuredGrid as vtkExplicitStructuredGridToUnstructuredGrid,
)
from vtkmodules.vtkFiltersCore import vtkFeatureEdges as vtkFeatureEdges
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D as vtkFlyingEdges3D
from vtkmodules.vtkFiltersCore import vtkGlyph3D as vtkGlyph3D
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance as vtkImplicitPolyDataDistance
from vtkmodules.vtkFiltersCore import vtkMarchingCubes as vtkMarchingCubes
from vtkmodules.vtkFiltersCore import vtkMassProperties as vtkMassProperties

with contextlib.suppress(ImportError):  # Introduced VTK 9.4
    from vtkmodules.vtkFiltersCore import vtkOrientPolyData as vtkOrientPolyData
from vtkmodules.vtkFiltersCore import vtkPointDataToCellData as vtkPointDataToCellData
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals as vtkPolyDataNormals
from vtkmodules.vtkFiltersCore import vtkQuadricDecimation as vtkQuadricDecimation
from vtkmodules.vtkFiltersCore import vtkResampleWithDataSet as vtkResampleWithDataSet
from vtkmodules.vtkFiltersCore import vtkReverseSense as vtkReverseSense
from vtkmodules.vtkFiltersCore import vtkSmoothPolyDataFilter as vtkSmoothPolyDataFilter
from vtkmodules.vtkFiltersCore import vtkStripper as vtkStripper
from vtkmodules.vtkFiltersCore import vtkThreshold as vtkThreshold
from vtkmodules.vtkFiltersCore import vtkTriangleFilter as vtkTriangleFilter
from vtkmodules.vtkFiltersCore import vtkTubeFilter as vtkTubeFilter
from vtkmodules.vtkFiltersCore import (
    vtkUnstructuredGridToExplicitStructuredGrid as vtkUnstructuredGridToExplicitStructuredGrid,
)
from vtkmodules.vtkFiltersCore import (
    vtkWindowedSincPolyDataFilter as vtkWindowedSincPolyDataFilter,
)
from vtkmodules.vtkFiltersExtraction import vtkExtractCellsByType as vtkExtractCellsByType
from vtkmodules.vtkFiltersExtraction import vtkExtractGeometry as vtkExtractGeometry
from vtkmodules.vtkFiltersExtraction import vtkExtractGrid as vtkExtractGrid
from vtkmodules.vtkFiltersExtraction import vtkExtractSelection as vtkExtractSelection
from vtkmodules.vtkFiltersFlowPaths import (
    vtkEvenlySpacedStreamlines2D as vtkEvenlySpacedStreamlines2D,
)
from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer as vtkStreamTracer

with contextlib.suppress(ImportError):  # Introduced VTK v9.1.0
    from vtkmodules.vtkFiltersGeneral import vtkRemovePolyData as vtkRemovePolyData

from vtkmodules.vtkFiltersGeneral import vtkAxes as vtkAxes
from vtkmodules.vtkFiltersGeneral import (
    vtkBooleanOperationPolyDataFilter as vtkBooleanOperationPolyDataFilter,
)
from vtkmodules.vtkFiltersGeneral import vtkBoxClipDataSet as vtkBoxClipDataSet
from vtkmodules.vtkFiltersGeneral import vtkClipClosedSurface as vtkClipClosedSurface
from vtkmodules.vtkFiltersGeneral import vtkContourTriangulator as vtkContourTriangulator
from vtkmodules.vtkFiltersGeneral import vtkCursor3D as vtkCursor3D
from vtkmodules.vtkFiltersGeneral import vtkCurvatures as vtkCurvatures
from vtkmodules.vtkFiltersGeneral import vtkDataSetTriangleFilter as vtkDataSetTriangleFilter
from vtkmodules.vtkFiltersGeneral import vtkGradientFilter as vtkGradientFilter
from vtkmodules.vtkFiltersGeneral import (
    vtkIntersectionPolyDataFilter as vtkIntersectionPolyDataFilter,
)
from vtkmodules.vtkFiltersGeneral import vtkOBBTree as vtkOBBTree
from vtkmodules.vtkFiltersGeneral import (
    vtkRectilinearGridToPointSet as vtkRectilinearGridToPointSet,
)
from vtkmodules.vtkFiltersGeneral import (
    vtkRectilinearGridToTetrahedra as vtkRectilinearGridToTetrahedra,
)
from vtkmodules.vtkFiltersGeneral import vtkShrinkFilter as vtkShrinkFilter
from vtkmodules.vtkFiltersGeneral import vtkTableBasedClipDataSet as vtkTableBasedClipDataSet
from vtkmodules.vtkFiltersGeneral import vtkTableToPolyData as vtkTableToPolyData
from vtkmodules.vtkFiltersGeneral import vtkTessellatorFilter as vtkTessellatorFilter
from vtkmodules.vtkFiltersGeneral import vtkTransformFilter as vtkTransformFilter
from vtkmodules.vtkFiltersGeneral import vtkWarpScalar as vtkWarpScalar
from vtkmodules.vtkFiltersGeneral import vtkWarpVector as vtkWarpVector
from vtkmodules.vtkFiltersGeometry import (
    vtkCompositeDataGeometryFilter as vtkCompositeDataGeometryFilter,
)
from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter as vtkDataSetSurfaceFilter
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter as vtkGeometryFilter
from vtkmodules.vtkFiltersGeometry import (
    vtkStructuredGridGeometryFilter as vtkStructuredGridGeometryFilter,
)
from vtkmodules.vtkFiltersHybrid import vtkPolyDataSilhouette as vtkPolyDataSilhouette
from vtkmodules.vtkFiltersModeling import (
    vtkAdaptiveSubdivisionFilter as vtkAdaptiveSubdivisionFilter,
)
from vtkmodules.vtkFiltersModeling import (
    vtkBandedPolyDataContourFilter as vtkBandedPolyDataContourFilter,
)
from vtkmodules.vtkFiltersModeling import (
    vtkButterflySubdivisionFilter as vtkButterflySubdivisionFilter,
)
from vtkmodules.vtkFiltersModeling import (
    vtkCollisionDetectionFilter as vtkCollisionDetectionFilter,
)
from vtkmodules.vtkFiltersModeling import (
    vtkDijkstraGraphGeodesicPath as vtkDijkstraGraphGeodesicPath,
)
from vtkmodules.vtkFiltersModeling import vtkFillHolesFilter as vtkFillHolesFilter
from vtkmodules.vtkFiltersModeling import vtkLinearExtrusionFilter as vtkLinearExtrusionFilter
from vtkmodules.vtkFiltersModeling import vtkLinearSubdivisionFilter as vtkLinearSubdivisionFilter
from vtkmodules.vtkFiltersModeling import vtkLoopSubdivisionFilter as vtkLoopSubdivisionFilter
from vtkmodules.vtkFiltersModeling import vtkOutlineFilter as vtkOutlineFilter
from vtkmodules.vtkFiltersModeling import vtkRibbonFilter as vtkRibbonFilter
from vtkmodules.vtkFiltersModeling import (
    vtkRotationalExtrusionFilter as vtkRotationalExtrusionFilter,
)
from vtkmodules.vtkFiltersModeling import vtkRuledSurfaceFilter as vtkRuledSurfaceFilter
from vtkmodules.vtkFiltersModeling import vtkSelectEnclosedPoints as vtkSelectEnclosedPoints
from vtkmodules.vtkFiltersModeling import vtkSubdivideTetra as vtkSubdivideTetra
from vtkmodules.vtkFiltersModeling import vtkTrimmedExtrusionFilter as vtkTrimmedExtrusionFilter
from vtkmodules.vtkFiltersParallel import vtkIntegrateAttributes as vtkIntegrateAttributes

with contextlib.suppress(ImportError):
    # `vtkmodules.vtkFiltersParallelDIY2` is unavailable in some versions of `vtk` from conda-forge
    from vtkmodules.vtkFiltersParallelDIY2 import (
        vtkRedistributeDataSetFilter as vtkRedistributeDataSetFilter,
    )

from vtkmodules.vtkFiltersPoints import vtkGaussianKernel as vtkGaussianKernel
from vtkmodules.vtkFiltersPoints import vtkPointInterpolator as vtkPointInterpolator
from vtkmodules.vtkFiltersSources import vtkArcSource as vtkArcSource
from vtkmodules.vtkFiltersSources import vtkArrowSource as vtkArrowSource

with contextlib.suppress(ImportError):
    # Deprecated in 9.3
    from vtkmodules.vtkFiltersSources import (  # type: ignore[attr-defined]
        vtkCapsuleSource as vtkCapsuleSource,
    )

from vtkmodules.vtkFiltersSources import vtkConeSource as vtkConeSource
from vtkmodules.vtkFiltersSources import vtkCubeSource as vtkCubeSource
from vtkmodules.vtkFiltersSources import vtkCylinderSource as vtkCylinderSource
from vtkmodules.vtkFiltersSources import vtkDiskSource as vtkDiskSource
from vtkmodules.vtkFiltersSources import vtkFrustumSource as vtkFrustumSource
from vtkmodules.vtkFiltersSources import vtkLineSource as vtkLineSource
from vtkmodules.vtkFiltersSources import vtkOutlineCornerFilter as vtkOutlineCornerFilter
from vtkmodules.vtkFiltersSources import vtkOutlineCornerSource as vtkOutlineCornerSource
from vtkmodules.vtkFiltersSources import vtkParametricFunctionSource as vtkParametricFunctionSource
from vtkmodules.vtkFiltersSources import vtkPlaneSource as vtkPlaneSource
from vtkmodules.vtkFiltersSources import vtkPlatonicSolidSource as vtkPlatonicSolidSource
from vtkmodules.vtkFiltersSources import vtkPointSource as vtkPointSource
from vtkmodules.vtkFiltersSources import vtkRegularPolygonSource as vtkRegularPolygonSource
from vtkmodules.vtkFiltersSources import vtkSphereSource as vtkSphereSource
from vtkmodules.vtkFiltersSources import vtkSuperquadricSource as vtkSuperquadricSource
from vtkmodules.vtkFiltersSources import vtkTessellatedBoxSource as vtkTessellatedBoxSource
from vtkmodules.vtkFiltersStatistics import vtkComputeQuartiles as vtkComputeQuartiles

with contextlib.suppress(ImportError):
    from vtkmodules.vtkFiltersStatistics import vtkLengthDistribution as vtkLengthDistribution
from vtkmodules.vtkFiltersTexture import vtkTextureMapToPlane as vtkTextureMapToPlane
from vtkmodules.vtkFiltersTexture import vtkTextureMapToSphere as vtkTextureMapToSphere
from vtkmodules.vtkFiltersVerdict import vtkCellQuality as vtkCellQuality
from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter as vtkCellSizeFilter
from vtkmodules.vtkFiltersVerdict import vtkMeshQuality as vtkMeshQuality

with contextlib.suppress(ImportError):
    from vtkmodules.vtkFiltersVerdict import vtkBoundaryMeshQuality as vtkBoundaryMeshQuality

from vtkmodules.vtkImagingCore import vtkAbstractImageInterpolator as vtkAbstractImageInterpolator
from vtkmodules.vtkImagingCore import vtkExtractVOI as vtkExtractVOI
from vtkmodules.vtkImagingCore import vtkImageConstantPad as vtkImageConstantPad
from vtkmodules.vtkImagingCore import vtkImageDifference as vtkImageDifference
from vtkmodules.vtkImagingCore import vtkImageExtractComponents as vtkImageExtractComponents
from vtkmodules.vtkImagingCore import vtkImageFlip as vtkImageFlip
from vtkmodules.vtkImagingCore import vtkImageInterpolator as vtkImageInterpolator
from vtkmodules.vtkImagingCore import vtkImageMirrorPad as vtkImageMirrorPad
from vtkmodules.vtkImagingCore import vtkImageResize as vtkImageResize
from vtkmodules.vtkImagingCore import vtkImageSincInterpolator as vtkImageSincInterpolator
from vtkmodules.vtkImagingCore import vtkImageThreshold as vtkImageThreshold
from vtkmodules.vtkImagingCore import vtkImageWrapPad as vtkImageWrapPad
from vtkmodules.vtkImagingCore import vtkRTAnalyticSource as vtkRTAnalyticSource
from vtkmodules.vtkImagingGeneral import vtkImageGaussianSmooth as vtkImageGaussianSmooth
from vtkmodules.vtkImagingGeneral import vtkImageMedian3D as vtkImageMedian3D
from vtkmodules.vtkImagingHybrid import vtkGaussianSplatter as vtkGaussianSplatter
from vtkmodules.vtkImagingHybrid import vtkSampleFunction as vtkSampleFunction
from vtkmodules.vtkImagingHybrid import (
    vtkSurfaceReconstructionFilter as vtkSurfaceReconstructionFilter,
)
from vtkmodules.vtkImagingMorphological import (
    vtkImageConnectivityFilter as vtkImageConnectivityFilter,
)
from vtkmodules.vtkImagingStencil import vtkImageStencil as vtkImageStencil
from vtkmodules.vtkImagingStencil import vtkPolyDataToImageStencil as vtkPolyDataToImageStencil
from vtkmodules.vtkIOGeometry import vtkHoudiniPolyDataWriter as vtkHoudiniPolyDataWriter
from vtkmodules.vtkIOGeometry import vtkIVWriter as vtkIVWriter
from vtkmodules.vtkIOGeometry import vtkOBJWriter as vtkOBJWriter
from vtkmodules.vtkIOGeometry import vtkProStarReader as vtkProStarReader
from vtkmodules.vtkIOGeometry import vtkSTLWriter as vtkSTLWriter

with contextlib.suppress(ImportError):  # Introduced VTK v9.4.0
    from vtkmodules.vtkIOHDF import vtkHDFWriter as vtkHDFWriter

from vtkmodules.vtkIOInfovis import vtkDelimitedTextReader as vtkDelimitedTextReader
from vtkmodules.vtkIOLegacy import vtkDataReader as vtkDataReader
from vtkmodules.vtkIOLegacy import vtkDataSetReader as vtkDataSetReader
from vtkmodules.vtkIOLegacy import vtkDataSetWriter as vtkDataSetWriter
from vtkmodules.vtkIOLegacy import vtkDataWriter as vtkDataWriter
from vtkmodules.vtkIOLegacy import vtkPolyDataReader as vtkPolyDataReader
from vtkmodules.vtkIOLegacy import vtkPolyDataWriter as vtkPolyDataWriter
from vtkmodules.vtkIOLegacy import vtkRectilinearGridReader as vtkRectilinearGridReader
from vtkmodules.vtkIOLegacy import vtkRectilinearGridWriter as vtkRectilinearGridWriter
from vtkmodules.vtkIOLegacy import vtkSimplePointsWriter as vtkSimplePointsWriter
from vtkmodules.vtkIOLegacy import vtkStructuredGridReader as vtkStructuredGridReader
from vtkmodules.vtkIOLegacy import vtkStructuredGridWriter as vtkStructuredGridWriter
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader as vtkUnstructuredGridReader
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridWriter as vtkUnstructuredGridWriter
from vtkmodules.vtkIOPLY import vtkPLYReader as vtkPLYReader
from vtkmodules.vtkIOPLY import vtkPLYWriter as vtkPLYWriter
from vtkmodules.vtkIOXML import vtkXMLImageDataReader as vtkXMLImageDataReader
from vtkmodules.vtkIOXML import vtkXMLImageDataWriter as vtkXMLImageDataWriter
from vtkmodules.vtkIOXML import vtkXMLMultiBlockDataReader as vtkXMLMultiBlockDataReader
from vtkmodules.vtkIOXML import vtkXMLMultiBlockDataWriter as vtkXMLMultiBlockDataWriter
from vtkmodules.vtkIOXML import vtkXMLPartitionedDataSetReader as vtkXMLPartitionedDataSetReader
from vtkmodules.vtkIOXML import vtkXMLPImageDataReader as vtkXMLPImageDataReader
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader as vtkXMLPolyDataReader
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter as vtkXMLPolyDataWriter
from vtkmodules.vtkIOXML import vtkXMLPRectilinearGridReader as vtkXMLPRectilinearGridReader
from vtkmodules.vtkIOXML import vtkXMLPUnstructuredGridReader as vtkXMLPUnstructuredGridReader
from vtkmodules.vtkIOXML import vtkXMLReader as vtkXMLReader
from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader as vtkXMLRectilinearGridReader
from vtkmodules.vtkIOXML import vtkXMLRectilinearGridWriter as vtkXMLRectilinearGridWriter
from vtkmodules.vtkIOXML import vtkXMLStructuredGridReader as vtkXMLStructuredGridReader
from vtkmodules.vtkIOXML import vtkXMLStructuredGridWriter as vtkXMLStructuredGridWriter
from vtkmodules.vtkIOXML import vtkXMLTableReader as vtkXMLTableReader
from vtkmodules.vtkIOXML import vtkXMLTableWriter as vtkXMLTableWriter
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader as vtkXMLUnstructuredGridReader
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter as vtkXMLUnstructuredGridWriter
from vtkmodules.vtkIOXML import vtkXMLWriter as vtkXMLWriter

with contextlib.suppress(ImportError):
    from vtkmodules.vtkImagingMorphological import (
        vtkImageContinuousDilate3D as vtkImageContinuousDilate3D,
    )
    from vtkmodules.vtkImagingMorphological import (
        vtkImageContinuousErode3D as vtkImageContinuousErode3D,
    )
    from vtkmodules.vtkImagingMorphological import vtkImageDilateErode3D as vtkImageDilateErode3D

try:
    from vtkmodules.vtkPythonContext2D import vtkPythonItem as vtkPythonItem
except ImportError:  # pragma: no cover
    # `vtkmodules.vtkPythonContext2D` is unavailable in some versions of `vtk` (see #3224)

    class vtkPythonItem:  # type: ignore[no-redef]  # noqa: N801
        """Empty placeholder."""

        def __init__(self):  # pragma: no cover
            """Raise version error on init."""
            from pyvista.core.errors import VTKVersionError  # noqa: PLC0415

            msg = 'Chart backgrounds require the vtkPythonContext2D module'
            raise VTKVersionError(msg)


from vtkmodules.vtkImagingFourier import vtkImageButterworthHighPass as vtkImageButterworthHighPass
from vtkmodules.vtkImagingFourier import vtkImageButterworthLowPass as vtkImageButterworthLowPass
from vtkmodules.vtkImagingFourier import vtkImageFFT as vtkImageFFT
from vtkmodules.vtkImagingFourier import vtkImageRFFT as vtkImageRFFT

# 9.1+ imports
with contextlib.suppress(ImportError):
    from vtkmodules.vtkFiltersPoints import vtkConvertToPointCloud as vtkConvertToPointCloud

with contextlib.suppress(ImportError):  # Introduced prior to VTK 9.3
    from vtkmodules.vtkRenderingCore import vtkViewport as vtkViewport

# 9.3+ imports
with contextlib.suppress(ImportError):
    from vtkmodules.vtkFiltersCore import vtkPackLabels as vtkPackLabels
    from vtkmodules.vtkFiltersCore import vtkSurfaceNets3D as vtkSurfaceNets3D

# 9.1+ imports
with contextlib.suppress(ImportError):
    from vtkmodules.vtkIOParallelXML import (
        vtkXMLPartitionedDataSetWriter as vtkXMLPartitionedDataSetWriter,
    )


class VersionInfo(NamedTuple):
    """Version information as a named tuple."""

    major: int
    minor: int
    micro: int

    def __str__(self):
        return str((self.major, self.minor, self.micro))


def VTKVersionInfo():  # noqa: N802
    """Return the vtk version as a namedtuple.

    Returns
    -------
    VersionInfo
        Version information as a named tuple.

    """
    try:
        ver = vtkVersion()
        major = ver.GetVTKMajorVersion()
        minor = ver.GetVTKMinorVersion()
        micro = ver.GetVTKBuildVersion()
    except AttributeError:  # pragma: no cover
        warnings.warn('Unable to detect VTK version. Defaulting to v4.0.0')
        major, minor, micro = (4, 0, 0)

    return VersionInfo(major, minor, micro)


vtk_version_info = VTKVersionInfo()


class vtkPyVistaOverride:  # noqa: N801
    """Base class to automatically override VTK classes with PyVista classes."""

    def __init_subclass__(cls, **kwargs):
        if vtk_version_info >= (9, 4):
            # Check for VTK base classes and call the override method
            for base in cls.__bases__:
                if (
                    hasattr(base, '__module__')
                    and base.__module__.startswith('vtkmodules.')
                    and hasattr(base, 'override')
                ):
                    # For now, just remove any overrides for these classes
                    # There are clear issues with the current implementation
                    # of overriding these classes upstream and until they are
                    # resolved, we will entirely remove the overrides.
                    # See https://gitlab.kitware.com/vtk/vtk/-/merge_requests/11698
                    # See https://gitlab.kitware.com/vtk/vtk/-/issues/19550#note_1598883
                    base.override(None)
                    break

        return cls


class DisableVtkSnakeCase:
    """Base class to raise error if using VTK's `snake_case` API."""

    @staticmethod
    def check_attribute(target, attr):
        # Check sys.meta_path to avoid dynamic imports when Python is shutting down
        if vtk_version_info >= (9, 4) and sys.meta_path is not None:
            # Raise error if accessing attributes from VTK's pythonic snake_case API

            import pyvista as pv  # noqa: PLC0415

            state = pv._VTK_SNAKE_CASE_STATE
            if state != 'allow':
                if (
                    attr not in ['__class__', '__init__']
                    and attr[0].islower()
                    and is_vtk_attribute(target, attr)
                ):
                    msg = (
                        f'The attribute {attr!r} is defined by VTK and is not part of the '
                        f'PyVista API'
                    )
                    if state == 'error':
                        raise pv.PyVistaAttributeError(msg)
                    else:
                        warnings.warn(msg, RuntimeWarning)

    def __getattribute__(self, item):
        DisableVtkSnakeCase.check_attribute(self, item)
        return object.__getattribute__(self, item)


def is_vtk_attribute(obj: object, attr: str):  # numpydoc ignore=RT01
    """Return True if the attribute is defined by a vtk class.

    Parameters
    ----------
    obj : object
        Class or instance to check.

    attr : str
        Name of the attribute to check.

    """

    def _find_defining_class(cls, attr):
        """Find the class that defines a given attribute."""
        for base in cls.__mro__:
            if attr in base.__dict__:
                return base
        return None

    cls = _find_defining_class(obj if isinstance(obj, type) else obj.__class__, attr)
    return cls is not None and cls.__module__.startswith('vtkmodules')


class VTKObjectWrapperCheckSnakeCase(VTKObjectWrapper):
    """Superclass for classes that wrap VTK objects with Python objects.

    This class overrides __getattr__ to disable the VTK snake case API.
    """

    def __getattr__(self, name: str):
        """Forward unknown attribute requests to VTKArray's __getattr__."""
        if self.VTKObject is not None:
            # Check if forwarding snake_case attributes
            DisableVtkSnakeCase.check_attribute(self.VTKObject, name)
            return getattr(self.VTKObject, name)
        raise AttributeError
