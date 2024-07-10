"""
Limited imports from VTK (excludes any GL-dependent).

These are the modules within VTK that must be loaded across pyvista's
core API. Here, we attempt to import modules using the ``vtkmodules``
package, which lets us only have to import from select modules and not
the entire library.

"""

from __future__ import annotations

from collections import namedtuple

# ruff: noqa: F401
import contextlib
from typing import NamedTuple
import warnings

from vtkmodules.vtkCommonCore import vtkVersion
from vtkmodules.vtkImagingSources import vtkImageEllipsoidSource
from vtkmodules.vtkImagingSources import vtkImageGaussianSource
from vtkmodules.vtkImagingSources import vtkImageGridSource
from vtkmodules.vtkImagingSources import vtkImageMandelbrotSource
from vtkmodules.vtkImagingSources import vtkImageNoiseSource
from vtkmodules.vtkImagingSources import vtkImageSinusoidSource

# vtkExtractEdges moved from vtkFiltersExtraction to vtkFiltersCore in
# VTK commit d9981b9aeb93b42d1371c6e295d76bfdc18430bd
try:
    from vtkmodules.vtkFiltersCore import vtkExtractEdges
except ImportError:
    from vtkmodules.vtkFiltersExtraction import vtkExtractEdges

# vtkCellTreeLocator moved from vtkFiltersGeneral to vtkCommonDataModel in
# VTK commit 4a29e6f7dd9acb460644fe487d2e80aac65f7be9
try:
    from vtkmodules.vtkCommonDataModel import vtkCellTreeLocator
except ImportError:
    from vtkmodules.vtkFiltersGeneral import vtkCellTreeLocator

from vtkmodules.numpy_interface.dataset_adapter import VTKArray
from vtkmodules.numpy_interface.dataset_adapter import VTKObjectWrapper
from vtkmodules.numpy_interface.dataset_adapter import numpyTovtkDataArray
from vtkmodules.util.numpy_support import get_vtk_array_type
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.util.numpy_support import numpy_to_vtkIdTypeArray
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonComputationalGeometry import vtkKochanekSpline
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricBohemianDome
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricBour
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricBoy
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricCatalanMinimal
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricConicSpiral
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricCrossCap
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricDini
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricEllipsoid
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricEnneper
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricFigure8Klein
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricFunction
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricHenneberg
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricKlein
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricKuen
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricMobius
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricPluckerConoid
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricPseudosphere
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricRandomHills
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricRoman
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricSpline
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricSuperEllipsoid
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricSuperToroid
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricTorus
from vtkmodules.vtkCommonCore import VTK_ARIAL
from vtkmodules.vtkCommonCore import VTK_COURIER
from vtkmodules.vtkCommonCore import VTK_FONT_FILE
from vtkmodules.vtkCommonCore import VTK_TIMES
from vtkmodules.vtkCommonCore import VTK_UNSIGNED_CHAR
from vtkmodules.vtkCommonCore import buffer_shared
from vtkmodules.vtkCommonCore import mutable
from vtkmodules.vtkCommonCore import reference
from vtkmodules.vtkCommonCore import vtkAbstractArray
from vtkmodules.vtkCommonCore import vtkBitArray
from vtkmodules.vtkCommonCore import vtkCharArray
from vtkmodules.vtkCommonCore import vtkCommand
from vtkmodules.vtkCommonCore import vtkDataArray
from vtkmodules.vtkCommonCore import vtkDoubleArray
from vtkmodules.vtkCommonCore import vtkFileOutputWindow
from vtkmodules.vtkCommonCore import vtkFloatArray
from vtkmodules.vtkCommonCore import vtkIdList
from vtkmodules.vtkCommonCore import vtkIdTypeArray
from vtkmodules.vtkCommonCore import vtkLogger
from vtkmodules.vtkCommonCore import vtkLookupTable
from vtkmodules.vtkCommonCore import vtkMath
from vtkmodules.vtkCommonCore import vtkOutputWindow
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonCore import vtkSignedCharArray
from vtkmodules.vtkCommonCore import vtkStringArray
from vtkmodules.vtkCommonCore import vtkStringOutputWindow
from vtkmodules.vtkCommonCore import vtkTypeInt32Array
from vtkmodules.vtkCommonCore import vtkTypeInt64Array
from vtkmodules.vtkCommonCore import vtkTypeUInt32Array
from vtkmodules.vtkCommonCore import vtkUnsignedCharArray
from vtkmodules.vtkCommonCore import vtkWeakReference
from vtkmodules.vtkCommonDataModel import VTK_BEZIER_CURVE
from vtkmodules.vtkCommonDataModel import VTK_BEZIER_HEXAHEDRON
from vtkmodules.vtkCommonDataModel import VTK_BEZIER_PYRAMID
from vtkmodules.vtkCommonDataModel import VTK_BEZIER_QUADRILATERAL
from vtkmodules.vtkCommonDataModel import VTK_BEZIER_TETRAHEDRON
from vtkmodules.vtkCommonDataModel import VTK_BEZIER_TRIANGLE
from vtkmodules.vtkCommonDataModel import VTK_BEZIER_WEDGE
from vtkmodules.vtkCommonDataModel import VTK_BIQUADRATIC_QUAD
from vtkmodules.vtkCommonDataModel import VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON
from vtkmodules.vtkCommonDataModel import VTK_BIQUADRATIC_QUADRATIC_WEDGE
from vtkmodules.vtkCommonDataModel import VTK_BIQUADRATIC_TRIANGLE
from vtkmodules.vtkCommonDataModel import VTK_CONVEX_POINT_SET
from vtkmodules.vtkCommonDataModel import VTK_CUBIC_LINE
from vtkmodules.vtkCommonDataModel import VTK_EMPTY_CELL
from vtkmodules.vtkCommonDataModel import VTK_HEXAGONAL_PRISM
from vtkmodules.vtkCommonDataModel import VTK_HEXAHEDRON
from vtkmodules.vtkCommonDataModel import VTK_HIGHER_ORDER_EDGE
from vtkmodules.vtkCommonDataModel import VTK_HIGHER_ORDER_HEXAHEDRON
from vtkmodules.vtkCommonDataModel import VTK_HIGHER_ORDER_POLYGON
from vtkmodules.vtkCommonDataModel import VTK_HIGHER_ORDER_PYRAMID
from vtkmodules.vtkCommonDataModel import VTK_HIGHER_ORDER_QUAD
from vtkmodules.vtkCommonDataModel import VTK_HIGHER_ORDER_TETRAHEDRON
from vtkmodules.vtkCommonDataModel import VTK_HIGHER_ORDER_TRIANGLE
from vtkmodules.vtkCommonDataModel import VTK_HIGHER_ORDER_WEDGE
from vtkmodules.vtkCommonDataModel import VTK_LAGRANGE_CURVE
from vtkmodules.vtkCommonDataModel import VTK_LAGRANGE_HEXAHEDRON
from vtkmodules.vtkCommonDataModel import VTK_LAGRANGE_PYRAMID
from vtkmodules.vtkCommonDataModel import VTK_LAGRANGE_QUADRILATERAL
from vtkmodules.vtkCommonDataModel import VTK_LAGRANGE_TETRAHEDRON
from vtkmodules.vtkCommonDataModel import VTK_LAGRANGE_TRIANGLE
from vtkmodules.vtkCommonDataModel import VTK_LAGRANGE_WEDGE
from vtkmodules.vtkCommonDataModel import VTK_LINE
from vtkmodules.vtkCommonDataModel import VTK_PARAMETRIC_CURVE
from vtkmodules.vtkCommonDataModel import VTK_PARAMETRIC_HEX_REGION
from vtkmodules.vtkCommonDataModel import VTK_PARAMETRIC_QUAD_SURFACE
from vtkmodules.vtkCommonDataModel import VTK_PARAMETRIC_SURFACE
from vtkmodules.vtkCommonDataModel import VTK_PARAMETRIC_TETRA_REGION
from vtkmodules.vtkCommonDataModel import VTK_PARAMETRIC_TRI_SURFACE
from vtkmodules.vtkCommonDataModel import VTK_PENTAGONAL_PRISM
from vtkmodules.vtkCommonDataModel import VTK_PIXEL
from vtkmodules.vtkCommonDataModel import VTK_POLY_LINE
from vtkmodules.vtkCommonDataModel import VTK_POLY_VERTEX
from vtkmodules.vtkCommonDataModel import VTK_POLYGON
from vtkmodules.vtkCommonDataModel import VTK_POLYHEDRON
from vtkmodules.vtkCommonDataModel import VTK_PYRAMID
from vtkmodules.vtkCommonDataModel import VTK_QUAD
from vtkmodules.vtkCommonDataModel import VTK_QUADRATIC_EDGE
from vtkmodules.vtkCommonDataModel import VTK_QUADRATIC_HEXAHEDRON
from vtkmodules.vtkCommonDataModel import VTK_QUADRATIC_LINEAR_QUAD
from vtkmodules.vtkCommonDataModel import VTK_QUADRATIC_LINEAR_WEDGE
from vtkmodules.vtkCommonDataModel import VTK_QUADRATIC_POLYGON
from vtkmodules.vtkCommonDataModel import VTK_QUADRATIC_PYRAMID
from vtkmodules.vtkCommonDataModel import VTK_QUADRATIC_QUAD
from vtkmodules.vtkCommonDataModel import VTK_QUADRATIC_TETRA
from vtkmodules.vtkCommonDataModel import VTK_QUADRATIC_TRIANGLE
from vtkmodules.vtkCommonDataModel import VTK_QUADRATIC_WEDGE
from vtkmodules.vtkCommonDataModel import VTK_TETRA
from vtkmodules.vtkCommonDataModel import VTK_TRIANGLE
from vtkmodules.vtkCommonDataModel import VTK_TRIANGLE_STRIP
from vtkmodules.vtkCommonDataModel import VTK_TRIQUADRATIC_HEXAHEDRON
from vtkmodules.vtkCommonDataModel import VTK_VERTEX
from vtkmodules.vtkCommonDataModel import VTK_VOXEL
from vtkmodules.vtkCommonDataModel import VTK_WEDGE
from vtkmodules.vtkCommonDataModel import vtkCell
from vtkmodules.vtkCommonDataModel import vtkCellArray
from vtkmodules.vtkCommonDataModel import vtkCellLocator
from vtkmodules.vtkCommonDataModel import vtkColor3ub
from vtkmodules.vtkCommonDataModel import vtkCompositeDataSet
from vtkmodules.vtkCommonDataModel import vtkDataObject
from vtkmodules.vtkCommonDataModel import vtkDataSet
from vtkmodules.vtkCommonDataModel import vtkDataSetAttributes
from vtkmodules.vtkCommonDataModel import vtkExplicitStructuredGrid
from vtkmodules.vtkCommonDataModel import vtkFieldData
from vtkmodules.vtkCommonDataModel import vtkGenericCell
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkCommonDataModel import vtkImplicitFunction
from vtkmodules.vtkCommonDataModel import vtkIterativeClosestPointTransform
from vtkmodules.vtkCommonDataModel import vtkMultiBlockDataSet
from vtkmodules.vtkCommonDataModel import vtkNonMergingPointLocator
from vtkmodules.vtkCommonDataModel import vtkPartitionedDataSet
from vtkmodules.vtkCommonDataModel import vtkPerlinNoise
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkCommonDataModel import vtkPlane
from vtkmodules.vtkCommonDataModel import vtkPlaneCollection
from vtkmodules.vtkCommonDataModel import vtkPlanes
from vtkmodules.vtkCommonDataModel import vtkPointLocator
from vtkmodules.vtkCommonDataModel import vtkPointSet
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkCommonDataModel import vtkPolyLine
from vtkmodules.vtkCommonDataModel import vtkPolyPlane
from vtkmodules.vtkCommonDataModel import vtkPyramid
from vtkmodules.vtkCommonDataModel import vtkRectf
from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid
from vtkmodules.vtkCommonDataModel import vtkSelection
from vtkmodules.vtkCommonDataModel import vtkSelectionNode
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator
from vtkmodules.vtkCommonDataModel import vtkStaticPointLocator
from vtkmodules.vtkCommonDataModel import vtkStructuredGrid
from vtkmodules.vtkCommonDataModel import vtkTable
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid

with contextlib.suppress(ImportError):  # Introduced prior to VTK 9.2
    from vtkmodules.vtkCommonDataModel import VTK_TRIQUADRATIC_PYRAMID

from vtkmodules.vtkCommonExecutionModel import vtkAlgorithm
from vtkmodules.vtkCommonExecutionModel import vtkAlgorithmOutput
from vtkmodules.vtkCommonExecutionModel import vtkCompositeDataPipeline
from vtkmodules.vtkCommonExecutionModel import vtkImageToStructuredGrid
from vtkmodules.vtkCommonMath import vtkMatrix3x3
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkDomainsChemistry import vtkProteinRibbonFilter
from vtkmodules.vtkFiltersCore import VTK_BEST_FITTING_PLANE
from vtkmodules.vtkFiltersCore import vtkAppendArcLength
from vtkmodules.vtkFiltersCore import vtkAppendFilter
from vtkmodules.vtkFiltersCore import vtkAppendPolyData
from vtkmodules.vtkFiltersCore import vtkCellCenters
from vtkmodules.vtkFiltersCore import vtkCellDataToPointData
from vtkmodules.vtkFiltersCore import vtkCenterOfMass
from vtkmodules.vtkFiltersCore import vtkCleanPolyData
from vtkmodules.vtkFiltersCore import vtkClipPolyData
from vtkmodules.vtkFiltersCore import vtkConnectivityFilter
from vtkmodules.vtkFiltersCore import vtkContourFilter
from vtkmodules.vtkFiltersCore import vtkCutter
from vtkmodules.vtkFiltersCore import vtkDecimatePro
from vtkmodules.vtkFiltersCore import vtkDelaunay2D
from vtkmodules.vtkFiltersCore import vtkDelaunay3D
from vtkmodules.vtkFiltersCore import vtkElevationFilter
from vtkmodules.vtkFiltersCore import vtkExplicitStructuredGridToUnstructuredGrid
from vtkmodules.vtkFiltersCore import vtkFeatureEdges
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D
from vtkmodules.vtkFiltersCore import vtkGlyph3D
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
from vtkmodules.vtkFiltersCore import vtkMarchingCubes
from vtkmodules.vtkFiltersCore import vtkMassProperties
from vtkmodules.vtkFiltersCore import vtkPointDataToCellData
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
from vtkmodules.vtkFiltersCore import vtkQuadricDecimation
from vtkmodules.vtkFiltersCore import vtkResampleWithDataSet
from vtkmodules.vtkFiltersCore import vtkSmoothPolyDataFilter
from vtkmodules.vtkFiltersCore import vtkStripper
from vtkmodules.vtkFiltersCore import vtkThreshold
from vtkmodules.vtkFiltersCore import vtkTriangleFilter
from vtkmodules.vtkFiltersCore import vtkTubeFilter
from vtkmodules.vtkFiltersCore import vtkUnstructuredGridToExplicitStructuredGrid
from vtkmodules.vtkFiltersCore import vtkWindowedSincPolyDataFilter
from vtkmodules.vtkFiltersExtraction import vtkExtractCellsByType
from vtkmodules.vtkFiltersExtraction import vtkExtractGeometry
from vtkmodules.vtkFiltersExtraction import vtkExtractGrid
from vtkmodules.vtkFiltersExtraction import vtkExtractSelection
from vtkmodules.vtkFiltersFlowPaths import vtkEvenlySpacedStreamlines2D
from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer

with contextlib.suppress(ImportError):  # Introduced VTK v9.1.0
    from vtkmodules.vtkFiltersGeneral import vtkRemovePolyData

from vtkmodules.vtkFiltersGeneral import vtkAxes
from vtkmodules.vtkFiltersGeneral import vtkBooleanOperationPolyDataFilter
from vtkmodules.vtkFiltersGeneral import vtkBoxClipDataSet
from vtkmodules.vtkFiltersGeneral import vtkClipClosedSurface
from vtkmodules.vtkFiltersGeneral import vtkContourTriangulator
from vtkmodules.vtkFiltersGeneral import vtkCursor3D
from vtkmodules.vtkFiltersGeneral import vtkCurvatures
from vtkmodules.vtkFiltersGeneral import vtkDataSetTriangleFilter
from vtkmodules.vtkFiltersGeneral import vtkGradientFilter
from vtkmodules.vtkFiltersGeneral import vtkIntersectionPolyDataFilter
from vtkmodules.vtkFiltersGeneral import vtkOBBTree
from vtkmodules.vtkFiltersGeneral import vtkRectilinearGridToPointSet
from vtkmodules.vtkFiltersGeneral import vtkRectilinearGridToTetrahedra
from vtkmodules.vtkFiltersGeneral import vtkShrinkFilter
from vtkmodules.vtkFiltersGeneral import vtkTableBasedClipDataSet
from vtkmodules.vtkFiltersGeneral import vtkTableToPolyData
from vtkmodules.vtkFiltersGeneral import vtkTessellatorFilter
from vtkmodules.vtkFiltersGeneral import vtkTransformFilter
from vtkmodules.vtkFiltersGeneral import vtkWarpScalar
from vtkmodules.vtkFiltersGeneral import vtkWarpVector
from vtkmodules.vtkFiltersGeometry import vtkCompositeDataGeometryFilter
from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkFiltersGeometry import vtkStructuredGridGeometryFilter
from vtkmodules.vtkFiltersHybrid import vtkPolyDataSilhouette
from vtkmodules.vtkFiltersModeling import vtkAdaptiveSubdivisionFilter
from vtkmodules.vtkFiltersModeling import vtkBandedPolyDataContourFilter
from vtkmodules.vtkFiltersModeling import vtkButterflySubdivisionFilter
from vtkmodules.vtkFiltersModeling import vtkCollisionDetectionFilter
from vtkmodules.vtkFiltersModeling import vtkDijkstraGraphGeodesicPath
from vtkmodules.vtkFiltersModeling import vtkFillHolesFilter
from vtkmodules.vtkFiltersModeling import vtkLinearExtrusionFilter
from vtkmodules.vtkFiltersModeling import vtkLinearSubdivisionFilter
from vtkmodules.vtkFiltersModeling import vtkLoopSubdivisionFilter
from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
from vtkmodules.vtkFiltersModeling import vtkRibbonFilter
from vtkmodules.vtkFiltersModeling import vtkRotationalExtrusionFilter
from vtkmodules.vtkFiltersModeling import vtkSelectEnclosedPoints
from vtkmodules.vtkFiltersModeling import vtkSubdivideTetra
from vtkmodules.vtkFiltersModeling import vtkTrimmedExtrusionFilter
from vtkmodules.vtkFiltersParallel import vtkIntegrateAttributes

with contextlib.suppress(ImportError):
    # `vtkmodules.vtkFiltersParallelDIY2` is unavailable in some versions of `vtk` from conda-forge
    from vtkmodules.vtkFiltersParallelDIY2 import vtkRedistributeDataSetFilter

from vtkmodules.vtkFiltersPoints import vtkGaussianKernel
from vtkmodules.vtkFiltersPoints import vtkPointInterpolator
from vtkmodules.vtkFiltersSources import vtkArcSource
from vtkmodules.vtkFiltersSources import vtkArrowSource
from vtkmodules.vtkFiltersSources import vtkCapsuleSource
from vtkmodules.vtkFiltersSources import vtkConeSource
from vtkmodules.vtkFiltersSources import vtkCubeSource
from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkFiltersSources import vtkDiskSource
from vtkmodules.vtkFiltersSources import vtkFrustumSource
from vtkmodules.vtkFiltersSources import vtkLineSource
from vtkmodules.vtkFiltersSources import vtkOutlineCornerFilter
from vtkmodules.vtkFiltersSources import vtkOutlineCornerSource
from vtkmodules.vtkFiltersSources import vtkParametricFunctionSource
from vtkmodules.vtkFiltersSources import vtkPlaneSource
from vtkmodules.vtkFiltersSources import vtkPlatonicSolidSource
from vtkmodules.vtkFiltersSources import vtkPointSource
from vtkmodules.vtkFiltersSources import vtkRegularPolygonSource
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkFiltersSources import vtkSuperquadricSource
from vtkmodules.vtkFiltersSources import vtkTessellatedBoxSource
from vtkmodules.vtkFiltersStatistics import vtkComputeQuartiles
from vtkmodules.vtkFiltersTexture import vtkTextureMapToPlane
from vtkmodules.vtkFiltersTexture import vtkTextureMapToSphere
from vtkmodules.vtkFiltersVerdict import vtkCellQuality
from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter

with contextlib.suppress(ImportError):
    from vtkmodules.vtkFiltersVerdict import vtkBoundaryMeshQuality

from vtkmodules.vtkImagingCore import vtkExtractVOI
from vtkmodules.vtkImagingCore import vtkImageConstantPad
from vtkmodules.vtkImagingCore import vtkImageDifference
from vtkmodules.vtkImagingCore import vtkImageExtractComponents
from vtkmodules.vtkImagingCore import vtkImageFlip
from vtkmodules.vtkImagingCore import vtkImageMirrorPad
from vtkmodules.vtkImagingCore import vtkImageThreshold
from vtkmodules.vtkImagingCore import vtkImageWrapPad
from vtkmodules.vtkImagingCore import vtkRTAnalyticSource
from vtkmodules.vtkImagingGeneral import vtkImageGaussianSmooth
from vtkmodules.vtkImagingGeneral import vtkImageMedian3D
from vtkmodules.vtkImagingHybrid import vtkSampleFunction
from vtkmodules.vtkImagingHybrid import vtkSurfaceReconstructionFilter
from vtkmodules.vtkIOGeometry import vtkHoudiniPolyDataWriter
from vtkmodules.vtkIOGeometry import vtkIVWriter
from vtkmodules.vtkIOGeometry import vtkOBJWriter
from vtkmodules.vtkIOGeometry import vtkProStarReader
from vtkmodules.vtkIOGeometry import vtkSTLWriter
from vtkmodules.vtkIOInfovis import vtkDelimitedTextReader
from vtkmodules.vtkIOLegacy import vtkDataReader
from vtkmodules.vtkIOLegacy import vtkDataSetReader
from vtkmodules.vtkIOLegacy import vtkDataSetWriter
from vtkmodules.vtkIOLegacy import vtkDataWriter
from vtkmodules.vtkIOLegacy import vtkPolyDataReader
from vtkmodules.vtkIOLegacy import vtkPolyDataWriter
from vtkmodules.vtkIOLegacy import vtkRectilinearGridReader
from vtkmodules.vtkIOLegacy import vtkRectilinearGridWriter
from vtkmodules.vtkIOLegacy import vtkSimplePointsWriter
from vtkmodules.vtkIOLegacy import vtkStructuredGridReader
from vtkmodules.vtkIOLegacy import vtkStructuredGridWriter
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridWriter
from vtkmodules.vtkIOPLY import vtkPLYReader
from vtkmodules.vtkIOPLY import vtkPLYWriter
from vtkmodules.vtkIOXML import vtkXMLImageDataReader
from vtkmodules.vtkIOXML import vtkXMLImageDataWriter
from vtkmodules.vtkIOXML import vtkXMLMultiBlockDataReader
from vtkmodules.vtkIOXML import vtkXMLMultiBlockDataWriter
from vtkmodules.vtkIOXML import vtkXMLPartitionedDataSetReader
from vtkmodules.vtkIOXML import vtkXMLPImageDataReader
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
from vtkmodules.vtkIOXML import vtkXMLPRectilinearGridReader
from vtkmodules.vtkIOXML import vtkXMLPUnstructuredGridReader
from vtkmodules.vtkIOXML import vtkXMLReader
from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
from vtkmodules.vtkIOXML import vtkXMLRectilinearGridWriter
from vtkmodules.vtkIOXML import vtkXMLStructuredGridReader
from vtkmodules.vtkIOXML import vtkXMLStructuredGridWriter
from vtkmodules.vtkIOXML import vtkXMLTableReader
from vtkmodules.vtkIOXML import vtkXMLTableWriter
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter
from vtkmodules.vtkIOXML import vtkXMLWriter

with contextlib.suppress(ImportError):
    from vtkmodules.vtkImagingMorphological import vtkImageDilateErode3D

try:
    from vtkmodules.vtkPythonContext2D import vtkPythonItem
except ImportError:  # pragma: no cover
    # `vtkmodules.vtkPythonContext2D` is unavailable in some versions of `vtk` (see #3224)

    class vtkPythonItem:  # type: ignore[no-redef]
        """Empty placeholder."""

        def __init__(self):  # pragma: no cover
            """Raise version error on init."""
            from pyvista.core.errors import VTKVersionError

            raise VTKVersionError('Chart backgrounds require the vtkPythonContext2D module')


from vtkmodules.vtkImagingFourier import vtkImageButterworthHighPass
from vtkmodules.vtkImagingFourier import vtkImageButterworthLowPass
from vtkmodules.vtkImagingFourier import vtkImageFFT
from vtkmodules.vtkImagingFourier import vtkImageRFFT

# 9.1+ imports
with contextlib.suppress(ImportError):
    from vtkmodules.vtkFiltersPoints import vtkConvertToPointCloud

with contextlib.suppress(ImportError):  # Introduced prior to VTK 9.3
    from vtkmodules.vtkRenderingCore import vtkViewport

# 9.3+ imports
with contextlib.suppress(ImportError):
    from vtkmodules.vtkFiltersCore import vtkPackLabels
    from vtkmodules.vtkFiltersCore import vtkSurfaceNets3D

# 9.1+ imports
with contextlib.suppress(ImportError):
    from vtkmodules.vtkIOParallelXML import vtkXMLPartitionedDataSetWriter


class VersionInfo(NamedTuple):
    """Version information as a named tuple."""

    major: int
    minor: int
    micro: int


def VTKVersionInfo():
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
        warnings.warn("Unable to detect VTK version. Defaulting to v4.0.0")
        major, minor, micro = (4, 0, 0)

    return VersionInfo(major, minor, micro)


vtk_version_info = VTKVersionInfo()
