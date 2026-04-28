"""Limited imports from VTK (excludes any GL-dependent).

These are the modules within VTK that must be loaded across pyvista's
core API. Here, we attempt to import modules using the ``vtkmodules``
package, which lets us only have to import from select modules and not
the entire library.

"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type checkers cannot resolve the dynamic lazy vtk imports, so we import everything
    # here as needed to ensure they can "see" the imported classes
    from vtk import *  # noqa: TID251
    from vtkmodules.numpy_interface.dataset_adapter import *
    from vtkmodules.util.vtkAlgorithm import *

_THIS_MODULE = sys.modules[__name__]

# Canonical mapping: vtkmodule -> classes
_VTK_MODULES: dict[str, tuple[str, ...]] = {
    'numpy_interface.dataset_adapter': (
        'VTKArray',
        'VTKObjectWrapper',
        'numpyTovtkDataArray',
    ),
    'util.numpy_support': (
        'get_vtk_array_type',
        'numpy_to_vtk',
        'numpy_to_vtkIdTypeArray',
        'vtk_to_numpy',
    ),
    'util.pickle_support': ('serialize_VTK_data_object',),
    'util.vtkAlgorithm': ('VTKPythonAlgorithmBase',),
    'vtkCommonComputationalGeometry': (
        'vtkKochanekSpline',
        'vtkParametricBohemianDome',
        'vtkParametricBour',
        'vtkParametricBoy',
        'vtkParametricCatalanMinimal',
        'vtkParametricConicSpiral',
        'vtkParametricCrossCap',
        'vtkParametricDini',
        'vtkParametricEllipsoid',
        'vtkParametricEnneper',
        'vtkParametricFigure8Klein',
        'vtkParametricFunction',
        'vtkParametricHenneberg',
        'vtkParametricKlein',
        'vtkParametricKuen',
        'vtkParametricMobius',
        'vtkParametricPluckerConoid',
        'vtkParametricPseudosphere',
        'vtkParametricRandomHills',
        'vtkParametricRoman',
        'vtkParametricSpline',
        'vtkParametricSuperEllipsoid',
        'vtkParametricSuperToroid',
        'vtkParametricTorus',
    ),
    'vtkCommonCore': (
        'VTK_ARIAL',
        'VTK_COURIER',
        'VTK_DOUBLE_MAX',
        'VTK_DOUBLE_MIN',
        'VTK_FONT_FILE',
        'VTK_TIMES',
        'VTK_UNSIGNED_CHAR',
        'buffer_shared',
        'mutable',
        'reference',
        'vtkAbstractArray',
        'vtkBitArray',
        'vtkCharArray',
        'vtkCommand',
        'vtkDataArray',
        'vtkDoubleArray',
        'vtkFileOutputWindow',
        'vtkFloatArray',
        'vtkIdList',
        'vtkIdTypeArray',
        'vtkInformation',
        'vtkLogger',
        'vtkLookupTable',
        'vtkMath',
        'vtkObjectBase',
        'vtkOutputWindow',
        'vtkPoints',
        'vtkSMPTools',
        'vtkSignedCharArray',
        'vtkStringArray',
        'vtkStringOutputWindow',
        'vtkTypeInt32Array',
        'vtkTypeInt64Array',
        'vtkTypeUInt32Array',
        'vtkUnsignedCharArray',
        'vtkVersion',
        'vtkWeakReference',
    ),
    'vtkCommonDataModel': (
        'VTK_BEZIER_CURVE',
        'VTK_BEZIER_HEXAHEDRON',
        'VTK_BEZIER_PYRAMID',
        'VTK_BEZIER_QUADRILATERAL',
        'VTK_BEZIER_TETRAHEDRON',
        'VTK_BEZIER_TRIANGLE',
        'VTK_BEZIER_WEDGE',
        'VTK_BIQUADRATIC_QUAD',
        'VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON',
        'VTK_BIQUADRATIC_QUADRATIC_WEDGE',
        'VTK_BIQUADRATIC_TRIANGLE',
        'VTK_CONVEX_POINT_SET',
        'VTK_CUBIC_LINE',
        'VTK_EMPTY_CELL',
        'VTK_HEXAGONAL_PRISM',
        'VTK_HEXAHEDRON',
        'VTK_HIGHER_ORDER_EDGE',
        'VTK_HIGHER_ORDER_HEXAHEDRON',
        'VTK_HIGHER_ORDER_POLYGON',
        'VTK_HIGHER_ORDER_PYRAMID',
        'VTK_HIGHER_ORDER_QUAD',
        'VTK_HIGHER_ORDER_TETRAHEDRON',
        'VTK_HIGHER_ORDER_TRIANGLE',
        'VTK_HIGHER_ORDER_WEDGE',
        'VTK_LAGRANGE_CURVE',
        'VTK_LAGRANGE_HEXAHEDRON',
        'VTK_LAGRANGE_PYRAMID',
        'VTK_LAGRANGE_QUADRILATERAL',
        'VTK_LAGRANGE_TETRAHEDRON',
        'VTK_LAGRANGE_TRIANGLE',
        'VTK_LAGRANGE_WEDGE',
        'VTK_LINE',
        'VTK_PARAMETRIC_CURVE',
        'VTK_PARAMETRIC_HEX_REGION',
        'VTK_PARAMETRIC_QUAD_SURFACE',
        'VTK_PARAMETRIC_SURFACE',
        'VTK_PARAMETRIC_TETRA_REGION',
        'VTK_PARAMETRIC_TRI_SURFACE',
        'VTK_PENTAGONAL_PRISM',
        'VTK_PIXEL',
        'VTK_POLY_LINE',
        'VTK_POLY_VERTEX',
        'VTK_POLYGON',
        'VTK_POLYHEDRON',
        'VTK_PYRAMID',
        'VTK_QUAD',
        'VTK_QUADRATIC_EDGE',
        'VTK_QUADRATIC_HEXAHEDRON',
        'VTK_QUADRATIC_LINEAR_QUAD',
        'VTK_QUADRATIC_LINEAR_WEDGE',
        'VTK_QUADRATIC_POLYGON',
        'VTK_QUADRATIC_PYRAMID',
        'VTK_QUADRATIC_QUAD',
        'VTK_QUADRATIC_TETRA',
        'VTK_QUADRATIC_TRIANGLE',
        'VTK_QUADRATIC_WEDGE',
        'VTK_TETRA',
        'VTK_TRIANGLE',
        'VTK_TRIANGLE_STRIP',
        'VTK_TRIQUADRATIC_HEXAHEDRON',
        'VTK_TRIQUADRATIC_PYRAMID',
        'VTK_VERTEX',
        'VTK_VOXEL',
        'VTK_WEDGE',
        'vtkAbstractCellLocator',
        'vtkBezierCurve',
        'vtkBezierHexahedron',
        'vtkBezierQuadrilateral',
        'vtkBezierTetra',
        'vtkBezierTriangle',
        'vtkBezierWedge',
        'vtkBiQuadraticQuad',
        'vtkBiQuadraticQuadraticHexahedron',
        'vtkBiQuadraticQuadraticWedge',
        'vtkBiQuadraticTriangle',
        'vtkCell',
        'vtkCellArray',
        'vtkCellLocator',
        'vtkCellTreeLocator',
        'vtkCellTypes',
        'vtkColor3ub',
        'vtkCompositeDataSet',
        'vtkConvexPointSet',
        'vtkCubicLine',
        'vtkDataObject',
        'vtkDataSet',
        'vtkDataSetAttributes',
        'vtkEmptyCell',
        'vtkExplicitStructuredGrid',
        'vtkFieldData',
        'vtkGenericCell',
        'vtkHexagonalPrism',
        'vtkHexahedron',
        'vtkImageData',
        'vtkImplicitBoolean',
        'vtkImplicitFunction',
        'vtkIterativeClosestPointTransform',
        'vtkLagrangeCurve',
        'vtkLagrangeHexahedron',
        'vtkLagrangeQuadrilateral',
        'vtkLagrangeTetra',
        'vtkLagrangeTriangle',
        'vtkLagrangeWedge',
        'vtkLine',
        'vtkMultiBlockDataSet',
        'vtkNonMergingPointLocator',
        'vtkPartitionedDataSet',
        'vtkPentagonalPrism',
        'vtkPerlinNoise',
        'vtkPiecewiseFunction',
        'vtkPixel',
        'vtkPlane',
        'vtkPlaneCollection',
        'vtkPlanes',
        'vtkPointLocator',
        'vtkPointSet',
        'vtkPolyData',
        'vtkPolygon',
        'vtkPolyhedron',
        'vtkPolyLine',
        'vtkPolyPlane',
        'vtkPolyVertex',
        'vtkPyramid',
        'vtkQuad',
        'vtkQuadraticEdge',
        'vtkQuadraticHexahedron',
        'vtkQuadraticLinearQuad',
        'vtkQuadraticLinearWedge',
        'vtkQuadraticPolygon',
        'vtkQuadraticPyramid',
        'vtkQuadraticQuad',
        'vtkQuadraticTetra',
        'vtkQuadraticTriangle',
        'vtkQuadraticWedge',
        'vtkRectf',
        'vtkRectilinearGrid',
        'vtkSelection',
        'vtkSelectionNode',
        'vtkStaticCellLocator',
        'vtkStaticPointLocator',
        'vtkStructuredGrid',
        'vtkStructuredPoints',
        'vtkTable',
        'vtkTetra',
        'vtkTriangle',
        'vtkTriangleStrip',
        'vtkTriQuadraticHexahedron',
        'vtkTriQuadraticPyramid',
        'vtkUnstructuredGrid',
        'vtkVertex',
        'vtkVoxel',
        'vtkWedge',
    ),
    'vtkCommonExecutionModel': (
        'vtkAlgorithm',
        'vtkAlgorithmOutput',
        'vtkCompositeDataPipeline',
        'vtkImageToStructuredGrid',
    ),
    'vtkCommonMath': (
        'vtkMatrix3x3',
        'vtkMatrix4x4',
    ),
    'vtkCommonTransforms': ('vtkTransform',),
    'vtkDomainsChemistry': ('vtkProteinRibbonFilter',),
    'vtkFiltersCore': (
        'VTK_BEST_FITTING_PLANE',
        'vtkAppendArcLength',
        'vtkAppendFilter',
        'vtkAppendPolyData',
        'vtkCellCenters',
        'vtkCellDataToPointData',
        'vtkCenterOfMass',
        'vtkCleanPolyData',
        'vtkClipPolyData',
        'vtkConnectivityFilter',
        'vtkContourFilter',
        'vtkConvertToMultiBlockDataSet',
        'vtkCutter',
        'vtkDecimatePolylineFilter',
        'vtkDecimatePro',
        'vtkDelaunay2D',
        'vtkDelaunay3D',
        'vtkElevationFilter',
        'vtkExplicitStructuredGridToUnstructuredGrid',
        'vtkExtractEdges',
        'vtkFeatureEdges',
        'vtkFlyingEdges3D',
        'vtkGlyph3D',
        'vtkImageAppend',
        'vtkImplicitPolyDataDistance',
        'vtkMarchingCubes',
        'vtkMassProperties',
        'vtkOrientPolyData',
        'vtkPackLabels',
        'vtkPointDataToCellData',
        'vtkPolyDataNormals',
        'vtkQuadricDecimation',
        'vtkResampleWithDataSet',
        'vtkReverseSense',
        'vtkSmoothPolyDataFilter',
        'vtkStripper',
        'vtkSurfaceNets3D',
        'vtkThreshold',
        'vtkTriangleFilter',
        'vtkTubeFilter',
        'vtkUnstructuredGridToExplicitStructuredGrid',
        'vtkWindowedSincPolyDataFilter',
    ),
    'vtkFiltersExtraction': (
        'vtkExtractCellsByType',
        'vtkExtractGeometry',
        'vtkExtractGrid',
        'vtkExtractSelection',
    ),
    'vtkFiltersFlowPaths': (
        'vtkEvenlySpacedStreamlines2D',
        'vtkStreamTracer',
    ),
    'vtkFiltersGeneral': (
        'vtkAxes',
        'vtkBooleanOperationPolyDataFilter',
        'vtkBoxClipDataSet',
        'vtkCellValidator',
        'vtkClipClosedSurface',
        'vtkContourTriangulator',
        'vtkCursor3D',
        'vtkCurvatures',
        'vtkDataSetTriangleFilter',
        'vtkGradientFilter',
        'vtkIntersectionPolyDataFilter',
        'vtkOBBTree',
        'vtkRectilinearGridToPointSet',
        'vtkRectilinearGridToTetrahedra',
        'vtkRemovePolyData',
        'vtkShrinkFilter',
        'vtkTableBasedClipDataSet',
        'vtkTableToPolyData',
        'vtkTessellatorFilter',
        'vtkTransformFilter',
        'vtkWarpScalar',
        'vtkWarpVector',
    ),
    'vtkFiltersGeometry': (
        'vtkCompositeDataGeometryFilter',
        'vtkDataSetSurfaceFilter',
        'vtkGeometryFilter',
        'vtkStructuredGridGeometryFilter',
    ),
    'vtkFiltersHybrid': ('vtkPolyDataSilhouette',),
    'vtkFiltersModeling': (
        'vtkAdaptiveSubdivisionFilter',
        'vtkBandedPolyDataContourFilter',
        'vtkButterflySubdivisionFilter',
        'vtkCollisionDetectionFilter',
        'vtkDijkstraGraphGeodesicPath',
        'vtkFillHolesFilter',
        'vtkLinearExtrusionFilter',
        'vtkLinearSubdivisionFilter',
        'vtkLoopSubdivisionFilter',
        'vtkOutlineFilter',
        'vtkRibbonFilter',
        'vtkRotationalExtrusionFilter',
        'vtkRuledSurfaceFilter',
        'vtkSelectEnclosedPoints',
        'vtkSubdivideTetra',
        'vtkTrimmedExtrusionFilter',
    ),
    'vtkFiltersParallel': ('vtkIntegrateAttributes',),
    'vtkFiltersParallelDIY2': ('vtkRedistributeDataSetFilter',),
    'vtkFiltersPoints': (
        'vtkConvertToPointCloud',
        'vtkGaussianKernel',
        'vtkPointInterpolator',
    ),
    'vtkFiltersSources': (
        'vtkArcSource',
        'vtkArrowSource',
        'vtkCapsuleSource',  # Deprecated 9.3.0
        'vtkCellTypeSource',
        'vtkConeSource',
        'vtkCubeSource',
        'vtkCylinderSource',
        'vtkDiskSource',
        'vtkFrustumSource',
        'vtkLineSource',
        'vtkOutlineCornerFilter',
        'vtkOutlineCornerSource',
        'vtkParametricFunctionSource',
        'vtkPlaneSource',
        'vtkPlatonicSolidSource',
        'vtkPointSource',
        'vtkRegularPolygonSource',
        'vtkSphereSource',
        'vtkSuperquadricSource',
        'vtkTessellatedBoxSource',
    ),
    'vtkFiltersStatistics': (
        'vtkComputeQuartiles',
        'vtkLengthDistribution',
    ),
    'vtkFiltersTexture': (
        'vtkTextureMapToPlane',
        'vtkTextureMapToSphere',
    ),
    'vtkFiltersVerdict': (
        'vtkBoundaryMeshQuality',
        'vtkCellQuality',
        'vtkCellSizeFilter',
        'vtkMeshQuality',
    ),
    'vtkImagingCore': (
        'vtkAbstractImageInterpolator',
        'vtkExtractVOI',
        'vtkImageBinaryThreshold',
        'vtkImageBSplineCoefficients',
        'vtkImageBSplineInterpolator',
        'vtkImageConstantPad',
        'vtkImageDifference',
        'vtkImageExtractComponents',
        'vtkImageFlip',
        'vtkImageInterpolator',
        'vtkImageMirrorPad',
        'vtkImageResize',
        'vtkImageSincInterpolator',
        'vtkImageThreshold',
        'vtkImageWrapPad',
        'vtkRTAnalyticSource',
    ),
    'vtkImagingFourier': (
        'vtkImageButterworthHighPass',
        'vtkImageButterworthLowPass',
        'vtkImageFFT',
        'vtkImageRFFT',
    ),
    'vtkImagingGeneral': (
        'vtkImageGaussianSmooth',
        'vtkImageMedian3D',
    ),
    'vtkImagingHybrid': (
        'vtkGaussianSplatter',
        'vtkSampleFunction',
        'vtkSurfaceReconstructionFilter',
    ),
    'vtkImagingMorphological': (
        'vtkImageConnectivityFilter',
        'vtkImageContinuousDilate3D',
        'vtkImageContinuousErode3D',
        'vtkImageDilateErode3D',
    ),
    'vtkImagingSources': (
        'vtkImageEllipsoidSource',
        'vtkImageGaussianSource',
        'vtkImageGridSource',
        'vtkImageMandelbrotSource',
        'vtkImageNoiseSource',
        'vtkImageSinusoidSource',
    ),
    'vtkImagingStencil': (
        'vtkImageStencil',
        'vtkPolyDataToImageStencil',
    ),
    'vtkIOInfovis': ('vtkDelimitedTextReader',),
}

# Derived mapping: class -> module
_VTK_CLASS_TO_MODULE: dict[str, str] = {
    cls: module for module, classes in _VTK_MODULES.items() for cls in classes
}


def __getattr__(name: str):
    """Lazy attribute access.

    VTK modules are only imported when first accessed.
    """
    # Handle special cases
    if name == 'vtkPythonItem':
        obj = _import_vtkPythonItem()
    elif name == 'vtkExtractCells':
        obj = _import_vtkExtractCells()
    elif name == 'vtkCellTypeUtilities':
        obj = _import_vtkCellTypeUtilities()
    else:
        # Default case: lazily import based on module mapping
        module_name = _VTK_CLASS_TO_MODULE.get(name)
        if module_name is None:
            msg = f'module {__name__!r} has no attribute {name!r}'
            raise AttributeError(msg)

        module = importlib.import_module(f'vtkmodules.{module_name}')

        try:
            obj = getattr(module, name)
        except AttributeError as e:
            msg = f"module 'vtkmodules.{module_name}' has no attribute {name!r}"
            raise AttributeError(msg) from e

    # Cache object for next access
    _THIS_MODULE.__dict__[name] = obj
    return obj


def _import_vtkPythonItem():  # noqa: N802
    try:
        from vtkmodules.vtkPythonContext2D import vtkPythonItem
    except ImportError:  # pragma: no cover
        # Suppress for ParaView shell https://github.com/pyvista/pyvista/issues/3224

        class vtkPythonItem:  # type: ignore[no-redef]  # noqa: N801
            """Empty placeholder."""

            def __init__(self) -> None:  # pragma: no cover
                """Raise version error on init."""
                from pyvista.core.errors import VTKVersionError

                msg = 'Chart backgrounds require the vtkPythonContext2D module'
                raise VTKVersionError(msg)

    return vtkPythonItem


def _import_vtkExtractCells():  # noqa: N802
    try:  # Module changed in VTK 9.3.0
        from vtkmodules.vtkFiltersCore import vtkExtractCells
    except ImportError:
        from vtkmodules.vtkFiltersExtraction import (  # type: ignore[attr-defined, no-redef]
            vtkExtractCells,
        )
    return vtkExtractCells


def _import_vtkCellTypeUtilities():  # noqa: N802
    try:  # Introduced VTK 9.6.0
        from vtkmodules.vtkCommonDataModel import vtkCellTypeUtilities
    except ImportError:
        from vtkmodules.vtkCommonDataModel import (  # type:ignore[assignment]
            vtkCellTypes as vtkCellTypeUtilities,
        )
    return vtkCellTypeUtilities
