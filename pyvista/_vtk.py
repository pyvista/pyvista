"""Lazy-loaded imports from VTK.

These are the modules within VTK that must be loaded across pyvista's
core and plotting API. The modules are lazily-loaded, and are only
imported on first access. We import from ``vtkmodules`` instead of
``vtk`` to selectively import modules and not the entire library.

"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import os
import sys
from typing import TYPE_CHECKING


def _resolve_vtk_backend() -> str:
    """Return the root package PyVista resolves VTK imports against.

    Selection order:

    1. The ``PYVISTA_VTK_BACKEND`` environment variable, if set, always wins
       (e.g. ``vtkmodules`` to force stock VTK, or ``cvista`` to force the fork).
    2. Otherwise, if the community ``cvista`` fork is installed, it is auto-selected
       in preference to stock VTK. Installing ``pyvista[cvista]`` is therefore the
       only action needed to opt in -- ``cvista`` imports as its own package and
       coexists with stock ``vtk``, so this never clobbers a stock install.
    3. Otherwise fall back to stock ``vtkmodules``.

    Resolved once when this module is first imported, so ``PYVISTA_VTK_BACKEND``
    must be set before importing :mod:`pyvista`.
    """
    backend = os.environ.get('PYVISTA_VTK_BACKEND')
    if backend:
        return backend
    if importlib.util.find_spec('cvista') is not None:
        return 'cvista'
    return 'vtkmodules'


# Root package every VTK import is resolved against (``vtkmodules`` or ``cvista``).
_VTK_BACKEND = _resolve_vtk_backend()


class _VtkmodulesToCvistaFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Route ``import vtkmodules[.*]`` to ``cvista[.*]``.

    When PyVista runs on the cvista backend, third-party packages that import
    ``vtkmodules`` directly (e.g. ``pyvista-zstd``, ``trame-vtk``) should resolve
    to cvista as well. As of cvista 9.6.2.2 the fork ships VTK libraries under
    cvista-distinct SONAMEs, so cvista and stock VTK *can* coexist in one process
    without ``undefined symbol`` errors -- but they are then two separate VTK type
    systems: a ``vtkPolyData`` from stock ``vtkmodules`` is a different C++ class
    than one from ``cvista`` and cannot be handed across the boundary. This finder
    keeps a single VTK build in the process by aliasing each requested
    ``vtkmodules`` name to its ``cvista`` counterpart in :data:`sys.modules`, so
    objects created by third-party code interoperate with PyVista's.
    """

    _PREFIX = 'vtkmodules'

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,  # noqa: ARG002
        target: ModuleType | None = None,  # noqa: ARG002
    ) -> ModuleSpec | None:
        if fullname == self._PREFIX or fullname.startswith(self._PREFIX + '.'):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        module = importlib.import_module('cvista' + spec.name[len(self._PREFIX) :])
        sys.modules[spec.name] = module  # alias under the requested vtkmodules name
        return module

    def exec_module(self, module: ModuleType) -> None:
        """No-op: the target module was already executed by ``import_module``."""


if _VTK_BACKEND == 'cvista' and not any(
    isinstance(finder, _VtkmodulesToCvistaFinder) for finder in sys.meta_path
):
    sys.meta_path.insert(0, _VtkmodulesToCvistaFinder())


def _import_from(module_name: str, class_name: str) -> Any:
    """Import ``class_name`` from the backend's ``module_name``.

    Mirrors ``from {backend}.{module_name} import {class_name}`` semantics:
    a missing module or missing attribute both raise ``ImportError``.
    """
    module = importlib.import_module(f'{_VTK_BACKEND}.{module_name}')
    try:
        return getattr(module, class_name)
    except AttributeError as e:  # match `from m import c` (raises ImportError)
        raise ImportError(str(e)) from e


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    from importlib.machinery import ModuleSpec
    from types import ModuleType
    from typing import Any

    # Type checkers cannot resolve the dynamic lazy vtk imports, so we import everything
    # here as needed to ensure they can "see" the imported classes
    from vtk import *  # noqa: TID251
    from vtkmodules.numpy_interface.dataset_adapter import *  # noqa: TID251
    from vtkmodules.util.vtkAlgorithm import *  # noqa: TID251

# Canonical mapping: vtkmodule -> classes
# Modules imported for pyvista's core API
_CORE_MODULES: dict[str, tuple[str, ...]] = {
    'numpy_interface.dataset_adapter': (
        'VTKArray',
        'VTKObjectWrapper',
        'numpyTovtkDataArray',
    ),
    'numpy_interface.vtk_aos_array': ('VTKAOSArray',),
    'numpy_interface.vtk_implicit_array': ('VTKImplicitArray',),
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
        'vtkCellStatus',
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
        'vtkMergeFilter',
        'vtkOrientPolyData',
        'vtkPackLabels',
        'vtkPointDataToCellData',
        'vtkPolyDataNormals',
        'vtkQuadricDecimation',
        'vtkResampleWithDataSet',
        'vtkReverseSense',
        'vtkSmoothPolyDataFilter',
        'vtkStaticCleanUnstructuredGrid',
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
    'vtkIOExodus': ('vtkExodusIIReader',),
    'vtkIOExport': (
        'vtkGLTFExporter',
        'vtkOBJExporter',
        'vtkVRMLExporter',
    ),
    'vtkIOExportGL2PS': ('vtkGL2PSExporter',),
    'vtkIOImport': (
        'vtk3DSImporter',
        'vtkGLTFImporter',
        'vtkOBJImporter',
        'vtkVRMLImporter',
    ),
    'vtkIOInfovis': ('vtkDelimitedTextReader',),
    'vtkIOLegacy': (
        'vtkDataSetReader',
        'vtkDataSetWriter',
        'vtkDataWriter',
    ),
    'vtkIOXML': (
        'vtkXMLImageDataReader',
        'vtkXMLImageDataWriter',
        'vtkXMLPolyDataReader',
        'vtkXMLPolyDataWriter',
        'vtkXMLRectilinearGridReader',
        'vtkXMLRectilinearGridWriter',
        'vtkXMLStructuredGridReader',
        'vtkXMLStructuredGridWriter',
        'vtkXMLTableReader',
        'vtkXMLTableWriter',
        'vtkXMLUnstructuredGridReader',
        'vtkXMLUnstructuredGridWriter',
    ),
}

# Rendering modules for pyvista's plotting API
_PLOTTING_MODULES: dict[str, tuple[str, ...]] = {
    'vtkChartsCore': (
        'vtkAxis',
        'vtkChart',
        'vtkChartBox',
        'vtkChartPie',
        'vtkChartXY',
        'vtkChartXYZ',
        'vtkPlotArea',
        'vtkPlotBar',
        'vtkPlotBox',
        'vtkPlotLine',
        'vtkPlotLine3D',
        'vtkPlotPie',
        'vtkPlotPoints',
        'vtkPlotPoints3D',
        'vtkPlotStacked',
        'vtkPlotSurface',
    ),
    'vtkCommonColor': (
        'vtkColorSeries',
        'vtkNamedColors',
    ),
    'vtkInteractionStyle': (
        'vtkInteractorStyleImage',
        'vtkInteractorStyleJoystickActor',
        'vtkInteractorStyleJoystickCamera',
        'vtkInteractorStyleRubberBand2D',
        'vtkInteractorStyleRubberBandPick',
        'vtkInteractorStyleRubberBandZoom',
        'vtkInteractorStyleTerrain',
        'vtkInteractorStyleTrackballActor',
        'vtkInteractorStyleTrackballCamera',
    ),
    'vtkInteractionWidgets': (
        'vtkBoxWidget',
        'vtkButtonWidget',
        'vtkCamera3DRepresentation',
        'vtkCamera3DWidget',
        'vtkCameraOrientationWidget',
        'vtkDistanceRepresentation3D',
        'vtkDistanceWidget',
        'vtkImplicitPlaneWidget',
        'vtkLineWidget',
        'vtkLogoRepresentation',
        'vtkLogoWidget',
        'vtkOrientationMarkerWidget',
        'vtkPlaneWidget',
        'vtkPointHandleRepresentation3D',
        'vtkResliceCursorPicker',
        'vtkScalarBarWidget',
        'vtkSliderRepresentation2D',
        'vtkSliderWidget',
        'vtkSphereWidget',
        'vtkSplineWidget',
        'vtkTexturedButtonRepresentation2D',
    ),
    'vtkRenderingAnnotation': (
        'vtkAnnotatedCubeActor',
        'vtkAxesActor',
        'vtkAxisActor',
        'vtkAxisActor2D',
        'vtkCornerAnnotation',
        'vtkCubeAxesActor',
        'vtkLegendBoxActor',
        'vtkLegendScaleActor',
        'vtkScalarBarActor',
    ),
    'vtkRenderingContext2D': (
        'vtkBlockItem',
        'vtkBrush',
        'vtkContext2D',
        'vtkContextActor',
        'vtkContextScene',
        'vtkImageItem',
        'vtkPen',
    ),
    'vtkRenderingCore': (
        'VTK_RESOLVE_OFF',
        'VTK_RESOLVE_POLYGON_OFFSET',
        'VTK_RESOLVE_SHIFT_ZBUFFER',
        'vtkAbstractMapper',
        'vtkActor',
        'vtkActor2D',
        'vtkAreaPicker',
        'vtkCamera',
        'vtkCellPicker',
        'vtkColorTransferFunction',
        'vtkCompositeDataDisplayAttributes',
        'vtkCompositePolyDataMapper',
        'vtkCoordinate',
        'vtkDataSetMapper',
        'vtkFollower',
        'vtkHardwarePicker',
        'vtkImageActor',
        'vtkInteractorStyle',
        'vtkLight',
        'vtkLightActor',
        'vtkLightKit',
        'vtkMapper',
        'vtkPointGaussianMapper',
        'vtkPointPicker',
        'vtkPolyDataMapper',
        'vtkPolyDataMapper2D',
        'vtkProp',
        'vtkProp3D',
        'vtkPropAssembly',
        'vtkPropCollection',
        'vtkProperty',
        'vtkPropPicker',
        'vtkRenderedAreaPicker',
        'vtkRenderer',
        'vtkRenderWindow',
        'vtkRenderWindowInteractor',
        'vtkScenePicker',
        'vtkSelectVisiblePoints',
        'vtkSkybox',
        'vtkTextActor',
        'vtkTextProperty',
        'vtkTexture',
        'vtkViewport',
        'vtkVolume',
        'vtkVolumeProperty',
        'vtkWindowToImageFilter',
        'vtkWorldPointPicker',
    ),
    'vtkRenderingFreeType': (
        'vtkMathTextFreeTypeTextRenderer',
        'vtkVectorText',
    ),
    'vtkRenderingLabel': (
        'vtkLabelPlacementMapper',
        'vtkPointSetToLabelHierarchy',
    ),
    'vtkRenderingUI': ('vtkGenericRenderWindowInteractor',),
    'vtkRenderingVolume': (
        'vtkFixedPointVolumeRayCastMapper',
        'vtkGPUVolumeRayCastMapper',
        'vtkUnstructuredGridVolumeRayCastMapper',
        'vtkVolumeMapper',
        'vtkVolumePicker',
    ),
    'vtkViewsContext2D': ('vtkContextInteractorStyle',),
}

# GL-dependent imports from VTK.
# These are the modules within VTK requiring libGL that must be loaded
# across pyvista's plotting API. These imports have the potential to
# raise an ImportError if the user does not have libGL installed.
#
#     ImportError: libGL.so.1: cannot open shared object file: No such file or directory
_OPENGL_MODULES: dict[str, tuple[str, ...]] = {
    'vtkRenderingOpenGL2': (
        'vtkCameraPass',
        'vtkCompositePolyDataMapper2',  # optional (contextlib.suppress)
        'vtkDepthOfFieldPass',
        'vtkEDLShading',
        'vtkGaussianBlurPass',
        'vtkOpenGLFXAAPass',
        'vtkOpenGLHardwareSelector',
        'vtkOpenGLRenderer',
        'vtkOpenGLSkybox',
        'vtkOpenGLTexture',
        'vtkRenderPassCollection',
        'vtkRenderStepsPass',
        'vtkSequencePass',
        'vtkShader',
        'vtkShadowMapPass',
        'vtkSSAAPass',
        'vtkSSAOPass',
    ),
    'vtkRenderingVolumeOpenGL2': (
        'vtkOpenGLGPUVolumeRayCastMapper',
        'vtkSmartVolumeMapper',
    ),
}

# Derived mapping: class -> module
_VTK_CLASS_TO_MODULE: dict[str, str] = {
    cls: module
    for module, classes in (_CORE_MODULES | _PLOTTING_MODULES | _OPENGL_MODULES).items()
    for cls in classes
}

# Fallback modules for classes whose home module differs between VTK builds. The
# tiered ``cvista`` backend relocates the view/camera-dependent FiltersHybrid
# filters into ``vtkFiltersHybridRendering`` (so the core tier stays rendering-free);
# stock VTK keeps them in ``vtkFiltersHybrid``. ``__getattr__`` tries the primary
# module first, then these alternates, so either build resolves the class.
_VTK_CLASS_ALT_MODULES: dict[str, tuple[str, ...]] = {
    'vtkPolyDataSilhouette': ('vtkFiltersHybridRendering',),
    'vtkRenderLargeImage': ('vtkFiltersHybridRendering',),
    'vtkAdaptiveDataSetSurfaceFilter': ('vtkFiltersHybridRendering',),
}


def __getattr__(name: str):
    """Lazy attribute access.

    VTK modules are only imported when first accessed.
    """
    # Handle special cases
    if importer := _SPECIAL_LOADERS.get(name):
        obj = importer()
    else:  # Default case: lazily import based on module mapping
        module_name = _VTK_CLASS_TO_MODULE.get(name)
        if module_name is None:
            msg = (
                f"{name!r} is not defined in PyVista's vtk namespace.\n"
                f'Developers should add a new `module:{name}` mapping to the `_vtk` module.'
            )
            raise AttributeError(msg)

        # Attempt to import the vtkmodule and the desired attribute
        # Convert module or attribute errors into a similar message that would otherwise be
        # seen when doing `from vtkmodules.vtkModule import vtkClass`
        module_full_name = f'{_VTK_BACKEND}.{module_name}'
        error_msg = (
            f'Cannot import name {name!r} from {module_full_name!r}.\n'
            'The cause is likely attributable to VTK version or a custom VTK build.'
        )
        candidate_modules = (module_name, *_VTK_CLASS_ALT_MODULES.get(name, ()))
        obj = _missing = object()
        last_exc: Exception | None = None
        for candidate in candidate_modules:
            try:
                module = importlib.import_module(f'{_VTK_BACKEND}.{candidate}')
                obj = getattr(module, name)
                break
            except (ModuleNotFoundError, AttributeError) as e:
                last_exc = e
                continue
        if obj is _missing:
            raise ImportError(error_msg) from last_exc

    # Cache object for next access
    globals()[name] = obj
    return obj


def has_attr(name: str) -> bool:
    """Return ``True`` if *name* resolves to a VTK class on this build.

    ``hasattr(_vtk, 'X')`` does not work as expected because the lazy
    ``__getattr__`` raises ``ImportError`` (not ``AttributeError``) when a
    class is mapped but missing from the underlying ``vtkmodules`` module on
    the installed VTK build. ``hasattr`` only catches ``AttributeError``, so
    the raised ``ImportError`` propagates and breaks version-probe call sites.
    Use this helper instead.

    Parameters
    ----------
    name : str
        Attribute name to probe on the ``_vtk`` namespace.

    Returns
    -------
    bool
        ``True`` if the attribute resolves on this VTK build, ``False`` if it
        is missing or its underlying ``vtkmodules`` module is missing.

    """
    if name in globals():
        return True
    try:
        __getattr__(name)
    except (AttributeError, ImportError):
        return False
    return True


# Specialized loading functions for irregular imports


def _import_vtkPythonItem():  # noqa: N802
    try:
        return _import_from('vtkPythonContext2D', 'vtkPythonItem')
    except ImportError:  # pragma: no cover
        # Suppress for ParaView shell https://github.com/pyvista/pyvista/issues/3224

        class vtkPythonItem:  # noqa: N801
            """Empty placeholder."""

            def __init__(self) -> None:  # pragma: no cover
                """Raise version error on init."""
                from pyvista.core.errors import VTKVersionError

                msg = 'Chart backgrounds require the vtkPythonContext2D module'
                raise VTKVersionError(msg)

    return vtkPythonItem


def _import_vtkExtractCells():  # noqa: N802
    try:  # Module changed in VTK 9.3.0
        return _import_from('vtkFiltersCore', 'vtkExtractCells')
    except ImportError:
        return _import_from('vtkFiltersExtraction', 'vtkExtractCells')


def _import_vtkCellTypeUtilities():  # noqa: N802
    try:  # Introduced VTK 9.6.0
        return _import_from('vtkCommonDataModel', 'vtkCellTypeUtilities')
    except ImportError:
        return _import_from('vtkCommonDataModel', 'vtkCellTypes')


_SPECIAL_LOADERS: dict[str, Callable[[], type[Any]]] = {
    'vtkPythonItem': _import_vtkPythonItem,
    'vtkExtractCells': _import_vtkExtractCells,
    'vtkCellTypeUtilities': _import_vtkCellTypeUtilities,
}
