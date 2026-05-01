"""Lazy-loaded imports from VTK.

These are the modules within VTK that must be loaded across pyvista's
core and plotting API. The modules are lazily-loaded, and are only
imported on first access. We import from ``vtkmodules`` instead of
``vtk`` to selectively import modules and not the entire library.

"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
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
        module_full_name = f'vtkmodules.{module_name}'
        error_msg = (
            f'Cannot import name {name!r} from {module_full_name!r}.\n'
            'The cause is likely attributable to VTK version or a custom VTK build.'
        )
        try:
            module = importlib.import_module(module_full_name)
        except ModuleNotFoundError as e:
            raise ImportError(error_msg) from e
        try:
            obj = getattr(module, name)
        except AttributeError as e:
            raise ImportError(error_msg) from e

    # Cache object for next access
    globals()[name] = obj
    return obj


# Specialized loading functions for irregular imports


def _import_vtkPythonItem():  # noqa: N802
    try:
        from vtkmodules.vtkPythonContext2D import vtkPythonItem  # noqa: TID251
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
        from vtkmodules.vtkFiltersCore import vtkExtractCells  # noqa: TID251
    except ImportError:
        from vtkmodules.vtkFiltersExtraction import (  # type: ignore[attr-defined, no-redef] # noqa: TID251
            vtkExtractCells,
        )
    return vtkExtractCells


def _import_vtkCellTypeUtilities():  # noqa: N802
    try:  # Introduced VTK 9.6.0
        from vtkmodules.vtkCommonDataModel import vtkCellTypeUtilities  # noqa: TID251
    except ImportError:
        from vtkmodules.vtkCommonDataModel import (  # type:ignore[assignment] # noqa: TID251
            vtkCellTypes as vtkCellTypeUtilities,
        )
    return vtkCellTypeUtilities


_SPECIAL_LOADERS: dict[str, Callable[[], type[Any]]] = {
    'vtkPythonItem': _import_vtkPythonItem,
    'vtkExtractCells': _import_vtkExtractCells,
    'vtkCellTypeUtilities': _import_vtkCellTypeUtilities,
}
