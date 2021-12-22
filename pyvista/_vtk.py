"""
Import from vtk.

These are the modules within vtk that must be loaded across pyvista.
Here, we attempt to import modules using the VTK9 ``vtkmodules``
package, which lets us only have to import from select modules and not
the entire library.

"""
# flake8: noqa: F401

# Checking for VTK9 here even though 8.2 contains vtkmodules.  There
# are enough idiosyncrasies to VTK 8.2, and supporting it would lead
# to obscure code.
try:
    from vtkmodules.vtkCommonCore import vtkVersion
    VTK9 = vtkVersion().GetVTKMajorVersion() >= 9
except ImportError:  # pragma: no cover
    VTK9 = False


if VTK9:

    from vtkmodules.numpy_interface.dataset_adapter import (
        VTKArray,
        VTKObjectWrapper,
        numpyTovtkDataArray,
    )
    from vtkmodules.util.numpy_support import (
        get_vtk_array_type,
        numpy_to_vtk,
        numpy_to_vtkIdTypeArray,
        vtk_to_numpy,
    )
    from vtkmodules.vtkChartsCore import (
        vtkAxis,
        vtkChart,
        vtkChartBox,
        vtkChartPie,
        vtkChartXY,
        vtkChartXYZ,
        vtkPlotArea,
        vtkPlotBar,
        vtkPlotBox,
        vtkPlotLine,
        vtkPlotLine3D,
        vtkPlotPie,
        vtkPlotPoints,
        vtkPlotPoints3D,
        vtkPlotStacked,
        vtkPlotSurface,
    )
    from vtkmodules.vtkCommonColor import vtkColorSeries
    from vtkmodules.vtkCommonComputationalGeometry import (
        vtkKochanekSpline,
        vtkParametricBohemianDome,
        vtkParametricBour,
        vtkParametricBoy,
        vtkParametricCatalanMinimal,
        vtkParametricConicSpiral,
        vtkParametricCrossCap,
        vtkParametricDini,
        vtkParametricEllipsoid,
        vtkParametricEnneper,
        vtkParametricFigure8Klein,
        vtkParametricFunction,
        vtkParametricHenneberg,
        vtkParametricKlein,
        vtkParametricKuen,
        vtkParametricMobius,
        vtkParametricPluckerConoid,
        vtkParametricPseudosphere,
        vtkParametricRandomHills,
        vtkParametricRoman,
        vtkParametricSpline,
        vtkParametricSuperEllipsoid,
        vtkParametricSuperToroid,
        vtkParametricTorus,
    )
    from vtkmodules.vtkCommonCore import (
        VTK_ARIAL,
        VTK_COURIER,
        VTK_TIMES,
        VTK_UNSIGNED_CHAR,
        buffer_shared,
        mutable,
        vtkAbstractArray,
        vtkBitArray,
        vtkCharArray,
        vtkCommand,
        vtkDataArray,
        vtkDoubleArray,
        vtkFileOutputWindow,
        vtkFloatArray,
        vtkIdList,
        vtkIdTypeArray,
        vtkLookupTable,
        vtkOutputWindow,
        vtkPoints,
        vtkSignedCharArray,
        vtkStringArray,
        vtkStringOutputWindow,
        vtkTypeInt32Array,
        vtkTypeInt64Array,
        vtkTypeUInt32Array,
        vtkUnsignedCharArray,
        vtkWeakReference,
    )
    from vtkmodules.vtkCommonDataModel import (
        VTK_HEXAHEDRON,
        VTK_PYRAMID,
        VTK_QUAD,
        VTK_QUADRATIC_HEXAHEDRON,
        VTK_QUADRATIC_PYRAMID,
        VTK_QUADRATIC_QUAD,
        VTK_QUADRATIC_TETRA,
        VTK_QUADRATIC_TRIANGLE,
        VTK_QUADRATIC_WEDGE,
        VTK_TETRA,
        VTK_TRIANGLE,
        VTK_WEDGE,
        vtkCellArray,
        vtkCellLocator,
        vtkColor3ub,
        vtkCompositeDataSet,
        vtkDataObject,
        vtkDataSet,
        vtkDataSetAttributes,
        vtkExplicitStructuredGrid,
        vtkFieldData,
        vtkGenericCell,
        vtkImageData,
        vtkImplicitFunction,
        vtkMultiBlockDataSet,
        vtkPerlinNoise,
        vtkPiecewiseFunction,
        vtkPlane,
        vtkPlaneCollection,
        vtkPlanes,
        vtkPointLocator,
        vtkPolyData,
        vtkPolyLine,
        vtkPolyPlane,
        vtkPyramid,
        vtkRectf,
        vtkRectilinearGrid,
        vtkSelection,
        vtkSelectionNode,
        vtkStaticCellLocator,
        vtkStaticPointLocator,
        vtkStructuredGrid,
        vtkTable,
        vtkUnstructuredGrid,
    )
    from vtkmodules.vtkCommonExecutionModel import vtkImageToStructuredGrid
    from vtkmodules.vtkCommonMath import vtkMatrix3x3, vtkMatrix4x4
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersCore import (
        VTK_BEST_FITTING_PLANE,
        vtkAppendArcLength,
        vtkAppendFilter,
        vtkAppendPolyData,
        vtkCellCenters,
        vtkCellDataToPointData,
        vtkCenterOfMass,
        vtkCleanPolyData,
        vtkClipPolyData,
        vtkConnectivityFilter,
        vtkContourFilter,
        vtkCutter,
        vtkDecimatePro,
        vtkDelaunay2D,
        vtkDelaunay3D,
        vtkElevationFilter,
        vtkExplicitStructuredGridToUnstructuredGrid,
        vtkFeatureEdges,
        vtkFlyingEdges3D,
        vtkGlyph3D,
        vtkImplicitPolyDataDistance,
        vtkMarchingCubes,
        vtkMassProperties,
        vtkPointDataToCellData,
        vtkPolyDataNormals,
        vtkProbeFilter,
        vtkQuadricDecimation,
        vtkResampleWithDataSet,
        vtkSmoothPolyDataFilter,
        vtkStripper,
        vtkThreshold,
        vtkTriangleFilter,
        vtkTubeFilter,
        vtkUnstructuredGridToExplicitStructuredGrid,
    )
    from vtkmodules.vtkFiltersExtraction import (
        vtkExtractEdges,
        vtkExtractGeometry,
        vtkExtractGrid,
        vtkExtractSelection,
    )
    from vtkmodules.vtkFiltersFlowPaths import vtkEvenlySpacedStreamlines2D, vtkStreamTracer
    from vtkmodules.vtkFiltersGeneral import (
        vtkAxes,
        vtkBooleanOperationPolyDataFilter,
        vtkBoxClipDataSet,
        vtkCellTreeLocator,
        vtkClipClosedSurface,
        vtkCursor3D,
        vtkCurvatures,
        vtkDataSetTriangleFilter,
        vtkGradientFilter,
        vtkIntersectionPolyDataFilter,
        vtkOBBTree,
        vtkRectilinearGridToPointSet,
        vtkShrinkFilter,
        vtkTableBasedClipDataSet,
        vtkTableToPolyData,
        vtkTransformFilter,
        vtkWarpScalar,
        vtkWarpVector,
    )
    from vtkmodules.vtkFiltersGeometry import (
        vtkCompositeDataGeometryFilter,
        vtkDataSetSurfaceFilter,
        vtkGeometryFilter,
        vtkStructuredGridGeometryFilter,
    )
    from vtkmodules.vtkFiltersHybrid import vtkPolyDataSilhouette
    from vtkmodules.vtkFiltersModeling import (
        vtkAdaptiveSubdivisionFilter,
        vtkButterflySubdivisionFilter,
        vtkCollisionDetectionFilter,
        vtkDijkstraGraphGeodesicPath,
        vtkFillHolesFilter,
        vtkLinearExtrusionFilter,
        vtkLinearSubdivisionFilter,
        vtkLoopSubdivisionFilter,
        vtkOutlineFilter,
        vtkRibbonFilter,
        vtkRotationalExtrusionFilter,
        vtkSelectEnclosedPoints,
    )
    from vtkmodules.vtkFiltersPoints import vtkGaussianKernel, vtkPointInterpolator
    from vtkmodules.vtkFiltersSources import (
        vtkArcSource,
        vtkArrowSource,
        vtkConeSource,
        vtkCubeSource,
        vtkCylinderSource,
        vtkDiskSource,
        vtkFrustumSource,
        vtkLineSource,
        vtkOutlineCornerFilter,
        vtkOutlineCornerSource,
        vtkParametricFunctionSource,
        vtkPlaneSource,
        vtkPlatonicSolidSource,
        vtkPointSource,
        vtkRegularPolygonSource,
        vtkSphereSource,
        vtkSuperquadricSource,
        vtkTessellatedBoxSource,
    )
    from vtkmodules.vtkFiltersStatistics import vtkComputeQuartiles
    from vtkmodules.vtkFiltersTexture import vtkTextureMapToPlane, vtkTextureMapToSphere
    from vtkmodules.vtkFiltersVerdict import vtkCellQuality, vtkCellSizeFilter
    from vtkmodules.vtkIOEnSight import vtkGenericEnSightReader
    from vtkmodules.vtkIOGeometry import (
        vtkAVSucdReader,
        vtkBYUReader,
        vtkFLUENTReader,
        vtkGLTFReader,
        vtkMCubesReader,
        vtkMFIXReader,
        vtkOBJReader,
        vtkOpenFOAMReader,
        vtkPTSReader,
        vtkSTLReader,
        vtkSTLWriter,
    )
    from vtkmodules.vtkIOImage import (
        vtkBMPReader,
        vtkDEMReader,
        vtkDICOMImageReader,
        vtkHDRReader,
        vtkJPEGReader,
        vtkMetaImageReader,
        vtkNrrdReader,
        vtkPNGReader,
        vtkPNMReader,
        vtkSLCReader,
        vtkTIFFReader,
    )
    from vtkmodules.vtkIOInfovis import vtkDelimitedTextReader
    from vtkmodules.vtkIOLegacy import (
        vtkDataReader,
        vtkDataSetReader,
        vtkDataSetWriter,
        vtkDataWriter,
        vtkPolyDataReader,
        vtkPolyDataWriter,
        vtkRectilinearGridReader,
        vtkRectilinearGridWriter,
        vtkStructuredGridReader,
        vtkStructuredGridWriter,
        vtkUnstructuredGridReader,
        vtkUnstructuredGridWriter,
    )
    from vtkmodules.vtkIOPLY import vtkPLYReader, vtkPLYWriter
    from vtkmodules.vtkIOXML import (
        vtkXMLImageDataReader,
        vtkXMLImageDataWriter,
        vtkXMLMultiBlockDataReader,
        vtkXMLMultiBlockDataWriter,
        vtkXMLPImageDataReader,
        vtkXMLPolyDataReader,
        vtkXMLPolyDataWriter,
        vtkXMLPRectilinearGridReader,
        vtkXMLPUnstructuredGridReader,
        vtkXMLReader,
        vtkXMLRectilinearGridReader,
        vtkXMLRectilinearGridWriter,
        vtkXMLStructuredGridReader,
        vtkXMLStructuredGridWriter,
        vtkXMLUnstructuredGridReader,
        vtkXMLUnstructuredGridWriter,
        vtkXMLWriter,
    )
    from vtkmodules.vtkImagingCore import (
        vtkExtractVOI,
        vtkImageDifference,
        vtkImageExtractComponents,
        vtkImageFlip,
        vtkRTAnalyticSource,
    )
    from vtkmodules.vtkImagingGeneral import vtkImageGaussianSmooth
    from vtkmodules.vtkImagingHybrid import vtkSampleFunction, vtkSurfaceReconstructionFilter
    from vtkmodules.vtkInteractionWidgets import (
        vtkBoxWidget,
        vtkButtonWidget,
        vtkImplicitPlaneWidget,
        vtkLineWidget,
        vtkOrientationMarkerWidget,
        vtkPlaneWidget,
        vtkScalarBarWidget,
        vtkSliderRepresentation2D,
        vtkSliderWidget,
        vtkSphereWidget,
        vtkSplineWidget,
        vtkTexturedButtonRepresentation2D,
    )
    from vtkmodules.vtkPythonContext2D import vtkPythonItem
    from vtkmodules.vtkRenderingAnnotation import (
        vtkAnnotatedCubeActor,
        vtkAxesActor,
        vtkCornerAnnotation,
        vtkCubeAxesActor,
        vtkLegendBoxActor,
        vtkScalarBarActor,
    )
    from vtkmodules.vtkRenderingContext2D import (
        vtkBlockItem,
        vtkBrush,
        vtkContext2D,
        vtkContextActor,
        vtkContextScene,
        vtkImageItem,
        vtkPen,
    )
    import vtkmodules.vtkRenderingContextOpenGL2  # Necessary for displaying charts, otherwise crashes on rendering
    from vtkmodules.vtkRenderingCore import (
        vtkActor,
        vtkActor2D,
        vtkCamera,
        vtkColorTransferFunction,
        vtkCoordinate,
        vtkDataSetMapper,
        vtkImageActor,
        vtkLight,
        vtkLightActor,
        vtkLightKit,
        vtkMapper,
        vtkPointPicker,
        vtkPolyDataMapper,
        vtkPolyDataMapper2D,
        vtkPropAssembly,
        vtkProperty,
        vtkRenderedAreaPicker,
        vtkRenderer,
        vtkRenderWindow,
        vtkRenderWindowInteractor,
        vtkSelectVisiblePoints,
        vtkSkybox,
        vtkTextActor,
        vtkTexture,
        vtkVolume,
        vtkVolumeProperty,
        vtkWindowToImageFilter,
        vtkWorldPointPicker,
    )
    from vtkmodules.vtkRenderingFreeType import vtkVectorText
    from vtkmodules.vtkRenderingLabel import vtkLabelPlacementMapper, vtkPointSetToLabelHierarchy
    from vtkmodules.vtkRenderingOpenGL2 import (
        vtkCameraPass,
        vtkEDLShading,
        vtkOpenGLHardwareSelector,
        vtkOpenGLRenderer,
        vtkOpenGLTexture,
        vtkRenderPassCollection,
        vtkRenderStepsPass,
        vtkSequencePass,
        vtkShadowMapPass,
    )
    from vtkmodules.vtkRenderingUI import vtkGenericRenderWindowInteractor
    from vtkmodules.vtkRenderingVolume import (
        vtkFixedPointVolumeRayCastMapper,
        vtkGPUVolumeRayCastMapper,
    )
    from vtkmodules.vtkRenderingVolumeOpenGL2 import (
        vtkOpenGLGPUVolumeRayCastMapper,
        vtkSmartVolumeMapper,
    )
    from vtkmodules.vtkViewsContext2D import vtkContextInteractorStyle

    # lazy import for some of the less used readers
    def lazy_vtkGL2PSExporter():
        """Lazy import of the vtkGL2PSExporter."""
        from vtkmodules.vtkIOExportGL2PS import vtkGL2PSExporter
        return vtkGL2PSExporter()

    def lazy_vtkFacetReader():
        """Lazy import of the vtkFacetReader."""
        from vtkmodules.vtkFiltersHybrid import vtkFacetReader
        return vtkFacetReader()

    def lazy_vtkPDataSetReader():
        """Lazy import of the vtkPDataSetReader."""
        from vtkmodules.vtkIOParallel import vtkPDataSetReader
        return vtkPDataSetReader()

    def lazy_vtkMultiBlockPLOT3DReader():
        """Lazy import of the vtkMultiBlockPLOT3DReader."""
        from vtkmodules.vtkIOParallel import vtkMultiBlockPLOT3DReader
        return vtkMultiBlockPLOT3DReader()

    def lazy_vtkPlot3DMetaReader():
        """Lazy import of the vtkPlot3DMetaReader."""
        from vtkmodules.vtkIOParallel import vtkPlot3DMetaReader
        return vtkPlot3DMetaReader()

    def lazy_vtkSegYReader():
        """Lazy import of the vtkSegYReader."""
        from vtkmodules.vtkIOSegY import vtkSegYReader
        return vtkSegYReader()

else:  # pragma: no cover

    # maintain VTK 8.2 compatibility
    try:
        from vtk.vtkCommonKitPython import buffer_shared, vtkAbstractArray, vtkWeakReference
    except ImportError:
        from vtk.vtkCommonCore import (buffer_shared,
                                       vtkAbstractArray,
                                       vtkWeakReference)

    import vtk

    # vtk8 already has an import all, so we can just mirror it here at
    # no cost
    from vtk import *
    from vtk.numpy_interface.dataset_adapter import VTKArray, VTKObjectWrapper, numpyTovtkDataArray
    from vtk.util.numpy_support import (
        get_vtk_array_type,
        numpy_to_vtk,
        numpy_to_vtkIdTypeArray,
        vtk_to_numpy,
    )

    # match the imports for VTK9
    def lazy_vtkGL2PSExporter():
        """Lazy import of the vtkGL2PSExporter."""
        return vtk.vtkGL2PSExporter()

    def lazy_vtkFacetReader():
        """Lazy import of the vtkFacetReader."""
        return vtk.vtkFacetReader()

    def lazy_vtkPDataSetReader():
        """Lazy import of the vtkPDataSetReader."""
        return vtk.vtkPDataSetReader()

    def lazy_vtkMultiBlockPLOT3DReader():
        """Lazy import of the vtkMultiBlockPLOT3DReader."""
        return vtk.vtkMultiBlockPLOT3DReader()

    def lazy_vtkPlot3DMetaReader():
        """Lazy import of the vtkPlot3DMetaReader."""
        return vtk.vtkPlot3DMetaReader()

    class vtkExplicitStructuredGrid():  # type: ignore
        """Empty placeholder for VTK9 compatibility."""

        def __init__(self):  # pragma: no cover
            """Raise version error on init."""
            from pyvista.core.errors import VTKVersionError
            raise VTKVersionError('vtkHDRReader requires VTK v9 or newer')

    class vtkHDRReader():   # type: ignore
        """Empty placeholder for VTK9 compatibility."""

        def __init__(self):  # pragma: no cover
            """Raise version error on init."""
            from pyvista.core.errors import VTKVersionError
            raise VTKVersionError('vtkHDRReader requires VTK v9 or newer')

    class vtkGLTFReader():   # type: ignore
        """Empty placeholder for VTK9 compatibility."""

        def __init__(self):  # pragma: no cover
            """Raise version error on init."""
            from pyvista.core.errors import VTKVersionError
            raise VTKVersionError('vtkGLTFReader requires VTK v9 or newer')


# lazy import as this was added in 9.1.0
def lazy_vtkCameraOrientationWidget():
    """Lazy import of the vtkCameraOrientationWidget."""
    try:
        from vtkmodules.vtkInteractionWidgets import vtkCameraOrientationWidget
    except ImportError:  # pragma: no cover
        raise ImportError('vtkCameraOrientationWidget requires vtk>=9.1.0')
    return vtkCameraOrientationWidget()
