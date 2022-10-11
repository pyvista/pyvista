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

# for charts
_has_vtkRenderingContextOpenGL2 = False

if VTK9:
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
        reference,
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
        VTK_BIQUADRATIC_QUAD,
        VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON,
        VTK_BIQUADRATIC_QUADRATIC_WEDGE,
        VTK_BIQUADRATIC_TRIANGLE,
        VTK_CONVEX_POINT_SET,
        VTK_CUBIC_LINE,
        VTK_EMPTY_CELL,
        VTK_HEXAGONAL_PRISM,
        VTK_HEXAHEDRON,
        VTK_HIGHER_ORDER_EDGE,
        VTK_HIGHER_ORDER_HEXAHEDRON,
        VTK_HIGHER_ORDER_POLYGON,
        VTK_HIGHER_ORDER_PYRAMID,
        VTK_HIGHER_ORDER_QUAD,
        VTK_HIGHER_ORDER_TETRAHEDRON,
        VTK_HIGHER_ORDER_TRIANGLE,
        VTK_HIGHER_ORDER_WEDGE,
        VTK_LAGRANGE_CURVE,
        VTK_LAGRANGE_HEXAHEDRON,
        VTK_LAGRANGE_PYRAMID,
        VTK_LAGRANGE_QUADRILATERAL,
        VTK_LAGRANGE_TETRAHEDRON,
        VTK_LAGRANGE_TRIANGLE,
        VTK_LAGRANGE_WEDGE,
        VTK_LINE,
        VTK_PARAMETRIC_CURVE,
        VTK_PARAMETRIC_HEX_REGION,
        VTK_PARAMETRIC_QUAD_SURFACE,
        VTK_PARAMETRIC_SURFACE,
        VTK_PARAMETRIC_TETRA_REGION,
        VTK_PARAMETRIC_TRI_SURFACE,
        VTK_PENTAGONAL_PRISM,
        VTK_PIXEL,
        VTK_POLY_LINE,
        VTK_POLY_VERTEX,
        VTK_POLYGON,
        VTK_POLYHEDRON,
        VTK_PYRAMID,
        VTK_QUAD,
        VTK_QUADRATIC_EDGE,
        VTK_QUADRATIC_HEXAHEDRON,
        VTK_QUADRATIC_LINEAR_QUAD,
        VTK_QUADRATIC_LINEAR_WEDGE,
        VTK_QUADRATIC_POLYGON,
        VTK_QUADRATIC_PYRAMID,
        VTK_QUADRATIC_QUAD,
        VTK_QUADRATIC_TETRA,
        VTK_QUADRATIC_TRIANGLE,
        VTK_QUADRATIC_WEDGE,
        VTK_TETRA,
        VTK_TRIANGLE,
        VTK_TRIANGLE_STRIP,
        VTK_TRIQUADRATIC_HEXAHEDRON,
        VTK_VERTEX,
        VTK_VOXEL,
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
        vtkNonMergingPointLocator,
        vtkPerlinNoise,
        vtkPiecewiseFunction,
        vtkPlane,
        vtkPlaneCollection,
        vtkPlanes,
        vtkPointLocator,
        vtkPointSet,
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

    try:
        from vtkmodules.vtkCommonDataModel import (
            VTK_BEZIER_CURVE,
            VTK_BEZIER_HEXAHEDRON,
            VTK_BEZIER_PYRAMID,
            VTK_BEZIER_QUADRILATERAL,
            VTK_BEZIER_TETRAHEDRON,
            VTK_BEZIER_TRIANGLE,
            VTK_BEZIER_WEDGE,
            VTK_TRIQUADRATIC_PYRAMID,
        )
    except ImportError:  # pragma: no cover
        pass

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
        vtkWindowedSincPolyDataFilter,
    )
    from vtkmodules.vtkFiltersExtraction import (
        vtkExtractGeometry,
        vtkExtractGrid,
        vtkExtractSelection,
    )
    from vtkmodules.vtkFiltersFlowPaths import vtkEvenlySpacedStreamlines2D, vtkStreamTracer
    from vtkmodules.vtkFiltersGeneral import (
        vtkAxes,
        vtkBooleanOperationPolyDataFilter,
        vtkBoxClipDataSet,
        vtkClipClosedSurface,
        vtkCursor3D,
        vtkCurvatures,
        vtkDataSetTriangleFilter,
        vtkGradientFilter,
        vtkIntersectionPolyDataFilter,
        vtkOBBTree,
        vtkRectilinearGridToPointSet,
        vtkRectilinearGridToTetrahedra,
        vtkShrinkFilter,
        vtkTableBasedClipDataSet,
        vtkTableToPolyData,
        vtkTessellatorFilter,
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
        vtkSubdivideTetra,
        vtkTrimmedExtrusionFilter,
    )
    from vtkmodules.vtkFiltersParallel import vtkIntegrateAttributes

    try:
        from vtkmodules.vtkFiltersParallelDIY2 import vtkRedistributeDataSetFilter
    except ModuleNotFoundError:  # pragma: no cover
        # `vtkmodules.vtkFiltersParallelDIY2` is unavailable in some versions of `vtk` from conda-forge
        pass
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
        vtkTecplotReader,
    )
    from vtkmodules.vtkIOImage import (
        vtkBMPReader,
        vtkDEMReader,
        vtkDICOMImageReader,
        vtkHDRReader,
        vtkJPEGReader,
        vtkMetaImageReader,
        vtkNIFTIImageReader,
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
        vtkSimplePointsWriter,
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
        vtkXMLTableReader,
        vtkXMLTableWriter,
        vtkXMLUnstructuredGridReader,
        vtkXMLUnstructuredGridWriter,
        vtkXMLWriter,
    )
    from vtkmodules.vtkImagingCore import (
        vtkExtractVOI,
        vtkImageDifference,
        vtkImageExtractComponents,
        vtkImageFlip,
        vtkImageThreshold,
        vtkRTAnalyticSource,
    )
    from vtkmodules.vtkImagingGeneral import vtkImageGaussianSmooth, vtkImageMedian3D
    from vtkmodules.vtkImagingHybrid import vtkSampleFunction, vtkSurfaceReconstructionFilter
    from vtkmodules.vtkImagingMorphological import vtkImageDilateErode3D
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
        vtkAxisActor2D,
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

    try:
        # Necessary for displaying charts, otherwise crashes on rendering
        import vtkmodules.vtkRenderingContextOpenGL2

        _has_vtkRenderingContextOpenGL2 = True
    except ImportError:  # pragma: no cover
        pass

    from vtkmodules.vtkImagingFourier import (
        vtkImageButterworthHighPass,
        vtkImageButterworthLowPass,
        vtkImageFFT,
        vtkImageRFFT,
    )
    from vtkmodules.vtkRenderingCore import (
        vtkAbstractMapper,
        vtkActor,
        vtkActor2D,
        vtkCamera,
        vtkCellPicker,
        vtkColorTransferFunction,
        vtkCompositeDataDisplayAttributes,
        vtkCoordinate,
        vtkDataSetMapper,
        vtkImageActor,
        vtkLight,
        vtkLightActor,
        vtkLightKit,
        vtkMapper,
        vtkPointGaussianMapper,
        vtkPointPicker,
        vtkPolyDataMapper,
        vtkPolyDataMapper2D,
        vtkPropAssembly,
        vtkProperty,
        vtkPropPicker,
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
        vtkCompositePolyDataMapper2,
        vtkDepthOfFieldPass,
        vtkEDLShading,
        vtkGaussianBlurPass,
        vtkOpenGLFXAAPass,
        vtkOpenGLHardwareSelector,
        vtkOpenGLRenderer,
        vtkOpenGLTexture,
        vtkRenderPassCollection,
        vtkRenderStepsPass,
        vtkSequencePass,
        vtkShadowMapPass,
        vtkSSAAPass,
        vtkSSAOPass,
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
    def lazy_vtkOBJExporter():
        """Lazy import of the vtkOBJExporter."""
        from vtkmodules.vtkIOExport import vtkOBJExporter

        return vtkOBJExporter()

    def lazy_vtkVRMLImporter():
        """Lazy import of the vtkVRMLImporter."""
        from vtkmodules.vtkIOImport import vtkVRMLImporter

        return vtkVRMLImporter()

    def lazy_vtkVRMLExporter():
        """Lazy import of the vtkVRMLExporter."""
        from vtkmodules.vtkIOExport import vtkVRMLExporter

        return vtkVRMLExporter()

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

    def lazy_vtkHDFReader():
        """Lazy import of the vtkHDFReader."""
        from vtkmodules.vtkIOHDF import vtkHDFReader

        return vtkHDFReader()

    def lazy_vtkCGNSReader():
        """Lazy import of the vtkCGNSReader."""
        from vtkmodules.vtkIOCGNSReader import vtkCGNSReader

        return vtkCGNSReader()

    def lazy_vtkPOpenFOAMReader():
        """Lazy import of the vtkPOpenFOAMReader."""
        from vtkmodules.vtkIOParallel import vtkPOpenFOAMReader
        from vtkmodules.vtkParallelCore import vtkDummyController

        # Workaround waiting for the fix to be upstream (MR 9195 gitlab.kitware.com/vtk/vtk)
        reader = vtkPOpenFOAMReader()
        reader.SetController(vtkDummyController())
        return reader

else:  # pragma: no cover

    # maintain VTK 8.2 compatibility
    try:
        from vtk.vtkCommonKitPython import buffer_shared, vtkAbstractArray, vtkWeakReference
    except ImportError:
        from vtk.vtkCommonCore import buffer_shared, vtkAbstractArray, vtkWeakReference

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
    def lazy_vtkOBJExporter():
        """Lazy import of the vtkOBJExporter."""
        return vtk.vtkOBJExporter()

    def lazy_vtkVRMLImporter():
        """Lazy import of the vtkVRMLImporter."""
        return vtk.vtkVRMLImporter()

    def lazy_vtkVRMLExporter():
        """Lazy import of the vtkVRMLExporter."""
        return vtk.vtkVRMLExporter()

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

    def lazy_vtkCGNSReader():
        """Lazy import of the vtkCGNSReader."""
        raise VTKVersionError('vtk.CGNSReader requires VTK v9.1.0 or newer')

    def lazy_vtkPOpenFOAMReader():
        """Lazy import of the vtkPOpenFOAMReader."""
        # Workaround to fix the following issue: https://gitlab.kitware.com/vtk/vtk/-/issues/18143
        # Fixed in vtk > 9.1.0
        reader = vtk.vtkPOpenFOAMReader()
        reader.SetController(vtk.vtkDummyController())
        return reader

    def lazy_vtkHDFReader():
        """Lazy import of the vtkHDFReader."""
        raise VTKVersionError('vtk.HDFReader requires VTK v9.1.0 or newer')

    def lazy_vtkSegYReader():
        """Lazy import of the vtkSegYReader."""
        return vtk.vtkSegYReader()

    class vtkExplicitStructuredGrid:  # type: ignore
        """Empty placeholder for VTK9 compatibility."""

        def __init__(self):  # pragma: no cover
            """Raise version error on init."""
            from pyvista.core.errors import VTKVersionError

            raise VTKVersionError('vtkHDRReader requires VTK v9 or newer')

    class vtkHDRReader:  # type: ignore
        """Empty placeholder for VTK9 compatibility."""

        def __init__(self):  # pragma: no cover
            """Raise version error on init."""
            from pyvista.core.errors import VTKVersionError

            raise VTKVersionError('vtkHDRReader requires VTK v9 or newer')

    class vtkGLTFReader:  # type: ignore
        """Empty placeholder for VTK9 compatibility."""

        def __init__(self):  # pragma: no cover
            """Raise version error on init."""
            from pyvista.core.errors import VTKVersionError

            raise VTKVersionError('vtkGLTFReader requires VTK v9 or newer')

    class vtkPythonItem:  # type: ignore
        """Empty placeholder for VTK9 compatibility."""

        def __init__(self):  # pragma: no cover
            """Raise version error on init."""
            from pyvista.core.errors import VTKVersionError

            raise VTKVersionError('Charts requires VTK v9 or newer')

    class vtkHDFReader:  # type: ignore
        """Empty placeholder for VTK9 compatibility."""

        def __init__(self):  # pragma: no cover
            """Raise version error on init."""
            from pyvista.core.errors import VTKVersionError

            raise VTKVersionError('vtkHDFReader requires VTK v9 or newer')


# lazy import as this was added in 9.1.0
def lazy_vtkCameraOrientationWidget():
    """Lazy import of the vtkCameraOrientationWidget."""
    try:
        from vtkmodules.vtkInteractionWidgets import vtkCameraOrientationWidget
    except ImportError:  # pragma: no cover
        from pyvista.core.errors import VTKVersionError

        raise VTKVersionError('vtkCameraOrientationWidget requires vtk>=9.1.0')
    return vtkCameraOrientationWidget()
