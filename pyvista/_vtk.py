"""
Import from vtk.

These are the modules within vtk that must be loaded across pyvista.
Here, we attempt to import modules using the VTK9 ``vtkmodules``
package, which lets us only have to import from select modules and not
the entire library.

"""

# Checking for VTK9 here even though 8.2 contains vtkmodules.  There
# are enough idiosyncrasies to VTK 8.2, and supporting it would lead
# to obscure code.
try:
    from vtkmodules.vtkCommonCore import vtkVersion
    VTK9 = vtkVersion().GetVTKMajorVersion() >= 9
except ImportError:  # pragma: no cover
    VTK9 = False


if VTK9:

    from vtkmodules.vtkImagingHybrid import vtkSampleFunction
    from vtkmodules.vtkInteractionWidgets import (vtkScalarBarWidget,
                                                  vtkSplineWidget,
                                                  vtkSphereWidget,
                                                  vtkTexturedButtonRepresentation2D,
                                                  vtkButtonWidget,
                                                  vtkLineWidget,
                                                  vtkSliderRepresentation2D,
                                                  vtkSliderWidget,
                                                  vtkImplicitPlaneWidget,
                                                  vtkPlaneWidget,
                                                  vtkBoxWidget,
                                                  vtkOrientationMarkerWidget)
    from vtkmodules.vtkRenderingFreeType import vtkVectorText
    from vtkmodules.vtkRenderingLabel import (vtkPointSetToLabelHierarchy,
                                              vtkLabelPlacementMapper)
    from vtkmodules.vtkRenderingVolume import (vtkFixedPointVolumeRayCastMapper,
                                               vtkGPUVolumeRayCastMapper)
    from vtkmodules.vtkRenderingVolumeOpenGL2 import (vtkOpenGLGPUVolumeRayCastMapper,
                                                      vtkSmartVolumeMapper)
    from vtkmodules.vtkRenderingOpenGL2 import (vtkOpenGLHardwareSelector,
                                                vtkRenderStepsPass,
                                                vtkEDLShading,
                                                vtkOpenGLRenderer,
                                                vtkShadowMapPass,
                                                vtkSequencePass,
                                                vtkCameraPass,
                                                vtkRenderPassCollection,
                                                vtkOpenGLTexture)
    from vtkmodules.vtkIOInfovis import vtkDelimitedTextReader
    from vtkmodules.vtkIOPLY import (vtkPLYReader,
                                     vtkPLYWriter)
    from vtkmodules.vtkIOGeometry import (vtkSTLReader,
                                          vtkFLUENTReader,
                                          vtkPTSReader,
                                          vtkMCubesReader,
                                          vtkAVSucdReader,
                                          vtkMFIXReader,
                                          vtkOpenFOAMReader,
                                          vtkSTLWriter,
                                          vtkBYUReader,
                                          vtkOBJReader)
    from vtkmodules.vtkIOImage import (vtkBMPReader,
                                       vtkDEMReader,
                                       vtkDICOMImageReader,
                                       vtkDICOMImageReader,
                                       vtkJPEGReader,
                                       vtkJPEGReader,
                                       vtkMetaImageReader,
                                       vtkNrrdReader,
                                       vtkNrrdReader,
                                       vtkPNGReader,
                                       vtkPNMReader,
                                       vtkSLCReader,
                                       vtkTIFFReader,
                                       vtkTIFFReader)
    from vtkmodules.vtkIOXML import (vtkXMLReader,
                                     vtkXMLWriter,
                                     vtkXMLPRectilinearGridReader,
                                     vtkXMLPUnstructuredGridReader,
                                     vtkXMLPImageDataReader,
                                     vtkXMLImageDataReader,
                                     vtkXMLImageDataWriter,
                                     vtkXMLPolyDataReader,
                                     vtkXMLPolyDataWriter,
                                     vtkXMLRectilinearGridReader,
                                     vtkXMLRectilinearGridWriter,
                                     vtkXMLUnstructuredGridReader,
                                     vtkXMLUnstructuredGridWriter,
                                     vtkXMLStructuredGridReader,
                                     vtkXMLStructuredGridWriter,
                                     vtkXMLMultiBlockDataReader,
                                     vtkXMLMultiBlockDataWriter)
    from vtkmodules.vtkIOEnSight import vtkGenericEnSightReader
    from vtkmodules.vtkIOLegacy import (vtkDataWriter,
                                        vtkDataReader,
                                        vtkStructuredGridReader,
                                        vtkStructuredGridWriter,
                                        vtkPolyDataWriter,
                                        vtkPolyDataReader,
                                        vtkUnstructuredGridReader,
                                        vtkUnstructuredGridWriter,
                                        vtkRectilinearGridReader,
                                        vtkRectilinearGridWriter,
                                        vtkDataSetWriter,
                                        vtkPolyDataReader,
                                        vtkDataSetReader)
    from vtkmodules.vtkCommonDataModel import (vtkImplicitFunction,
                                               vtkDataObject,
                                               vtkExplicitStructuredGrid,
                                               vtkPyramid,
                                               vtkPlane,
                                               vtkPlanes,
                                               vtkPiecewiseFunction,
                                               vtkPolyPlane,
                                               vtkCellArray,
                                               vtkStructuredGrid,
                                               vtkUnstructuredGrid,
                                               vtkDataSetAttributes,
                                               vtkTable,
                                               vtkPlaneCollection,
                                               vtkDataSet,
                                               vtkPointLocator,
                                               vtkCellLocator,
                                               vtkMultiBlockDataSet,
                                               vtkCompositeDataSet,
                                               vtkFieldData,
                                               vtkPolyData,
                                               vtkPolyLine,
                                               vtkRectilinearGrid,
                                               vtkImageData,
                                               vtkStaticPointLocator,
                                               vtkSelectionNode,
                                               vtkSelection,
                                               vtkPerlinNoise,
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
                                               VTK_WEDGE)
    from vtkmodules.vtkRenderingAnnotation import (vtkScalarBarActor,
                                                   vtkCornerAnnotation,
                                                   vtkAxesActor,
                                                   vtkAnnotatedCubeActor,
                                                   vtkLegendBoxActor,
                                                   vtkCubeAxesActor)
    from vtkmodules.vtkRenderingCore import (vtkTexture,
                                             vtkSkybox,
                                             vtkPropAssembly,
                                             vtkRenderer,
                                             vtkMapper,
                                             vtkPolyDataMapper2D,
                                             vtkPolyDataMapper,
                                             vtkCoordinate,
                                             vtkRenderWindow,
                                             vtkRenderWindowInteractor,
                                             vtkActor2D,
                                             vtkSelectVisiblePoints,
                                             vtkTextActor,
                                             vtkVolume,
                                             vtkVolumeProperty,
                                             vtkColorTransferFunction,
                                             vtkActor,
                                             vtkDataSetMapper,
                                             vtkProperty,
                                             vtkWorldPointPicker,
                                             vtkPointPicker,
                                             vtkRenderedAreaPicker,
                                             vtkWindowToImageFilter,
                                             vtkLight,
                                             vtkLightActor,
                                             vtkLightKit,
                                             vtkCamera,
                                             vtkImageActor)
    from vtkmodules.vtkCommonComputationalGeometry import (vtkParametricSpline,
                                                           vtkParametricBour,
                                                           vtkParametricBoy,
                                                           vtkParametricCatalanMinimal,
                                                           vtkParametricConicSpiral,
                                                           vtkParametricCrossCap,
                                                           vtkParametricDini,
                                                           vtkParametricEllipsoid,
                                                           vtkParametricEnneper,
                                                           vtkParametricFigure8Klein,
                                                           vtkParametricHenneberg,
                                                           vtkParametricKlein,
                                                           vtkParametricKuen,
                                                           vtkParametricMobius,
                                                           vtkParametricPluckerConoid,
                                                           vtkParametricPseudosphere,
                                                           vtkParametricRandomHills,
                                                           vtkParametricRoman,
                                                           vtkParametricSuperEllipsoid,
                                                           vtkParametricSuperToroid,
                                                           vtkParametricTorus,
                                                           vtkParametricFunction,
                                                           vtkParametricBohemianDome)
    from vtkmodules.vtkCommonCore import (VTK_COURIER,
                                          VTK_TIMES,
                                          VTK_ARIAL,
                                          vtkUnsignedCharArray,
                                          vtkIdTypeArray,
                                          vtkCharArray,
                                          vtkBitArray,
                                          vtkFileOutputWindow,
                                          vtkOutputWindow,
                                          vtkStringOutputWindow,
                                          vtkIdList,
                                          vtkStringArray,
                                          vtkCommand,
                                          vtkTypeUInt32Array,
                                          vtkDataArray,
                                          vtkPoints,
                                          vtkIdList,
                                          vtkLookupTable,
                                          VTK_UNSIGNED_CHAR,
                                          vtkAbstractArray,
                                          vtkDoubleArray)
    from vtkmodules.vtkCommonMath import (vtkMatrix4x4,
                                          vtkMatrix3x3)
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersCore import (vtkAppendFilter,
                                           vtkUnstructuredGridToExplicitStructuredGrid,
                                           vtkExplicitStructuredGridToUnstructuredGrid,
                                           vtkCutter,
                                           vtkPointDataToCellData,
                                           vtkMassProperties,
                                           vtkCenterOfMass,
                                           vtkStripper,
                                           vtkDelaunay2D,
                                           VTK_BEST_FITTING_PLANE,
                                           vtkAppendArcLength,
                                           vtkCleanPolyData,
                                           vtkQuadricDecimation,
                                           vtkPolyDataNormals,
                                           vtkTriangleFilter,
                                           vtkSmoothPolyDataFilter,
                                           vtkDecimatePro,
                                           vtkTubeFilter,
                                           vtkAppendPolyData,
                                           vtkFeatureEdges,
                                           vtkGlyph3D,
                                           vtkResampleWithDataSet,
                                           vtkProbeFilter,
                                           vtkClipPolyData,
                                           vtkImplicitPolyDataDistance,
                                           vtkThreshold,
                                           vtkElevationFilter,
                                           vtkContourFilter,
                                           vtkMarchingCubes,
                                           vtkFlyingEdges3D,
                                           vtkCellCenters,
                                           vtkConnectivityFilter,
                                           vtkCellDataToPointData,
                                           vtkDelaunay3D,
                                           vtkCutter)
    from vtkmodules.vtkFiltersGeneral import (vtkTableBasedClipDataSet,
                                              vtkTableToPolyData,
                                              vtkOBBTree,
                                              vtkRectilinearGridToPointSet,
                                              vtkClipClosedSurface,
                                              vtkIntersectionPolyDataFilter,
                                              vtkCurvatures,
                                              vtkBoxClipDataSet,
                                              vtkTableBasedClipDataSet,
                                              vtkWarpScalar,
                                              vtkWarpVector,
                                              vtkDataSetTriangleFilter,
                                              vtkGradientFilter,
                                              vtkShrinkFilter,
                                              vtkBooleanOperationPolyDataFilter,
                                              vtkTransformFilter)
    from vtkmodules.vtkFiltersModeling import (vtkOutlineFilter,
                                               vtkRibbonFilter,
                                               vtkLinearExtrusionFilter,
                                               vtkRotationalExtrusionFilter,
                                               vtkDijkstraGraphGeodesicPath,
                                               vtkFillHolesFilter,
                                               vtkLinearSubdivisionFilter,
                                               vtkButterflySubdivisionFilter,
                                               vtkLoopSubdivisionFilter,
                                               vtkSelectEnclosedPoints)
    from vtkmodules.vtkFiltersSources import (vtkOutlineCornerFilter,
                                              vtkParametricFunctionSource,
                                              vtkParametricFunctionSource,
                                              vtkPlaneSource,
                                              vtkArcSource,
                                              vtkTessellatedBoxSource,
                                              vtkOutlineCornerSource,
                                              vtkCubeSource,
                                              vtkArrowSource,
                                              vtkCylinderSource,
                                              vtkSphereSource,
                                              vtkPlaneSource,
                                              vtkLineSource,
                                              vtkCubeSource,
                                              vtkConeSource,
                                              vtkDiskSource,
                                              vtkRegularPolygonSource,
                                              vtkLineSource,
                                              vtkPointSource,
                                              vtkArrowSource,
                                              vtkFrustumSource)
    from vtkmodules.vtkFiltersGeometry import (vtkGeometryFilter,
                                               vtkStructuredGridGeometryFilter,
                                               vtkCompositeDataGeometryFilter,
                                               vtkDataSetSurfaceFilter)
    from vtkmodules.vtkFiltersHybrid import vtkPolyDataSilhouette
    from vtkmodules.vtkFiltersExtraction import (vtkExtractEdges,
                                                 vtkExtractGeometry,
                                                 vtkExtractGrid,
                                                 vtkExtractSelection)
    from vtkmodules.vtkFiltersTexture import (vtkTextureMapToPlane,
                                              vtkTextureMapToSphere)
    from vtkmodules.vtkFiltersPoints import (vtkGaussianKernel,
                                             vtkPointInterpolator)
    from vtkmodules.vtkFiltersVerdict import (vtkCellSizeFilter,
                                              vtkCellQuality)
    from vtkmodules.vtkImagingGeneral import vtkImageGaussianSmooth
    from vtkmodules.vtkImagingCore import (vtkExtractVOI,
                                           vtkImageExtractComponents,
                                           vtkImageDifference,
                                           vtkImageFlip,
                                           vtkRTAnalyticSource)
    from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer
    from vtkmodules.vtkCommonExecutionModel import vtkImageToStructuredGrid
    from vtkmodules.numpy_interface.dataset_adapter import (VTKObjectWrapper,
                                                            numpyTovtkDataArray,
                                                            VTKArray)
    from vtkmodules.util.numpy_support import (vtk_to_numpy,
                                               numpy_to_vtk,
                                               numpy_to_vtkIdTypeArray,
                                               get_vtk_array_type)
    from vtkmodules.vtkCommonCore import (buffer_shared,
                                          vtkAbstractArray,
                                          vtkWeakReference)

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
        from vtk.vtkCommonKitPython import (buffer_shared,
                                            vtkAbstractArray,
                                            vtkWeakReference)
    except ImportError:
        from vtk.vtkCommonCore import (buffer_shared,
                                       vtkAbstractArray,
                                       vtkWeakReference)

    from vtk.util.numpy_support import vtk_to_numpy
    from vtk.numpy_interface.dataset_adapter import (VTKObjectWrapper,
                                                     VTKArray,
                                                     numpyTovtkDataArray)
    from vtk.util.numpy_support import (vtk_to_numpy,
                                        numpy_to_vtk,
                                        numpy_to_vtkIdTypeArray,
                                        get_vtk_array_type)

    # vtk8 already has an import all, so we can just mirror it here at
    # no cost
    from vtk import *

    import vtk

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

        pass
