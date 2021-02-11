"""
Common vtk imports.

These are the default modules within vtk that must be loaded across
pyvista.  Here, we attempt to import modules using the VTK9 vtkmodules
import strategy, which lets us only have to import from select modules
and not the entire library.
"""

try:
    from vtkmodules.vtkCommonCore import vtkVersion
    _vtk9 = vtkVersion().GetVTKMajorVersion() >= 9
except ImportError:
    _vtk9 = False

if _vtk9:
    from vtkmodules.vtkIOExport import vtkOBJExporter
    from vtkmodules.vtkIOExportGL2PS import vtkGL2PSExporter
    from vtkmodules.vtkInteractionWidgets import vtkScalarBarWidget
    from vtkmodules.vtkRenderingLabel import (vtkPointSetToLabelHierarchy,
                                              vtkLabelPlacementMapper)
    from vtkmodules.vtkRenderingVolume import (vtkFixedPointVolumeRayCastMapper,
                                               vtkGPUVolumeRayCastMapper)
    from vtkmodules.vtkRenderingVolumeOpenGL2 import (vtkOpenGLGPUVolumeRayCastMapper,
                                                      vtkSmartVolumeMapper)
    from vtkmodules.vtkRenderingOpenGL2 import (vtkOpenGLHardwareSelector,
                                                vtkOpenGLTexture)
    from vtkmodules.vtkIOInfovis import vtkDelimitedTextReader
    from vtkmodules.vtkIOPLY import vtkPLYReader, vtkPLYWriter
    from vtkmodules.vtkIOGeometry import (vtkSTLReader,
                                          vtkSTLWriter,
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
                                       vtkTIFFReader,
    )

    from vtkmodules.vtkIOXML import (vtkXMLReader,
                                     vtkXMLWriter,
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
    from vtkmodules.vtkCommonDataModel import (vtkDataObject,
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
                                               VTK_WEDGE
    )
    from vtkmodules.vtkRenderingAnnotation import (vtkScalarBarActor,
                                                   vtkCornerAnnotation,
                                                   vtkLegendBoxActor,
                                                   )
    from vtkmodules.vtkRenderingCore import (vtkTexture,
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
                                             vtkImageActor,
                                             )
    from vtkmodules.vtkCommonCore import (vtkIdList,
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
    from vtkmodules.vtkCommonMath import vtkMatrix4x4
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersCore import (vtkAppendFilter,
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
                                              )
    from vtkmodules.vtkFiltersModeling import (vtkOutlineFilter,
                                               vtkRibbonFilter,
                                               vtkLinearExtrusionFilter,
                                               vtkRotationalExtrusionFilter,
                                               vtkDijkstraGraphGeodesicPath,
                                               vtkFillHolesFilter,
                                               vtkLinearSubdivisionFilter,
                                               vtkButterflySubdivisionFilter,
                                               vtkLoopSubdivisionFilter,
                                               vtkSelectEnclosedPoints,
                                               )
    from vtkmodules.vtkFiltersSources import (vtkOutlineCornerFilter,
                                              vtkLineSource,
                                              vtkPointSource,
                                              vtkArrowSource)
    from vtkmodules.vtkFiltersGeometry import (vtkGeometryFilter,
                                               vtkStructuredGridGeometryFilter,
                                               vtkCompositeDataGeometryFilter,
                                               vtkDataSetSurfaceFilter)
    from vtkmodules.vtkFiltersExtraction import (vtkExtractEdges,
                                                 vtkExtractGeometry,
                                                 vtkExtractGrid,
                                                 vtkExtractSelection)
    from vtkmodules.vtkFiltersTexture import (vtkTextureMapToPlane,
                                              vtkTextureMapToSphere)
    from vtkmodules.vtkFiltersPoints import (vtkGaussianKernel,
                                             vtkPointInterpolator)
    from vtkmodules.vtkFiltersVerdict import (vtkCellSizeFilter,
                                              vtkCellQuality,
                                              )
    from vtkmodules.vtkImagingGeneral import vtkImageGaussianSmooth
    from vtkmodules.vtkImagingCore import vtkExtractVOI
    from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer
    from vtkmodules.vtkCommonExecutionModel import vtkImageToStructuredGrid

    from vtkmodules.numpy_interface.dataset_adapter import (VTKObjectWrapper,
                                                            numpyTovtkDataArray,
                                                            VTKArray)
    from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

    from vtkmodules.vtkCommonCore import (buffer_shared,
                                          vtkAbstractArray,
                                          vtkWeakReference)

else:
    from vtk.vtkCommonKitPython import (buffer_shared,
                                        vtkAbstractArray,
                                        vtkWeakReference
                                        )
    from vtk.util.numpy_support import vtk_to_numpy
    from vtk.numpy_interface.dataset_adapter import (VTKObjectWrapper,
                                                     VTKArray,
                                                     numpyTovtkDataArray)
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

    # vtk8 already has an import all, so we can just mirror it here at no cost
    from vtk import *
