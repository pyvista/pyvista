"""
Common vtk imports.

These are the default modules within vtk that must be loaded across
pyvista.  Here, we attempt to import modules using the VTK9 vtkmodules
import strategy, which lets us only have to import from select modules
and not the entire library.
"""

# perhaps use:
# vtk.vtkVersion().GetVTKMajorVersion() >= 9:

try:
    from vtkmodules.numpy_interface.dataset_adapter import (VTKObjectWrapper,
                                                            numpyTovtkDataArray)
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOXML import (vtkXMLReader,
                                     vtkXMLWriter,
                                     vtkXMLMultiBlockDataReader,
                                     vtkXMLMultiBlockDataWriter)
    from vtkmodules.vtkIOEnSight import vtkGenericEnSightReader
    from vtkmodules.vtkIOLegacy import (vtkDataWriter, vtkDataReader,
                                        vtkDataSetWriter,
                                        vtkDataSetReader)
    from vtkmodules.vtkCommonDataModel import (vtkDataObject,
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
                                               )
    from vtkmodules.vtkRenderingCore import vtkTexture
    from vtkmodules.vtkCommonCore import (vtkIdList, vtkDataArray,
                                          vtkPoints, vtkIdList,
                                          vtkAbstractArray, vtkDoubleArray)
    from vtkmodules.vtkCommonMath import vtkMatrix4x4
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersCore import (vtkAppendFilter,
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
                                               vtkCompositeDataGeometryFilter,
                                               vtkDataSetSurfaceFilter)
    from vtkmodules.vtkFiltersExtraction import (vtkExtractEdges,
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
    from vtkmodules.util.numpy_support import vtk_to_numpy

except:
    from vtk.util.numpy_support import vtk_to_numpy
    from vtk.numpy_interface.dataset_adapter import (VTKObjectWrapper,
                                                     numpyTovtkDataArray)
    from vtk.util.numpy_support import vtk_to_numpy

    # vtk8 already has an import all, so we can just mirror it here at no cost
    from vtk import *
