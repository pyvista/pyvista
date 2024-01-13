"""
Limited imports from VTK (excludes any GL-dependent).

These are the modules within VTK that must be loaded across pyvista's
core API. Here, we attempt to import modules using the ``vtkmodules``
package, which lets us only have to import from select modules and not
the entire library.

"""
# flake8: noqa: F401

from vtkmodules.vtkCommonCore import vtkVersion

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
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
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
    VTK_FONT_FILE,
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
    vtkLogger,
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
    VTK_BEZIER_CURVE,
    VTK_BEZIER_HEXAHEDRON,
    VTK_BEZIER_PYRAMID,
    VTK_BEZIER_QUADRILATERAL,
    VTK_BEZIER_TETRAHEDRON,
    VTK_BEZIER_TRIANGLE,
    VTK_BEZIER_WEDGE,
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
    vtkCell,
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
    vtkIterativeClosestPointTransform,
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

try:  # Introduced prior to VTK 9.2
    from vtkmodules.vtkCommonDataModel import VTK_TRIQUADRATIC_PYRAMID
except ImportError:  # pragma: no cover
    pass

from vtkmodules.vtkCommonExecutionModel import (
    vtkAlgorithm,
    vtkAlgorithmOutput,
    vtkImageToStructuredGrid,
)
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
    vtkExtractCellsByType,
    vtkExtractGeometry,
    vtkExtractGrid,
    vtkExtractSelection,
)
from vtkmodules.vtkFiltersFlowPaths import vtkEvenlySpacedStreamlines2D, vtkStreamTracer

try:  # Introduced VTK v9.1.0
    from vtkmodules.vtkFiltersGeneral import vtkRemovePolyData
except ImportError:  # pragma: no cover
    pass
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
    vtkBandedPolyDataContourFilter,
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
except ImportError:  # pragma: no cover
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
from vtkmodules.vtkIOGeometry import vtkSTLWriter
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

try:
    from vtkmodules.vtkImagingMorphological import vtkImageDilateErode3D
except ImportError:  # pragma: no cover
    pass

try:
    from vtkmodules.vtkPythonContext2D import vtkPythonItem
except ImportError:  # pragma: no cover
    # `vtkmodules.vtkPythonContext2D` is unavailable in some versions of `vtk` (see #3224)

    class vtkPythonItem:  # type: ignore
        """Empty placeholder."""

        def __init__(self):  # pragma: no cover
            """Raise version error on init."""
            from pyvista.core.errors import VTKVersionError

            raise VTKVersionError('Chart backgrounds require the vtkPythonContext2D module')


from vtkmodules.vtkImagingFourier import (
    vtkImageButterworthHighPass,
    vtkImageButterworthLowPass,
    vtkImageFFT,
    vtkImageRFFT,
)

# 9.1+ imports
try:
    from vtkmodules.vtkFiltersPoints import vtkConvertToPointCloud
except ImportError:  # pragma: no cover
    pass

try:  # Introduced prior to VTK 9.3
    from vtkmodules.vtkRenderingCore import vtkViewport
except ImportError:  # pragma: no cover
    pass

# 9.3+ imports
try:
    from vtkmodules.vtkFiltersCore import vtkPackLabels, vtkSurfaceNets3D
except ImportError:  # pragma: no cover
    pass
