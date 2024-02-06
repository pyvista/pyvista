"""
All imports from VTK (including GL-dependent).

These are the modules within VTK that must be loaded across pyvista's
plotting API. Here, we attempt to import modules using the ``vtkmodules``
package, which lets us only have to import from select modules and not
the entire library.

"""
# flake8: noqa: F401

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
from vtkmodules.vtkInteractionWidgets import (
    vtkBoxWidget,
    vtkButtonWidget,
    vtkDistanceRepresentation3D,
    vtkDistanceWidget,
    vtkImplicitPlaneWidget,
    vtkLineWidget,
    vtkLogoRepresentation,
    vtkLogoWidget,
    vtkOrientationMarkerWidget,
    vtkPlaneWidget,
    vtkPointHandleRepresentation3D,
    vtkResliceCursorPicker,
    vtkScalarBarWidget,
    vtkSliderRepresentation2D,
    vtkSliderWidget,
    vtkSphereWidget,
    vtkSplineWidget,
    vtkTexturedButtonRepresentation2D,
)
from vtkmodules.vtkRenderingAnnotation import (
    vtkAnnotatedCubeActor,
    vtkAxesActor,
    vtkAxisActor2D,
    vtkCornerAnnotation,
    vtkCubeAxesActor,
    vtkLegendBoxActor,
    vtkLegendScaleActor,
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
    from vtkmodules.vtkRenderingCore import vtkHardwarePicker
except ImportError:  # pragma: no cover
    # VTK < 9.2 is missing this class
    vtkHardwarePicker = None
from vtkmodules.vtkRenderingCore import (
    vtkAbstractMapper,
    vtkActor,
    vtkActor2D,
    vtkAreaPicker,
    vtkCamera,
    vtkCellPicker,
    vtkColorTransferFunction,
    vtkCompositeDataDisplayAttributes,
    vtkCompositePolyDataMapper,
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
    vtkProp3D,
    vtkPropAssembly,
    vtkProperty,
    vtkPropPicker,
    vtkRenderedAreaPicker,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkScenePicker,
    vtkSelectVisiblePoints,
    vtkSkybox,
    vtkTextActor,
    vtkTextProperty,
    vtkTexture,
    vtkVolume,
    vtkVolumeProperty,
    vtkWindowToImageFilter,
    vtkWorldPointPicker,
)
from vtkmodules.vtkRenderingFreeType import vtkMathTextFreeTypeTextRenderer, vtkVectorText
from vtkmodules.vtkRenderingLabel import vtkLabelPlacementMapper, vtkPointSetToLabelHierarchy
from vtkmodules.vtkRenderingUI import vtkGenericRenderWindowInteractor
from vtkmodules.vtkRenderingVolume import (
    vtkFixedPointVolumeRayCastMapper,
    vtkGPUVolumeRayCastMapper,
    vtkUnstructuredGridVolumeRayCastMapper,
    vtkVolumePicker,
)
from vtkmodules.vtkViewsContext2D import vtkContextInteractorStyle

from pyvista.core._vtk_core import *

from ._vtk_gl import *
