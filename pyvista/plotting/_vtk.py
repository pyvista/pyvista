"""All imports from VTK (including GL-dependent).

These are the modules within VTK that must be loaded across pyvista's
plotting API. Here, we attempt to import modules using the ``vtkmodules``
package, which lets us only have to import from select modules and not
the entire library.

"""

from __future__ import annotations

from vtkmodules.vtkChartsCore import vtkAxis as vtkAxis
from vtkmodules.vtkChartsCore import vtkChart as vtkChart
from vtkmodules.vtkChartsCore import vtkChartBox as vtkChartBox
from vtkmodules.vtkChartsCore import vtkChartPie as vtkChartPie
from vtkmodules.vtkChartsCore import vtkChartXY as vtkChartXY
from vtkmodules.vtkChartsCore import vtkChartXYZ as vtkChartXYZ
from vtkmodules.vtkChartsCore import vtkPlotArea as vtkPlotArea
from vtkmodules.vtkChartsCore import vtkPlotBar as vtkPlotBar
from vtkmodules.vtkChartsCore import vtkPlotBox as vtkPlotBox
from vtkmodules.vtkChartsCore import vtkPlotLine as vtkPlotLine
from vtkmodules.vtkChartsCore import vtkPlotLine3D as vtkPlotLine3D
from vtkmodules.vtkChartsCore import vtkPlotPie as vtkPlotPie
from vtkmodules.vtkChartsCore import vtkPlotPoints as vtkPlotPoints
from vtkmodules.vtkChartsCore import vtkPlotPoints3D as vtkPlotPoints3D
from vtkmodules.vtkChartsCore import vtkPlotStacked as vtkPlotStacked
from vtkmodules.vtkChartsCore import vtkPlotSurface as vtkPlotSurface
from vtkmodules.vtkCommonColor import vtkColorSeries as vtkColorSeries
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleImage as vtkInteractorStyleImage
from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleJoystickActor as vtkInteractorStyleJoystickActor,
)
from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleJoystickCamera as vtkInteractorStyleJoystickCamera,
)
from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleRubberBand2D as vtkInteractorStyleRubberBand2D,
)
from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleRubberBandPick as vtkInteractorStyleRubberBandPick,
)
from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleRubberBandZoom as vtkInteractorStyleRubberBandZoom,
)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTerrain as vtkInteractorStyleTerrain
from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleTrackballActor as vtkInteractorStyleTrackballActor,
)
from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleTrackballCamera as vtkInteractorStyleTrackballCamera,
)
from vtkmodules.vtkInteractionWidgets import vtkBoxWidget as vtkBoxWidget
from vtkmodules.vtkInteractionWidgets import vtkButtonWidget as vtkButtonWidget
from vtkmodules.vtkInteractionWidgets import (
    vtkCameraOrientationWidget as vtkCameraOrientationWidget,
)
from vtkmodules.vtkInteractionWidgets import (
    vtkDistanceRepresentation3D as vtkDistanceRepresentation3D,
)
from vtkmodules.vtkInteractionWidgets import vtkDistanceWidget as vtkDistanceWidget
from vtkmodules.vtkInteractionWidgets import vtkImplicitPlaneWidget as vtkImplicitPlaneWidget
from vtkmodules.vtkInteractionWidgets import vtkLineWidget as vtkLineWidget
from vtkmodules.vtkInteractionWidgets import vtkLogoRepresentation as vtkLogoRepresentation
from vtkmodules.vtkInteractionWidgets import vtkLogoWidget as vtkLogoWidget
from vtkmodules.vtkInteractionWidgets import (
    vtkOrientationMarkerWidget as vtkOrientationMarkerWidget,
)
from vtkmodules.vtkInteractionWidgets import vtkPlaneWidget as vtkPlaneWidget
from vtkmodules.vtkInteractionWidgets import (
    vtkPointHandleRepresentation3D as vtkPointHandleRepresentation3D,
)
from vtkmodules.vtkInteractionWidgets import vtkResliceCursorPicker as vtkResliceCursorPicker
from vtkmodules.vtkInteractionWidgets import vtkScalarBarWidget as vtkScalarBarWidget
from vtkmodules.vtkInteractionWidgets import vtkSliderRepresentation2D as vtkSliderRepresentation2D
from vtkmodules.vtkInteractionWidgets import vtkSliderWidget as vtkSliderWidget
from vtkmodules.vtkInteractionWidgets import vtkSphereWidget as vtkSphereWidget
from vtkmodules.vtkInteractionWidgets import vtkSplineWidget as vtkSplineWidget
from vtkmodules.vtkInteractionWidgets import (
    vtkTexturedButtonRepresentation2D as vtkTexturedButtonRepresentation2D,
)
from vtkmodules.vtkRenderingAnnotation import vtkAnnotatedCubeActor as vtkAnnotatedCubeActor
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor as vtkAxesActor
from vtkmodules.vtkRenderingAnnotation import vtkAxisActor as vtkAxisActor
from vtkmodules.vtkRenderingAnnotation import vtkAxisActor2D as vtkAxisActor2D
from vtkmodules.vtkRenderingAnnotation import vtkCornerAnnotation as vtkCornerAnnotation
from vtkmodules.vtkRenderingAnnotation import vtkCubeAxesActor as vtkCubeAxesActor
from vtkmodules.vtkRenderingAnnotation import vtkLegendBoxActor as vtkLegendBoxActor
from vtkmodules.vtkRenderingAnnotation import vtkLegendScaleActor as vtkLegendScaleActor
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor as vtkScalarBarActor
from vtkmodules.vtkRenderingContext2D import vtkBlockItem as vtkBlockItem
from vtkmodules.vtkRenderingContext2D import vtkBrush as vtkBrush
from vtkmodules.vtkRenderingContext2D import vtkContext2D as vtkContext2D
from vtkmodules.vtkRenderingContext2D import vtkContextActor as vtkContextActor
from vtkmodules.vtkRenderingContext2D import vtkContextScene as vtkContextScene
from vtkmodules.vtkRenderingContext2D import vtkImageItem as vtkImageItem
from vtkmodules.vtkRenderingContext2D import vtkPen as vtkPen
from vtkmodules.vtkRenderingCore import VTK_RESOLVE_OFF as VTK_RESOLVE_OFF
from vtkmodules.vtkRenderingCore import VTK_RESOLVE_POLYGON_OFFSET as VTK_RESOLVE_POLYGON_OFFSET
from vtkmodules.vtkRenderingCore import VTK_RESOLVE_SHIFT_ZBUFFER as VTK_RESOLVE_SHIFT_ZBUFFER
from vtkmodules.vtkRenderingCore import vtkAbstractMapper as vtkAbstractMapper
from vtkmodules.vtkRenderingCore import vtkActor as vtkActor
from vtkmodules.vtkRenderingCore import vtkActor2D as vtkActor2D
from vtkmodules.vtkRenderingCore import vtkAreaPicker as vtkAreaPicker
from vtkmodules.vtkRenderingCore import vtkCamera as vtkCamera
from vtkmodules.vtkRenderingCore import vtkCellPicker as vtkCellPicker
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction as vtkColorTransferFunction
from vtkmodules.vtkRenderingCore import (
    vtkCompositeDataDisplayAttributes as vtkCompositeDataDisplayAttributes,
)
from vtkmodules.vtkRenderingCore import vtkCompositePolyDataMapper as vtkCompositePolyDataMapper
from vtkmodules.vtkRenderingCore import vtkCoordinate as vtkCoordinate
from vtkmodules.vtkRenderingCore import vtkDataSetMapper as vtkDataSetMapper
from vtkmodules.vtkRenderingCore import vtkFollower as vtkFollower
from vtkmodules.vtkRenderingCore import vtkHardwarePicker as vtkHardwarePicker
from vtkmodules.vtkRenderingCore import vtkImageActor as vtkImageActor
from vtkmodules.vtkRenderingCore import vtkInteractorStyle as vtkInteractorStyle
from vtkmodules.vtkRenderingCore import vtkLight as vtkLight
from vtkmodules.vtkRenderingCore import vtkLightActor as vtkLightActor
from vtkmodules.vtkRenderingCore import vtkLightKit as vtkLightKit
from vtkmodules.vtkRenderingCore import vtkMapper as vtkMapper
from vtkmodules.vtkRenderingCore import vtkPointGaussianMapper as vtkPointGaussianMapper
from vtkmodules.vtkRenderingCore import vtkPointPicker as vtkPointPicker
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper as vtkPolyDataMapper
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper2D as vtkPolyDataMapper2D
from vtkmodules.vtkRenderingCore import vtkProp as vtkProp
from vtkmodules.vtkRenderingCore import vtkProp3D as vtkProp3D
from vtkmodules.vtkRenderingCore import vtkPropAssembly as vtkPropAssembly
from vtkmodules.vtkRenderingCore import vtkPropCollection as vtkPropCollection
from vtkmodules.vtkRenderingCore import vtkProperty as vtkProperty
from vtkmodules.vtkRenderingCore import vtkPropPicker as vtkPropPicker
from vtkmodules.vtkRenderingCore import vtkRenderedAreaPicker as vtkRenderedAreaPicker
from vtkmodules.vtkRenderingCore import vtkRenderer as vtkRenderer
from vtkmodules.vtkRenderingCore import vtkRenderWindow as vtkRenderWindow
from vtkmodules.vtkRenderingCore import vtkRenderWindowInteractor as vtkRenderWindowInteractor
from vtkmodules.vtkRenderingCore import vtkScenePicker as vtkScenePicker
from vtkmodules.vtkRenderingCore import vtkSelectVisiblePoints as vtkSelectVisiblePoints
from vtkmodules.vtkRenderingCore import vtkSkybox as vtkSkybox
from vtkmodules.vtkRenderingCore import vtkTextActor as vtkTextActor
from vtkmodules.vtkRenderingCore import vtkTextProperty as vtkTextProperty
from vtkmodules.vtkRenderingCore import vtkTexture as vtkTexture
from vtkmodules.vtkRenderingCore import vtkViewport as vtkViewport
from vtkmodules.vtkRenderingCore import vtkVolume as vtkVolume
from vtkmodules.vtkRenderingCore import vtkVolumeProperty as vtkVolumeProperty
from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter as vtkWindowToImageFilter
from vtkmodules.vtkRenderingCore import vtkWorldPointPicker as vtkWorldPointPicker
from vtkmodules.vtkRenderingFreeType import (
    vtkMathTextFreeTypeTextRenderer as vtkMathTextFreeTypeTextRenderer,
)
from vtkmodules.vtkRenderingFreeType import vtkVectorText as vtkVectorText
from vtkmodules.vtkRenderingLabel import vtkLabelPlacementMapper as vtkLabelPlacementMapper
from vtkmodules.vtkRenderingLabel import vtkPointSetToLabelHierarchy as vtkPointSetToLabelHierarchy
from vtkmodules.vtkRenderingUI import (
    vtkGenericRenderWindowInteractor as vtkGenericRenderWindowInteractor,
)
from vtkmodules.vtkRenderingVolume import (
    vtkFixedPointVolumeRayCastMapper as vtkFixedPointVolumeRayCastMapper,
)
from vtkmodules.vtkRenderingVolume import vtkGPUVolumeRayCastMapper as vtkGPUVolumeRayCastMapper
from vtkmodules.vtkRenderingVolume import (
    vtkUnstructuredGridVolumeRayCastMapper as vtkUnstructuredGridVolumeRayCastMapper,
)
from vtkmodules.vtkRenderingVolume import vtkVolumePicker as vtkVolumePicker
from vtkmodules.vtkViewsContext2D import vtkContextInteractorStyle as vtkContextInteractorStyle

from pyvista.core._vtk_core import *

from ._vtk_gl import *
