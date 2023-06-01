"""
Import from vtk.

These are the modules within vtk that must be loaded across pyvista.
Here, we attempt to import modules using the ``vtkmodules``
package, which lets us only have to import from select modules and not
the entire library.

"""
# flake8: noqa: F401

from vtkmodules.vtkCommonColor import vtkColorSeries
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

from pyvista.core._vtk_core import *

try:
    # Necessary for displaying charts, otherwise crashes on rendering
    import vtkmodules.vtkRenderingContextOpenGL2

    _has_vtkRenderingContextOpenGL2 = True
except ImportError:  # pragma: no cover
    _has_vtkRenderingContextOpenGL2 = False

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
    vtkProp3D,
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
from vtkmodules.vtkRenderingFreeType import vtkMathTextFreeTypeTextRenderer, vtkVectorText
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
    vtkUnstructuredGridVolumeRayCastMapper,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import (
    vtkOpenGLGPUVolumeRayCastMapper,
    vtkSmartVolumeMapper,
)
from vtkmodules.vtkViewsContext2D import vtkContextInteractorStyle
