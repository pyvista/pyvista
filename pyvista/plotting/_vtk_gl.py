"""
GL-dependent imports from VTK.

These are the modules within VTK requiring libGL that must be loaded
across pyvista's plotting API. These imports have the potential to
raise an ImportError if the user does not have libGL installed.

    ImportError: libGL.so.1: cannot open shared object file: No such file or directory

"""
# flake8: noqa: F401

try:
    # Necessary for displaying charts, otherwise crashes on rendering
    from vtkmodules import vtkRenderingContextOpenGL2
except ImportError:  # pragma: no cover
    vtkRenderingContextOpenGL2 = None


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
from vtkmodules.vtkRenderingVolumeOpenGL2 import (
    vtkOpenGLGPUVolumeRayCastMapper,
    vtkSmartVolumeMapper,
)
