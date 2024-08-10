"""
GL-dependent imports from VTK.

These are the modules within VTK requiring libGL that must be loaded
across pyvista's plotting API. These imports have the potential to
raise an ImportError if the user does not have libGL installed.

    ImportError: libGL.so.1: cannot open shared object file: No such file or directory

"""

# ruff: noqa: F401
from __future__ import annotations

try:
    # Necessary for displaying charts, otherwise crashes on rendering
    from vtkmodules import vtkRenderingContextOpenGL2
except ImportError:  # pragma: no cover
    vtkRenderingContextOpenGL2 = None


from vtkmodules.vtkRenderingOpenGL2 import vtkCameraPass
from vtkmodules.vtkRenderingOpenGL2 import vtkCompositePolyDataMapper2
from vtkmodules.vtkRenderingOpenGL2 import vtkDepthOfFieldPass
from vtkmodules.vtkRenderingOpenGL2 import vtkEDLShading
from vtkmodules.vtkRenderingOpenGL2 import vtkGaussianBlurPass
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLFXAAPass
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLHardwareSelector
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLRenderer
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLTexture
from vtkmodules.vtkRenderingOpenGL2 import vtkRenderPassCollection
from vtkmodules.vtkRenderingOpenGL2 import vtkRenderStepsPass
from vtkmodules.vtkRenderingOpenGL2 import vtkSequencePass
from vtkmodules.vtkRenderingOpenGL2 import vtkShadowMapPass
from vtkmodules.vtkRenderingOpenGL2 import vtkSSAAPass
from vtkmodules.vtkRenderingOpenGL2 import vtkSSAOPass
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLGPUVolumeRayCastMapper
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
