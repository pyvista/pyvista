"""GL-dependent imports from VTK.

These are the modules within VTK requiring libGL that must be loaded
across pyvista's plotting API. These imports have the potential to
raise an ImportError if the user does not have libGL installed.

    ImportError: libGL.so.1: cannot open shared object file: No such file or directory

"""

from __future__ import annotations

import contextlib

try:
    # Necessary for displaying charts, otherwise crashes on rendering
    from vtkmodules import vtkRenderingContextOpenGL2 as vtkRenderingContextOpenGL2
except ImportError:  # pragma: no cover
    vtkRenderingContextOpenGL2 = None  # type: ignore[assignment] # noqa: N816


from vtkmodules.vtkRenderingOpenGL2 import vtkCameraPass as vtkCameraPass

with contextlib.suppress(ImportError):
    from vtkmodules.vtkRenderingOpenGL2 import (  # type: ignore[attr-defined]
        vtkCompositePolyDataMapper2 as vtkCompositePolyDataMapper2,
    )
from vtkmodules.vtkRenderingOpenGL2 import vtkDepthOfFieldPass as vtkDepthOfFieldPass
from vtkmodules.vtkRenderingOpenGL2 import vtkEDLShading as vtkEDLShading
from vtkmodules.vtkRenderingOpenGL2 import vtkGaussianBlurPass as vtkGaussianBlurPass
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLFXAAPass as vtkOpenGLFXAAPass
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLHardwareSelector as vtkOpenGLHardwareSelector
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLRenderer as vtkOpenGLRenderer
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLSkybox as vtkOpenGLSkybox
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLTexture as vtkOpenGLTexture
from vtkmodules.vtkRenderingOpenGL2 import vtkRenderPassCollection as vtkRenderPassCollection
from vtkmodules.vtkRenderingOpenGL2 import vtkRenderStepsPass as vtkRenderStepsPass
from vtkmodules.vtkRenderingOpenGL2 import vtkSequencePass as vtkSequencePass
from vtkmodules.vtkRenderingOpenGL2 import vtkShadowMapPass as vtkShadowMapPass
from vtkmodules.vtkRenderingOpenGL2 import vtkSSAAPass as vtkSSAAPass
from vtkmodules.vtkRenderingOpenGL2 import vtkSSAOPass as vtkSSAOPass
from vtkmodules.vtkRenderingVolumeOpenGL2 import (
    vtkOpenGLGPUVolumeRayCastMapper as vtkOpenGLGPUVolumeRayCastMapper,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper as vtkSmartVolumeMapper
