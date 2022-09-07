"""Render passes module for PyVista."""
import weakref

from pyvista import _vtk

from ..utilities.misc import vtk_version_info

# The order of both the pre and post-passes matters.
PRE_PASS = [
    'vtkEDLShading',
]

POST_PASS = [
    'vtkDepthOfFieldPass',
    'vtkGaussianBlurPass',
    'vtkOpenGLFXAAPass',
    'vtkSSAOPass',
    'vtkSSAAPass',  # should be last
]


class RenderPasses:
    """Class to support multiple render passes for a renderer.

    Notes
    -----
    Passes are organized here as "primary" (vtkOpenGLRenderPass) that act
    within the renderer and "post-processing" (vtkImageProcessingPass) passes,
    which act on the image generated from the renderer.

    The primary passes are added as part of a vtk.vtkRenderPassCollection or
    are "stacked", while the post-processing passes are added as a final pass
    to the rendered image.

    """

    def __init__(self, renderer):
        """Initialize render passes."""
        self._renderer_ref = weakref.ref(renderer)

        self._passes = {}
        self._fxaa_pass = None
        self._shadow_map_pass = None
        self._edl_pass = None
        self._dof_pass = None
        self._ssaa_pass = None
        self._ssao_pass = None
        self._blur_passes = []
        self.__pass_collection = None
        self.__seq_pass = None
        self.__camera_pass = None

    @property
    def _pass_collection(self):
        """Initialize (when necessary) the pass collection and return it.

        This lets us lazily generate the pass collection only when we need it
        rather than at initialization of the class.

        """
        if self.__pass_collection is None:
            self._init_passes()
        return self.__pass_collection

    @property
    def _seq_pass(self):
        """Initialize (when necessary) the sequence collection and return it.

        This lets us lazily generate the sequence collection only when we need it
        rather than at initialization of the class.

        """
        if self.__seq_pass is None:
            self._init_passes()
        return self.__seq_pass

    @property
    def _camera_pass(self):
        """Initialize (when necessary) the camera pass and return it.

        This lets us lazily generate the camera pass only when we need it
        rather than at initialization of the class.

        """
        if self.__camera_pass is None:
            self._init_passes()
        return self.__camera_pass

    def _init_passes(self):
        """Initialize the renderer's standard passes."""
        # simulate the standard VTK rendering passes and put them in a sequence
        self.__pass_collection = _vtk.vtkRenderPassCollection()
        self.__pass_collection.AddItem(_vtk.vtkRenderStepsPass())

        self.__seq_pass = _vtk.vtkSequencePass()
        self.__seq_pass.SetPasses(self._pass_collection)

        # Make the sequence the delegate of a camera pass.
        self.__camera_pass = _vtk.vtkCameraPass()
        self.__camera_pass.SetDelegatePass(self._seq_pass)

    @property
    def _renderer(self):
        """Return the renderer."""
        if self._renderer_ref is not None:
            return self._renderer_ref()

    def deep_clean(self):
        """Delete all render passes."""
        if self._renderer is not None:
            self._renderer.SetPass(None)
        self._renderer_ref = None
        if self.__seq_pass is not None:
            self.__seq_pass.SetPasses(None)
        self.__seq_pass = None
        self.__pass_collection = None
        self.__camera_pass = None
        self._passes = {}
        self._shadow_map_pass = None
        self._edl_pass = None
        self._dof_pass = None
        self._ssaa_pass = None
        self._ssao_pass = None
        self._blur_passes = []

    def enable_edl_pass(self):
        """Enable the EDL pass."""
        if self._edl_pass is not None:
            return
        self._edl_pass = _vtk.vtkEDLShading()
        self._add_pass(self._edl_pass)
        return self._edl_pass

    def disable_edl_pass(self):
        """Disable the EDL pass."""
        if self._edl_pass is None:
            return
        self._remove_pass(self._edl_pass)
        self._edl_pass = None

    def add_blur_pass(self):
        """Add a vtkGaussianBlurPass pass.

        This is a vtkImageProcessingPass and delegates to the last pass.

        """
        blur_pass = _vtk.vtkGaussianBlurPass()
        self._add_pass(blur_pass)
        self._blur_passes.append(blur_pass)
        return blur_pass

    def remove_blur_pass(self):
        """Remove a single vtkGaussianBlurPass pass."""
        if self._blur_passes:
            # order of the blur passes does not matter
            self._remove_pass(self._blur_passes.pop())

    def enable_shadow_pass(self):
        """Enable shadow pass."""
        # shadow pass can be directly added to the base pass collection
        if self._shadow_map_pass is not None:
            return
        self._shadow_map_pass = _vtk.vtkShadowMapPass()
        self._pass_collection.AddItem(self._shadow_map_pass.GetShadowMapBakerPass())
        self._pass_collection.AddItem(self._shadow_map_pass)
        self._update_passes()
        return self._shadow_map_pass

    def disable_shadow_pass(self):
        """Disable shadow pass."""
        if self._shadow_map_pass is None:
            return
        self._pass_collection.RemoveItem(self._shadow_map_pass.GetShadowMapBakerPass())
        self._pass_collection.RemoveItem(self._shadow_map_pass)
        self._update_passes()

    def enable_depth_of_field_pass(self, automatic_focal_distance=True):
        """Enable the depth of field pass."""
        if self._dof_pass is not None:
            return

        if self._ssao_pass is not None:
            raise RuntimeError('Depth of field pass is incompatible with the SSAO pass.')

        self._dof_pass = _vtk.vtkDepthOfFieldPass()
        self._dof_pass.SetAutomaticFocalDistance(automatic_focal_distance)
        self._add_pass(self._dof_pass)
        return self._dof_pass

    def disable_depth_of_field_pass(self):
        """Disable the depth of field pass."""
        if self._dof_pass is None:
            return
        self._remove_pass(self._dof_pass)
        self._dof_pass = None

    def enable_ssao_pass(self, radius, bias, kernel_size, blur):
        """Enable the screen space ambient occlusion pass."""
        if vtk_version_info < (9,):  # pragma: no cover
            from pyvista.core.errors import VTKVersionError  # avoid circular import

            raise VTKVersionError('SSAO pass requires VTK 9 or newer.')

        if self._dof_pass is not None:
            raise RuntimeError('SSAO pass is incompatible with the depth of field pass.')

        if self._ssao_pass is not None:
            return
        self._ssao_pass = _vtk.vtkSSAOPass()
        self._ssao_pass.SetRadius(radius)
        self._ssao_pass.SetBias(bias)
        self._ssao_pass.SetKernelSize(kernel_size)
        self._ssao_pass.SetBlur(blur)
        self._add_pass(self._ssao_pass)
        return self._ssao_pass

    def disable_ssao_pass(self):
        """Disable the screen space ambient occlusion pass."""
        if self._ssao_pass is None:
            return
        self._remove_pass(self._ssao_pass)
        self._ssao_pass = None

    def enable_ssaa_pass(self):
        """Enable super-sample anti-aliasing pass."""
        if self._ssaa_pass is not None:
            return
        self._ssaa_pass = _vtk.vtkSSAAPass()
        self._add_pass(self._ssaa_pass)
        return self._ssaa_pass

    def disable_ssaa_pass(self):
        """Disable super-sample anti-aliasing pass."""
        if self._ssaa_pass is None:
            return
        self._remove_pass(self._ssaa_pass)
        self._ssaa_pass = None

    def _update_passes(self):
        """Reassemble pass delegation."""
        if self._renderer is None:  # pragma: no cover
            raise RuntimeError('The renderer has been closed.')

        current_pass = self._camera_pass
        for class_name in PRE_PASS + POST_PASS:
            if class_name in self._passes:
                for render_pass in self._passes[class_name]:
                    render_pass.SetDelegatePass(current_pass)
                    current_pass = render_pass

        # reset to the default rendering if no special passes have been added
        if current_pass is self._camera_pass and self._shadow_map_pass is None:
            self._renderer.SetPass(None)
        else:
            self._renderer.SetPass(current_pass)

    def _add_pass(self, render_pass):
        """Add a render pass."""
        class_name = render_pass.GetClassName()

        if class_name in PRE_PASS and render_pass in self._passes:
            return

        if class_name not in self._passes:
            self._passes[class_name] = [render_pass]
        else:
            self._passes[class_name].append(render_pass)

        self._update_passes()

    def _remove_pass(self, render_pass):
        """Remove a pass.

        Remove a pass and reassemble the pass ordering.

        """
        class_name = render_pass.GetClassName()

        if class_name not in self._passes:  # pragma: no cover
            return
        else:
            self._passes[class_name].remove(render_pass)
            if not self._passes[class_name]:
                self._passes.pop(class_name)

        self._update_passes()
