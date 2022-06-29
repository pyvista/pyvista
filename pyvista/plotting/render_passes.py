from pyvista import _vtk

# ordered
PRE_PASS = [
    'vtkEDLShading',
]

# ordered
POST_PASS = [
    'vtkDepthOfFieldPass',
    'vtkGaussianBlurPass',
    'vtkOpenGLFXAAPass',
    'vtkSSAAPass',  # should really be last
]


class RenderPasses:
    """Class to support multiple render passes for a renderer.

    Notes
    -----
    Passes are organized here as "Primary" (vtkOpenGLRenderPass) that act
    within the renderer and "post-processing" (vtkImageProcessingPass) passes,
    which act on the image generated from the renderer.

    The primary passes are added as part of a vtk.vtkRenderPassCollection or
    are "stacked", while the post-processing passes are added as a final pass
    to the rendered image.

    """

    def __init__(self, renderer):

        self._renderer = renderer

        self._pass_collection = _vtk.vtkRenderPassCollection()
        self._pass_collection.AddItem(_vtk.vtkLightsPass())
        self._pass_collection.AddItem(_vtk.vtkDefaultPass())

        self._seq_pass = _vtk.vtkSequencePass()
        self._seq_pass.SetPasses(self._pass_collection)

        # Tell the renderer to use our render pass pipeline
        camera_pass = _vtk.vtkCameraPass()
        camera_pass.SetDelegatePass(self._seq_pass)

        self._base_pass = camera_pass
        self._renderer.SetPass(self._base_pass)

        self._passes = {}
        self._shadow_map_pass = None
        self._edl_pass = None
        self._dof_pass = None
        self._ssaa_pass = None
        self._blur_passes = []

    def deep_clean(self):
        """Delete all render passes."""
        if self._renderer is not None:
            self._renderer.SetPass(None)
        self._renderer = None
        if self._seq_pass is not None:
            self._seq_pass.SetPasses(None)
        self._seq_pass = None
        self._pass_collection = None
        self._base_pass = None
        self._passes = {}
        self._shadow_map_pass = None
        self._edl_pass = None
        self._dof_pass = None
        self._ssaa_pass = None
        self._blur_passes = []

    def enable_edl_pass(self):
        if self._edl_pass is not None:
            return
        self._edl_pass = _vtk.vtkEDLShading()
        self.add_pass(self._edl_pass)
        return self._edl_pass

    def disable_edl_pass(self):
        if self._edl_pass is None:
            return
        self.remove_pass(self._edl_pass)
        self._edl_pass = None

    def add_blur_pass(self):
        """Add a vtkGaussianBlurPass pass.

        This is an vtkImageProcessingPass and delegates to the last pass.

        """
        blur_pass = _vtk.vtkGaussianBlurPass()
        self.add_pass(blur_pass)
        self._blur_passes.append(blur_pass)
        return blur_pass

    def remove_blur_pass(self):
        """Add a vtkGaussianBlurPass pass."""
        if self._blur_passes:
            self.remove_pass(self._blur_passes.pop(0))

    def enable_shadow_pass(self):
        """Enable shadow pass."""
        # shadow pass can be directly added to the base pass collection
        if self._shadow_map_pass is not None:
            return
        self._shadow_map_pass = _vtk.vtkShadowMapPass()
        self._pass_collection.AddItem(self._shadow_map_pass.GetShadowMapBakerPass())
        self._pass_collection.AddItem(self._shadow_map_pass)

    def disable_shadow_pass(self):
        """Disable shadow pass."""
        if self._shadow_map_pass is None:
            return
        self._pass_collection.RemoveItem(self._shadow_map_pass.GetShadowMapBakerPass())
        self._pass_collection.RemoveItem(self._shadow_map_pass)

    def enable_depth_of_field_pass(self, automatic_focal_distance=True):
        if self._dof_pass is not None:
            return
        self._dof_pass = _vtk.vtkDepthOfFieldPass()
        self._dof_pass.SetAutomaticFocalDistance(automatic_focal_distance)
        self.add_pass(self._dof_pass)
        return self._dof_pass

    def disable_depth_of_field_pass(self):
        if self._dof_pass is None:
            return
        self.remove_pass(self._dof_pass)
        self._dof_pass = None

    def enable_ssaa_pass(self):
        """Enable screen space anti-aliasing pass."""
        if self._ssaa_pass is not None:
            return
        self._ssaa_pass = _vtk.vtkSSAAPass()
        self.add_pass(self._ssaa_pass)

    def disable_ssaa_pass(self):
        """Disable screen space anti-aliasing pass."""
        if self._ssaa_pass is None:
            return
        self.remove_pass(self._ssaa_pass)
        self._ssaa_pass = None

    def _update_passes(self):
        """Reassemble pass delegation."""
        current_pass = self._base_pass
        for class_name in PRE_PASS + POST_PASS:
            if class_name in self._passes:
                for render_pass in self._passes[class_name]:
                    render_pass.SetDelegatePass(current_pass)
                    current_pass = render_pass

        self._renderer.SetPass(current_pass)

    @staticmethod
    def _class_name_from_vtk_obj(obj):
        """Return the class name from a vtk object."""
        return str(type(obj)).split('.')[-1].split("'")[0]

    def add_pass(self, render_pass):
        """Add a render pass."""
        class_name = RenderPasses._class_name_from_vtk_obj(render_pass)

        if class_name in PRE_PASS and render_pass in self._passes:
            return

        if class_name not in self._passes:
            self._passes[class_name] = [render_pass]
        else:
            self._passes[class_name].append(render_pass)

        self._update_passes()

    def remove_pass(self, render_pass):
        """Remove a pass.

        Remove a pass and reassembles the pass ordering

        """
        class_name = RenderPasses._class_name_from_vtk_obj(render_pass)

        if class_name not in self._passes:
            return
        else:
            self._passes[class_name].remove(render_pass)
            if not self._passes[class_name]:
                self._passes.pop(class_name)

        self._update_passes()
