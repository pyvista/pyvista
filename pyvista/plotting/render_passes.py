"""Render passes module for PyVista."""

from __future__ import annotations

import itertools
import weakref

from pyvista import _vtk
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core.utilities.misc import _NoNewAttrMixin

# The order of both the pre and post-passes matters.
PRE_PASS = [
    # SSAO needs geometry positions and normals before image filtering.
    'vtkSSAOPass',
    'vtkEDLShading',
]

POST_PASS = [
    'vtkDepthOfFieldPass',
    'vtkGaussianBlurPass',
    'vtkOpenGLFXAAPass',
    'vtkSSAAPass',
    'vtkToneMappingPass',  # requires the renderer's final viewport size
]


class RenderPasses(_NoNewAttrMixin):
    """Class to support multiple render passes for a renderer.

    Parameters
    ----------
    renderer : :vtk:`vtkRenderer`
        Renderer to initialize render passes for.

    Notes
    -----
    Passes are organized here as "primary" (:vtk:`vtkOpenGLRenderPass`) that act
    within the renderer and "post-processing" (:vtk:`vtkImageProcessingPass`) passes,
    which act on the image generated from the renderer.

    The primary passes are added as part of a :vtk:`vtkRenderPassCollection` or
    are "stacked", while the post-processing passes are added as a final pass
    to the rendered image.

    .. versionchanged:: 0.50
        Preserve an externally installed base pipeline when composing or removing
        managed effects. Image passes are ordered around its geometry stages;
        tone mapping follows SSAA. Removing the last managed effect
        restores the external pipeline and its original delegates.

    External pipelines must not contain managed passes or cycles. External pass
    objects remain owned by their caller. Native depth peeling and shadows are
    composed into the translucent and opaque stages of a
    :vtk:`vtkRenderStepsPass` base, respectively.

    """

    def __init__(self, renderer):
        """Initialize render passes."""
        self._renderer_ref = weakref.ref(renderer)
        self._closed = False

        self._passes = {}
        self._base_pass = None
        self._base_passes = []
        self._base_links_installed = []
        self._managed_active = False
        self._installed_pass = None
        self._render_steps_state = None
        self._render_steps_installed = None
        self._render_steps_camera = None
        self._depth_peeling_pass = None
        self._shadow_sequence = None
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
        return None  # type: ignore[unreachable]

    def _check_closed(self):
        """Raise if the renderer has already been closed."""
        if self._closed:
            msg = 'The renderer has been closed.'
            raise RuntimeError(msg)

    def close(self):
        """Delete all render passes and mark them permanently unusable.

        Unlike plain ``deep_clean()``, this is only called once the owning
        renderer itself is closed, so it also latches ``_closed`` -- any
        further attempt to enable/disable a pass then raises instead of
        silently no-op'ing.
        """
        self._closed = True
        self.deep_clean()

    def deep_clean(self):
        """Delete all render passes."""
        self._restore_render_steps()
        self._release_depth_peeling()
        for render_pass in (
            *itertools.chain.from_iterable(self._passes.values()),
            self._fxaa_pass,
            self._shadow_map_pass,
            self.__camera_pass,
        ):
            if render_pass is not None:
                if hasattr(render_pass, 'SetDelegatePass'):
                    render_pass.SetDelegatePass(None)
                if render_pass is self._shadow_map_pass:
                    render_pass.SetOpaqueSequence(None)
                    render_pass.GetShadowMapBakerPass().SetOpaqueSequence(None)
                self._release_graphics_resources(render_pass)
        self._restore_base_passes()
        if self._renderer is not None and self._renderer.GetPass() is self._installed_pass:
            self._renderer.SetPass(self._base_pass)
        self._base_pass = None
        self._base_passes = []
        self._base_links_installed = []
        self._managed_active = False
        self._installed_pass = None
        self._shadow_sequence = None
        self._fxaa_pass = None
        self._renderer_ref = None  # type: ignore[assignment]
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
        """Enable the EDL pass.

        Returns
        -------
        :vtk:`vtkEDLShading`
            The enabled EDL pass.

        """
        self._check_closed()
        self._update_base_pass()
        if self._edl_pass is not None:
            self._update_passes()
            return None
        self._edl_pass = _vtk.vtkEDLShading()
        self._add_pass(self._edl_pass)
        return self._edl_pass

    def disable_edl_pass(self):
        """Disable the EDL pass."""
        self._check_closed()
        if self._edl_pass is None:
            return
        self._remove_pass(self._edl_pass)
        self._edl_pass = None

    def add_blur_pass(self):
        """Add a :vtk:`vtkGaussianBlurPass` pass.

        This is a :vtk:`vtkImageProcessingPass` and delegates to the last pass.

        Returns
        -------
        :vtk:`vtkGaussianBlurPass`
            The added Gaussian blur pass.

        """
        self._check_closed()
        self._update_base_pass()
        blur_pass = _vtk.vtkGaussianBlurPass()
        self._add_pass(blur_pass)
        self._blur_passes.append(blur_pass)
        return blur_pass

    def remove_blur_pass(self):
        """Remove a single :vtk:`vtkGaussianBlurPass` pass."""
        self._check_closed()
        if self._blur_passes:
            # order of the blur passes does not matter
            self._remove_pass(self._blur_passes.pop())

    def enable_shadow_pass(self):
        """Enable shadow pass.

        Returns
        -------
        :vtk:`vtkShadowMapPass`
            The enabled shadow pass.

        """
        self._check_closed()
        self._update_base_pass()
        self._check_compatible('vtkShadowMapPass')
        if self._shadow_map_pass is not None:
            self._update_passes()
            return None
        if self._dof_pass is not None:
            msg = 'Shadows are incompatible with the depth of field pass.'
            raise RuntimeError(msg)
        if not self._pass_collection.GetItemAsObject(0).IsA('vtkRenderStepsPass'):
            msg = 'Shadows require a vtkRenderStepsPass geometry pipeline.'
            raise RuntimeError(msg)
        self._shadow_map_pass = _vtk.vtkShadowMapPass()
        self._update_passes()
        return self._shadow_map_pass

    def disable_shadow_pass(self):
        """Disable shadow pass."""
        self._check_closed()
        if self._shadow_map_pass is None:
            return
        shadow_pass = self._shadow_map_pass
        self._shadow_map_pass = None
        self._update_passes()
        shadow_pass.SetOpaqueSequence(None)
        shadow_pass.GetShadowMapBakerPass().SetOpaqueSequence(None)
        self._release_graphics_resources(shadow_pass)
        self._shadow_sequence = None

    @_deprecate_positional_args
    def enable_depth_of_field_pass(self, automatic_focal_distance: bool = True):  # noqa: FBT001, FBT002
        """Enable the depth of field pass.

        Parameters
        ----------
        automatic_focal_distance : bool, default: True
            If ``True``, the depth of field effect will automatically compute
            the focal distance. If ``False``, the user must specify the distance.

        Returns
        -------
        :vtk:`vtkDepthOfFieldPass`
            The enabled depth of field pass.

        """
        self._check_closed()
        self._update_base_pass()
        self._check_compatible('vtkDepthOfFieldPass')
        if self._dof_pass is not None:
            self._update_passes()
            return None
        if self._shadow_map_pass is not None:
            msg = 'Depth of field is incompatible with shadows.'
            raise RuntimeError(msg)

        if self._ssao_pass is not None:
            msg = 'Depth of field pass is incompatible with the SSAO pass.'
            raise RuntimeError(msg)

        self._dof_pass = _vtk.vtkDepthOfFieldPass()
        self._dof_pass.SetAutomaticFocalDistance(automatic_focal_distance)
        self._add_pass(self._dof_pass)
        return self._dof_pass

    def disable_depth_of_field_pass(self):
        """Disable the depth of field pass."""
        self._check_closed()
        if self._dof_pass is None:
            return
        self._remove_pass(self._dof_pass)
        self._dof_pass = None

    @_deprecate_positional_args
    def enable_ssao_pass(  # noqa: PLR0917
        self, radius, bias, kernel_size, blur
    ):
        """Enable the screen space ambient occlusion pass.

        Parameters
        ----------
        radius : float
            Radius of occlusion generation.
        bias : float
            Bias to adjust the occlusion generation.
        kernel_size : int
            Size of the kernel for occlusion generation.
        blur : bool
            If ``True``, the pass uses a blur stage.

        Returns
        -------
        :vtk:`vtkSSAOPass`
            The enabled screen space ambient occlusion pass.

        """
        self._check_closed()
        self._update_base_pass()
        if self._dof_pass is not None:
            msg = 'SSAO pass is incompatible with the depth of field pass.'
            raise RuntimeError(msg)

        self._check_compatible('vtkSSAOPass')
        new_pass = self._ssao_pass is None
        if new_pass:
            self._ssao_pass = _vtk.vtkSSAOPass()
        self._ssao_pass.SetRadius(radius)
        self._ssao_pass.SetBias(bias)
        self._ssao_pass.SetKernelSize(kernel_size)
        self._ssao_pass.SetBlur(blur)
        if new_pass:
            self._add_pass(self._ssao_pass)
        else:
            self._update_passes()
        return self._ssao_pass

    def disable_ssao_pass(self):
        """Disable the screen space ambient occlusion pass."""
        self._check_closed()
        if self._ssao_pass is None:
            return
        self._remove_pass(self._ssao_pass)
        self._ssao_pass = None

    def enable_ssaa_pass(self):
        """Enable super-sample anti-aliasing pass.

        Returns
        -------
        :vtk:`vtkSSAAPass`
            The enabled super-sample anti-aliasing pass.

        """
        self._check_closed()
        self._update_base_pass()
        if self._ssaa_pass is not None:
            self._update_passes()
            return None
        self._ssaa_pass = _vtk.vtkSSAAPass()
        self._add_pass(self._ssaa_pass)
        return self._ssaa_pass

    def disable_ssaa_pass(self):
        """Disable super-sample anti-aliasing pass."""
        self._check_closed()
        if self._ssaa_pass is None:
            return
        self._remove_pass(self._ssaa_pass)
        self._ssaa_pass = None

    def _check_compatible(self, candidate=None):
        """Reject incompatible depth-buffer consumers before changing the pipeline."""
        external = self._validate_base_pass(self._base_pass, self._base_overrides())
        names = {render_pass.GetClassName() for render_pass in external}
        names.update(self._passes)
        names.add(candidate)
        if self._shadow_map_pass is not None:
            names.add('vtkShadowMapPass')
        if self._renderer.GetUseDepthPeeling():
            names.add('vtkDualDepthPeelingPass')
        if 'vtkDepthOfFieldPass' in names:
            for name, label in (
                ('vtkSSAOPass', 'the SSAO pass'),
                ('vtkShadowMapPass', 'shadows'),
                ('vtkDualDepthPeelingPass', 'depth peeling'),
                ('vtkDepthPeelingPass', 'depth peeling'),
            ):
                if name in names:
                    msg = f'Depth of field pass is incompatible with {label}.'
                    raise RuntimeError(msg)

    def _restore_render_steps(self):
        """Restore opaque, translucent, and volume passes borrowed from the base."""
        if self._render_steps_state is not None:
            steps, *original = self._render_steps_state
            for name, before, installed in zip(
                ('OpaquePass', 'TranslucentPass', 'VolumetricPass'),
                original,
                self._render_steps_installed,
                strict=True,
            ):
                if getattr(steps, 'Get' + name)() is installed:
                    getattr(steps, 'Set' + name)(before)
            self._restore_camera_cache()
            self._render_steps_state = None
            self._render_steps_installed = None
            self._render_steps_camera = None

    def _restore_camera_cache(self):
        """Replace cached geometry stages when managed effects are removed."""
        if self._render_steps_installed is not None:
            steps = self._render_steps_state[0]
            # VTK also retains the previous frame's stages in the camera's
            # sequence. Restore only our substituted entries, so immediate
            # toggles do not see disabled effects in that cache.
            camera, sequence = self._render_steps_camera
            if (
                camera is not None
                and camera.GetDelegatePass() is sequence
                and sequence is not None
                and sequence.IsA('vtkSequencePass')
            ):
                collection = sequence.GetPasses()
                replacements = dict(
                    zip(
                        self._render_steps_installed,
                        (
                            steps.GetOpaquePass(),
                            steps.GetTranslucentPass(),
                            steps.GetVolumetricPass(),
                        ),
                        strict=True,
                    )
                )
                if collection is not None:
                    for index in reversed(range(collection.GetNumberOfItems())):
                        item = collection.GetItemAsObject(index)
                        if item in replacements and replacements[item] is not item:
                            replacement = replacements[item]
                            if replacement is None:
                                collection.RemoveItem(index)
                            else:
                                collection.ReplaceItem(index, replacement)

    def _release_depth_peeling(self):
        """Release only the managed peeling pass, retaining its geometry delegates."""
        if self._depth_peeling_pass is not None:
            self._depth_peeling_pass.SetTranslucentPass(None)
            self._depth_peeling_pass.SetVolumetricPass(None)
            self._release_graphics_resources(self._depth_peeling_pass)
            self._depth_peeling_pass = None

    def _configure_render_steps(self):
        """Apply shadows before translucent geometry and honor native peeling settings."""
        steps = self._pass_collection.GetItemAsObject(0)
        if not steps.IsA('vtkRenderStepsPass'):
            return None
        if self._render_steps_state is None:
            camera = steps.GetCameraPass()
            self._render_steps_camera = (
                camera,
                None if camera is None else camera.GetDelegatePass(),
            )
            self._render_steps_state = (
                steps,
                steps.GetOpaquePass(),
                steps.GetTranslucentPass(),
                steps.GetVolumetricPass(),
            )
        else:
            overrides = self._base_overrides()
            original = tuple(
                overrides[steps, getter]
                for getter in ('GetOpaquePass', 'GetTranslucentPass', 'GetVolumetricPass')
            )
            if original[0] is not self._render_steps_state[1]:
                self._shadow_sequence = None
            self._render_steps_state = (steps, *original)
        _, opaque, translucent, volumetric = self._render_steps_state
        if self._shadow_map_pass is not None:
            if self._shadow_sequence is None:
                opaque_passes = _vtk.vtkRenderPassCollection()
                if steps.GetLightsPass() is not None:
                    opaque_passes.AddItem(steps.GetLightsPass())
                if opaque is not None:
                    opaque_passes.AddItem(opaque)
                opaque_sequence = _vtk.vtkSequencePass()
                opaque_sequence.SetPasses(opaque_passes)
                self._shadow_map_pass.SetOpaqueSequence(opaque_sequence)
                baker = self._shadow_map_pass.GetShadowMapBakerPass()
                light_camera = _vtk.vtkCameraPass()
                light_camera.SetDelegatePass(opaque_sequence)
                baker.SetOpaqueSequence(light_camera)
                shadow_passes = _vtk.vtkRenderPassCollection()
                shadow_passes.AddItem(baker)
                shadow_passes.AddItem(self._shadow_map_pass)
                self._shadow_sequence = _vtk.vtkSequencePass()
                self._shadow_sequence.SetPasses(shadow_passes)
            steps.SetOpaquePass(self._shadow_sequence)
        else:
            steps.SetOpaquePass(opaque)

        if (
            self._renderer.GetUseDepthPeeling()
            and translucent is not None
            and not (
                translucent.IsA('vtkDepthPeelingPass')
                or translucent.IsA('vtkDualDepthPeelingPass')
            )
        ):
            if self._depth_peeling_pass is None:
                self._depth_peeling_pass = _vtk.vtkDualDepthPeelingPass()
            peeling = self._depth_peeling_pass
            peeling.SetTranslucentPass(translucent)
            peeling.SetMaximumNumberOfPeels(self._renderer.GetMaximumNumberOfPeels())
            peeling.SetOcclusionRatio(self._renderer.GetOcclusionRatio())
            if self._renderer.GetUseDepthPeelingForVolumes():
                peeling.SetVolumetricPass(volumetric)
                steps.SetVolumetricPass(None)
            else:
                peeling.SetVolumetricPass(None)
                steps.SetVolumetricPass(volumetric)
            steps.SetTranslucentPass(peeling)
        else:
            steps.SetTranslucentPass(translucent)
            steps.SetVolumetricPass(volumetric)
            self._release_depth_peeling()
        # Peeling uses its own framebuffer attachments. SSAO's geometry buffers
        # must be completed before that translucent stage changes the bindings.
        staged_ssao = None
        translucent = steps.GetTranslucentPass()
        if (
            self._passes.get('vtkSSAOPass')
            and translucent is not None
            and (
                translucent.IsA('vtkDepthPeelingPass')
                or translucent.IsA('vtkDualDepthPeelingPass')
            )
        ):
            staged_ssao = self._passes['vtkSSAOPass'][0]
            if steps.GetOpaquePass() is not None and (
                self._render_steps_installed is None
                or self._render_steps_installed[0] is not staged_ssao
            ):
                # Geometry cached for the previous framebuffer must be rebuilt
                # for SSAO's multiple color attachments on this transition.
                self._release_graphics_resources(steps.GetOpaquePass())
            opaque_passes = _vtk.vtkRenderPassCollection()
            if steps.GetLightsPass() is not None:
                opaque_passes.AddItem(steps.GetLightsPass())
            if steps.GetOpaquePass() is not None:
                opaque_passes.AddItem(steps.GetOpaquePass())
            opaque_sequence = _vtk.vtkSequencePass()
            opaque_sequence.SetPasses(opaque_passes)
            opaque_camera = _vtk.vtkCameraPass()
            opaque_camera.SetDelegatePass(opaque_sequence)
            staged_ssao.SetDelegatePass(opaque_camera)
            steps.SetOpaquePass(staged_ssao)
        self._restore_camera_cache()
        self._render_steps_installed = (
            steps.GetOpaquePass(),
            steps.GetTranslucentPass(),
            steps.GetVolumetricPass(),
        )
        return staged_ssao

    def _base_overrides(self):
        """Read caller-owned links without the stages temporarily inserted by this manager."""
        overrides = {}
        expected = dict(self._base_links_installed)
        for render_pass, original in self._base_passes:
            actual = render_pass.GetDelegatePass()
            overrides[render_pass, 'GetDelegatePass'] = (
                original if actual is expected.get(render_pass, original) else actual
            )
        if self._render_steps_state is not None:
            steps, *original = self._render_steps_state
            getters = ('GetOpaquePass', 'GetTranslucentPass', 'GetVolumetricPass')
            for getter, before, installed in zip(
                getters, original, self._render_steps_installed, strict=True
            ):
                actual = getattr(steps, getter)()
                overrides[steps, getter] = before if actual is installed else actual
        return overrides

    def _validate_base_pass(self, root, overrides):
        """Reject cycles and references back into the manager's active pipeline."""
        managed = {
            *itertools.chain.from_iterable(self._passes.values()),
            self.__camera_pass,
            self.__seq_pass,
            self._fxaa_pass,
            self._shadow_map_pass,
            self._shadow_sequence,
            self._depth_peeling_pass,
        }
        visited = set()
        active = set()

        def visit(render_pass, *, camera_cache=False):
            """Validate each delegate and each pass in a composite pipeline.

            Parameters
            ----------
            render_pass : :vtk:`vtkRenderPass`
                Pass to visit.
            camera_cache : bool, default: False
                Whether the pass belongs to the cached render-steps sequence.

            """
            if render_pass is None:
                return
            # RenderStepsPass caches its geometry stages beneath its camera.
            # Those cached references are replaced by VTK at the next render.
            cached_ssao = (
                self._render_steps_installed is not None
                and self._render_steps_installed[0] is self._ssao_pass
                and render_pass is self._ssao_pass
            )
            if camera_cache and (
                cached_ssao or render_pass in (self._shadow_sequence, self._depth_peeling_pass)
            ):
                return
            if render_pass in active or render_pass in managed:
                msg = 'An external base pipeline cannot contain cycles or managed render passes.'
                raise ValueError(msg)
            if render_pass in visited:
                return
            active.add(render_pass)
            for getter in (
                'GetDelegatePass',
                'GetCameraPass',
                'GetOpaqueSequence',
                'GetShadowMapBakerPass',
                'GetLightsPass',
                'GetOpaquePass',
                'GetTranslucentPass',
                'GetVolumetricPass',
                'GetOverlayPass',
                'GetPostProcessPass',
            ):
                if hasattr(render_pass, getter):
                    child = overrides.get((render_pass, getter), getattr(render_pass, getter)())
                    cached_camera = (
                        getter == 'GetCameraPass'
                        and self._render_steps_camera is not None
                        and child is self._render_steps_camera[0]
                        and child is not None
                        and child.GetDelegatePass() is self._render_steps_camera[1]
                    )
                    visit(child, camera_cache=camera_cache or cached_camera)
            if hasattr(render_pass, 'GetPasses'):
                collection = render_pass.GetPasses()
                if collection is not None:
                    for index in range(collection.GetNumberOfItems()):
                        visit(collection.GetItemAsObject(index), camera_cache=camera_cache)
            active.remove(render_pass)
            visited.add(render_pass)

        visit(root)
        return visited

    def _restore_base_passes(self):
        """Restore borrowed links while retaining changes made by the caller."""
        for (render_pass, getter), delegate in self._base_overrides().items():
            if getter == 'GetDelegatePass':
                render_pass.SetDelegatePass(delegate)
        self._base_links_installed = []

    def _update_base_pass(self):
        """Capture the caller's pipeline, including edits beneath an unchanged root."""
        current = self._renderer.GetPass()
        links_changed = any(
            render_pass.GetDelegatePass() is not delegate
            for render_pass, delegate in self._base_links_installed
        )
        if self._managed_active and current is self._installed_pass and not links_changed:
            self._validate_base_pass(self._base_pass, self._base_overrides())
            return
        if not self._managed_active and current is None and self._base_pass is None:
            return
        if self._managed_active and current is self._installed_pass:
            current = self._base_pass
        overrides = self._base_overrides()
        self._validate_base_pass(current, overrides)
        links = []
        delegate = current
        while delegate is not None and (
            delegate.IsA('vtkImageProcessingPass') or delegate.IsA('vtkSSAAPass')
        ):
            next_pass = overrides.get((delegate, 'GetDelegatePass'), delegate.GetDelegatePass())
            links.append((delegate, next_pass))
            delegate = next_pass
        self._restore_render_steps()
        self._restore_base_passes()
        self._base_pass = current
        self._base_passes = links
        self._shadow_sequence = None
        self._pass_collection.ReplaceItem(0, delegate or _vtk.vtkRenderStepsPass())

    def _update_passes(self):
        """Reassemble geometry and image passes while retaining an external base pipeline."""
        self._check_closed()
        self._update_base_pass()
        self._check_compatible()
        passes = {name: list(items) for name, items in self._passes.items()}
        for render_pass, _ in reversed(self._base_passes):
            passes.setdefault(render_pass.GetClassName(), []).append(render_pass)

        # Native FXAA is bypassed when VTK renders through a custom pass.
        if (
            not passes.get('vtkOpenGLFXAAPass')
            and self._renderer.GetUseFXAA()
            and (self._passes or self._shadow_map_pass is not None or self._base_pass is not None)
        ):
            if self._fxaa_pass is None:
                self._fxaa_pass = _vtk.vtkOpenGLFXAAPass()
            self._fxaa_pass.SetFXAAOptions(self._renderer.GetFXAAOptions())
            passes.setdefault('vtkOpenGLFXAAPass', []).append(self._fxaa_pass)
        elif self._fxaa_pass is not None:
            self._fxaa_pass.SetDelegatePass(None)
            self._release_graphics_resources(self._fxaa_pass)
            self._fxaa_pass = None

        native_peeling = self._base_pass is not None and self._renderer.GetUseDepthPeeling()
        if (
            not self._passes
            and self._shadow_map_pass is None
            and self._fxaa_pass is None
            and not native_peeling
        ):
            self._restore_render_steps()
            self._release_depth_peeling()
            self._restore_base_passes()
            self._renderer.SetPass(self._base_pass)
            self._installed_pass = self._base_pass
            self._managed_active = False
            return

        staged_ssao = self._configure_render_steps()
        current_pass = (
            self._pass_collection.GetItemAsObject(0)
            if staged_ssao is not None
            else self._camera_pass
        )
        # Unknown external image passes retain their order outside managed effects.
        order = (
            PRE_PASS + POST_PASS + [name for name in passes if name not in PRE_PASS + POST_PASS]
        )
        for class_name in order:
            for render_pass in passes.get(class_name, []):
                if render_pass is staged_ssao:
                    continue
                render_pass.SetDelegatePass(current_pass)
                current_pass = render_pass
        self._renderer.SetPass(current_pass)
        self._installed_pass = current_pass
        self._managed_active = True
        self._base_links_installed = [
            (render_pass, render_pass.GetDelegatePass()) for render_pass, _ in self._base_passes
        ]

    def _add_pass(self, render_pass):
        """Add a render pass."""
        class_name = render_pass.GetClassName()

        if render_pass in self._passes.get(class_name, []):
            return

        if class_name not in self._passes:
            self._passes[class_name] = [render_pass]
        else:
            self._passes[class_name].append(render_pass)

        self._update_passes()

    def _release_graphics_resources(self, render_pass):
        """Free the GPU resources a pass holds before it is dropped."""
        renderer = self._renderer
        ren_win = None if renderer is None else renderer.GetRenderWindow()
        if ren_win is not None:
            ren_win.MakeCurrent()
            render_pass.ReleaseGraphicsResources(ren_win)

    def _remove_pass(self, render_pass):
        """Remove a pass.

        Remove a pass and reassemble the pass ordering.

        """
        class_name = render_pass.GetClassName()

        if class_name not in self._passes:  # pragma: no cover
            return
        else:
            render_pass.SetDelegatePass(None)
            self._release_graphics_resources(render_pass)
            self._passes[class_name].remove(render_pass)
            if not self._passes[class_name]:
                self._passes.pop(class_name)

        self._update_passes()
