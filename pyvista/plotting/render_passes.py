"""Render passes module for PyVista."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING
from typing import cast
import weakref

from pyvista import _vtk
from pyvista.core.utilities.misc import _NoNewAttrMixin

if TYPE_CHECKING:
    from typing import TypeAlias

    # Every pass added to a renderer delegates to the one before it
    _DelegatingPass: TypeAlias = '_vtk.vtkImageProcessingPass | _vtk.vtkSSAAPass'

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


class RenderPasses(_NoNewAttrMixin):
    """Class to support multiple render passes for a renderer.

    Notes
    -----
    Passes are organized here as "primary" (:vtk:`vtkOpenGLRenderPass`) that act
    within the renderer and "post-processing" (:vtk:`vtkImageProcessingPass`) passes,
    which act on the image generated from the renderer.

    The primary passes are added as part of a :vtk:`vtkRenderPassCollection` or
    are "stacked", while the post-processing passes are added as a final pass
    to the rendered image.

    Parameters
    ----------
    renderer : :vtk:`vtkRenderer`
        Renderer to initialize render passes for.

    """

    def __init__(self, renderer: _vtk.vtkRenderer) -> None:
        """Initialize render passes."""
        self._renderer_ref = weakref.ref(renderer)
        self._closed = False

        self._passes: dict[str, list[_DelegatingPass]] = {}
        self._fxaa_pass: _vtk.vtkOpenGLFXAAPass | None = None
        self._shadow_map_pass: _vtk.vtkShadowMapPass | None = None
        self._edl_pass: _vtk.vtkEDLShading | None = None
        self._dof_pass: _vtk.vtkDepthOfFieldPass | None = None
        self._ssaa_pass: _vtk.vtkSSAAPass | None = None
        self._ssao_pass: _vtk.vtkSSAOPass | None = None
        self._blur_passes: list[_vtk.vtkGaussianBlurPass] = []
        self.__pass_collection: _vtk.vtkRenderPassCollection | None = None
        self.__seq_pass: _vtk.vtkSequencePass | None = None
        self.__camera_pass: _vtk.vtkCameraPass | None = None

    @property
    def _pass_collection(self) -> _vtk.vtkRenderPassCollection:
        """Initialize (when necessary) the pass collection and return it.

        This lets us lazily generate the pass collection only when we need it
        rather than at initialization of the class.

        """
        if self.__pass_collection is None:
            self._init_passes()
        return cast('_vtk.vtkRenderPassCollection', self.__pass_collection)

    @property
    def _seq_pass(self) -> _vtk.vtkSequencePass:
        """Initialize (when necessary) the sequence collection and return it.

        This lets us lazily generate the sequence collection only when we need it
        rather than at initialization of the class.

        """
        if self.__seq_pass is None:
            self._init_passes()
        return cast('_vtk.vtkSequencePass', self.__seq_pass)

    @property
    def _camera_pass(self) -> _vtk.vtkCameraPass:
        """Initialize (when necessary) the camera pass and return it.

        This lets us lazily generate the camera pass only when we need it
        rather than at initialization of the class.

        """
        if self.__camera_pass is None:
            self._init_passes()
        return cast('_vtk.vtkCameraPass', self.__camera_pass)

    def _init_passes(self) -> None:
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
    def _renderer(self) -> _vtk.vtkRenderer | None:
        """Return the renderer."""
        if self._renderer_ref is not None:
            return self._renderer_ref()
        return None  # type: ignore[unreachable]

    def _check_closed(self) -> None:
        """Raise if the renderer has already been closed."""
        if self._closed:
            msg = 'The renderer has been closed.'
            raise RuntimeError(msg)

    def close(self) -> None:
        """Delete all render passes and mark them permanently unusable.

        Unlike plain ``deep_clean()``, this is only called once the owning
        renderer itself is closed, so it also latches ``_closed`` -- any
        further attempt to enable/disable a pass then raises instead of
        silently no-op'ing.
        """
        self._closed = True
        self.deep_clean()

    def deep_clean(self) -> None:
        """Delete all render passes."""
        for render_pass in (
            *itertools.chain.from_iterable(self._passes.values()),
            self._shadow_map_pass,
            self.__camera_pass,
        ):
            if render_pass is not None:
                self._release_graphics_resources(render_pass)
        if self._renderer is not None:
            self._renderer.SetPass(None)  # type: ignore[arg-type]
        self._renderer_ref = None  # type: ignore[assignment]
        if self.__seq_pass is not None:
            self.__seq_pass.SetPasses(None)  # type: ignore[arg-type]
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

    def enable_edl_pass(self) -> _vtk.vtkEDLShading | None:
        """Enable the EDL pass.

        Returns
        -------
        :vtk:`vtkEDLShading`
            The enabled EDL pass.

        """
        if self._edl_pass is not None:
            return None
        self._edl_pass = _vtk.vtkEDLShading()
        self._add_pass(self._edl_pass)
        return self._edl_pass

    def disable_edl_pass(self) -> None:
        """Disable the EDL pass."""
        self._check_closed()
        if self._edl_pass is None:
            return
        self._remove_pass(self._edl_pass)
        self._edl_pass = None

    def add_blur_pass(self) -> _vtk.vtkGaussianBlurPass:
        """Add a :vtk:`vtkGaussianBlurPass` pass.

        This is a :vtk:`vtkImageProcessingPass` and delegates to the last pass.

        Returns
        -------
        :vtk:`vtkGaussianBlurPass`
            The added Gaussian blur pass.

        """
        blur_pass = _vtk.vtkGaussianBlurPass()
        self._add_pass(blur_pass)
        self._blur_passes.append(blur_pass)
        return blur_pass

    def remove_blur_pass(self) -> None:
        """Remove a single :vtk:`vtkGaussianBlurPass` pass."""
        self._check_closed()
        if self._blur_passes:
            # order of the blur passes does not matter
            self._remove_pass(self._blur_passes.pop())

    def enable_shadow_pass(self) -> _vtk.vtkShadowMapPass | None:
        """Enable shadow pass.

        Returns
        -------
        :vtk:`vtkShadowMapPass`
            The enabled shadow pass.

        """
        self._check_closed()
        # shadow pass can be directly added to the base pass collection
        if self._shadow_map_pass is not None:
            return None
        self._shadow_map_pass = _vtk.vtkShadowMapPass()
        self._pass_collection.AddItem(self._shadow_map_pass.GetShadowMapBakerPass())
        self._pass_collection.AddItem(self._shadow_map_pass)
        self._update_passes()
        return self._shadow_map_pass

    def disable_shadow_pass(self) -> None:
        """Disable shadow pass."""
        self._check_closed()
        if self._shadow_map_pass is None:
            return
        self._release_graphics_resources(self._shadow_map_pass)
        self._pass_collection.RemoveItem(self._shadow_map_pass.GetShadowMapBakerPass())
        self._pass_collection.RemoveItem(self._shadow_map_pass)
        self._shadow_map_pass = None
        self._update_passes()

    def enable_depth_of_field_pass(
        self, *, automatic_focal_distance: bool = True
    ) -> _vtk.vtkDepthOfFieldPass | None:
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
        if self._dof_pass is not None:
            return None

        if self._ssao_pass is not None:
            msg = 'Depth of field pass is incompatible with the SSAO pass.'
            raise RuntimeError(msg)

        self._dof_pass = _vtk.vtkDepthOfFieldPass()
        self._dof_pass.SetAutomaticFocalDistance(automatic_focal_distance)
        self._add_pass(self._dof_pass)
        return self._dof_pass

    def disable_depth_of_field_pass(self) -> None:
        """Disable the depth of field pass."""
        self._check_closed()
        if self._dof_pass is None:
            return
        self._remove_pass(self._dof_pass)
        self._dof_pass = None

    def enable_ssao_pass(
        self, *, radius: float, bias: float, kernel_size: int, blur: bool
    ) -> _vtk.vtkSSAOPass | None:
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
        if self._dof_pass is not None:
            msg = 'SSAO pass is incompatible with the depth of field pass.'
            raise RuntimeError(msg)

        if self._ssao_pass is not None:
            return None
        self._ssao_pass = _vtk.vtkSSAOPass()
        self._ssao_pass.SetRadius(radius)
        self._ssao_pass.SetBias(bias)
        self._ssao_pass.SetKernelSize(kernel_size)
        self._ssao_pass.SetBlur(blur)
        self._add_pass(self._ssao_pass)
        return self._ssao_pass

    def disable_ssao_pass(self) -> None:
        """Disable the screen space ambient occlusion pass."""
        self._check_closed()
        if self._ssao_pass is None:
            return
        self._remove_pass(self._ssao_pass)
        self._ssao_pass = None

    def enable_ssaa_pass(self) -> _vtk.vtkSSAAPass | None:
        """Enable super-sample anti-aliasing pass.

        Returns
        -------
        :vtk:`vtkSSAAPass`
            The enabled super-sample anti-aliasing pass.

        """
        if self._ssaa_pass is not None:
            return None
        self._ssaa_pass = _vtk.vtkSSAAPass()
        self._add_pass(self._ssaa_pass)
        return self._ssaa_pass

    def disable_ssaa_pass(self) -> None:
        """Disable super-sample anti-aliasing pass."""
        self._check_closed()
        if self._ssaa_pass is None:
            return
        self._remove_pass(self._ssaa_pass)
        self._ssaa_pass = None

    def _update_passes(self) -> None:
        """Reassemble pass delegation."""
        self._check_closed()

        current_pass: _vtk.vtkRenderPass = self._camera_pass
        for class_name in PRE_PASS + POST_PASS:
            if class_name in self._passes:
                for render_pass in self._passes[class_name]:
                    render_pass.SetDelegatePass(current_pass)
                    current_pass = render_pass

        # reset to the default rendering if no special passes have been added
        if current_pass is self._camera_pass and self._shadow_map_pass is None:
            self._renderer.SetPass(None)  # type: ignore[union-attr, arg-type]
        else:
            self._renderer.SetPass(current_pass)  # type: ignore[union-attr]

    def _add_pass(self, render_pass: _DelegatingPass) -> None:
        """Add a render pass."""
        class_name = render_pass.GetClassName()

        if class_name in PRE_PASS and render_pass in self._passes:
            return

        if class_name not in self._passes:
            self._passes[class_name] = [render_pass]
        else:
            self._passes[class_name].append(render_pass)

        self._update_passes()

    def _release_graphics_resources(self, render_pass: _vtk.vtkRenderPass) -> None:
        """Free the GPU resources a pass holds before it is dropped."""
        renderer = self._renderer
        ren_win = None if renderer is None else renderer.GetRenderWindow()
        if ren_win is not None:
            render_pass.ReleaseGraphicsResources(ren_win)

    def _remove_pass(self, render_pass: _DelegatingPass) -> None:
        """Remove a pass.

        Remove a pass and reassemble the pass ordering.

        """
        class_name = render_pass.GetClassName()

        if class_name not in self._passes:  # pragma: no cover
            return
        else:
            self._release_graphics_resources(render_pass)
            self._passes[class_name].remove(render_pass)
            if not self._passes[class_name]:
                self._passes.pop(class_name)

        self._update_passes()
