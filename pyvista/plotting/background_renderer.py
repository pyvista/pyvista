"""Contains the BackgroundRenderer class."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import numpy as np
import pyvista_validation as _validation

import pyvista as pv
from pyvista import _vtk

from .renderer import Renderer

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from pyvista.core._typing_core import VectorLike

    from .plotter import BasePlotter


class BackgroundRenderer(Renderer):
    """BackgroundRenderer for visualizing a background image.

    Parameters
    ----------
    parent : pyvista.Plotter
        The plotter the background renderer is drawn by.
    image_path : str | Path
        Path to the image to use as a background.
    scale : float, default: 1
        Scaling factor for the background image.
    view_port : sequence[float], optional
        Viewport for the background renderer as ``(xmin, ymin, xmax, ymax)``.

    """

    def __init__(
        self,
        parent: BasePlotter,
        image_path: str | Path,
        *,
        scale: float = 1,
        view_port: VectorLike[float] | None = None,
    ) -> None:
        """Initialize BackgroundRenderer with an image."""
        # read the image first as we don't need to create a render if
        # the image path is invalid
        image_data = pv.read(image_path, cls=pv.ImageData)

        super().__init__(parent, border=False)
        self.SetLayer(0)
        self.InteractiveOff()
        self.SetBackground(self._plotter.renderer.GetBackground())
        self._scale = scale
        self._modified_observer = None
        self._prior_window_size: list[int] | None = None
        if view_port is not None:
            self.viewport = _validation.validate_arrayN(
                view_port, must_have_length=4, dtype_out=float, to_list=True, name='view_port'
            )

        # create image actor
        image_actor = _vtk.vtkImageActor()
        image_actor.SetInputData(image_data)
        self.add_actor(image_actor, name='background')
        self.camera.enable_parallel_projection()
        self.reset_camera()  # necessary to get first render
        self.resize()

    def resize(self, *args: Any) -> None:  # noqa: ARG002
        """Resize a background renderer.

        Parameters
        ----------
        *args : tuple
            Ignored arguments.

        """
        if self.parent is None:  # when deleted
            return
        if self.parent.render_window is None:  # BasePlotter
            return

        if self._actors is None:  # the renderer has been closed
            return

        if self._prior_window_size != self.parent.window_size:
            self._prior_window_size = self.parent.window_size

        actor = cast('_vtk.vtkImageActor', self._actors['background'])
        image_data = actor.GetInput()
        origin = image_data.GetOrigin()
        extent = image_data.GetExtent()
        spacing = image_data.GetSpacing()
        xc = origin[0] + 0.5 * (extent[0] + extent[1]) * spacing[0]
        yc = origin[1] + 0.5 * (extent[2] + extent[3]) * spacing[1]
        yd = (extent[3] - extent[2] + 1) * spacing[1]
        dist = self.camera.distance

        # make the longest dimensions match the plotting window
        img_dim = np.array(image_data.dimensions[:2])
        self.camera._focus = np.array([xc, yc, 0.0])
        self.camera.position = np.array([xc, yc, dist])

        ratio = img_dim / np.array(self.parent.window_size)
        scale_value: float = 1
        if ratio.max() > 1:
            # images are not scaled if larger than the window
            scale_value = ratio.max()

        if self._scale is not None:
            scale_value /= self._scale

        self.camera.parallel_scale = 0.5 * yd / self._scale
