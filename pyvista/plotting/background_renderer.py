"""Contains the BackgroundRenderer class."""
import numpy as np
import vtk

import pyvista
from .renderer import Renderer


class BackgroundRenderer(Renderer):
    """BackgroundRenderer for visualizing a background image."""

    def __init__(self, parent, image_path, scale=1, view_port=None):
        """Initialize BackgroundRenderer with an image."""
        # read the image first as we don't need to create a render if
        # the image path is invalid
        image_data = pyvista.read(image_path)

        super().__init__(parent, border=False)
        self.SetLayer(0)
        self.InteractiveOff()
        self.SetBackground(self.parent.renderer.GetBackground())
        self._scale = scale
        self._modified_observer = None
        self._prior_window_size = None
        if view_port is not None:
            self.SetViewport(view_port)

        # create image actor
        image_actor = vtk.vtkImageActor()
        image_actor.SetInputData(image_data)
        self.add_actor(image_actor, name='background')
        self.camera.enable_parallel_projection()
        self.reset_camera()  # necessary to get first render
        self.resize()

    def resize(self, *args):
        """Resize a background renderer."""
        if self.parent is None:  # when deleted
            return

        if self._prior_window_size != self.parent.window_size:
            self._prior_window_size = self.parent.window_size

        actor = self._actors['background']
        image_data = actor.GetInput()
        origin = image_data.GetOrigin()
        extent = image_data.GetExtent()
        spacing = image_data.GetSpacing()
        xc = origin[0] + 0.5*(extent[0] + extent[1]) * spacing[0]
        yc = origin[1] + 0.5*(extent[2] + extent[3]) * spacing[1]
        yd = (extent[3] - extent[2] + 1) * spacing[1]
        dist = self.camera.distance

        # make the longest dimensions match the plotting window
        img_dim = np.array(image_data.dimensions[:2])
        self.camera.focus = np.array([xc, yc, 0.0])
        self.camera.position = np.array([xc, yc, dist])

        ratio = img_dim/np.array(self.parent.window_size)
        scale_value = 1
        if ratio.max() > 1:
            # images are not scaled if larger than the window
            scale_value = ratio.max()

        if self._scale is not None:
            scale_value /= self._scale

        self.camera.parallel_scale = 0.5 * yd / self._scale
