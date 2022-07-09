"""Module to wrap vtkOpenGLRenderer."""

import pyvista


class RenderWindow(pyvista._vtk.vtkRenderWindow):
    """Wrap vtkOpenGLRenderer."""

    def __init__(self, renderers):
        """Initialize the render window."""
        super().__init__()
        self._camera_setup = False
        self._rendered = False
        self._renderers = renderers
        for renderer in renderers:
            self.AddRenderer(renderer)

    def render(self):
        """Render this render window.

        This does nothing until the cameras have been setup.
        """
        # do not render until the camera has been setup
        if not self._camera_setup:
            return

        self.Render()
        self._rendered = True

    @property
    def rendered(self):
        """Return if this render window has ever been rendered."""
        return self._rendered

    @property
    def renderers(self):
        """Return the renderers of this plotter."""
        return self._renderers

    def finalize(self):
        """Finalize and clear out the render window."""
        self._renderers = None
        self.Finalize()

    def setup_camera(self, cpos=None):
        """Set up the camera on the very first render request.

        For example on the show call or any screenshot producing code.
        """
        # reset unless camera for the first render unless camera is set
        if not self._camera_setup:
            for renderer in self.renderers:
                if not isinstance(renderer, pyvista.Renderer):
                    continue
                if not renderer.camera_set and cpos is None:
                    renderer.camera_position = renderer.get_default_cam_pos()
                    renderer.ResetCamera()
                elif cpos is not None:
                    renderer.camera_position = cpos
            self._camera_setup = True

    def show(self):
        """Set up the cameras if needed and render."""
        self.setup_camera()
        self.render()
