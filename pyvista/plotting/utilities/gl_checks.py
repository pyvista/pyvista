"""Plotting GL checks."""

from pyvista.plotting import _vtk


def check_depth_peeling(number_of_peels=100, occlusion_ratio=0.0):
    """Check if depth peeling is available.

    Attempts to use depth peeling to see if it is available for the
    current environment. Returns ``True`` if depth peeling is
    available and has been successfully leveraged, otherwise
    ``False``.

    Parameters
    ----------
    number_of_peels : int, default: 100
        Maximum number of depth peels.

    occlusion_ratio : float, default: 0.0
        Occlusion ratio.

    Returns
    -------
    bool
        ``True`` when system supports depth peeling with the specified
        settings.

    """
    # Try Depth Peeling with a basic scene
    source = _vtk.vtkSphereSource()
    mapper = _vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    actor = _vtk.vtkActor()
    actor.SetMapper(mapper)
    # requires opacity < 1
    actor.GetProperty().SetOpacity(0.5)
    renderer = _vtk.vtkRenderer()
    renderWindow = _vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetOffScreenRendering(True)
    renderWindow.SetAlphaBitPlanes(True)
    renderWindow.SetMultiSamples(0)
    renderer.AddActor(actor)
    renderer.SetUseDepthPeeling(True)
    renderer.SetMaximumNumberOfPeels(number_of_peels)
    renderer.SetOcclusionRatio(occlusion_ratio)
    renderWindow.Render()
    return renderer.GetLastRenderingUsedDepthPeeling() == 1


def uses_egl() -> bool:
    """Check if VTK has been compiled with EGL support via OSMesa.

    Returns
    -------
    bool
        ``True`` if VTK has been compiled with EGL support via OSMesa,
        otherwise ``False``.
    """
    ren_win_str = str(type(_vtk.vtkRenderWindow()))
    return 'EGL' in ren_win_str or 'OSOpenGL' in ren_win_str
