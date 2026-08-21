"""Plotting GL checks."""

from __future__ import annotations

import functools
import os
import sys

from pyvista import _vtk
from pyvista.plotting.tools import _prepare_offscreen_macos_render_window


def _qt_platform_name() -> str | None:
    """Return the platform name of a live Qt application, if there is one.

    Only bindings that are *already imported* are consulted: this must never
    import Qt, and must never construct an application.

    Returns
    -------
    str | None
        The platform Qt actually connected to, or ``None`` when no Qt
        application is running in this process.

    """
    for module_name in ('PySide6.QtGui', 'PyQt6.QtGui', 'PySide2.QtGui', 'PyQt5.QtGui'):
        # getattr rather than an import, and tolerant of a module that is in
        # sys.modules but still initializing
        qgui = getattr(sys.modules.get(module_name), 'QGuiApplication', None)
        if qgui is None:
            continue
        app = qgui.instance()
        if app is not None:
            return str(app.platformName())
    return None


def _process_uses_egl() -> bool:
    """Return whether this process draws through EGL rather than GLX.

    ``WAYLAND_DISPLAY`` only says a Wayland compositor is running, not that
    *this* process talks to it: ``QT_QPA_PLATFORM=xcb`` runs a Qt application
    through XWayland inside a Wayland session, and then its OpenGL is GLX. An
    EGL render window cannot make its context current in such a process, so
    every render through it logs ``Unable to eglMakeCurrent`` with
    ``EGL_BAD_ACCESS`` (pyvista/pyvistaqt#445 follow-up).

    A running Qt application is authoritative, since it is the thing that did
    or did not connect to the compositor. Without one, the session variable is
    all there is to go on.

    Returns
    -------
    bool
        ``True`` when this process's OpenGL goes through EGL.

    """
    platform_name = _qt_platform_name()
    if platform_name is not None:
        return platform_name.startswith('wayland')
    return bool(os.environ.get('WAYLAND_DISPLAY'))


def _offscreen_probe_render_window():
    """Create an offscreen render window suitable for GL capability probes.

    Under a Wayland session the process may already be using EGL for OpenGL,
    for example through a Qt application running on the native ``wayland``
    platform (pyvista/pyvistaqt#445). Making a GLX context current in such a
    process aborts it with ``X Error ... BadAccess (X_GLXMakeCurrent)``, so
    the default (GLX-based) ``vtkXOpenGLRenderWindow`` cannot be used for the
    probe. The converse mix is harmless: an EGL render window works in a
    process that already uses GLX. So prefer EGL whenever a Wayland session is
    detected, and keep the factory default everywhere else.

    An explicit ``VTK_DEFAULT_OPENGL_WINDOW`` override always wins: the
    factory honors it, and the user's choice also determines the backend the
    rest of the process uses, so matching it keeps the probe consistent (and
    safe) with the actual rendering backend.
    """
    if (
        not os.environ.get('VTK_DEFAULT_OPENGL_WINDOW')
        and _process_uses_egl()
        and _vtk.has_attr('vtkEGLRenderWindow')
    ):
        return _vtk.vtkEGLRenderWindow()
    return _vtk.vtkRenderWindow()


@functools.cache
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
    renderWindow = _offscreen_probe_render_window()
    renderWindow.SetOffScreenRendering(True)
    _prepare_offscreen_macos_render_window(renderWindow)
    renderWindow.AddRenderer(renderer)
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
    if _process_uses_egl():
        # Instantiating the factory-default render window is not safe here:
        # constructing (and destroying) the default GLX-based window aborts a
        # process that already uses EGL, e.g. a Qt application running on the
        # native ``wayland`` platform (pyvista/pyvistaqt#445). Answer without
        # instantiation instead: honor an explicit backend override, otherwise
        # infer from the build -- headless EGL/OSMesa wheels are compiled
        # without X support.
        backend = os.environ.get('VTK_DEFAULT_OPENGL_WINDOW')
        if backend:
            return 'EGL' in backend or 'OSOpenGL' in backend
        return not _vtk.has_attr('vtkXOpenGLRenderWindow')
    ren_win_str = str(type(_vtk.vtkRenderWindow()))
    return 'EGL' in ren_win_str or 'OSOpenGL' in ren_win_str
