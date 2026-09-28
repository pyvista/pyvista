"""Plotting GL checks."""

from __future__ import annotations

import functools
import os
from pathlib import Path
import re
import sys

from pyvista import _vtk
from pyvista.plotting.tools import _prepare_offscreen_macos_render_window

# Qt GUI modules that would expose a live QGuiApplication, newest binding first
_QT_GUI_MODULES = ('PySide6.QtGui', 'PyQt6.QtGui', 'PySide2.QtGui', 'PyQt5.QtGui')

_PROC_MAPS = Path('/proc/self/maps')


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
    for module_name in _QT_GUI_MODULES:
        # getattr rather than an import, and tolerant of a module that is in
        # sys.modules but still initializing
        qgui = getattr(sys.modules.get(module_name), 'QGuiApplication', None)
        if qgui is None:
            continue
        app = qgui.instance()
        if app is not None:
            return str(app.platformName())
    return None


def _gl_platform_from_maps(maps: str) -> str | None:
    """Return the GL platform implied by the libraries listed in a memory map.

    Parameters
    ----------
    maps : str
        The contents of ``/proc/<pid>/maps``.

    Returns
    -------
    str | None
        ``'egl'`` or ``'glx'`` when exactly one of ``libEGL`` and ``libGLX`` is
        mapped, otherwise ``None``.

    """
    # not libGL: EGL programs may link it for the GL entry points
    has_egl = re.search(r'/libEGL[^/\s]*\.so', maps) is not None
    has_glx = re.search(r'/libGLX[^/\s]*\.so', maps) is not None
    if has_egl == has_glx:
        return None
    return 'egl' if has_egl else 'glx'


def _loaded_gl_platform() -> str | None:
    """Return the GL platform implied by the libraries mapped into this process.

    Returns
    -------
    str | None
        ``'egl'`` or ``'glx'``, or ``None`` when the libraries do not settle
        it or the platform has no ``/proc/self/maps``.

    """
    try:
        maps = _PROC_MAPS.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return None
    return _gl_platform_from_maps(maps)


def _process_uses_egl() -> bool:
    """Return whether this process draws through EGL rather than GLX.

    ``WAYLAND_DISPLAY`` only says a Wayland compositor is running, not that
    *this* process talks to it: ``QT_QPA_PLATFORM=xcb`` runs a Qt application
    through XWayland inside a Wayland session, and then its OpenGL is GLX. An
    EGL render window cannot make its context current in such a process, so
    every render through it logs ``Unable to eglMakeCurrent`` with
    ``EGL_BAD_ACCESS`` (pyvista/pyvistaqt#445 follow-up).

    A running Qt application is authoritative, since it is the thing that did
    or did not connect to the compositor. Failing that, the GL libraries mapped
    into the process answer for any other host that has created a window, such
    as a GTK application; importing VTK maps neither. Without either, VTK's
    factory picks the window, and it prefers GLX (through XWayland) when
    ``DISPLAY`` is set, so only a Wayland session without ``DISPLAY`` means EGL.

    Returns
    -------
    bool
        ``True`` when this process's OpenGL goes through EGL.

    """
    platform_name = _qt_platform_name()
    if platform_name is not None:
        return platform_name.startswith('wayland')
    loaded = _loaded_gl_platform()
    if loaded is not None:
        return loaded == 'egl'
    return bool(os.environ.get('WAYLAND_DISPLAY')) and not os.environ.get('DISPLAY')


def _offscreen_probe_render_window() -> _vtk.vtkRenderWindow:
    """Create an offscreen render window suitable for GL capability probes.

    Under a Wayland session the process may already be using EGL for OpenGL,
    for example through a Qt application running on the native ``wayland``
    platform (pyvista/pyvistaqt#445). Making a GLX context current in such a
    process aborts it with ``X Error ... BadAccess (X_GLXMakeCurrent)``, so
    the default (GLX-based) ``vtkXOpenGLRenderWindow`` cannot be used for the
    probe. The converse mix is less severe but still broken: an EGL render
    window in a process that already uses GLX logs ``EGL_BAD_ACCESS`` on every
    render. So use EGL when ``_process_uses_egl`` says this process draws
    through EGL, and keep the factory default everywhere else.

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
def check_depth_peeling(number_of_peels: int = 100, occlusion_ratio: float = 0.0) -> bool:
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
        # instantiation instead: honor an explicit backend override, then a
        # mapped libEGL, otherwise infer from the build -- headless EGL/OSMesa
        # wheels are compiled without X support.
        backend = os.environ.get('VTK_DEFAULT_OPENGL_WINDOW')
        if backend:
            return 'EGL' in backend or 'OSOpenGL' in backend
        return _loaded_gl_platform() == 'egl' or not _vtk.has_attr('vtkXOpenGLRenderWindow')
    ren_win_str = str(type(_vtk.vtkRenderWindow()))
    return 'EGL' in ren_win_str or 'OSOpenGL' in ren_win_str
