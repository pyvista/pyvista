# flake8: noqa: D102,D103,D107
"""PyVista Trame User Interface.

This module builds a base UI for manipulating a PyVista Plotter.
The UI generated here is the default for rendering in Jupyter
environments and provides a starting point for custom user-built
applications.
"""
from typing import Dict
import warnings

from trame.app import get_server

from .base_viewer import BaseViewer
from .vuetify2 import Viewer as Vue2Viewer
from .vuetify3 import Viewer as Vue3Viewer

_VIEWERS: Dict[str, BaseViewer] = {}
UI_TITLE = 'PyVista'


def get_viewer(plotter, server=None, suppress_rendering=False):
    """Get a Viewer instance for a given Plotter.

    There should be only one Viewer instance per plotter. A Viewer
    can have multiple UI views though.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Plotter to return or create the viewer instance for.

    server : trame.Server, optional
        Current server for Trame application.

    suppress_rendering : bool, default: False
        Suppress rendering on the plotter.

    Returns
    -------
    pyvista.trame.ui.Viewer
        Trame viewer.

    """
    if plotter._id_name in _VIEWERS:
        viewer = _VIEWERS[plotter._id_name]
        if suppress_rendering != plotter.suppress_rendering:
            plotter.suppress_rendering = suppress_rendering
            warnings.warn(
                "Suppress rendering on the plotter is changed to " + str(suppress_rendering),
                UserWarning,
            )
        return viewer

    if not server:
        server = get_server()
    if server.client_type == 'vue2':
        viewer = Vue2Viewer(plotter, suppress_rendering=suppress_rendering, server=server)
    else:
        viewer = Vue3Viewer(plotter, suppress_rendering=suppress_rendering, server=server)

    _VIEWERS[plotter._id_name] = viewer
    return viewer


def plotter_ui(
    plotter, mode=None, default_server_rendering=True, collapse_menu=False, add_menu=True, **kwargs
):
    """Create a UI view for the given Plotter.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Plotter to create the UI for.

    mode : str, default: 'trame'
        The UI view mode. Options are:

        * ``'trame'``: Uses a view that can switch between client and server
          rendering modes.
        * ``'server'``: Uses a view that is purely server rendering.
        * ``'client'``: Uses a view that is purely client rendering (generally
          safe without a virtual frame buffer)

    default_server_rendering : bool, default: True
        Whether to use server-side or client-side rendering on-start when
        using the ``'trame'`` mode.

    collapse_menu : bool, default: False
        Collapse the UI menu (camera controls, etc.) on start.

    add_menu : bool, default: True
        Add a UI controls VCard to the VContainer.

    **kwargs : dict, optional
        Additional keyword arguments are passed to the viewer being created.

    Returns
    -------
    PyVistaRemoteLocalView | PyVistaRemoteView | PyVistaLocalView
        Trame view interface for pyvista.

    """
    viewer = get_viewer(plotter, server=kwargs.get('server'), suppress_rendering=mode == 'client')
    return viewer.ui(
        mode=mode,
        default_server_rendering=default_server_rendering,
        collapse_menu=collapse_menu,
        add_menu=add_menu,
        **kwargs,
    )
