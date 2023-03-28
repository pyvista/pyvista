"""
Support dynamic or static jupyter notebook plotting.

Includes:

* ``ipyvtklink``
* ``panel``
* ``pythreejs``
* ``ipygany``
* ``client``
* ``server``

"""
import os
import warnings

import numpy as np

from pyvista import _vtk
from pyvista.utilities.misc import PyVistaDeprecationWarning

PANEL_EXTENSION_SET = [False]


def handle_plotter(plotter, backend=None, screenshot=None, **kwargs):
    """Show the ``pyvista`` plot in a jupyter environment.

    Returns
    -------
    IPython Widget
        IPython widget or image.

    """
    if screenshot is False:
        screenshot = None

    try:
        if backend == 'pythreejs':
            return show_pythreejs(plotter, **kwargs)
        if backend == 'ipyvtklink':
            warnings.warn(
                '`ipyvtklink` backend is deprecated and has been replaced by the `trame` backend.',
                PyVistaDeprecationWarning,
            )
            return show_ipyvtk(plotter)
        if backend == 'panel':
            return show_panel(plotter)
        if backend == 'ipygany':
            from pyvista.jupyter.pv_ipygany import show_ipygany

            return show_ipygany(plotter, **kwargs)
        if backend in ['server', 'client', 'trame']:
            from pyvista.trame.jupyter import show_trame

            return show_trame(plotter, mode=backend, **kwargs)

    except ImportError as e:
        warnings.warn(
            f'Failed to use notebook backend: \n\n{e}\n\nFalling back to a static output.'
        )

    return show_static_image(plotter, screenshot)


def show_static_image(plotter, screenshot):
    """Display a static image to be displayed within a jupyter notebook."""
    import PIL.Image

    if plotter.last_image is None:
        # Must render here, otherwise plotter will segfault.
        plotter.render()
        plotter.last_image = plotter.screenshot(screenshot, return_img=True)
    image = PIL.Image.fromarray(plotter.last_image)

    # Simply display the result: either ipyvtklink object or image display
    return image


def show_ipyvtk(plotter):
    """Display an interactive viewer widget using ``ipyvtklink``."""
    if any('SPYDER' in name for name in os.environ):
        warnings.warn(
            '``ipyvtklink`` backend is incompatible with Spyder.\n'
            'Use notebook=False for interactive '
            'plotting within spyder or disable it globally with:\n'
            'pyvista.set_jupyter_backend(None)'
        )

    try:
        from ipyvtklink.viewer import ViewInteractiveWidget
    except ImportError:  # pragma: no cover
        raise ImportError(
            'Please install `ipyvtklink` to use this feature: '
            'https://github.com/Kitware/ipyvtklink'
        )

    # Have to leave the Plotter open for the widget to use
    disp = ViewInteractiveWidget(
        plotter.render_window,
        on_close=plotter.close,
        transparent_background=plotter.image_transparent_background,
    )

    for renderer in plotter.renderers:
        renderer.AddObserver(_vtk.vtkCommand.ModifiedEvent, lambda *args: disp.update_canvas())

    return disp


def show_panel(plotter):
    """Take the active renderer or renderers from a plotter and show them using ``panel``."""
    try:
        import panel as pn
    except ImportError:  # pragma: no cover
        raise ImportError('Install ``panel`` to use this feature')

    # check if panel extension has been set
    if not PANEL_EXTENSION_SET[0]:
        pn.extension('vtk')
        PANEL_EXTENSION_SET[0] = True

    # only set window size if explicitly set within the plotter
    sizing = {}
    if not plotter._window_size_unset:
        width, height = plotter.window_size
        sizing = {'width': width, 'height': height}

    axes_enabled = plotter.renderer.axes_enabled
    pan = pn.panel(
        plotter.render_window,
        sizing_mode='stretch_width',
        orientation_widget=axes_enabled,
        enable_keybindings=False,
        **sizing,
    )

    # if plotter.renderer.axes_enabled:
    # pan.axes = build_panel_axes()

    if plotter.renderer.cube_axes_actor is not None:
        pan.axes = build_panel_bounds(plotter.renderer.cube_axes_actor)

    return pan


def build_panel_bounds(actor):
    """Build a panel bounds actor using the plotter cube_axes_actor."""
    bounds = {}

    n_ticks = 5
    if actor.GetXAxisVisibility():
        xmin, xmax = actor.GetXRange()
        bounds['xticker'] = {'ticks': np.linspace(xmin, xmax, n_ticks)}

    if actor.GetYAxisVisibility():
        ymin, ymax = actor.GetYRange()
        bounds['yticker'] = {'ticks': np.linspace(ymin, ymax, n_ticks)}

    if actor.GetZAxisVisibility():
        zmin, zmax = actor.GetZRange()
        bounds['zticker'] = {'ticks': np.linspace(zmin, zmax, n_ticks)}

    bounds['origin'] = [xmin, ymin, zmin]
    bounds['grid_opacity'] = 0.5
    bounds['show_grid'] = True
    bounds['digits'] = 3
    bounds['fontsize'] = actor.GetLabelTextProperty(0).GetFontSize()

    return bounds


def show_pythreejs(plotter, **kwargs):
    """Show a pyvista plotting scene using pythreejs."""
    from .pv_pythreejs import convert_plotter

    renderer = convert_plotter(plotter)
    return renderer
