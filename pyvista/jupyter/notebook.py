"""
Support dynamic or static jupyter notebook plotting.

Includes:

* ``ipyvtk_simple``
# ``panel``

"""
import warnings
import os

import numpy as np

# This module should not be imported at the __init__ level, only as a
# lazy import
try:
    import IPython
    from IPython import display
except ImportError:
    raise ImportError('Install IPython to display an image in a notebook')

PANEL_EXTENSION_SET = [False]


def handle_plotter(plotter, backend=None, screenshot=None,
                   return_viewer=False):
    """Show the ``pyvista`` plot in a jupyter environment.

    Parameters
    ----------
    return_viewer : bool, optional
        Return the jupyterlab viewer, scene, or display object
        when plotting with jupyter notebook.

    Returns
    -------
    IPython Widget
        IPython widget when ``return_viewer==True``.  Otherwise, ``None``.

    """
    if screenshot is False:
        screenshot = None
    if backend == 'ipyvtk_simple':
        return show_ipyvtk(plotter, return_viewer)
    if backend == 'panel':
        return show_panel(plotter, return_viewer)
    if backend == 'ipygany':
        from pyvista.jupyter.pv_ipygany import show_ipygany
        return show_ipygany(plotter, return_viewer)

    return show_static_image(plotter, screenshot, return_viewer)


def show_static_image(plotter, screenshot, return_viewer):
    """Display a static image to be displayed within a jupyter notebook."""
    # Must render here, otherwise plotter will segfault.  Consider
    # moving to screenshot logic
    plotter.render()

    import PIL.Image

    if not hasattr(plotter, 'last_image'):
        plotter.last_image = plotter.screenshot(screenshot, return_img=True)
    disp = display.display(PIL.Image.fromarray(plotter.last_image))

    # close plotter as this will be a static image and there is no
    # point to keeping the plotter around.
    plotter.close()

    # Simply display the result: either ipyvtk_simple object or image display
    if return_viewer:
        return disp
    else:
        display.display_html(disp)


def show_ipyvtk(plotter, return_viewer):
    """Display an interactive viewer widget using ``ipyvtk_simple``."""
    if any('SPYDER' in name for name in os.environ):
        warnings.warn('``use_ipyvtk`` is incompatible with Spyder.\n'
                      'Use notebook=False for interactive '
                      'plotting within spyder or disable it globally with:\n'
                      'pyvista.set_jupyter_backend(None)')

    try:
        from ipyvtk_simple.viewer import ViewInteractiveWidget
    except ImportError:
        raise ImportError('Please install `ipyvtk_simple` to use this feature:'
                          '\thttps://github.com/Kitware/ipyvtk-simple')

    # Have to leave the Plotter open for the widget to use
    disp = ViewInteractiveWidget(plotter.ren_win, on_close=plotter.close,
                                 transparent_background=plotter.image_transparent_background)

    if return_viewer:
        return disp
    display.display_html(disp)


def show_panel(plotter, return_viewer):
    """Take the active renderer from a plotter and show it using ``panel``."""
    if len(plotter.renderers) != 1:
        warnings.warn('Plotter has multiple renderers.  Only showing the active '
                      'renderer using ``panel``')

    try:
        import panel as pn
    except ImportError:
        raise ImportError('Install ``panel`` to use this feature')

    if not PANEL_EXTENSION_SET[0]:
        pn.extension('vtk')
        PANEL_EXTENSION_SET[0] = True

    width, height = plotter.window_size
    axes_enabled = plotter.renderer.axes_enabled
    pan = pn.panel(plotter.ren_win, width=width, height=height,
                   orientation_widget=axes_enabled,
                   enable_keybindings=False)

    # sizing_mode='stretch_width'  # doesn't work when graphics saved/buggy
    # ìnteractive_orientation_widget  # can't seem to be enabled

    # if plotter.renderer.axes_enabled:
        # pan.axes = build_panel_axes()

    if hasattr(plotter.renderer, 'cube_axes_actor'):
        pan.axes = build_panel_bounds(plotter.renderer.cube_axes_actor)

    if return_viewer:
        return pan
    display.display_html(pan)


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
