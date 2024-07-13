"""
Support dynamic or static jupyter notebook plotting.

Includes:

* ``trame``
* ``client``
* ``server``
* ``html``

"""

from __future__ import annotations

import warnings


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
        if backend in ['server', 'client', 'trame', 'html']:
            from pyvista.trame.jupyter import show_trame

            return show_trame(plotter, mode=backend, **kwargs)

    except ImportError as e:
        warnings.warn(
            f'Failed to use notebook backend: \n\n{e}\n\nFalling back to a static output.',
        )

    return show_static_image(plotter, screenshot)


def show_static_image(plotter, screenshot):
    """Display a static image to be displayed within a jupyter notebook."""
    import PIL.Image

    if plotter.last_image is None:
        # Must render here, otherwise plotter will segfault.
        plotter.render()
        plotter.last_image = plotter.screenshot(screenshot, return_img=True)
    return PIL.Image.fromarray(plotter.last_image)
