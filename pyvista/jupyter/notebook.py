"""Support dynamic or static jupyter notebook plotting.

Includes:

* ``trame``
* ``client``
* ``server``
* ``html``

"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from pyvista._warn_external import warn_external
from pyvista.jupyter import _custom_backends
from pyvista.jupyter import _discover_entry_points
from pyvista.jupyter import _get_custom_backend_handler
from pyvista.jupyter import _resolve_backend

if TYPE_CHECKING:
    import io
    from pathlib import Path

    from IPython.lib.display import IFrame
    from PIL.Image import Image

    from pyvista import pyvista_ndarray
    from pyvista.jupyter import JupyterBackendOptions
    from pyvista.plotting.plotter import Plotter
    from pyvista.trame.jupyter import EmbeddableWidget
    from pyvista.trame.jupyter import Widget


def handle_plotter(
    plotter: Plotter,
    backend: JupyterBackendOptions | str | None = None,
    screenshot: str | Path | io.BytesIO | bool | None = None,  # noqa: FBT001
    **kwargs,
) -> EmbeddableWidget | IFrame | Widget | Image:
    """Show the ``pyvista`` plot in a jupyter environment.

    Returns
    -------
    IPython Widget
        IPython widget or image.

    """
    if screenshot is False:
        screenshot = None

    # Auto-detect the best available backend when not specified
    if backend is None:
        backend = _resolve_backend()
        if backend == 'static':
            warn_external(
                'Using static image for notebook display.\n'
                'Install trame for interactive backends:'
                ' pip install "pyvista[jupyter]"'
            )

    # Custom backends (registered or from entry points)
    custom_handler = _get_custom_backend_handler(backend)
    if custom_handler is not None:
        return custom_handler(plotter, screenshot=screenshot, **kwargs)

    # Built-in trame backends
    try:
        if backend in ['server', 'client', 'trame', 'html']:
            from pyvista.trame.jupyter import show_trame  # noqa: PLC0415

            return show_trame(plotter, mode=backend, **kwargs)

    except ImportError as e:
        # Trame was explicitly requested but not available
        _discover_entry_points()
        if _custom_backends:
            fallback_name, fallback_handler = next(iter(_custom_backends.items()))
            available = [f'"{b}"' for b in sorted(_custom_backends.keys())]
            available += ['"static"', '"none"']
            warn_external(
                f'Failed to use notebook backend "{backend}": {e}\n\n'
                f'Using registered backend "{fallback_name}" instead.\n'
                f'Available backends: {", ".join(available)}'
            )
            return fallback_handler(plotter, screenshot=screenshot, **kwargs)

        warn_external(
            f'Failed to use notebook backend "{backend}": {e}\n\n'
            'Falling back to a static output.\n'
            'Available backends: "static", "none"\n'
            'Install trame for interactive backends:'
            ' pip install "pyvista[jupyter]"'
        )

    return show_static_image(plotter, screenshot)


def show_static_image(
    plotter: Plotter,
    screenshot: str | Path | io.BytesIO | bool | None,  # noqa: FBT001
) -> Image:  # numpydoc ignore=RT01
    """Display a static image to be displayed within a jupyter notebook."""
    import PIL.Image  # noqa: PLC0415

    if plotter.last_image is None:
        # Must render here, otherwise plotter will segfault.
        plotter.render()
        plotter.last_image = plotter.screenshot(screenshot, return_img=True)
    last_image = cast('pyvista_ndarray', plotter.last_image)
    return PIL.Image.fromarray(last_image)
