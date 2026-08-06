"""Support dynamic or static jupyter notebook plotting.

Trame backends (``trame``, ``client``, ``server``, ``html``) are
provided by the optional :mod:`trame_pyvista` package, which registers
them via the ``pyvista.jupyter_backends`` entry-point group.

"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from pyvista._warn_external import warn_external
from pyvista.jupyter import _custom_backends
from pyvista.jupyter import _ensure_entry_points
from pyvista.jupyter import _get_custom_backend_handler
from pyvista.jupyter import _resolve_backend

if TYPE_CHECKING:
    import io
    from pathlib import Path

    from IPython.lib.display import IFrame
    from PIL.Image import Image
    from trame_pyvista.jupyter import EmbeddableWidget
    from trame_pyvista.jupyter import Widget

    from pyvista import pyvista_ndarray
    from pyvista.jupyter import JupyterBackendOptions
    from pyvista.plotting.plotter import Plotter


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
                ' pip install trame-pyvista'
            )

    # Custom backends (registered or from entry points — including trame-pyvista)
    custom_handler = _get_custom_backend_handler(backend)
    if custom_handler is not None:
        return cast(
            'EmbeddableWidget | IFrame | Widget | Image',
            custom_handler(plotter, screenshot=screenshot, **kwargs),
        )

    # Trame backend names with no registered handler — fall back with a hint
    if backend in ('server', 'client', 'trame', 'html'):
        _ensure_entry_points()
        if _custom_backends:
            fallback_name, fallback_handler = next(iter(_custom_backends.items()))
            available = [f'"{b}"' for b in sorted(_custom_backends.keys())]
            available += ['"static"', '"none"']
            warn_external(
                f'No handler registered for notebook backend "{backend}".\n\n'
                f'Using registered backend "{fallback_name}" instead.\n'
                f'Available backends: {", ".join(available)}'
            )
            return cast(
                'EmbeddableWidget | IFrame | Widget | Image',
                fallback_handler(plotter, screenshot=screenshot, **kwargs),
            )

        warn_external(
            f'No handler registered for notebook backend "{backend}".\n\n'
            'Falling back to a static output.\n'
            'Available backends: "static", "none"\n'
            'Install trame for interactive backends:'
            ' pip install trame-pyvista'
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
