"""Support dynamic or static jupyter notebook plotting.

Trame backends (``trame``, ``client``, ``server``, ``html``) are
provided by the optional :mod:`trame_pyvista` package, which registers
them via the ``pyvista.jupyter_backends`` entry-point group.

"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from pyvista._warn_external import warn_external
from pyvista.jupyter import _custom_backends
from pyvista.jupyter import _ensure_entry_points
from pyvista.jupyter import _get_custom_backend_handler
from pyvista.jupyter import _resolve_backend

if TYPE_CHECKING:
    from io import BytesIO
    from pathlib import Path

    from IPython.lib.display import IFrame
    from PIL.Image import Image
    from trame_pyvista.jupyter import EmbeddableWidget
    from trame_pyvista.jupyter import Widget

    from pyvista.jupyter import JupyterBackendOptions
    from pyvista.plotting.plotter import Plotter


def handle_plotter(
    plotter: Plotter,
    backend: JupyterBackendOptions | str | None = None,
    screenshot: str | Path | BytesIO | bool | None = None,  # noqa: FBT001
    **kwargs: Any,
) -> EmbeddableWidget | IFrame | Widget | Image:
    """Show the ``pyvista`` plot in a jupyter environment.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Plotter to display.

    backend : str, optional
        Jupyter backend to use.

    screenshot : str | pathlib.Path | io.BytesIO | bool, optional
        Save a screenshot to this path when set.

    **kwargs : dict, optional
        Passed to the backend handler.

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

    _check_server_proxy(backend, kwargs)

    # Custom backends (registered or from entry points—including trame-pyvista)
    custom_handler = _get_custom_backend_handler(backend)
    if custom_handler is not None:
        return cast(
            'EmbeddableWidget | IFrame | Widget | Image',
            custom_handler(plotter, screenshot=screenshot, **kwargs),
        )

    # Trame backend names with no registered handler—fall back with a hint
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


def _check_server_proxy(backend: str, kwargs: dict[str, Any]) -> None:
    """Warn when a trame iframe is routed through an unavailable ``jupyter-server-proxy``."""
    if (kwargs.get('mode') or backend) not in ('client', 'server', 'trame', 'trame-pyvista'):
        return
    import pyvista as pv  # noqa: PLC0415

    def option(name: str) -> bool:
        value = kwargs.get(name)
        return getattr(pv.global_theme.trame, name) if value is None else value

    if option('jupyter_extension_enabled') or not option('server_proxy_enabled'):
        return
    # The kernel and Jupyter Server can live in different environments, so warn, not raise.
    if importlib.util.find_spec('jupyter_server_proxy') is None:
        warn_external(
            f'The "{backend}" notebook backend serves the plot through '
            'jupyter-server-proxy, which is not importable from the kernel environment. '
            'If the plot shows a 404 page, install it where Jupyter Server runs:\n'
            '    pip install jupyter-server-proxy'
        )


def show_static_image(
    plotter: Plotter,
    screenshot: str | Path | BytesIO | bool | None,  # noqa: FBT001
) -> Image:
    """Display a static image to be displayed within a jupyter notebook.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Plotter to take the screenshot from.

    screenshot : str | pathlib.Path | io.BytesIO | bool, optional
        Save the screenshot to this path when set.

    Returns
    -------
    PIL.Image.Image
        Static image of the plotter.

    """
    import PIL.Image  # noqa: PLC0415

    if plotter.last_image is None:
        # Must render here, otherwise plotter will segfault.
        plotter.render()
        plotter.last_image = plotter.screenshot(screenshot, return_img=True, render=False)
    return PIL.Image.fromarray(plotter.last_image)
