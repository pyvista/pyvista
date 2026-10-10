"""Support dynamic or static jupyter notebook plotting.

Trame backends (``trame``, ``client``, ``server``, ``html``) are
provided by the optional :mod:`trame_pyvista` package, which registers
them via the ``pyvista.jupyter_backends`` entry-point group.

"""

from __future__ import annotations

import functools
import importlib.util
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from pyvista._warn_external import warn_external
from pyvista.jupyter import _TRAME_BACKENDS
from pyvista.jupyter import _TRAME_SERVER_BACKENDS
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
                ' pip install pyvista[jupyter]'
            )

    # Custom backends (registered or from entry points—including trame-pyvista)
    custom_handler = _get_custom_backend_handler(backend)
    if custom_handler is not None:
        _check_server_proxy(backend, kwargs)
        return cast(
            'EmbeddableWidget | IFrame | Widget | Image',
            custom_handler(plotter, screenshot=screenshot, **kwargs),
        )

    # Trame backend names with no registered handler—fall back with a hint
    if backend in _TRAME_BACKENDS:
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
            ' pip install pyvista[jupyter]'
        )

    return show_static_image(plotter, screenshot)


def _check_server_proxy(backend: str, kwargs: dict[str, Any]) -> None:
    """Warn when a trame iframe is routed through an unavailable ``jupyter-server-proxy``."""
    mode = kwargs.get('mode') or backend
    if mode not in _TRAME_SERVER_BACKENDS:
        return

    import pyvista as pv  # noqa: PLC0415

    trame_theme = pv.global_theme.trame
    extension_enabled = kwargs.get('jupyter_extension_enabled')
    if extension_enabled is None:
        extension_enabled = trame_theme.jupyter_extension_enabled
    proxy_enabled = kwargs.get('server_proxy_enabled')
    if proxy_enabled is None:
        proxy_enabled = trame_theme.server_proxy_enabled
    proxy_prefix = kwargs.get('server_proxy_prefix')
    if proxy_prefix is None:
        proxy_prefix = trame_theme.server_proxy_prefix
    if extension_enabled or not proxy_enabled or str(proxy_prefix).startswith('http'):
        return

    # The kernel and Jupyter Server can live in different environments, so warn, not raise.
    if importlib.util.find_spec('jupyter_server_proxy') is None:
        _warn_missing_server_proxy()


@functools.cache
def _warn_missing_server_proxy() -> None:
    """Warn once per process that ``jupyter-server-proxy`` is not importable."""
    warn_external(
        'Trame notebook plots are served through jupyter-server-proxy, which is not '
        'importable from the kernel environment. If plots show a 404 page, install it '
        'where Jupyter Server runs:\n'
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
