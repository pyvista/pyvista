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
            available += ['"static"', '"wasm"', '"none"']
            warn_external(
                f'Failed to use notebook backend "{backend}": {e}\n\n'
                f'Using registered backend "{fallback_name}" instead.\n'
                f'Available backends: {", ".join(available)}'
            )
            return fallback_handler(plotter, screenshot=screenshot, **kwargs)

        warn_external(
            f'Failed to use notebook backend "{backend}": {e}\n\n'
            'Falling back to a static output.\n'
            'Available backends: "static", "wasm", "none"\n'
            'Install trame for interactive backends:'
            ' pip install "pyvista[jupyter]"\n'
            'Or use WASM for browser-based rendering:'
            ' pip install pyvista-wasm'
        )

    # WASM backend for Pyodide/JupyterLite environments
    if backend == 'wasm':
        return show_wasm(plotter)  # type: ignore[return-value]

    return show_static_image(plotter, screenshot)


def show_wasm(
    plotter: Plotter,
) -> object:
    """Show the plotter using WASM/VTK.wasm for browser-based rendering.

    This function enables interactive 3D visualization in JupyterLite,
    Pyodide, and other browser-based Python environments using VTK.wasm.

    Parameters
    ----------
    plotter : pyvista.Plotter
        The PyVista plotter to display.

    Returns
    -------
    object
        An IPython-displayable widget for the WASM visualization.

    Raises
    ------
    ImportError
        If ``pyvista-wasm`` is not installed.

    Examples
    --------
    >>> import pyvista as pv
    >>> plotter = pv.Plotter()
    >>> _ = plotter.add_mesh(pv.Sphere())
    >>> pv.set_jupyter_backend('wasm')
    >>> plotter.show()  # doctest: +SKIP

    """
    try:
        import pyvista_wasm  # noqa: PLC0415
    except ImportError as e:
        msg = (
            'The WASM backend requires pyvista-wasm.\n'
            'Install it with: pip install "pyvista[wasm]"\n\n'
            'For Pyodide/JupyterLite, use:\n'
            'import micropip\n'
            'await micropip.install("pyvista-wasm")'
        )
        raise ImportError(msg) from e

    from IPython.display import HTML  # noqa: PLC0415

    # Create a WASM plotter from the current plotter
    wasm_plotter = pyvista_wasm.Plotter()

    # Transfer meshes and settings from the original plotter
    for actor in plotter.actors.values():
        if hasattr(actor, 'mapper') and hasattr(actor.mapper, 'dataset'):
            mesh = actor.mapper.dataset
            # Convert to WASM mesh format
            wasm_mesh = _convert_to_wasm_mesh(mesh)
            if wasm_mesh is not None:
                # Transfer actor properties
                prop = getattr(actor, 'prop', None)
                color = getattr(prop, 'color', None) if prop is not None else None
                opacity = getattr(prop, 'opacity', 1.0) if prop is not None else 1.0
                show_edges = getattr(prop, 'show_edges', False) if prop is not None else False
                wasm_plotter.add_mesh(
                    wasm_mesh,
                    color=color,
                    opacity=opacity,
                    show_edges=show_edges,
                )

    # Generate standalone HTML for embedding
    html_content = wasm_plotter.generate_standalone_html()
    return HTML(html_content)


def _convert_to_wasm_mesh(mesh):
    """Convert a pyvista mesh to a pyvista_wasm mesh.

    Parameters
    ----------
    mesh : pyvista.DataSet
        The pyvista mesh to convert.

    Returns
    -------
    pyvista_wasm.PolyData or None
        The converted WASM mesh, or None if conversion fails.

    """
    import numpy as np  # noqa: PLC0415
    import pyvista_wasm  # noqa: PLC0415

    try:
        # Extract points and faces from the mesh
        points = np.array(mesh.points)

        # Handle face data conversion
        if hasattr(mesh, 'faces') and mesh.faces is not None:
            faces = np.array(mesh.faces)
        elif hasattr(mesh, 'cells') and mesh.cells is not None:
            faces = np.array(mesh.cells)
        else:
            # Try to extract from cell connectivity
            try:
                faces = mesh.cell_connectivity
                if faces is None:
                    faces = np.array([])
            except (AttributeError, ValueError):
                faces = np.array([])

        # Create WASM PolyData
        wasm_mesh = pyvista_wasm.PolyData(points, faces)

        # Transfer point and cell data
        for name in mesh.point_data.keys():
            wasm_mesh.point_data[name] = np.array(mesh.point_data[name])

        for name in mesh.cell_data.keys():
            wasm_mesh.cell_data[name] = np.array(mesh.cell_data[name])

        # Transfer field data
        for name in mesh.field_data.keys():
            wasm_mesh.field_data[name] = np.array(mesh.field_data[name])
    except Exception:  # noqa: BLE001
        # If conversion fails, return None
        return None
    else:
        return wasm_mesh


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
