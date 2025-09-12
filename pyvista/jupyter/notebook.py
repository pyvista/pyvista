"""Support dynamic or static jupyter notebook plotting.

Includes:

* ``trame``
* ``client``
* ``server``
* ``html``

"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING
import warnings

import numpy as np

if TYPE_CHECKING:
    import io
    from pathlib import Path

    from IPython.lib.display import IFrame
    from PIL.Image import Image

    from pyvista.jupyter import JupyterBackendOptions
    from pyvista.plotting.plotter import Plotter
    from pyvista.trame.jupyter import EmbeddableWidget
    from pyvista.trame.jupyter import Widget


def handle_plotter(
    plotter: Plotter,
    backend: JupyterBackendOptions | None = None,
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

    try:
        if backend in ['server', 'client', 'trame', 'html']:
            from pyvista.trame.jupyter import show_trame  # noqa: PLC0415

            return show_trame(plotter, mode=backend, **kwargs)

        if backend == 'vtk-wasm':
            return show_vtk_wasm(plotter, **kwargs)

    except ImportError as e:
        warnings.warn(
            f'Failed to use notebook backend: \n\n{e}\n\nFalling back to a static output.',
        )

    return show_static_image(plotter, screenshot)


def show_vtk_wasm(plotter: Plotter, **kwargs) -> IFrame:  # noqa: ARG001  # numpydoc ignore=RT01
    """Display a VTK WebAssembly widget in a jupyter notebook."""
    try:
        from IPython.display import IFrame  # noqa: PLC0415
        import trame_vtklocal  # noqa: F401, PLC0415
        import vtk  # noqa: F401, PLC0415
    except ImportError as e:
        msg = (
            'VTK-WASM backend requires VTK>=9.4 and trame-vtklocal. Install with:'
            '    pip install --extra-index-url vtk>=9.4 trame-vtklocal'
        )
        raise ImportError(msg) from e

    # Export the scene to a format compatible with vtk-wasm
    # Create a temporary HTML file with the vtk-wasm viewer
    # TODO: Integrate actual VTK scene data with vtk-wasm viewer

    # Export the VTK scene data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        # Create HTML content with vtk-wasm viewer
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>VTK-WASM Viewer</title>
            <script src="https://unpkg.com/vtk-wasm/dist/vtk-wasm.js"></script>
        </head>
        <body>
            <div id="vtk-container" style="width: 100%; height: 600px;"></div>
            <script>
                // Initialize vtk-wasm viewer
                const container = document.getElementById('vtk-container');
                // VTK-WASM integration code will be added here
                console.log('VTK-WASM viewer initialized');
            </script>
        </body>
        </html>
        """
        f.write(html_content)
        temp_file = f.name

    # Return an IFrame pointing to the temporary file
    return IFrame(temp_file, width='100%', height=600)


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
    assert isinstance(plotter.last_image, np.ndarray)
    return PIL.Image.fromarray(plotter.last_image)
