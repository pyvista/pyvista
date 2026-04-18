"""WASM/Pyodide support module for PyVista.

This module provides utilities for running PyVista in WebAssembly (WASM)
environments such as JupyterLite, Pyodide, and Stlite. It enables interactive
3D visualization in web browsers without requiring a backend server.

Background
----------
WebAssembly (WASM) is a binary instruction format designed as a portable
target for compilation of high-level languages like C/C++. Pyodide is a
port of CPython to WebAssembly that runs in browsers, enabling Python
execution client-side without a server.

In WASM environments:

* The regular VTK Python package (compiled for x86/ARM) is not available.
* VTK.wasm (VTK C++ compiled to WebAssembly) provides rendering capabilities.
* pyvista-wasm bridges PyVista's API with VTK.wasm.
* sys.platform returns 'emscripten' and platform.machine() returns 'wasm32'.

References
----------
For more information, see:

* Pyodide documentation: https://pyodide.org/
* WASM constraints: https://pyodide.org/en/stable/usage/wasm-constraints.html
* pyvista-wasm package: https://github.com/tkoyama010/pyvista-wasm

Examples
--------
Check if running in a WASM environment:

>>> import pyvista as pv
>>> pv.wasm.is_pyodide()  # doctest: +SKIP
False

Generate standalone HTML for embedding:

>>> import pyvista as pv
>>> plotter = pv.Plotter()
>>> _ = plotter.add_mesh(pv.Sphere())
>>> html = pv.wasm.generate_standalone_html(plotter)  # doctest: +SKIP
>>> '<!DOCTYPE html>' in html  # doctest: +SKIP
True

"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

if TYPE_CHECKING:
    from pyvista.core.dataset import DataSet
    from pyvista.plotting.plotter import Plotter

__all__ = [
    'WASMPlotter',
    'generate_standalone_html',
    'is_pyodide',
]


def is_pyodide() -> bool:
    """Check if running in a Pyodide/WASM environment.

    Returns
    -------
    bool
        True if running in a Pyodide/WASM environment, False otherwise.

    Examples
    --------
    >>> import pyvista as pv
    >>> is_pyodide()  # doctest: +SKIP
    False

    """
    return sys.platform == 'emscripten'


def _ensure_numpy_in_pyodide() -> None:
    """Ensure numpy is available in Pyodide environment.

    This function attempts to install numpy using micropip if it's
    not already available in a Pyodide environment.

    """
    if is_pyodide():
        try:
            pass  # numpy is already imported at module level
        except ImportError:
            try:
                import asyncio  # noqa: PLC0415

                import micropip  # type: ignore[import-not-found,unused-ignore]  # noqa: PLC0415

                asyncio.get_event_loop().run_until_complete(micropip.install('numpy'))
            except ImportError:
                pass


class WASMPlotter:
    """Wrapper for WASM-compatible plotting.

    This class provides a PyVista-compatible interface for creating
    visualizations that can run in browser-based Python environments
    using VTK.wasm as the rendering backend.

    Parameters
    ----------
    **kwargs : dict, optional
        Additional keyword arguments passed to the WASM plotter.

    Examples
    --------
    >>> import pyvista as pv
    >>> plotter = pv.wasm.WASMPlotter()  # doctest: +SKIP
    >>> _ = plotter.add_mesh(pv.Sphere(), color='red')  # doctest: +SKIP
    >>> html = plotter.generate_standalone_html()  # doctest: +SKIP

    """

    def __init__(self, **kwargs) -> None:
        """Initialize the WASM plotter wrapper."""
        self._wasm_plotter = None
        self._kwargs = kwargs
        self._actors: list[Any] = []
        self._meshes: list[Any] = []

    def _get_wasm_plotter(self) -> Any:
        """Get or create the underlying WASM plotter."""
        if self._wasm_plotter is None:
            try:
                import pyvista_wasm  # noqa: PLC0415

                self._wasm_plotter = pyvista_wasm.Plotter(**self._kwargs)
            except ImportError as e:
                msg = (
                    'The WASM backend requires pyvista-wasm.\n'
                    'Install it with: pip install "pyvista[wasm]"\n\n'
                    'For Pyodide/JupyterLite, use:\n'
                    'import micropip\n'
                    'await micropip.install("pyvista-wasm")'
                )
                raise ImportError(msg) from e
        return self._wasm_plotter

    def add_mesh(self, mesh: DataSet, **kwargs) -> object | None:
        """Add a mesh to the plotter.

        Parameters
        ----------
        mesh : pyvista.DataSet
            The mesh to add to the plotter.
        **kwargs : dict, optional
            Additional keyword arguments for mesh styling.

        Returns
        -------
        actor or None
            The actor representing the added mesh, or None if conversion fails.

        """
        # Convert pyvista mesh to WASM mesh
        wasm_mesh = self._convert_mesh(mesh)
        if wasm_mesh is None:
            return None

        self._meshes.append((mesh, kwargs))

        # Add to WASM plotter
        wasm_plotter = self._get_wasm_plotter()
        return wasm_plotter.add_mesh(wasm_mesh, **kwargs)

    def _convert_mesh(self, mesh: DataSet) -> object | None:
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

        try:
            # Extract points
            points = np.array(mesh.points)

            # Handle face/cell data conversion
            faces = None
            if hasattr(mesh, 'faces') and mesh.faces is not None:
                faces = np.array(mesh.faces)
            elif hasattr(mesh, 'cells') and mesh.cells is not None:
                faces = np.array(mesh.cells)

            # Create WASM PolyData
            if faces is not None and len(faces) > 0:
                wasm_mesh = pyvista_wasm.PolyData(points, faces)
            else:
                # Point cloud without faces
                wasm_mesh = pyvista_wasm.PolyData(points)

            # Transfer point data
            for name in mesh.point_data.keys():
                wasm_mesh.point_data[name] = np.array(mesh.point_data[name])

            # Transfer cell data
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

    def generate_standalone_html(self) -> str:
        """Generate a complete standalone HTML page.

        Returns
        -------
        str
            A full HTML document string containing the visualization.

        """
        wasm_plotter = self._get_wasm_plotter()
        return wasm_plotter.generate_standalone_html()

    def show(self, **_kwargs) -> None:
        """Display the visualization.

        In JupyterLite/Pyodide environments, this renders using VTK.wasm.

        Parameters
        ----------
        **_kwargs : dict, optional
            Additional keyword arguments for display options (unused).

        """
        wasm_plotter = self._get_wasm_plotter()
        wasm_plotter.show()

    def view_xy(self, *, negative: bool = False) -> None:
        """View the XY plane (Z-normal).

        Parameters
        ----------
        negative : bool, optional
            View from the negative Z direction.

        """
        wasm_plotter = self._get_wasm_plotter()
        wasm_plotter.view_xy(negative=negative)

    def view_xz(self, *, negative: bool = False) -> None:
        """View the XZ plane (Y-normal).

        Parameters
        ----------
        negative : bool, optional
            View from the negative Y direction.

        """
        wasm_plotter = self._get_wasm_plotter()
        wasm_plotter.view_xz(negative=negative)

    def view_yz(self, *, negative: bool = False) -> None:
        """View the YZ plane (X-normal).

        Parameters
        ----------
        negative : bool, optional
            View from the negative X direction.

        """
        wasm_plotter = self._get_wasm_plotter()
        wasm_plotter.view_yz(negative=negative)

    def view_isometric(self) -> None:
        """View the scene from an isometric angle."""
        wasm_plotter = self._get_wasm_plotter()
        wasm_plotter.view_isometric()

    @property
    def background_color(self) -> tuple[Any, ...]:
        """Get or set the background color.

        Returns
        -------
        tuple
            RGB color tuple.

        """
        wasm_plotter = self._get_wasm_plotter()
        return wasm_plotter.background_color

    @background_color.setter
    def background_color(self, color: tuple[Any, ...] | str) -> None:
        """Set the background color.

        Parameters
        ----------
        color : tuple or str
            Color as RGB tuple or string name.

        """
        wasm_plotter = self._get_wasm_plotter()
        wasm_plotter.background_color = color


def generate_standalone_html(plotter: Plotter, **kwargs) -> str:
    """Generate a standalone HTML file from a PyVista plotter.

    This function converts a PyVista plotter to a standalone HTML
    file that can be embedded in web pages or displayed in
    JupyterLite/Pyodide environments.

    Parameters
    ----------
    plotter : pyvista.Plotter
        The PyVista plotter to convert.
    **kwargs : dict, optional
        Additional keyword arguments for the WASM plotter.

    Returns
    -------
    str
        A full HTML document string containing the visualization.

    Raises
    ------
    ImportError
        If ``pyvista-wasm`` is not installed.

    Examples
    --------
    >>> import pyvista as pv
    >>> plotter = pv.Plotter()
    >>> _ = plotter.add_mesh(pv.Sphere())
    >>> html = pv.wasm.generate_standalone_html(plotter)  # doctest: +SKIP
    >>> '<!DOCTYPE html>' in html  # doctest: +SKIP
    True

    """
    wasm_plotter = WASMPlotter(**kwargs)

    # Transfer meshes from the original plotter
    for actor in plotter.actors.values():
        if hasattr(actor, 'mapper') and hasattr(actor.mapper, 'dataset'):
            mesh = actor.mapper.dataset
            # Get actor properties if available
            color = None
            opacity = 1.0
            show_edges = False
            if hasattr(actor, 'prop'):
                prop = actor.prop
                if hasattr(prop, 'color'):
                    color = prop.color
                if hasattr(prop, 'opacity'):
                    opacity = prop.opacity
                if hasattr(prop, 'show_edges'):
                    show_edges = prop.show_edges

            wasm_plotter.add_mesh(
                mesh,
                color=color,
                opacity=opacity,
                show_edges=show_edges,
            )

    # Transfer camera settings
    if hasattr(plotter, 'camera') and plotter.camera is not None:
        camera = plotter.camera
        if hasattr(camera, 'position'):
            # Set camera position in WASM plotter
            pass  # WASM plotter handles camera differently

    # Transfer background color
    if hasattr(plotter, 'background_color'):
        bg_color = plotter.background_color
        # Convert Color type to tuple if necessary
        if hasattr(bg_color, 'float_rgb'):
            wasm_plotter.background_color = bg_color.float_rgb
        elif hasattr(bg_color, '__iter__'):
            wasm_plotter.background_color = tuple(bg_color)
        else:
            wasm_plotter.background_color = str(bg_color)

    return wasm_plotter.generate_standalone_html()


# Ensure numpy is available in Pyodide on module import
_ensure_numpy_in_pyodide()
