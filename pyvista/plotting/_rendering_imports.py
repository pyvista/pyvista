"""Side-effect VTK rendering imports, routed through the active backend.

These name *modules* imported purely to register rendering factories, so they are
resolved against the active backend root (see :mod:`pyvista._vtk`) rather than the
flat class namespace. Kept in their own module, imported first by
``pyvista.plotting``, so the factories register before any rendering class and the
ordering is a plain ``import`` rather than E402-triggering top-level statements.
"""

from __future__ import annotations

import contextlib
import importlib

from pyvista import _vtk

# Charts (2D context rendering).
importlib.import_module(f'{_vtk._VTK_ROOT}.vtkRenderingContextOpenGL2')
# Text rendering; vtkRenderingMatplotlib adds MathText/LaTeX and is optional
# (absent when VTK is built without Matplotlib -- check_math_text_support() is then False).
# https://discourse.vtk.org/t/how-to-check-if-mathtext-is-supported-without-importing-all-of-vtk/16038
importlib.import_module(f'{_vtk._VTK_ROOT}.vtkRenderingFreeType')
with contextlib.suppress(ImportError):
    importlib.import_module(f'{_vtk._VTK_ROOT}.vtkRenderingMatplotlib')
