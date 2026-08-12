"""Side-effect VTK rendering imports, routed through the active backend.

Magic vtk imports needed to make LaTeX rendering work. See
https://discourse.vtk.org/t/how-to-check-if-mathtext-is-supported-without-importing-all-of-vtk/16038

These name a *module* rather than a class -- they are imported purely for their
side effects (registering rendering factories) -- so they cannot go through the
flat class namespace and must be resolved against the active backend root; see
:mod:`pyvista._vtk`.

They live in their own module, imported first by ``pyvista.plotting``, so the
ordering guarantee (factories registered before any rendering class is imported)
is expressed as a real ``import`` statement rather than as executable statements
ahead of the package's imports, which would make every later import an E402.
"""

from __future__ import annotations

import contextlib
import importlib

from pyvista import _vtk

importlib.import_module(f'{_vtk._VTK_ROOT}.vtkRenderingContextOpenGL2')
importlib.import_module(f'{_vtk._VTK_ROOT}.vtkRenderingFreeType')

# VTK may be built without its Matplotlib module. MathText/LaTeX rendering is then
# unavailable (``check_math_text_support()`` returns False), but plotting otherwise
# works.
with contextlib.suppress(ImportError):
    importlib.import_module(f'{_vtk._VTK_ROOT}.vtkRenderingMatplotlib')
