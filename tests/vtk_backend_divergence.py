"""Reasons a test diverges on an alternative VTK build.

Defined once and imported at each site, so when the fork's module set changes a
single edit covers every test rather than a grep for repeated string literals.
Each is used with the ``skip_vtk_backend`` marker::

    @pytest.mark.skip_vtk_backend('cvista', reason=NO_SNAKE_CASE)
    def test_snake_case_api(): ...
"""

from __future__ import annotations

# Behaviour that differs by design rather than by omission.
NO_SNAKE_CASE = 'cvista omits the VTK snake_case wrapper API'
TRIMMED_CLASS_SET = 'cvista wraps a trimmed VTK class set'
TRIMMED_MODULE_SET = 'cvista wraps a trimmed VTK module set'
CVISTA_NAMESPACE = 'module loads under the cvista namespace'
INT32_CELL_STORAGE = 'cvista stores connectivity as int32 (no zero-copy share)'
INT32_COMPRESSION = 'cvista stores indices as int32 (smaller, less compressible)'
CELL_STATUS_ENUM = 'cvista diverges on vtkCellStatus enum exposure'
FIXED_SIZE_CELL_STORAGE = (
    'cvista uses fixed-size cell storage for uniform cells where stock VTK does not'
)
