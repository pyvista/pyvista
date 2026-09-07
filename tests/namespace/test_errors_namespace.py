from __future__ import annotations

import re
import warnings

from pyvista.core.errors import PyVistaDeprecationWarning

CORE_ERRORS = [
    'AmbiguousDataError',
    'DeprecationError',
    'MissingDataError',
    'NotAllTrianglesError',
    'PointSetCellOperationError',
    'PointSetDimensionReductionError',
    'PointSetNotSupported',
    'PyVistaDeprecationWarning',
    'PyVistaEfficiencyWarning',
    'PyVistaFutureWarning',
    'PyVistaPipelineError',
    'VTKVersionError',
]

PLOTTING_ERRORS = [
    'InvalidCameraError',
    'RenderWindowUnavailable',
]


def _not_forwarded(names: list[str], module_path: str) -> list[str]:
    """Return the names `pyvista.errors` does not forward with the expected warning."""
    import pyvista.errors as errors  # noqa: PLR0402

    failed = []
    for name in names:
        pattern = rf'now imported as: `from {module_path} import {name}`\.'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            found = hasattr(errors, name)
        deprecations = [
            str(w.message) for w in caught if issubclass(w.category, PyVistaDeprecationWarning)
        ]
        if not found or not any(re.search(pattern, message) for message in deprecations):
            failed.append(name)
    return failed


def test_core_errors_namespace():
    """Every core error still forwards from the deprecated `pyvista.errors`."""
    assert not _not_forwarded(CORE_ERRORS, r'pyvista\.core\.errors')


def test_plotting_errors_namespace():
    """Every plotting error still forwards from the deprecated `pyvista.errors`."""
    assert not _not_forwarded(PLOTTING_ERRORS, r'pyvista\.plotting\.errors')
