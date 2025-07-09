from __future__ import annotations

import pytest

from pyvista.core.errors import PyVistaDeprecationWarning


@pytest.mark.parametrize(
    'name',
    [
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
    ],
)
def test_core_errors_namespace(name):
    with pytest.warns(
        PyVistaDeprecationWarning,
        match=rf'now imported as: `from pyvista\.core\.errors import {name}`\.',
    ):
        import pyvista.errors as errors  # noqa: PLR0402

        assert hasattr(errors, name)


@pytest.mark.parametrize(
    'name',
    [
        'InvalidCameraError',
        'RenderWindowUnavailable',
    ],
)
def test_plotting_errors_namespace(name):
    with pytest.warns(
        PyVistaDeprecationWarning,
        match=rf'now imported as: `from pyvista\.plotting\.errors import {name}`\.',
    ):
        import pyvista.errors as errors  # noqa: PLR0402

        assert hasattr(errors, name)
