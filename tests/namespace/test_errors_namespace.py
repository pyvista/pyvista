import importlib
import pathlib

import pytest

from pyvista.core.errors import PyVistaDeprecationWarning


@pytest.mark.parametrize('name', [
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
    'InvalidCameraError',
    'RenderWindowUnavailable',
])
def test_utilities_namespace(name):
    with pytest.warns(PyVistaDeprecationWarning):
        import pyvista.errors as errors

        assert hasattr(errors, name)
