from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from pyvista.core.errors import PyVistaDeprecationWarning

namespace_data = Path(__file__).parent / 'namespace-plotting.txt'
with namespace_data.open() as f:
    namespace = f.read().splitlines()
    # ignore commented data
    namespace = [n.split(', ')[0] for n in namespace if not n.startswith('#')]


def test_plotting_top_namespace():
    """Every recorded name is still reachable from `pyvista.plotting`."""
    module = importlib.import_module('pyvista.plotting')
    missing = [name for name in namespace if not hasattr(module, name)]
    assert not missing, f'Missing from the `pyvista.plotting` namespace: {missing}'


def test_common_plotting_import_paths():
    # These are `pyvista.plotting.plotting` imports found via search on GitHub
    # across multiple public repositories
    with pytest.warns(
        PyVistaDeprecationWarning,
        match='The `pyvista.plotting.plotting` module has been deprecated',
    ):
        from pyvista.plotting.plotting import _ALL_PLOTTERS  # noqa: F401
    with pytest.warns(
        PyVistaDeprecationWarning,
        match='The `pyvista.plotting.plotting` module has been deprecated',
    ):
        from pyvista.plotting.plotting import BasePlotter  # noqa: F401
    with pytest.warns(
        PyVistaDeprecationWarning,
        match='The `pyvista.plotting.plotting` module has been deprecated',
    ):
        from pyvista.plotting.plotting import Plotter  # noqa: F401
