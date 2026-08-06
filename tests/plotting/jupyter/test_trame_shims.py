"""Tests for the deprecated ``pyvista.trame`` re-export shims.

The Trame integration moved to the standalone :mod:`trame_pyvista`
package in PyVista 0.49. The ``pyvista.trame`` modules remain as
deprecation shims that emit :class:`PyVistaDeprecationWarning` on
import and re-export the public symbols from their new home so older
code keeps working until the next breaking release.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

import pytest

from pyvista.core.errors import PyVistaDeprecationWarning

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec('trame_pyvista') is None,
    reason='trame_pyvista not installed',
)

SHIMS = [
    pytest.param(
        'pyvista.trame',
        'trame_pyvista',
        'pyvista.trame is deprecated',
        id='pyvista.trame',
    ),
    pytest.param(
        'pyvista.trame.jupyter',
        'trame_pyvista.jupyter',
        'pyvista.trame.jupyter is deprecated',
        id='pyvista.trame.jupyter',
    ),
    pytest.param(
        'pyvista.trame.views',
        'trame_pyvista.widgets',
        'pyvista.trame.views is deprecated',
        id='pyvista.trame.views',
    ),
    pytest.param(
        'pyvista.trame.ui',
        'trame_pyvista.ui',
        'pyvista.trame.ui is deprecated',
        id='pyvista.trame.ui',
    ),
    pytest.param(
        'pyvista.trame.ui.base_viewer',
        'trame_pyvista.ui.base_viewer',
        'pyvista.trame.ui.base_viewer is deprecated',
        id='pyvista.trame.ui.base_viewer',
    ),
    pytest.param(
        'pyvista.trame.ui.vuetify2',
        'trame_pyvista.ui.vuetify2',
        'pyvista.trame.ui.vuetify2 is deprecated',
        id='pyvista.trame.ui.vuetify2',
    ),
    pytest.param(
        'pyvista.trame.ui.vuetify3',
        'trame_pyvista.ui.vuetify3',
        'pyvista.trame.ui.vuetify3 is deprecated',
        id='pyvista.trame.ui.vuetify3',
    ),
]


def _fresh_import(module_name: str):
    """Drop ``module_name`` from ``sys.modules`` and import it again."""
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.mark.parametrize(('shim_name', 'upstream_name', 'expected_message'), SHIMS)
def test_shim_emits_deprecation_warning(shim_name, upstream_name, expected_message):
    """Importing a shim emits a :class:`PyVistaDeprecationWarning`."""
    del upstream_name
    with pytest.warns(PyVistaDeprecationWarning, match=expected_message):
        _fresh_import(shim_name)


@pytest.mark.parametrize(('shim_name', 'upstream_name', 'expected_message'), SHIMS)
def test_shim_reexports_match_upstream(shim_name, upstream_name, expected_message):
    """Every name in the shim's ``__all__`` is the object from upstream."""
    del expected_message
    with pytest.warns(PyVistaDeprecationWarning):
        shim = _fresh_import(shim_name)
    upstream = importlib.import_module(upstream_name)

    assert shim.__all__, f'{shim_name} has empty __all__'
    for name in shim.__all__:
        assert hasattr(upstream, name), f'{upstream_name} is missing re-exported symbol {name!r}'
        assert getattr(shim, name) is getattr(upstream, name), (
            f'{shim_name}.{name} is not the object from {upstream_name}.{name}'
        )
