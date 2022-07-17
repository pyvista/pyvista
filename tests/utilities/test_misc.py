import os

import pytest

from pyvista.utilities.misc import _set_plot_theme_from_env, has_module


def test_set_plot_theme_from_env():
    os.environ['PYVISTA_PLOT_THEME'] = 'not a valid theme'
    try:
        with pytest.warns(UserWarning, match='Invalid'):
            _set_plot_theme_from_env()
    finally:
        os.environ.pop('PYVISTA_PLOT_THEME', None)


def test_has_module():
    assert has_module('pytest')
    assert not has_module('not_a_module')
