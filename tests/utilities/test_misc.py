import os

import pytest

from pyvista.utilities.misc import _set_plot_theme_from_env


def test_set_plot_theme_from_env():
    os.environ['PYVISTA_PLOT_THEME'] = 'not a valid theme'
    try:
        with pytest.warns(UserWarning, match='Invalid'):
            _set_plot_theme_from_env()
    finally:
        os.environ.pop('PYVISTA_PLOT_THEME', None)
