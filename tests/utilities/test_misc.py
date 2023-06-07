import os

import numpy as np
import pytest

from pyvista import examples
from pyvista.core.utilities.fileio import _try_imageio_imread
from pyvista.core.utilities.misc import has_module
from pyvista.themes import _set_plot_theme_from_env

HAS_IMAGEIO = True
try:
    import imageio
except ModuleNotFoundError:
    HAS_IMAGEIO = False


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


@pytest.mark.skipif(not HAS_IMAGEIO, reason="Requires imageio")
def test_try_imageio_imread():
    img = _try_imageio_imread(examples.mapfile)
    assert isinstance(img, (imageio.core.util.Array, np.ndarray))
