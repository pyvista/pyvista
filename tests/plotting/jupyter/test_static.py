from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import numpy as np
import pytest

import pyvista as pv
from pyvista.jupyter import _validate_jupyter_backend

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

has_ipython = bool(importlib.util.find_spec('IPython'))
if has_ipython:
    from PIL.Image import Image


skip_no_ipython = pytest.mark.skipif(not has_ipython, reason='Requires IPython package')


def test_set_jupyter_backend_fail():
    with pytest.raises(ValueError, match='Invalid Jupyter notebook plotting backend'):
        pv.set_jupyter_backend('not a backend')


def test_validate_jupyter_backend_raises(mocker: MockerFixture):
    from pyvista import jupyter

    m = mocker.patch.object(jupyter, 'importlib')
    m.util.find_spec.return_value = False
    with pytest.raises(
        ImportError, match='Install IPython to display with pyvista in a notebook.'
    ):
        _validate_jupyter_backend('foo')


@pytest.mark.parametrize('backend', [None, 'none'])
def test_set_jupyter_backend_none(backend):
    pv.set_jupyter_backend(backend)
    assert pv.global_theme.jupyter_backend == 'none'


@skip_no_ipython
def test_set_jupyter_backend_static():
    pv.set_jupyter_backend('static')
    assert pv.global_theme.jupyter_backend == 'static'
    pv.set_jupyter_backend(None)


@skip_no_ipython
@pytest.mark.skip_plotting
@pytest.mark.parametrize('return_viewer', [True, False])
def test_static_from_show(sphere, return_viewer):
    window_size = (100, 100)
    image = sphere.plot(
        window_size=window_size,
        notebook=True,
        jupyter_backend='static',
        return_viewer=return_viewer,
    )
    if return_viewer:
        assert isinstance(image, Image)
        assert window_size == image.size


@skip_no_ipython
@pytest.mark.skip_plotting
def test_show_return_values(sphere: pv.PolyData):
    # Three possible return values: (cpos, image, widget)
    img, viewer = sphere.plot(
        notebook=True,
        return_viewer=True,
        return_cpos=False,
        return_img=True,
        jupyter_backend='static',
    )
    assert isinstance(img, np.ndarray)
    assert isinstance(viewer, Image)

    cpos, img, viewer = sphere.plot(
        notebook=True,
        return_viewer=True,
        return_cpos=True,
        return_img=True,
        jupyter_backend='static',
    )
    assert isinstance(cpos, pv.CameraPosition)
    assert isinstance(img, np.ndarray)
    assert isinstance(viewer, Image)
