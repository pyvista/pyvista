import numpy as np
import pytest

import pyvista as pv
from pyvista.plotting import system_supports_plotting

has_ipython = True
try:
    import IPython  # noqa
    from PIL.Image import Image
except:  # noqa: E722
    has_ipython = False

skip_no_ipython = pytest.mark.skipif(not has_ipython, reason="Requires IPython package")

skip_no_plotting = pytest.mark.skipif(
    not system_supports_plotting(), reason="Requires system to support plotting"
)


def test_set_jupyter_backend_fail():
    with pytest.raises(ValueError, match='Invalid Jupyter notebook plotting backend'):
        pv.set_jupyter_backend('not a backend')


@pytest.mark.parametrize('backend', [None, 'none'])
def test_set_jupyter_backend_none(backend):
    pv.set_jupyter_backend(backend)
    assert pv.global_theme.jupyter_backend is None


@skip_no_ipython
def test_set_jupyter_backend_static():
    pv.set_jupyter_backend('static')
    assert pv.global_theme.jupyter_backend == 'static'
    pv.set_jupyter_backend(None)


@skip_no_ipython
@skip_no_plotting
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
@skip_no_plotting
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
