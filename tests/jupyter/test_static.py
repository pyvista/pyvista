import pytest

import pyvista as pv

has_ipython = True
try:
    import IPython  # noqa
    from PIL.Image import Image
except:  # noqa: E722
    has_ipython = False

skip_no_ipython = pytest.mark.skipif(not has_ipython, reason="Requires IPython package")


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
