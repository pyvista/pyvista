import pytest

import pyvista as pv

has_panel = True
try:
    from panel.pane.vtk.vtk import VTKRenderWindowSynchronized
except:
    has_panel = False

skip_no_panel = pytest.mark.skipif(not has_panel, reason='Requires panel')


def test_set_jupyter_backend_ipygany_fail():
    with pytest.raises(ValueError, match='Invalid Jupyter notebook plotting backend'):
        pv.set_jupyter_backend('not a backend')


@pytest.mark.parametrize('backend', [None, 'none'])
def test_set_jupyter_backend_none(backend):
    pv.set_jupyter_backend(backend)
    assert pv.rcParams['jupyter_backend'] is None


@skip_no_panel
def test_set_jupyter_backend_ipygany():
    pv.set_jupyter_backend('panel')
    assert pv.rcParams['jupyter_backend'] == 'panel'


@skip_no_panel
def test_panel(sphere):
    viewer = sphere.plot(notebook=True, jupyter_backend='panel', return_viewer=True)
    assert isinstance(viewer, VTKRenderWindowSynchronized)
