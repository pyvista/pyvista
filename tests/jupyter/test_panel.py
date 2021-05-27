import pytest

import pyvista as pv

has_panel = True
try:
    from panel.pane.vtk.vtk import VTKRenderWindowSynchronized
except:
    has_panel = False

skip_no_panel = pytest.mark.skipif(not has_panel, reason='Requires panel')


@skip_no_panel
def test_set_jupyter_backend_panel():
    pv.set_jupyter_backend('panel')
    assert pv.global_theme.jupyter_backend == 'panel'
    pv.set_jupyter_backend(None)


@skip_no_panel
@pytest.mark.parametrize('return_viewer', [True, False])
def test_panel(sphere, return_viewer):
    viewer = sphere.plot(notebook=True, jupyter_backend='panel',
                         return_viewer=return_viewer,
                         window_size=(100, 100), show_bounds=True)
    if return_viewer:
        assert isinstance(viewer, VTKRenderWindowSynchronized)
    else:
        return viewer is None
