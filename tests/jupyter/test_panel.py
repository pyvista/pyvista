import os

import pytest

import pyvista as pv
from pyvista.utilities.misc import PyVistaDeprecationWarning

has_panel = True
try:
    from panel.pane.vtk.vtk import VTKRenderWindowSynchronized
except:  # noqa: E722
    has_panel = False

skip_no_panel = pytest.mark.skipif(not has_panel, reason='Requires panel')


@skip_no_panel
def test_set_jupyter_backend_panel():
    try:
        with pytest.warns(PyVistaDeprecationWarning):
            pv.set_jupyter_backend('panel')
        assert pv.global_theme.jupyter_backend == 'panel'
    finally:
        pv.set_jupyter_backend(None)


@skip_no_panel
@pytest.mark.parametrize('return_viewer', [True, False])
def test_panel(sphere, return_viewer):
    viewer = sphere.plot(
        notebook=True,
        jupyter_backend='panel',
        return_viewer=return_viewer,
        window_size=(100, 100),
        show_bounds=True,
    )
    if return_viewer:
        assert isinstance(viewer, VTKRenderWindowSynchronized)
    else:
        return viewer is None


@skip_no_panel
def test_save(sphere, tmpdir):
    filename = str(tmpdir.join('tmp.html'))
    plotter = pv.Plotter(shape=(1, 2))

    plotter.add_text("Airplane 1\n", font_size=30, color='grey')
    plotter.add_mesh(sphere, show_edges=False, color='grey')

    plotter.subplot(0, 1)
    plotter.add_text("Airplane 2\n", font_size=30, color='grey')
    plotter.add_mesh(sphere, show_edges=False, color='grey')

    plotter.export_html(filename, backend='panel')
    assert os.path.isfile(filename)
