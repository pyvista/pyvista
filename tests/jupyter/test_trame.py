from IPython.display import IFrame
import numpy as np
import pytest

import pyvista as pv
from pyvista.plotting import system_supports_plotting

has_trame = True
try:
    from trame.app import get_server

    from pyvista.trame.jupyter import Widget
    from pyvista.trame.ui import get_or_create_viewer
    from pyvista.trame.views import PyVistaLocalView, PyVistaRemoteLocalView, PyVistaRemoteView
except:  # noqa: E722
    has_trame = False

# skip all tests if VTK<9.1.0
if pv.vtk_version_info < (9, 1):
    pytestmark = pytest.mark.skip

skip_no_plotting = pytest.mark.skipif(
    not system_supports_plotting(), reason="Requires system to support plotting"
)

skip_no_trame = pytest.mark.skipif(not has_trame, reason="Requires trame")


@skip_no_trame
def test_set_jupyter_backend_trame():
    try:
        pv.global_theme.jupyter_backend = 'trame'
        assert pv.global_theme.jupyter_backend == 'trame'
        pv.global_theme.jupyter_backend = 'client'
        assert pv.global_theme.jupyter_backend == 'client'
        pv.global_theme.jupyter_backend = 'server'
        assert pv.global_theme.jupyter_backend == 'server'
    finally:
        pv.global_theme.jupyter_backend = None


@skip_no_trame
@skip_no_plotting
def test_trame_server_launch():
    pv.set_jupyter_backend('trame')
    server = get_server(name=pv.global_theme.trame.jupyter_server_name)
    assert server.running


@skip_no_trame
@skip_no_plotting
def test_trame():
    pv.set_jupyter_backend('trame')
    server = get_server(name=pv.global_theme.trame.jupyter_server_name)
    assert server.running

    pl = pv.Plotter(notebook=True)
    actor = pl.add_mesh(pv.Cone())
    widget = pl.show(return_viewer=True)
    assert isinstance(widget, Widget)
    assert 'http://' in widget.src

    viewer = get_or_create_viewer(pl)

    for cp in ['xy', 'xz', 'yz', 'isometric']:
        exec(f'viewer.view_{cp}()')
        cpos = list(pl.camera_position)
        pl.camera_position = cp[:3]
        assert cpos == pl.camera_position

    orig_value = actor.prop.show_edges
    server.state[viewer.EDGES] = not orig_value
    viewer.on_edge_visiblity_change(**server.state.to_dict())
    assert actor.prop.show_edges != orig_value

    server.state[viewer.GRID] = True
    assert len(pl.actors) == 1
    viewer.on_grid_visiblity_change(**server.state.to_dict())
    assert len(pl.actors) == 2
    server.state[viewer.GRID] = False
    viewer.on_grid_visiblity_change(**server.state.to_dict())
    assert len(pl.actors) == 1

    server.state[viewer.OUTLINE] = True
    assert len(pl.actors) == 1
    viewer.on_outline_visiblity_change(**server.state.to_dict())
    assert len(pl.actors) == 2
    server.state[viewer.OUTLINE] = False
    viewer.on_outline_visiblity_change(**server.state.to_dict())
    assert len(pl.actors) == 1

    server.state[viewer.AXIS] = True
    assert not hasattr(pl.renderer, 'axes_actor')
    viewer.on_axis_visiblity_change(**server.state.to_dict())
    assert hasattr(pl.renderer, 'axes_actor')
    server.state[viewer.AXIS] = False
    viewer.on_axis_visiblity_change(**server.state.to_dict())
    assert not pl.renderer.axes_widget.GetEnabled()

    server.state[viewer.SERVER_RENDERING] = False
    viewer.on_rendering_mode_change(**server.state.to_dict())
    server.state[viewer.SERVER_RENDERING] = True
    viewer.on_rendering_mode_change(**server.state.to_dict())

    assert viewer.actors == pl.actors

    assert isinstance(viewer.screenshot(), memoryview)


@skip_no_trame
@skip_no_plotting
def test_trame_jupyter_modes():
    pv.set_jupyter_backend('trame')

    pl = pv.Plotter(notebook=True)
    pl.add_mesh(pv.Cone())
    widget = pl.show(jupyter_backend='client', return_viewer=True)
    assert isinstance(widget, Widget)
    assert pl.suppress_rendering

    pl = pv.Plotter(notebook=True)
    pl.add_mesh(pv.Cone())
    widget = pl.show(jupyter_backend='server', return_viewer=True)
    assert isinstance(widget, Widget)
    assert not pl.suppress_rendering

    pl = pv.Plotter(notebook=True)
    pl.add_mesh(pv.Cone())
    widget = pl.show(jupyter_backend='trame', return_viewer=True)
    assert isinstance(widget, Widget)
    assert not pl.suppress_rendering


@skip_no_trame
@skip_no_plotting
def test_trame_closed_plotter():
    pl = pv.Plotter(notebook=True)
    pl.add_mesh(pv.Cone())
    pl.close()
    with pytest.raises(RuntimeError, match='The render window for this plotter has been destroyed'):
        PyVistaRemoteLocalView(pl)


@skip_no_trame
@skip_no_plotting
def test_trame_views():
    server = get_server('foo')

    pl = pv.Plotter(notebook=True)
    pl.add_mesh(pv.Cone())

    assert PyVistaRemoteLocalView(pl, trame_server=server)
    assert PyVistaRemoteView(pl, trame_server=server)
    assert PyVistaLocalView(pl, trame_server=server)


@skip_no_trame
@skip_no_plotting
def test_trame_jupyter_custom_size():
    w, h = 200, 150
    plotter = pv.Plotter(notebook=True, window_size=(w, h))
    _ = plotter.add_mesh(pv.Cone())
    widget = plotter.show(jupyter_backend='trame', return_viewer=True)
    html = widget.value
    assert f'width: {w}px' in html
    assert f'height: {h}px' in html

    plotter = pv.Plotter(notebook=True)
    plotter.window_size = (w, h)
    _ = plotter.add_mesh(pv.Cone())
    widget = plotter.show(jupyter_backend='trame', return_viewer=True)
    html = widget.value
    assert f'width: {w}px' in html
    assert f'height: {h}px' in html

    # Make sure that if size is default theme, it uses 99%/600px
    previous_size = pv.global_theme.window_size
    pv.global_theme.window_size = pv.themes.DefaultTheme().window_size
    try:
        plotter = pv.Plotter(notebook=True)
        _ = plotter.add_mesh(pv.Cone())
        widget = plotter.show(jupyter_backend='trame', return_viewer=True)
        html = widget.value
        assert 'width: 99%' in html
        assert 'height: 600px' in html
    finally:
        pv.global_theme.window_size = previous_size


@skip_no_trame
@skip_no_plotting
def test_trame_jupyter_custom_handler():
    def handler(viewer, src, **kwargs):
        return IFrame(src, '75%', '500px')

    plotter = pv.Plotter(notebook=True)
    _ = plotter.add_mesh(pv.Cone())
    iframe = plotter.show(
        jupyter_backend='trame',
        jupyter_kwargs=dict(handler=handler),
        return_viewer=True,
    )
    assert isinstance(iframe, IFrame)


@skip_no_trame
@skip_no_plotting
def test_trame_int64():
    mesh = pv.Sphere()
    mesh['int64'] = np.arange(mesh.n_cells, dtype=np.int64)

    plotter = pv.Plotter(notebook=True)
    _ = plotter.add_mesh(mesh, scalars='int64')
    widget = plotter.show(
        jupyter_backend='trame',
        return_viewer=True,
    )
    # Basically just assert that it didn't error out
    assert isinstance(widget, Widget)
