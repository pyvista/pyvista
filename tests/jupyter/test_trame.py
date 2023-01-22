import gc

import pytest

import pyvista as pv
from pyvista.plotting import system_supports_plotting

has_trame = True
try:
    from IPython.display import IFrame
    from trame.app import get_server

    from pyvista.trame import show_trame  # noqa
except:  # noqa: E722
    has_trame = False


skip_no_plotting = pytest.mark.skipif(
    not system_supports_plotting(), reason="Requires system to support plotting"
)

skip_no_trame = pytest.mark.skipif(not has_trame, reason="Requires trame")


@skip_no_trame
def test_set_jupyter_backend_ipyvtklink():
    pv.global_theme.jupyter_backend = 'trame'
    assert pv.global_theme.jupyter_backend == 'trame'
    pv.global_theme.jupyter_backend = 'client'
    assert pv.global_theme.jupyter_backend == 'client'
    pv.global_theme.jupyter_backend = 'server'
    assert pv.global_theme.jupyter_backend == 'server'
    pv.global_theme.jupyter_backend = None


@skip_no_trame
@skip_no_plotting
@pytest.mark.asyncio
async def test_trame(sphere):
    await pv.set_jupyter_backend('trame')

    view_obj_orig = {obj for obj in gc.get_objects() if isinstance(obj, pv.trame.ui.Viewer)}

    pl = pv.Plotter(notebook=True)
    actor = pl.add_mesh(sphere)
    widget = pl.show()
    assert isinstance(widget, IFrame)

    server = get_server(name=pv.global_theme.trame.jupyter_server_name)
    assert server.running

    # access the active Viewer object
    view_obj_now = {obj for obj in gc.get_objects() if isinstance(obj, pv.trame.ui.Viewer)}
    view_obj = view_obj_now - view_obj_orig
    assert len(view_obj) == 1
    viewer = view_obj.pop()
    assert viewer._server is server

    for cp in ['xy', 'xz', 'yz', 'isometric']:
        exec(f'viewer.view_{cp}()')
        cpos = list(pl.camera_position)
        pl.camera_position = cp[:3]
        assert cpos == pl.camera_position

    orig_value = actor.prop.show_edges
    viewer._state[viewer.EDGES] = not orig_value
    viewer.on_edge_visiblity_change()
    assert actor.prop.show_edges != orig_value

    # pl.camera.zoom(2)
    # cpos = list(pl.camera_position)
    viewer.reset_camera()
    # assert cpos != pl.camera_position

    viewer._state[viewer.GRID] = True
    assert len(pl.actors) == 1
    viewer.on_grid_visiblity_change()
    assert len(pl.actors) == 2
    viewer._state[viewer.GRID] = False
    viewer.on_grid_visiblity_change()
    assert len(pl.actors) == 1

    viewer._state[viewer.OUTLINE] = True
    assert len(pl.actors) == 1
    viewer.on_outline_visiblity_change()
    assert len(pl.actors) == 2
    viewer._state[viewer.OUTLINE] = False
    viewer.on_outline_visiblity_change()
    assert len(pl.actors) == 1

    viewer._state[viewer.AXIS] = True
    assert not hasattr(pl.renderer, 'axes_actor')
    viewer.on_axis_visiblity_change()
    assert hasattr(pl.renderer, 'axes_actor')
    viewer._state[viewer.AXIS] = False
    viewer.on_axis_visiblity_change()
    assert not pl.renderer.axes_widget.GetEnabled()

    viewer._state[viewer.SERVER_RENDERING] = False
    viewer.on_rendering_mode_change()
    viewer._state[viewer.SERVER_RENDERING] = True
    viewer.on_rendering_mode_change()

    assert viewer.actors == pl.actors

    n_called = [0]

    def callback(*args):
        n_called[0] += 1

    pl.add_on_render_callback(callback)
    viewer.screenshot()
    assert n_called[0] == 1
