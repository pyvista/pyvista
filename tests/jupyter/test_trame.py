import pytest

import pyvista as pv
from pyvista.plotting import system_supports_plotting

has_trame = True
try:
    from IPython.display import IFrame
    from trame.app import get_server

    from pyvista.trame.ui import get_or_create_viewer
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
async def test_trame_server_launch():
    await pv.set_jupyter_backend('trame')
    server = get_server(name=pv.global_theme.trame.jupyter_server_name)
    assert server.running


@skip_no_trame
@skip_no_plotting
@pytest.mark.asyncio
async def test_trame():
    await pv.set_jupyter_backend('trame')
    server = get_server(name=pv.global_theme.trame.jupyter_server_name)
    assert server.running

    pl = pv.Plotter(notebook=True)
    actor = pl.add_mesh(pv.Cone())
    widget = pl.show()
    assert isinstance(widget, IFrame)

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
