import pytest

import pyvista as pv
from pyvista.plotting import system_supports_plotting

has_trame = True
try:
    from IPython.display import IFrame

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
async def test_ipyvtk(sphere):
    await pv.set_jupyter_backend('trame')
    pl = pv.Plotter(notebook=True)
    pl.add_mesh(sphere)
    viewer = pl.show()
    assert isinstance(viewer, IFrame)
