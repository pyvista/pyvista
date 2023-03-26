import os

import pytest

import pyvista as pv
from pyvista.plotting import system_supports_plotting
from pyvista.utilities.misc import PyVistaDeprecationWarning

has_ipyvtklink = True
try:
    from ipyvtklink.viewer import ViewInteractiveWidget
except:  # noqa: E722
    has_ipyvtklink = False


skip_no_plotting = pytest.mark.skipif(
    not system_supports_plotting(), reason="Requires system to support plotting"
)

skip_no_ipyvtk = pytest.mark.skipif(not has_ipyvtklink, reason="Requires IPython package")


@skip_no_ipyvtk
def test_set_jupyter_backend_ipyvtklink():
    try:
        with pytest.warns(PyVistaDeprecationWarning):
            pv.global_theme.jupyter_backend = 'ipyvtklink'
        assert pv.global_theme.jupyter_backend == 'ipyvtklink'
    finally:
        pv.global_theme.jupyter_backend = None


@skip_no_ipyvtk
@skip_no_plotting
def test_ipyvtk(sphere):
    pl = pv.Plotter(notebook=True)
    pl.add_mesh(sphere)
    with pytest.warns(PyVistaDeprecationWarning):
        viewer = pl.show(jupyter_backend='ipyvtklink', return_viewer=True)
    assert isinstance(viewer, ViewInteractiveWidget)


@skip_no_ipyvtk
@skip_no_plotting
def test_ipyvtk_warn(sphere):
    os.environ['SPYDER'] = 'exists'
    with pytest.warns(UserWarning, match='incompatible with Spyder'):
        sphere.plot(notebook=True, jupyter_backend='ipyvtklink')
    del os.environ['SPYDER']
