import os

import pytest

import pyvista as pv


has_ipyvtk_simple = True
try:
    from ipyvtk_simple.viewer import ViewInteractiveWidget
except:
    has_ipyvtk_simple = False


skip_no_ipyvtk = pytest.mark.skipif(not has_ipyvtk_simple,
                                    reason="Requires IPython package")

@skip_no_ipyvtk
def test_set_jupyter_backend_ipyvtk_simple():
    pv.set_jupyter_backend('ipyvtk_simple')
    assert pv.rcParams['jupyter_backend'] == 'ipyvtk_simple'
    pv.set_jupyter_backend('panel')


@skip_no_ipyvtk
def test_ipyvtk(sphere):
    pl = pv.Plotter(notebook=True)
    pl.add_mesh(sphere)
    viewer = pl.show(jupyter_backend='ipyvtk_simple',
                     return_viewer=True)
    assert isinstance(viewer, ViewInteractiveWidget)


@skip_no_ipyvtk
def test_ipyvtk_warn(sphere):
    os.environ['SPYDER'] = 'exists'
    with pytest.warns(UserWarning, match='incompatible with Spyder'):
        sphere.plot(notebook=True, jupyter_backend='ipyvtk_simple')
    del os.environ['SPYDER']
