from ipyvtk_simple.viewer import ViewInteractiveWidget

import pyvista


def test_ipyvtk(sphere):
    pl = pyvista.Plotter(notebook=True)
    pl.add_mesh(sphere)
    viewer = pl.show(jupyter_backend='ipyvtk_simple',
                     return_viewer=True)
    assert isinstance(viewer, ViewInteractiveWidget)
