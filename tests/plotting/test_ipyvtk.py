from ipyvtk_simple.viewer import ViewInteractiveWidget

import pyvista


def test_ipyvtk(sphere):
    pl = pyvista.Plotter(notebook=True)
    pl.add_mesh(sphere)
    viewer = pl.show(use_ipyvtk=True, return_viewer=True)
    assert isinstance(viewer, ViewInteractiveWidget)


def test_ipyvtk_sub_render(sphere, cube):
    pl = pyvista.Plotter(notebook=True, shape=(2, 1))
    pl.add_mesh(sphere)
    plotter.subplot(0, 1)
    pl.add_mesh()
    breakpoint()
    viewer = pl.show(use_ipyvtk=True, return_viewer=True)
    assert isinstance(viewer, ViewInteractiveWidget)
