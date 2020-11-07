from ipyvtk_simple.viewer import ViewInteractiveWidget
import pytest

import pyvista


def test_ipyvtk(sphere):
    pl = pyvista.Plotter(notebook=True)
    pl.add_mesh(sphere)
    viewer = pl.show(use_ipyvtk=True, return_viewer=True)
    assert isinstance(viewer, ViewInteractiveWidget)


def test_ipyvtk_sub_render_fail():
    pl = pyvista.Plotter(notebook=True, shape=(2, 1))
    with pytest.raises(NotImplementedError):
        pl.show(use_ipyvtk=True)
