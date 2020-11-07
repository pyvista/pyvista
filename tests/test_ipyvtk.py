import gc

from ipyvtk_simple.viewer import ViewInteractiveWidget
import pytest

import pyvista


def test_ipyvtk(sphere):
    pl = pyvista.Plotter(notebook=True)
    pl.add_mesh(sphere)
    viewer = pl.show(use_ipyvtk=True, return_viewer=True)
    assert isinstance(viewer, ViewInteractiveWidget)
