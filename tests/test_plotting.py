from subprocess import Popen, PIPE
import os

import numpy as np
import pytest
import vtkInterface as vtki

from vtkInterface import examples
from vtkInterface.plotting import RunningXServer


@pytest.mark.skipif(not RunningXServer(), reason="Requires active X Server")
class TestPlotting(object):

    def test_init(self):
        plotter = vtki.PlotClass()
        assert hasattr(plotter, 'renWin')

    def test_plotarrow(self):
        cent = np.random.random(3)
        direction = np.random.random(3)
        cpos, img = vtki.PlotArrows(cent, direction, off_screen=True, screenshot=True)
        assert np.any(img)

    def test_plotarrows(self):
        cent = np.random.random((100, 3))
        direction = np.random.random((100, 3))
        cpos, img = vtki.PlotArrows(cent, direction, off_screen=True, screenshot=True)
        assert np.any(img)


def test_axes():
    plotter = vtki.PlotClass(off_screen=True)
    plotter.AddAxes()
    plotter.AddMesh(vtki.Sphere())    
    plotter.Plot()
    
