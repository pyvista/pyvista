import numpy as np
import pytest

import vtkInterface as vtki
from vtkInterface import examples
from vtkInterface.plotting import RunningXServer

def test_merge():
    from vtkInterface import examples
    beamA = vtki.UnstructuredGrid(examples.hexbeamfile)
    beamB = beamA.Copy()
    beamB.points[:, 1] += 1
    beamA.Merge(beamB)


@pytest.mark.skipif(not RunningXServer(), reason="Requires active X Server")
def test_struct_example():
    # Make data
    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    z = np.sin(r)

    # create and plot structured grid
    grid = vtki.StructuredGrid(x, y, z)
    cpos = grid.Plot(off_screen=True)  # basic plot
    assert isinstance(cpos, list)

    # Plot mean curvature
    cpos_curv = grid.PlotCurvature(off_screen=True)
    assert isinstance(cpos_curv, list)
