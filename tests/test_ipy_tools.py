import sys

import numpy as np
import pytest

import pyvista
from pyvista import QtInteractor, examples
from pyvista.plotting import system_supports_plotting

NO_PLOTTING = NO_PLOTTING

try:
    import PyQt5
    has_pyqt5 = True
except:
    has_pyqt5 = False



@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_ipy_scaled_plotter(qtbot):
    data = examples.load_uniform()
    p = pyvista.ScaledPlotter(show=False)
    p.add_mesh(data)
    p.close()

@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_ipy_orthoganl_slicer(qtbot):
    data = examples.load_uniform()
    tool = pyvista.OrthogonalSlicer(data, show=False)
    g = tool.tool()
    g.widget.update()
    tool.plotter.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_ipy_many_slices_along_axis(qtbot):
    data = examples.load_uniform()
    tool = pyvista.ManySlicesAlongAxis(data, show=False)
    g = tool.tool()
    g.widget.update()
    tool.plotter.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_ipy_threshold(qtbot):
    data = examples.load_uniform()
    tool = pyvista.Threshold(data, show=False)
    g = tool.tool()
    g.widget.update()
    tool.plotter.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_ipy_clip(qtbot):
    data = examples.load_uniform()
    tool = pyvista.Clip(data, show=False)
    g = tool.tool()
    g.widget.update()
    tool.plotter.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_ipy_integrated(qtbot):
    data = examples.load_uniform()
    p = pyvista.ScaledPlotter(show=False)
    p.add_mesh(data)
    slicer = pyvista.OrthogonalSlicer(data, plotter=p)
    many = pyvista.ManySlicesAlongAxis(data, plotter=p)
    thresher = pyvista.Threshold(data, plotter=p)
    clipper = pyvista.Clip(data, plotter=p)
    p.close()

@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_ipy_isocontour(qtbot):
    data = examples.load_uniform()
    tool = pyvista.Isocontour(data, show=False)
    g = tool.tool()
    g.widget.update()
    tool.plotter.close()
