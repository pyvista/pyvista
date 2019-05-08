import sys

import numpy as np
import pytest

import vista
from vista import QtInteractor, examples
from vista.plotting import system_supports_plotting

try:
    import PyQt5
    has_pyqt5 = True
except:
    has_pyqt5 = False



@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_ipy_scaled_plotter(qtbot):
    data = examples.load_uniform()
    p = vista.ScaledPlotter(show=False)
    p.add_mesh(data)
    p.close()

@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_ipy_orthoganl_slicer(qtbot):
    data = examples.load_uniform()
    tool = vista.OrthogonalSlicer(data, show=False)
    g = tool.tool()
    g.widget.update()
    tool.plotter.close()


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_ipy_many_slices_along_axis(qtbot):
    data = examples.load_uniform()
    tool = vista.ManySlicesAlongAxis(data, show=False)
    g = tool.tool()
    g.widget.update()
    tool.plotter.close()


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_ipy_threshold(qtbot):
    data = examples.load_uniform()
    tool = vista.Threshold(data, show=False)
    g = tool.tool()
    g.widget.update()
    tool.plotter.close()


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_ipy_clip(qtbot):
    data = examples.load_uniform()
    tool = vista.Clip(data, show=False)
    g = tool.tool()
    g.widget.update()
    tool.plotter.close()


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_ipy_integrated(qtbot):
    data = examples.load_uniform()
    p = vista.ScaledPlotter(show=False)
    p.add_mesh(data)
    slicer = vista.OrthogonalSlicer(data, plotter=p)
    many = vista.ManySlicesAlongAxis(data, plotter=p)
    thresher = vista.Threshold(data, plotter=p)
    clipper = vista.Clip(data, plotter=p)
    p.close()

@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
@pytest.mark.skipif(not has_pyqt5, reason="requires pyqt5")
def test_ipy_isocontour(qtbot):
    data = examples.load_uniform()
    tool = vista.Isocontour(data, show=False)
    g = tool.tool()
    g.widget.update()
    tool.plotter.close()
