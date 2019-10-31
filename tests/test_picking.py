import os
import sys
from subprocess import PIPE, Popen
from weakref import proxy

import imageio
import numpy as np
import pytest

import pyvista
from pyvista import examples
from pyvista.plotting import system_supports_plotting

NO_PLOTTING = not system_supports_plotting()

# We need an interactive renderer
OFF_SCREEN = False

callback = lambda *args: args[0]


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.parametrize('through', [True, False])
def test_cell_picking(through):
    mesh = examples.load_airplane()
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(mesh)
    plotter.enable_cell_picking(callback=callback, through=through)
    plotter.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
@pytest.mark.parametrize('use_mesh', [True, False])
def test_point_picking(use_mesh):
    mesh = examples.load_airplane()
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(mesh)
    plotter.enable_point_picking(callback=callback, use_mesh=use_mesh)
    plotter.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_path_picking():
    mesh = examples.load_airplane()
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(mesh)
    plotter.enable_path_picking(callback=callback)
    plotter.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_geodesic_picking():
    mesh = examples.load_airplane()
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(mesh)
    plotter.enable_geodesic_picking(callback=callback)
    plotter.close()


@pytest.mark.skipif(NO_PLOTTING, reason="Requires system to support plotting")
def test_horizon_picking():
    mesh = examples.load_airplane()
    plotter = pyvista.Plotter(off_screen=OFF_SCREEN)
    plotter.add_mesh(mesh)
    plotter.enable_horizon_picking(callback=callback)
    plotter.close()
