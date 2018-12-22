from subprocess import Popen, PIPE
import os

import numpy as np
import pytest

import vtki
from vtki import examples
from vtki.plotting import running_xserver


@pytest.mark.skipif(not running_xserver(), reason="Requires X11")
def test_docexample_advancedplottingwithnumpy():
    import vtki
    import numpy as np

    # Make a grid
    x, y, z = np.meshgrid(np.linspace(-5, 5, 20),
                          np.linspace(-5, 5, 20),
                          np.linspace(-5, 5, 5))

    points = np.empty((x.size, 3))
    points[:, 0] = x.ravel('F')
    points[:, 1] = y.ravel('F')
    points[:, 2] = z.ravel('F')

    # Compute a direction for the vector field
    direction = np.sin(points)**3

    # plot using the plotting class
    plotter = vtki.Plotter(off_screen=True)
    plotter.add_arrows(points, direction, 0.5)
    plotter.set_background([0, 0, 0]) # RGB set to black
    plotter.plot(autoclose=False)
    np.any(plotter.screenshot())
    plotter.close()


@pytest.mark.skipif(not running_xserver(), reason="Requires X11")
def test_creatingagifmovie(tmpdir, off_screen=True):
    if tmpdir:
        filename = str(tmpdir.mkdir("tmpdir").join('wave.gif'))
    else:
        filename = '/tmp/wave.gif'

    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    z = np.sin(r)
    
    # Create and structured surface
    grid = vtki.StructuredGrid(x, y, z)
    
    # Make copy of points
    pts = grid.points.copy()
    
    # Start a plotter object and set the scalars to the Z height
    plotter = vtki.Plotter(off_screen=off_screen)
    plotter.add_mesh(grid, scalars=z.ravel())
    plotter.plot(autoclose=False)
    
    # Open a gif
    plotter.open_gif(filename)
    
    # Update Z and write a frame for each updated position
    nframe = 5
    for phase in np.linspace(0, 2*np.pi, nframe + 1)[:nframe]:
        z = np.sin(r + phase)
        pts[:, -1] = z.ravel()
        plotter.update_coordinates(pts)
        plotter.update_scalars(z.ravel())
        plotter.write_frame()
    
    # Close movie and delete object
    plotter.close()


@pytest.mark.skipif(not running_xserver(), reason="Requires X11")
def test_plot_wave():
    points = examples.plot_wave(wavetime=0.1, off_screen=True)
    assert isinstance(points, np.ndarray)


@pytest.mark.skipif(not running_xserver(), reason="Requires X11")
def test_beam_example():
    examples.beam_example(off_screen=True)


@pytest.mark.skipif(not running_xserver(), reason="Requires X11")
def test_plot_ants_plane():
    examples.plot_ants_plane(off_screen=True)


def test_load_ant():
    """ Load ply ant mesh """
    mesh = examples.load_ant()
    assert mesh.n_points


def test_load_airplane():
    """ Load ply airplane mesh """
    mesh = examples.load_airplane()
    assert mesh.n_points


def test_load_sphere():
    """ Loads sphere ply mesh """
    mesh = examples.load_sphere()
    assert mesh.n_points
