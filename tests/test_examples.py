from subprocess import Popen, PIPE
import os

import numpy as np
import pytest

import vtkInterface as vtki
from vtkInterface import examples
from vtkInterface.plotting import RunningXServer


@pytest.mark.skipif(not RunningXServer(), reason="Requires active X Server")
def test_docexample_advancedplottingwithnumpy():
    import vtkInterface as vtki
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
    plobj = vtki.PlotClass(off_screen=True)
    plobj.AddArrows(points, direction, 0.5)
    plobj.SetBackground([0, 0, 0]) # RGB set to black
    plobj.Plot(autoclose=False)
    img = plobj.TakeScreenShot()
    assert np.any(img)
    plobj.Close()

@pytest.mark.skipif(not RunningXServer(), reason="Requires active X Server")
def test_creatingagifmovie(tmpdir, off_screen=True):
    if tmpdir:
        filename = str(tmpdir.mkdir("tmpdir").join('wave.gif'))
    else:
        filename = '/tmp/wave.gif'
    import vtkInterface as vtki
    import numpy as np

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
    plobj = vtki.PlotClass(off_screen=off_screen)
    plobj.AddMesh(grid, scalars=z.ravel())
    plobj.Plot(autoclose=False)
    
    # Open a gif
    plobj.OpenGif(filename)
    
    # Update Z and write a frame for each updated position
    nframe = 15
    for phase in np.linspace(0, 2*np.pi, nframe + 1)[:nframe]:
        z = np.sin(r + phase)
        pts[:, -1] = z.ravel()
        plobj.UpdateCoordinates(pts)
        plobj.UpdateScalars(z.ravel())
    
        plobj.WriteFrame()
    
    # Close movie and delete object
    plobj.Close()


@pytest.mark.skipif(not RunningXServer(), reason="Requires active X Server")
def test_show_wave():
    points = examples.ShowWave(wavetime=0.1, off_screen=True)
    assert isinstance(points, np.ndarray)
