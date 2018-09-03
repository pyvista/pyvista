"""
Tests if vtk and numpy have been loaded
"""

import time
import numpy as np
import vtkInterface


def ShowWave(fps=30, frequency=1, wavetime=3, interactive=False):
    """

    Plot a 3D moving wave in a render window.

    Parameters
    ----------
    fps : int, optional
        Maximum frames per second to display.  Defaults to 30.

    frequency: float, optional
        Wave cycles per second.  Defaults to 1

    wavetime : float, optional
        The desired total display time in seconds.  Defaults to 3 seconds.

    interactive: bool, optional
        Allows the user to set the camera position before the start of the wave
        movement.  Default False.

    """
    # camera position
    cpos = [(6.879481857604187, -32.143727535933195, 23.05622921691103),
            (-0.2336056403734026, -0.6960083534590372, -0.7226721553894022),
            (-0.008900669873416645, 0.6018246347860926, 0.7985786667826725)]

    # Make data
    X = np.arange(-10, 10, 0.25)
    Y = np.arange(-10, 10, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

    # Create and plot structured grid
    sgrid = vtkInterface.StructuredGrid(X, Y, Z)

    # Get pointer to points
    pts = sgrid.GetNumpyPoints(deep=True)

    # Start a plotter object and set the scalars to the Z height
    plobj = vtkInterface.PlotClass()
    plobj.AddMesh(sgrid, scalars=Z.ravel())
    plobj.SetCameraPosition(cpos)
    plobj.Plot(title='Wave Example', window_size=[800, 600],
               autoclose=False, interactive=interactive)

    # Update Z and display a frame for each updated position
    tdelay = 1. / fps
    tlast = time.time()
    tstart = time.time()
    while time.time() - tstart < wavetime:
        # get phase from start
        telap = time.time() - tstart
        phase = telap * 2 * np.pi * frequency
        Z = np.sin(R + phase)
        pts[:, -1] = Z.ravel()

        # update plotting object, but don't automatically render
        plobj.UpdateCoordinates(pts, render=False)
        plobj.UpdateScalars(Z.ravel(), render=False)

        # Render and get time to render
        rstart = time.time()
        plobj.Render()
        rstop = time.time()

        # compute time delay
        tpast = time.time() - tlast
        if tpast < tdelay and tpast >= 0:
            time.sleep(tdelay - tpast)

        # Print render time and actual FPS
        rtime = rstop - rstart
        act_fps = 1 / (time.time() - tlast + 1E-10)
        print(
            'Frame render time: {:f} sec.  Actual FPS: {:.2f}'.format(
                rtime, act_fps))
        tlast = time.time()

    # Close movie and delete object
    plobj.Close()
