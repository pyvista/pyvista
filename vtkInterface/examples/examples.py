import inspect
import sys
import os
import time

import vtkInterface
import numpy as np

# get location of this folder and the example files
dir_path = os.path.dirname(os.path.realpath(__file__))
antfile = os.path.join(dir_path, 'ant.ply')
planefile = os.path.join(dir_path, 'airplane.ply')
hexbeamfile = os.path.join(dir_path, 'hexbeam.vtk')
spherefile = os.path.join(dir_path, 'sphere.ply')


# get location of this folder
dir_path = os.path.dirname(os.path.realpath(__file__))


def LoadAnt():
    """ Load ply ant mesh """
    return vtkInterface.PolyData(antfile)


def LoadAirplane():
    """ Load ply airplane mesh """
    return vtkInterface.PolyData(planefile)


def LoadSphere():
    """ Loads sphere ply mesh """
    return vtkInterface.PolyData(spherefile)


def PlotSphere():
    """ Plot a white airplane """
    sphere = vtkInterface.PolyData(spherefile)
    sphere.Plot()


def PlotAirplane():
    """ Plot a white airplane """
    airplane = vtkInterface.PolyData(planefile)
    airplane.Plot()


def PlotAnt():
    """ Plot a red ant in wireframe"""
    ant = vtkInterface.PolyData(antfile)
    ant.Plot(color='r', style='wireframe')


def PlotAntsPlane():
    """
    Demonstrate how to create a plot class to plot multiple meshes while
    adding scalars and text.
    Plot two ants and airplane
    """

    # load and shrink airplane
    airplane = vtkInterface.PolyData(planefile)
    airplane.points /= 10
    # pts = airplane.points # gets pointer to array
    # pts /= 10  # shrink

    # rotate and translate ant so it is on the plane
    ant = vtkInterface.PolyData(antfile)
    ant.RotateX(90)
    ant.Translate([90, 60, 15])

    # Make a copy and add another ant
    ant_copy = ant.Copy()
    ant_copy.Translate([30, 0, -10])

    # Create plotting object
    plobj = vtkInterface.PlotClass()
    plobj.AddMesh(ant, 'r')
    plobj.AddMesh(ant_copy, 'b')

    # Add airplane mesh and make the color equal to the Y position
    plane_scalars = airplane.points[:, 1]
    plobj.AddMesh(airplane, scalars=plane_scalars, stitle='Plane Y\nLocation')
    plobj.AddText('Ants and Plane Example')
    plobj.Plot()


def BeamExample():
    # Load module and example file

    hexfile = hexbeamfile

    # Load Grid
    grid = vtkInterface.UnstructuredGrid(hexfile)

    # Create fiticious displacements as a function of Z location
    pts = grid.GetNumpyPoints(deep=True)
    d = np.zeros_like(pts)
    d[:, 1] = pts[:, 2]**3/250

    # Displace original grid
    grid.SetNumpyPoints(pts + d)

    # Camera position
    cpos = [(11.915126303095157, 6.11392754955802, 3.6124956735471914),
            (0.0, 0.375, 2.0),
            (-0.42546442225230097, 0.9024244135964158, -0.06789847673314177)]

    try:
        import matplotlib
        colormap = 'bwr'
    except ImportError:
        colormap = None

    # plot this displaced beam
    plobj = vtkInterface.PlotClass()
    plobj.AddMesh(grid, scalars=d[:, 1], stitle='Y Displacement',
                  rng=[-d.max(), d.max()], colormap=colormap)
    plobj.SetCameraPosition(cpos)
    plobj.AddText('Static Beam Example')
    cpos = plobj.Plot(autoclose=False)  # store camera position
    # plobj.TakeScreenShot('beam.png')
    plobj.Close()

    # Animate plot
    plobj = vtkInterface.PlotClass()
    plobj.AddMesh(grid, scalars=d[:, 1], stitle='Y Displacement', showedges=True,
                  rng=[-d.max(), d.max()], interpolatebeforemap=True,
                  colormap=colormap)
    plobj.SetCameraPosition(cpos)
    plobj.AddText('Beam Animation Example')
    plobj.Plot(interactive=False, autoclose=False, window_size=[800, 600])

    #plobj.OpenMovie('beam.mp4')
#    plobj.OpenGif('beam.gif')
    for phase in np.linspace(0, 4*np.pi, 100):
        plobj.UpdateCoordinates(pts + d*np.cos(phase), render=False)
        plobj.UpdateScalars(d[:, 1]*np.cos(phase), render=False)
        plobj.Render()
#        plobj.WriteFrame()
        time.sleep(0.01)

    plobj.Close()

    # Animate plot as a wireframe
    plobj = vtkInterface.PlotClass()
    plobj.AddMesh(grid, scalars=d[:, 1], stitle='Y Displacement',
                  showedges=True, rng=[-d.max(), d.max()], colormap=colormap,
                  interpolatebeforemap=True, style='wireframe')
    plobj.SetCameraPosition(cpos)
    plobj.AddText('Beam Animation Example 2')
    plobj.Plot(interactive=False, autoclose=False, window_size=[800, 600])

    # plobj.OpenMovie('beam.mp4')
    # plobj.OpenGif('beam_wireframe.gif')
    for phase in np.linspace(0, 4*np.pi, 100):
        plobj.UpdateCoordinates(pts + d*np.cos(phase), render=False)
        plobj.UpdateScalars(d[:, 1]*np.cos(phase), render=False)
        plobj.Render()
#        plobj.WriteFrame()
        time.sleep(0.01)

    plobj.Close()


def ShowWave(fps=30, frequency=1, wavetime=3, interactive=False,
             off_screen=False):
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
        Allows the user to set the camera position before the start of the
        wave movement.  Default False.

    off_screen : bool, optional
        Enables off screen rendering when True.  Used for automated testing.
        Disabled by default.

    Returns
    -------
    points : np.ndarray
        Position of points at last frame.

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
    points = sgrid.points.copy()

    # Start a plotter object and set the scalars to the Z height
    plotter = vtkInterface.PlotClass(off_screen=off_screen)
    plotter.AddMesh(sgrid, scalars=Z.ravel())
    plotter.SetCameraPosition(cpos)
    plotter.Plot(title='Wave Example', window_size=[800, 600],
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
        points[:, -1] = Z.ravel()

        # update plotting object, but don't automatically render
        plotter.UpdateCoordinates(points, render=False)
        plotter.UpdateScalars(Z.ravel(), render=False)

        # Render and get time to render
        rstart = time.time()
        plotter.Render()
        rstop = time.time()

        # compute time delay
        tpast = time.time() - tlast
        if tpast < tdelay and tpast >= 0:
            time.sleep(tdelay - tpast)

        # Print render time and actual FPS
        rtime = rstop - rstart
        act_fps = 1 / (time.time() - tlast + 1E-10)
        # print('Actual FPS: %.2f' % act_fps)
        tlast = time.time()

    # Close movie and delete object
    plotter.Close()

    return points


def RunAll():
    """ Runs all the functions within this module """
    testfunctions = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(obj) and name != 'RunAll':
            testfunctions.append(obj)

    # run all the functions
    for f in testfunctions:
        print('Running %s' % str(f))
        f()
