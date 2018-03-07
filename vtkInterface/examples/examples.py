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
    # pts = airplane.GetNumpyPoints() # gets pointer to array
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
    plobj.AddAxes()
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
    plobj.AddAxes()
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
    plobj.AddAxes()
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
