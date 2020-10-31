"""Module managing examples and toy datasets."""

import os
import time

import numpy as np

import pyvista

# get location of this folder and the example files
dir_path = os.path.dirname(os.path.realpath(__file__))
antfile = os.path.join(dir_path, 'ant.ply')
planefile = os.path.join(dir_path, 'airplane.ply')
hexbeamfile = os.path.join(dir_path, 'hexbeam.vtk')
spherefile = os.path.join(dir_path, 'sphere.ply')
uniformfile = os.path.join(dir_path, 'uniform.vtk')
rectfile = os.path.join(dir_path, 'rectilinear.vtk')
globefile = os.path.join(dir_path, 'globe.vtk')
mapfile = os.path.join(dir_path, '2k_earth_daymap.jpg')
channelsfile = os.path.join(dir_path, 'channels.vti')

# get location of this folder
dir_path = os.path.dirname(os.path.realpath(__file__))


def load_ant():
    """Load ply ant mesh."""
    return pyvista.PolyData(antfile)


def load_airplane():
    """Load ply airplane mesh."""
    return pyvista.PolyData(planefile)


def load_sphere():
    """Load sphere ply mesh."""
    return pyvista.PolyData(spherefile)


def load_uniform():
    """Load a sample uniform grid."""
    return pyvista.UniformGrid(uniformfile)


def load_rectilinear():
    """Load a sample uniform grid."""
    return pyvista.RectilinearGrid(rectfile)

def load_hexbeam():
    """Load a sample UnstructuredGrid."""
    return pyvista.UnstructuredGrid(hexbeamfile)


def load_structured():
    """Load a simple StructuredGrid."""
    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    z = np.sin(r)
    return pyvista.StructuredGrid(x, y, z)

def load_globe():
    """Load a globe source."""
    globe = pyvista.PolyData(globefile)
    globe.textures['2k_earth_daymap'] = load_globe_texture()
    return globe

def load_globe_texture():
    """Load a vtk.vtkTexture that can be applied to the globe source."""
    return pyvista.read_texture(mapfile)


def load_channels():
    """Load a uniform grid of fluvial channels in the subsurface."""
    return pyvista.read(channelsfile)

def plot_ants_plane(off_screen=None, notebook=None):
    """Plot two ants and airplane.

    Demonstrate how to create a plot class to plot multiple meshes while
    adding scalars and text.

    """
    # load and shrink airplane
    airplane = pyvista.PolyData(planefile)
    airplane.points /= 10
    # pts = airplane.points # gets pointer to array
    # pts /= 10  # shrink

    # rotate and translate ant so it is on the plane
    ant = pyvista.PolyData(antfile)
    ant.rotate_x(90)
    ant.translate([90, 60, 15])

    # Make a copy and add another ant
    ant_copy = ant.copy()
    ant_copy.translate([30, 0, -10])

    # Create plotting object
    plotter = pyvista.Plotter(off_screen=off_screen, notebook=notebook)
    plotter.add_mesh(ant, 'r')
    plotter.add_mesh(ant_copy, 'b')

    # Add airplane mesh and make the color equal to the Y position
    plane_scalars = airplane.points[:, 1]
    plotter.add_mesh(airplane, scalars=plane_scalars, stitle='Plane Y\nLocation')
    plotter.add_text('Ants and Plane Example')
    plotter.show()


def beam_example(off_screen=None, notebook=None):
    """Create the beam example."""
    # Load module and example file
    hexfile = hexbeamfile

    # Load Grid
    grid = pyvista.UnstructuredGrid(hexfile)

    # Create fiticious displacements as a function of Z location
    d = grid.points[:, 2]**3/250
    grid.points[:, 1] += d

    # Camera position
    cpos = [(11.915126303095157, 6.11392754955802, 3.6124956735471914),
            (0.0, 0.375, 2.0),
            (-0.42546442225230097, 0.9024244135964158, -0.06789847673314177)]

    try:
        import matplotlib
        cmap = 'bwr'
    except ImportError:
        cmap = None

    # plot this displaced beam
    plotter = pyvista.Plotter(off_screen=off_screen, notebook=notebook)
    plotter.add_mesh(grid, scalars=d, stitle='Y Displacement',
                     rng=[-d.max(), d.max()], cmap=cmap)
    plotter.camera_position = cpos
    plotter.add_text('Static Beam Example')
    cpos = plotter.show()  # store camera position


def plot_wave(fps=30, frequency=1, wavetime=3, interactive=False,
              off_screen=None, notebook=None):
    """Plot a 3D moving wave in a render window.

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

    Return
    ------
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
    sgrid = pyvista.StructuredGrid(X, Y, Z)

    # Get pointer to points
    points = sgrid.points.copy()

    # Start a plotter object and set the scalars to the Z height
    plotter = pyvista.Plotter(off_screen=off_screen, notebook=notebook)
    plotter.add_mesh(sgrid, scalars=Z.ravel())
    plotter.camera_position = cpos
    plotter.show(title='Wave Example', window_size=[800, 600],
                 auto_close=False, interactive_update=True)

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
        plotter.update_coordinates(points, render=False)
        plotter.update_scalars(Z.ravel(), render=False)

        # Render and get time to render
        #rstart = time.time()
        plotter.update()
        # plotter.render()
        #rstop = time.time()

        # time delay
        tpast = time.time() - tlast
        if tpast < tdelay and tpast >= 0:
            time.sleep(tdelay - tpast)

        # get render time and actual FPS
        # rtime = rstop - rstart
        # act_fps = 1 / (time.time() - tlast + 1E-10)
        tlast = time.time()

    # Close movie and delete object
    plotter.close()

    return points


def load_spline():
    """Load an example spline mesh."""
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    points = np.column_stack((x, y, z))
    return pyvista.Spline(points, 1000)


def load_random_hills():
    """Create random hills toy example.

    Uses the parametric random hill function to create hills oriented
    like topography and adds an elevation array.
    """
    mesh = pyvista.ParametricRandomHills()
    return mesh.elevation()


def load_sphere_vectors():
    """Create example sphere with a swirly vector field defined on nodes."""
    sphere = pyvista.Sphere(radius=3.14)

    # make cool swirly pattern
    vectors = np.vstack(
        (
            np.sin(sphere.points[:, 0]),
            np.cos(sphere.points[:, 1]),
            np.cos(sphere.points[:, 2]),
        )
    ).T

    # add and scale
    sphere.vectors = vectors * 0.3
    return sphere
