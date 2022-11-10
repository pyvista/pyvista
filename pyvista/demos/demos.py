"""Demos to show off the functionality of PyVista."""

import time

import numpy as np

import pyvista as pv
from pyvista import examples

from .logo import text_3d


def glyphs(grid_sz=3):
    """Create several parametric supertoroids using VTK's glyph table functionality.

    Parameters
    ----------
    grid_sz : int, optional
        Create ``grid_sz x grid_sz`` supertoroids.

    Returns
    -------
    pyvista.PolyData
        Mesh of supertoroids.

    See Also
    --------
    plot_glyphs

    Examples
    --------
    >>> from pyvista import demos
    >>> mesh = demos.glyphs()
    >>> mesh.plot()

    """
    n = 10
    values = np.arange(n)  # values for scalars to look up glyphs by

    # taken from:
    rng = np.random.default_rng()
    params = rng.uniform(0.5, 2, size=(n, 2))  # (n1, n2) parameters for the toroids

    geoms = [pv.ParametricSuperToroid(n1=n1, n2=n2) for n1, n2 in params]

    # get dataset where to put glyphs
    grid_sz = float(grid_sz)
    x, y, z = np.mgrid[:grid_sz, :grid_sz, :grid_sz]
    mesh = pv.StructuredGrid(x, y, z)

    # add random scalars
    rng_int = rng.integers(0, n, size=x.size)
    mesh.point_data['scalars'] = rng_int

    # construct the glyphs on top of the mesh; don't scale by scalars now
    return mesh.glyph(
        geom=geoms, indices=values, scale=False, factor=0.3, rng=(0, n - 1), orient=False
    )


def plot_glyphs(grid_sz=3, **kwargs):
    """Plot several parametric supertoroids using VTK's glyph table functionality.

    Parameters
    ----------
    grid_sz : int, optional
        Create ``grid_sz x grid_sz`` supertoroids.

    **kwargs : dict, optional
        All additional keyword arguments will be passed to
        :func:`pyvista.Plotter.add_mesh`.

    Returns
    -------
    various
        See :func:`show <pyvista.Plotter.show>`.

    Examples
    --------
    >>> from pyvista import demos
    >>> demos.plot_glyphs()

    """
    # construct the glyphs on top of the mesh; don't scale by scalars now
    mesh = glyphs(grid_sz)

    kwargs.setdefault('specular', 1)
    kwargs.setdefault('specular_power', 15)
    kwargs.setdefault('smooth_shading', True)

    # create plotter and add our glyphs with some nontrivial lighting
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_scalar_bar=False, **kwargs)
    return plotter.show()


def orientation_cube():
    """Return a dictionary containing the meshes composing an orientation cube.

    Returns
    -------
    dict
        Dictionary containing the meshes composing an orientation cube.

    Examples
    --------
    Load the cube mesh and plot it

    >>> import pyvista
    >>> from pyvista import demos
    >>> ocube = demos.orientation_cube()
    >>> pl = pyvista.Plotter()
    >>> _ = pl.add_mesh(ocube['cube'], show_edges=True)
    >>> _ = pl.add_mesh(ocube['x_p'], color='blue')
    >>> _ = pl.add_mesh(ocube['x_n'], color='blue')
    >>> _ = pl.add_mesh(ocube['y_p'], color='green')
    >>> _ = pl.add_mesh(ocube['y_n'], color='green')
    >>> _ = pl.add_mesh(ocube['z_p'], color='red')
    >>> _ = pl.add_mesh(ocube['z_n'], color='red')
    >>> pl.show_axes()
    >>> pl.show()

    """
    cube = pv.Cube()

    x_p = text_3d('X+', depth=0.2)
    x_p.points *= 0.45
    x_p.rotate_y(90, inplace=True)
    x_p.rotate_x(90, inplace=True)
    x_p.translate(-np.array(x_p.center), inplace=True)
    x_p.translate([0.5, 0, 0], inplace=True)
    # x_p.point_data['mesh'] = 1

    x_n = text_3d('X-', depth=0.2)
    x_n.points *= 0.45
    x_n.rotate_y(90, inplace=True)
    x_n.rotate_x(90, inplace=True)
    x_n.rotate_z(180, inplace=True)
    x_n.translate(-np.array(x_n.center), inplace=True)
    x_n.translate([-0.5, 0, 0], inplace=True)
    # x_n.point_data['mesh'] = 2

    y_p = text_3d('Y+', depth=0.2)
    y_p.points *= 0.45
    y_p.rotate_x(90, inplace=True)
    y_p.rotate_z(180, inplace=True)
    y_p.translate(-np.array(y_p.center), inplace=True)
    y_p.translate([0, 0.5, 0], inplace=True)
    # y_p.point_data['mesh'] = 3

    y_n = text_3d('Y-', depth=0.2)
    y_n.points *= 0.45
    y_n.rotate_x(90, inplace=True)
    y_n.translate(-np.array(y_n.center), inplace=True)
    y_n.translate([0, -0.5, 0], inplace=True)
    # y_n.point_data['mesh'] = 4

    z_p = text_3d('Z+', depth=0.2)
    z_p.points *= 0.45
    z_p.rotate_z(90, inplace=True)
    z_p.translate(-np.array(z_p.center), inplace=True)
    z_p.translate([0, 0, 0.5], inplace=True)
    # z_p.point_data['mesh'] = 5

    z_n = text_3d('Z-', depth=0.2)
    z_n.points *= 0.45
    z_n.rotate_x(180, inplace=True)
    z_n.translate(-np.array(z_n.center), inplace=True)
    z_n.translate([0, 0, -0.5], inplace=True)

    return {
        'cube': cube,
        'x_p': x_p,
        'x_n': x_n,
        'y_p': y_p,
        'y_n': y_n,
        'z_p': z_p,
        'z_n': z_n,
    }


def orientation_plotter():
    """Return a plotter containing the orientation cube.

    Returns
    -------
    pyvista.Plotter
        Orientation cube plotter.

    Examples
    --------
    >>> from pyvista import demos
    >>> plotter = demos.orientation_plotter()
    >>> plotter.show()

    """
    ocube = orientation_cube()
    pl = pv.Plotter()
    pl.add_mesh(ocube['cube'], show_edges=True)
    pl.add_mesh(ocube['x_p'], color='blue')
    pl.add_mesh(ocube['x_n'], color='blue')
    pl.add_mesh(ocube['y_p'], color='green')
    pl.add_mesh(ocube['y_n'], color='green')
    pl.add_mesh(ocube['z_p'], color='red')
    pl.add_mesh(ocube['z_n'], color='red')
    pl.show_axes()
    return pl


def plot_wave(fps=30, frequency=1, wavetime=3, interactive=False, notebook=None):
    """Plot a 3D moving wave in a render window.

    Parameters
    ----------
    fps : int, optional
        Maximum frames per second to display.  Defaults to 30.

    frequency : float, optional
        Wave cycles per second.  Defaults to 1 Hz.

    wavetime : float, optional
        The desired total display time in seconds.  Defaults to 3 seconds.

    interactive : bool, optional
        Allows the user to set the camera position before the start of the
        wave movement.  Default ``False``.

    notebook : bool, optional
        When ``True``, the resulting plot is placed inline a jupyter
        notebook.  Assumes a jupyter console is active.

    Returns
    -------
    numpy.ndarray
        Position of points at last frame.

    Examples
    --------
    >>> from pyvista import demos
    >>> out = demos.plot_wave()

    """
    # camera position
    cpos = [
        (6.879481857604187, -32.143727535933195, 23.05622921691103),
        (-0.2336056403734026, -0.6960083534590372, -0.7226721553894022),
        (-0.008900669873416645, 0.6018246347860926, 0.7985786667826725),
    ]

    # Make data
    X = np.arange(-10, 10, 0.25)
    Y = np.arange(-10, 10, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

    # Create and plot structured grid
    sgrid = pv.StructuredGrid(X, Y, Z)

    # Get pointer to points
    points = sgrid.points.copy()
    mesh = sgrid.extract_surface()

    # Start a plotter object and set the scalars to the Z height
    plotter = pv.Plotter(notebook=notebook)
    plotter.add_mesh(mesh, scalars=Z.ravel(), show_scalar_bar=False, smooth_shading=True)
    plotter.camera_position = cpos
    plotter.show(
        title='Wave Example', window_size=[800, 600], auto_close=False, interactive_update=True
    )

    # Update Z and display a frame for each updated position
    tdelay = 1.0 / fps
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
        mesh.compute_normals(inplace=True)

        # Render and get time to render
        plotter.update()

        # time delay
        tpast = time.time() - tlast
        if tpast < tdelay and tpast >= 0 and not plotter.off_screen:
            time.sleep(tdelay - tpast)

        # store when rendering complete
        tlast = time.time()

    # Close movie and delete object
    plotter.close()
    return points


def plot_ants_plane(notebook=None):
    """Plot two ants and airplane.

    Demonstrate how to create a plot class to plot multiple meshes while
    adding scalars and text.

    This example plots the following:

    .. code:: python

       >>> import pyvista
       >>> from pyvista import examples

       Load and shrink airplane

       >>> airplane = examples.load_airplane()
       >>> airplane.points /= 10

       Rotate and translate ant so it is on the plane.

       >>> ant = examples.load_ant()
       >>> _ = ant.rotate_x(90, inplace=True)
       >>> _ = ant.translate([90, 60, 15], inplace=True)

       Make a copy and add another ant.

       >>> ant_copy = ant.translate([30, 0, -10], inplace=False)

       Create plotting object.

       >>> plotter = pyvista.Plotter()
       >>> _ = plotter.add_mesh(ant, 'r')
       >>> _ = plotter.add_mesh(ant_copy, 'b')

       Add airplane mesh and make the color equal to the Y position.

       >>> plane_scalars = airplane.points[:, 1]
       >>> _ = plotter.add_mesh(airplane, scalars=plane_scalars,
       ...                      scalar_bar_args={'title': 'Plane Y Location'})
       >>> _ = plotter.add_text('Ants and Plane Example')
       >>> plotter.show()

    Parameters
    ----------
    notebook : bool, optional
        When ``True``, the resulting plot is placed inline a jupyter
        notebook.  Assumes a jupyter console is active.

    Examples
    --------
    >>> from pyvista import demos
    >>> demos.plot_ants_plane()

    """
    # load and shrink airplane
    airplane = examples.load_airplane()
    airplane.points /= 10

    # rotate and translate ant so it is on the plane
    ant = examples.load_ant()
    ant.rotate_x(90, inplace=True)
    ant.translate([90, 60, 15], inplace=True)

    # Make a copy and add another ant
    ant_copy = ant.copy()
    ant_copy.translate([30, 0, -10], inplace=True)

    # Create plotting object
    plotter = pv.Plotter(notebook=notebook)
    plotter.add_mesh(ant, 'r')
    plotter.add_mesh(ant_copy, 'b')

    # Add airplane mesh and make the color equal to the Y position
    plane_scalars = airplane.points[:, 1]
    plotter.add_mesh(
        airplane, scalars=plane_scalars, scalar_bar_args={'title': 'Plane Y\nLocation'}
    )
    plotter.add_text('Ants and Plane Example')
    plotter.show()


def plot_beam(notebook=None):
    """Plot a beam with displacement.

    Parameters
    ----------
    notebook : bool, optional
        When ``True``, the resulting plot is placed inline a jupyter
        notebook.  Assumes a jupyter console is active.

    Examples
    --------
    >>> from pyvista import demos
    >>> demos.plot_beam()

    """
    # Create fiticious displacements as a function of Z location
    grid = examples.load_hexbeam()
    d = grid.points[:, 2] ** 3 / 250
    grid.points[:, 1] += d

    # Camera position
    cpos = [
        (11.915126303095157, 6.11392754955802, 3.6124956735471914),
        (0.0, 0.375, 2.0),
        (-0.42546442225230097, 0.9024244135964158, -0.06789847673314177),
    ]

    try:
        import matplotlib  # noqa

        cmap = 'bwr'
    except ImportError:  # pragma: no cover
        cmap = None

    # plot this displaced beam
    plotter = pv.Plotter(notebook=notebook)
    plotter.add_mesh(
        grid,
        scalars=d,
        scalar_bar_args={'title': 'Y Displacement'},
        rng=[-d.max(), d.max()],
        cmap=cmap,
    )
    plotter.camera_position = cpos
    plotter.add_text('Static Beam Example')
    plotter.show()


def plot_datasets(dataset_type=None):
    """Plot the pyvista dataset types.

    This demo plots the following PyVista dataset types:

    * :class:`pyvista.PolyData`
    * :class:`pyvista.UnstructuredGrid`
    * :class:`pyvista.UniformGrid`
    * :class:`pyvista.RectilinearGrid`
    * :class:`pyvista.StructuredGrid`

    Parameters
    ----------
    dataset_type : str, optional
        If set, plot just that dataset.  Must be one of the following:

        * ``'PolyData'``
        * ``'UnstructuredGrid'``
        * ``'UniformGrid'``
        * ``'RectilinearGrid'``
        * ``'StructuredGrid'``

    Examples
    --------
    >>> from pyvista import demos
    >>> demos.plot_datasets()

    """
    allowable_types = [
        'PolyData',
        'UnstructuredGrid',
        'UniformGrid',
        'RectilinearGrid',
        'StructuredGrid',
    ]
    if dataset_type is not None:
        if dataset_type not in allowable_types:
            raise ValueError(
                f'Invalid dataset_type {dataset_type}.  Must be one '
                f'of the following: {allowable_types}'
            )

    ###########################################################################
    # uniform grid
    image = pv.UniformGrid(dimensions=(6, 6, 1))
    image.spacing = (3, 2, 1)

    ###########################################################################
    # RectilinearGrid
    xrng = np.array([0, 0.3, 1, 4, 5, 6, 6.2, 6.6])
    yrng = np.linspace(-2, 2, 5)
    zrng = [1]
    rec_grid = pv.RectilinearGrid(xrng, yrng, zrng)

    ###########################################################################
    # structured grid
    ang = np.linspace(0, np.pi / 2, 10)
    r = np.linspace(6, 10, 8)
    z = [0]
    ang, r, z = np.meshgrid(ang, r, z)

    x = r * np.sin(ang)
    y = r * np.cos(ang)

    struct_grid = pv.StructuredGrid(x[::-1], y[::-1], z[::-1])

    ###########################################################################
    # polydata
    points = pv.PolyData([[1.0, 2.0, 2.0], [2.0, 2.0, 2.0]])

    line = pv.Line()
    line.points += np.array((2, 0, 0))
    line.clear_data()

    tri = pv.Triangle()
    tri.points += np.array([0, 1, 0])
    circ = pv.Circle()
    circ.points += np.array([1.5, 1.5, 0])

    poly = tri + circ

    ###########################################################################
    # unstructuredgrid
    pyr = pv.Pyramid()
    pyr.points *= 0.7
    cube = pv.Cube(center=(2, 0, 0))
    ugrid = circ + pyr + cube + tri

    if dataset_type is not None:
        pl = pv.Plotter()
    else:
        pl = pv.Plotter(shape='3/2')

    # polydata
    if dataset_type is None:
        pl.subplot(0)
        pl.add_text('4. PolyData')
    if dataset_type in [None, 'PolyData']:
        pl.add_points(points, point_size=20)
        pl.add_mesh(line, line_width=5)
        pl.add_mesh(poly)
        pl.add_mesh(poly.extract_all_edges(), line_width=2, color='k')

    # unstructuredgrid
    if dataset_type is None:
        pl.subplot(1)
        pl.add_text('5. UnstructuredGrid')
    if dataset_type in [None, 'UnstructuredGrid']:
        pl.add_mesh(ugrid)
        pl.add_mesh(ugrid.extract_all_edges(), line_width=2, color='k')

    # UniformGrid
    if dataset_type is None:
        pl.subplot(2)
        pl.add_text('1. UniformGrid')
    if dataset_type in [None, 'UniformGrid']:
        pl.add_mesh(image)
        pl.add_mesh(image.extract_all_edges(), color='k', style='wireframe', line_width=2)
        pl.camera_position = 'xy'

    # RectilinearGrid
    if dataset_type is None:
        pl.subplot(3)
        pl.add_text('2. RectilinearGrid')
    if dataset_type in [None, 'RectilinearGrid']:
        pl.add_mesh(rec_grid)
        pl.add_mesh(rec_grid.extract_all_edges(), color='k', style='wireframe', line_width=2)
        pl.camera_position = 'xy'

    # StructuredGrid
    if dataset_type is None:
        pl.subplot(4)
        pl.add_text('3. StructuredGrid')
    if dataset_type in [None, 'StructuredGrid']:
        pl.add_mesh(struct_grid)
        pl.add_mesh(struct_grid.extract_all_edges(), color='k', style='wireframe', line_width=2)
        pl.camera_position = 'xy'

    pl.show()
