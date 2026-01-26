"""Demos to show off the functionality of PyVista."""

from __future__ import annotations

import time

import numpy as np

import pyvista as pv
from pyvista import examples
from pyvista._deprecate_positional_args import _deprecate_positional_args

from .logo import text_3d


def glyphs(grid_sz=3):
    """Create several parametric supertoroids using VTK's glyph table functionality.

    Parameters
    ----------
    grid_sz : int, default: 3
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
    # Seed rng for reproducible plots
    rng = np.random.default_rng(seed=0)

    n = 10
    values = np.arange(n)  # values for scalars to look up glyphs by

    # taken from:
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
        geom=geoms,
        indices=values,
        scale=False,
        factor=0.3,
        rng=(0, n - 1),
        orient=False,
    )


def plot_glyphs(grid_sz=3, **kwargs):
    """Plot several parametric supertoroids using VTK's glyph table functionality.

    Parameters
    ----------
    grid_sz : int, default: 3
        Create ``grid_sz x grid_sz`` supertoroids.

    **kwargs : dict, optional
        All additional keyword arguments will be passed to
        :func:`pyvista.Plotter.add_mesh`.

    Returns
    -------
    output : list | np.ndarray | ipywidgets.Widget
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
    pl = pv.Plotter()
    pl.add_mesh(mesh, show_scalar_bar=False, **kwargs)
    return pl.show()


def orientation_cube():
    """Return a dictionary containing the meshes composing an orientation cube.

    Returns
    -------
    dict
        Dictionary containing the meshes composing an orientation cube.

    Examples
    --------
    Load the cube mesh and plot it

    >>> import pyvista as pv
    >>> from pyvista import demos
    >>> ocube = demos.orientation_cube()
    >>> pl = pv.Plotter()
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
    >>> pl = demos.orientation_plotter()
    >>> pl.show()

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
    pl.show_axes()  # type: ignore[call-arg]
    return pl


@_deprecate_positional_args
def plot_wave(fps=30, frequency=1, wavetime=3, notebook=None):  # noqa: PLR0917
    """Plot a 3D moving wave in a render window.

    Parameters
    ----------
    fps : int, default: 30
        Maximum frames per second to display.

    frequency : float, default: 1.0
        Wave cycles per second (Hz).

    wavetime : float, default: 3.0
        The desired total display time in seconds.

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
    cpos = pv.CameraPosition(
        position=(6.879481857604187, -32.143727535933195, 23.05622921691103),
        focal_point=(-0.2336056403734026, -0.6960083534590372, -0.7226721553894022),
        viewup=(-0.008900669873416645, 0.6018246347860926, 0.7985786667826725),
    )

    # Make data
    X = np.arange(-10, 10, 0.25)
    Y = np.arange(-10, 10, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

    # Create and plot structured grid
    sgrid = pv.StructuredGrid(X, Y, Z)

    mesh = sgrid.extract_surface(algorithm='geometry', pass_pointid=False, pass_cellid=False)
    mesh['Height'] = Z.ravel()

    # Start a plotter object and set the scalars to the Z height
    pl = pv.Plotter(notebook=notebook)
    pl.add_mesh(mesh, scalars='Height', show_scalar_bar=False, smooth_shading=True)
    pl.camera_position = cpos
    pl.show(
        title='Wave Example',
        window_size=[800, 600],
        auto_close=False,
        interactive_update=True,
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
        mesh.points[:, -1] = Z.ravel()  # type: ignore[index]
        mesh['Height'] = Z.ravel()

        mesh.compute_normals(inplace=True)

        # Render and get time to render
        pl.update()

        # time delay
        tpast = time.time() - tlast
        if tpast < tdelay and tpast >= 0 and not pl.off_screen:
            time.sleep(tdelay - tpast)

        # store when rendering complete
        tlast = time.time()

    # Close movie and delete object
    pl.close()
    return mesh.points


def plot_ants_plane(notebook=None):
    """Plot two ants and airplane.

    Demonstrate how to create a plot class to plot multiple meshes while
    adding scalars and text.

    This example plots the following:

    .. code-block:: python

       >>> import pyvista as pv
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

       >>> pl = pv.Plotter()
       >>> _ = pl.add_mesh(ant, color='r')
       >>> _ = pl.add_mesh(ant_copy, color='b')

       Add airplane mesh and make the color equal to the Y position.

       >>> plane_scalars = airplane.points[:, 1]
       >>> _ = pl.add_mesh(
       ...     airplane,
       ...     scalars=plane_scalars,
       ...     scalar_bar_args={'title': 'Plane Y Location'},
       ... )
       >>> _ = pl.add_text('Ants and Plane Example')
       >>> pl.show()

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
    pl = pv.Plotter(notebook=notebook)
    pl.add_mesh(ant, color='r')
    pl.add_mesh(ant_copy, color='b')

    # Add airplane mesh and make the color equal to the Y position
    plane_scalars = airplane.points[:, 1]
    pl.add_mesh(
        airplane,
        scalars=plane_scalars,
        scalar_bar_args={'title': 'Plane Y\nLocation'},
    )
    pl.add_text('Ants and Plane Example')
    pl.show()


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
    cpos = pv.CameraPosition(
        position=(11.915126303095157, 6.11392754955802, 3.6124956735471914),
        focal_point=(0.0, 0.375, 2.0),
        viewup=(-0.42546442225230097, 0.9024244135964158, -0.06789847673314177),
    )

    cmap = 'bwr'

    # plot this displaced beam
    pl = pv.Plotter(notebook=notebook)
    pl.add_mesh(
        grid,
        scalars=d,
        scalar_bar_args={'title': 'Y Displacement'},
        rng=[-d.max(), d.max()],
        cmap=cmap,  # type: ignore[arg-type]
    )
    pl.camera_position = cpos
    pl.add_text('Static Beam Example')
    pl.show()


def plot_datasets(dataset_type=None):
    """Plot the pyvista dataset types.

    This demo plots the following PyVista dataset types:

    * :class:`pyvista.PolyData`
    * :class:`pyvista.UnstructuredGrid`
    * :class:`pyvista.ImageData`
    * :class:`pyvista.RectilinearGrid`
    * :class:`pyvista.StructuredGrid`

    Parameters
    ----------
    dataset_type : str, optional
        If set, plot just that dataset.  Must be one of the following:

        * ``'PolyData'``
        * ``'UnstructuredGrid'``
        * ``'ImageData'``
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
        'ImageData',
        'RectilinearGrid',
        'StructuredGrid',
    ]
    if dataset_type is not None and dataset_type not in allowable_types:
        msg = (
            f'Invalid dataset_type {dataset_type}.  '
            f'Must be one of the following: {allowable_types}'
        )
        raise ValueError(msg)

    ###########################################################################
    # uniform grid
    image = pv.ImageData(dimensions=(6, 6, 1))
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
    ang, r, z = np.meshgrid(ang, r, z)  # type: ignore[assignment]

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

    pl = pv.Plotter() if dataset_type is not None else pv.Plotter(shape='3/2')

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

    # ImageData
    if dataset_type is None:
        pl.subplot(2)
        pl.add_text('1. ImageData')
    if dataset_type in [None, 'ImageData']:
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
