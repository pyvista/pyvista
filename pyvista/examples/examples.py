"""Built-in examples that ship with PyVista and do not need to be downloaded.

Examples
--------
>>> from pyvista import examples
>>> mesh = examples.load_ant()
>>> mesh.plot()

"""

import os

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


def load_ant():
    """Load ply ant mesh.

    Returns
    -------
    pyvista.PolyData
        Dataset.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.load_ant()
    >>> dataset.plot()

    """
    return pyvista.PolyData(antfile)


def load_airplane():
    """Load ply airplane mesh.

    Returns
    -------
    pyvista.PolyData
        Dataset.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.load_airplane()
    >>> dataset.plot()

    """
    return pyvista.PolyData(planefile)


def load_sphere():
    """Load sphere ply mesh.

    Returns
    -------
    pyvista.PolyData
        Dataset.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.load_sphere()
    >>> dataset.plot()

    """
    return pyvista.PolyData(spherefile)


def load_uniform():
    """Load a sample uniform grid.

    Returns
    -------
    pyvista.UniformGrid
        Dataset.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.load_uniform()
    >>> dataset.plot()

    """
    return pyvista.UniformGrid(uniformfile)


def load_rectilinear():
    """Load a sample uniform grid.

    Returns
    -------
    pyvista.RectilinearGrid
        Dataset.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.load_rectilinear()
    >>> dataset.plot()

    """
    return pyvista.RectilinearGrid(rectfile)


def load_hexbeam():
    """Load a sample UnstructuredGrid.

    Returns
    -------
    pyvista.UnstructuredGrid
        Dataset.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.load_hexbeam()
    >>> dataset.plot()

    """
    return pyvista.UnstructuredGrid(hexbeamfile)


def load_tetbeam():
    """Load a sample UnstructuredGrid containing only tetrahedral cells.

    Returns
    -------
    pyvista.UnstructuredGrid
        Dataset.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.load_tetbeam()
    >>> dataset.plot()

    """
    # make the geometry identical to the hexbeam
    xrng = np.linspace(0, 1, 3)
    yrng = np.linspace(0, 1, 3)
    zrng = np.linspace(0, 5, 11)
    grid = pyvista.RectilinearGrid(xrng, yrng, zrng)
    return grid.to_tetrahedra()


def load_structured():
    """Load a simple StructuredGrid.

    Returns
    -------
    pyvista.StructuredGrid
        Dataset.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.load_structured()
    >>> dataset.plot()

    """
    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    z = np.sin(r)
    return pyvista.StructuredGrid(x, y, z)


def load_globe():
    """Load a globe source.

    Returns
    -------
    pyvista.PolyData
        Globe dataset with earth texture.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.load_globe()
    >>> dataset.plot()

    """
    globe = pyvista.PolyData(globefile)
    globe.textures['2k_earth_daymap'] = load_globe_texture()
    return globe


def load_globe_texture():
    """Load a vtk.vtkTexture that can be applied to the globe source.

    Returns
    -------
    pyvista.Texture
        Dataset.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.load_globe_texture()
    >>> dataset.plot()

    """
    return pyvista.read_texture(mapfile)


def load_channels():
    """Load a uniform grid of fluvial channels in the subsurface.

    Returns
    -------
    pyvista.UniformGrid
        Dataset.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.load_channels()
    >>> dataset.plot()

    """
    return pyvista.read(channelsfile)


def load_spline():
    """Load an example spline mesh.

    This example data was created with:

    .. code:: python

       >>> import numpy as np
       >>> import pyvista
       >>> theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
       >>> z = np.linspace(-2, 2, 100)
       >>> r = z**2 + 1
       >>> x = r * np.sin(theta)
       >>> y = r * np.cos(theta)
       >>> points = np.column_stack((x, y, z))
       >>> mesh = pyvista.Spline(points, 1000)

    Returns
    -------
    pyvista.PolyData
        Spline mesh.

    Examples
    --------
    >>> from pyvista import examples
    >>> spline = examples.load_spline()
    >>> spline.plot()

    """
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

    This example dataset was created with:

    .. code:: python

       >>> mesh = pyvista.ParametricRandomHills()  # doctest:+SKIP
       >>> mesh = mesh.elevation()  # doctest:+SKIP

    Returns
    -------
    pyvista.PolyData
        Random hills mesh.

    Examples
    --------
    >>> from pyvista import examples
    >>> mesh = examples.load_random_hills()
    >>> mesh.plot()

    """
    mesh = pyvista.ParametricRandomHills()
    return mesh.elevation()


def load_sphere_vectors():
    """Create example sphere with a swirly vector field defined on nodes.

    Returns
    -------
    pyvista.PolyData
        Mesh containing vectors.

    Examples
    --------
    >>> from pyvista import examples
    >>> mesh = examples.load_sphere_vectors()
    >>> mesh.point_data
    pyvista DataSetAttributes
    Association     : POINT
    Active Scalars  : vectors
    Active Vectors  : vectors
    Active Texture  : None
    Active Normals  : Normals
    Contains arrays :
        Normals                 float32    (842, 3)             NORMALS
        vectors                 float32    (842, 3)             VECTORS

    """
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
    sphere["vectors"] = vectors * 0.3
    sphere.set_active_vectors("vectors")
    return sphere


def load_explicit_structured(dimensions=(5, 6, 7), spacing=(20, 10, 1)):
    """Load a simple explicit structured grid.

    Parameters
    ----------
    dimensions : tuple(int), optional
        Grid dimensions. Default is (5, 6, 7).
    spacing : tuple(int), optional
        Grid spacing. Default is (20, 10, 1).

    Returns
    -------
    pyvista.ExplicitStructuredGrid
        An explicit structured grid.

    Examples
    --------
    >>> from pyvista import examples
    >>> grid = examples.load_explicit_structured()
    >>> grid.plot(show_edges=True)

    """
    ni, nj, nk = np.asarray(dimensions) - 1
    si, sj, sk = spacing

    xcorn = np.arange(0, (ni + 1) * si, si)
    xcorn = np.repeat(xcorn, 2)
    xcorn = xcorn[1:-1]
    xcorn = np.tile(xcorn, 4 * nj * nk)

    ycorn = np.arange(0, (nj + 1) * sj, sj)
    ycorn = np.repeat(ycorn, 2)
    ycorn = ycorn[1:-1]
    ycorn = np.tile(ycorn, (2 * ni, 2 * nk))
    ycorn = np.transpose(ycorn)
    ycorn = ycorn.flatten()

    zcorn = np.arange(0, (nk + 1) * sk, sk)
    zcorn = np.repeat(zcorn, 2)
    zcorn = zcorn[1:-1]
    zcorn = np.repeat(zcorn, (4 * ni * nj))

    corners = np.stack((xcorn, ycorn, zcorn))
    corners = corners.transpose()

    grid = pyvista.ExplicitStructuredGrid(dimensions, corners)
    return grid


def load_nut():
    """Load an example nut mesh.

    Returns
    -------
    pyvista.PolyData
        A sample nut surface dataset.

    Examples
    --------
    Load an example nut and plot with smooth shading.

    >>> from pyvista import examples
    >>> mesh = examples.load_nut()
    >>> mesh.plot(smooth_shading=True, split_sharp_edges=True)

    """
    return pyvista.read(os.path.join(dir_path, 'nut.ply'))


def _cart2sphe(x, y, z):
    """Convert 3D Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    x, y, z : np.ndarray
        Coordinates.

    """
    xy2 = x**2 + y**2
    r = np.sqrt(xy2 + z**2)
    theta = np.arctan2(np.sqrt(xy2), z)  # the polar angle in radian angles
    phi = np.arctan2(y, x)  # the azimuth angle in radian angles
    phi[phi < 0] += np.pi * 2  # np.arctan2 returns the angle in the range [-pi, pi]

    return r, theta, phi


def load_hydrogen_orbital(n=1, l=0, m=0, norm=True):
    """Load the hydrogen wave function for a :class:`pyvista.UniformGrid`.

    Ispired by `Hydrogen Wave Function
    <http://staff.ustc.edu.cn/~zqj/posts/Hydrogen-Wavefunction/>`_

    Notes
    -----
    This examples requires `sympy <https://www.sympy.org/>`_.
    """
    try:
        from sympy import lambdify
        from sympy.abc import phi, r, theta
        from sympy.physics.hydrogen import Psi_nlm
    except ImportError:
        raise ImportError('Install sympy to run this example.') from None

    if n < 1 or n > 5:
        raise ValueError('`n` must be between 1 and 5')
    if l not in range(n):
        raise ValueError(f'`l` must be one of: {list(range(n))}')
    if m not in range(l + 1):
        raise ValueError(f'`l` must be one of: {list(range(l+1))}')

    psi = lambdify((r, phi, theta), Psi_nlm(n, l, m, r, phi, theta), 'numpy')

    if n == 1:
        l, n = 1.5 * n**2 + 1.0, 200
    else:
        l, n = 1.5 * n**2 + 10.0, 200

    n = 100
    s = (l * 2) / (n - 1)
    grid = pyvista.UniformGrid(dimensions=(n, n, n), spacing=(s, s, s), origin=(-l, -l, -l))

    r0, theta, phi = _cart2sphe(grid.x, grid.y, grid.z)
    wfc = psi(r0, phi, theta).reshape(grid.dimensions)

    grid['norm_hwf'] = np.abs(wfc.ravel())
    grid['hwf'] = wfc.ravel()

    return grid
