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

# get location of this folder
dir_path = os.path.dirname(os.path.realpath(__file__))


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

       >>> mesh = pyvista.ParametricRandomHills()
       >>> mesh = mesh.elevation()

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
        Normals                 float32  (842, 3)             NORMALS
        vectors                 float32  (842, 3)             VECTORS

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


def load_explicit_structured(dims=(5, 6, 7), spacing=(20, 10, 1)):
    """Load a simple explicit structured grid.

    Parameters
    ----------
    dims : tuple(int), optional
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
    ni, nj, nk = np.asarray(dims)-1
    si, sj, sk = spacing

    xcorn = np.arange(0, (ni+1)*si, si)
    xcorn = np.repeat(xcorn, 2)
    xcorn = xcorn[1:-1]
    xcorn = np.tile(xcorn, 4*nj*nk)

    ycorn = np.arange(0, (nj+1)*sj, sj)
    ycorn = np.repeat(ycorn, 2)
    ycorn = ycorn[1:-1]
    ycorn = np.tile(ycorn, (2*ni, 2*nk))
    ycorn = np.transpose(ycorn)
    ycorn = ycorn.flatten()

    zcorn = np.arange(0, (nk+1)*sk, sk)
    zcorn = np.repeat(zcorn, 2)
    zcorn = zcorn[1:-1]
    zcorn = np.repeat(zcorn, (4*ni*nj))

    corners = np.stack((xcorn, ycorn, zcorn))
    corners = corners.transpose()

    grid = pyvista.ExplicitStructuredGrid(dims, corners)
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
