"""Built-in examples that ship with PyVista and do not need to be downloaded.

Examples
--------
>>> from pyvista import examples
>>> mesh = examples.load_ant()
>>> mesh.plot()

"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

import pyvista
from pyvista.examples._dataset_loader import _DatasetLoader
from pyvista.examples._dataset_loader import _SingleFileDownloadableDatasetLoader

# get location of this folder and the example files
dir_path = str(Path(os.path.realpath(__file__)).parent)
antfile = str(Path(dir_path) / 'ant.ply')
planefile = str(Path(dir_path) / 'airplane.ply')
hexbeamfile = str(Path(dir_path) / 'hexbeam.vtk')
spherefile = str(Path(dir_path) / 'sphere.ply')
uniformfile = str(Path(dir_path) / 'uniform.vtk')
rectfile = str(Path(dir_path) / 'rectilinear.vtk')
globefile = str(Path(dir_path) / 'globe.vtk')
mapfile = str(Path(dir_path) / '2k_earth_daymap.jpg')
channelsfile = str(Path(dir_path) / 'channels.vti')
logofile = str(Path(dir_path) / 'pyvista_logo.png')
nutfile = str(Path(dir_path) / 'nut.ply')
frogtissuesfile = str(Path(dir_path) / 'frog_tissues.vti')


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

    .. seealso::

        :ref:`Ant Dataset <ant_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_ant.load()


_dataset_ant = _SingleFileDownloadableDatasetLoader(antfile, read_func=pyvista.PolyData)  # type: ignore[arg-type]


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

    .. seealso::

        :ref:`Airplane Dataset <airplane_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_airplane.load()


_dataset_airplane = _SingleFileDownloadableDatasetLoader(planefile, read_func=pyvista.PolyData)  # type: ignore[arg-type]


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

    .. seealso::

        :ref:`Sphere Dataset <sphere_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_sphere.load()


_dataset_sphere = _SingleFileDownloadableDatasetLoader(spherefile, read_func=pyvista.PolyData)  # type: ignore[arg-type]


def load_uniform():
    """Load a sample uniform grid.

    Returns
    -------
    pyvista.ImageData
        Dataset.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.load_uniform()
    >>> dataset.plot()

    .. seealso::

        :ref:`Uniform Dataset <uniform_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_uniform.load()


_dataset_uniform = _SingleFileDownloadableDatasetLoader(uniformfile, read_func=pyvista.ImageData)  # type: ignore[arg-type]


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

    .. seealso::

        :ref:`Rectilinear Dataset <rectilinear_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_rectilinear.load()


_dataset_rectilinear = _SingleFileDownloadableDatasetLoader(
    rectfile,
    read_func=pyvista.RectilinearGrid,  # type: ignore[arg-type]
)


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

    .. seealso::

        :ref:`Hexbeam Dataset <hexbeam_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_hexbeam.load()


_dataset_hexbeam = _SingleFileDownloadableDatasetLoader(
    hexbeamfile,
    read_func=pyvista.UnstructuredGrid,  # type: ignore[arg-type]
)


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

    .. seealso::

        :ref:`Tetbeam Dataset <tetbeam_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_tetbeam.load()


def _tetbeam_load_func():
    # make the geometry identical to the hexbeam
    xrng = np.linspace(0, 1, 3)
    yrng = np.linspace(0, 1, 3)
    zrng = np.linspace(0, 5, 11)
    grid = pyvista.RectilinearGrid(xrng, yrng, zrng)
    return grid.to_tetrahedra()


_dataset_tetbeam = _DatasetLoader(_tetbeam_load_func)


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

    .. seealso::

        :ref:`Structured Dataset <structured_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_structured.load()


def _structured_load_func():
    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    z = np.sin(r)
    return pyvista.StructuredGrid(x, y, z)


_dataset_structured = _DatasetLoader(_structured_load_func)


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
    >>> texture = examples.load_globe_texture()
    >>> dataset.plot(texture=texture)

    .. seealso::

        :ref:`Globe Dataset <globe_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_globe.load()


_dataset_globe = _SingleFileDownloadableDatasetLoader(globefile, read_func=pyvista.PolyData)  # type: ignore[arg-type]


def load_globe_texture():
    """Load a pyvista.Texture that can be applied to the globe source.

    Returns
    -------
    pyvista.Texture
        Dataset.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.load_globe_texture()
    >>> dataset.plot()

    .. seealso::

        :ref:`Globe Texture Dataset <globe_texture_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_globe_texture.load()


_dataset_globe_texture = _SingleFileDownloadableDatasetLoader(
    mapfile,
    read_func=pyvista.read_texture,  # type: ignore[arg-type]
)


def load_channels():
    """Load a uniform grid of fluvial channels in the subsurface.

    Returns
    -------
    pyvista.ImageData
        Dataset.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.load_channels()
    >>> dataset.plot()

    .. seealso::

        :ref:`Channels Dataset <channels_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_channels.load()


_dataset_channels = _SingleFileDownloadableDatasetLoader(channelsfile)


def load_spline():
    """Load an example spline mesh.

    This example data was created with:

    .. code-block:: python

       >>> import numpy as np
       >>> import pyvista as pv
       >>> theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
       >>> z = np.linspace(-2, 2, 100)
       >>> r = z**2 + 1
       >>> x = r * np.sin(theta)
       >>> y = r * np.cos(theta)
       >>> points = np.column_stack((x, y, z))
       >>> mesh = pv.Spline(points, 1000)

    Returns
    -------
    pyvista.PolyData
        Spline mesh.

    Examples
    --------
    >>> from pyvista import examples
    >>> spline = examples.load_spline()
    >>> spline.plot()

    .. seealso::

        :ref:`Spline Dataset <spline_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_spline.load()


def _spline_load_func():
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    points = np.column_stack((x, y, z))
    return pyvista.Spline(points, 1000)


_dataset_spline = _DatasetLoader(_spline_load_func)


def load_random_hills():
    """Create random hills toy example.

    Uses the parametric random hill function to create hills oriented
    like topography and adds an elevation array.

    This example dataset was created with:

    .. code-block:: python

       >>> mesh = pv.ParametricRandomHills()  # doctest:+SKIP
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

    .. seealso::

        :ref:`Random Hills Dataset <random_hills_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_random_hills.load()


def _random_hills_load_func():
    mesh = pyvista.ParametricRandomHills()
    return mesh.elevation()


_dataset_random_hills = _DatasetLoader(_random_hills_load_func)


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

    .. seealso::

        :ref:`Sphere Vectors Dataset <sphere_vectors_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_sphere_vectors.load()


def _sphere_vectors_load_func() -> pyvista.PolyData:
    sphere = pyvista.Sphere(radius=math.pi)

    # make cool swirly pattern
    vectors = np.vstack(
        (
            np.sin(sphere.points[:, 0]),
            np.cos(sphere.points[:, 1]),
            np.cos(sphere.points[:, 2]),
        ),
    ).T

    # add and scale
    sphere['vectors'] = vectors * 0.3
    sphere.set_active_vectors('vectors')
    return sphere


_dataset_sphere_vectors = _DatasetLoader(_sphere_vectors_load_func)


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

    .. seealso::

        :ref:`Explicit Structured Dataset <explicit_structured_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_explicit_structured.load(dimensions=dimensions, spacing=spacing)


def _explicit_structured_load_func(dimensions=(5, 6, 7), spacing=(20, 10, 1)):
    ni, nj, nk = np.asarray(dimensions) - 1
    si, sj, sk = spacing
    xi = np.arange(0.0, (ni + 1) * si, si)
    yi = np.arange(0.0, (nj + 1) * sj, sj)
    zi = np.arange(0.0, (nk + 1) * sk, sk)

    return pyvista.StructuredGrid(
        *np.meshgrid(xi, yi, zi, indexing='ij')
    ).cast_to_explicit_structured_grid()


_dataset_explicit_structured = _DatasetLoader(_explicit_structured_load_func)


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

    .. seealso::

        :ref:`Nut Dataset <nut_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_nut.load()


_dataset_nut = _SingleFileDownloadableDatasetLoader(nutfile)


def load_hydrogen_orbital(n=1, l=0, m=0, zoom_fac=1.0):  # noqa: PLR0917
    """Load the hydrogen wave function for a :class:`pyvista.ImageData`.

    This is the solution to the Schrödinger equation for hydrogen
    evaluated in three-dimensional Cartesian space.

    Inspired by `Hydrogen Wave Function
    <http://staff.ustc.edu.cn/~zqj/posts/Hydrogen-Wavefunction/>`_.

    Parameters
    ----------
    n : int, default: 1
        Principal quantum number. Must be a positive integer. This is often
        referred to as the "energy level" or "shell".

    l : int, default: 0
        Azimuthal quantum number. Must be a non-negative integer strictly
        smaller than ``n``. By convention this value is represented by the
        letters s, p, d, f, etc.

    m : int, default: 0
        Magnetic quantum number. Must be an integer ranging from ``-l`` to
        ``l`` (inclusive). This is the orientation of the angular momentum in
        space.

    zoom_fac : float, default: 1.0
        Zoom factor for the electron cloud. Increase this value to focus on the
        center of the electron cloud.

    Returns
    -------
    pyvista.ImageData
        ImageData containing two ``point_data`` arrays:

        * ``'real_wf'`` - Real part of the wave function.
        * ``'wf'`` - Complex wave function.

    Notes
    -----
    This example requires `sympy <https://www.sympy.org/>`_.

    Examples
    --------
    Plot the 3dxy orbital of a hydrogen atom. This corresponds to the quantum
    numbers ``n=3``, ``l=2``, and ``m=-2``.

    >>> from pyvista import examples
    >>> grid = examples.load_hydrogen_orbital(3, 2, -2)
    >>> grid.plot(volume=True, opacity=[1, 0, 1], cmap='magma')

    See :ref:`atomic_orbitals_example` for additional examples using
    this function.

    .. seealso::

        :ref:`Hydrogen Orbital Dataset <hydrogen_orbital_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_hydrogen_orbital.load(n=n, l=l, m=m, zoom_fac=zoom_fac)


def _hydrogen_orbital_load_func(n=1, l=0, m=0, zoom_fac=1.0):  # noqa: PLR0917
    try:
        from sympy import lambdify
        from sympy.abc import phi
        from sympy.abc import r
        from sympy.abc import theta
        from sympy.physics.hydrogen import Psi_nlm
    except ImportError:  # pragma: no cover
        msg = '\n\nInstall sympy to run this example. Run:\n\n    pip install sympy\n'
        raise ImportError(msg) from None

    if n < 1:
        msg = '`n` must be at least 1'
        raise ValueError(msg)
    if l not in range(n):
        msg = f'`l` must be one of: {list(range(n))}'
        raise ValueError(msg)
    if m not in range(-l, l + 1):
        msg = f'`m` must be one of: {list(range(-l, l + 1))}'
        raise ValueError(msg)

    psi = lambdify((r, phi, theta), Psi_nlm(n, l, m, r, phi, theta, 1), 'numpy')

    org = 1.5 * n**2 + 1.0 if n == 1 else 1.5 * n**2 + 10.0

    org /= zoom_fac

    dim = 100
    sp = (org * 2) / (dim - 1)
    grid = pyvista.ImageData(
        dimensions=(dim, dim, dim),
        spacing=(sp, sp, sp),
        origin=(-org, -org, -org),
    )

    r, theta, phi = pyvista.cartesian_to_spherical(grid.x, grid.y, grid.z)
    wfc = psi(r, phi, theta).reshape(grid.dimensions)

    grid['real_wf'] = np.real(wfc.ravel())
    grid['wf'] = wfc.ravel()
    return grid


_dataset_hydrogen_orbital = _DatasetLoader(_hydrogen_orbital_load_func)


def load_logo():
    """Load the PyVista logo as a :class:`pyvista.ImageData`.

    .. note::

        Alternative versions of the logo file are also available from the ``logo``
        directory at https://github.com/pyvista/pyvista/. This includes
        higher-resolution ``.png`` files and a vectorized ``.svg`` version.

    .. versionchanged:: 0.45

        The dimensions of the image is now ``1389 x 592``.
        Previously, it was ``1920 x 718``.

    Returns
    -------
    pyvista.ImageData
        ImageData of the PyVista logo.

    Examples
    --------
    >>> from pyvista import examples
    >>> image = examples.load_logo()
    >>> image.dimensions
    (1389, 592, 1)

    >>> image.plot(cpos='xy', zoom='tight', rgb=True, show_axes=False)

    .. seealso::

        :ref:`Logo Dataset <logo_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _dataset_logo.load()


_dataset_logo = _SingleFileDownloadableDatasetLoader(logofile)


def load_frog_tissues():
    """Load frog tissues dataset.

    This dataset contains tissue segmentation labels for the frog dataset.

    .. versionadded:: 0.44.0

    Returns
    -------
    pyvista.ImageData
        Dataset.

    Examples
    --------
    Load data

    >>> import numpy as np
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> data = examples.load_frog_tissues()

    Plot tissue labels as a volume

    First, define plotting parameters

    >>> # Configure colors / color bar
    >>> clim = data.get_data_range()  # Set color bar limits to match data
    >>> cmap = 'glasbey'  # Use a categorical colormap
    >>> categories = True  # Ensure n_colors matches number of labels
    >>> opacity = 'foreground'  # Make foreground opaque, background transparent
    >>> opacity_unit_distance = 1

    Set plotting resolution to half the image's spacing

    >>> res = np.array(data.spacing) / 2

    Define rendering parameters

    >>> mapper = 'gpu'
    >>> shade = True
    >>> ambient = 0.3
    >>> diffuse = 0.6
    >>> specular = 0.5
    >>> specular_power = 40

    Make and show plot

    >>> p = pv.Plotter()
    >>> _ = p.add_volume(
    ...     data,
    ...     clim=clim,
    ...     ambient=ambient,
    ...     shade=shade,
    ...     diffuse=diffuse,
    ...     specular=specular,
    ...     specular_power=specular_power,
    ...     mapper=mapper,
    ...     opacity=opacity,
    ...     opacity_unit_distance=opacity_unit_distance,
    ...     categories=categories,
    ...     cmap=cmap,
    ...     resolution=res,
    ... )
    >>> p.camera_position = 'yx'  # Set camera to provide a dorsal view
    >>> p.show()

    .. seealso::

        :ref:`Frog Tissues Dataset <frog_tissues_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Frog Dataset <frog_dataset>`

        :ref:`medical_dataset_gallery`
            Browse other medical datasets.

    """
    return _dataset_frog_tissues.load()


_dataset_frog_tissues = _SingleFileDownloadableDatasetLoader(frogtissuesfile)
