import numpy as np
from numpy.random import default_rng
from pytest import fixture, skip

import pyvista
from pyvista import examples

pyvista.OFF_SCREEN = True


@fixture(scope='session', autouse=True)
def set_mpl():
    """Avoid matplotlib windows popping up."""
    try:
        import matplotlib
    except ImportError:
        pass
    else:
        matplotlib.use('agg', force=True)


@fixture()
def cube():
    return pyvista.Cube()


@fixture()
def airplane():
    return examples.load_airplane()


@fixture()
def rectilinear():
    return examples.load_rectilinear()


@fixture()
def sphere():
    return examples.load_sphere()


@fixture()
def uniform():
    return examples.load_uniform()


@fixture()
def ant():
    return examples.load_ant()


@fixture()
def globe():
    return examples.load_globe()


@fixture()
def hexbeam():
    return examples.load_hexbeam()


@fixture()
def tetbeam():
    return examples.load_tetbeam()


@fixture()
def struct_grid():
    x, y, z = np.meshgrid(
        np.arange(-10, 10, 2, dtype=np.float32),
        np.arange(-10, 10, 2, dtype=np.float32),
        np.arange(-10, 10, 2, dtype=np.float32),
    )
    return pyvista.StructuredGrid(x, y, z)


@fixture()
def plane():
    return pyvista.Plane()


@fixture()
def spline():
    return examples.load_spline()


@fixture()
def tri_cylinder():
    """Triangulated cylinder"""
    return pyvista.Cylinder().triangulate()


@fixture()
def datasets():
    return [
        examples.load_uniform(),  # UniformGrid
        examples.load_rectilinear(),  # RectilinearGrid
        examples.load_hexbeam(),  # UnstructuredGrid
        examples.load_airplane(),  # PolyData
        examples.load_structured(),  # StructuredGrid
    ]


@fixture()
def datasets_vtk9():
    return [
        examples.load_explicit_structured(),
    ]


@fixture()
def pointset():
    rng = default_rng(0)
    points = rng.random((10, 3))
    return pyvista.PointSet(points)


@fixture()
def noise_2d():
    freq = [10, 5, 0]
    noise = pyvista.perlin_noise(1, freq, (0, 0, 0))
    return pyvista.sample_function(noise, bounds=(0, 10, 0, 10, 0, 10), dim=(2**4, 2**4, 1))


def pytest_addoption(parser):
    parser.addoption("--reset_image_cache", action='store_true', default=False)
    parser.addoption("--ignore_image_cache", action='store_true', default=False)
    parser.addoption("--fail_extra_image_cache", action='store_true', default=False)


def pytest_runtest_setup(item):
    """Custom setup to handle skips based on VTK version.

    See pytest.mark.needs_vtk9 and pytest.mark.needs_vtk_version
    in pytest.ini.

    """
    for mark in item.iter_markers('needs_vtk9'):
        # this test needs VTK 9 or newer
        if not pyvista._vtk.VTK9:
            skip('Test needs VTK 9 or newer.')
    for mark in item.iter_markers('needs_vtk_version'):
        # this test needs the given VTK version
        # allow both needs_vtk_version(9, 1) and needs_vtk_version((9, 1))
        args = mark.args
        if len(args) == 1 and isinstance(args[0], tuple):
            version_needed = args[0]
        else:
            version_needed = args
        if pyvista.vtk_version_info < version_needed:
            version_str = '.'.join(map(str, version_needed))
            skip(f'Test needs VTK {version_str} or newer.')
