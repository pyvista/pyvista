import numpy as np
from numpy.random import default_rng
from pytest import fixture

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
def multiblock_poly():
    # format and order of data (including missing) is intentional
    mesh_a = pyvista.Sphere(center=(0, 0, 0))
    mesh_a['data_a'] = mesh_a.points[:, 0] * 10
    mesh_a['data_b'] = mesh_a.points[:, 1] * 10
    mesh_a['cell_data'] = mesh_a.cell_centers().points[:, 0]
    mesh_a.point_data.set_array(mesh_a.points[:, 2] * 10, 'all_data')

    mesh_b = pyvista.Sphere(center=(1, 0, 0))
    mesh_b['data_a'] = mesh_b.points[:, 0] * 10
    mesh_b['data_b'] = mesh_b.points[:, 1] * 10
    mesh_b['cell_data'] = mesh_b.cell_centers().points[:, 0]
    mesh_b.point_data.set_array(mesh_b.points[:, 2] * 10, 'all_data')

    mesh_c = pyvista.Sphere(center=(2, 0, 0))
    mesh_c.point_data.set_array(mesh_c.points, 'multi-comp')
    mesh_c.point_data.set_array(mesh_c.points[:, 2] * 10, 'all_data')

    mblock = pyvista.MultiBlock()
    mblock.append(mesh_a)
    mblock.append(mesh_b)
    mblock.append(mesh_c)
    return mblock


@fixture()
def pointset():
    rng = default_rng(0)
    points = rng.random((10, 3))
    return pyvista.PointSet(points)


@fixture()
def multiblock_all(datasets):
    """Return datasets fixture combined in a pyvista multiblock."""
    return pyvista.MultiBlock(datasets)


@fixture()
def noise_2d():
    freq = [10, 5, 0]
    noise = pyvista.perlin_noise(1, freq, (0, 0, 0))
    return pyvista.sample_function(noise, bounds=(0, 10, 0, 10, 0, 10), dim=(2**4, 2**4, 1))


def pytest_addoption(parser):
    parser.addoption("--reset_image_cache", action='store_true', default=False)
    parser.addoption("--ignore_image_cache", action='store_true', default=False)
    parser.addoption("--fail_extra_image_cache", action='store_true', default=False)
