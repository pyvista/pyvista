import numpy as np
from pytest import fixture

import pyvista
from pyvista import examples
pyvista.rcParams['off_screen'] = True

# necessary for image regression.  Xvfb doesn't support
# multi-sampling, so we disable it here for consistency between
# desktops and remote testing
pyvista.rcParams['multi_samples'] = 1
pyvista.rcParams['window_size'] = [400, 400]  # default is [1024, 768]
pyvista.OFF_SCREEN = True


@fixture(scope='session')
def set_mpl():
    """Avoid matplotlib windows popping up."""
    try:
        import matplotlib
    except Exception:
        pass
    else:
        matplotlib.use('agg', force=True)


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
    x, y, z = np.meshgrid(np.arange(-10, 10, 2),
                          np.arange(-10, 10, 2),
                          np.arange(-10, 10, 2))
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
