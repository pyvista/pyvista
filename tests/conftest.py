import os
import numpy as np
from pytest import fixture

# env var testing, but be here before any pyvista imports
os.environ['PYVISTA_JUPYTER_BACKEND'] = 'none'

import pyvista
from pyvista import examples

pyvista.set_plot_theme('testing')
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
