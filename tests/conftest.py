import numpy as np
from pytest import fixture

import pyvista
from pyvista import examples


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
