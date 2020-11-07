import gc

import numpy as np
from pytest import fixture
import pytest

import pyvista
from pyvista import examples
pyvista.rcParams['off_screen'] = True
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


def _is_vtk(obj):
    try:
        return obj.__class__.__name__.startswith('vtk')
    except Exception:  # old Python sometimes no __class__.__name__
        return False


@pytest.fixture(autouse=True)
def check_gc():
    """Ensure that all VTK objects are garbage-collected by Python."""
    before = set(id(o) for o in gc.get_objects() if _is_vtk(o))
    yield
    pyvista.close_all()
    gc.collect()
    after = [o for o in gc.get_objects() if _is_vtk(o) and id(o) not in before]
    assert len(after) == 0, \
        'Not all objects GCed:\n' + \
        '\n'.join(sorted(o.__class__.__name__ for o in after))
