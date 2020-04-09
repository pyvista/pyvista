from pytest import fixture
from pyvista import examples, MultiBlock, UnstructuredGrid
import vtk


@fixture()
def grid():
    return UnstructuredGrid(examples.hexbeamfile)


@fixture()
def vtk_multi():
    return vtk.vtkMultiBlockDataSet()


@fixture()
def pyvista_multi():
    return MultiBlock


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
