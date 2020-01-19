import numpy as np
from pytest import fixture
import pyvista
from vtk.numpy_interface.dataset_adapter import ArrayAssociation


@fixture()
def example_grid():
    return pyvista.UnstructuredGrid(pyvista.examples.hexbeamfile).copy()


def test_init(example_grid):
    attributes = pyvista.DataSetAttributes(
        example_grid.GetPointData(), dataset=example_grid, association=ArrayAssociation.POINT)
    assert attributes.VTKObject == example_grid.GetPointData()
    assert attributes.dataset == example_grid
    assert attributes.association == ArrayAssociation.POINT