import numpy as np
import pytest

from pyvista import pyvista_ndarray
from pyvista import examples


def test_slices_are_associated():
    dataset = examples.load_structured()
    points = pyvista_ndarray(dataset.GetPoints().GetData(), dataset=dataset)

    # check that slices of pyvista_ndarray are associated correctly
    assert points[1, :].VTKObject == points.VTKObject
    assert points[1, :].dataset.Get() == points.dataset.Get()
    assert points[1, :].association == points.association


def test_copies_are_not_associated():
    dataset = examples.load_structured()
    points = pyvista_ndarray(dataset.GetPoints().GetData(), dataset=dataset)
    points_2 = points.copy()

    # check that copies of pyvista_ndarray are dissociated from the original dataset
    assert points_2.VTKObject is None
    assert points_2.dataset.Get() is None
    assert points_2.association.name == 'NONE'
    assert not np.shares_memory(points, points_2)


def test_isequal():
    dataset = examples.load_structured()
    dataset_2 = examples.load_structured()

    dataset.points[:, -1] = np.inf
    dataset_2.points[:, -1] = np.inf

    assert np.allclose(dataset.points, dataset_2.points, equal_nan=True)

# TODO: This currently doesn't work for single element indexing operations!
# in these cases, the __array_finalize__ method is not called
@pytest.mark.skip
def test_slices_are_associated_single_index():
    dataset = examples.load_structured()
    points = pyvista_ndarray(dataset.GetPoints().GetData(), dataset=dataset)

    assert points[1, 1].VTKObject == points.VTKObject
    assert points[1, 1].dataset.Get() == points.dataset.Get()
    assert points[1, 1].association == points.association
