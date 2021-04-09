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


# TODO: This currently doesn't work for single element indexing operations!
# in these cases, the __array_finalize__ method is not called
@pytest.mark.skip
def test_slices_are_associated_single_index():
    dataset = examples.load_structured()
    points = pyvista_ndarray(dataset.GetPoints().GetData(), dataset=dataset)

    assert points[1, 1].VTKObject == points.VTKObject
    assert points[1, 1].dataset.Get() == points.dataset.Get()
    assert points[1, 1].association == points.association
