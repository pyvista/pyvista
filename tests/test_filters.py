import pytest

import vtki
from vtki import examples


def test_uniform_grid_filters():
    """This tests all avaialble filters"""
    dataset = examples.load_uniform()
    dataset.set_active_scalar('Spatial Point Data')
    # Threshold
    thresh = dataset.threshold([100, 500])
    assert thresh is not None
    # Slice
    slc = dataset.slice()
    assert slc is not None
    # Clip
    clp = dataset.clip(invert=True)
    assert clp is not None
    # Contour
    iso = dataset.contour()
    assert iso is not None
