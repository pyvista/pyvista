import pytest

import vtki
from vtki import examples



datasets = [
    examples.load_uniform(), # UniformGrid
    examples.load_rectilinear(), # RectilinearGrid
    examples.load_hexbean(), # UnstructuredGrid
    examples.load_airplane(), # PolyData
    examples.load_structured(), # StructuredGrid
]
normals = ['x', 'y', '-z', (1,1,1), (3.3, 5.4, 0.8)]



def test_clip_filter():
    """This tests the clip filter on all datatypes avaialble filters"""
    for i, dataset in enumerate(datasets):
        clp = dataset.clip(normal=normals[i], invert=True)
        assert clp is not None
        assert isinstance(clp, vtki.UnstructuredGrid)


def test_slice_filter():
    """This tests the slice filter on all datatypes avaialble filters"""
    for i, dataset in enumerate(datasets):
        slc = dataset.slice(normal=normals[i])
        assert slc is not None
        assert isinstance(slc, vtki.PolyData)


def test_slice_orthogonal_filter():
    """This tests the slice filter on all datatypes avaialble filters"""

    for i, dataset in enumerate(datasets):
        slices = dataset.slice_orthogonal()
        assert slices is not None
        assert isinstance(slices, vtki.MultiBlock)
        assert slices.n_blocks == 3
        for slc in slices:
            assert isinstance(slc, vtki.PolyData)


def test_slice_along_axis():
    """Test the many slices along axis filter """
    axii = ['x', 'y', 'z', 'y', 'x']
    ns =  [2, 3, 4, 10, 20, 13]
    for i, dataset in enumerate(datasets):
        slices = dataset.slice_along_axis(n=ns[i], axis=axii[i])
        assert slices is not None
        assert isinstance(slices, vtki.MultiBlock)
        assert slices.n_blocks == ns[i]
        for slc in slices:
            assert isinstance(slc, vtki.PolyData)

def test_threshold():
    for i, dataset in enumerate(datasets[0:3]):
        thresh = dataset.threshold()
        assert thresh is not None
        assert isinstance(thresh, vtki.UnstructuredGrid)
    # Test value ranges
    dataset = examples.load_uniform() # UniformGrid
    thresh = dataset.threshold(100, invert=False)
    assert thresh is not None
    assert isinstance(thresh, vtki.UnstructuredGrid)
    thresh = dataset.threshold([100, 500], invert=False)
    assert thresh is not None
    assert isinstance(thresh, vtki.UnstructuredGrid)
    thresh = dataset.threshold([100, 500], invert=True)
    assert thresh is not None
    assert isinstance(thresh, vtki.UnstructuredGrid)
    # Now test datasets without arrays
    with pytest.raises(RuntimeError):
        for i, dataset in enumerate(datasets[3:-1]):
            thresh = dataset.threshold()
            assert thresh is not None
            assert isinstance(thresh, vtki.UnstructuredGrid)


def test_threshold_percent():
    percents = [25, 50, [18.0, 85.0], [19.0, 80.0], 0.70]
    inverts = [False, True, False, True, False]
    # Only test data sets that have arrays
    for i, dataset in enumerate(datasets[0:3]):
        thresh = dataset.threshold_percent(percent=percents[i], invert=inverts[i])
        assert thresh is not None
        assert isinstance(thresh, vtki.UnstructuredGrid)


def test_outline():
    for i, dataset in enumerate(datasets):
        outline = dataset.outline()
        assert outline is not None
        assert isinstance(outline, vtki.PolyData)

def test_outline_corners():
    for i, dataset in enumerate(datasets):
        outline = dataset.outline_corners()
        assert outline is not None
        assert isinstance(outline, vtki.PolyData)


def test_extract_geometry():
    for i, dataset in enumerate(datasets):
        outline = dataset.extract_geometry()
        assert outline is not None
        assert isinstance(outline, vtki.PolyData)

def test_extract_edges():
    for i, dataset in enumerate(datasets):
        wire = dataset.extract_geometry()
        assert wire is not None
        assert isinstance(wire, vtki.PolyData)
        wire = dataset.wireframe()
        assert wire is not None
        assert isinstance(wire, vtki.PolyData)


def test_contour():
    dataset = examples.load_uniform()
    iso = dataset.contour()
    assert iso is not None
    iso = dataset.contour(isosurfaces=[100, 300, 500])
    assert iso is not None
