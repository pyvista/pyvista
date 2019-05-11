import numpy as np
import pytest

import pyvista
from pyvista import examples

datasets = [
    examples.load_uniform(), # UniformGrid
    examples.load_rectilinear(), # RectilinearGrid
    examples.load_hexbeam(), # UnstructuredGrid
    examples.load_airplane(), # PolyData
    examples.load_structured(), # StructuredGrid
]
normals = ['x', 'y', '-z', (1,1,1), (3.3, 5.4, 0.8)]



def test_clip_filter():
    """This tests the clip filter on all datatypes avaialble filters"""
    for i, dataset in enumerate(datasets):
        clp = dataset.clip(normal=normals[i], invert=True)
        assert clp is not None
        assert isinstance(clp, pyvista.UnstructuredGrid)

def test_clip_box():
    for i, dataset in enumerate(datasets):
        clp = dataset.clip_box(invert=True)
        assert clp is not None
        assert isinstance(clp, pyvista.UnstructuredGrid)
    dataset = examples.load_airplane()
    # test length 3 bounds
    result = dataset.clip_box(bounds=(900, 900, 200), invert=False)
    dataset = examples.load_uniform()
    result = dataset.clip_box(bounds=0.5)
    with pytest.raises(AssertionError):
        dataset.clip_box(bounds=(5, 6,))

def test_slice_filter():
    """This tests the slice filter on all datatypes avaialble filters"""
    for i, dataset in enumerate(datasets):
        slc = dataset.slice(normal=normals[i])
        assert slc is not None
        assert isinstance(slc, pyvista.PolyData)
    dataset = examples.load_uniform()
    with pytest.raises(AssertionError):
        dataset.slice(origin=(10, 15, 15))


def test_slice_orthogonal_filter():
    """This tests the slice filter on all datatypes avaialble filters"""

    for i, dataset in enumerate(datasets):
        slices = dataset.slice_orthogonal()
        assert slices is not None
        assert isinstance(slices, pyvista.MultiBlock)
        assert slices.n_blocks == 3
        for slc in slices:
            assert isinstance(slc, pyvista.PolyData)


def test_slice_along_axis():
    """Test the many slices along axis filter """
    axii = ['x', 'y', 'z', 'y', 0]
    ns =  [2, 3, 4, 10, 20, 13]
    for i, dataset in enumerate(datasets):
        slices = dataset.slice_along_axis(n=ns[i], axis=axii[i])
        assert slices is not None
        assert isinstance(slices, pyvista.MultiBlock)
        assert slices.n_blocks == ns[i]
        for slc in slices:
            assert isinstance(slc, pyvista.PolyData)
    dataset = examples.load_uniform()
    with pytest.raises(RuntimeError):
        dataset.slice_along_axis(axis='u')

def test_threshold():
    for i, dataset in enumerate(datasets[0:3]):
        thresh = dataset.threshold()
        assert thresh is not None
        assert isinstance(thresh, pyvista.UnstructuredGrid)
    # Test value ranges
    dataset = examples.load_uniform() # UniformGrid
    thresh = dataset.threshold(100, invert=False)
    assert thresh is not None
    assert isinstance(thresh, pyvista.UnstructuredGrid)
    thresh = dataset.threshold([100, 500], invert=False)
    assert thresh is not None
    assert isinstance(thresh, pyvista.UnstructuredGrid)
    thresh = dataset.threshold([100, 500], invert=True)
    assert thresh is not None
    assert isinstance(thresh, pyvista.UnstructuredGrid)
    # Now test datasets without arrays
    with pytest.raises(AssertionError):
        for i, dataset in enumerate(datasets[3:-1]):
            thresh = dataset.threshold()
            assert thresh is not None
            assert isinstance(thresh, pyvista.UnstructuredGrid)
    dataset = examples.load_uniform()
    with pytest.raises(AssertionError):
        dataset.threshold([10, 100, 300])


def test_threshold_percent():
    percents = [25, 50, [18.0, 85.0], [19.0, 80.0], 0.70]
    inverts = [False, True, False, True, False]
    # Only test data sets that have arrays
    for i, dataset in enumerate(datasets[0:3]):
        thresh = dataset.threshold_percent(percent=percents[i], invert=inverts[i])
        assert thresh is not None
        assert isinstance(thresh, pyvista.UnstructuredGrid)
    dataset = examples.load_uniform()
    result = dataset.threshold_percent(0.75, scalars='Spatial Cell Data')
    with pytest.raises(RuntimeError):
        result = dataset.threshold_percent(20000)
    with pytest.raises(RuntimeError):
        result = dataset.threshold_percent(0.0)


def test_outline():
    for i, dataset in enumerate(datasets):
        outline = dataset.outline()
        assert outline is not None
        assert isinstance(outline, pyvista.PolyData)

def test_outline_corners():
    for i, dataset in enumerate(datasets):
        outline = dataset.outline_corners()
        assert outline is not None
        assert isinstance(outline, pyvista.PolyData)


def test_extract_geometry():
    for i, dataset in enumerate(datasets):
        outline = dataset.extract_geometry()
        assert outline is not None
        assert isinstance(outline, pyvista.PolyData)

def test_wireframe():
    for i, dataset in enumerate(datasets):
        wire = dataset.wireframe()
        assert wire is not None
        assert isinstance(wire, pyvista.PolyData)


def test_contour():
    dataset = examples.load_uniform()
    iso = dataset.contour()
    assert iso is not None
    iso = dataset.contour(isosurfaces=[100, 300, 500])
    assert iso is not None
    with pytest.raises(AssertionError):
        result = dataset.contour(scalars='Spatial Cell Data')
    with pytest.raises(RuntimeError):
        result = dataset.contour(isosurfaces=pyvista.PolyData())
    dataset = examples.load_airplane()
    with pytest.raises(AssertionError):
        result = dataset.contour()


def test_elevation():
    dataset = examples.load_uniform()
    # Test default params
    elev = dataset.elevation()
    assert 'Elevation' in elev.scalar_names
    assert 'Elevation' == elev.active_scalar_name
    assert elev.get_data_range() == (dataset.bounds[4], dataset.bounds[5])
    # test vector args
    c = list(dataset.center)
    t = list(c) # cast so it doesnt point to `c`
    t[2] = dataset.bounds[-1]
    elev = dataset.elevation(low_point=c, high_point=t)
    assert 'Elevation' in elev.scalar_names
    assert 'Elevation' == elev.active_scalar_name
    assert elev.get_data_range() == (dataset.center[2], dataset.bounds[5])
    # Test not setting active
    elev = dataset.elevation(set_active=False)
    assert 'Elevation' in elev.scalar_names
    assert 'Elevation' != elev.active_scalar_name
    # Set use a range by scalar name
    elev = dataset.elevation(scalar_range='Spatial Point Data')
    assert 'Elevation' in elev.scalar_names
    assert 'Elevation' == elev.active_scalar_name
    assert dataset.get_data_range('Spatial Point Data') == (elev.get_data_range('Elevation'))
    # Set use a user defined range
    elev = dataset.elevation(scalar_range=[1.0, 100.0])
    assert 'Elevation' in elev.scalar_names
    assert 'Elevation' == elev.active_scalar_name
    assert elev.get_data_range('Elevation') == (1.0, 100.0)
    # test errors
    with pytest.raises(RuntimeError):
        elev = dataset.elevation(scalar_range=0.5)


def test_texture_map_to_plane():
    dataset = examples.load_airplane()
    # Automatically decide plane
    out = dataset.texture_map_to_plane(inplace=False)
    assert isinstance(out, type(dataset))
    # Define the plane explicitly
    bnds = dataset.bounds
    origin = bnds[0::2]
    point_u = (bnds[1], bnds[2], bnds[4])
    point_v = (bnds[0], bnds[3], bnds[4])
    out = dataset.texture_map_to_plane(origin=origin, point_u=point_u, point_v=point_v)
    assert isinstance(out, type(dataset))
    assert 'Texture Coordinates' in out.scalar_names
    # FINAL: Test in place modifiacation
    dataset.texture_map_to_plane(inplace=True)
    assert 'Texture Coordinates' in dataset.scalar_names



def test_compute_cell_sizes():
    for i, dataset in enumerate(datasets):
        result = dataset.compute_cell_sizes()
        assert result is not None
        assert isinstance(result, type(dataset))
        assert 'Area' in result.scalar_names
        assert 'Volume' in result.scalar_names
    # Test the volume property
    grid = pyvista.UniformGrid((10,10,10))
    volume = float(np.prod(np.array(grid.dimensions) - 1))
    assert np.allclose(grid.volume, volume)



def test_cell_centers():
    for i, dataset in enumerate(datasets):
        result = dataset.cell_centers()
        assert result is not None
        assert isinstance(result, pyvista.PolyData)

def test_glyph():
    for i, dataset in enumerate(datasets):
        result = dataset.glyph()
        assert result is not None
        assert isinstance(result, pyvista.PolyData)
    # Test different options for glyph filter
    sphere = pyvista.Sphere(radius=3.14)
    # make cool swirly pattern
    vectors = np.vstack((np.sin(sphere.points[:, 0]),
    np.cos(sphere.points[:, 1]),
    np.cos(sphere.points[:, 2]))).T
    # add and scale
    sphere.vectors = vectors*0.3
    sphere.point_arrays['foo'] = np.random.rand(sphere.n_points)
    sphere.point_arrays['arr'] = np.ones(sphere.n_points)
    result = sphere.glyph(scale='arr')
    result = sphere.glyph(scale='arr', orient='Normals', factor=0.1)


def test_split_and_connectivity():
    # Load a simple example mesh
    dataset = examples.load_uniform()
    dataset.set_active_scalar('Spatial Cell Data')
    threshed = dataset.threshold_percent([0.15, 0.50], invert=True)

    bodies = threshed.split_bodies()

    volumes = [518.0, 35.0]
    assert len(volumes) == bodies.n_blocks
    for i, body in enumerate(bodies):
        assert np.allclose(body.volume, volumes[i], rtol=0.1)


def test_warp_by_scalar():
    data = examples.load_uniform()
    warped = data.warp_by_scalar()
    assert data.n_points == warped.n_points


def test_cell_data_to_point_data():
    data = examples.load_uniform()
    foo = data.cell_data_to_point_data()
    assert foo.n_scalars == 2
    assert len(foo.cell_arrays.keys()) == 0

def test_point_data_to_cell_data():
    data = examples.load_uniform()
    foo = data.point_data_to_cell_data()
    assert foo.n_scalars == 2
    assert len(foo.point_arrays.keys()) == 0


def test_triangulate():
    data = examples.load_uniform()
    tri = data.triangulate()
    assert isinstance(tri, pyvista.UnstructuredGrid)
    assert np.any(tri.cells)


def test_delaunay_3d():
    data = examples.load_uniform().threshold_percent(30)
    result = data.delaunay_3d()
    assert np.any(result.points)


def test_smooth():
    data = examples.load_uniform()
    vol = data.threshold_percent(30)
    surf = vol.extract_geometry()
    smooth = surf.smooth()
    assert np.any(smooth.points)


def test_resample():
    mesh = pyvista.Sphere(center=(4.5,4.5,4.5), radius=4.5)
    data_to_probe = examples.load_uniform()
    result = mesh.sample(data_to_probe)
    name = 'Spatial Point Data'
    assert name in result.scalar_names
    assert isinstance(result, type(mesh))


def test_streamlines():
    mesh = examples.download_carotid()
    stream, src = mesh.streamlines(return_source=True, max_time=100.0,
                            initial_step_length=2., terminal_speed=0.1,
                           n_points=25, source_radius=2.0,
                           source_center=(133.1, 116.3, 5.0) )
    assert stream.n_points > 0
    assert src.n_points == 25
