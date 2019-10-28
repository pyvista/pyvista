import numpy as np
import pytest
import sys

import pyvista
from pyvista import examples

PYTHON_2 = int(sys.version[0]) < 3

DATASETS = [
    examples.load_uniform(), # UniformGrid
    examples.load_rectilinear(), # RectilinearGrid
    examples.load_hexbeam(), # UnstructuredGrid
    examples.load_airplane(), # PolyData
    examples.load_structured(), # StructuredGrid
]
normals = ['x', 'y', '-z', (1,1,1), (3.3, 5.4, 0.8)]

COMPOSITE = pyvista.MultiBlock(DATASETS, deep=True)


def test_clip_filter():
    """This tests the clip filter on all datatypes avaialble filters"""
    for i, dataset in enumerate(DATASETS):
        clp = dataset.clip(normal=normals[i], invert=True)
        assert clp is not None
        if isinstance(dataset, pyvista.PolyData):
            assert isinstance(clp, pyvista.PolyData)
        else:
            assert isinstance(clp, pyvista.UnstructuredGrid)


@pytest.mark.skipif(PYTHON_2, reason="Python 2 doesn't support binding methods")
def test_clip_filter_composite():
    # Now test composite data structures
    output = COMPOSITE.clip(normal=normals[0], invert=False)
    assert output.n_blocks == COMPOSITE.n_blocks


def test_clip_box():
    for i, dataset in enumerate(DATASETS):
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


@pytest.mark.skipif(PYTHON_2, reason="Python 2 doesn't support binding methods")
def test_clip_box_composite():
    # Now test composite data structures
    output = COMPOSITE.clip_box(invert=False)
    assert output.n_blocks == COMPOSITE.n_blocks


def test_clip_surface():
    surface = pyvista.Cone(direction=(0,0,-1),
                           height=3.0, radius=1, resolution=50, )
    xx = yy = zz = 1 - np.linspace(0, 51, 51) * 2 / 50
    dataset = pyvista.RectilinearGrid(xx, yy, zz)
    clipped = dataset.clip_surface(surface, invert=False)
    assert clipped.n_points < dataset.n_points
    clipped = dataset.clip_surface(surface, invert=False, compute_distance=True)
    assert clipped.n_points < dataset.n_points
    assert 'implicit_distance' in clipped.array_names


def test_slice_filter():
    """This tests the slice filter on all datatypes avaialble filters"""
    for i, dataset in enumerate(DATASETS):
        slc = dataset.slice(normal=normals[i])
        assert slc is not None
        assert isinstance(slc, pyvista.PolyData)
    dataset = examples.load_uniform()
    slc = dataset.slice(contour=True)
    assert slc is not None
    assert isinstance(slc, pyvista.PolyData)
    result = dataset.slice(origin=(10, 15, 15))
    assert result.n_points < 1


@pytest.mark.skipif(PYTHON_2, reason="Python 2 doesn't support binding methods")
def test_slice_filter_composite():
    # Now test composite data structures
    output = COMPOSITE.slice(normal=normals[0])
    assert output.n_blocks == COMPOSITE.n_blocks


def test_slice_orthogonal_filter():
    """This tests the slice filter on all datatypes avaialble filters"""

    for i, dataset in enumerate(DATASETS):
        slices = dataset.slice_orthogonal()
        assert slices is not None
        assert isinstance(slices, pyvista.MultiBlock)
        assert slices.n_blocks == 3
        for slc in slices:
            assert isinstance(slc, pyvista.PolyData)


@pytest.mark.skipif(PYTHON_2, reason="Python 2 doesn't support binding methods")
def test_slice_orthogonal_filter_composite():
    # Now test composite data structures
    output = COMPOSITE.slice_orthogonal()
    assert output.n_blocks == COMPOSITE.n_blocks


def test_slice_along_axis():
    """Test the many slices along axis filter """
    axii = ['x', 'y', 'z', 'y', 0]
    ns = [2, 3, 4, 10, 20, 13]
    for i, dataset in enumerate(DATASETS):
        slices = dataset.slice_along_axis(n=ns[i], axis=axii[i])
        assert slices is not None
        assert isinstance(slices, pyvista.MultiBlock)
        assert slices.n_blocks == ns[i]
        for slc in slices:
            assert isinstance(slc, pyvista.PolyData)
    dataset = examples.load_uniform()
    with pytest.raises(RuntimeError):
        dataset.slice_along_axis(axis='u')


@pytest.mark.skipif(PYTHON_2, reason="Python 2 doesn't support binding methods")
def test_slice_along_axis_composite():
    # Now test composite data structures
    output = COMPOSITE.slice_along_axis()
    assert output.n_blocks == COMPOSITE.n_blocks


def test_threshold():
    for i, dataset in enumerate(DATASETS[0:3]):
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
    # Now test DATASETS without arrays
    with pytest.raises(AssertionError):
        for i, dataset in enumerate(DATASETS[3:-1]):
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
    for i, dataset in enumerate(DATASETS[0:3]):
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
    for i, dataset in enumerate(DATASETS):
        outline = dataset.outline()
        assert outline is not None
        assert isinstance(outline, pyvista.PolyData)


@pytest.mark.skipif(PYTHON_2, reason="Python 2 doesn't support binding methods")
def test_outline_composite():
    # Now test composite data structures
    output = COMPOSITE.outline()
    assert isinstance(output, pyvista.PolyData)
    output = COMPOSITE.outline(nested=True)
    assert output.n_blocks == COMPOSITE.n_blocks


def test_outline_corners():
    for i, dataset in enumerate(DATASETS):
        outline = dataset.outline_corners()
        assert outline is not None
        assert isinstance(outline, pyvista.PolyData)


@pytest.mark.skipif(PYTHON_2, reason="Python 2 doesn't support binding methods")
def test_outline_corners_composite():
    # Now test composite data structures
    output = COMPOSITE.outline_corners()
    assert isinstance(output, pyvista.PolyData)
    output = COMPOSITE.outline_corners(nested=True)
    assert output.n_blocks == COMPOSITE.n_blocks


def test_extract_geometry():
    for i, dataset in enumerate(DATASETS):
        outline = dataset.extract_geometry()
        assert outline is not None
        assert isinstance(outline, pyvista.PolyData)
    # Now test composite data structures
    output = COMPOSITE.extract_geometry()
    assert isinstance(output, pyvista.PolyData)


def test_wireframe():
    for i, dataset in enumerate(DATASETS):
        wire = dataset.wireframe()
        assert wire is not None
        assert isinstance(wire, pyvista.PolyData)


@pytest.mark.skipif(PYTHON_2, reason="Python 2 doesn't support binding methods")
def test_wireframe_composite():
    # Now test composite data structures
    output = COMPOSITE.wireframe()
    assert output.n_blocks == COMPOSITE.n_blocks

@pytest.mark.parametrize('method', ['contour', 'marching_cubes',
                                    'flying_edges'])
def test_contour(method):
    dataset = examples.load_uniform()
    iso = dataset.contour(method=method)
    assert iso is not None
    iso = dataset.contour(isosurfaces=[100, 300, 500], method=method)
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
    assert 'Elevation' in elev.array_names
    assert 'Elevation' == elev.active_scalar_name
    assert elev.get_data_range() == (dataset.bounds[4], dataset.bounds[5])
    # test vector args
    c = list(dataset.center)
    t = list(c) # cast so it doesnt point to `c`
    t[2] = dataset.bounds[-1]
    elev = dataset.elevation(low_point=c, high_point=t)
    assert 'Elevation' in elev.array_names
    assert 'Elevation' == elev.active_scalar_name
    assert elev.get_data_range() == (dataset.center[2], dataset.bounds[5])
    # Test not setting active
    elev = dataset.elevation(set_active=False)
    assert 'Elevation' in elev.array_names
    assert 'Elevation' != elev.active_scalar_name
    # Set use a range by scalar name
    elev = dataset.elevation(scalar_range='Spatial Point Data')
    assert 'Elevation' in elev.array_names
    assert 'Elevation' == elev.active_scalar_name
    assert dataset.get_data_range('Spatial Point Data') == (elev.get_data_range('Elevation'))
    # Set use a user defined range
    elev = dataset.elevation(scalar_range=[1.0, 100.0])
    assert 'Elevation' in elev.array_names
    assert 'Elevation' == elev.active_scalar_name
    assert elev.get_data_range('Elevation') == (1.0, 100.0)
    # test errors
    with pytest.raises(RuntimeError):
        elev = dataset.elevation(scalar_range=0.5)


@pytest.mark.skipif(PYTHON_2, reason="Python 2 doesn't support binding methods")
def test_elevation_composite():
    # Now test composite data structures
    output = COMPOSITE.elevation()
    assert output.n_blocks == COMPOSITE.n_blocks


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
    assert 'Texture Coordinates' in out.array_names
    # FINAL: Test in place modifiacation
    dataset.texture_map_to_plane(inplace=True)
    assert 'Texture Coordinates' in dataset.array_names



def test_compute_cell_sizes():
    for i, dataset in enumerate(DATASETS):
        result = dataset.compute_cell_sizes()
        assert result is not None
        assert isinstance(result, type(dataset))
        assert 'Area' in result.array_names
        assert 'Volume' in result.array_names
    # Test the volume property
    grid = pyvista.UniformGrid((10,10,10))
    volume = float(np.prod(np.array(grid.dimensions) - 1))
    assert np.allclose(grid.volume, volume)


@pytest.mark.skipif(PYTHON_2, reason="Python 2 doesn't support binding methods")
def test_compute_cell_sizes_composite():
    # Now test composite data structures
    output = COMPOSITE.compute_cell_sizes()
    assert output.n_blocks == COMPOSITE.n_blocks


def test_cell_centers():
    for i, dataset in enumerate(DATASETS):
        result = dataset.cell_centers()
        assert result is not None
        assert isinstance(result, pyvista.PolyData)


@pytest.mark.skipif(PYTHON_2, reason="Python 2 doesn't support binding methods")
def test_cell_centers_composite():
    # Now test composite data structures
    output = COMPOSITE.cell_centers()
    assert output.n_blocks == COMPOSITE.n_blocks


def test_glyph():
    for i, dataset in enumerate(DATASETS):
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
    result = sphere.glyph(scale='arr', orient='Normals', factor=0.1, tolerance=0.1)


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
    warped = data.warp_by_scalar(scale_factor=3)
    assert data.n_points == warped.n_points
    warped = data.warp_by_scalar(normal=[1,1,3])
    assert data.n_points == warped.n_points
    # Test in place!
    foo = examples.load_hexbeam()
    warped = foo.warp_by_scalar()
    foo.warp_by_scalar(inplace=True)
    assert np.allclose(foo.points, warped.points)



def test_cell_data_to_point_data():
    data = examples.load_uniform()
    foo = data.cell_data_to_point_data()
    assert foo.n_arrays == 2
    assert len(foo.cell_arrays.keys()) == 0


@pytest.mark.skipif(PYTHON_2, reason="Python 2 doesn't support binding methods")
def test_cell_data_to_point_data_composite():
    # Now test composite data structures
    output = COMPOSITE.cell_data_to_point_data()
    assert output.n_blocks == COMPOSITE.n_blocks



def test_point_data_to_cell_data():
    data = examples.load_uniform()
    foo = data.point_data_to_cell_data()
    assert foo.n_arrays == 2
    assert len(foo.point_arrays.keys()) == 0


@pytest.mark.skipif(PYTHON_2, reason="Python 2 doesn't support binding methods")
def test_point_data_to_cell_data_composite():
    # Now test composite data structures
    output = COMPOSITE.point_data_to_cell_data()
    assert output.n_blocks == COMPOSITE.n_blocks


def test_triangulate():
    data = examples.load_uniform()
    tri = data.triangulate()
    assert isinstance(tri, pyvista.UnstructuredGrid)
    assert np.any(tri.cells)


@pytest.mark.skipif(PYTHON_2, reason="Python 2 doesn't support binding methods")
def test_triangulate_composite():
    # Now test composite data structures
    output = COMPOSITE.triangulate()
    assert output.n_blocks == COMPOSITE.n_blocks


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
    assert name in result.array_names
    assert isinstance(result, type(mesh))
    result = mesh.sample(data_to_probe, tolerance=1.0)
    name = 'Spatial Point Data'
    assert name in result.array_names
    assert isinstance(result, type(mesh))


def test_streamlines():
    mesh = examples.download_carotid()
    stream, src = mesh.streamlines(return_source=True, max_time=100.0,
                                   initial_step_length=2., terminal_speed=0.1,
                                   n_points=25, source_radius=2.0,
                                   source_center=(133.1, 116.3, 5.0))
    assert stream.n_points > 0
    assert src.n_points == 25


def test_plot_over_line():
    """this requires matplotlib"""
    mesh = examples.load_uniform()
    # Make two points to construct the line between
    a = [mesh.bounds[0], mesh.bounds[2], mesh.bounds[4]]
    b = [mesh.bounds[1], mesh.bounds[3], mesh.bounds[5]]
    mesh.plot_over_line(a, b, resolution=1000, show=False)
    # Test multicomponent
    mesh['foo'] = np.random.rand(mesh.n_cells, 3)
    mesh.plot_over_line(a, b, resolution=None, scalars='foo',
                        title='My Stuff', ylabel='3 Values', show=False)


def test_slice_along_line():
    model = examples.load_uniform()
    n = 5
    x = y = z = np.linspace(model.bounds[0], model.bounds[1], num=n)
    points = np.c_[x,y,z]
    spline = pyvista.Spline(points, n)
    slc = model.slice_along_line(spline)
    assert slc.n_points > 0
    slc = model.slice_along_line(spline, contour=True)
    assert slc.n_points > 0
    # Now check a simple line
    a = [model.bounds[0], model.bounds[2], model.bounds[4]]
    b = [model.bounds[1], model.bounds[3], model.bounds[5]]
    line = pyvista.Line(a, b, resolution=10)
    slc = model.slice_along_line(line)
    assert slc.n_points > 0
    # Now check a bad input
    a = [model.bounds[0], model.bounds[2], model.bounds[4]]
    b = [model.bounds[1], model.bounds[2], model.bounds[5]]
    line2 = pyvista.Line(a, b, resolution=10)
    line = line2.cast_to_unstructured_grid().merge(line.cast_to_unstructured_grid())
    with pytest.raises(AssertionError):
        slc = model.slice_along_line(line)


@pytest.mark.skipif(PYTHON_2, reason="Python 2 doesn't support binding methods")
def test_slice_along_line_composite():
    # Now test composite data structures
    a = [COMPOSITE.bounds[0], COMPOSITE.bounds[2], COMPOSITE.bounds[4]]
    b = [COMPOSITE.bounds[1], COMPOSITE.bounds[3], COMPOSITE.bounds[5]]
    line = pyvista.Line(a, b, resolution=10)
    output = COMPOSITE.slice_along_line(line)
    assert output.n_blocks == COMPOSITE.n_blocks


def test_interpolate():
    surface = examples.download_saddle_surface()
    points = examples.download_sparse_points()
    # Run the interpolation
    interpolated = surface.interpolate(points, radius=12.0)
    assert interpolated.n_points
    assert interpolated.n_arrays


def test_select_enclosed_points():
    mesh = examples.load_uniform()
    surf = pyvista.Sphere(center=mesh.center, radius=mesh.length/2.)
    result = mesh.select_enclosed_points(surf)
    assert isinstance(result, type(mesh))
    assert 'SelectedPoints' in result.array_names
    assert result.n_arrays == mesh.n_arrays + 1
    # Now check non-closed surface
    mesh = pyvista.ParametricEllipsoid(0.2, 0.7, 0.7, )
    surf = mesh.copy()
    surf.rotate_x(90)
    result = mesh.select_enclosed_points(surf, check_surface=False)
    assert isinstance(result, type(mesh))
    assert 'SelectedPoints' in result.array_names
    assert result.n_arrays == mesh.n_arrays + 1
    with pytest.raises(RuntimeError):
        result = mesh.select_enclosed_points(surf, check_surface=True)


def test_decimate_boundary():
    mesh = examples.load_uniform()
    boundary = mesh.decimate_boundary()
    assert boundary.n_points


def test_merge_general():
    mesh = examples.load_uniform()
    thresh = mesh.threshold_percent([0.2, 0.5]) # unstructured grid
    con = mesh.contour() # poly data
    merged = thresh + con
    assert isinstance(merged, pyvista.UnstructuredGrid)
    merged = con + thresh
    assert isinstance(merged, pyvista.UnstructuredGrid)
    # Pure PolyData inputs should yield poly data output
    merged = mesh.extract_surface() + con
    assert isinstance(merged, pyvista.PolyData)


def test_compute_cell_quality():
    mesh = pyvista.ParametricEllipsoid().decimate(0.8)
    qual = mesh.compute_cell_quality()
    assert 'CellQuality' in qual.array_names
    with pytest.raises(KeyError):
        qual = mesh.compute_cell_quality(quality_measure='foo')


def test_compute_gradients():
    mesh = examples.load_random_hills()
    grad = mesh.compute_gradient()
    assert 'gradient' in grad.array_names
    assert np.shape(grad['gradient'])[0] == mesh.n_points
    assert np.shape(grad['gradient'])[1] == 3
