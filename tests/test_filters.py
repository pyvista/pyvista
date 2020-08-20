import sys

import numpy as np
import pytest

import pyvista
from pyvista import examples

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False

DATASETS = [
    examples.load_uniform(), # UniformGrid
    examples.load_rectilinear(), # RectilinearGrid
    examples.load_hexbeam(), # UnstructuredGrid
    examples.load_airplane(), # PolyData
    examples.load_structured(), # StructuredGrid
]
normals = ['x', 'y', '-z', (1,1,1), (3.3, 5.4, 0.8)]

COMPOSITE = pyvista.MultiBlock(DATASETS, deep=True)

skip_py2_nobind = pytest.mark.skipif(int(sys.version[0]) < 3,
                                     reason="Python 2 doesn't support binding methods")


@pytest.fixture(scope='module')
def uniform_vec():
    nx, ny, nz = 20, 15, 5
    origin = (-(nx - 1)*0.1/2, -(ny - 1)*0.1/2, -(nz - 1)*0.1/2)
    mesh = pyvista.UniformGrid((nx, ny, nz), (.1, .1, .1), origin)
    mesh['vectors'] = mesh.points
    return mesh


def test_datasetfilters_init():
    with pytest.raises(TypeError):
        pyvista.core.filters.DataSetFilters()


def test_clip_filter():
    """This tests the clip filter on all datatypes available filters"""
    for i, dataset in enumerate(DATASETS):
        clp = dataset.clip(normal=normals[i], invert=True)
        assert clp is not None
        if isinstance(dataset, pyvista.PolyData):
            assert isinstance(clp, pyvista.PolyData)
        else:
            assert isinstance(clp, pyvista.UnstructuredGrid)


@skip_py2_nobind
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
    assert result.n_cells
    with pytest.raises(ValueError):
        dataset.clip_box(bounds=(5, 6,))
    # allow Sequence but not Iterable bounds
    with pytest.raises(TypeError):
        dataset.clip_box(bounds={5, 6, 7})
    # Test with a poly data box
    mesh = examples.load_airplane()
    box = pyvista.Cube(center=(0.9e3, 0.2e3, mesh.center[2]),
                       x_length=500, y_length=500, z_length=500)
    box.rotate_z(33)
    result = mesh.clip_box(box, invert=False)
    assert result.n_cells
    result = mesh.clip_box(box, invert=True)
    assert result.n_cells

    with pytest.raises(ValueError):
        dataset.clip_box(bounds=pyvista.Sphere())


@skip_py2_nobind
def test_clip_box_composite():
    # Now test composite data structures
    output = COMPOSITE.clip_box(invert=False)
    assert output.n_blocks == COMPOSITE.n_blocks


def test_clip_surface():
    surface = pyvista.Cone(direction=(0,0,-1),
                           height=3.0, radius=1, resolution=50, )
    xx = yy = zz = 1 - np.linspace(0, 51, 11) * 2 / 50
    dataset = pyvista.RectilinearGrid(xx, yy, zz)
    clipped = dataset.clip_surface(surface, invert=False)
    assert isinstance(clipped, pyvista.UnstructuredGrid)
    clipped = dataset.clip_surface(surface, invert=False, compute_distance=True)
    assert isinstance(clipped, pyvista.UnstructuredGrid)
    assert 'implicit_distance' in clipped.array_names
    clipped = dataset.clip_surface(surface.cast_to_unstructured_grid(),)
    assert isinstance(clipped, pyvista.UnstructuredGrid)
    assert 'implicit_distance' in clipped.array_names


def test_clip_closed_surface():
    closed_surface = pyvista.Sphere()
    clipped = closed_surface.clip_closed_surface()
    assert closed_surface.n_open_edges == 0
    open_surface = closed_surface.clip()
    with pytest.raises(ValueError):
        _ = open_surface.clip_closed_surface()


def test_implicit_distance():
    surface = pyvista.Cone(direction=(0,0,-1),
                           height=3.0, radius=1, resolution=50, )
    xx = yy = zz = 1 - np.linspace(0, 51, 11) * 2 / 50
    dataset = pyvista.RectilinearGrid(xx, yy, zz)
    res = dataset.compute_implicit_distance(surface)
    assert "implicit_distance" in res.point_arrays
    dataset.compute_implicit_distance(surface, inplace=True)
    assert "implicit_distance" in dataset.point_arrays


def test_slice_filter():
    """This tests the slice filter on all datatypes available filters"""
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


@skip_py2_nobind
def test_slice_filter_composite():
    # Now test composite data structures
    output = COMPOSITE.slice(normal=normals[0])
    assert output.n_blocks == COMPOSITE.n_blocks


def test_slice_orthogonal_filter():
    """This tests the slice filter on all datatypes available filters"""

    for i, dataset in enumerate(DATASETS):
        slices = dataset.slice_orthogonal()
        assert slices is not None
        assert isinstance(slices, pyvista.MultiBlock)
        assert slices.n_blocks == 3
        for slc in slices:
            assert isinstance(slc, pyvista.PolyData)


@skip_py2_nobind
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
    with pytest.raises(ValueError):
        dataset.slice_along_axis(axis='u')


@skip_py2_nobind
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
    # allow Sequence but not Iterable
    with pytest.raises(TypeError):
        dataset.threshold({100, 500})
    # Now test DATASETS without arrays
    with pytest.raises(ValueError):
        for i, dataset in enumerate(DATASETS[3:-1]):
            thresh = dataset.threshold()
            assert thresh is not None
            assert isinstance(thresh, pyvista.UnstructuredGrid)
    dataset = examples.load_uniform()
    with pytest.raises(ValueError):
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
    with pytest.raises(ValueError):
        result = dataset.threshold_percent(20000)
    with pytest.raises(ValueError):
        result = dataset.threshold_percent(0.0)
    # allow Sequence but not Iterable
    with pytest.raises(TypeError):
        dataset.threshold_percent({18.0, 85.0})


def test_outline():
    for i, dataset in enumerate(DATASETS):
        outline = dataset.outline()
        assert outline is not None
        assert isinstance(outline, pyvista.PolyData)


@skip_py2_nobind
def test_outline_composite():
    # Now test composite data structures
    output = COMPOSITE.outline()
    assert isinstance(output, pyvista.PolyData)
    output = COMPOSITE.outline(nested=True)

    # vtk 9.0.0 returns polydata
    assert isinstance(output, (pyvista.MultiBlock, pyvista.PolyData))
    if isinstance(output, pyvista.MultiBlock):
        assert output.n_blocks == COMPOSITE.n_blocks


def test_outline_corners():
    for i, dataset in enumerate(DATASETS):
        outline = dataset.outline_corners()
        assert outline is not None
        assert isinstance(outline, pyvista.PolyData)


@skip_py2_nobind
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
        wire = dataset.extract_all_edges()
        assert wire is not None
        assert isinstance(wire, pyvista.PolyData)


@skip_py2_nobind
def test_wireframe_composite():
    # Now test composite data structures
    output = COMPOSITE.extract_all_edges()
    assert output.n_blocks == COMPOSITE.n_blocks


def test_delaunay_2d():
    mesh = DATASETS[2].delaunay_2d()  # UnstructuredGrid
    assert isinstance(mesh, pyvista.PolyData)
    assert mesh.n_points


@pytest.mark.parametrize('method', ['contour', 'marching_cubes',
                                    'flying_edges'])
def test_contour(uniform, method):
    iso = uniform.contour(method=method)
    assert iso is not None
    iso = uniform.contour(isosurfaces=[100, 300, 500], method=method)
    assert iso is not None


def test_contour_errors(uniform):
    with pytest.raises(TypeError):
        uniform.contour(scalars='Spatial Cell Data')
    with pytest.raises(TypeError):
        uniform.contour(isosurfaces=pyvista.PolyData())
    with pytest.raises(TypeError):
        uniform.contour(isosurfaces={100, 300, 500})
    uniform = examples.load_airplane()
    with pytest.raises(ValueError):
        uniform.contour()
    with pytest.raises(ValueError):
        uniform.contour(method='invalid method')


def test_elevation():
    dataset = examples.load_uniform()
    # Test default params
    elev = dataset.elevation()
    assert 'Elevation' in elev.array_names
    assert 'Elevation' == elev.active_scalars_name
    assert elev.get_data_range() == (dataset.bounds[4], dataset.bounds[5])
    # test vector args
    c = list(dataset.center)
    t = list(c) # cast so it does not point to `c`
    t[2] = dataset.bounds[-1]
    elev = dataset.elevation(low_point=c, high_point=t)
    assert 'Elevation' in elev.array_names
    assert 'Elevation' == elev.active_scalars_name
    assert elev.get_data_range() == (dataset.center[2], dataset.bounds[5])
    # Test not setting active
    elev = dataset.elevation(set_active=False)
    assert 'Elevation' in elev.array_names
    assert 'Elevation' != elev.active_scalars_name
    # Set use a range by scalar name
    elev = dataset.elevation(scalar_range='Spatial Point Data')
    assert 'Elevation' in elev.array_names
    assert 'Elevation' == elev.active_scalars_name
    assert dataset.get_data_range('Spatial Point Data') == (elev.get_data_range('Elevation'))
    # Set use a user defined range
    elev = dataset.elevation(scalar_range=[1.0, 100.0])
    assert 'Elevation' in elev.array_names
    assert 'Elevation' == elev.active_scalars_name
    assert elev.get_data_range('Elevation') == (1.0, 100.0)
    # test errors
    with pytest.raises(TypeError):
        elev = dataset.elevation(scalar_range=0.5)
    with pytest.raises(ValueError):
        elev = dataset.elevation(scalar_range=[1, 2, 3])
    with pytest.raises(TypeError):
        elev = dataset.elevation(scalar_range={1, 2})


@skip_py2_nobind
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


@skip_py2_nobind
def test_compute_cell_sizes_composite():
    # Now test composite data structures
    output = COMPOSITE.compute_cell_sizes()
    assert output.n_blocks == COMPOSITE.n_blocks


def test_cell_centers():
    for i, dataset in enumerate(DATASETS):
        result = dataset.cell_centers()
        assert result is not None
        assert isinstance(result, pyvista.PolyData)


@skip_py2_nobind
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
    result = sphere.glyph(scale=False)
    result = sphere.glyph(scale='arr')
    result = sphere.glyph(scale='arr', orient='Normals', factor=0.1)
    result = sphere.glyph(scale='arr', orient='Normals', factor=0.1, tolerance=0.1)
    result = sphere.glyph(scale='arr', orient='Normals', factor=0.1, tolerance=0.1,
                          clamping=False, rng=[1, 1])


def test_split_and_connectivity():
    # Load a simple example mesh
    dataset = examples.load_uniform()
    dataset.set_active_scalars('Spatial Cell Data')
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


def test_warp_by_vector():
    # Test when inplace=False (default)
    data = examples.load_sphere_vectors()
    warped = data.warp_by_vector()
    assert data.n_points == warped.n_points
    assert not np.allclose(data.points, warped.points)
    warped = data.warp_by_vector(factor=3)
    assert data.n_points == warped.n_points
    assert not np.allclose(data.points, warped.points)
    # Test when inplace=True
    foo = examples.load_sphere_vectors()
    warped = foo.warp_by_vector()
    foo.warp_by_vector(inplace=True)
    assert np.allclose(foo.points, warped.points)


def test_invalid_warp_scalar(sphere):
    sphere['cellscalars'] = np.random.random(sphere.n_cells)
    sphere.point_arrays.clear()
    with pytest.raises(TypeError):
        sphere.warp_by_scalar()


def test_invalid_warp_scalar_inplace(uniform):
    with pytest.raises(TypeError):
        uniform.warp_by_scalar(inplace=True)


def test_invalid_warp_vector(sphere):
    # bad vectors
    sphere.point_arrays['Normals'] = np.empty((sphere.n_points, 2))
    with pytest.raises(ValueError):
        sphere.warp_by_vector('Normals')

    # no vectors
    sphere.point_arrays.clear()
    with pytest.raises(TypeError):
        sphere.warp_by_vector()


def test_cell_data_to_point_data():
    data = examples.load_uniform()
    foo = data.cell_data_to_point_data()
    assert foo.n_arrays == 2
    assert len(foo.cell_arrays.keys()) == 0
    _ = data.ctp()


@skip_py2_nobind
def test_cell_data_to_point_data_composite():
    # Now test composite data structures
    output = COMPOSITE.cell_data_to_point_data()
    assert output.n_blocks == COMPOSITE.n_blocks


def test_point_data_to_cell_data():
    data = examples.load_uniform()
    foo = data.point_data_to_cell_data()
    assert foo.n_arrays == 2
    assert len(foo.point_arrays.keys()) == 0
    _ = data.ptc()


@skip_py2_nobind
def test_point_data_to_cell_data_composite():
    # Now test composite data structures
    output = COMPOSITE.point_data_to_cell_data()
    assert output.n_blocks == COMPOSITE.n_blocks


def test_triangulate():
    data = examples.load_uniform()
    tri = data.triangulate()
    assert isinstance(tri, pyvista.UnstructuredGrid)
    assert np.any(tri.cells)


@skip_py2_nobind
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


@pytest.mark.parametrize('use_points', [True, False])
@pytest.mark.parametrize('categorical', [True, False])
def test_probe(categorical, use_points):
    mesh = pyvista.Sphere(center=(4.5, 4.5, 4.5), radius=4.5)
    data_to_probe = examples.load_uniform()
    if use_points:
        dataset = np.array(mesh.points)
    else:
        dataset = mesh
    result = data_to_probe.probe(dataset, tolerance=1E-5, categorical=categorical)
    name = 'Spatial Point Data'
    assert name in result.array_names
    assert isinstance(result, type(mesh))
    result = mesh.sample(data_to_probe, tolerance=1.0)
    name = 'Spatial Point Data'
    assert name in result.array_names
    assert isinstance(result, type(mesh))


@pytest.mark.parametrize('integration_direction', ['forward', 'backward', 'both'])
def test_streamlines_dir(uniform_vec, integration_direction):
    stream = uniform_vec.streamlines('vectors',
                                     integration_direction=integration_direction)
    assert all([stream.n_points, stream.n_cells])


@pytest.mark.parametrize('integrator_type', [2, 4, 45])
def test_streamlines_type(uniform_vec, integrator_type):
    stream = uniform_vec.streamlines('vectors', integrator_type=integrator_type)
    assert all([stream.n_points, stream.n_cells])


@pytest.mark.parametrize('interpolator_type', ['point', 'cell'])
def test_streamlines(uniform_vec, interpolator_type):
    stream = uniform_vec.streamlines('vectors',
                                     interpolator_type=interpolator_type)
    assert all([stream.n_points, stream.n_cells])


def test_streamlines(uniform_vec):
    stream, src = uniform_vec.streamlines('vectors', return_source=True,
                                          pointa=(0.0, 0.0, 0.0),
                                          pointb=(1.1, 1.1, 0.1))
    assert all([stream.n_points, stream.n_cells, src.n_points])

    with pytest.raises(ValueError):
        uniform_vec.streamlines('vectors', integration_direction='not valid')

    with pytest.raises(ValueError):
        uniform_vec.streamlines('vectors', integrator_type=42)

    with pytest.raises(ValueError):
        uniform_vec.streamlines('vectors', interpolator_type='not valid')

    with pytest.raises(ValueError):
        uniform_vec.streamlines('vectors', step_unit='not valid')


def test_sample_over_line():
    """Test that we get a sampled line."""
    name = 'values'

    line = pyvista.Line([0, 0, 0], [0, 0, 10], 9)
    line[name] = np.linspace(0, 10, 10)

    sampled_line = line.sample_over_line([0, 0, 0.5], [0, 0, 1.5], 2)

    expected_result = np.array([0.5, 1, 1.5])
    assert np.allclose(sampled_line[name], expected_result)
    assert name in line.array_names # is name in sampled result

    # test no resolution
    sphere = pyvista.Sphere(center=(4.5,4.5,4.5), radius=4.5)
    sampled_from_sphere = sphere.sample_over_line([3, 1, 1], [-3, -1, -1])
    assert sampled_from_sphere.n_points == sphere.n_cells + 1
    # is sampled result a polydata object
    assert isinstance(sampled_from_sphere, pyvista.PolyData)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Requires matplotlib")
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
    with pytest.raises(ValueError):
        slc = model.slice_along_line(line)

    with pytest.raises(TypeError):
        one_cell = model.extract_cells(0)
        model.slice_along_line(one_cell)


def extract_points_invalid(sphere):
    with pytest.raises(ValueError):
        sphere.extract_points('invalid')

    with pytest.raises(TypeError):
        sphere.extract_points(object)


@skip_py2_nobind
def test_slice_along_line_composite():
    # Now test composite data structures
    a = [COMPOSITE.bounds[0], COMPOSITE.bounds[2], COMPOSITE.bounds[4]]
    b = [COMPOSITE.bounds[1], COMPOSITE.bounds[3], COMPOSITE.bounds[5]]
    line = pyvista.Line(a, b, resolution=10)
    output = COMPOSITE.slice_along_line(line)
    assert output.n_blocks == COMPOSITE.n_blocks


def test_interpolate():
    pdata = pyvista.PolyData()
    pdata.points = np.random.random((10, 3))
    pdata['scalars'] = np.random.random(10)
    surf = pyvista.Sphere(theta_resolution=10, phi_resolution=10)
    interp = surf.interpolate(pdata, radius=0.01)
    assert interp.n_points
    assert interp.n_arrays


def test_select_enclosed_points(uniform, hexbeam):
    surf = pyvista.Sphere(center=uniform.center, radius=uniform.length/2.)
    result = uniform.select_enclosed_points(surf)
    assert isinstance(result, type(uniform))
    assert 'SelectedPoints' in result.array_names
    assert result['SelectedPoints'].any()
    assert result.n_arrays == uniform.n_arrays + 1

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
    with pytest.raises(TypeError):
        result = mesh.select_enclosed_points(hexbeam, check_surface=True)


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

    with pytest.raises(TypeError):
        grad = mesh.compute_gradient(object)

    mesh.point_arrays.clear()
    with pytest.raises(TypeError):
        grad = mesh.compute_gradient()

def test_compute_derivatives():
    mesh = examples.load_random_hills()
    vector = np.zeros((mesh.n_points, 3))
    vector[:,1] = np.ones(mesh.n_points)
    mesh['vector'] = vector
    derv = mesh.compute_derivative(scalars='vector', gradient=True,
                                   divergence=True, vorticity=True,
                                   qcriterion=True)
    assert 'gradient' in derv.array_names
    assert np.shape(derv['gradient'])[0] == mesh.n_points
    assert np.shape(derv['gradient'])[1] == 9

    assert 'divergence' in derv.array_names
    assert np.shape(derv['divergence'])[0] == mesh.n_points
    assert len(np.shape(derv['divergence'])) == 1

    assert 'vorticity' in derv.array_names
    assert np.shape(derv['vorticity'])[0] == mesh.n_points
    assert np.shape(derv['vorticity'])[1] == 3

    assert 'qcriterion' in derv.array_names
    assert np.shape(derv['qcriterion'])[0] == mesh.n_points
    assert len(np.shape(derv['qcriterion'])) == 1

    derv = mesh.compute_derivative(scalars='vector', gradient='gradienttest',
                                   divergence='divergencetest', vorticity='vorticitytest',
                                   qcriterion='qcriteriontest')
    assert 'gradienttest' in derv.array_names
    assert np.shape(derv['gradienttest'])[0] == mesh.n_points
    assert np.shape(derv['gradienttest'])[1] == 9

    assert 'divergencetest' in derv.array_names
    assert np.shape(derv['divergencetest'])[0] == mesh.n_points
    assert len(np.shape(derv['divergencetest'])) == 1

    assert 'vorticitytest' in derv.array_names
    assert np.shape(derv['vorticitytest'])[0] == mesh.n_points
    assert np.shape(derv['vorticitytest'])[1] == 3

    assert 'qcriteriontest' in derv.array_names
    assert np.shape(derv['qcriteriontest'])[0] == mesh.n_points
    assert len(np.shape(derv['qcriteriontest'])) == 1

    grad = mesh.compute_derivative(scalars='Elevation', gradient=True)
    assert 'gradient' in grad.array_names
    assert np.shape(grad['gradient'])[0] == mesh.n_points
    assert np.shape(grad['gradient'])[1] == 3

    grad = mesh.compute_derivative(scalars='Elevation', gradient=True, faster=True)
    assert 'gradient' in grad.array_names
    assert np.shape(grad['gradient'])[0] == mesh.n_points
    assert np.shape(grad['gradient'])[1] == 3

    grad = mesh.compute_derivative(scalars='vector', gradient=True, faster=True)
    assert 'gradient' in grad.array_names
    assert np.shape(grad['gradient'])[0] == mesh.n_points
    assert np.shape(grad['gradient'])[1] == 9

    with pytest.raises(ValueError):
        grad = mesh.compute_derivative(scalars='Elevation', gradient=False)

    with pytest.raises(TypeError):
        derv = mesh.compute_derivative(object)

    mesh.point_arrays.clear()
    with pytest.raises(TypeError):
        derv = mesh.compute_derivative()

def test_extract_subset():
    volume = examples.load_uniform()
    voi = volume.extract_subset([0,3,1,4,5,7])
    assert isinstance(voi, pyvista.UniformGrid)
    # Test that we fix the confusing issue from extents in
    #   https://gitlab.kitware.com/vtk/vtk/-/issues/17938
    assert voi.origin == voi.bounds[::2]


def test_poly_data_strip():
    mesh = examples.load_airplane()
    slc = mesh.slice(normal='z', origin=(0,0,-10))
    stripped = slc.strip()
    assert stripped.n_cells == 1
