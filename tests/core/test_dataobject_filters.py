from __future__ import annotations

import re

import numpy as np
import pytest

import pyvista as pv
from pyvista import VTKVersionError
from pyvista import examples
from pyvista.core import _vtk_core
from tests.core.test_dataset_filters import aprox_le
from tests.core.test_dataset_filters import normals
from tests.core.test_dataset_filters import skip_mac


@pytest.fixture
def composite(datasets):
    return pv.MultiBlock(datasets)


def test_clip_filter(datasets):
    """This tests the clip filter on all datatypes available filters"""
    for i, dataset in enumerate(datasets):
        clp = dataset.clip(normal=normals[i], invert=True)
        assert clp is not None
        if isinstance(dataset, pv.PolyData):
            assert isinstance(clp, pv.PolyData)
        else:
            assert isinstance(clp, pv.UnstructuredGrid)

    # clip with get_clipped=True
    for i, dataset in enumerate(datasets):
        clp1, clp2 = dataset.clip(normal=normals[i], invert=True, return_clipped=True)
        for clp in (clp1, clp2):
            if isinstance(dataset, pv.PolyData):
                assert isinstance(clp, pv.PolyData)
            else:
                assert isinstance(clp, pv.UnstructuredGrid)

    # crinkle clip
    mesh = pv.Wavelet()
    clp = mesh.clip(normal=(1, 1, 1), crinkle=True)
    assert clp is not None
    clp1, clp2 = mesh.clip(normal=(1, 1, 1), return_clipped=True, crinkle=True)
    assert clp1 is not None
    assert clp2 is not None
    set_a = set(clp1.cell_data['cell_ids'])
    set_b = set(clp2.cell_data['cell_ids'])
    assert set_a.isdisjoint(set_b)
    assert set_a.union(set_b) == set(range(mesh.n_cells))


@skip_mac
@pytest.mark.parametrize('both', [False, True])
@pytest.mark.parametrize('invert', [False, True])
def test_clip_by_scalars_filter(datasets, both, invert):
    """This tests the clip filter on all datatypes available filters"""
    for dataset_in in datasets:
        dataset = dataset_in.copy()  # don't modify in-place
        dataset.point_data['to_clip'] = np.arange(dataset.n_points)

        clip_value = dataset.n_points / 2

        if both:
            clps = dataset.clip_scalar(
                scalars='to_clip',
                value=clip_value,
                both=True,
                invert=invert,
            )
            assert len(clps) == 2
            expect_les = (invert, not invert)
        else:
            clps = (
                dataset.clip_scalar(scalars='to_clip', value=clip_value, both=False, invert=invert),
            )
            assert len(clps) == 1
            expect_les = (invert,)

        for clp, expect_le in zip(clps, expect_les):
            assert clp is not None
            if isinstance(dataset, pv.PolyData):
                assert isinstance(clp, pv.PolyData)
            else:
                assert isinstance(clp, pv.UnstructuredGrid)

            if expect_le:
                # VTK clip filter appears to not clip exactly to the clip value.
                # here we allow for a wider range of acceptable values
                assert aprox_le(clp.point_data['to_clip'].max(), clip_value, rtol=1e-1)
            else:
                assert clp.point_data['to_clip'].max() >= clip_value


def test_clip_filter_no_active(sphere):
    # test no active scalars case
    sphere.point_data.set_array(sphere.points[:, 2], 'data')
    assert sphere.active_scalars_name is None
    clp = sphere.clip_scalar()
    assert clp.n_points < sphere.n_points


def test_clip_filter_scalar_multiple():
    mesh = pv.Plane()
    mesh['x'] = mesh.points[:, 0].copy()
    mesh['y'] = mesh.points[:, 1].copy()
    mesh['z'] = mesh.points[:, 2].copy()

    mesh_clip_x = mesh.clip_scalar(scalars='x', value=0.0)
    assert np.isclose(mesh_clip_x['x'].max(), 0.0)
    mesh_clip_y = mesh.clip_scalar(scalars='y', value=0.0)
    assert np.isclose(mesh_clip_y['y'].max(), 0.0)
    mesh_clip_z = mesh.clip_scalar(scalars='z', value=0.0)
    assert np.isclose(mesh_clip_z['z'].max(), 0.0)


def test_clip_filter_composite(composite):
    # Now test composite data structures
    output = composite.clip(normal=normals[0], invert=False)
    assert output.n_blocks == composite.n_blocks


def test_clip_box(datasets):
    for dataset in datasets:
        clp = dataset.clip_box(invert=True, progress_bar=True)
        assert clp is not None
        assert isinstance(clp, pv.UnstructuredGrid)
        clp2 = dataset.clip_box(merge_points=False)
        assert clp2 is not None

    dataset = examples.load_airplane()
    # test length 3 bounds
    result = dataset.clip_box(bounds=(900, 900, 200), invert=False, progress_bar=True)
    dataset = examples.load_uniform()
    result = dataset.clip_box(bounds=0.5, progress_bar=True)
    assert result.n_cells
    with pytest.raises(ValueError):  # noqa: PT011
        dataset.clip_box(bounds=(5, 6), progress_bar=True)
    # allow Sequence but not Iterable bounds
    with pytest.raises(TypeError):
        dataset.clip_box(bounds={5, 6, 7}, progress_bar=True)
    # Test with a poly data box
    mesh = examples.load_airplane()
    box = pv.Cube(center=(0.9e3, 0.2e3, mesh.center[2]), x_length=500, y_length=500, z_length=500)
    box.rotate_z(33, inplace=True)
    result = mesh.clip_box(box, invert=False, progress_bar=True)
    assert result.n_cells
    result = mesh.clip_box(box, invert=True, progress_bar=True)
    assert result.n_cells

    with pytest.raises(ValueError):  # noqa: PT011
        dataset.clip_box(bounds=pv.Sphere(), progress_bar=True)

    # crinkle clip
    surf = pv.Sphere(radius=3)
    vol = pv.voxelize(surf)
    cube = pv.Cube().rotate_x(33, inplace=False)
    clp = vol.clip_box(bounds=cube, invert=False, crinkle=True)
    assert clp is not None


def test_clip_box_composite(composite):
    # Now test composite data structures
    output = composite.clip_box(invert=False, progress_bar=True)
    assert output.n_blocks == composite.n_blocks


def test_clip_surface():
    surface = pv.Cone(
        direction=(0, 0, -1),
        height=3.0,
        radius=1,
        resolution=50,
    )
    xx = yy = zz = 1 - np.linspace(0, 51, 11) * 2 / 50
    dataset = pv.RectilinearGrid(xx, yy, zz)
    clipped = dataset.clip_surface(surface, invert=False, progress_bar=True)
    assert isinstance(clipped, pv.UnstructuredGrid)
    clipped = dataset.clip_surface(surface, invert=False, compute_distance=True, progress_bar=True)
    assert isinstance(clipped, pv.UnstructuredGrid)
    assert 'implicit_distance' in clipped.array_names
    clipped = dataset.clip_surface(surface.cast_to_unstructured_grid(), progress_bar=True)
    assert isinstance(clipped, pv.UnstructuredGrid)
    assert 'implicit_distance' in clipped.array_names
    # Test crinkle
    clipped = dataset.clip_surface(surface, invert=False, progress_bar=True, crinkle=True)
    assert isinstance(clipped, pv.UnstructuredGrid)


def test_clip_closed_surface():
    closed_surface = pv.Sphere()
    clipped = closed_surface.clip_closed_surface(progress_bar=True)
    assert clipped.n_open_edges == 0
    open_surface = closed_surface.clip(progress_bar=True)
    with pytest.raises(ValueError):  # noqa: PT011
        _ = open_surface.clip_closed_surface()


def test_slice_filter(datasets):
    """This tests the slice filter on all datatypes available filters"""
    for i, dataset in enumerate(datasets):
        slc = dataset.slice(normal=normals[i], progress_bar=True)
        assert slc is not None
        assert isinstance(slc, pv.PolyData)
    dataset = examples.load_uniform()
    slc = dataset.slice(contour=True, progress_bar=True)
    assert slc is not None
    assert isinstance(slc, pv.PolyData)
    result = dataset.slice(origin=(10, 15, 15), progress_bar=True)
    assert result.n_points < 1


def test_slice_filter_composite(composite):
    # Now test composite data structures
    output = composite.slice(normal=normals[0], progress_bar=True)
    assert output.n_blocks == composite.n_blocks


def test_slice_orthogonal_filter(datasets):
    """This tests the slice filter on all datatypes available filters"""
    for dataset in datasets:
        slices = dataset.slice_orthogonal(progress_bar=True)
        assert slices is not None
        assert isinstance(slices, pv.MultiBlock)
        assert slices.n_blocks == 3
        for slc in slices:
            assert isinstance(slc, pv.PolyData)


def test_slice_orthogonal_filter_composite(composite):
    # Now test composite data structures
    output = composite.slice_orthogonal(progress_bar=True)
    assert output.n_blocks == composite.n_blocks


def test_slice_along_axis(datasets):
    """Test the many slices along axis filter"""
    axii = ['x', 'y', 'z', 'y', 0]
    ns = [2, 3, 4, 10, 20, 13]
    for i, dataset in enumerate(datasets):
        slices = dataset.slice_along_axis(n=ns[i], axis=axii[i], progress_bar=True)
        assert slices is not None
        assert isinstance(slices, pv.MultiBlock)
        assert slices.n_blocks == ns[i]
        for slc in slices:
            assert isinstance(slc, pv.PolyData)
    dataset = examples.load_uniform()
    with pytest.raises(ValueError):  # noqa: PT011
        dataset.slice_along_axis(axis='u')


def test_slice_along_axis_composite(composite):
    # Now test composite data structures
    output = composite.slice_along_axis(progress_bar=True)
    assert output.n_blocks == composite.n_blocks


def test_slice_along_line():
    model = examples.load_uniform()
    n = 5
    x = y = z = np.linspace(model.bounds.x_min, model.bounds.x_max, num=n)
    points = np.c_[x, y, z]
    spline = pv.Spline(points, n)
    slc = model.slice_along_line(spline, progress_bar=True)
    assert slc.n_points > 0
    slc = model.slice_along_line(spline, contour=True, progress_bar=True)
    assert slc.n_points > 0
    # Now check a simple line
    a = [model.bounds.x_min, model.bounds.y_min, model.bounds.z_min]
    b = [model.bounds.x_max, model.bounds.y_max, model.bounds.z_max]
    line = pv.Line(a, b, resolution=10)
    slc = model.slice_along_line(line, progress_bar=True)
    assert slc.n_points > 0
    # Now check a bad input
    a = [model.bounds.x_min, model.bounds.y_min, model.bounds.z_min]
    b = [model.bounds.x_max, model.bounds.y_min, model.bounds.z_max]
    line2 = pv.Line(a, b, resolution=10)
    line = line2.cast_to_unstructured_grid().merge(line.cast_to_unstructured_grid())
    with pytest.raises(ValueError):  # noqa: PT011
        slc = model.slice_along_line(line, progress_bar=True)

    one_cell = model.extract_cells(0, progress_bar=True)
    with pytest.raises(TypeError):
        model.slice_along_line(one_cell, progress_bar=True)


def test_slice_along_line_composite(composite):
    # Now test composite data structures
    a = [composite.bounds.x_min, composite.bounds.y_min, composite.bounds.z_min]
    b = [composite.bounds.x_max, composite.bounds.y_max, composite.bounds.z_max]
    line = pv.Line(a, b, resolution=10)
    output = composite.slice_along_line(line, progress_bar=True)
    assert output.n_blocks == composite.n_blocks


def test_outline(datasets):
    for dataset in datasets:
        outline = dataset.outline(progress_bar=True)
        assert outline is not None
        assert isinstance(outline, pv.PolyData)


def test_outline_composite(composite):
    # Now test composite data structures
    output = composite.outline(progress_bar=True)
    assert isinstance(output, pv.PolyData)
    output = composite.outline(nested=True, progress_bar=True)

    # vtk 9.0.0 returns polydata
    assert isinstance(output, (pv.MultiBlock, pv.PolyData))
    if isinstance(output, pv.MultiBlock):
        assert output.n_blocks == composite.n_blocks


def test_outline_corners(datasets):
    for dataset in datasets:
        outline = dataset.outline_corners(progress_bar=True)
        assert outline is not None
        assert isinstance(outline, pv.PolyData)


def test_outline_corners_composite(composite):
    # Now test composite data structures
    output = composite.outline_corners(progress_bar=True)
    assert isinstance(output, pv.PolyData)
    output = composite.outline_corners(nested=True)
    assert output.n_blocks == composite.n_blocks


def test_extract_all_edges(datasets):
    for dataset in datasets:
        edges = dataset.extract_all_edges()
        assert edges is not None
        assert isinstance(edges, pv.PolyData)

    if pv.vtk_version_info < (9, 1):
        with pytest.raises(VTKVersionError):
            datasets[0].extract_all_edges(use_all_points=True)
    else:
        edges = datasets[0].extract_all_edges(use_all_points=True)
        assert edges.n_lines


def test_extract_all_edges_no_data():
    mesh = pv.Wavelet()
    edges = mesh.extract_all_edges(clear_data=True)
    assert edges is not None
    assert isinstance(edges, pv.PolyData)
    assert edges.n_arrays == 0


def test_wireframe_composite(composite):
    # Now test composite data structures
    output = composite.extract_all_edges(progress_bar=True)
    assert output.n_blocks == composite.n_blocks


def test_elevation(uniform):
    dataset = uniform
    # Test default params
    elev = dataset.elevation(progress_bar=True)
    assert 'Elevation' in elev.array_names
    assert elev.active_scalars_name == 'Elevation'
    assert elev.get_data_range() == (dataset.bounds.z_min, dataset.bounds.z_max)
    # test vector args
    c = list(dataset.center)
    t = list(c)  # cast so it does not point to `c`
    t[2] = dataset.bounds[-1]
    elev = dataset.elevation(low_point=c, high_point=t, progress_bar=True)
    assert 'Elevation' in elev.array_names
    assert elev.active_scalars_name == 'Elevation'
    assert elev.get_data_range() == (dataset.center[2], dataset.bounds.z_max)
    # Test not setting active
    elev = dataset.elevation(set_active=False, progress_bar=True)
    assert 'Elevation' in elev.array_names
    assert elev.active_scalars_name != 'Elevation'
    # Set use a range by scalar name
    elev = dataset.elevation(scalar_range='Spatial Point Data', progress_bar=True)
    assert 'Elevation' in elev.array_names
    assert elev.active_scalars_name == 'Elevation'
    assert dataset.get_data_range('Spatial Point Data') == (elev.get_data_range('Elevation'))
    # Set use a user defined range
    elev = dataset.elevation(scalar_range=[1.0, 100.0], progress_bar=True)
    assert 'Elevation' in elev.array_names
    assert elev.active_scalars_name == 'Elevation'
    assert elev.get_data_range('Elevation') == (1.0, 100.0)
    # test errors
    match = 'Data Range has shape () which is not allowed. Shape must be 2.'
    with pytest.raises(ValueError, match=re.escape(match)):
        elev = dataset.elevation(scalar_range=0.5, progress_bar=True)
    with pytest.raises(ValueError):  # noqa: PT011
        elev = dataset.elevation(scalar_range=[1, 2, 3], progress_bar=True)
    with pytest.raises(TypeError):
        elev = dataset.elevation(scalar_range={1, 2}, progress_bar=True)


def test_elevation_composite(composite):
    # Now test composite data structures
    output = composite.elevation(progress_bar=True)
    assert output.n_blocks == composite.n_blocks


def test_compute_cell_sizes(datasets):
    for dataset in datasets:
        result = dataset.compute_cell_sizes(progress_bar=True, vertex_count=True)
        assert result is not None
        assert isinstance(result, type(dataset))
        assert 'Length' in result.array_names
        assert 'Area' in result.array_names
        assert 'Volume' in result.array_names
        assert 'VertexCount' in result.array_names
    # Test the volume property
    grid = pv.ImageData(dimensions=(10, 10, 10))
    volume = float(np.prod(np.array(grid.dimensions) - 1))
    assert np.allclose(grid.volume, volume)


def test_compute_cell_sizes_composite(composite):
    # Now test composite data structures
    output = composite.compute_cell_sizes(progress_bar=True)
    assert output.n_blocks == composite.n_blocks


def test_cell_centers(datasets):
    for dataset in datasets:
        result = dataset.cell_centers(progress_bar=True)
        assert result is not None
        assert isinstance(result, pv.PolyData)


@pytest.mark.needs_vtk_version(9, 1, 0)
def test_cell_center_pointset(airplane):
    pointset = airplane.cast_to_pointset()
    result = pointset.cell_centers(progress_bar=True)
    assert result is not None
    assert isinstance(result, pv.PolyData)


def test_cell_centers_composite(composite):
    # Now test composite data structures
    output = composite.cell_centers(progress_bar=True)
    assert output.n_blocks == composite.n_blocks


def test_cell_data_to_point_data():
    data = examples.load_uniform()
    foo = data.cell_data_to_point_data(progress_bar=True)
    assert foo.n_arrays == 2
    assert len(foo.cell_data.keys()) == 0
    _ = data.ctp()


def test_cell_data_to_point_data_composite(composite):
    # Now test composite data structures
    output = composite.cell_data_to_point_data(progress_bar=True)
    assert output.n_blocks == composite.n_blocks


def test_point_data_to_cell_data():
    data = examples.load_uniform()
    foo = data.point_data_to_cell_data(progress_bar=True)
    assert foo.n_arrays == 2
    assert len(foo.point_data.keys()) == 0
    _ = data.ptc()


def test_point_data_to_cell_data_composite(composite):
    # Now test composite data structures
    output = composite.point_data_to_cell_data(progress_bar=True)
    assert output.n_blocks == composite.n_blocks


def test_triangulate():
    data = examples.load_uniform()
    tri = data.triangulate(progress_bar=True)
    assert isinstance(tri, pv.UnstructuredGrid)
    assert np.any(tri.cells)


def test_triangulate_composite(composite):
    # Now test composite data structures
    output = composite.triangulate(progress_bar=True)
    assert output.n_blocks == composite.n_blocks


def test_sample():
    mesh = pv.Sphere(center=(4.5, 4.5, 4.5), radius=4.5)
    data_to_probe = examples.load_uniform()

    def sample_test(**kwargs):
        """Test `sample` with kwargs."""
        result = mesh.sample(data_to_probe, **kwargs)
        name = 'Spatial Point Data'
        assert name in result.array_names
        assert isinstance(result, type(mesh))

    sample_test()
    sample_test(tolerance=1.0)
    sample_test(progress_bar=True)
    sample_test(categorical=True)
    sample_test(locator=_vtk_core.vtkStaticCellLocator())
    for locator in ['cell', 'cell_tree', 'obb_tree', 'static_cell']:
        sample_test(locator=locator)
    with pytest.raises(ValueError):  # noqa: PT011
        sample_test(locator='invalid')
    sample_test(pass_cell_data=False)
    sample_test(pass_point_data=False)
    sample_test(pass_field_data=False)
    if pv.vtk_version_info >= (9, 3):
        sample_test(snap_to_closest_point=True)
    else:
        with pytest.raises(VTKVersionError, match='snap_to_closest_point'):
            sample_test(snap_to_closest_point=True)


def test_sample_composite():
    mesh0 = pv.ImageData(dimensions=(11, 11, 1), origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    mesh1 = pv.ImageData(dimensions=(11, 11, 1), origin=(10.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    mesh0['common_data'] = np.zeros(mesh0.n_points)
    mesh1['common_data'] = np.ones(mesh1.n_points)
    mesh0['partial_data'] = np.zeros(mesh0.n_points)

    composite = pv.MultiBlock([mesh0, mesh1])

    probe_points = pv.PolyData(
        [
            [5.0, 5.0, 0.0],
            [15.0, 5.0, 0.0],
            [25.0, 5.0, 0.0],  # outside domain
        ],
    )

    result = probe_points.sample(composite)
    assert 'common_data' in result.point_data
    # Need pass partial arrays?
    assert 'partial_data' not in result.point_data
    assert 'vtkValidPointMask' in result.point_data
    assert 'vtkGhostType' in result.point_data
    # data outside domain is 0
    assert np.array_equal(result['common_data'], [0.0, 1.0, 0.0])
    assert np.array_equal(result['vtkValidPointMask'], [1, 1, 0])

    result = probe_points.sample(composite, mark_blank=False)
    assert 'vtkGhostType' not in result.point_data

    small_mesh_0 = pv.ImageData(
        dimensions=(6, 6, 1),
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
    )
    small_mesh_1 = pv.ImageData(
        dimensions=(6, 6, 1),
        origin=(10.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
    )

    probe_composite = pv.MultiBlock([small_mesh_0, small_mesh_1])
    result = probe_composite.sample(composite)
    assert 'common_data' in result[0].point_data
    # Need pass partial arrays?
    assert 'partial_data' not in result[0].point_data
    assert 'vtkValidPointMask' in result[0].point_data
    assert 'vtkGhostType' in result[0].point_data


def test_extract_geometry(datasets, composite):
    for dataset in datasets:
        geom = dataset.extract_geometry(progress_bar=True)
        assert geom is not None
        assert isinstance(geom, pv.PolyData)
    # Now test composite data structures
    output = composite.extract_geometry()
    assert isinstance(output, pv.PolyData)


def test_extract_geometry_composite(multiblock_all_with_nested_and_none):
    geom = multiblock_all_with_nested_and_none.extract_geometry()
    assert isinstance(geom, pv.PolyData)


def test_extract_geometry_extent(uniform):
    geom = uniform.extract_geometry(extent=(0, 5, 0, 100, 0, 100))
    assert isinstance(geom, pv.PolyData)
    assert geom.bounds == (0.0, 5.0, 0.0, 9.0, 0.0, 9.0)
