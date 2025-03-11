from __future__ import annotations

import itertools
import re

from hypothesis import HealthCheck
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis.extra._array_helpers import array_shapes
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import one_of
import numpy as np
import pytest
import vtk

import pyvista as pv
from pyvista import PyVistaDeprecationWarning
from pyvista import VTKVersionError
from pyvista import examples
from pyvista.core import _vtk_core
from tests.core.test_dataset_filters import HYPOTHESIS_MAX_EXAMPLES
from tests.core.test_dataset_filters import n_numbers
from tests.core.test_dataset_filters import normals


@pytest.mark.parametrize('return_clipped', [True, False])
def test_clip_filter(multiblock_all_with_nested_and_none, return_clipped):
    """This tests the clip filter on all datatypes available filters"""
    # Remove None blocks in the root block but keep the none block in the nested MultiBlock
    multi = multiblock_all_with_nested_and_none
    for i, block in enumerate(multi):
        if block is None:
            del multi[i]
    assert None not in multi
    assert None in multi.recursive_iterator()

    for dataset in multi:
        clips = dataset.clip(normal='x', invert=True, return_clipped=return_clipped)
        assert clips is not None

        if return_clipped:
            assert isinstance(clips, tuple)
            assert len(clips) == 2
        else:
            assert isinstance(clips, pv.DataObject)
            # Make dataset iterable
            clips = [clips]

        for clip in clips:
            if isinstance(dataset, pv.PolyData):
                assert isinstance(clip, pv.PolyData)
            elif isinstance(dataset, pv.MultiBlock):
                assert isinstance(clip, pv.MultiBlock)
                assert clip.n_blocks == dataset.n_blocks
            else:
                assert isinstance(clip, pv.UnstructuredGrid)


def test_clip_filter_normal(datasets):
    # Test no errors are raised
    for i, dataset in enumerate(datasets):
        dataset.clip(normal=normals[i], invert=True)


@pytest.mark.parametrize('dataset', [pv.PolyData(), pv.MultiBlock()])
def test_clip_filter_empty_inputs(dataset):
    dataset.clip('x')


def test_clip_filter_crinkle_disjoint(uniform):
    def assert_array_names(clipped):
        assert cell_ids in clipped.array_names
        assert 'vtkOriginalPointIds' not in clipped.array_names
        assert 'vtkOriginalCellIds' not in clipped.array_names

    # crinkle clip
    cell_ids = 'cell_ids'
    clp = uniform.clip(normal=(1, 1, 1), crinkle=True)
    assert_array_names(clp)

    assert clp is not None
    clp1, clp2 = uniform.clip(normal=(1, 1, 1), return_clipped=True, crinkle=True)
    assert clp1 is not None
    assert clp2 is not None
    assert_array_names(clp1)
    assert_array_names(clp2)
    set_a = set(clp1.cell_data[cell_ids])
    set_b = set(clp2.cell_data[cell_ids])
    assert set_a.isdisjoint(set_b)
    assert set_a.union(set_b) == set(range(uniform.n_cells))


@pytest.mark.parametrize('has_active_scalars', [True, False])
def test_clip_filter_crinkle_active_scalars(uniform, has_active_scalars):
    if not has_active_scalars:
        uniform.set_active_scalars(None)
        assert uniform.active_scalars is None
    else:
        assert uniform.active_scalars is not None

    scalars_before = uniform.active_scalars_name
    uniform.clip('x', crinkle=True)
    scalars_after = uniform.active_scalars_name
    assert scalars_before == scalars_after


def test_clip_filter_composite(multiblock_all):
    # Now test composite data structures
    output = multiblock_all.clip(normal=normals[0], invert=False)
    assert output.n_blocks == multiblock_all.n_blocks


def test_transform_raises(sphere):
    matrix = np.diag((1, 1, 1, 0))
    match = re.escape('Transform element (3,3), the inverse scale term, is zero')
    with pytest.raises(ValueError, match=match):
        sphere.transform(matrix, inplace=False)


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


def test_clip_box_composite(multiblock_all):
    # Now test composite data structures
    output = multiblock_all.clip_box(invert=False, progress_bar=True)
    assert output.n_blocks == multiblock_all.n_blocks


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


def test_slice_filter_composite(multiblock_all):
    # Now test composite data structures
    output = multiblock_all.slice(normal=normals[0], progress_bar=True)
    assert output.n_blocks == multiblock_all.n_blocks


def test_slice_orthogonal_filter(datasets):
    """This tests the slice filter on all datatypes available filters"""
    for dataset in datasets:
        slices = dataset.slice_orthogonal(progress_bar=True)
        assert slices is not None
        assert isinstance(slices, pv.MultiBlock)
        assert slices.n_blocks == 3
        for slc in slices:
            assert isinstance(slc, pv.PolyData)


def test_slice_orthogonal_filter_composite(multiblock_all):
    # Now test composite data structures
    output = multiblock_all.slice_orthogonal(progress_bar=True)
    assert output.n_blocks == multiblock_all.n_blocks


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


def test_slice_along_axis_composite(multiblock_all):
    # Now test composite data structures
    output = multiblock_all.slice_along_axis(progress_bar=True)
    assert output.n_blocks == multiblock_all.n_blocks


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


def test_extract_all_edges_composite(multiblock_all):
    # Now test composite data structures
    output = multiblock_all.extract_all_edges(progress_bar=True)
    assert output.n_blocks == multiblock_all.n_blocks


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


def test_elevation_composite(multiblock_all):
    # Now test composite data structures
    output = multiblock_all.elevation(progress_bar=True)
    assert output.n_blocks == multiblock_all.n_blocks


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


def test_compute_cell_sizes_composite(multiblock_all):
    # Now test composite data structures
    output = multiblock_all.compute_cell_sizes(progress_bar=True)
    assert output.n_blocks == multiblock_all.n_blocks


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


def test_cell_centers_composite(multiblock_all):
    # Now test composite data structures
    output = multiblock_all.cell_centers(progress_bar=True)
    assert output.n_blocks == multiblock_all.n_blocks


def test_cell_data_to_point_data():
    data = examples.load_uniform()
    foo = data.cell_data_to_point_data(progress_bar=True)
    assert foo.n_arrays == 2
    assert len(foo.cell_data.keys()) == 0
    _ = data.ctp()


def test_cell_data_to_point_data_composite(multiblock_all):
    # Now test composite data structures
    output = multiblock_all.cell_data_to_point_data(progress_bar=True)
    assert output.n_blocks == multiblock_all.n_blocks


def test_point_data_to_cell_data():
    data = examples.load_uniform()
    foo = data.point_data_to_cell_data(progress_bar=True)
    assert foo.n_arrays == 2
    assert len(foo.point_data.keys()) == 0
    _ = data.ptc()


def test_point_data_to_cell_data_composite(multiblock_all):
    # Now test composite data structures
    output = multiblock_all.point_data_to_cell_data(progress_bar=True)
    assert output.n_blocks == multiblock_all.n_blocks


def test_triangulate():
    data = examples.load_uniform()
    tri = data.triangulate(progress_bar=True)
    assert isinstance(tri, pv.UnstructuredGrid)
    assert np.any(tri.cells)


def test_triangulate_composite(multiblock_all):
    # Now test composite data structures
    output = multiblock_all.triangulate(progress_bar=True)
    assert output.n_blocks == multiblock_all.n_blocks


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


def test_slice_along_line_composite(multiblock_all):
    # Now test composite data structures
    a = [multiblock_all.bounds.x_min, multiblock_all.bounds.y_min, multiblock_all.bounds.z_min]
    b = [multiblock_all.bounds.x_max, multiblock_all.bounds.y_max, multiblock_all.bounds.z_max]
    line = pv.Line(a, b, resolution=10)
    output = multiblock_all.slice_along_line(line, progress_bar=True)
    assert output.n_blocks == multiblock_all.n_blocks


def test_compute_cell_quality():
    mesh = pv.ParametricEllipsoid().triangulate().decimate(0.8)
    qual = mesh.compute_cell_quality(progress_bar=True)
    assert 'CellQuality' in qual.array_names
    with pytest.raises(KeyError):
        qual = mesh.compute_cell_quality(quality_measure='foo', progress_bar=True)


@pytest.mark.parametrize(
    ('num_cell_arrays', 'num_point_data'),
    itertools.product([0, 1, 2], [0, 1, 2]),
)
def test_transform_mesh(datasets, num_cell_arrays, num_point_data):
    # rotate about x-axis by 90 degrees
    for dataset in datasets:
        tf = pv.core.utilities.transformations.axis_angle_rotation((1, 0, 0), 90)

        for i in range(num_cell_arrays):
            dataset.cell_data[f'C{i}'] = np.random.default_rng().random((dataset.n_cells, 3))

        for i in range(num_point_data):
            dataset.point_data[f'P{i}'] = np.random.default_rng().random((dataset.n_points, 3))

        # deactivate any active vectors!
        # even if transform_all_input_vectors is False, vtkTransformfilter will
        # transform active vectors
        dataset.set_active_vectors(None)

        transformed = dataset.transform(tf, transform_all_input_vectors=False, inplace=False)

        assert dataset.points[:, 0] == pytest.approx(transformed.points[:, 0])
        assert dataset.points[:, 2] == pytest.approx(-transformed.points[:, 1])
        assert dataset.points[:, 1] == pytest.approx(transformed.points[:, 2])

        # ensure that none of the vector data is changed
        for name, array in dataset.point_data.items():
            assert transformed.point_data[name] == pytest.approx(array)

        for name, array in dataset.cell_data.items():
            assert transformed.cell_data[name] == pytest.approx(array)

        # verify that the cell connectivity is a deep copy
        if hasattr(dataset, '_connectivity_array'):
            transformed._connectivity_array[0] += 1
            assert not np.array_equal(dataset._connectivity_array, transformed._connectivity_array)
        if hasattr(dataset, 'cell_connectivity'):
            transformed.cell_connectivity[0] += 1
            assert not np.array_equal(dataset.cell_connectivity, transformed.cell_connectivity)


@pytest.mark.parametrize(
    ('num_cell_arrays', 'num_point_data'),
    itertools.product([0, 1, 2], [0, 1, 2]),
)
def test_transform_mesh_and_vectors(datasets, num_cell_arrays, num_point_data):
    for dataset in datasets:
        # rotate about x-axis by 90 degrees
        tf = pv.core.utilities.transformations.axis_angle_rotation((1, 0, 0), 90)

        for i in range(num_cell_arrays):
            dataset.cell_data[f'C{i}'] = np.random.default_rng().random((dataset.n_cells, 3))

        for i in range(num_point_data):
            dataset.point_data[f'P{i}'] = np.random.default_rng().random((dataset.n_points, 3))

        # track original untransformed dataset
        orig_dataset = dataset.copy(deep=True)

        transformed = dataset.transform(tf, transform_all_input_vectors=True, inplace=False)

        # verify that the dataset has not modified
        if num_cell_arrays:
            assert dataset.cell_data == orig_dataset.cell_data
        if num_point_data:
            assert dataset.point_data == orig_dataset.point_data

        assert dataset.points[:, 0] == pytest.approx(transformed.points[:, 0])
        assert dataset.points[:, 2] == pytest.approx(-transformed.points[:, 1])
        assert dataset.points[:, 1] == pytest.approx(transformed.points[:, 2])

        for i in range(num_cell_arrays):
            assert dataset.cell_data[f'C{i}'][:, 0] == pytest.approx(
                transformed.cell_data[f'C{i}'][:, 0],
            )
            assert dataset.cell_data[f'C{i}'][:, 2] == pytest.approx(
                -transformed.cell_data[f'C{i}'][:, 1],
            )
            assert dataset.cell_data[f'C{i}'][:, 1] == pytest.approx(
                transformed.cell_data[f'C{i}'][:, 2],
            )

        for i in range(num_point_data):
            assert dataset.point_data[f'P{i}'][:, 0] == pytest.approx(
                transformed.point_data[f'P{i}'][:, 0],
            )
            assert dataset.point_data[f'P{i}'][:, 2] == pytest.approx(
                -transformed.point_data[f'P{i}'][:, 1],
            )
            assert dataset.point_data[f'P{i}'][:, 1] == pytest.approx(
                transformed.point_data[f'P{i}'][:, 2],
            )

        # Verify active scalars are not changed
        expected_point_scalars_name = orig_dataset.point_data.active_scalars_name
        actual_point_scalars_name = transformed.point_data.active_scalars_name
        assert actual_point_scalars_name == expected_point_scalars_name

        expected_cell_scalars_name = orig_dataset.cell_data.active_scalars_name
        actual_cell_scalars_name = transformed.cell_data.active_scalars_name
        assert actual_cell_scalars_name == expected_cell_scalars_name


@pytest.mark.parametrize(
    ('num_cell_arrays', 'num_point_data'),
    itertools.product([0, 1, 2], [0, 1, 2]),
)
def test_transform_int_vectors_warning(datasets, num_cell_arrays, num_point_data):
    for dataset in datasets:
        tf = pv.core.utilities.transformations.axis_angle_rotation((1, 0, 0), 90)
        dataset.clear_data()
        for i in range(num_cell_arrays):
            dataset.cell_data[f'C{i}'] = np.random.default_rng().integers(
                np.iinfo(int).max,
                size=(dataset.n_cells, 3),
            )
        for i in range(num_point_data):
            dataset.point_data[f'P{i}'] = np.random.default_rng().integers(
                np.iinfo(int).max,
                size=(dataset.n_points, 3),
            )
        if not (num_cell_arrays == 0 and num_point_data == 0):
            with pytest.warns(UserWarning, match='Integer'):
                _ = dataset.transform(tf, transform_all_input_vectors=True, inplace=False)


def test_transform_inplace_rectilinear(rectilinear):
    # assert that transformations of this type raises the correct error
    tf = pv.core.utilities.transformations.axis_angle_rotation(
        (1, 0, 0),
        90,
    )  # rotate about x-axis by 90 degrees
    with pytest.raises(TypeError):
        rectilinear.transform(tf, inplace=True)


@pytest.mark.parametrize('spacing', [(1, 1, 1), (0.5, 0.6, 0.7)])
def test_transform_imagedata(uniform, spacing):
    # Transformations affect origin, spacing, and direction, so test these here
    uniform.spacing = spacing

    # Test scaling
    vector123 = np.array((1, 2, 3))
    uniform.scale(vector123, inplace=True)
    expected_spacing = spacing * vector123
    assert np.allclose(uniform.spacing, expected_spacing)

    # Test direction
    rotation = pv.Transform().rotate_vector(vector123, 30).matrix[:3, :3]
    uniform.rotate(rotation, inplace=True)
    assert np.allclose(uniform.direction_matrix, rotation)

    # Test translation by centering data
    vector = np.array(uniform.center) * -1
    translation = pv.Transform().translate(vector)
    uniform.transform(translation, inplace=True)
    assert isinstance(uniform, pv.ImageData)
    assert np.array_equal(uniform.origin, vector)

    # Test applying a second translation
    translated = uniform.transform(translation, inplace=False)
    assert np.allclose(translated.origin, vector * 2)
    assert np.allclose(translated.center, uniform.origin)


def test_transform_imagedata_warns_with_shear(uniform):
    shear = np.eye(4)
    shear[0, 1] = 0.1

    with pytest.warns(
        Warning,
        match='The transformation matrix has a shear component which has been removed. \n'
        'Shear is not supported when setting `ImageData` `index_to_physical_matrix`.',
    ):
        uniform.transform(shear, inplace=True)


def test_transform_filter_inplace_default_warns(cube):
    expected_msg = 'The default value of `inplace` for the filter `PolyData.transform` will change in the future.'
    with pytest.warns(PyVistaDeprecationWarning, match=expected_msg):
        _ = cube.transform(np.eye(4))


def test_reflect_mesh_about_point(datasets):
    for dataset in datasets:
        x_plane = 500
        reflected = dataset.reflect((1, 0, 0), point=(x_plane, 0, 0), progress_bar=True)
        assert reflected.n_cells == dataset.n_cells
        assert reflected.n_points == dataset.n_points
        assert np.allclose(x_plane - dataset.points[:, 0], reflected.points[:, 0] - x_plane)
        assert np.allclose(dataset.points[:, 1:], reflected.points[:, 1:])


def test_reflect_mesh_with_vectors(datasets):
    for dataset in datasets:
        if hasattr(dataset, 'compute_normals'):
            dataset.compute_normals(inplace=True, progress_bar=True)

        # add vector data to cell and point arrays
        dataset.cell_data['C'] = np.arange(dataset.n_cells)[:, np.newaxis] * np.array(
            [1, 2, 3],
            dtype=float,
        ).reshape((1, 3))
        dataset.point_data['P'] = np.arange(dataset.n_points)[:, np.newaxis] * np.array(
            [1, 2, 3],
            dtype=float,
        ).reshape((1, 3))

        reflected = dataset.reflect(
            (1, 0, 0),
            transform_all_input_vectors=True,
            inplace=False,
            progress_bar=True,
        )

        # assert isinstance(reflected, type(dataset))
        assert reflected.n_cells == dataset.n_cells
        assert reflected.n_points == dataset.n_points
        assert np.allclose(dataset.points[:, 0], -reflected.points[:, 0])
        assert np.allclose(dataset.points[:, 1:], reflected.points[:, 1:])

        # assert normals are reflected
        if hasattr(dataset, 'compute_normals'):
            assert np.allclose(
                dataset.cell_data['Normals'][:, 0],
                -reflected.cell_data['Normals'][:, 0],
            )
            assert np.allclose(
                dataset.cell_data['Normals'][:, 1:],
                reflected.cell_data['Normals'][:, 1:],
            )
            assert np.allclose(
                dataset.point_data['Normals'][:, 0],
                -reflected.point_data['Normals'][:, 0],
            )
            assert np.allclose(
                dataset.point_data['Normals'][:, 1:],
                reflected.point_data['Normals'][:, 1:],
            )

        # assert other vector fields are reflected
        assert np.allclose(dataset.cell_data['C'][:, 0], -reflected.cell_data['C'][:, 0])
        assert np.allclose(dataset.cell_data['C'][:, 1:], reflected.cell_data['C'][:, 1:])
        assert np.allclose(dataset.point_data['P'][:, 0], -reflected.point_data['P'][:, 0])
        assert np.allclose(dataset.point_data['P'][:, 1:], reflected.point_data['P'][:, 1:])


@pytest.mark.parametrize(
    'dataset',
    [
        examples.load_hexbeam(),  # UnstructuredGrid
        examples.load_airplane(),  # PolyData
        examples.load_structured(),  # StructuredGrid
    ],
)
def test_reflect_inplace(dataset):
    orig = dataset.copy()
    dataset.reflect((1, 0, 0), inplace=True, progress_bar=True)
    assert dataset.n_cells == orig.n_cells
    assert dataset.n_points == orig.n_points
    assert np.allclose(dataset.points[:, 0], -orig.points[:, 0])
    assert np.allclose(dataset.points[:, 1:], orig.points[:, 1:])


def test_transform_inplace_bad_types_2(rectilinear):
    # assert that transformations of these types throw the correct error
    with pytest.raises(TypeError):
        rectilinear.reflect((1, 0, 0), inplace=True)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(rotate_amounts=n_numbers(4), translate_amounts=n_numbers(3))
def test_transform_should_match_vtk_transformation(rotate_amounts, translate_amounts, grid):
    trans = pv.Transform()
    trans.check_finite = False
    trans.RotateWXYZ(*rotate_amounts)
    trans.translate(translate_amounts)
    trans.Update()

    # Apply transform with pyvista filter
    grid_a = grid.copy()
    grid_a.transform(trans, inplace=True)

    # Apply transform with vtk filter
    grid_b = grid.copy()
    f = vtk.vtkTransformFilter()
    f.SetInputDataObject(grid_b)
    f.SetTransform(trans)
    f.Update()
    grid_b = pv.wrap(f.GetOutput())

    # treat INF as NAN (necessary for allclose)
    grid_a.points[np.isinf(grid_a.points)] = np.nan
    assert np.allclose(grid_a.points, grid_b.points, equal_nan=True)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(rotate_amounts=n_numbers(4))
def test_transform_should_match_vtk_transformation_non_homogeneous(rotate_amounts, grid):
    # test non homogeneous transform
    trans_rotate_only = pv.Transform()
    trans_rotate_only.check_finite = False
    trans_rotate_only.RotateWXYZ(*rotate_amounts)
    trans_rotate_only.Update()

    grid_copy = grid.copy()
    grid_copy.transform(trans_rotate_only, inplace=True)

    from pyvista.core.utilities.transformations import apply_transformation_to_points

    trans_arr = trans_rotate_only.matrix[:3, :3]
    trans_pts = apply_transformation_to_points(trans_arr, grid.points)
    assert np.allclose(grid_copy.points, trans_pts, equal_nan=True)


def test_translate_should_not_fail_given_none(grid):
    bounds = grid.bounds
    grid.transform(None, inplace=True)
    assert grid.bounds == bounds


def test_translate_should_fail_bad_points_or_transform(grid):
    points = np.random.default_rng().random((10, 2))
    bad_points = np.random.default_rng().random((10, 2))
    trans = np.random.default_rng().random((4, 4))
    bad_trans = np.random.default_rng().random((2, 4))
    with pytest.raises(ValueError):  # noqa: PT011
        pv.core.utilities.transformations.apply_transformation_to_points(trans, bad_points)

    with pytest.raises(ValueError):  # noqa: PT011
        pv.core.utilities.transformations.apply_transformation_to_points(bad_trans, points)


@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=HYPOTHESIS_MAX_EXAMPLES,
)
@given(array=arrays(dtype=np.float32, shape=array_shapes(max_dims=5, max_side=5)))
def test_transform_should_fail_given_wrong_numpy_shape(array, grid):
    assume(array.shape not in [(3, 3), (4, 4)])
    match = 'Shape must be one of [(3, 3), (4, 4)]'
    with pytest.raises(ValueError, match=re.escape(match)):
        grid.transform(array, inplace=True)


@pytest.mark.parametrize('axis_amounts', [[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
def test_translate_should_translate_grid(grid, axis_amounts):
    grid_copy = grid.copy()
    grid_copy.translate(axis_amounts, inplace=True)

    grid_points = grid.points.copy() + np.array(axis_amounts)
    assert np.allclose(grid_copy.points, grid_points)


@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=HYPOTHESIS_MAX_EXAMPLES,
)
@given(angle=one_of(floats(allow_infinity=False, allow_nan=False), integers()))
@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
def test_rotate_should_match_vtk_rotation(angle, axis, grid):
    trans = vtk.vtkTransform()
    getattr(trans, f'Rotate{axis.upper()}')(angle)
    trans.Update()

    trans_filter = vtk.vtkTransformFilter()
    trans_filter.SetTransform(trans)
    trans_filter.SetInputData(grid)
    trans_filter.Update()
    grid_a = pv.UnstructuredGrid(trans_filter.GetOutput())

    grid_b = grid.copy()
    getattr(grid_b, f'rotate_{axis}')(angle, inplace=True)
    assert np.allclose(grid_a.points, grid_b.points, equal_nan=True)


def test_rotate_90_degrees_four_times_should_return_original_geometry():
    sphere = pv.Sphere()
    sphere.rotate_y(90, inplace=True)
    sphere.rotate_y(90, inplace=True)
    sphere.rotate_y(90, inplace=True)
    sphere.rotate_y(90, inplace=True)
    assert np.all(sphere.points == pv.Sphere().points)


def test_rotate_180_degrees_two_times_should_return_original_geometry():
    sphere = pv.Sphere()
    sphere.rotate_x(180, inplace=True)
    sphere.rotate_x(180, inplace=True)
    assert np.all(sphere.points == pv.Sphere().points)


def test_rotate_vector_90_degrees_should_not_distort_geometry():
    cylinder = pv.Cylinder()
    rotated = cylinder.rotate_vector(vector=(1, 1, 0), angle=90)
    assert np.isclose(cylinder.volume, rotated.volume)


def test_rotations_should_match_by_a_360_degree_difference():
    mesh = examples.load_airplane()

    point = np.random.default_rng().random(3) - 0.5
    angle = (np.random.default_rng().random() - 0.5) * 360.0
    vector = np.random.default_rng().random(3) - 0.5

    # Rotate about x axis.
    rot1 = mesh.copy()
    rot2 = mesh.copy()
    rot1.rotate_x(angle=angle, point=point, inplace=True)
    rot2.rotate_x(angle=angle - 360.0, point=point, inplace=True)
    assert np.allclose(rot1.points, rot2.points)

    # Rotate about y axis.
    rot1 = mesh.copy()
    rot2 = mesh.copy()
    rot1.rotate_y(angle=angle, point=point, inplace=True)
    rot2.rotate_y(angle=angle - 360.0, point=point, inplace=True)
    assert np.allclose(rot1.points, rot2.points)

    # Rotate about z axis.
    rot1 = mesh.copy()
    rot2 = mesh.copy()
    rot1.rotate_z(angle=angle, point=point, inplace=True)
    rot2.rotate_z(angle=angle - 360.0, point=point, inplace=True)
    assert np.allclose(rot1.points, rot2.points)

    # Rotate about custom vector.
    rot1 = mesh.copy()
    rot2 = mesh.copy()
    rot1.rotate_vector(vector=vector, angle=angle, point=point, inplace=True)
    rot2.rotate_vector(vector=vector, angle=angle - 360.0, point=point, inplace=True)
    assert np.allclose(rot1.points, rot2.points)


def test_rotate_x():
    # Test non-point-based mesh doesn't fail
    mesh = examples.load_uniform()
    out = mesh.rotate_x(30)
    assert isinstance(out, pv.ImageData)
    match = 'Shape must be one of [(3,), (1, 3), (3, 1)]'
    with pytest.raises(ValueError, match=re.escape(match)):
        out = mesh.rotate_x(30, point=5)
    with pytest.raises(ValueError, match=re.escape(match)):
        out = mesh.rotate_x(30, point=[1, 3])


def test_rotate_y():
    # Test non-point-based mesh doesn't fail
    mesh = examples.load_uniform()
    out = mesh.rotate_y(30)
    assert isinstance(out, pv.ImageData)
    match = 'Shape must be one of [(3,), (1, 3), (3, 1)]'
    with pytest.raises(ValueError, match=re.escape(match)):
        out = mesh.rotate_y(30, point=5)
    with pytest.raises(ValueError, match=re.escape(match)):
        out = mesh.rotate_y(30, point=[1, 3])


def test_rotate_z():
    # Test non-point-based mesh doesn't fail
    mesh = examples.load_uniform()
    out = mesh.rotate_z(30)
    assert isinstance(out, pv.ImageData)
    match = 'Shape must be one of [(3,), (1, 3), (3, 1)]'
    with pytest.raises(ValueError, match=re.escape(match)):
        out = mesh.rotate_z(30, point=5)
    with pytest.raises(ValueError, match=re.escape(match)):
        out = mesh.rotate_z(30, point=[1, 3])


def test_rotate_vector():
    # Test non-point-based mesh doesn't fail
    mesh = examples.load_uniform()
    out = mesh.rotate_vector([1, 1, 1], 33)
    assert isinstance(out, pv.ImageData)
    match = 'Shape must be one of [(3,), (1, 3), (3, 1)]'
    with pytest.raises(ValueError, match=re.escape(match)):
        out = mesh.rotate_vector([1, 1], 33)
    with pytest.raises(ValueError, match=re.escape(match)):
        out = mesh.rotate_vector(30, 33)


def test_rotate():
    # Test non-point-based mesh doesn't fail
    mesh = examples.load_uniform()
    out = mesh.rotate([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert isinstance(out, pv.ImageData)


def test_transform_integers():
    # regression test for gh-1943
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ]
    # build vtkPolyData from scratch to enforce int data
    poly = vtk.vtkPolyData()
    poly.SetPoints(pv.vtk_points(points))
    poly = pv.wrap(poly)
    poly.verts = [1, 0, 1, 1, 1, 2]
    # define active and inactive vectors with int values
    for dataset_attrs in poly.point_data, poly.cell_data:
        for key in 'active_v', 'inactive_v', 'active_n', 'inactive_n':
            dataset_attrs[key] = poly.points
        dataset_attrs.active_vectors_name = 'active_v'
        dataset_attrs.active_normals_name = 'active_n'

    # active vectors and normals should be converted by default
    for key in 'active_v', 'inactive_v', 'active_n', 'inactive_n':
        assert poly.point_data[key].dtype == np.int_
        assert poly.cell_data[key].dtype == np.int_

    with pytest.warns(UserWarning):
        poly.rotate_x(angle=10, inplace=True)

    # check that points were converted and transformed correctly
    assert poly.points.dtype == np.float32
    assert poly.points[-1, 1] != 0
    # assert that exactly active vectors and normals were converted
    for key in 'active_v', 'active_n':
        assert poly.point_data[key].dtype == np.float32
        assert poly.cell_data[key].dtype == np.float32
    for key in 'inactive_v', 'inactive_n':
        assert poly.point_data[key].dtype == np.int_
        assert poly.cell_data[key].dtype == np.int_


@pytest.mark.xfail(reason='VTK bug')
def test_transform_integers_vtkbug_present():
    # verify that the VTK transform bug is still there
    # if this test starts to pass, we can remove the
    # automatic float conversion from ``DataSet.transform``
    # along with this test
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ]
    # build vtkPolyData from scratch to enforce int data
    poly = vtk.vtkPolyData()
    poly.SetPoints(pv.vtk_points(points))

    # manually put together a rotate_x(10) transform
    trans_arr = pv.core.utilities.transformations.axis_angle_rotation((1, 0, 0), 10, deg=True)
    trans_mat = pv.vtkmatrix_from_array(trans_arr)
    trans = vtk.vtkTransform()
    trans.SetMatrix(trans_mat)
    trans_filt = vtk.vtkTransformFilter()
    trans_filt.SetInputDataObject(poly)
    trans_filt.SetTransform(trans)
    trans_filt.Update()
    poly = pv.wrap(trans_filt.GetOutputDataObject(0))
    # the bug is that e.g. 0.98 gets truncated to 0
    assert poly.points[-1, 1] != 0


def test_scale():
    mesh = examples.load_airplane()

    xyz = np.random.default_rng().random(3)
    scale1 = mesh.copy()
    scale2 = mesh.copy()
    scale1.scale(xyz, inplace=True)
    scale2.points *= xyz
    scale3 = mesh.scale(xyz, inplace=False)
    assert np.allclose(scale1.points, scale2.points)
    assert np.allclose(scale3.points, scale2.points)
    # test scalar scale case
    scale1 = mesh.copy()
    scale2 = mesh.copy()
    xyz = 4.0
    scale1.scale(xyz, inplace=True)
    scale2.scale([xyz] * 3, inplace=True)
    assert np.allclose(scale1.points, scale2.points)
    # test non-point-based mesh doesn't fail
    mesh = examples.load_uniform()
    out = mesh.scale(xyz)
    assert isinstance(out, pv.ImageData)


def test_flip_x():
    mesh = examples.load_airplane()
    flip_x1 = mesh.copy()
    flip_x2 = mesh.copy()
    flip_x1.flip_x(point=(0, 0, 0), inplace=True)
    flip_x2.points[:, 0] *= -1.0
    assert np.allclose(flip_x1.points, flip_x2.points)
    # Test non-point-based mesh doesn't fail
    mesh = examples.load_uniform()
    out = mesh.flip_x()
    assert isinstance(out, pv.ImageData)


def test_flip_y():
    mesh = examples.load_airplane()
    flip_y1 = mesh.copy()
    flip_y2 = mesh.copy()
    flip_y1.flip_y(point=(0, 0, 0), inplace=True)
    flip_y2.points[:, 1] *= -1.0
    assert np.allclose(flip_y1.points, flip_y2.points)
    # Test non-point-based mesh doesn't fail
    mesh = examples.load_uniform()
    out = mesh.flip_y()
    assert isinstance(out, pv.ImageData)


def test_flip_z():
    mesh = examples.load_airplane()
    flip_z1 = mesh.copy()
    flip_z2 = mesh.copy()
    flip_z1.flip_z(point=(0, 0, 0), inplace=True)
    flip_z2.points[:, 2] *= -1.0
    assert np.allclose(flip_z1.points, flip_z2.points)
    # Test non-point-based mesh doesn't fail
    mesh = examples.load_uniform()
    out = mesh.flip_z()
    assert isinstance(out, pv.ImageData)


def test_flip_normal():
    mesh = examples.load_airplane()
    flip_normal1 = mesh.copy()
    flip_normal2 = mesh.copy()
    flip_normal1.flip_normal(normal=[1.0, 0.0, 0.0], inplace=True)
    flip_normal2.flip_x(inplace=True)
    assert np.allclose(flip_normal1.points, flip_normal2.points)

    flip_normal3 = mesh.copy()
    flip_normal4 = mesh.copy()
    flip_normal3.flip_normal(normal=[0.0, 1.0, 0.0], inplace=True)
    flip_normal4.flip_y(inplace=True)
    assert np.allclose(flip_normal3.points, flip_normal4.points)

    flip_normal5 = mesh.copy()
    flip_normal6 = mesh.copy()
    flip_normal5.flip_normal(normal=[0.0, 0.0, 1.0], inplace=True)
    flip_normal6.flip_z(inplace=True)
    assert np.allclose(flip_normal5.points, flip_normal6.points)

    # Test non-point-based mesh doesn't fail
    mesh = examples.load_uniform()
    out = mesh.flip_normal(normal=[1.0, 0.0, 0.5])
    assert isinstance(out, pv.ImageData)
