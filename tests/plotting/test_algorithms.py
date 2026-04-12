"""Tests for pyvista.plotting.utilities.algorithms."""

from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv
from pyvista.plotting.utilities.algorithms import ActiveScalarsAlgorithm
from pyvista.plotting.utilities.algorithms import AddIDsAlgorithm
from pyvista.plotting.utilities.algorithms import CallbackFilterAlgorithm
from pyvista.plotting.utilities.algorithms import PointSetToPolyDataAlgorithm
from pyvista.plotting.utilities.algorithms import SmoothShadingAlgorithm
from pyvista.plotting.utilities.algorithms import SourceAlgorithm
from pyvista.plotting.utilities.algorithms import active_scalars_algorithm
from pyvista.plotting.utilities.algorithms import add_ids_algorithm
from pyvista.plotting.utilities.algorithms import algorithm_to_mesh_handler
from pyvista.plotting.utilities.algorithms import callback_algorithm
from pyvista.plotting.utilities.algorithms import cell_data_to_point_data_algorithm
from pyvista.plotting.utilities.algorithms import crinkle_algorithm
from pyvista.plotting.utilities.algorithms import decimation_algorithm
from pyvista.plotting.utilities.algorithms import extract_surface_algorithm
from pyvista.plotting.utilities.algorithms import outline_algorithm
from pyvista.plotting.utilities.algorithms import point_data_to_cell_data_algorithm
from pyvista.plotting.utilities.algorithms import pointset_to_polydata_algorithm
from pyvista.plotting.utilities.algorithms import set_algorithm_input
from pyvista.plotting.utilities.algorithms import smooth_shading_algorithm
from pyvista.plotting.utilities.algorithms import triangulate_algorithm

ORIGINAL_POINT_IDS_NAME = SmoothShadingAlgorithm.ORIGINAL_POINT_IDS_NAME


@pytest.fixture
def sphere_with_scalars():
    mesh = pv.Sphere()
    mesh.point_data['z'] = mesh.points[:, 2]
    return mesh


@pytest.fixture
def cube_with_cell_data():
    mesh = pv.Cube()
    mesh.cell_data['cell_id'] = np.arange(mesh.n_cells)
    return mesh


def test_algorithm_to_mesh_handler_passthrough(sphere_with_scalars):
    mesh, algo = algorithm_to_mesh_handler(sphere_with_scalars)
    assert mesh is sphere_with_scalars
    assert algo is None


def test_algorithm_to_mesh_handler_with_algorithm(sphere_with_scalars):
    alg = outline_algorithm(sphere_with_scalars)
    mesh, algo = algorithm_to_mesh_handler(alg)
    assert isinstance(mesh, pv.DataSet)
    assert algo is alg


def test_set_algorithm_input_dataset(sphere_with_scalars):
    alg = ActiveScalarsAlgorithm(name='z', preference='point')
    set_algorithm_input(alg, sphere_with_scalars)
    alg.Update()
    out = pv.wrap(alg.GetOutputDataObject(0))
    assert 'z' in out.point_data


def test_set_algorithm_input_algorithm(sphere_with_scalars):
    alg1 = outline_algorithm(sphere_with_scalars)
    alg2 = extract_surface_algorithm(alg1)
    alg2.Update()
    out = pv.wrap(alg2.GetOutputDataObject(0))
    assert out.n_points > 0


def test_active_scalars_algo_point(sphere_with_scalars):
    algo = ActiveScalarsAlgorithm(name='z', preference='point')
    set_algorithm_input(algo, sphere_with_scalars)
    algo.Update()
    out = algo.GetOutputDataObject(0)
    scalars = out.GetPointData().GetScalars()
    assert scalars is not None
    assert scalars.GetName() == 'z'


def test_active_scalars_algo_cell(cube_with_cell_data):
    algo = ActiveScalarsAlgorithm(name='cell_id', preference='cell')
    set_algorithm_input(algo, cube_with_cell_data)
    algo.Update()
    out = algo.GetOutputDataObject(0)
    scalars = out.GetCellData().GetScalars()
    assert scalars is not None
    assert scalars.GetName() == 'cell_id'


def test_active_scalars_algo_shallow_copy(sphere_with_scalars):
    algo = ActiveScalarsAlgorithm(name='z', preference='point')
    set_algorithm_input(algo, sphere_with_scalars)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert np.shares_memory(sphere_with_scalars['z'], out['z'])


def test_active_scalars_algo_propagates_changes(sphere_with_scalars):
    algo = ActiveScalarsAlgorithm(name='z', preference='point')
    set_algorithm_input(algo, sphere_with_scalars)
    algo.Update()

    # Modify the original mesh in place
    sphere_with_scalars['z'][:] = 42.0
    algo.Modified()
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert np.allclose(out['z'], 42.0)


def test_active_scalars_algo_wrapper(sphere_with_scalars):
    algo = active_scalars_algorithm(sphere_with_scalars, 'z', preference='point')
    algo.Update()
    out = algo.GetOutputDataObject(0)
    assert out.GetPointData().GetScalars().GetName() == 'z'


def test_pointset_to_polydata_conversion():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    pointset = pv.PointSet(points)
    algo = PointSetToPolyDataAlgorithm()
    set_algorithm_input(algo, pointset)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert isinstance(out, pv.PolyData)
    assert out.n_points == 3


def test_pointset_to_polydata_wrapper():
    pointset = pv.PointSet(np.random.default_rng().random((10, 3)))
    algo = pointset_to_polydata_algorithm(pointset)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert isinstance(out, pv.PolyData)


def test_add_ids_point_and_cell(sphere_with_scalars):
    algo = AddIDsAlgorithm(point_ids=True, cell_ids=True)
    set_algorithm_input(algo, sphere_with_scalars)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert 'point_ids' in out.point_data
    assert 'cell_ids' in out.cell_data
    assert np.array_equal(out['point_ids'], np.arange(sphere_with_scalars.n_points))
    assert np.array_equal(out['cell_ids'], np.arange(sphere_with_scalars.n_cells))


def test_add_ids_point_only(sphere_with_scalars):
    algo = AddIDsAlgorithm(point_ids=True, cell_ids=False)
    set_algorithm_input(algo, sphere_with_scalars)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert 'point_ids' in out.point_data
    assert 'cell_ids' not in out.cell_data


def test_add_ids_cell_only(sphere_with_scalars):
    algo = AddIDsAlgorithm(point_ids=False, cell_ids=True)
    set_algorithm_input(algo, sphere_with_scalars)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert 'point_ids' not in out.point_data
    assert 'cell_ids' in out.cell_data


def test_add_ids_shallow_copy(sphere_with_scalars):
    algo = AddIDsAlgorithm(point_ids=True, cell_ids=False)
    set_algorithm_input(algo, sphere_with_scalars)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert np.shares_memory(sphere_with_scalars['z'], out['z'])


def test_add_ids_wrapper(sphere_with_scalars):
    algo = add_ids_algorithm(sphere_with_scalars, point_ids=True, cell_ids=True)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert 'point_ids' in out.point_data
    assert 'cell_ids' in out.cell_data


def test_crinkle():
    source = pv.Sphere()
    source.cell_data['data'] = np.arange(source.n_cells)
    clipped = source.clip()
    # Add cell_ids needed by crinkle
    clipped.cell_data['cell_ids'] = np.arange(clipped.n_cells)
    algo = crinkle_algorithm(clipped, source)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert out.n_cells > 0


def test_outline(sphere_with_scalars):
    algo = outline_algorithm(sphere_with_scalars)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert out.n_points == 8  # Bounding box corners
    assert out.n_cells > 0


def test_outline_with_faces(sphere_with_scalars):
    algo = outline_algorithm(sphere_with_scalars, generate_faces=True)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert out.n_points == 8


def test_extract_surface():
    grid = pv.ImageData(dimensions=(5, 5, 5))
    algo = extract_surface_algorithm(grid)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert isinstance(out, pv.PolyData)
    assert out.n_points > 0


def test_extract_surface_pass_ids():
    grid = pv.ImageData(dimensions=(5, 5, 5))
    algo = extract_surface_algorithm(grid, pass_pointid=True, pass_cellid=True)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert 'vtkOriginalPointIds' in out.point_data
    assert 'vtkOriginalCellIds' in out.cell_data


def test_cell_data_to_point_data(cube_with_cell_data):
    algo = cell_data_to_point_data_algorithm(cube_with_cell_data)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert 'cell_id' in out.point_data


def test_cell_data_to_point_data_pass_cell(cube_with_cell_data):
    algo = cell_data_to_point_data_algorithm(cube_with_cell_data, pass_cell_data=True)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert 'cell_id' in out.point_data
    assert 'cell_id' in out.cell_data


def test_point_data_to_cell_data(sphere_with_scalars):
    algo = point_data_to_cell_data_algorithm(sphere_with_scalars)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert 'z' in out.cell_data


def test_point_data_to_cell_data_pass_point(sphere_with_scalars):
    algo = point_data_to_cell_data_algorithm(sphere_with_scalars, pass_point_data=True)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert 'z' in out.cell_data
    assert 'z' in out.point_data


def test_triangulate():
    mesh = pv.Plane()
    algo = triangulate_algorithm(mesh)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert out.n_cells >= mesh.n_cells


def test_decimation(sphere_with_scalars):
    algo = decimation_algorithm(sphere_with_scalars, target_reduction=0.5)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))
    assert out.n_points < sphere_with_scalars.n_points


def test_active_scalars_algo_modified_propagates(sphere_with_scalars):
    """Re-running after Modified() must reflect the new array name.

    Targets the rewritten ``RequestData`` — the Modified() pathway is what
    lets ``DataSetMapper.set_active_scalars`` swap arrays in place without
    re-splicing the pipeline.
    """
    sphere_with_scalars.point_data['x'] = sphere_with_scalars.points[:, 0]

    algo = ActiveScalarsAlgorithm(name='x', preference='point')
    set_algorithm_input(algo, sphere_with_scalars)
    algo.Update()
    assert algo.GetOutputDataObject(0).GetPointData().GetScalars().GetName() == 'x'

    algo.scalars_name = 'z'  # setter calls Modified()
    algo.Update()
    assert algo.GetOutputDataObject(0).GetPointData().GetScalars().GetName() == 'z'


def test_active_scalars_algo_does_not_mutate_input(sphere_with_scalars):
    """The original mesh's active scalars must not change when the algo runs."""
    sphere_with_scalars.point_data['x'] = sphere_with_scalars.points[:, 0]
    sphere_with_scalars.set_active_scalars('x')

    algo = active_scalars_algorithm(sphere_with_scalars, 'z', preference='point')
    algo.Update()

    assert sphere_with_scalars.active_scalars_name == 'x'


def test_add_ids_preserves_active_scalars(sphere_with_scalars):
    """Adding ID arrays must not steal the active-scalars slot."""
    sphere_with_scalars.set_active_scalars('z')

    algo = add_ids_algorithm(sphere_with_scalars, point_ids=True, cell_ids=True)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))

    assert out.point_data.active_scalars_name == 'z'
    # Original is also untouched
    assert sphere_with_scalars.active_scalars_name == 'z'


def _run_smooth_shading(inp, **kwargs):
    algo = SmoothShadingAlgorithm(**kwargs)
    set_algorithm_input(algo, inp)
    algo.Update()
    return pv.wrap(algo.GetOutputDataObject(0)), algo


def test_smooth_shading_algo_output_is_polydata_from_polydata():
    sphere = pv.Sphere()
    out, _ = _run_smooth_shading(sphere)

    assert isinstance(out, pv.PolyData)
    assert out.point_data.active_normals is not None
    assert out.point_data.active_normals_name == 'Normals'
    assert out.n_points == sphere.n_points
    normals = np.asarray(out.point_data.active_normals)
    magnitudes = np.linalg.norm(normals, axis=1)
    assert np.allclose(magnitudes, 1.0, atol=1e-5)


def test_smooth_shading_algo_output_is_polydata_from_unstructured(hexbeam):
    out, _ = _run_smooth_shading(hexbeam)

    # Output type is always polydata (the surface), not the input type.
    assert isinstance(out, pv.PolyData)
    assert out.point_data.active_normals is not None
    # Surface points are a subset of the original grid's points.
    assert 0 < out.n_points <= hexbeam.n_points
    normals = np.asarray(out.point_data.active_normals)
    mags = np.linalg.norm(normals, axis=1)
    assert np.allclose(mags, 1.0, atol=1e-5)


def test_smooth_shading_algo_output_is_polydata_from_imagedata():
    grid = pv.ImageData(dimensions=(5, 5, 5))
    out, _ = _run_smooth_shading(grid)

    assert isinstance(out, pv.PolyData)
    assert out.point_data.active_normals is not None
    # Surface of a 5x5x5 ImageData: total minus interior = 5**3 - 3**3 = 98.
    assert out.n_points == 98
    normals = np.asarray(out.point_data.active_normals)
    magnitudes = np.linalg.norm(normals, axis=1)
    assert np.allclose(magnitudes, 1.0, atol=1e-5)


def test_smooth_shading_algo_split_sharp_edges_increases_points():
    cube = pv.Cube()
    smooth_out, _ = _run_smooth_shading(cube)
    split_out, _ = _run_smooth_shading(cube, split_sharp_edges=True, feature_angle=10.0)

    # Without splitting: point count preserved (8 shared vertices).
    assert smooth_out.n_points == cube.n_points
    # With splitting: every 90-degree edge exceeds 10-degree threshold, so each
    # of the 6 faces gets its own 4 corners: 6 * 4 = 24.
    assert split_out.n_points == 24


def test_smooth_shading_algo_split_sharp_edges_changes_normals():
    cube = pv.Cube()
    smooth_out, _ = _run_smooth_shading(cube)
    split_out, _ = _run_smooth_shading(cube, split_sharp_edges=True, feature_angle=10.0)

    # Smooth-shaded cube corners average three face normals; split corners
    # point along a single face normal. They must differ.
    smooth_norms = np.asarray(smooth_out.point_data.active_normals)
    split_norms = np.asarray(split_out.point_data.active_normals)
    assert smooth_norms.shape == (smooth_out.n_points, 3)
    assert split_norms.shape == (split_out.n_points, 3)
    # Split normals must each be axis-aligned (one component == ±1).
    max_abs = np.max(np.abs(split_norms), axis=1)
    assert np.allclose(max_abs, 1.0, atol=1e-5)


def test_smooth_shading_algo_tracker_maps_output_to_input(hexbeam):
    out, _ = _run_smooth_shading(hexbeam)

    assert ORIGINAL_POINT_IDS_NAME in out.point_data
    tracker = np.asarray(out.point_data[ORIGINAL_POINT_IDS_NAME])
    assert tracker.shape == (out.n_points,)
    # Tracker values must be valid indices into the input mesh.
    assert tracker.min() >= 0
    assert tracker.max() < hexbeam.n_points
    # Output surface points are a subset of input points at the tracked indices.
    assert np.allclose(out.points, hexbeam.points[tracker])


def test_smooth_shading_algo_tracker_chains_through_split():
    """Split-vertex points must map back to their original input vertices."""
    cube = pv.Cube()
    out, _ = _run_smooth_shading(cube, split_sharp_edges=True, feature_angle=10.0)

    tracker = np.asarray(out.point_data[ORIGINAL_POINT_IDS_NAME])
    assert tracker.shape == (out.n_points,)
    assert tracker.min() >= 0
    assert tracker.max() < cube.n_points
    # Each split copy of a vertex must sit at the same spatial location as
    # the original (splitting only duplicates, doesn't move).
    assert np.allclose(out.points, cube.points[tracker])
    # And split vertices really did get duplicated — at least one original
    # index appears more than once in the tracker.
    _, counts = np.unique(tracker, return_counts=True)
    assert (counts > 1).any()


def test_smooth_shading_algo_scalars_carry_through_split(sphere_with_scalars):
    """Point-data arrays on the input are split alongside their points."""
    out, _ = _run_smooth_shading(sphere_with_scalars, split_sharp_edges=True, feature_angle=10.0)

    assert 'z' in out.point_data
    assert out.point_data['z'].shape == (out.n_points,)
    # After split, each output point's 'z' value must equal the input 'z'
    # value at the tracked original index.
    tracker = np.asarray(out.point_data[ORIGINAL_POINT_IDS_NAME])
    assert np.allclose(
        np.asarray(out.point_data['z']),
        np.asarray(sphere_with_scalars.point_data['z'])[tracker],
    )


def test_smooth_shading_algo_texture_coords_survive_split():
    """Texture coordinates must survive extract_surface + compute_normals."""
    globe = pv.examples.load_globe()
    assert 'Texture Coordinates' in globe.point_data

    out, _ = _run_smooth_shading(globe, split_sharp_edges=True, feature_angle=30.0)

    assert isinstance(out, pv.PolyData)
    assert 'Texture Coordinates' in out.point_data
    assert out.point_data.active_normals is not None
    assert out.point_data['Texture Coordinates'].shape[1] == 2
    # Verify values via the tracker — each output point's texture coordinates
    # must match the input's at the tracked original index.
    tracker = np.asarray(out.point_data[ORIGINAL_POINT_IDS_NAME])
    original_tcoords = np.asarray(globe.point_data['Texture Coordinates'])
    output_tcoords = np.asarray(out.point_data['Texture Coordinates'])
    assert np.allclose(output_tcoords, original_tcoords[tracker])


def test_smooth_shading_algo_non_polydata_scalars_match_reference(hexbeam):
    """Scalars on the smooth-shaded output must match a manually constructed reference."""
    scalars_name = 'sample_point_scalars'
    assert scalars_name in hexbeam.point_data

    out, _ = _run_smooth_shading(hexbeam, split_sharp_edges=True)

    # Build expected output independently: extract surface then compute normals.
    expected = hexbeam.extract_surface(algorithm=None).compute_normals(
        cell_normals=False,
        split_vertices=True,
    )

    assert scalars_name in out.point_data
    assert np.allclose(
        np.asarray(out.point_data[scalars_name]),
        np.asarray(expected.point_data[scalars_name]),
    )


def test_smooth_shading_algo_feature_angle_setter_modifies():
    cube = pv.Cube()
    algo = SmoothShadingAlgorithm(split_sharp_edges=True, feature_angle=10.0)
    set_algorithm_input(algo, cube)
    algo.Update()
    initial_mtime = algo.GetMTime()

    algo.feature_angle = 60.0
    assert algo.GetMTime() > initial_mtime
    assert algo.feature_angle == 60.0

    # Setting to the same value does not bump MTime.
    mtime_before = algo.GetMTime()
    algo.feature_angle = 60.0
    assert algo.GetMTime() == mtime_before


def test_smooth_shading_algo_split_sharp_edges_setter_modifies():
    algo = SmoothShadingAlgorithm()
    set_algorithm_input(algo, pv.Sphere())
    algo.Update()
    initial_mtime = algo.GetMTime()

    algo.split_sharp_edges = True
    assert algo.GetMTime() > initial_mtime
    assert algo.split_sharp_edges is True

    mtime_before = algo.GetMTime()
    algo.split_sharp_edges = True
    assert algo.GetMTime() == mtime_before


def test_smooth_shading_algo_passthrough_line_input():
    """Line-only meshes pass through without computed normals."""
    line = pv.Line()
    out, _ = _run_smooth_shading(line)

    assert isinstance(out, pv.PolyData)
    # Lines can't have normals computed; output passes through unchanged.
    assert out.point_data.active_normals is None
    assert out.n_points == line.n_points


@pytest.mark.parametrize('split_sharp_edges', [True, False])
def test_smooth_shading_algo_passthrough_point_cloud(split_sharp_edges):
    """Vertex-only point clouds pass through without normals (parametrized)."""
    point_cloud = pv.PolyData([0.0, 0.0, 0.0])
    assert point_cloud.n_verts == point_cloud.n_cells

    out, _ = _run_smooth_shading(point_cloud, split_sharp_edges=split_sharp_edges)

    assert isinstance(out, pv.PolyData)
    assert 'Normals' not in out.point_data
    assert out.n_points == point_cloud.n_points


def test_smooth_shading_algo_passthrough_empty_input():
    empty = pv.PolyData()
    out, _ = _run_smooth_shading(empty)

    assert out.n_points == 0


def test_smooth_shading_algo_does_not_mutate_input():
    sphere = pv.Sphere()
    sphere.point_data['z'] = sphere.points[:, 2]
    original_array_names = set(sphere.point_data.keys())

    _run_smooth_shading(sphere, split_sharp_edges=True, feature_angle=10.0)

    # No new arrays should appear on the original input.
    assert set(sphere.point_data.keys()) == original_array_names
    assert 'pyvistaOriginalPointIds' not in sphere.point_data
    assert ORIGINAL_POINT_IDS_NAME not in sphere.point_data


def test_smooth_shading_algo_propagates_changes_on_modified():
    cube = pv.Cube()

    algo = SmoothShadingAlgorithm(split_sharp_edges=False)
    set_algorithm_input(algo, cube)
    algo.Update()
    n_points_before = algo.GetOutputDataObject(0).GetNumberOfPoints()

    algo.split_sharp_edges = True
    algo.feature_angle = 10.0
    algo.Update()
    n_points_after = algo.GetOutputDataObject(0).GetNumberOfPoints()

    # Enabling split_sharp_edges must re-run the pipeline and produce more
    # points at cube corners.
    assert n_points_after > n_points_before


def test_smooth_shading_algo_wrapper():
    sphere = pv.Sphere()
    algo = smooth_shading_algorithm(sphere, split_sharp_edges=True, feature_angle=10.0)
    algo.Update()
    out = pv.wrap(algo.GetOutputDataObject(0))

    assert isinstance(out, pv.PolyData)
    assert out.point_data.active_normals is not None


def test_smooth_shading_algo_composes_with_active_scalars(sphere_with_scalars):
    stage1 = active_scalars_algorithm(sphere_with_scalars, 'z', preference='point')
    stage2 = smooth_shading_algorithm(stage1, split_sharp_edges=True, feature_angle=10.0)
    stage2.Update()
    out = pv.wrap(stage2.GetOutputDataObject(0))

    # Active scalars survives the smooth shading stage (carried through the
    # split by vtkPolyDataNormals along with every other point-data array).
    assert out.point_data.active_scalars_name == 'z'
    assert out.point_data['z'].shape == (out.n_points,)
    # And normals were computed.
    assert out.point_data.active_normals is not None
    # Verify scalar values are correct via the tracker.
    tracker = np.asarray(out.point_data[ORIGINAL_POINT_IDS_NAME])
    expected_z = np.asarray(sphere_with_scalars.point_data['z'])[tracker]
    assert np.allclose(out.point_data['z'], expected_z)


def test_smooth_shading_algo_cell_scalars_carry_through(hexbeam):
    """Cell-data arrays on a non-polydata input survive surface extraction."""
    hexbeam.cell_data['cid'] = np.arange(hexbeam.n_cells, dtype=float)

    out, _ = _run_smooth_shading(hexbeam)

    assert isinstance(out, pv.PolyData)
    # Cell data carried through extract_surface -> compute_normals.
    assert 'cid' in out.cell_data
    assert out.cell_data['cid'].shape == (out.n_cells,)
    # Verify values: each surface cell's 'cid' must map back to a valid
    # cell index from the original hexbeam via the cell tracker.
    assert 'vtkOriginalCellIds' in out.cell_data
    cell_tracker = np.asarray(out.cell_data['vtkOriginalCellIds'])
    expected_cid = np.asarray(hexbeam.cell_data['cid'])[cell_tracker]
    assert np.allclose(out.cell_data['cid'], expected_cid)


def test_smooth_shading_pipeline_propagates_input_data_changes():
    """Modifying input data and re-updating the pipeline must produce new output.

    This tests the full pipeline: SourceAlgorithm → CallbackFilterAlgorithm →
    ActiveScalarsAlgorithm → SmoothShadingAlgorithm.  When the source data
    changes, the end-of-pipeline output must reflect those changes.
    """
    mesh = pv.Wavelet()
    mesh.point_data['z'] = mesh.points[:, 2].astype(float)

    source = SourceAlgorithm(lambda: mesh, output_type=type(mesh))
    to_surface = callback_algorithm(
        source,
        lambda m: m.extract_surface(algorithm=None),
        output_type=pv.PolyData,
    )
    stage1 = active_scalars_algorithm(to_surface, 'z', preference='point')
    stage2 = smooth_shading_algorithm(stage1)
    stage2.Update()
    out = pv.wrap(stage2.GetOutputDataObject(0))

    z_before = np.array(out['z'])
    assert z_before.min() < 0

    mesh.point_data['z'][:] = 0
    source.Modified()

    stage2.Update()
    out_after = pv.wrap(stage2.GetOutputDataObject(0))

    z_after = np.array(out_after['z'])
    assert np.allclose(z_after, 0.0)
    assert out_after.point_data.active_scalars_name == 'z'
    assert out_after.point_data.active_normals is not None


def test_source_algorithm_default_output():
    sphere = pv.Sphere()
    alg = SourceAlgorithm(lambda: sphere)
    alg.Update()
    out = pv.wrap(alg.GetOutputDataObject(0))

    assert isinstance(out, pv.UnstructuredGrid)
    assert out.n_points == sphere.n_points


def test_source_algorithm_output_type_pyvista_class():
    alg = SourceAlgorithm(pv.Sphere, output_type=pv.PolyData)
    alg.Update()
    out = pv.wrap(alg.GetOutputDataObject(0))

    assert isinstance(out, pv.PolyData)
    assert out.n_points > 0


def test_source_algorithm_output_type_string():
    alg = SourceAlgorithm(pv.Sphere, output_type='vtkPolyData')
    alg.Update()
    out = pv.wrap(alg.GetOutputDataObject(0))

    assert isinstance(out, pv.PolyData)


def test_source_algorithm_modified_re_executes():
    counter = [0]

    def gen():
        counter[0] += 1
        return pv.Sphere()

    alg = SourceAlgorithm(gen, output_type=pv.PolyData)
    alg.Update()
    assert counter[0] == 1

    alg.Modified()
    alg.Update()
    assert counter[0] == 2


def test_source_algorithm_feeds_pipeline():
    mesh = pv.Sphere()
    mesh.point_data['z'] = mesh.points[:, 2]

    source = SourceAlgorithm(lambda: mesh, output_type=pv.PolyData)
    stage = active_scalars_algorithm(source, 'z', preference='point')
    stage.Update()
    out = pv.wrap(stage.GetOutputDataObject(0))

    assert out.point_data.active_scalars_name == 'z'


def test_callback_filter_preserves_type(sphere_with_scalars):
    alg = CallbackFilterAlgorithm(callback=lambda m: m.copy())
    set_algorithm_input(alg, sphere_with_scalars)
    alg.Update()
    out = pv.wrap(alg.GetOutputDataObject(0))

    assert isinstance(out, pv.PolyData)
    assert 'z' in out.point_data
    assert out.n_points == sphere_with_scalars.n_points


def test_callback_filter_override_output_type(sphere_with_scalars):
    alg = CallbackFilterAlgorithm(
        callback=lambda m: m.cast_to_unstructured_grid(),
        output_type=pv.UnstructuredGrid,
    )
    set_algorithm_input(alg, sphere_with_scalars)
    alg.Update()
    out = pv.wrap(alg.GetOutputDataObject(0))

    assert isinstance(out, pv.UnstructuredGrid)
    assert 'z' in out.point_data


def test_callback_filter_output_type_string():
    alg = CallbackFilterAlgorithm(
        callback=lambda m: m,
        output_type='vtkPolyData',
    )
    set_algorithm_input(alg, pv.Sphere())
    alg.Update()
    out = pv.wrap(alg.GetOutputDataObject(0))

    assert isinstance(out, pv.PolyData)


def test_callback_filter_wrapper(sphere_with_scalars):
    alg = callback_algorithm(sphere_with_scalars, lambda m: m.copy())
    alg.Update()
    out = pv.wrap(alg.GetOutputDataObject(0))

    assert isinstance(out, pv.PolyData)
    assert out.n_points == sphere_with_scalars.n_points


def test_callback_filter_chains_in_pipeline(sphere_with_scalars):
    stage1 = callback_algorithm(sphere_with_scalars, lambda m: m.copy())
    stage2 = active_scalars_algorithm(stage1, 'z', preference='point')
    stage2.Update()
    out = pv.wrap(stage2.GetOutputDataObject(0))

    assert out.point_data.active_scalars_name == 'z'


def test_callback_filter_propagates_input_changes(sphere_with_scalars):
    alg = callback_algorithm(sphere_with_scalars, lambda m: m.copy())
    alg.Update()
    out = pv.wrap(alg.GetOutputDataObject(0))
    assert not np.allclose(out['z'], 42.0)

    sphere_with_scalars['z'][:] = 42.0
    alg.Modified()
    alg.Update()
    out = pv.wrap(alg.GetOutputDataObject(0))
    assert np.allclose(out['z'], 42.0)
