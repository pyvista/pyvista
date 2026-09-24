from __future__ import annotations

from collections.abc import Sized
import itertools
import re
import sys
from typing import TYPE_CHECKING
from typing import Literal
from typing import get_args
import warnings

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

import pyvista as pv
from pyvista import PyVistaDeprecationWarning
from pyvista import _vtk
from pyvista import examples
from pyvista.core.cell import _get_connectivity_array
from pyvista.core.errors import DeprecationError
from pyvista.core.errors import PointSetCellOperationError
from pyvista.core.errors import PointSetDimensionReductionError
from pyvista.core.errors import PointSetNotSupported
from pyvista.core.filters.data_object import _PYVISTA_CELL_STATUS_INFO
from pyvista.core.filters.data_object import _SENTINEL
from pyvista.core.filters.data_object import _VTK_CELL_STATUS_INFO
from pyvista.core.filters.data_object import _convex_hull_scipy
from pyvista.core.filters.data_object import _get_cell_quality_measures
from pyvista.core.utilities._cell_lengths import _cell_edge_lengths
from pyvista.core.utilities.cell_quality import _CellQualityLiteral
from pyvista.core.utilities.helpers import _NORMALS
from pyvista.core.utilities.helpers import generate_plane
from tests.core.test_dataset_filters import HYPOTHESIS_MAX_EXAMPLES
from tests.core.test_dataset_filters import n_numbers
from tests.core.test_dataset_filters import normals
from tests.vtk_backend_divergence import CELL_STATUS_ENUM

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# CellStatus sorted by lower-case names, excluding VALID state
CELL_STATUS_ARRAY_NAMES = [
    val.name.lower()
    for val in sorted(
        pv.CellStatus,
        key=lambda v: v.name.lower(),
    )
    if val != pv.CellStatus.VALID
]


@pytest.mark.parametrize('return_clipped', [True, False])
@pytest.mark.parametrize('crinkle', [True, False])
def test_clip_filter(multiblock_all_with_nested_and_none, return_clipped, crinkle):
    """This tests the clip filter on all datatypes available filters"""
    # Remove None blocks in the root block but keep the none block in the nested MultiBlock
    multi = multiblock_all_with_nested_and_none
    for i, block in enumerate(multi):
        if block is None:
            del multi[i]
    assert None not in multi
    assert None in multi.recursive_iterator()

    # Center datasets at origin so that clip actually removes part of the mesh
    for block in multi.recursive_iterator(skip_none=True):
        center = np.array(block.center)
        block.translate(-center, inplace=True)

    for dataset in multi:
        bounds_before_clip = dataset.bounds
        clips = dataset.clip(
            normal='x', invert=True, return_clipped=return_clipped, crinkle=crinkle
        )
        assert clips is not None

        if return_clipped:
            assert isinstance(clips, tuple)
            assert len(clips) == 2
        else:
            assert isinstance(clips, pv.DataObject)
            # Make dataset iterable
            clips = [clips]

        for clip in clips:
            if isinstance(dataset, pv.PointSet):
                assert isinstance(clip, pv.PointSet)
            elif isinstance(dataset, pv.PolyData):
                assert isinstance(clip, pv.PolyData)
            elif isinstance(dataset, pv.MultiBlock):
                assert isinstance(clip, pv.MultiBlock)
                assert clip.n_blocks == dataset.n_blocks
            else:
                assert isinstance(clip, pv.UnstructuredGrid)

            bounds_after_clip = clip.bounds
            if (
                isinstance(dataset, pv.PointSet)
                and pv.vtk_version_info >= (9, 4)
                and pv.vtk_version_info < (9, 5)
            ):
                pytest.xfail("VTK 9.4 bug where clipping PointSet doesn't work")
            assert not np.allclose(bounds_before_clip, bounds_after_clip)


@pytest.mark.parametrize('as_composite', [True, False])
def test_clip_filter_pointset_no_points_removed(pointset, as_composite):
    n_points_in = pointset.n_points
    mesh = pv.MultiBlock([pointset]) if as_composite else pointset
    # Make sure we clip such that none of the points are removed
    # This ensures output bounds == input bounds which hits a branch where
    # remove_unused_points may be called
    bounds = pointset.bounds
    clipped = mesh.clip(origin=(bounds.x_max + 1, bounds.y_max, bounds.z_max))
    pointset_out = clipped[0] if as_composite else clipped

    if as_composite and pv.vtk_version_info >= (9, 4) and pv.vtk_version_info < (9, 5):
        assert pointset_out.is_empty
        pytest.xfail("VTK 9.4 bug where clipping PointSet doesn't work")
    assert np.allclose(clipped.bounds, bounds)

    assert isinstance(pointset_out, pv.PointSet)
    n_points_out = pointset_out.n_points
    assert n_points_in == n_points_out


@pytest.mark.parametrize(
    'mesh',
    [
        pv.Sphere(),
        pv.PointSet(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])),
        pv.ImageData(dimensions=(5, 5, 5)).cast_to_unstructured_grid(),
    ],
    ids=['polydata', 'pointset', 'unstructured'],
)
def test_clip_inplace(mesh):
    mesh = mesh.copy()
    n_points_in = mesh.n_points
    clipped = mesh.clip(inplace=True)
    assert clipped is mesh
    assert mesh.n_points < n_points_in


@pytest.mark.parametrize(
    'mesh',
    [
        pv.ImageData(dimensions=(5, 5, 5)),
        pv.RectilinearGrid(*[np.linspace(-1, 1, 5)] * 3),
        pv.MultiBlock([pv.Sphere()]),
    ],
    ids=['image', 'rectilinear', 'composite'],
)
def test_clip_inplace_raises(mesh):
    match = f'Cannot use inplace=True for {type(mesh).__name__} input'
    with pytest.raises(TypeError, match=match):
        mesh.clip(inplace=True)


def _seam_polydata():
    """Two triangles meeting along a diagonal, sharing coordinates but no points."""
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float
    )
    mesh = pv.PolyData(points, np.array([3, 0, 1, 2, 3, 3, 4, 5]))
    mesh.cell_data['half'] = np.array([0.0, 1.0])
    mesh.point_data['height'] = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    mesh.point_data['scalars'] = mesh.point_data['height']
    return mesh


def _joins_the_halves(mesh):
    """Whether any point is shared between cells of both halves."""
    grid = mesh if isinstance(mesh, pv.UnstructuredGrid) else mesh.cast_to_unstructured_grid()
    half = np.asarray(mesh.cell_data['half'])
    halves_of_point = {}
    for i in range(mesh.n_cells):
        for point_id in grid.get_cell(i).point_ids:
            halves_of_point.setdefault(point_id, set()).add(round(float(half[i])))
    return any(len(halves) > 1 for halves in halves_of_point.values())


@pytest.mark.parametrize('name', ['clip', 'clip_box', 'clip_slab', 'clip_surface', 'clip_scalar'])
def test_clip_strips_does_not_duplicate_points(name):
    """A mesh mixing strips with other cells keeps one point per position, quietly."""
    points = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [1, 3, 0]], dtype=float)
    mesh = pv.PolyData(points, faces=[3, 0, 1, 2], strips=[4, 0, 1, 3, 2])
    mesh.point_data['scalars'] = np.linspace(0.0, 1.0, mesh.n_points)

    with pv.VtkErrorCatcher() as catcher:
        clipped = _clip_that_removes_nothing(mesh, name)

    assert not catcher.events
    positions = {point.tobytes() for point in np.ascontiguousarray(clipped.points)}
    assert clipped.n_points == len(positions)


@pytest.mark.parametrize('name', ['clip', 'clip_slab'])
def test_clip_composite_empty_block_keeps_array_names(name):
    """An empty block says which arrays it would have had, as a lone dataset does."""
    plane = pv.Plane()
    plane.point_data['height'] = plane.points[:, 2]
    composite = pv.MultiBlock({'plane': plane, 'empty': None})

    clipped = {
        'clip': lambda: composite.clip(normal='z', origin=(0, 0, 99), invert=False),
        'clip_slab': lambda: composite.clip_slab(thickness=0.001, normal='z', origin=(0, 0, 99)),
    }[name]()

    assert clipped['plane'].is_empty
    assert sorted(clipped['plane'].array_names) == sorted(plane.array_names)
    assert clipped['empty'] is None


@pytest.mark.parametrize('invert', [True, False])
def test_clip_box_merge_points_welds_when_asked(invert):
    """``merge_points`` decides whether coincident points are joined."""
    grid = pv.ImageData(dimensions=(5, 5, 5)).cast_to_unstructured_grid()
    centers = grid.cell_centers().points[:, 2]
    lower = grid.extract_cells(np.flatnonzero(centers < 2)).cast_to_unstructured_grid()
    upper = grid.extract_cells(np.flatnonzero(centers > 2)).cast_to_unstructured_grid()
    lower.cell_data['half'] = np.zeros(lower.n_cells)
    upper.cell_data['half'] = np.ones(upper.n_cells)
    seam = lower.merge(upper, merge_points=False)
    # A box that cuts in x only, so both halves survive either way
    bounds = [3.0, 99.0, -99.0, 99.0, -99.0, 99.0]

    merged = seam.clip_box(bounds, invert=invert, merge_points=True)
    unmerged = seam.clip_box(bounds, invert=invert, merge_points=False)

    assert _joins_the_halves(merged)
    assert not _joins_the_halves(unmerged)
    assert merged.n_points < unmerged.n_points
    assert merged.volume == pytest.approx(unmerged.volume)


def test_clip_empty_output_keeps_array_names():
    """An empty clip still says which arrays the input had."""
    mesh = pv.Plane()
    mesh.point_data['height'] = mesh.points[:, 2].astype(np.float32)
    mesh.cell_data['ids'] = np.arange(mesh.n_cells, dtype=np.uint16)

    clipped = mesh.clip(normal='z', origin=(0, 0, 99), invert=False)

    assert clipped.is_empty
    assert sorted(clipped.array_names) == sorted(mesh.array_names)
    assert clipped.point_data['height'].dtype == np.float32
    assert clipped.cell_data['ids'].dtype == np.uint16


def _cell_type_meshes():
    """One mesh per input class and per cell type a clip has to preserve."""
    axis = np.linspace(-1.0, 1.0, 4)
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    explicit = pv.StructuredGrid(x, y, z)
    explicit.dimensions = [4, 4, 4]
    image = pv.ImageData(dimensions=(4, 4, 4), spacing=(0.6, 0.6, 0.6), origin=(-1, -1, -1))
    meshes = {
        'PolyData triangles': pv.Sphere(theta_resolution=8, phi_resolution=8),
        'PolyData quads': pv.Plane(i_resolution=3, j_resolution=3),
        'PolyData lines': pv.PolyData(
            np.array([[x, 0.0, 0.0] for x in np.linspace(-1.0, 1.0, 5)]),
            lines=[2, 0, 1, 2, 1, 2, 2, 2, 3, 2, 3, 4],
        ),
        'PolyData verts': pv.PolyData(np.random.default_rng(0).uniform(-0.6, 0.6, (60, 3))),
        'ImageData': image,
        'RectilinearGrid': pv.RectilinearGrid(axis, axis, axis),
        'StructuredGrid': pv.StructuredGrid(x, y, z),
        'ExplicitStructuredGrid': explicit.cast_to_explicit_structured_grid(),
        'UnstructuredGrid hexahedra': image.cast_to_unstructured_grid(),
        'UnstructuredGrid tetrahedra': pv.Sphere(
            theta_resolution=8, phi_resolution=8
        ).delaunay_3d(),
    }
    for mesh in meshes.values():
        mesh.point_data['scalars'] = np.linspace(0.0, 1.0, mesh.n_points)
    return meshes


def _cell_types(mesh):
    """The cell types of a mesh, as an unstructured grid reports them."""
    grid = mesh if isinstance(mesh, pv.UnstructuredGrid) else mesh.cast_to_unstructured_grid()
    # A voxel is the same eight points as a hexahedron, ordered differently
    return {
        pv.CellType.HEXAHEDRON if t == pv.CellType.VOXEL else pv.CellType(t)
        for t in grid.celltypes
    }


_CELL_TYPE_MESHES = _cell_type_meshes()


def _clip_that_removes_nothing(mesh, name):
    """Apply one clip whose region contains the whole mesh."""
    enclosing = pv.Cube(center=(0, 0, 0), x_length=99, y_length=99, z_length=99)
    return {
        'clip': lambda: mesh.clip(normal='z', origin=(0, 0, -99), invert=False),
        'clip_box': lambda: mesh.clip_box([-99.0, 99.0] * 3, invert=False),
        'clip_slab': lambda: mesh.clip_slab(thickness=999.0, normal='z'),
        'clip_surface': lambda: mesh.clip_surface(enclosing),
        'clip_scalar': lambda: mesh.clip_scalar(scalars='scalars', value=-99.0, invert=False),
    }[name]()


CLIP_FILTERS = ['clip', 'clip_box', 'clip_slab', 'clip_surface', 'clip_scalar']


@pytest.mark.parametrize('name', CLIP_FILTERS)
@pytest.mark.parametrize('mesh_type', list(_CELL_TYPE_MESHES))
def test_clip_keeps_the_cell_types_it_does_not_cut(mesh_type, name):
    """A clip that removes nothing leaves every cell as the type it was."""
    mesh = _CELL_TYPE_MESHES[mesh_type].copy()

    clipped = _clip_that_removes_nothing(mesh, name)

    assert clipped.n_cells == mesh.n_cells
    assert _cell_types(clipped) == _cell_types(mesh)


@pytest.mark.parametrize('name', CLIP_FILTERS)
@pytest.mark.parametrize('mesh_type', list(_CELL_TYPE_MESHES))
def test_clip_keeps_the_points_it_does_not_cut(mesh_type, name):
    """A clip that removes nothing neither adds nor merges points."""
    mesh = _CELL_TYPE_MESHES[mesh_type].copy()

    clipped = _clip_that_removes_nothing(mesh, name)

    assert clipped.n_points == mesh.n_points
    assert np.allclose(np.sort(clipped.points, axis=0), np.sort(mesh.points, axis=0))


@pytest.mark.parametrize('name', CLIP_FILTERS)
@pytest.mark.parametrize('mesh_type', list(_CELL_TYPE_MESHES))
def test_clip_splits_the_mesh_in_two(mesh_type, name):
    """What a clip keeps and what it removes add up to the whole mesh."""
    mesh = _CELL_TYPE_MESHES[mesh_type].copy()
    center = np.array(mesh.center)
    surface = pv.Sphere(
        radius=0.6,
        center=(center[0] + 0.3, center[1], center[2]),
        theta_resolution=16,
        phi_resolution=16,
    )
    kept, removed = {
        'clip': lambda: (
            mesh.clip(normal='x', origin=center, invert=False),
            mesh.clip(normal='x', origin=center, invert=True),
        ),
        'clip_box': lambda: (
            mesh.clip_box([-0.4, 0.4] * 3, invert=False),
            mesh.clip_box([-0.4, 0.4] * 3, invert=True),
        ),
        'clip_slab': lambda: (
            mesh.clip_slab(thickness=0.8, normal='x', origin=center),
            mesh.clip_slab(thickness=0.8, normal='x', origin=center, invert=True),
        ),
        'clip_surface': lambda: (
            mesh.clip_surface(surface, invert=True),
            mesh.clip_surface(surface, invert=False),
        ),
        'clip_scalar': lambda: (
            mesh.clip_scalar(scalars='scalars', value=0.5, invert=True),
            mesh.clip_scalar(scalars='scalars', value=0.5, invert=False),
        ),
    }[name]()

    # A split, not a pass-through, on both sides
    assert kept.n_cells
    assert removed.n_cells
    if mesh_type == 'PolyData verts':
        assert kept.n_cells + removed.n_cells == mesh.n_cells
        return
    if mesh_type == 'PolyData lines':

        def measure(part):
            return part.compute_cell_sizes(length=True)['Length'].sum()
    else:
        attr = 'area' if isinstance(mesh, pv.PolyData) else 'volume'

        def measure(part):
            return getattr(part, attr)

    assert measure(mesh) > 0
    assert measure(kept) + measure(removed) == pytest.approx(measure(mesh), rel=1e-6)


def _seam_grid():
    """Two grid halves that touch but share no points."""
    grid = pv.ImageData(dimensions=(5, 5, 5)).cast_to_unstructured_grid()
    centers = grid.cell_centers().points[:, 2]
    lower = grid.extract_cells(np.flatnonzero(centers < 2)).cast_to_unstructured_grid()
    upper = grid.extract_cells(np.flatnonzero(centers > 2)).cast_to_unstructured_grid()
    lower.cell_data['half'] = np.zeros(lower.n_cells)
    upper.cell_data['half'] = np.ones(upper.n_cells)
    seam = lower.merge(upper, merge_points=False)
    seam.point_data['scalars'] = np.linspace(0.0, 1.0, seam.n_points)
    return seam


@pytest.mark.parametrize(
    'name',
    [
        'clip',
        'clip_slab',
        'clip_surface',
        'clip_scalar',
        'clip_scalar both=True',
        'clip_box merge_points=False',
    ],
)
@pytest.mark.parametrize('mesh_type', ['PolyData', 'UnstructuredGrid'])
def test_clip_keeps_coincident_points_apart_for_every_filter(mesh_type, name):
    """No clip welds points the input held apart, except when asked to."""
    mesh = _seam_polydata() if mesh_type == 'PolyData' else _seam_grid()
    if name == 'clip_box merge_points=False':
        clipped = mesh.clip_box([-99.0, 99.0] * 3, invert=False, merge_points=False)
    elif name == 'clip_scalar both=True':
        clipped = mesh.clip_scalar(scalars='scalars', value=-99.0, invert=False, both=True)[0]
    else:
        clipped = _clip_that_removes_nothing(mesh, name)

    assert clipped.n_points == mesh.n_points
    assert not _joins_the_halves(clipped)


@pytest.mark.parametrize('mesh_type', ['PolyData', 'UnstructuredGrid'])
def test_clip_box_merge_points_true_welds_every_input(mesh_type):
    """``merge_points=True`` joins points that share a position."""
    mesh = _seam_polydata() if mesh_type == 'PolyData' else _seam_grid()

    merged = mesh.clip_box([-99.0, 99.0] * 3, invert=False, merge_points=True)

    assert merged.n_points < mesh.n_points
    assert _joins_the_halves(merged)


@pytest.mark.parametrize('invert', [True, False])
@pytest.mark.parametrize(
    'make',
    [
        pytest.param(lambda: (pv.Cube(), list(pv.Cube().bounds)), id='PolyData, six bounds'),
        pytest.param(lambda: (pv.Cube(), pv.Cube()), id='PolyData, box mesh'),
        pytest.param(
            lambda: (pv.Cube().cast_to_unstructured_grid(), list(pv.Cube().bounds)),
            id='UnstructuredGrid',
        ),
        pytest.param(
            lambda: (
                pv.Cube(center=(5e6, 0, 0), x_length=0.1, y_length=0.1, z_length=0.1),
                list(pv.Cube(center=(5e6, 0, 0), x_length=0.1, y_length=0.1, z_length=0.1).bounds),
            ),
            id='PolyData far from the origin',
        ),
    ],
)
def test_clip_box_keeps_a_face_lying_in_a_box_plane(make, invert):
    """A box that only touches a face keeps it, whichever way the box is given."""
    mesh, box = make()

    clipped = mesh.clip_box(box, invert=invert)

    assert clipped.n_cells == (0 if invert else mesh.n_cells)


def test_clip_filter_normal(datasets):
    # Test no errors are raised
    for i, dataset in enumerate(datasets):
        dataset.clip(normal=normals[i % len(normals)], invert=True)


@pytest.mark.parametrize('dataset', [pv.PolyData(), pv.MultiBlock()])
def test_clip_filter_empty_inputs(dataset):
    dataset.clip('x')


@pytest.mark.parametrize('as_composite', [True, False])
@pytest.mark.parametrize(
    'clip',
    [
        lambda mesh: mesh.clip(crinkle=True),
        lambda mesh: mesh.clip(crinkle=True, return_clipped=True),
        lambda mesh: mesh.clip_box(crinkle=True),
        lambda mesh: mesh.clip_slab(thickness=1.0, crinkle=True),
    ],
)
def test_clip_crinkle_does_not_modify_input(uniform, as_composite, clip):
    mesh = pv.MultiBlock([uniform, uniform.copy()]) if as_composite else uniform
    blocks = list(mesh.recursive_iterator()) if as_composite else [mesh]
    before = [(block.array_names, block.active_scalars_name) for block in blocks]

    clip(mesh)

    assert [(block.array_names, block.active_scalars_name) for block in blocks] == before


def test_clip_box_does_not_modify_bounds_mesh(uniform):
    box = pv.Cube(center=uniform.center, x_length=3, y_length=3, z_length=3)
    before = box.copy()
    clipped = uniform.clip_box(box, invert=False)
    assert clipped.n_cells
    assert box == before


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


@pytest.mark.parametrize(
    'filt',
    [
        pv.DataObjectFilters.clip,
        pv.DataObjectFilters.slice,
        pv.PolyDataFilters.clip_closed_surface,
        pv.PolyDataFilters.project_points_to_plane,
    ],
)
def test_filters_with_plane_keyword(filt, ant):
    origin = (1, 2, 3)
    normal = (4, 5, 6)
    plane = pv.Plane(center=origin, direction=normal)
    output_no_plane = filt(ant, origin=origin, normal=normal)
    output_with_plane = filt(ant, plane=plane)
    assert np.allclose(output_no_plane.bounds, output_with_plane.bounds)

    match = 'The plane mesh must be planar. Got a non-planar mesh with dimensionality of 3.'
    with pytest.raises(ValueError, match=match):
        filt(ant, plane=pv.Box())

    match = 'The `normal` and `origin` parameters cannot be set when `plane` is specified.'
    with pytest.raises(ValueError, match=match):
        filt(ant, plane=plane, normal='x')
    with pytest.raises(ValueError, match=match):
        filt(ant, plane=plane, origin=(0, 0, 0))


def test_transform_raises(sphere):
    matrix = np.diag((1, 1, 1, 0))
    match = re.escape('Transform element (3,3), the inverse scale term, is zero')
    with pytest.raises(ValueError, match=match):
        sphere.transform(matrix, inplace=False)


@pytest.mark.parametrize('crinkle', [True, False])
def test_clip_box_output_type(multiblock_all_with_nested_and_none, crinkle):
    multiblock_all_with_nested_and_none.clean()
    for dataset in multiblock_all_with_nested_and_none:
        clp = dataset.clip_box(invert=True, progress_bar=True, crinkle=crinkle)
        assert clp is not None
        assert isinstance(clp, (pv.UnstructuredGrid, pv.PolyData, pv.MultiBlock, pv.PointSet))
        if isinstance(clp, pv.MultiBlock):
            # Every block keeps the class its own type gives, as a lone input does
            assert all(
                isinstance(block, (pv.UnstructuredGrid, pv.PolyData, pv.PointSet))
                for block in clp.recursive_iterator(skip_none=True)
            )
        clp2 = dataset.clip_box(merge_points=False)
        assert clp2 is not None


def test_clip_box():
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
    vol = surf.voxelize()
    cube = pv.Cube().rotate_x(33, inplace=False)
    clp = vol.clip_box(bounds=cube, invert=False, crinkle=True)
    assert clp is not None


@pytest.mark.parametrize('crinkle', [True, False])
def test_clip_empty(crinkle):
    out = pv.PolyData().clip(crinkle=crinkle, return_clipped=False)
    assert out.is_empty

    out1, _out2 = pv.PolyData().clip(crinkle=crinkle, return_clipped=True)
    assert out1.is_empty

    out = pv.PolyData().clip_box(crinkle=crinkle)
    assert out.is_empty


@pytest.mark.parametrize('as_composite', [True, False])
def test_clip_box_no_unused_points(as_composite):
    mesh = pv.Cube()
    mesh = pv.MultiBlock([mesh]) if as_composite else mesh
    new_bounds = (
        mesh.bounds.x_min,
        mesh.bounds.x_max,
        mesh.bounds.y_min,
        mesh.bounds.y_max,
        mesh.bounds.z_min + (mesh.bounds.z_max - mesh.bounds.z_min) * 7 / 10,
        mesh.bounds.z_min + (mesh.bounds.z_max - mesh.bounds.z_min) * 8 / 10,
    )
    clipped = mesh.clip_box(bounds=new_bounds, invert=False)
    assert np.allclose(clipped.bounds, new_bounds)


@pytest.mark.parametrize('invert', [True, False])
def test_clip_box_polydata_no_unused_points(invert):
    mesh = pv.Sphere(theta_resolution=16, phi_resolution=16)
    clipped = mesh.clip_box([0.1, 1.0, 0.1, 1.0, 0.1, 1.0], invert=invert)
    used = np.unique(clipped.cast_to_unstructured_grid().cell_connectivity)
    assert clipped.n_points == len(used)


@pytest.mark.parametrize('invert', [True, False])
def test_clip_box_polydata_keeps_cells_and_arrays(invert):
    """The clipped surface covers the same area and keeps its own cell types."""
    mesh = pv.Sphere(theta_resolution=16, phi_resolution=16)
    mesh.point_data['data'] = mesh.points[:, 2]
    mesh.cell_data['cells'] = np.arange(mesh.n_cells, dtype=float)
    bounds = [0.1, 1.0, 0.1, 1.0, 0.1, 1.0]

    clipped = mesh.clip_box(bounds, invert=invert)
    expected = _box_clip_filter(mesh, bounds, invert=invert).remove_unused_points()

    assert type(clipped) is pv.PolyData
    assert clipped.area == pytest.approx(expected.area)
    assert sorted(clipped.array_names) == sorted(expected.array_names)
    # The box planes keep the cells the box does not cut, which the box filter splits
    assert set(clipped.cast_to_unstructured_grid().celltypes) != {pv.CellType.TRIANGLE}


def test_clip_box_polydata_empty_is_polydata():
    mesh = pv.Sphere(theta_resolution=8, phi_resolution=8)

    clipped = mesh.clip_box([5.0, 6.0, 5.0, 6.0, 5.0, 6.0], invert=False)

    assert type(clipped) is pv.PolyData
    assert clipped.is_empty


def test_clip_box_polydata_empty_output_has_no_points():
    mesh = pv.Sphere(theta_resolution=16, phi_resolution=16)
    clipped = mesh.clip_box([0.3, 1.0, 0.3, 1.0, 0.3, 1.0], invert=False)
    assert clipped.n_cells == 0
    assert clipped.n_points == 0


def _box_clip_filter(mesh, bounds, *, invert):
    alg = _vtk.vtkBoxClipDataSet()
    alg.SetInputDataObject(mesh)
    alg.SetBoxClip(*bounds)
    if invert:
        alg.GenerateClippedOutputOn()
    alg.Update()
    return pv.wrap(alg.GetOutputDataObject(1 if invert else 0))


@pytest.mark.parametrize('invert', [True, False])
@pytest.mark.parametrize(
    ('cast', 'regular'),
    [
        pytest.param(lambda mesh: mesh, True, id='image'),
        pytest.param(lambda mesh: mesh.cast_to_rectilinear_grid(), True, id='rectilinear'),
        pytest.param(lambda mesh: mesh.cast_to_structured_grid(), True, id='structured'),
        pytest.param(lambda mesh: mesh.cast_to_unstructured_grid(), True, id='unstructured'),
        pytest.param(lambda _mesh: examples.load_explicit_structured(), False, id='explicit'),
    ],
)
def test_clip_box_planes_match_box_filter(uniform, invert, cast, regular):
    mesh = cast(uniform)
    lower, upper = np.array(mesh.bounds[::2]), np.array(mesh.bounds[1::2])
    lower = lower + 0.3 * (upper - lower)
    bounds = [lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]]
    clipped = mesh.clip_box(bounds, invert=invert)
    expected = _box_clip_filter(mesh, bounds, invert=invert)
    assert isinstance(clipped, pv.UnstructuredGrid)
    assert np.isclose(clipped.volume, expected.volume)
    # The box filter keeps unused input points, so compare with the box, not its bounds
    assert np.allclose(clipped.bounds, mesh.bounds if invert else bounds)

    # Whole cells are kept instead of being split into tetrahedra
    assert set(expected.celltypes) == {pv.CellType.TETRA}
    assert pv.CellType.HEXAHEDRON in set(clipped.celltypes)
    if regular:
        # Cells the box cuts stay hexahedral for a grid with axis-aligned cells
        assert set(clipped.celltypes) == {pv.CellType.HEXAHEDRON}
        assert clipped.n_cells < expected.n_cells
    else:
        # How VTK splits the cut cells of a curvilinear grid varies by version
        assert set(clipped.celltypes) <= {
            pv.CellType.HEXAHEDRON,
            pv.CellType.POLYHEDRON,
            pv.CellType.TETRA,
            pv.CellType.WEDGE,
            pv.CellType.PYRAMID,
        }


def test_clip_box_planes_invert_is_complement(uniform):
    bounds = [3.0, 9.0, 2.0, 9.0, 4.0, 9.0]
    inside = uniform.clip_box(bounds, invert=False)
    outside = uniform.clip_box(bounds, invert=True)
    assert np.isclose(inside.volume + outside.volume, uniform.volume)
    assert np.allclose(inside.bounds, bounds)


def test_clip_box_planes_oriented(uniform):
    box = pv.Cube(center=uniform.center, x_length=5, y_length=5, z_length=5).rotate_z(30)
    inside = uniform.clip_box(box, invert=False)
    outside = uniform.clip_box(box, invert=True)
    assert 0 < inside.volume < uniform.volume
    assert np.isclose(inside.volume + outside.volume, uniform.volume)


@pytest.mark.parametrize('invert', [True, False])
def test_clip_box_planes_crinkle(uniform, invert):
    bounds = [3.0, 9.0, 2.0, 9.0, 4.0, 9.0]
    crinkled = uniform.clip_box(bounds, invert=invert, crinkle=True)
    assert 'cell_ids' in crinkled.cell_data
    assert set(crinkled.celltypes) == {pv.CellType.VOXEL}
    expected = _box_clip_filter(uniform, bounds, invert=invert)
    assert crinkled.volume >= expected.volume


def test_clip_box_planes_box_outside_or_containing_mesh(uniform):
    bounds = np.array(uniform.bounds)
    larger = bounds + np.array([-1, 1] * 3)
    assert uniform.clip_box(larger, invert=False).n_cells == uniform.n_cells
    assert uniform.clip_box(larger, invert=True).n_cells == 0
    far = [bounds[1] + 1, bounds[1] + 2, 0, 1, 0, 1]
    assert uniform.clip_box(far, invert=False).n_cells == 0
    assert uniform.clip_box(far, invert=True).n_cells == uniform.n_cells


@pytest.mark.parametrize('invert', [True, False])
def test_clip_box_pointset(invert):
    points = pv.PointSet(np.random.default_rng(0).random((200, 3)))
    points.point_data['ids'] = np.arange(points.n_points)
    points.field_data['meta'] = [1.0]
    bounds = (0.0, 0.5, 0.0, 0.5, 0.0, 0.5)
    inside = np.all((points.points >= 0.0) & (points.points <= 0.5), axis=1)

    clipped = points.clip_box(bounds, invert=invert)

    assert isinstance(clipped, pv.PointSet)
    assert clipped.n_points == np.count_nonzero(~inside if invert else inside)
    assert np.array_equal(
        np.sort(clipped.point_data['ids']), np.flatnonzero(~inside if invert else inside)
    )
    assert np.allclose(clipped.field_data['meta'], [1.0])


@pytest.mark.parametrize('invert', [True, False])
def test_clip_box_merge_points_keeps_cell_types(uniform, invert):
    merged = uniform.clip_box(invert=invert)
    unmerged = uniform.clip_box(invert=invert, merge_points=False)
    assert set(merged.celltypes) == set(unmerged.celltypes)
    assert merged.n_cells == unmerged.n_cells
    assert merged.volume == pytest.approx(unmerged.volume)
    if invert:
        # Only an inverted clip appends pieces that share points
        assert merged.n_points < unmerged.n_points
    else:
        assert merged.n_points == unmerged.n_points


def test_clip_box_merge_points_false_keeps_coincident_points_apart():
    grid = pv.ImageData(dimensions=(5, 5, 5)).cast_to_unstructured_grid()
    centers = grid.cell_centers().points[:, 2]
    lower = grid.extract_cells(np.flatnonzero(centers < 2)).cast_to_unstructured_grid()
    upper = grid.extract_cells(np.flatnonzero(centers > 2)).cast_to_unstructured_grid()
    lower.cell_data['half'] = np.zeros(lower.n_cells)
    upper.cell_data['half'] = np.ones(upper.n_cells)
    # The halves meet at z == 2 but share no points, so the seam is a discontinuity
    seam = lower.merge(upper, merge_points=False)

    def shared_across_seam(mesh):
        half = np.asarray(mesh.cell_data['half'])
        halves_of_point = {}
        for i in range(mesh.n_cells):
            for point_id in mesh.get_cell(i).point_ids:
                halves_of_point.setdefault(point_id, set()).add(round(float(half[i])))
        return sum(1 for halves in halves_of_point.values() if len(halves) > 1)

    assert shared_across_seam(seam) == 0
    bounds = [2.5, 5.0, 2.5, 5.0, 2.5, 5.0]
    # An inverted clip appends the pieces outside the box, welding the seam unless asked not to
    assert shared_across_seam(seam.clip_box(bounds, invert=True)) > 0
    assert shared_across_seam(seam.clip_box(bounds, invert=True, merge_points=False)) == 0


def test_clip_box_composite(multiblock_all):
    # Now test composite data structures
    output = multiblock_all.clip_box(invert=False, progress_bar=True)
    assert output.n_blocks == multiblock_all.n_blocks


def test_clip_slab_axis_aligned():
    mesh = pv.ImageData(dimensions=(21, 21, 21), spacing=(0.1, 0.1, 0.1), origin=(-1, -1, -1))
    slab = mesh.clip_slab(thickness=0.4, normal='z', origin=(0, 0, 0))
    assert slab.n_cells > 0
    assert slab.bounds.z_min == pytest.approx(-0.2, abs=1e-6)
    assert slab.bounds.z_max == pytest.approx(0.2, abs=1e-6)
    # Untouched axes should span the original bounds
    assert slab.bounds.x_min == pytest.approx(mesh.bounds.x_min)
    assert slab.bounds.x_max == pytest.approx(mesh.bounds.x_max)


def test_clip_slab_default_origin_is_mesh_center():
    mesh = pv.ImageData(dimensions=(21, 21, 21), spacing=(0.1, 0.1, 0.1), origin=(-1, -1, -1))
    slab = mesh.clip_slab(thickness=0.4, normal='x')
    cx = mesh.center[0]
    assert slab.bounds.x_min == pytest.approx(cx - 0.2, abs=1e-6)
    assert slab.bounds.x_max == pytest.approx(cx + 0.2, abs=1e-6)


def test_clip_slab_string_normal():
    mesh = pv.ImageData(dimensions=(21, 21, 21), spacing=(0.1, 0.1, 0.1), origin=(-1, -1, -1))
    pos = mesh.clip_slab(thickness=0.4, normal='y', origin=(0, 0, 0))
    neg = mesh.clip_slab(thickness=0.4, normal='-y', origin=(0, 0, 0))
    assert pos.n_cells == neg.n_cells
    assert np.allclose(pos.bounds, neg.bounds)


def test_clip_slab_invert():
    mesh = pv.ImageData(dimensions=(21, 21, 21), spacing=(0.1, 0.1, 0.1), origin=(-1, -1, -1))
    slab = mesh.clip_slab(thickness=0.4, normal='z', origin=(0, 0, 0))
    outside = mesh.clip_slab(thickness=0.4, normal='z', origin=(0, 0, 0), invert=True)
    assert slab.n_cells > 0
    assert outside.n_cells > 0
    # Tolerance is loose enough to absorb VTK 9.2 floating-point ordering
    # differences in cell-center computation.
    tol = 1e-6
    slab_z = slab.cell_centers().points[:, 2]
    assert slab_z.min() >= -0.2 - tol
    assert slab_z.max() <= 0.2 + tol
    out_z = outside.cell_centers().points[:, 2]
    assert not np.any((out_z > -0.2 + tol) & (out_z < 0.2 - tol))


def test_clip_slab_oblique_normal():
    mesh = pv.ImageData(dimensions=(21, 21, 21), spacing=(0.1, 0.1, 0.1), origin=(-1, -1, -1))
    normal = np.array([1.0, 1.0, 1.0])
    unit = normal / np.linalg.norm(normal)
    slab = mesh.clip_slab(thickness=0.3, normal=normal, origin=(0, 0, 0))
    assert slab.n_cells > 0
    # Compute signed distance to the reference plane without matmul to avoid
    # spurious `divide by zero` warnings in certain BLAS builds when operating
    # on pyvista_ndarray views.
    points = np.asarray(slab.points)
    projections = (points * unit).sum(axis=1)
    assert projections.min() >= -0.15 - 1e-6
    assert projections.max() <= 0.15 + 1e-6


def test_clip_slab_plane_argument():
    mesh = pv.ImageData(dimensions=(21, 21, 21), spacing=(0.1, 0.1, 0.1), origin=(-1, -1, -1))
    plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1))
    slab = mesh.clip_slab(thickness=0.4, plane=plane)
    assert slab.bounds.z_min == pytest.approx(-0.2, abs=1e-6)
    assert slab.bounds.z_max == pytest.approx(0.2, abs=1e-6)


def test_clip_slab_polydata_preserves_type():
    sphere = pv.Sphere()
    slab = sphere.clip_slab(thickness=0.2, normal='y')
    assert isinstance(slab, pv.PolyData)
    assert slab.n_cells > 0


def test_clip_slab_composite(multiblock_all):
    output = multiblock_all.clip_slab(thickness=5.0, normal='x', progress_bar=True)
    assert isinstance(output, pv.MultiBlock)
    assert output.n_blocks == multiblock_all.n_blocks


def test_clip_slab_crinkle():
    mesh = pv.ImageData(dimensions=(21, 21, 21), spacing=(0.1, 0.1, 0.1), origin=(-1, -1, -1))
    slab = mesh.clip_slab(thickness=0.4, normal='z', origin=(0, 0, 0), crinkle=True)
    assert 'cell_ids' in slab.cell_data


@pytest.mark.parametrize('thickness', [0.0, -1.0])
def test_clip_slab_invalid_thickness(thickness):
    mesh = pv.Sphere()
    with pytest.raises(ValueError, match='strictly positive'):
        mesh.clip_slab(thickness=thickness, normal='x')


@pytest.mark.parametrize('normal', [(0, 0, 0), [0.0, 0.0, 0.0]])
@pytest.mark.parametrize(
    'method',
    [
        pv.DataObjectFilters.clip,
        pv.DataObjectFilters.slice,
        pv.DataObjectFilters.clip_slab,
        pv.PolyDataFilters.clip_closed_surface,
    ],
)
def test_zero_normal_raises(method, normal):
    kwargs = dict(thickness=1.0) if method is pv.DataObjectFilters.clip_slab else {}
    with pytest.raises(ValueError, match=re.escape('`normal` must be a non-zero vector.')):
        method(pv.Sphere(), normal=normal, **kwargs)


def test_clip_slab_zero_normal():
    mesh = pv.Sphere()
    with pytest.raises(ValueError, match='non-zero'):
        mesh.clip_slab(thickness=0.2, normal=(0.0, 0.0, 0.0))


def _two_clip_reference(mesh, normal, origin, thickness):
    """Reference implementation: two chained clips with opposing normals."""
    unit = np.asarray(normal, dtype=float)
    unit = unit / np.linalg.norm(unit)
    origin = np.asarray(origin, dtype=float)
    half = thickness / 2.0
    return mesh.clip(normal=unit, origin=origin + unit * half).clip(
        normal=-unit, origin=origin - unit * half
    )


@pytest.mark.parametrize(
    ('normal', 'thickness'),
    [
        ((0.0, 0.0, 1.0), 0.4),
        ((1.0, 1.0, 1.0), 0.3),
        ((2.0, -1.0, 0.5), 0.25),
    ],
)
def test_clip_slab_matches_two_clip_workaround(normal, thickness):
    """``clip_slab`` must match the historical two-clip chain geometrically.

    Only geometric equivalence (same region, same volume, same bounds) is
    asserted — not topological equivalence. The single-pass
    ``vtkImplicitBoolean`` clipper and the two-pass chain produce different
    cell decompositions on some VTK versions (notably 9.2), but both are valid
    discretizations of the same slab region.
    """
    mesh = pv.ImageData(dimensions=(21, 21, 21), spacing=(0.1, 0.1, 0.1), origin=(-1, -1, -1))
    origin = (0.0, 0.0, 0.0)
    new = mesh.clip_slab(thickness=thickness, normal=normal, origin=origin)
    old = _two_clip_reference(mesh, normal, origin, thickness)
    assert new.n_cells > 0
    assert old.n_cells > 0
    assert np.allclose(new.bounds, old.bounds, atol=1e-6)
    assert new.volume == pytest.approx(old.volume, rel=1e-6)


def test_clip_slab_matches_two_clip_workaround_polydata():
    """``clip_slab`` on surface data must match the two-clip chain geometrically."""
    mesh = pv.Sphere()
    new = mesh.clip_slab(thickness=0.2, normal='y', origin=(0, 0, 0))
    old = _two_clip_reference(mesh, (0, 1, 0), (0, 0, 0), 0.2)
    assert new.n_cells > 0
    assert old.n_cells > 0
    assert np.allclose(new.bounds, old.bounds, atol=1e-6)
    assert new.area == pytest.approx(old.area, rel=1e-6)


def _image_for_slicing():
    image = pv.ImageData(dimensions=(9, 7, 6), spacing=(1.0, 1.5, 2.0), origin=(1.0, 2.0, 3.0))
    image.point_data['ints'] = np.arange(image.n_points, dtype=np.int16) - 100
    image.point_data['floats'] = np.linspace(-1.0, 1.0, image.n_points) ** 3
    image.point_data['rgb'] = np.tile(np.arange(image.n_points)[:, None], (1, 3)).astype(np.uint8)
    image.cell_data['cell'] = np.arange(image.n_cells, dtype=np.int32)
    image.field_data['meta'] = ['x']
    image.set_active_scalars('floats')
    return image


def _cutter_slice(mesh, normal, origin):
    normal = _NORMALS[normal] if isinstance(normal, str) else normal
    return mesh.slice_implicit(generate_plane(normal, origin))


def _assert_slices_match(actual, expected):
    assert actual.n_points == expected.n_points
    assert actual.n_cells == expected.n_cells
    assert actual.array_names == expected.array_names
    assert actual.active_scalars_name == expected.active_scalars_name
    assert list(actual.field_data.keys()) == list(expected.field_data.keys())
    if actual.n_points == 0:
        return
    assert set(np.unique(actual.faces[::5])) == {4}
    order_actual = np.lexsort(np.asarray(actual.points).T[::-1])
    order_expected = np.lexsort(np.asarray(expected.points).T[::-1])
    assert np.allclose(actual.points[order_actual], expected.points[order_expected])
    for name in expected.point_data.keys():
        got = np.asarray(actual.point_data[name])[order_actual]
        want = np.asarray(expected.point_data[name])[order_expected]
        assert got.dtype == want.dtype
        # Integer values are truncated from a lerp that can differ in the last bit
        atol = 1 if got.dtype.kind in 'iu' else 1e-12
        assert np.allclose(got, want, atol=atol)
    for name in expected.cell_data.keys():
        assert np.array_equal(
            np.sort(actual.cell_data[name], axis=0), np.sort(expected.cell_data[name], axis=0)
        )


@pytest.mark.parametrize('normal', ['x', 'y', 'z', (-1, 0, 0), (0, -2, 0), (0, 0, -1)])
@pytest.mark.parametrize('fraction', [0.0, 0.2, 0.5, 0.731, 1.0])
def test_slice_image_axis_aligned_matches_cutter(normal, fraction):
    image = _image_for_slicing()
    axis = int(np.flatnonzero(_NORMALS[normal] if isinstance(normal, str) else normal)[0])
    bounds = image.bounds
    origin = list(image.center)
    origin[axis] = bounds[2 * axis] + fraction * (bounds[2 * axis + 1] - bounds[2 * axis])
    _assert_slices_match(image.slice(normal, origin=origin), _cutter_slice(image, normal, origin))


def test_slice_image_axis_aligned_on_grid_plane_and_offset():
    image = _image_for_slicing()
    image.offset = (3, 4, 5)
    for k in (0, 1, 4, 8):
        origin = [image.origin[0] + (image.offset[0] + k) * image.spacing[0], 0.0, 0.0]
        _assert_slices_match(image.slice('x', origin=origin), _cutter_slice(image, 'x', origin))


def test_slice_image_plane_outside_is_empty():
    image = _image_for_slicing()
    sliced = image.slice('x', origin=(image.bounds[1] + 1.0, 0.0, 0.0))
    assert sliced.n_points == 0
    assert sliced.array_names == image.array_names


@pytest.mark.parametrize(
    'kwargs',
    [
        dict(normal=(1, 1, 0)),
        dict(normal='x', generate_triangles=True),
        dict(normal='x', contour=True),
    ],
)
def test_slice_image_other_paths(kwargs):
    image = _image_for_slicing()
    sliced = image.slice(**kwargs)
    assert isinstance(sliced, pv.PolyData)
    assert sliced.n_cells


def test_slice_image_axis_aligned_keeps_active_cell_attributes():
    image = _image_for_slicing()
    n_cells = image.n_cells
    image.cell_data['cell_vectors'] = np.tile(np.arange(n_cells, dtype=float)[:, None], (1, 3))
    image.cell_data.active_vectors_name = 'cell_vectors'

    sliced = image.slice('x')

    assert sliced.cell_data.active_vectors_name == 'cell_vectors'


def test_slice_image_axis_aligned_keeps_active_attributes():
    image = _image_for_slicing()
    n_points = image.n_points
    image.point_data['vectors'] = np.tile(np.arange(n_points, dtype=float)[:, None], (1, 3))
    image.point_data['normals'] = np.tile([[0.0, 0.0, 1.0]], (n_points, 1))
    image.point_data['tcoords'] = np.tile(np.linspace(0, 1, n_points)[:, None], (1, 2))
    image.point_data['tensors'] = np.tile(np.arange(9, dtype=float), (n_points, 1))
    image.point_data.active_vectors_name = 'vectors'
    image.point_data.active_normals_name = 'normals'
    image.point_data.active_texture_coordinates_name = 'tcoords'
    image.GetPointData().SetActiveTensors('tensors')

    sliced = image.slice('x')
    assert sliced.point_data.active_vectors_name == 'vectors'
    assert sliced.point_data.active_normals_name == 'normals'
    assert sliced.point_data.active_texture_coordinates_name == 'tcoords'
    assert sliced.GetPointData().GetTensors().GetName() == 'tensors'
    assert sliced.point_data.active_scalars_name == 'floats'


def test_slice_image_rotated_uses_cutter():
    image = _image_for_slicing()
    image.direction_matrix = pv.Transform().rotate_z(30).matrix[:3, :3]
    sliced = image.slice('x')
    expected = _cutter_slice(image, 'x', image.center)
    assert sliced.n_points == expected.n_points
    assert np.allclose(np.sort(sliced.points, axis=0), np.sort(expected.points, axis=0))


def test_slice_image_orthogonal_and_along_axis():
    image = _image_for_slicing()
    orthogonal = image.slice_orthogonal()
    assert orthogonal.n_blocks == 3
    for block, axis in zip(orthogonal, (0, 1, 2), strict=True):
        assert np.allclose(block.points[:, axis], image.center[axis])
    along = image.slice_along_axis(n=4, axis='z')
    assert along.n_blocks == 4
    assert all(block.n_cells == 8 * 6 for block in along)


def test_slice_filter(datasets_no_pointset):
    """This tests the slice filter on all datatypes available filters"""
    for i, dataset in enumerate(datasets_no_pointset):
        slc = dataset.slice(normal=normals[i], progress_bar=True)
        assert slc is not None
        assert isinstance(slc, pv.PolyData)
    dataset = examples.load_uniform()
    slc = dataset.slice(contour=True, progress_bar=True)
    assert slc is not None
    assert isinstance(slc, pv.PolyData)
    result = dataset.slice(origin=(10, 15, 15), progress_bar=True)
    assert result.n_points < 1


def test_slice_filter_composite(multiblock_all_no_pointset):
    output = multiblock_all_no_pointset.slice(normal=normals[0], progress_bar=True)
    assert output.n_blocks == multiblock_all_no_pointset.n_blocks


def _output_type_meshes():
    """Return one mesh of every wrappable class, all spanning the same bounds."""
    axis = np.linspace(-1.0, 1.0, 5)
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    explicit = pv.StructuredGrid(x, y, z)
    explicit.dimensions = [5, 5, 5]
    image = pv.ImageData(dimensions=(5, 5, 5), spacing=(0.5, 0.5, 0.5), origin=(-1, -1, -1))
    return {
        'PolyData': pv.Sphere(radius=1.0, theta_resolution=8, phi_resolution=8),
        'PointSet': pv.PointSet(np.random.default_rng(0).uniform(-1, 1, (30, 3))),
        'ImageData': image,
        'RectilinearGrid': pv.RectilinearGrid(axis, axis, axis),
        'StructuredGrid': pv.StructuredGrid(x, y, z),
        'ExplicitStructuredGrid': explicit.cast_to_explicit_structured_grid(),
        'UnstructuredGrid': image.cast_to_unstructured_grid(),
    }


_OUTPUT_TYPE_MESHES = _output_type_meshes()


def _output_type_call(mesh, name):
    """Call one clip or slice filter with arguments valid for every input class."""
    kwargs = {
        'clip_slab': dict(normal='x', thickness=0.8),
        'slice_implicit': dict(implicit_function=generate_plane((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
        'slice_along_axis': dict(n=2),
        'slice_along_line': dict(line=pv.Line((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0), resolution=4)),
    }.get(name, {})
    return getattr(mesh, name)(**kwargs)


_GRID = pv.UnstructuredGrid
_POLY = pv.PolyData
_MULTI = pv.MultiBlock

_CLIP_LIKE = {
    'PolyData': _POLY,
    'PointSet': pv.PointSet,
    'ImageData': _GRID,
    'RectilinearGrid': _GRID,
    'StructuredGrid': _GRID,
    'ExplicitStructuredGrid': _GRID,
    'UnstructuredGrid': _GRID,
}
_BOX_LIKE = _CLIP_LIKE
_SLICE_LIKE = dict.fromkeys(_CLIP_LIKE, _POLY)
_SLICES_LIKE = dict.fromkeys(_CLIP_LIKE, _MULTI)

# The class each filter gives back for each input class, as the docstrings state it
OUTPUT_TYPES = {
    'clip': _CLIP_LIKE,
    'clip_slab': _CLIP_LIKE,
    'clip_box': _BOX_LIKE,
    'slice': _SLICE_LIKE,
    'slice_implicit': _SLICE_LIKE,
    'slice_along_line': _SLICE_LIKE,
    'slice_orthogonal': _SLICES_LIKE,
    'slice_along_axis': _SLICES_LIKE,
}


@pytest.mark.parametrize('name', list(OUTPUT_TYPES))
@pytest.mark.parametrize('mesh_type', list(_CLIP_LIKE))
def test_clip_slice_output_type(name, mesh_type):
    """Each filter gives back the class its docstring names, for every input class."""
    mesh = _OUTPUT_TYPE_MESHES[mesh_type].copy()

    if mesh_type == 'PointSet' and name.startswith('slice'):
        with pytest.raises(pv.PointSetDimensionReductionError):
            _output_type_call(mesh, name)
        return

    output = _output_type_call(mesh, name)

    expected = OUTPUT_TYPES[name][mesh_type]
    assert type(output) is expected
    if expected is _MULTI:
        assert all(type(block) is _POLY for block in output)


@pytest.mark.parametrize('name', list(OUTPUT_TYPES))
def test_clip_slice_output_type_composite(name):
    """A composite stays a composite, with every block following the same rule."""
    meshes = {key: mesh.copy() for key, mesh in _OUTPUT_TYPE_MESHES.items()}
    if name.startswith('slice'):
        meshes.pop('PointSet')
    flat, nested = list(meshes)[:3], list(meshes)[3:]
    composite = pv.MultiBlock(
        {
            'flat': pv.MultiBlock({key: meshes[key] for key in flat}),
            'nested': pv.MultiBlock({key: meshes[key] for key in nested}),
            'empty': None,
        }
    )

    output = _output_type_call(composite, name)

    assert type(output) is _MULTI
    assert output.keys() == composite.keys()
    assert output['empty'] is None
    for group, block_names in (('flat', flat), ('nested', nested)):
        assert output[group].keys() == block_names
        for mesh_type in block_names:
            block = output[group][mesh_type]
            expected = OUTPUT_TYPES[name][mesh_type]
            assert type(block) is expected
            if expected is _MULTI:
                assert all(type(sub) is _POLY for sub in block)


_SAME_CLASS = {name: getattr(pv, name) for name in _CLIP_LIKE}
_TO_POLY = dict.fromkeys(_CLIP_LIKE, _POLY)

# The class each filter gives back for each input class, as the docstrings state it
DATA_OBJECT_OUTPUT_TYPES = {
    'cell_centers': _TO_POLY,
    'extract_all_edges': _TO_POLY,
    'triangulate': {
        **dict.fromkeys(_CLIP_LIKE, _GRID),
        'PolyData': _POLY,
        'PointSet': pv.PointSet,
    },
    'elevation': _SAME_CLASS,
    'compute_cell_sizes': _SAME_CLASS,
    'cell_validator': _SAME_CLASS,
    'cell_data_to_point_data': _SAME_CLASS,
    'ctp': _SAME_CLASS,
    'point_data_to_cell_data': _SAME_CLASS,
    'ptc': _SAME_CLASS,
    'sample': _SAME_CLASS,
}

# The filters a bare `PointSet` rejects, since it has no cells
_POINTSET_REJECTS = frozenset(DATA_OBJECT_OUTPUT_TYPES) - {'cell_centers', 'elevation', 'sample'}


def _data_object_call(mesh, name):
    """Call one `DataObjectFilters` filter with arguments valid for every input class."""
    kwargs = {'sample': dict(target=_OUTPUT_TYPE_MESHES['ImageData'].copy())}.get(name, {})
    return getattr(mesh, name)(**kwargs)


@pytest.mark.parametrize('name', list(DATA_OBJECT_OUTPUT_TYPES))
@pytest.mark.parametrize('mesh_type', list(_CLIP_LIKE))
def test_data_object_filter_output_type(name, mesh_type):
    """Each filter gives back the class its docstring names, for every input class."""
    mesh = _OUTPUT_TYPE_MESHES[mesh_type].copy()

    if mesh_type == 'PointSet' and name in _POINTSET_REJECTS:
        with pytest.raises((PointSetCellOperationError, PointSetNotSupported)):
            _data_object_call(mesh, name)
        return

    output = _data_object_call(mesh, name)

    assert type(output) is DATA_OBJECT_OUTPUT_TYPES[name][mesh_type]


@pytest.mark.parametrize('name', list(DATA_OBJECT_OUTPUT_TYPES))
def test_data_object_filter_output_type_composite(name):
    """A composite stays a composite, with every block following the same rule."""
    meshes = {
        key: mesh.copy()
        for key, mesh in _OUTPUT_TYPE_MESHES.items()
        if not (key == 'PointSet' and name in _POINTSET_REJECTS)
    }
    expected_types = DATA_OBJECT_OUTPUT_TYPES[name]
    flat, nested = list(meshes)[:3], list(meshes)[3:]
    composite = pv.MultiBlock(
        {
            'flat': pv.MultiBlock({key: meshes[key] for key in flat}),
            'nested': pv.MultiBlock({key: meshes[key] for key in nested}),
            'empty': None,
        }
    )

    output = _data_object_call(composite, name)

    assert type(output) is _MULTI
    assert output.keys() == composite.keys()
    assert output['empty'] is None
    for group, block_names in (('flat', flat), ('nested', nested)):
        assert output[group].keys() == block_names
        for mesh_type in block_names:
            assert type(output[group][mesh_type]) is expected_types[mesh_type]


@pytest.mark.parametrize(
    'name',
    ['slice', 'slice_implicit', 'slice_along_line', 'slice_orthogonal', 'slice_along_axis'],
)
def test_slice_composite_pointset_block_raises(name):
    """A PointSet block raises the same error a bare PointSet does, naming the block."""
    points = pv.PointSet(np.random.default_rng(0).uniform(-1, 1, (30, 3)))
    with pytest.raises(pv.PointSetDimensionReductionError, match='type PointSet'):
        _output_type_call(pv.MultiBlock({'points': points}), name)


def test_slice_orthogonal_filter(datasets_no_pointset):
    """This tests the slice filter on all datatypes available filters"""
    for dataset in datasets_no_pointset:
        slices = dataset.slice_orthogonal(progress_bar=True)
        assert slices is not None
        assert isinstance(slices, pv.MultiBlock)
        assert slices.n_blocks == 3
        for slc in slices:
            assert isinstance(slc, pv.PolyData)


def test_slice_orthogonal_filter_composite(multiblock_all_no_pointset):
    output = multiblock_all_no_pointset.slice_orthogonal(progress_bar=True)
    assert output.n_blocks == multiblock_all_no_pointset.n_blocks


def test_slice_along_axis(datasets_no_pointset):
    """Test the many slices along axis filter"""
    axii = ['x', 'y', 'z', 'y', 0]
    ns = [2, 3, 4, 10, 20, 13]
    for i, dataset in enumerate(datasets_no_pointset):
        slices = dataset.slice_along_axis(n=ns[i], axis=axii[i], progress_bar=True)
        assert slices is not None
        assert isinstance(slices, pv.MultiBlock)
        assert slices.n_blocks == ns[i]
        for slc in slices:
            assert isinstance(slc, pv.PolyData)
    dataset = examples.load_uniform()
    with pytest.raises(ValueError):  # noqa: PT011
        dataset.slice_along_axis(axis='u')


def test_slice_along_axis_composite(multiblock_all_no_pointset):
    output = multiblock_all_no_pointset.slice_along_axis(progress_bar=True)
    assert output.n_blocks == multiblock_all_no_pointset.n_blocks


def test_extract_all_edges(datasets_no_pointset):
    for dataset in datasets_no_pointset:
        edges = dataset.extract_all_edges()
        assert edges is not None
        assert isinstance(edges, pv.PolyData)

    # Test that use_all_points parameter raises a deprecation warning
    with pytest.warns(PyVistaDeprecationWarning, match='use_all_points.*deprecated'):
        edges = datasets_no_pointset[0].extract_all_edges(use_all_points=True)
    assert edges.n_lines

    # Test that use_all_points=False also raises a deprecation warning
    with pytest.warns(PyVistaDeprecationWarning, match='use_all_points.*deprecated'):
        edges = datasets_no_pointset[0].extract_all_edges(use_all_points=False)
    assert edges.n_lines


def test_extract_all_edges_no_data():
    mesh = pv.Wavelet()
    edges = mesh.extract_all_edges(clear_data=True)
    assert edges is not None
    assert isinstance(edges, pv.PolyData)
    assert edges.n_arrays == 0


def test_extract_all_edges_composite(multiblock_all_no_pointset):
    # Now test composite data structures
    output = multiblock_all_no_pointset.extract_all_edges(progress_bar=True)
    assert output.n_blocks == multiblock_all_no_pointset.n_blocks


def test_cell_validator_composite(multiblock_all_no_pointset):
    output = multiblock_all_no_pointset.cell_validator()
    assert output.n_blocks == multiblock_all_no_pointset.n_blocks
    for block, source in zip(output, multiblock_all_no_pointset, strict=True):
        assert type(block) is type(source)
        assert block.active_scalars_name == 'validity_state'
        assert block.cell_data['validity_state'].shape == (source.n_cells,)
        assert block.field_data['invalid'].size == 0


def test_cell_validator_composite_pointset_raises(multiblock_all):
    with pytest.raises(pv.PointSetCellOperationError, match='type PointSet'):
        multiblock_all.cell_validator()


def test_extract_all_edges_composite_pointset_raises(multiblock_all):
    with pytest.raises(pv.PointSetCellOperationError, match='type PointSet'):
        multiblock_all.extract_all_edges(progress_bar=True)


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
    output = multiblock_all.elevation(progress_bar=True)
    assert output.n_blocks == multiblock_all.n_blocks
    assert [type(block) for block in output] == [type(block) for block in multiblock_all]


def test_compute_cell_sizes(datasets_no_pointset):
    for dataset in datasets_no_pointset:
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


@pytest.mark.parametrize(
    ('keyword', 'array_name'),
    [
        ('vertex_count', 'VertexCount'),
        ('length', 'Length'),
        ('area', 'Area'),
        ('volume', 'Volume'),
    ],
)
@pytest.mark.parametrize('empty', [True, False])
def test_compute_single_cell_sizes(datasets, keyword, array_name, empty):
    kwargs = {'vertex_count': False, 'length': False, 'area': False, 'volume': False}
    kwargs[keyword] = True

    for dataset in datasets:
        dataset_ = dataset.__class__() if empty else dataset
        dataset_.clear_data()
        if isinstance(dataset_, pv.PointSet):
            # PointSet has no cells, so cell sizes cannot be computed
            with pytest.raises(pv.PointSetCellOperationError):
                dataset_.compute_cell_sizes(**kwargs)
            continue
        result = dataset_.compute_cell_sizes(**kwargs)
        assert result.array_names == [array_name]


@pytest.mark.parametrize('empty', [True, False])
def test_compute_cell_sizes_multiblock_vertex_count(empty):
    content = [] if empty else [pv.PolyData()]
    multi = pv.MultiBlock(content)
    result = multi.compute_cell_sizes(vertex_count=True)
    if content:
        poly = result[0]
        assert 'Length' in poly.array_names
        assert 'Area' in poly.array_names
        assert 'Volume' in poly.array_names
        assert 'VertexCount' in poly.array_names


def test_compute_cell_sizes_composite(multiblock_all_no_pointset):
    # Now test composite data structures
    output = multiblock_all_no_pointset.compute_cell_sizes(progress_bar=True)
    assert output.n_blocks == multiblock_all_no_pointset.n_blocks


def test_compute_cell_sizes_composite_pointset_raises(multiblock_all):
    with pytest.raises(pv.PointSetCellOperationError, match='type PointSet'):
        multiblock_all.compute_cell_sizes(progress_bar=True)


def test_cell_centers(datasets):
    for dataset in datasets:
        result = dataset.cell_centers(progress_bar=True)
        assert result is not None
        assert isinstance(result, pv.PolyData)


def test_cell_centers_no_cell_data(cube):
    # test passing cell data kwarg works
    assert cube.cell_centers(pass_cell_data=True).cell_data
    assert not cube.cell_centers(pass_cell_data=False).cell_data


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


def test_cell_data_to_point_data_active_scalars_not_converted():
    # Older VTK declines to convert an id array, so the active name may not survive
    mesh = pv.Sphere(phi_resolution=8, theta_resolution=8)
    mesh.clear_data()
    ids = _vtk.vtkIdTypeArray()
    ids.SetName('RegionId')
    ids.SetNumberOfTuples(mesh.n_cells)
    ids.Fill(0)
    mesh.GetCellData().AddArray(ids)
    mesh.set_active_scalars('RegionId')

    converted = mesh.cell_data_to_point_data()
    assert converted.active_scalars_name in (None, *converted.array_names)


def test_cell_data_to_point_data_composite(multiblock_all_no_pointset):
    # Now test composite data structures
    output = multiblock_all_no_pointset.cell_data_to_point_data(progress_bar=True)
    assert output.n_blocks == multiblock_all_no_pointset.n_blocks


def test_cell_data_to_point_data_composite_pointset_raises(multiblock_all):
    with pytest.raises(pv.PointSetNotSupported, match='type PointSet'):
        multiblock_all.cell_data_to_point_data(progress_bar=True)


def test_point_data_to_cell_data():
    data = examples.load_uniform()
    foo = data.point_data_to_cell_data(progress_bar=True)
    assert foo.n_arrays == 2
    assert len(foo.point_data.keys()) == 0
    _ = data.ptc()


_STRING_CONVERSIONS = [
    ('point_data_to_cell_data', {}),
    ('point_data_to_cell_data', {'categorical': True}),
    ('cell_data_to_point_data', {}),
]
_STRING_CONVERSION_IDS = ['ptc', 'ptc-categorical', 'ctp']


def _conversion_associations(filter_name):
    """Return the source and target attribute names of a data conversion filter."""
    source_name, _, target_name = filter_name.partition('_to_')
    return source_name, target_name


@pytest.mark.parametrize('unstructured', [False, True])
@pytest.mark.parametrize('dtype', ['U', 'S'])
@pytest.mark.parametrize('pass_data', [False, True])
@pytest.mark.parametrize(
    ('filter_name', 'kwargs'), _STRING_CONVERSIONS, ids=_STRING_CONVERSION_IDS
)
def test_data_conversion_excludes_strings(filter_name, kwargs, pass_data, dtype, unstructured):
    mesh = pv.ImageData(dimensions=(3, 2, 2))
    if unstructured:
        mesh = mesh.cast_to_unstructured_grid()
    source_name, target_name = _conversion_associations(filter_name)
    sizes = {'point_data': mesh.n_points, 'cell_data': mesh.n_cells}
    source = getattr(mesh, source_name)
    source['labels'] = np.arange(sizes[source_name]).astype(dtype)
    source['values'] = np.arange(sizes[source_name], dtype=float)
    source['names'] = np.full(sizes[source_name], 'name', dtype=dtype)
    getattr(mesh, target_name)['existing'] = np.full(sizes[target_name], 42.0)
    mesh.field_data['description'] = ['metadata']
    kwargs = {**kwargs, f'pass_{source_name}': pass_data}

    # The conversion must match the same mesh without any string arrays
    numeric = mesh.copy()
    del getattr(numeric, source_name)['labels']
    del getattr(numeric, source_name)['names']
    expected = getattr(numeric, filter_name)(**kwargs)

    original = mesh.copy()
    match = 'excluded from the {}-to-{} data conversion'.format(
        source_name.removesuffix('_data'), target_name.removesuffix('_data')
    )
    with pytest.warns(UserWarning, match=match + re.escape(": ['labels', 'names']")):
        result = getattr(mesh, filter_name)(**kwargs)

    assert getattr(result, target_name) == getattr(expected, target_name)
    assert result.field_data == original.field_data
    if pass_data:
        assert getattr(result, source_name) == source
        assert np.shares_memory(getattr(result, source_name)['values'], source['values'])
    else:
        assert getattr(result, source_name) == getattr(expected, source_name)
    assert mesh == original


@pytest.mark.parametrize(
    ('filter_name', 'kwargs'), _STRING_CONVERSIONS, ids=_STRING_CONVERSION_IDS
)
def test_data_conversion_excludes_strings_only(filter_name, kwargs):
    mesh = pv.ImageData(dimensions=(3, 2, 2))
    source_name, target_name = _conversion_associations(filter_name)
    sizes = {'point_data': mesh.n_points, 'cell_data': mesh.n_cells}
    getattr(mesh, source_name)['labels'] = np.arange(sizes[source_name]).astype(str)
    original = mesh.copy()
    with pytest.warns(UserWarning, match=re.escape("conversion: ['labels']")):
        result = getattr(mesh, filter_name)(**kwargs, **{f'pass_{source_name}': True})
    assert getattr(result, source_name).keys() == ['labels']
    assert not getattr(result, target_name)
    assert mesh == original


def test_point_data_to_cell_data_excludes_unnamed_strings():
    mesh = pv.ImageData(dimensions=(3, 2, 2))
    mesh.point_data['values'] = np.arange(mesh.n_points, dtype=float)
    mesh.point_data['labels'] = np.arange(mesh.n_points).astype(str)
    mesh.point_data.VTKObject.GetAbstractArray('labels').SetName(None)
    with pytest.warns(UserWarning, match='String arrays cannot be converted'):
        result = mesh.point_data_to_cell_data()
    assert result.cell_data.keys() == ['values']


def test_point_data_to_cell_data_excludes_strings_composite():
    numeric = pv.ImageData(dimensions=(3, 2, 2))
    numeric.point_data['values'] = np.arange(numeric.n_points, dtype=float)
    strings = numeric.copy()
    strings.point_data['values'] = np.arange(strings.n_points).astype(str)
    mesh = pv.MultiBlock({'numeric': numeric, 'nested': pv.MultiBlock([strings, None])})
    original = mesh.copy()
    with pytest.warns(UserWarning, match=re.escape("conversion: ['values']")):
        result = mesh.point_data_to_cell_data()
    assert result['numeric'].cell_data['values'][0] == 5.0
    assert not result['nested'][0].cell_data
    assert result['nested'][1] is None
    assert mesh == original


def test_point_data_to_cell_data_composite(multiblock_all_no_pointset):
    # Now test composite data structures
    output = multiblock_all_no_pointset.point_data_to_cell_data(progress_bar=True)
    assert output.n_blocks == multiblock_all_no_pointset.n_blocks


def test_point_data_to_cell_data_composite_pointset_raises(multiblock_all):
    with pytest.raises(pv.PointSetNotSupported, match='type PointSet'):
        multiblock_all.point_data_to_cell_data(progress_bar=True)


def test_triangulate():
    data = examples.load_uniform()
    tri = data.triangulate(progress_bar=True)
    assert isinstance(tri, pv.UnstructuredGrid)
    assert np.any(tri.cells)


def test_triangulate_composite(multiblock_all_no_pointset):
    output = multiblock_all_no_pointset.triangulate(progress_bar=True)
    assert output.n_blocks == multiblock_all_no_pointset.n_blocks
    for block, source in zip(output, multiblock_all_no_pointset, strict=True):
        expected = pv.PolyData if isinstance(source, pv.PolyData) else pv.UnstructuredGrid
        assert type(block) is expected


def test_triangulate_composite_pointset_raises(multiblock_all):
    with pytest.raises(pv.PointSetCellOperationError, match='type PointSet'):
        multiblock_all.triangulate(progress_bar=True)


def test_sample():
    mesh = pv.Sphere(center=(4.5, 4.5, 4.5), radius=4.5)
    data_to_probe = examples.load_uniform()

    def sample_test(**kwargs):
        """Test ``sample`` with kwargs."""
        result = mesh.sample(data_to_probe, **kwargs)
        name = 'Spatial Point Data'
        assert name in result.array_names
        assert isinstance(result, type(mesh))

    sample_test()
    sample_test(tolerance=1.0)
    sample_test(progress_bar=True)
    sample_test(categorical=True)
    sample_test(locator=_vtk.vtkStaticCellLocator())
    with pytest.raises(ValueError):  # noqa: PT011
        sample_test(locator='invalid')
    sample_test(pass_cell_data=False)
    sample_test(pass_point_data=False)
    sample_test(pass_field_data=False)
    sample_test(snap_to_closest_point=True)


@pytest.mark.parametrize(
    'locator', ['cell', 'cell_tree', 'static_cell', _vtk.vtkStaticCellLocator]
)
def test_sample_locator(locator):
    # An unstructured target is required: image data is probed without a cell locator
    target = pv.Sphere(theta_resolution=10, phi_resolution=10).delaunay_3d()
    # A linear field interpolates to the same value from whichever cell is found
    target.point_data['x'] = target.points[:, 0]
    mesh = pv.Sphere(theta_resolution=8, phi_resolution=8, radius=0.4)

    result = mesh.sample(target, locator=locator() if callable(locator) else locator)

    assert result['vtkValidPointMask'].all()
    assert np.allclose(result['x'], mesh.points[:, 0])


@pytest.mark.needs_vtk_version(9, 7, 0)
@pytest.mark.parametrize('locator', ['obb_tree', _vtk.vtkOBBTree])
def test_sample_obb_tree_locator_raises(locator):
    locator = locator() if callable(locator) else locator
    target = pv.Sphere(theta_resolution=10, phi_resolution=10).delaunay_3d()
    target.point_data['pdata'] = np.arange(target.n_points, dtype=float)

    match = "The 'obb_tree' locator is deprecated"
    with pytest.raises(ValueError, match=match):
        pv.Sphere().sample(target, locator=locator)


@pytest.mark.needs_vtk_version(less_than=(9, 7, 0))
@pytest.mark.parametrize('locator', ['obb_tree', _vtk.vtkOBBTree])
def test_sample_obb_tree_locator_deprecated(locator):
    locator = locator() if callable(locator) else locator
    target = pv.Sphere(theta_resolution=10, phi_resolution=10).delaunay_3d()
    target.point_data['pdata'] = np.arange(target.n_points, dtype=float)

    match = "The 'obb_tree' locator is deprecated"
    with pytest.warns(pv.PyVistaDeprecationWarning, match=match):
        pv.Sphere().sample(target, locator=locator)


@pytest.fixture
def categorical_probe():
    """Image data overlapping the categorical target."""
    return pv.ImageData(dimensions=(5, 5, 5), spacing=(0.3, 0.3, 0.3), origin=(-0.6, -0.6, -0.6))


@pytest.fixture
def categorical_target():
    """Target whose active point scalars are spaced apart so interpolation is detectable."""
    target = pv.Sphere(theta_resolution=10, phi_resolution=10).delaunay_3d()
    target.clear_point_data()
    target.point_data['labels'] = (np.arange(target.n_points) % 3) * 10.0
    target.set_active_scalars('labels')
    return target


@pytest.mark.parametrize('categorical', [True, False])
def test_sample_categorical(categorical_probe, categorical_target, categorical):
    result = categorical_probe.sample(categorical_target, categorical=categorical)

    sampled = result['labels'][result['vtkValidPointMask'] == 1]
    assert sampled.size
    assert bool(np.isin(sampled, categorical_target['labels']).all()) is categorical


@pytest.mark.parametrize('composite', [pv.MultiBlock, pv.PartitionedDataSet])
@pytest.mark.parametrize('categorical', [True, False])
def test_sample_categorical_composite_target(
    categorical_probe, categorical_target, composite, categorical
):
    target = composite([categorical_target])

    result = categorical_probe.sample(target, categorical=categorical)

    sampled = result['labels'][result['vtkValidPointMask'] == 1]
    assert sampled.size
    assert bool(np.isin(sampled, categorical_target['labels']).all()) is categorical


@pytest.mark.parametrize(
    'kwargs',
    [
        {},
        {'mark_blank': False},
        {'pass_cell_data': False},
        {'pass_point_data': False},
        {'locator': 'cell'},
        {'tolerance': 1e-3},
    ],
)
def test_sample_composite_categorical_merge_matches_vtk(kwargs):
    # A constant per block interpolates to the same values either way, so the whole
    # output must match the composite probe VTK runs when `categorical` is off
    lower = pv.ImageData(dimensions=(6, 6, 6), spacing=(0.2,) * 3, origin=(-0.6,) * 3)
    upper = pv.ImageData(dimensions=(6, 6, 6), spacing=(0.2,) * 3, origin=(-0.1,) * 3)
    for index, block in enumerate((lower, upper)):
        block.point_data['labels'] = np.full(block.n_points, 100.0 * index)
    target = pv.MultiBlock([lower, upper])
    mesh = pv.PolyData(np.random.default_rng(0).random((200, 3)) * 1.4 - 0.7)

    expected = mesh.sample(target, **kwargs)
    merged = mesh.sample(target, categorical=True, **kwargs)

    assert sorted(merged.point_data.keys()) == sorted(expected.point_data.keys())
    assert sorted(merged.cell_data.keys()) == sorted(expected.cell_data.keys())
    assert merged['labels'].any()
    # Interpolating a constant is exact only to rounding, but the masks and ghosts are
    for name in expected.point_data:
        assert np.allclose(merged.point_data[name], expected.point_data[name]), name
    for name in expected.cell_data:
        assert np.array_equal(merged.cell_data[name], expected.cell_data[name]), name
    assert np.array_equal(merged['vtkValidPointMask'], expected['vtkValidPointMask'])


@pytest.fixture
def side_by_side_blocks():
    """Two blocks a probe of [5, 15, 25] hits in turn, with a constant `common` array."""
    lower = pv.ImageData(dimensions=(11, 11, 1), spacing=(1.0, 1.0, 1.0))
    upper = pv.ImageData(dimensions=(11, 11, 1), origin=(10.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    lower['common'] = np.zeros(lower.n_points)
    upper['common'] = np.ones(upper.n_points)
    for block in (lower, upper):
        block.set_active_scalars('common')
    return lower, upper


@pytest.fixture
def side_by_side_probe():
    """Probe points inside the lower block, inside the upper block, and outside both."""
    return pv.PolyData([[5.0, 5.0, 0.0], [15.0, 5.0, 0.0], [25.0, 5.0, 0.0]])


@pytest.mark.parametrize('on_upper', [True, False])
def test_sample_composite_categorical_drops_partial_arrays(
    side_by_side_blocks, side_by_side_probe, on_upper
):
    lower, upper = side_by_side_blocks
    block = upper if on_upper else lower
    block['partial'] = np.zeros(block.n_points)

    merged = side_by_side_probe.sample(pv.MultiBlock([lower, upper]), categorical=True)

    assert 'partial' not in merged.point_data
    assert np.array_equal(merged['common'], [0.0, 1.0, 0.0])
    assert np.array_equal(merged['vtkValidPointMask'], [1, 1, 0])


@pytest.mark.parametrize(
    ('lower_array', 'upper_array'),
    [
        (np.zeros((121, 3)), np.ones(121)),
        (np.full(121, 2.75), np.full(121, 7, dtype=np.int32)),
    ],
    ids=['components', 'dtype'],
)
def test_sample_composite_categorical_drops_mismatched_arrays(
    side_by_side_blocks, side_by_side_probe, lower_array, upper_array
):
    lower, upper = side_by_side_blocks
    lower['mismatched'] = lower_array
    upper['mismatched'] = upper_array
    target = pv.MultiBlock([lower, upper])

    merged = side_by_side_probe.sample(target, categorical=True)

    # VTK keeps an array only where every block agrees on its components and type
    assert sorted(merged.point_data.keys()) == sorted(
        side_by_side_probe.sample(target).point_data.keys()
    )
    assert 'mismatched' not in merged.point_data


def test_sample_empty_composite_categorical(categorical_probe):
    result = categorical_probe.sample(pv.MultiBlock(), categorical=True)

    assert result.n_points == categorical_probe.n_points
    assert not np.any(result['vtkValidPointMask'])


@pytest.mark.parametrize(
    'mesh',
    [pv.PolyData(), pv.PointSet(np.array([[0.0, 0.0, 0.0], [9.0, 9.0, 9.0]]))],
    ids=['empty', 'pointset'],
)
def test_sample_composite_categorical_without_ghost_arrays(mesh, categorical_target):
    target = pv.MultiBlock([categorical_target, categorical_target.copy()])

    result = mesh.sample(target, categorical=True)

    assert result.n_points == mesh.n_points
    assert 'labels' in result.point_data


@pytest.mark.parametrize('as_composite', [True, False])
def test_sample_categorical_no_point_scalars_raises(categorical_probe, as_composite):
    target = pv.Sphere(theta_resolution=10, phi_resolution=10).delaunay_3d()
    target.cell_data['labels'] = np.ones(target.n_cells)
    subject = "block 'Block-00'" if as_composite else 'target'
    target = pv.MultiBlock([target]) if as_composite else target

    match = f'Categorical sampling requires single-component point scalars on the {subject}'
    with pytest.raises(pv.MissingDataError, match=re.escape(match)):
        categorical_probe.sample(target, categorical=True)


def test_sample_categorical_string_scalars_raise(categorical_probe, categorical_target):
    categorical_target.clear_point_data()
    categorical_target.point_data['names'] = ['a'] * categorical_target.n_points

    match = 'Categorical sampling requires single-component point scalars on the target'
    with pytest.raises(pv.MissingDataError, match=match):
        categorical_probe.sample(categorical_target, categorical=True)


def test_sample_categorical_activates_the_only_candidate(categorical_probe, categorical_target):
    categorical_target.point_data.active_scalars_name = None

    result = categorical_probe.sample(categorical_target, categorical=True)

    sampled = result['labels'][result['vtkValidPointMask'] == 1]
    assert sampled.size
    assert np.isin(sampled, categorical_target['labels']).all()
    assert categorical_target.point_data.active_scalars_name is None


def test_sample_categorical_ambiguous_scalars_raises(categorical_probe, categorical_target):
    categorical_target.point_data['other'] = np.ones(categorical_target.n_points)
    categorical_target.point_data.active_scalars_name = None

    match = re.escape("Make one of ['labels', 'other'] active")
    with pytest.raises(pv.AmbiguousDataError, match=match):
        categorical_probe.sample(categorical_target, categorical=True)


def test_sample_non_dataset_target_raises(categorical_probe):
    match = 'Sampling target must be a dataset or a composite of datasets, got NoneType.'
    with pytest.raises(TypeError, match=re.escape(match)):
        categorical_probe.sample(None)


def test_sample_categorical_multi_component_raises(categorical_probe):
    target = pv.Sphere(theta_resolution=10, phi_resolution=10).delaunay_3d()
    target.point_data['vectors'] = np.ones((target.n_points, 3))
    target.point_data.active_scalars_name = 'vectors'

    match = "active point scalars 'vectors' have 3 components"
    with pytest.raises(ValueError, match=match):
        categorical_probe.sample(target, categorical=True)


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


def test_sample_composite_target():
    from pyvista import _vtk

    def _solid(center):
        mesh = pv.SolidSphere(outer_radius=0.4, center=center)
        mesh['height'] = mesh.points[:, 2]
        mesh.cell_data['cval'] = np.arange(mesh.n_cells, dtype=float)
        return mesh

    a, b = _solid((0.0, 0.0, 0.0)), _solid((0.7, 0.0, 0.0))
    grid = pv.ImageData(dimensions=(20, 20, 20), spacing=(0.09,) * 3, origin=(-0.8,) * 3)

    flat = grid.sample(pv.MultiBlock([a, b]))
    assert flat['vtkValidPointMask'].sum() > 0
    assert 'height' in flat.point_data
    assert 'cval' in flat.point_data

    # Nesting and empty blocks are handled by the composite probe
    nested = grid.sample(pv.MultiBlock([a, pv.MultiBlock([b])]))
    assert np.array_equal(nested['vtkValidPointMask'], flat['vtkValidPointMask'])

    with_none = grid.sample(pv.MultiBlock([a, None]))
    assert 0 < with_none['vtkValidPointMask'].sum() < flat['vtkValidPointMask'].sum()

    partitioned = grid.sample(pv.PartitionedDataSet([a, b]))
    assert np.array_equal(partitioned['vtkValidPointMask'], flat['vtkValidPointMask'])

    # Unwrapped composites are accepted too
    raw = _vtk.vtkMultiBlockDataSet()
    raw.SetNumberOfBlocks(2)
    raw.SetBlock(0, a)
    raw.SetBlock(1, b)
    assert np.array_equal(grid.sample(raw)['vtkValidPointMask'], flat['vtkValidPointMask'])

    raw_partitions = _vtk.vtkPartitionedDataSet()
    raw_partitions.SetNumberOfPartitions(2)
    raw_partitions.SetPartition(0, a)
    raw_partitions.SetPartition(1, b)
    assert np.array_equal(
        grid.sample(raw_partitions)['vtkValidPointMask'], flat['vtkValidPointMask']
    )


@pytest.mark.parametrize('as_composite', [True, False])
def test_slice_along_line_bad_line_raises(as_composite):
    mesh = pv.Sphere()
    mesh = pv.MultiBlock([mesh]) if as_composite else mesh
    with pytest.raises(ValueError, match='Input line must have only one cell'):
        mesh.slice_along_line(pv.Line() + pv.Line((1, 1, 1), (2, 2, 2)))
    with pytest.raises(TypeError, match='Input line must have a PolyLine cell'):
        mesh.slice_along_line(pv.PolyData([[0.0, 0.0, 0.0]]))


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


def test_slice_along_line_composite(multiblock_all_no_pointset):
    bounds = multiblock_all_no_pointset.bounds
    line = pv.Line(bounds[::2], bounds[1::2], resolution=10)
    output = multiblock_all_no_pointset.slice_along_line(line, progress_bar=True)
    assert output.n_blocks == multiblock_all_no_pointset.n_blocks


@pytest.mark.parametrize('generate_triangles', [True, False])
@pytest.mark.parametrize('name', ['slice', 'slice_implicit', 'slice_along_line'])
def test_slice_contour_of_an_empty_slice(name, generate_triangles):
    """A plane that cuts nothing has nothing to contour, arrays or not."""
    points = pv.PolyData(np.random.default_rng(0).uniform(-1, 1, (30, 3)))
    points.point_data['data'] = np.arange(points.n_points, dtype=float)

    sliced = _output_type_call_with(
        points, name, generate_triangles=generate_triangles, contour=True
    )

    assert type(sliced) is pv.PolyData
    assert sliced.is_empty


def _output_type_call_with(mesh, name, **kwargs):
    """Call one slice filter with arguments valid for every input class."""
    extra = {
        'slice_implicit': dict(implicit_function=generate_plane((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
        'slice_along_line': dict(line=pv.Line((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0), resolution=4)),
    }.get(name, {})
    return getattr(mesh, name)(**extra, **kwargs)


def test_slice_generate_triangles_true_emits_only_triangles():
    grid = examples.load_uniform().cast_to_unstructured_grid()
    out = grid.slice(normal=(1, 1, 1), generate_triangles=True)
    ug = out.cast_to_unstructured_grid()
    assert set(ug.celltypes.tolist()) <= {pv.CellType.TRIANGLE}


def test_slice_generate_triangles_false_default_preserves_polygons():
    """The ``False`` default keeps the historical polygon output for
    plane-cut UnstructuredGrids (no triangulation forced).
    """
    grid = examples.load_uniform().cast_to_unstructured_grid()
    out = grid.slice(normal=(1, 1, 1))  # default generate_triangles=False
    ug = out.cast_to_unstructured_grid()
    # Polygon path produces quads (and maybe other polygons) for hex
    # cell intersections, not just triangles.
    types = set(ug.celltypes.tolist())
    assert pv.CellType.QUAD in types or pv.CellType.POLYGON in types


def test_compute_cell_quality_removed():
    mesh = pv.ParametricEllipsoid().triangulate().decimate(0.8)
    with pytest.raises(AttributeError):
        _ = mesh.compute_cell_quality(progress_bar=True)


SHAPE = 'shape'
CELL_QUALITY = 'CellQuality'
AREA = 'area'
VOLUME = 'volume'


def test_cell_quality():
    mesh = pv.ParametricEllipsoid().triangulate().decimate(0.8)
    qual = mesh.cell_quality(SHAPE, progress_bar=True)
    assert SHAPE in qual.array_names

    expected_names = [SHAPE, AREA]
    qual = mesh.cell_quality(expected_names, progress_bar=True)
    assert qual.array_names == expected_names

    with pytest.raises(ValueError, match="quality_measure 'foo' is not valid"):
        mesh.cell_quality(quality_measure='foo', progress_bar=True)


def test_cell_quality_measures(ant):
    # Get quality measures from type hints
    hinted_measures = list(get_args(_CellQualityLiteral))

    # Get quality measures from the VTK class
    actual_measures = list(_get_cell_quality_measures().keys())
    msg = 'VTK API has changed. Update type hints and docstring for `cell_quality`.'
    assert actual_measures == hinted_measures, msg

    # Test 'all' measure keys
    qual = ant.cell_quality('all')
    assert qual.array_names == actual_measures


@pytest.mark.parametrize(
    'cell_mesh',
    [
        examples.cells.Triangle(),
        examples.cells.Quadrilateral(),
        examples.cells.Hexahedron(),
        examples.cells.Tetrahedron(),
    ],
)
@pytest.mark.parametrize('measure', ['relative_size_squared', 'shape_and_size'])
def test_cell_quality_size_measures(cell_mesh, measure):
    quality = cell_mesh.cell_quality(measure)
    assert np.isclose(quality[measure][0], 1.0)


def test_cell_quality_all_valid(ant):
    qual = ant.cell_quality('all_valid')
    assert AREA in qual.array_names
    assert SHAPE in qual.array_names
    assert VOLUME not in qual.array_names


@pytest.mark.parametrize(
    'mesh',
    [
        examples.cells.QuadraticTriangle(),
        examples.cells.QuadraticQuadrilateral(),
        examples.cells.QuadraticTetrahedron(),
        examples.cells.QuadraticHexahedron(),
    ],
)
def test_cell_quality_no_vtk_warnings(mesh):
    with pv.VtkErrorCatcher() as catcher:
        mesh.cell_quality('all_valid')
    assert catcher.warning_events == []


def test_cell_quality_composite(
    multiblock_all_with_nested_and_none, multiblock_all_no_pointset_with_nested_and_none
):
    match = "could not be applied to the block at index 5 with name 'Block-05' and type PointSet"
    with pytest.raises(pv.PointSetCellOperationError, match=match):
        qual = multiblock_all_with_nested_and_none.cell_quality([SHAPE])

    qual = multiblock_all_no_pointset_with_nested_and_none.cell_quality([SHAPE])
    for block in qual.recursive_iterator(skip_none=True):
        assert SHAPE in block.array_names


def test_cell_quality_return_type(multiblock_all_no_pointset_with_nested_and_none):
    iter_in = multiblock_all_no_pointset_with_nested_and_none.recursive_iterator()
    qual = multiblock_all_no_pointset_with_nested_and_none.cell_quality([SHAPE])
    iter_out = qual.recursive_iterator()
    for block_in, block_out in zip(iter_in, iter_out, strict=True):
        assert type(block_in) is type(block_out)


@pytest.mark.parametrize(
    ('num_cell_arrays', 'num_point_data'),
    itertools.product([0, 1, 2], [0, 1, 2]),
)
def test_transform_mesh(datasets, num_cell_arrays, num_point_data):
    dx, dy, dz = -1.0, 2.0, 3.0
    tf = pv.Transform().translate((dx, dy, dz))
    for dataset in datasets:
        for i in range(num_cell_arrays):
            if not isinstance(dataset, pv.PointSet):
                dataset.cell_data[f'C{i}'] = np.random.default_rng().random((dataset.n_cells, 3))

        for i in range(num_point_data):
            dataset.point_data[f'P{i}'] = np.random.default_rng().random((dataset.n_points, 3))

        # deactivate any active vectors!
        # even if transform_all_input_vectors is False, vtkTransformfilter will
        # transform active vectors
        dataset.set_active_vectors(None)

        transformed = dataset.transform(tf, transform_all_input_vectors=False, inplace=False)

        assert np.allclose(dataset.points[:, 0] + dx, transformed.points[:, 0])
        assert np.allclose(dataset.points[:, 1] + dy, transformed.points[:, 1])
        assert np.allclose(dataset.points[:, 2] + dz, transformed.points[:, 2])

        # ensure that none of the vector data is changed
        for name, array in dataset.point_data.items():
            assert transformed.point_data[name] == pytest.approx(array)

        for name, array in dataset.cell_data.items():
            assert transformed.cell_data[name] == pytest.approx(array)

        # verify that the cell connectivity is a deep copy. The public `connectivity`
        # property is read-only, so mutate the underlying vtkCellArray in place to
        # prove the two meshes do not share a buffer.
        cell_array = None
        if isinstance(dataset, pv.PolyData):
            cell_array = pv.core.pointset.PolyData.GetPolys
        elif isinstance(dataset, pv.UnstructuredGrid):
            cell_array = pv.core.pointset.UnstructuredGrid._get_cells
        if cell_array is not None:
            _get_connectivity_array(cell_array(transformed))[0] += 1
            assert not np.array_equal(
                _get_connectivity_array(cell_array(dataset)),
                _get_connectivity_array(cell_array(transformed)),
            )


def _matching_orders(actual_points, expected_points):
    """Return the orders which sort two matching point sets the same way."""
    return np.lexsort(np.round(actual_points, 8).T), np.lexsort(np.round(expected_points, 8).T)


@pytest.mark.parametrize(
    ('num_cell_arrays', 'num_point_data'),
    itertools.product([0, 1, 2], [0, 1, 2]),
)
def test_transform_mesh_and_vectors(datasets, num_cell_arrays, num_point_data):
    sx, sy, sz = -1.1, 2.2, 3.3
    tf = pv.Transform().scale((sx, sy, sz))
    for dataset in datasets:
        if not isinstance(dataset, pv.PointSet):
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

        scale = (sx, sy, sz)
        expected_points = dataset.points * scale
        # A rectilinear grid reverses the axes a negative scale would make descend, so
        # match the points of both meshes before comparing their arrays
        order, expected_order = _matching_orders(transformed.points, expected_points)
        if not isinstance(dataset, pv.RectilinearGrid):
            assert np.array_equal(order, expected_order)
        assert np.allclose(transformed.points[order], expected_points[expected_order])

        if not isinstance(dataset, pv.PointSet):
            cell_order, expected_cell_order = _matching_orders(
                transformed.cell_centers().points, dataset.cell_centers().points * scale
            )
            for i in range(num_cell_arrays):
                assert np.allclose(
                    transformed.cell_data[f'C{i}'][cell_order],
                    dataset.cell_data[f'C{i}'][expected_cell_order] * scale,
                )

        for i in range(num_point_data):
            assert np.allclose(
                transformed.point_data[f'P{i}'][order],
                dataset.point_data[f'P{i}'][expected_order] * scale,
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
def test_transform_int_vectors_warning(datasets_no_pointset, num_cell_arrays, num_point_data):
    tf = pv.Transform().scale((1, 2, 3))
    for dataset in datasets_no_pointset:
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


def test_transform_inplace(datasets):
    tf = pv.Transform().scale(1, 2, 3)
    for dataset in datasets:
        dataset.clear_data()
        pdata_array = np.arange(dataset.n_points)
        cdata_array = np.arange(dataset.n_cells)
        pdata_name = 'pdata'
        cdata_name = 'cdata'
        dataset[pdata_name] = pdata_array
        if not isinstance(dataset, pv.PointSet):
            dataset[cdata_name] = cdata_array

        copied = dataset.copy()
        inplace = copied.transform(tf, inplace=True)
        assert inplace is copied
        assert np.shares_memory(inplace[pdata_name], copied[pdata_name])
        if not isinstance(dataset, pv.PointSet):
            assert np.shares_memory(inplace[cdata_name], copied[cdata_name])

        not_inplace = dataset.transform(tf, inplace=False)
        assert inplace == not_inplace
        assert not np.shares_memory(not_inplace[pdata_name], copied[pdata_name])
        if not isinstance(dataset, pv.PointSet):
            assert not np.shares_memory(not_inplace[cdata_name], copied[cdata_name])


def test_transform_rectilinear_raises(rectilinear):
    tf = pv.Transform().rotate_x(30)
    match = (
        'The transformation has a rotation component which is not axis-aligned and is not\n'
        'supported by RectilinearGrid. Cast to StructuredGrid first to fully support '
        'rotations,\nor use `Transform.decompose()` to remove this component.'
    )

    with pytest.raises(ValueError, match=re.escape(match)):
        rectilinear.transform(tf, inplace=False)

    matrix = np.eye(4)
    matrix[0, 1] = 0.1
    matrix[1, 0] = 0.1
    match = (
        'The transformation has a shear component which is not supported by RectilinearGrid.\n'
        'Cast to StructuredGrid first to support shear transformations.'
    )

    with pytest.raises(ValueError, match=match):
        rectilinear.transform(matrix, inplace=False)


SHEAR_MATRIX = np.eye(4)
SHEAR_MATRIX[0, 1] = 0.1
SHEAR_MATRIX[1, 0] = 0.1


@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize(
    ('grid', 'transformation', 'match'),
    [
        (
            'rectilinear',
            pv.Transform().rotate_x(30),
            'rotation component which is not axis-aligned',
        ),
        ('rectilinear', SHEAR_MATRIX, 'shear component'),
        ('uniform', SHEAR_MATRIX, 'shear component'),
    ],
    ids=['rectilinear-rotation', 'rectilinear-shear', 'image-shear'],
)
def test_transform_raises_leaves_input_unchanged(grid, transformation, match, inplace, request):
    mesh = request.getfixturevalue(grid)
    mesh['a'] = np.arange(mesh.n_points, dtype=float)
    mesh['b'] = np.arange(mesh.n_points, dtype=float)
    mesh.set_active_scalars('a')
    before = mesh.copy()

    with pytest.raises(ValueError, match=match):
        mesh.transform(transformation, inplace=inplace)

    assert mesh.active_scalars_name == 'a'
    assert mesh == before


def test_transform_rectilinear(rectilinear):
    # Test that various transformations applied sequentially work

    def transform(mesh):
        return (
            mesh.flip_x()
            .flip_y()
            .flip_z()
            .rotate_x(360)
            .rotate(np.diag((-1, -1, -1)))
            .scale((1, 2, 3))
            .translate((4, 5, 6))
        )

    transform_then_cast = transform(rectilinear).cast_to_unstructured_grid()
    cast_then_transform = transform(rectilinear.cast_to_unstructured_grid())

    assert transform_then_cast == cast_then_transform


@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize(
    'transformation',
    [
        pv.Transform().rotate_z(90),
        pv.Transform().rotate_x(90).scale((2, 3, 4)).translate((5, -1, 2)),
        pv.Transform().rotate_y(-90),
        pv.Transform().rotate_z(180),
        pv.Transform().rotate_z(90).rotate_x(90),
    ],
    ids=[
        'rotate-z',
        'rotate-x-scale-translate',
        'rotate-y',
        'rotate-z-180',
        'rotate-z-then-x',
    ],
)
def test_transform_rectilinear_axis_aligned_rotation(rectilinear, transformation, inplace):
    rectilinear.point_data['p'] = np.arange(rectilinear.n_points, dtype=float)
    rectilinear.point_data['v'] = np.arange(rectilinear.n_points * 3, dtype=float).reshape(-1, 3)
    rectilinear.cell_data['c'] = np.arange(rectilinear.n_cells, dtype=float)
    expected = rectilinear.cast_to_structured_grid().transform(transformation, inplace=False)

    transformed = rectilinear.transform(transformation, inplace=inplace)

    assert isinstance(transformed, pv.RectilinearGrid)
    assert np.allclose(transformed.bounds, expected.bounds)
    # Points are ordered along the grid's own axes, so sort both before comparing
    actual_order = np.lexsort(np.round(transformed.points, 8).T)
    expected_order = np.lexsort(np.round(expected.points, 8).T)
    assert np.allclose(transformed.points[actual_order], expected.points[expected_order])
    assert np.array_equal(transformed['p'][actual_order], expected['p'][expected_order])
    assert np.array_equal(transformed['v'][actual_order], expected['v'][expected_order])
    actual_cells = np.lexsort(np.round(transformed.cell_centers().points, 8).T)
    expected_cells = np.lexsort(np.round(expected.cell_centers().points, 8).T)
    assert np.array_equal(
        transformed.cell_data['c'][actual_cells], expected.cell_data['c'][expected_cells]
    )


@pytest.mark.parametrize(
    'transformation',
    [pv.Transform().rotate_z(90), pv.Transform().scale((-1, 1, 1))],
    ids=['rotation', 'reflection'],
)
def test_transform_rectilinear_axes_ascend(rectilinear, transformation):
    transformed = rectilinear.transform(transformation, inplace=False)

    for coordinates in (transformed.x, transformed.y, transformed.z):
        assert np.all(np.diff(coordinates) > 0)
    # Descending coordinates are not supported by the cell locators
    probe = pv.PolyData(transformed.cell_centers().points)
    assert np.all(probe.sample(transformed)['vtkValidPointMask'] == 1)


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


def test_transform_imagedata_raises_with_shear(uniform):
    shear = np.eye(4)
    shear[0, 1] = 0.1

    match = (
        'The transformation has a shear component which is not supported by ImageData.\n'
        'Cast to StructuredGrid first to fully support shear transformations, or use\n'
        '`Transform.decompose()` to remove this component.'
    )

    with pytest.raises(ValueError, match=re.escape(match)):
        uniform.transform(shear, inplace=True)


def test_transform_filter_inplace_default_raises(cube):
    expected_msg = (
        'The default value of `inplace` for the filter `PolyData.transform` '
        'will change in the future.'
    )
    with pytest.raises(DeprecationError, match=expected_msg):
        _ = cube.transform(np.eye(4))


def test_reflect_mesh_about_point(datasets):
    for dataset in datasets:
        x_plane = 500
        reflected = dataset.reflect((1, 0, 0), point=(x_plane, 0, 0), progress_bar=True)
        assert reflected.n_cells == dataset.n_cells
        assert reflected.n_points == dataset.n_points
        expected_points = dataset.points * (-1, 1, 1) + (2 * x_plane, 0, 0)
        order, expected_order = _matching_orders(reflected.points, expected_points)
        if not isinstance(dataset, pv.RectilinearGrid):
            assert np.array_equal(order, expected_order)
        assert np.allclose(reflected.points[order], expected_points[expected_order])


def test_reflect_mesh_with_vectors(datasets):
    for dataset in datasets:
        if hasattr(dataset, 'compute_normals'):
            dataset.compute_normals(inplace=True, progress_bar=True)

        # add vector data to cell and point arrays
        if not isinstance(dataset, pv.PointSet):
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
        reflection = (-1, 1, 1)
        expected_points = dataset.points * reflection
        order, expected_order = _matching_orders(reflected.points, expected_points)
        if not isinstance(dataset, pv.RectilinearGrid):
            assert np.array_equal(order, expected_order)
        assert np.allclose(reflected.points[order], expected_points[expected_order])

        # assert vector fields and normals are reflected
        if not isinstance(dataset, pv.PointSet):
            cell_order, expected_cell_order = _matching_orders(
                reflected.cell_centers().points, dataset.cell_centers().points * reflection
            )
            if hasattr(dataset, 'compute_normals'):
                assert np.allclose(
                    reflected.cell_data['Normals'][cell_order],
                    dataset.cell_data['Normals'][expected_cell_order] * reflection,
                )
            assert np.allclose(
                reflected.cell_data['C'][cell_order],
                dataset.cell_data['C'][expected_cell_order] * reflection,
            )

        if hasattr(dataset, 'compute_normals'):
            assert np.allclose(
                reflected.point_data['Normals'][order],
                dataset.point_data['Normals'][expected_order] * reflection,
            )
        assert np.allclose(
            reflected.point_data['P'][order],
            dataset.point_data['P'][expected_order] * reflection,
        )


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


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(rotate_amounts=n_numbers(4), translate_amounts=n_numbers(3))
def test_transform_should_match_vtk_transformation(rotate_amounts, translate_amounts, hexbeam):
    trans = pv.Transform()
    trans.check_finite = False
    trans.RotateWXYZ(*rotate_amounts)
    trans.translate(translate_amounts)
    trans.Update()

    # Apply transform with pyvista filter
    grid_a = hexbeam.copy()
    grid_a.transform(trans, inplace=True)

    # Apply transform with vtk filter
    grid_b = hexbeam.copy()
    f = _vtk.vtkTransformFilter()
    f.SetInputDataObject(grid_b)
    f.SetTransform(trans)
    f.Update()
    grid_b = pv.wrap(f.GetOutput())

    # treat INF as NAN (necessary for allclose)
    grid_a.points[np.isinf(grid_a.points)] = np.nan
    assert np.allclose(grid_a.points, grid_b.points, equal_nan=True)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(rotate_amounts=n_numbers(4))
def test_transform_should_match_vtk_transformation_non_homogeneous(rotate_amounts, hexbeam):
    # test non homogeneous transform
    trans_rotate_only = pv.Transform()
    trans_rotate_only.check_finite = False
    trans_rotate_only.RotateWXYZ(*rotate_amounts)
    trans_rotate_only.Update()

    grid_copy = hexbeam.copy()
    grid_copy.transform(trans_rotate_only, inplace=True)

    from pyvista.core.utilities.transformations import apply_transformation_to_points

    trans_arr = trans_rotate_only.matrix[:3, :3]
    trans_pts = apply_transformation_to_points(trans_arr, hexbeam.points)
    assert np.allclose(grid_copy.points, trans_pts, equal_nan=True)


def test_translate_should_not_fail_given_none(hexbeam):
    bounds = hexbeam.bounds
    hexbeam.transform(None, inplace=True)
    assert hexbeam.bounds == bounds


def test_translate_should_fail_bad_points_or_transform():
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
def test_transform_should_fail_given_wrong_numpy_shape(array, hexbeam):
    assume(array.shape not in [(3, 3), (4, 4)])
    match = 'Shape must be one of [(3, 3), (4, 4)]'
    with pytest.raises(ValueError, match=re.escape(match)):
        hexbeam.transform(array, inplace=True)


@pytest.mark.parametrize('inplace', [True, False])
def test_translate_transform_all_input_vectors(datasets, inplace):
    """Translating honors ``transform_all_input_vectors`` whether or not it is in place."""
    for dataset in datasets:
        dataset.point_data['int_vectors'] = np.ones((dataset.n_points, 3), dtype=np.int64)

        match = 'have been converted to ``np.float32``'
        with pytest.warns(UserWarning, match=match):
            output = dataset.translate(
                (-1.0, 2.0, 3.0), transform_all_input_vectors=True, inplace=inplace
            )

        assert output.point_data['int_vectors'].dtype == np.float32
        assert (output is dataset) is inplace


@pytest.mark.parametrize('inplace', [True, False])
def test_translate_transform_all_input_vectors_false(datasets, inplace):
    """Inactive vector data is left alone when ``transform_all_input_vectors`` is off."""
    for dataset in datasets:
        dataset.point_data['int_vectors'] = np.ones((dataset.n_points, 3), dtype=np.int64)
        dataset.set_active_vectors(None)

        output = dataset.translate(
            (-1.0, 2.0, 3.0), transform_all_input_vectors=False, inplace=inplace
        )

        assert output.point_data['int_vectors'].dtype == np.int64


@pytest.mark.parametrize('axis_amounts', [[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
def test_translate_should_translate_grid(hexbeam, axis_amounts):
    grid_copy = hexbeam.copy()
    grid_copy.translate(axis_amounts, inplace=True)

    grid_points = hexbeam.points.copy() + np.array(axis_amounts)
    assert np.allclose(grid_copy.points, grid_points)


@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=HYPOTHESIS_MAX_EXAMPLES,
)
@given(angle=one_of(floats(allow_infinity=False, allow_nan=False), integers()))
@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
def test_rotate_should_match_vtk_rotation(angle, axis, hexbeam):
    trans = _vtk.vtkTransform()
    getattr(trans, f'Rotate{axis.upper()}')(angle)
    trans.Update()

    trans_filter = _vtk.vtkTransformFilter()
    trans_filter.SetTransform(trans)
    trans_filter.SetInputData(hexbeam)
    trans_filter.Update()
    grid_a = pv.UnstructuredGrid(trans_filter.GetOutput())

    grid_b = hexbeam.copy()
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
    poly = _vtk.vtkPolyData()
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

    with pytest.warns(UserWarning, match=r'Integer points.*converted.*float32'):
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
    poly = _vtk.vtkPolyData()
    poly.SetPoints(pv.vtk_points(points))

    # manually put together a rotate_x(10) transform
    trans_arr = pv.core.utilities.transformations.axis_angle_rotation((1, 0, 0), 10, deg=True)
    trans_mat = pv.vtkmatrix_from_array(trans_arr)
    trans = _vtk.vtkTransform()
    trans.SetMatrix(trans_mat)
    trans_filt = _vtk.vtkTransformFilter()
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


@pytest.mark.parametrize('bounds', [(-1, 1, -1, 1, -1, 1), (0, 10, -5, 5, 2, 8)])
@pytest.mark.parametrize('inplace', [True, False])
def test_resize_bounds(sphere, bounds, inplace):
    """Test resize method with bounds parameter."""
    resized = sphere.resize(bounds=bounds, inplace=inplace)

    assert np.allclose(resized.bounds, bounds, atol=1e-10)
    assert (sphere is resized) == inplace


def test_resize_bounds_preserve_aspect_ratio(sphere):
    """Test preserving aspect ratio when resizing bounds."""
    bounds = (-1, 1, -2, 2, -3, 3)

    resized = sphere.resize(
        bounds=bounds,
        preserve_aspect_ratio=True,
    )

    target_size = np.array([2, 4, 6])

    # Fits within requested bounds
    assert np.all(np.array(resized.bounds_size) <= target_size + 1e-10)

    # Aspect ratio preserved
    original_ratio = np.array(sphere.bounds_size) / sphere.length
    resized_ratio = np.array(resized.bounds_size) / resized.length
    assert np.allclose(original_ratio, resized_ratio)

    # Center should still match requested bounds center
    assert np.allclose(
        resized.center,
        (
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        ),
    )


@pytest.mark.parametrize('bounds_size', [2.0, (0.5, 2.5, 3.5)])
@pytest.mark.parametrize('center', [None, (0.0, 0.0, 0.0), (1.5, 2.5, 3.5)])
def test_resize_bounds_size(sphere, bounds_size, center):
    """Test resize method with bounds_size parameter."""
    expected_center = sphere.center if center is None else center

    resized = sphere.resize(bounds_size=bounds_size, center=center)
    new_size = resized.bounds_size
    assert np.allclose(new_size, bounds_size)
    assert np.allclose(resized.center, expected_center)


def test_resize_bounds_size_preserve_aspect_ratio(sphere):
    """Test preserving aspect ratio when resizing bounds_size."""
    target_size = (1.0, 2.0, 3.0)

    resized = sphere.resize(
        bounds_size=target_size,
        preserve_aspect_ratio=True,
    )

    # Fits within requested size
    assert np.all(np.array(resized.bounds_size) <= np.array(target_size) + 1e-10)

    # Aspect ratio preserved
    original_ratio = np.array(sphere.bounds_size) / sphere.length
    resized_ratio = np.array(resized.bounds_size) / resized.length
    assert np.allclose(original_ratio, resized_ratio)


@pytest.mark.parametrize('length', [42, 5.0])
@pytest.mark.parametrize('center', [None, (0.0, 0.0, 0.0), (1.5, 2.5, 3.5)])
def test_resize_length(sphere, length, center):
    """Test resize method with length parameter."""
    expected_center = sphere.center if center is None else center

    resized = sphere.resize(length=length, center=center)
    assert np.isclose(resized.length, length)
    assert np.allclose(resized.center, expected_center)

    # Default preserve_aspect_ratio=None should preserve aspect ratio for length
    original_ratio = np.array(sphere.bounds_size) / sphere.length
    resized_ratio = np.array(resized.bounds_size) / resized.length
    assert np.allclose(original_ratio, resized_ratio)


@pytest.mark.parametrize('mesh', [pv.MultiBlock(), pv.PolyData()])
def test_resize_empty(mesh):
    resized = mesh.resize()
    assert resized.is_empty
    assert isinstance(resized, type(mesh))
    assert resized is not mesh


def test_resize_raises(sphere):
    """Test resize method error handling."""

    match = (
        'Cannot specify more than one resizing method. '
        'Choose either `bounds`, `bounds_size`, or `length` independently.'
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        sphere.resize(bounds=[-1, 1, -1, 1, -1, 1], bounds_size=2.0)
    with pytest.raises(ValueError, match=re.escape(match)):
        sphere.resize(length=5, bounds_size=2.0)
    with pytest.raises(ValueError, match=re.escape(match)):
        sphere.resize(bounds=[-1, 1, -1, 1, -1, 1], length=5)

    match = '`bounds`, `bounds_size`, and `length` cannot all be None. Choose one resizing method.'
    with pytest.raises(ValueError, match=re.escape(match)):
        sphere.resize()

    match = '`center` can only be used with the `bounds_size` and `length` parameters.'
    with pytest.raises(ValueError, match=re.escape(match)):
        sphere.resize(bounds=[-1, 1, -1, 1, -1, 1], center=(0, 0, 0))

    match = '{name} values must all be greater than 0.0.'
    with pytest.raises(ValueError, match=match.format(name='length')):
        sphere.resize(length=0)
    with pytest.raises(ValueError, match=match.format(name='bounds_size')):
        sphere.resize(bounds_size=[-1, 2, 3])

    match = (
        '`preserve_aspect_ratio=False` cannot be used with `length` since '
        '`length` resizing always preserves the aspect ratio.'
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        sphere.resize(length=5, preserve_aspect_ratio=False)


def test_resize_zero_extent(plane):
    # This should not fail even with zero Z extent
    target_bounds = [-1, 1, -1, 1, -1, 1]
    resized = plane.resize(bounds=target_bounds)

    # X and Y should be resized, Z should remain at the target Z center
    expected_z_center = (target_bounds[4] + target_bounds[5]) / 2
    assert np.allclose(resized.points[:, 2], expected_z_center)


def test_resize_multiblock():
    sphere = pv.Sphere(center=(1, 2, 3))
    cube = pv.Cube(center=(-1, -2, -3))
    multi = pv.MultiBlock({'sphere': sphere, 'cube': cube})

    new_size = (7, 8, 9)
    resized = multi.resize(bounds_size=new_size)
    assert np.allclose(resized.bounds_size, new_size)
    # Test that blocks were not resized individually, but were
    # instead resized as part of the whole
    assert not np.allclose(resized['sphere'].bounds_size, new_size)
    assert not np.allclose(resized['cube'].bounds_size, new_size)


def _add_vtk_array(dataset, name, values, association: Literal['point', 'cell']):
    arr = _vtk.vtkFloatArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(1)

    for v in values:
        arr.InsertNextValue(float(v))

    if association == 'point':
        dataset.GetPointData().AddArray(arr)
    else:  # association == "cell":
        dataset.GetCellData().AddArray(arr)


def _add_invalid_arrays(mesh):
    # Invalid point arrays (multiple), need more than 4 to test truncated repr
    _add_vtk_array(mesh, 'foo', range(10), association='point')
    _add_vtk_array(mesh, 'bar', range(15), association='point')
    _add_vtk_array(mesh, 'baz', range(12), association='point')
    _add_vtk_array(mesh, 'qux', range(13), association='point')
    _add_vtk_array(mesh, 'fred', range(14), association='point')
    _add_vtk_array(mesh, 'waldo', range(16), association='point')
    _add_vtk_array(mesh, 'thud', range(17), association='point')

    # Invalid cell array (single)
    _add_vtk_array(mesh, 'ham', range(11), association='cell')


@pytest.fixture
def sphere_with_invalid_arrays(sphere):
    _add_invalid_arrays(sphere)
    return sphere


@pytest.fixture
def grid_with_invalid_arrays(hexbeam):
    _add_invalid_arrays(hexbeam)
    return hexbeam


@pytest.mark.parametrize('as_composite', [True, False])
def test_validate_mesh_is_valid(sphere_with_invalid_arrays, as_composite):
    mesh = pv.PolyData()
    mesh = mesh.cast_to_multiblock() if as_composite else mesh
    report = mesh.validate_mesh()
    assert report.is_valid
    assert isinstance(report.mesh, pv.MultiBlock if as_composite else pv.PolyData)
    assert report.mesh is not mesh
    output_polydata = report.mesh[0] if as_composite else report.mesh
    assert 'validity_state' in output_polydata.array_names

    invalid_mesh = (
        pv.MultiBlock([sphere_with_invalid_arrays]) if as_composite else sphere_with_invalid_arrays
    )
    assert not invalid_mesh.validate_mesh().is_valid


def test_validate_mesh_default_fields():
    mesh = pv.UnstructuredGrid()
    report1 = str(mesh.validate_mesh())
    report2 = str(mesh.validate_mesh(['data', 'cells', 'points']))
    assert report1 == report2


def test_validate_mesh_exclude_fields():
    mesh = pv.PolyData()
    exclude = str(mesh.validate_mesh(exclude_fields='cells'))
    include = str(mesh.validate_mesh(['data', 'points']))
    assert exclude == include

    exclude = str(mesh.validate_mesh(exclude_fields=['cells', 'points', 'cell_data_wrong_length']))
    include = str(mesh.validate_mesh('point_data_wrong_length'))
    assert exclude == include

    match = "Excluded field 'cells' must be a subset of the validation fields."
    with pytest.raises(ValueError, match=match):
        mesh.validate_mesh('points', exclude_fields='cells')

    match = "Excluded field 'points' must be a subset of the validation fields."
    with pytest.raises(ValueError, match=match):
        mesh.validate_mesh('cells', exclude_fields='points')

    match = "Excluded field 'data' must be a subset of the validation fields."
    with pytest.raises(ValueError, match=match):
        mesh.validate_mesh('points', exclude_fields='data')

    match = "Excluded field 'negative_size' must be a subset of the validation fields."
    with pytest.raises(ValueError, match=match):
        mesh.validate_mesh('points', exclude_fields='negative_size')


def test_validate_mesh_exclude_fields_subset(invalid_tetra_negative_volume):
    NEGATIVE_SIZE = 'negative_size'
    mesh = invalid_tetra_negative_volume
    assert not mesh.validate_mesh().is_valid
    assert not mesh.validate_mesh(NEGATIVE_SIZE).is_valid
    assert not mesh.validate_mesh('cells').is_valid
    assert mesh.validate_mesh('cells', exclude_fields=NEGATIVE_SIZE).is_valid

    # Test all cell fields are included except for the excluded one
    report = str(mesh.validate_mesh('cells', exclude_fields=NEGATIVE_SIZE, report_body='fields'))
    expected = (
        'Invalid cell ids:\n'
        '    Coincident points        : []\n'
        '    Degenerate faces         : []\n'
        '    Intersecting edges       : []\n'
        '    Intersecting faces       : []\n'
        '    Invalid point references : []\n'
        '    Inverted faces           : []\n'
        '    Non-contiguous edges     : []\n'
        '    Non-convex               : []\n'
        '    Non-planar faces         : []\n'
        '    Wrong number of points   : []\n'
        '    Zero size                : []'
    )
    assert expected in report
    assert 'Negative size' not in report

    fields = ['data', 'points', NEGATIVE_SIZE]
    report = str(mesh.validate_mesh(fields, exclude_fields=fields, report_body='fields'))
    expected = (
        'Mesh Validation Report\n'
        '----------------------\n'
        'Mesh info:\n'
        '    Type                     : UnstructuredGrid\n'
        '    N Points                 : 4\n'
        '    N Cells                  : 1\n'
        '    Cell types               : {TETRA}\n'
        'Report summary:\n'
        '    Is valid                 : True\n'
        '    Invalid fields           : ()'
    )
    assert report == expected


def test_validate_mesh_cell_status():
    mesh = pv.PolyData()
    report_enum = mesh.validate_mesh(pv.CellStatus.DEGENERATE_FACES)
    report_str = mesh.validate_mesh('degenerate_faces')
    assert str(report_enum) == str(report_str)

    report_enum = mesh.validate_mesh(['unused_points', pv.CellStatus.DEGENERATE_FACES])
    report_str = mesh.validate_mesh(['unused_points', 'degenerate_faces'])
    assert str(report_enum) == str(report_str)


def test_validate_mesh_message(sphere_with_invalid_arrays):
    assert pv.PolyData().validate_mesh().message is None
    assert sphere_with_invalid_arrays.validate_mesh().message


def test_validate_mesh_point_arrays(sphere_with_invalid_arrays):
    # Dataset had invalid point AND cell arrays, but we validate point arrays only
    report = sphere_with_invalid_arrays.validate_mesh(['point_data_wrong_length'])
    assert report.point_data_wrong_length == ['foo', 'bar', 'baz', 'qux', 'fred', 'waldo', 'thud']
    assert report.cell_data_wrong_length is None

    # Clear cell arrays and validate ALL arrays
    sphere_with_invalid_arrays.cell_data.clear()
    report = sphere_with_invalid_arrays.validate_mesh('data')
    assert report.point_data_wrong_length == ['foo', 'bar', 'baz', 'qux', 'fred', 'waldo', 'thud']
    assert report.cell_data_wrong_length == []

    match = (
        'PolyData mesh is not valid:\n'
        ' - Mesh has 7 point arrays with incorrect length (length must be 422). '
        "Invalid arrays: 'foo' (10), 'bar' (15), 'baz' (12), 'qux' (13), ..."
    )
    with pytest.warns(pv.InvalidMeshWarning, match=re.escape(match)):
        report = sphere_with_invalid_arrays.validate_mesh(action='warn')
    assert report.message == match


def test_validate_mesh_cell_arrays(sphere_with_invalid_arrays):
    # Dataset had invalid point AND cell arrays, but we validate cell arrays only
    report = sphere_with_invalid_arrays.validate_mesh('cell_data_wrong_length')
    assert report.cell_data_wrong_length == ['ham']
    assert report.point_data_wrong_length is None

    # Clear point arrays and validate ALL arrays
    sphere_with_invalid_arrays.point_data.clear()
    report = sphere_with_invalid_arrays.validate_mesh('data')
    assert report.cell_data_wrong_length == ['ham']
    assert report.point_data_wrong_length == []

    match = (
        'PolyData mesh is not valid:\n'
        ' - Mesh has 1 cell array with incorrect length (length must be 840). '
        "Invalid array: 'ham' (11)"
    )
    with pytest.warns(pv.InvalidMeshWarning, match=re.escape(match)):
        report = sphere_with_invalid_arrays.validate_mesh(action='warn')
    assert report.message == match


def test_validate_mesh_raises(sphere_with_invalid_arrays):
    match = (
        'PolyData mesh is not valid:\n'
        ' - Mesh has 7 point arrays with incorrect length (length must be 422).'
        " Invalid arrays: 'foo' (10), 'bar' (15), 'baz' (12), 'qux' (13), ...\n"
        ' - Mesh has 1 cell array with incorrect length (length must be 840). '
        "Invalid array: 'ham' (11)"
    )
    with pytest.raises(pv.InvalidMeshError, match=re.escape(match)):
        sphere_with_invalid_arrays.validate_mesh(action='error')


@pytest.fixture
def image150():
    # Dims must be large enough to replicate VTK bug: https://gitlab.kitware.com/vtk/vtk/-/work_items/20096
    return pv.ImageData(dimensions=(150, 150, 150))


def test_validate_mesh_imagedata(image150):
    image150.cell_validator()
    image150.validate_mesh(action='error')

    match = "Cell field 'wrong_number_of_points' is not supported for ImageData."
    with pytest.raises(ValueError, match=match):
        image150.validate_mesh('cells')


@pytest.mark.parametrize('field', [f.lower() for f in _VTK_CELL_STATUS_INFO if f != 'VALID'])
def test_validate_mesh_imagedata_vtk_fields(image150, field):
    # Ensure vtk fields raise error
    match = f'Cell field {field!r} is not supported for ImageData.'
    with pytest.raises(ValueError, match=match):
        image150.validate_mesh([field])


@pytest.mark.parametrize('field', [f.lower() for f in _PYVISTA_CELL_STATUS_INFO if f != 'VALID'])
def test_validate_mesh_imagedata_pyvista_fields(image150, field):
    # Ensure pyvista fields DO NOT raise error
    image150.validate_mesh(field)


@pytest.mark.needs_vtk_version(less_than=(9, 6, 0))
def test_validate_mesh_planarity_tolerance():
    match = 'Planarity tolerance requires VTK 9.6 or later.'
    with pytest.raises(pv.VTKVersionError, match=match):
        pv.UnstructuredGrid().validate_mesh(planarity_tolerance=0.2)


@pytest.mark.needs_vtk_version(9, 6, 0)
def test_validate_mesh_planarity_tolerance_polyhedron():
    # Build a hex-shaped polyhedron whose top face is non-planar (one vertex
    # pushed up out of the plane). With a strict planarity tolerance the
    # mesh is flagged invalid; with a loose tolerance it is accepted.
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 2.0],  # pushed up out of the top face plane
        [0.0, 1.0, 1.0],
    ]
    faces = [
        [4, 0, 1, 2, 3],  # bottom
        [4, 4, 5, 6, 7],  # top (non-planar)
        [4, 0, 1, 5, 4],
        [4, 1, 2, 6, 5],
        [4, 2, 3, 7, 6],
        [4, 3, 0, 4, 7],
    ]
    polyhedron_connectivity = [len(faces), *[item for face in faces for item in face]]
    cells = [len(polyhedron_connectivity), *polyhedron_connectivity]
    mesh = pv.UnstructuredGrid(cells, [pv.CellType.POLYHEDRON], points)

    # Strict tolerance flags non-planarity
    report_strict = mesh.validate_mesh(planarity_tolerance=0.001)
    assert not report_strict.is_valid
    assert 'non-planar' in str(report_strict.message).lower()

    # Loose tolerance accepts the mesh's planarity (no NON_PLANAR_FACES status)
    report_loose = mesh.validate_mesh(planarity_tolerance=10.0)
    assert 'non-planar' not in str(report_loose.message or '').lower()


@pytest.fixture
def invalid_random_polydata():
    n = 20
    rng = np.random.default_rng(seed=103)
    points = rng.random(n * 3).reshape(-1, 3)

    faces = [[0, 1, n + 1]]
    faces = np.column_stack(
        (
            np.ones(
                len(faces),
            )
            * 3,
            faces,
        )
    ).astype(int)
    points = np.append(points, [[np.nan, 0, 0]], axis=0)
    return pv.PolyData(points, faces=faces)


def test_validate_mesh_name():
    name = 'poly'
    report = pv.PolyData().validate_mesh(name=name)
    assert report.name == name


def test_validate_mesh_report_str():
    report = pv.Sphere().validate_mesh(report_body='fields', name='Sphere')
    actual = str(report)
    expected = (
        'Mesh Validation Report\n'
        '----------------------\n'
        'Mesh info:\n'
        "    Name                     : 'Sphere'\n"
        '    Type                     : PolyData\n'
        '    N Points                 : 842\n'
        '    N Cells                  : 1680\n'
        '    Cell types               : {TRIANGLE}\n'
        'Report summary:\n'
        '    Is valid                 : True\n'
        '    Invalid fields           : ()\n'
        'Invalid data arrays:\n'
        '    Cell data wrong length   : []\n'
        '    Point data wrong length  : []\n'
        'Invalid point ids:\n'
        '    Non-finite points        : []\n'
        '    Unused points            : []\n'
        'Invalid cell ids:\n'
        '    Coincident points        : []\n'
        '    Degenerate faces         : []\n'
        '    Intersecting edges       : []\n'
        '    Intersecting faces       : []\n'
        '    Invalid point references : []\n'
        '    Inverted faces           : []\n'
        '    Negative size            : []\n'
        '    Non-contiguous edges     : []\n'
        '    Non-convex               : []\n'
        '    Non-planar faces         : []\n'
        '    Wrong number of points   : []\n'
        '    Zero size                : []'
    )
    assert actual == expected


def test_validate_mesh_composite_report_str():
    multi = pv.Sphere().cast_to_multiblock()
    report = multi.validate_mesh(report_body='fields', name='Sphere')
    actual = str(report)
    expected = (
        'Mesh Validation Report\n'
        '----------------------\n'
        'Mesh info:\n'
        "    Name                     : 'Sphere'\n"
        '    Type                     : MultiBlock\n'
        '    N Blocks                 : 1\n'
        'Report summary:\n'
        '    Is valid                 : True\n'
        '    Invalid fields           : ()\n'
        'Blocks with invalid data arrays:\n'
        '    Cell data wrong length   : []\n'
        '    Point data wrong length  : []\n'
        'Blocks with invalid points:\n'
        '    Non-finite points        : []\n'
        '    Unused points            : []\n'
        'Blocks with invalid cells:\n'
        '    Coincident points        : []\n'
        '    Degenerate faces         : []\n'
        '    Intersecting edges       : []\n'
        '    Intersecting faces       : []\n'
        '    Invalid point references : []\n'
        '    Inverted faces           : []\n'
        '    Negative size            : []\n'
        '    Non-contiguous edges     : []\n'
        '    Non-convex               : []\n'
        '    Non-planar faces         : []\n'
        '    Wrong number of points   : []\n'
        '    Zero size                : []'
    )
    assert actual == expected


def test_validate_mesh_str_invalid_mesh(invalid_random_polydata):
    report = invalid_random_polydata.validate_mesh(
        exclude_fields=['negative_size', 'zero_size'], report_body='fields'
    )
    actual = str(report)
    expected = (
        'Mesh Validation Report\n'
        '----------------------\n'
        'Mesh info:\n'
        '    Type                         : PolyData\n'
        '    N Points                     : 21\n'
        '    N Cells                      : 1\n'
        '    Cell types                   : {TRIANGLE}\n'
        'Report summary:\n'
        '    Is valid                     : False\n'
        "    Invalid fields (3)           : ('non_finite_points', 'unused_points', "
        "'invalid_point_references')\n"
        'Invalid data arrays:\n'
        '    Cell data wrong length       : []\n'
        '    Point data wrong length      : []\n'
        'Invalid point ids:\n'
        '    Non-finite points (1)        : [20]\n'
        '    Unused points (19)           : [2, 3, 4, 5, 6, 7, ...]\n'
        'Invalid cell ids:\n'
        '    Coincident points            : []\n'
        '    Degenerate faces             : []\n'
        '    Intersecting edges           : []\n'
        '    Intersecting faces           : []\n'
        '    Invalid point references (1) : [0]\n'
        '    Inverted faces               : []\n'
        '    Non-contiguous edges         : []\n'
        '    Non-convex                   : []\n'
        '    Non-planar faces             : []\n'
        '    Wrong number of points       : []'
    )
    assert actual == expected


@pytest.fixture
def invalid_nested_multiblock(invalid_random_polydata):
    return pv.MultiBlock(
        dict(
            none=None,
            poly_root=invalid_random_polydata,
            nested=pv.MultiBlock(dict(poly_nested=invalid_random_polydata.copy())),
            nested_valid=pv.MultiBlock(dict(valid=pv.PolyData())),
        )
    )


def test_validate_mesh_composite_str_invalid_mesh(invalid_nested_multiblock):
    report = invalid_nested_multiblock.validate_mesh(
        exclude_fields=['negative_size', 'zero_size'], report_body='fields'
    )
    actual = str(report)
    expected = (
        'Mesh Validation Report\n'
        '----------------------\n'
        'Mesh info:\n'
        '    Type                         : MultiBlock\n'
        '    N Blocks                     : 4\n'
        'Report summary:\n'
        '    Is valid                     : False\n'
        "    Invalid fields (3)           : ('non_finite_points', 'unused_points', "
        "'invalid_point_references')\n"
        'Blocks with invalid data arrays:\n'
        '    Cell data wrong length       : []\n'
        '    Point data wrong length      : []\n'
        'Blocks with invalid points:\n'
        '    Non-finite points (2)        : [1, 2]\n'
        '    Unused points (2)            : [1, 2]\n'
        'Blocks with invalid cells:\n'
        '    Coincident points            : []\n'
        '    Degenerate faces             : []\n'
        '    Intersecting edges           : []\n'
        '    Intersecting faces           : []\n'
        '    Invalid point references (2) : [1, 2]\n'
        '    Inverted faces               : []\n'
        '    Non-contiguous edges         : []\n'
        '    Non-convex                   : []\n'
        '    Non-planar faces             : []\n'
        '    Wrong number of points       : []'
    )
    assert actual == expected


def test_validate_mesh_composite_message(invalid_nested_multiblock):
    multi = invalid_nested_multiblock
    report = multi.validate_mesh(exclude_fields=['negative_size', 'zero_size'])
    actual = report.message
    expected = (
        'MultiBlock mesh is not valid:\n'
        " * Block id 1 'poly_root' PolyData mesh is not valid:\n"
        '   - Mesh has 19 unused points not referenced by any cell. Invalid point '
        'ids: [2, 3, 4, 5, 6, 7, ...]\n'
        '   - Mesh has 1 non-finite point. Invalid point id: [20]\n'
        '   - Mesh has 1 TRIANGLE cell with invalid point references. Invalid cell '
        'id: [0]\n'
        " * Block id 2 'nested' MultiBlock mesh is not valid:\n"
        "   * Block id 0 'poly_nested' PolyData mesh is not valid:\n"
        '     - Mesh has 19 unused points not referenced by any cell. Invalid '
        'point ids: [2, 3, 4, 5, 6, 7, ...]\n'
        '     - Mesh has 1 non-finite point. Invalid point id: [20]\n'
        '     - Mesh has 1 TRIANGLE cell with invalid point references. Invalid cell '
        'id: [0]'
    )
    assert actual == expected


def test_validate_mesh_composite_subreports(invalid_nested_multiblock):
    report = invalid_nested_multiblock.validate_mesh()
    # Test subreports
    assert len(report) == len(invalid_nested_multiblock)
    assert report[0] is None

    index = 1
    expected_subreport = str(invalid_nested_multiblock[index].validate_mesh())
    actual_subreport = str(report[index])
    assert actual_subreport == expected_subreport

    assert isinstance(report, Sized)
    for subreport in report:
        assert isinstance(subreport, (type(report), type(None)))

    match = 'Indexing mesh validation reports is only supported for composite meshes.'
    poly_subreport = report[1]
    assert isinstance(poly_subreport.mesh, pv.PolyData)
    with pytest.raises(TypeError, match=match):
        poly_subreport[0]

    multi_subreport = report[2]
    assert isinstance(multi_subreport.mesh, pv.MultiBlock)
    poly_subreport = multi_subreport[0]
    assert isinstance(poly_subreport.mesh, pv.PolyData)
    with pytest.raises(TypeError, match=match):
        poly_subreport[0]

    match = 'Length of mesh validation report is only defined for composite meshes.'
    with pytest.raises(TypeError, match=match):
        len(poly_subreport)


def test_validate_mesh_str_filtered():
    report = pv.PolyData().validate_mesh(['data', 'unused_points'], report_body='fields')
    actual = str(report)
    expected = (
        'Mesh Validation Report\n'
        '----------------------\n'
        'Mesh info:\n'
        '    Type                     : PolyData\n'
        '    N Points                 : 0\n'
        '    N Cells                  : 0\n'
        '    Cell types               : set()\n'
        'Report summary:\n'
        '    Is valid                 : True\n'
        '    Invalid fields           : ()\n'
        'Invalid data arrays:\n'
        '    Cell data wrong length   : []\n'
        '    Point data wrong length  : []\n'
        'Invalid point ids:\n'
        '    Unused points            : []'
    )
    assert actual == expected

    report = pv.PolyData().validate_mesh(['memory_safe'], report_body='fields')
    actual = str(report)
    expected = (
        'Mesh Validation Report\n'
        '----------------------\n'
        'Mesh info:\n'
        '    Type                     : PolyData\n'
        '    N Points                 : 0\n'
        '    N Cells                  : 0\n'
        '    Cell types               : set()\n'
        'Report summary:\n'
        '    Is valid                 : True\n'
        '    Invalid fields           : ()\n'
        'Invalid data arrays:\n'
        '    Cell data wrong length   : []\n'
        '    Point data wrong length  : []\n'
        'Invalid cell ids:\n'
        '    Invalid point references : []'
    )
    assert actual == expected


@pytest.mark.parametrize('fields', ['cells', 'non_convex', 'unused_points'])
def test_validate_mesh_composite_pointset_block(ant, fields):
    # By default the fields a PointSet cannot have are skipped for that block alone
    multi = pv.MultiBlock({'ant': ant, 'points': ant.cast_to_pointset()})
    report = multi.validate_mesh()
    assert report.is_valid
    assert str(report.mesh['points'].validate_mesh()) == str(
        ant.cast_to_pointset().validate_mesh()
    )

    # Asking for them explicitly raises, as it does for a bare PointSet
    match = f'field {fields!r} is not supported for PointSet' if fields != 'cells' else 'PointSet'
    with pytest.raises(ValueError, match=match):
        ant.cast_to_pointset().validate_mesh(fields)
    with pytest.raises(ValueError, match=match):
        multi.validate_mesh(fields)


def test_validate_mesh_composite_grid_block(uniform):
    multi = pv.MultiBlock({'sphere': pv.Sphere(), 'image': uniform})
    assert multi.validate_mesh().is_valid
    match = "Cell field 'non_convex' is not supported for ImageData"
    with pytest.raises(ValueError, match=match):
        uniform.validate_mesh('non_convex')
    with pytest.raises(ValueError, match=match):
        multi.validate_mesh('non_convex')


def test_validate_mesh_pointset(ant):
    pset = ant.cast_to_pointset()
    report = pset.validate_mesh(report_body='fields')
    actual = str(report)
    expected = (
        'Mesh Validation Report\n'
        '----------------------\n'
        'Mesh info:\n'
        '    Type                     : PointSet\n'
        '    N Points                 : 486\n'
        '    N Cells                  : 0\n'
        '    Cell types               : set()\n'
        'Report summary:\n'
        '    Is valid                 : True\n'
        '    Invalid fields           : ()\n'
        'Invalid data arrays:\n'
        '    Cell data wrong length   : []\n'
        '    Point data wrong length  : []\n'
        'Invalid point ids:\n'
        '    Non-finite points        : []'
    )
    assert actual == expected

    report = pset.validate_mesh('data', report_body='fields')
    actual = str(report)
    expected = (
        'Mesh Validation Report\n'
        '----------------------\n'
        'Mesh info:\n'
        '    Type                     : PointSet\n'
        '    N Points                 : 486\n'
        '    N Cells                  : 0\n'
        '    Cell types               : set()\n'
        'Report summary:\n'
        '    Is valid                 : True\n'
        '    Invalid fields           : ()\n'
        'Invalid data arrays:\n'
        '    Cell data wrong length   : []\n'
        '    Point data wrong length  : []'
    )
    assert actual == expected


def test_validate_mesh_report_body(invalid_tetra_negative_volume):
    report = pv.PolyData().validate_mesh(report_body='message')
    expected = (
        'Mesh Validation Report\n'
        '----------------------\n'
        'Mesh info:\n'
        '    Type           : PolyData\n'
        '    N Points       : 0\n'
        '    N Cells        : 0\n'
        '    Cell types     : set()\n'
        'Report summary:\n'
        '    Is valid       : True\n'
        '    Invalid fields : ()'
    )
    actual = str(report)
    assert actual == expected

    report = invalid_tetra_negative_volume.validate_mesh(report_body='message')
    expected = (
        'Mesh Validation Report\n'
        '----------------------\n'
        'Mesh info:\n'
        '    Type               : UnstructuredGrid\n'
        '    N Points           : 4\n'
        '    N Cells            : 1\n'
        '    Cell types         : {TETRA}\n'
        'Report summary:\n'
        '    Is valid           : False\n'
        "    Invalid fields (1) : ('negative_size',)\n"
        'Error message:\n'
        '    UnstructuredGrid mesh is not valid:\n'
        '     - Mesh has 1 TETRA cell with negative volume. Invalid cell id: [0]'
    )
    actual = str(report)
    assert actual == expected


def test_cell_validator_pointset_raises():
    match = 'Cell operations are not supported'
    with pytest.raises(pv.PointSetCellOperationError, match=match):
        pv.PointSet().cell_validator()


def test_cell_validator():
    sphere = pv.Sphere()
    sphere.cell_data['data'] = range(sphere.n_cells)
    validated = sphere.cell_validator()
    assert validated.active_scalars_name == 'validity_state'
    assert isinstance(validated, pv.PolyData)
    assert validated.field_data.keys() == ['invalid', *CELL_STATUS_ARRAY_NAMES]
    assert validated.array_names == [
        'validity_state',
        'invalid',
        *CELL_STATUS_ARRAY_NAMES,
        'Normals',
        'data',
    ]
    for name in CELL_STATUS_ARRAY_NAMES:
        array = validated.field_data[name]
        assert array.shape == (0,)


@pytest.mark.needs_vtk_version(9, 6, 0)
@pytest.mark.skip_vtk_backend('cvista', reason=CELL_STATUS_ENUM)
def test_cell_status():
    expected_pyvista_values = list(pv.CellStatus)
    expected_vtk_values = list(vars(_vtk.vtkCellStatus).values())

    # Map VTK enum members PyVista enum members
    VTK_TO_CELL_STATUS = {
        _vtk.vtkCellStatus.Valid: pv.CellStatus.VALID,
        _vtk.vtkCellStatus.WrongNumberOfPoints: pv.CellStatus.WRONG_NUMBER_OF_POINTS,
        _vtk.vtkCellStatus.IntersectingEdges: pv.CellStatus.INTERSECTING_EDGES,
        _vtk.vtkCellStatus.IntersectingFaces: pv.CellStatus.INTERSECTING_FACES,
        _vtk.vtkCellStatus.NoncontiguousEdges: pv.CellStatus.NON_CONTIGUOUS_EDGES,
        _vtk.vtkCellStatus.Nonconvex: pv.CellStatus.NON_CONVEX,
        _vtk.vtkCellStatus.FacesAreOrientedIncorrectly: pv.CellStatus.INVERTED_FACES,
        _vtk.vtkCellStatus.NonPlanarFaces: pv.CellStatus.NON_PLANAR_FACES,
        _vtk.vtkCellStatus.DegenerateFaces: pv.CellStatus.DEGENERATE_FACES,
        _vtk.vtkCellStatus.CoincidentPoints: pv.CellStatus.COINCIDENT_POINTS,
    }

    for vtk_val, pyvista_val in VTK_TO_CELL_STATUS.items():
        assert vtk_val == pyvista_val

        assert vtk_val in expected_vtk_values
        assert pyvista_val in expected_pyvista_values

        expected_vtk_values.remove(vtk_val)
        expected_pyvista_values.remove(pyvista_val)

    # Ensure all values are accounted for and we're not missing any
    assert expected_vtk_values == []
    # There should only be pyvista-only status values
    pyvista_specific_values = [info.value for info in _PYVISTA_CELL_STATUS_INFO.values()]
    assert expected_pyvista_values == pyvista_specific_values


@pytest.fixture
def invalid_tetra_missing_point():
    # Define tetra with one point missing
    cells = [3, 0, 1, 2]
    celltypes = [pv.CellType.TETRA]
    points = [
        [-1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
    ]
    return pv.UnstructuredGrid(cells, celltypes, points)


@pytest.fixture
def invalid_tetra_negative_volume():
    # Regular tetra but with first two points swapped
    cells = [4, 0, 1, 2, 3]
    celltypes = [pv.CellType.TETRA]
    points = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ]
    return pv.UnstructuredGrid(cells, celltypes, points)


@pytest.mark.parametrize('as_composite', [True, False])
def test_cell_validator_invalid_tetra(
    invalid_tetra_missing_point, invalid_tetra_negative_volume, as_composite
):
    # Use vtkAppend instead of pv.merge for consistent ordering
    # since pyvista merge order changed in VTK 9.5.
    append = _vtk.vtkAppendFilter()
    append.AddInputData(invalid_tetra_missing_point)
    append.AddInputData(invalid_tetra_negative_volume)
    append.Update()

    invalid_input = pv.wrap(append.GetOutput())
    mesh = invalid_input.cast_to_multiblock() if as_composite else invalid_input
    validated = mesh.cell_validator()
    assert type(validated) is type(mesh)
    single_mesh = validated[0] if as_composite else validated
    for name in CELL_STATUS_ARRAY_NAMES:
        if name in (
            pv.CellStatus.WRONG_NUMBER_OF_POINTS.name.lower(),
            pv.CellStatus.ZERO_SIZE.name.lower(),
            pv.CellStatus.COINCIDENT_POINTS.name.lower(),
        ):
            expected_cell_ids = [0]
            assert single_mesh[name].tolist() == expected_cell_ids
        elif name == pv.CellStatus.NEGATIVE_SIZE.name.lower():
            expected_cell_ids = [1]
            assert single_mesh[name].tolist() == expected_cell_ids
        else:
            array = single_mesh.field_data[name]
            assert array.shape == (0,)


def test_validate_mesh_negative_volume(invalid_tetra_negative_volume):
    message = invalid_tetra_negative_volume.validate_mesh().message
    expected = (
        'UnstructuredGrid mesh is not valid:\n'
        ' - Mesh has 1 TETRA cell with negative volume. Invalid cell id: [0]'
    )
    assert message == expected


def test_validate_mesh_degenerate_cells():
    def append_mixed_cells(dataset):
        # Use append, not pv.merge, due to change in merge order in vtk 9.5
        valid_tetra = examples.cells.Tetrahedron().translate((2, 2, 2))
        valid_vertex = examples.cells.Vertex().translate((-2, -2, -2))
        append = _vtk.vtkAppendFilter()
        append.AddInputData(dataset)
        append.AddInputData(valid_tetra)
        append.AddInputData(valid_vertex)
        append.Update()
        return pv.wrap(append.GetOutput())

    # Line with coincident points
    invalid_mesh = pv.Line((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    for mesh in [invalid_mesh, append_mixed_cells(invalid_mesh)]:
        state = mesh.cell_validator()['validity_state']
        assert state[0] & pv.CellStatus.ZERO_SIZE
        assert state[0] & pv.CellStatus.COINCIDENT_POINTS
    match = 'Mesh has 1 LINE cell with zero length. Invalid cell id: [0]'
    for mesh in [invalid_mesh, append_mixed_cells(invalid_mesh)]:
        with pytest.raises(pv.InvalidMeshError, match=re.escape(match)):
            mesh.validate_mesh(action='error')

    # Degenerate triangle
    points = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    invalid_mesh = pv.PolyData(points, faces=[3, 0, 1, 2])
    for mesh in [invalid_mesh, append_mixed_cells(invalid_mesh)]:
        state = mesh.cell_validator()['validity_state']
        assert state[0] & pv.CellStatus.ZERO_SIZE
        assert state[0] & pv.CellStatus.COINCIDENT_POINTS
    match = 'Mesh has 1 TRIANGLE cell with zero area. Invalid cell id: [0]'
    for mesh in [invalid_mesh, append_mixed_cells(invalid_mesh)]:
        with pytest.raises(pv.InvalidMeshError, match=re.escape(match)):
            mesh.validate_mesh(action='error')

    # Degenerate voxel
    invalid_mesh = pv.ImageData(dimensions=(2, 2, 2), spacing=(1.0, 1.0, 0.0))
    for mesh in [invalid_mesh, append_mixed_cells(invalid_mesh)]:
        state = mesh.cell_validator()['validity_state']
        assert state[0] & pv.CellStatus.ZERO_SIZE
    match = 'Mesh has 1 VOXEL cell with zero volume. Invalid cell id: [0]'
    for mesh in [invalid_mesh, append_mixed_cells(invalid_mesh)]:
        with pytest.raises(pv.InvalidMeshError, match=re.escape(match)):
            mesh.validate_mesh(action='error')

    # POLY_VERTEX with no points
    invalid_mesh = pv.UnstructuredGrid([0], [pv.CellType.POLY_VERTEX], [])
    match = 'Mesh has 1 POLY_VERTEX cell with zero size. Invalid cell id: [0]'
    with pytest.raises(pv.InvalidMeshError, match=re.escape(match)):
        invalid_mesh.validate_mesh(action='error')

    # Test valid voxel with tiny volume does not generate false positive
    mesh = pv.ImageData(dimensions=(2, 2, 2), spacing=(0.0001, 0.0002, 0.0003))
    assert np.isclose(mesh.volume, 6e-12)
    assert mesh.validate_mesh().is_valid

    # Force invalid with manual tolerance
    match = 'Mesh has 1 VOXEL cell with zero volume. Invalid cell id: [0]'
    with pytest.raises(pv.InvalidMeshError, match=re.escape(match)):
        mesh.validate_mesh(size_tolerance=1e-8, action='error')
    with pytest.raises(pv.InvalidMeshError, match=re.escape(match)):
        mesh.cast_to_multiblock().validate_mesh(size_tolerance=1e-8, action='error')


@pytest.mark.parametrize(
    'mesh_name', ['PolyVertex', 'PolyLine', 'Line', 'Quadrilateral', 'Polygon', 'Hexahedron']
)
def test_validate_mesh_coincident_points(mesh_name):
    """Test that a cell with a collapsed edge (two coincident points) is invalid."""
    mesh = getattr(examples.cells, mesh_name)()

    # Negative case: the pristine cell is valid -- the check must not false-positive.
    assert mesh.validate_mesh().is_valid

    # Collapse an edge so points[0] and points[1] coincide.
    mesh.points[1] = mesh.points[0]

    report = mesh.validate_mesh()
    assert report.is_valid is False
    assert 'coincident_points' in report.invalid_fields

    state = mesh.cell_validator()['validity_state']
    assert state[0] & pv.CellStatus.COINCIDENT_POINTS


@pytest.fixture
def degenerate_structured_grid_quad():
    x = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    y = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    z = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    # Collapse the "south" edge to a single point
    x[0, :] = 0.0
    y[0, :] = 0.0
    z[0, :] = 0.0

    grid = pv.StructuredGrid(x, y, z)
    assert grid.n_cells == 1
    assert grid.n_points == 4
    assert grid.distinct_cell_types == {pv.CellType.QUAD}

    cleaned = grid.extract_surface(algorithm='geometry').clean()
    assert cleaned.n_cells == 1
    assert cleaned.n_points == 3
    assert cleaned.distinct_cell_types == {pv.CellType.TRIANGLE}

    return grid


def test_validate_mesh_coincident_points_structured_grid(degenerate_structured_grid_quad):
    report = degenerate_structured_grid_quad.validate_mesh()
    assert report.invalid_fields == ('coincident_points',)


def test_validate_mesh_invalid_point_references():
    # Define mesh with a cell that has point indices > n_points
    cells = [3, 0, 1, 2]
    celltypes = [pv.CellType.TRIANGLE]
    points = [0.0, 0.0, 0.0]
    grid = pv.UnstructuredGrid(cells, celltypes, points)

    report = grid.validate_mesh('invalid_point_references')
    expected_cell_ids = [0]
    assert report.invalid_point_references == expected_cell_ids


@pytest.mark.parametrize('n_points', [3, 131072], ids=['small', 'large'])
@pytest.mark.parametrize(
    'mesh_type',
    [
        pytest.param(
            pv.PolyData,
            marks=pytest.mark.needs_vtk_version(
                (9, 5, 0),
                reason='Casting PolyData to UnstructuredGrid does not preserve invalid ids',
            ),
        ),
        pv.UnstructuredGrid,
    ],
)
def test_validate_mesh_invalid_point_references_is_only_status(mesh_type, n_points):
    # The large mesh is included since reading a point beyond the last one may fault
    points = np.zeros((n_points, 3))
    cells = [3, 0, 1, n_points]
    mesh = (
        pv.PolyData(points, faces=cells)
        if mesh_type is pv.PolyData
        else pv.UnstructuredGrid(cells, [pv.CellType.TRIANGLE], points)
    )

    validated = mesh.cell_validator()
    assert validated.cell_data['validity_state'][0] == pv.CellStatus.INVALID_POINT_REFERENCES
    assert validated.field_data['invalid'].tolist() == [0]
    for name in CELL_STATUS_ARRAY_NAMES:
        expected = [0] if name == 'invalid_point_references' else []
        assert validated.field_data[name].tolist() == expected

    report = mesh.validate_mesh(exclude_fields=['unused_points'])
    assert report.invalid_fields == ('invalid_point_references',)
    assert 'TRIANGLE cell with invalid point references' in report.message
    assert '{TRIANGLE}' in str(report)


@pytest.fixture
def invalid_hexahedron():
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]

    # Swap first two points to create a bad cell
    cells = [8, 1, 0, 2, 3, 4, 5, 6, 7]
    celltypes = [pv.CellType.HEXAHEDRON]

    return pv.UnstructuredGrid(cells, celltypes, points)


@pytest.fixture
def poly_with_invalid_point():
    poly = pv.PolyData()
    poly.points = [[np.nan, 0.0, 0.0]]
    return poly


@pytest.fixture
def single_cell_invalid_point_references():
    return pv.PolyData([0.0, 0.0, 0.0], [3, 0, 1, 1])


@pytest.fixture
def mixed_2d_cells_invalid_point_references():
    return pv.PolyData([0.0, 0.0, 0.0], [3, 0, 1, 1, 4, 0, 1, 1, 1])


@pytest.fixture
def mixed_dimension_cells_invalid_point_references():
    return pv.PolyData([0.0, 0.0, 0.0], faces=[3, 0, 1, 1], verts=[2, 0, -1])


@pytest.mark.needs_vtk_version(9, 6, 0)
def test_cell_validator_intersecting_edges_nonconvex(invalid_hexahedron):
    validated = invalid_hexahedron.cell_validator()
    expected_cell_ids = [0]
    expected_invalid_fields = ['intersecting_edges', 'non_planar_faces', 'inverted_faces']
    for name in CELL_STATUS_ARRAY_NAMES:
        if name in expected_invalid_fields:
            assert validated[name].tolist() == expected_cell_ids, name
        else:
            array = validated.field_data[name]
            assert array.shape == (0,), name
    assert validated['invalid'].tolist() == expected_cell_ids

    # Test validating specific fields
    report = invalid_hexahedron.validate_mesh('cells')
    assert report.intersecting_edges is not None
    assert report.non_convex is not None
    assert report.inverted_faces is not None

    report = invalid_hexahedron.validate_mesh('non_convex')
    assert report.intersecting_edges is None
    assert report.non_convex is not None
    assert report.inverted_faces is None


@pytest.mark.needs_vtk_version(9, 6, 0)
@pytest.mark.skipif(sys.platform == 'Darwin', reason='Results differ for macOS and older vtk')
def test_validate_mesh_error_message(invalid_hexahedron, poly_with_invalid_point):
    def _format_composite(match):
        prefix = "MultiBlock mesh is not valid:\n * Block id 0 'Block-00' "
        return prefix + match.replace(' - ', '   - ')

    # Test single cell
    match = (
        'UnstructuredGrid mesh is not valid:\n'
        ' - Mesh has 1 HEXAHEDRON cell with intersecting edges. Invalid cell id: [0]\n'
        ' - Mesh has 1 HEXAHEDRON cell with inverted faces. Invalid cell id: [0]\n'
        ' - Mesh has 1 HEXAHEDRON cell with non-planar faces. Invalid cell id: [0]'
    )
    with pytest.warns(pv.InvalidMeshWarning, match=re.escape(match)):
        invalid_hexahedron.validate_mesh(action='warn')
    with pytest.warns(pv.InvalidMeshWarning, match=re.escape(_format_composite(match))):
        invalid_hexahedron.cast_to_multiblock().validate_mesh(action='warn')

    match = (
        'UnstructuredGrid mesh is not valid:\n'
        ' - Mesh has 2 HEXAHEDRON cells with intersecting edges. Invalid cell ids: [0, 1]\n'
        ' - Mesh has 2 HEXAHEDRON cells with inverted faces. Invalid cell ids: [0, 1]\n'
        ' - Mesh has 2 HEXAHEDRON cells with non-planar faces. Invalid cell ids: [0, 1]'
    )
    invalid_hexahedrons = pv.merge([invalid_hexahedron, invalid_hexahedron.translate((3, 3, 3))])
    with pytest.warns(pv.InvalidMeshWarning, match=re.escape(match)):
        invalid_hexahedrons.validate_mesh(action='warn')
    with pytest.warns(pv.InvalidMeshWarning, match=re.escape(_format_composite(match))):
        invalid_hexahedrons.cast_to_multiblock().validate_mesh(action='warn')

    # Test points
    match = (
        'PolyData mesh is not valid:\n'
        ' - Mesh has 1 unused point not referenced by any cell. Invalid point id: [0]\n'
        ' - Mesh has 1 non-finite point. Invalid point id: [0]'
    )
    with pytest.warns(pv.InvalidMeshWarning, match=re.escape(match)):
        poly_with_invalid_point.validate_mesh(action='warn')
    with pytest.warns(pv.InvalidMeshWarning, match=re.escape(_format_composite(match))):
        poly_with_invalid_point.cast_to_multiblock().validate_mesh(action='warn')

    poly_with_invalid_point.points = poly_with_invalid_point.points.tolist() * 100
    # Test multiple points
    ids = '[0, 1, 2, 3, 4, 5, ...]'
    match = (
        'PolyData mesh is not valid:\n'
        f' - Mesh has 100 unused points not referenced by any cell. Invalid point ids: {ids}\n'
        f' - Mesh has 100 non-finite points. Invalid point ids: {ids}'
    )
    with pytest.warns(pv.InvalidMeshWarning, match=re.escape(match)):
        poly_with_invalid_point.validate_mesh(action='warn')
    with pytest.warns(pv.InvalidMeshWarning, match=re.escape(_format_composite(match))):
        poly_with_invalid_point.cast_to_multiblock().validate_mesh(action='warn')


@pytest.mark.needs_vtk_version((9, 5, 0), reason='Suspected issue with fixtures for older VTK')
def test_validate_mesh_distinct_cell_types(
    single_cell_invalid_point_references,
    mixed_2d_cells_invalid_point_references,
    mixed_dimension_cells_invalid_point_references,
):
    kwargs = dict(exclude_fields=['negative_size', 'zero_size'])
    message = single_cell_invalid_point_references.validate_mesh(**kwargs).message
    expected = (
        'PolyData mesh is not valid:\n'
        ' - Mesh has 1 TRIANGLE cell with invalid point references. Invalid cell id: [0]'
    )
    assert expected == message

    message = mixed_2d_cells_invalid_point_references.validate_mesh(**kwargs).message
    expected = (
        'PolyData mesh is not valid:\n'
        ' - Mesh has 1 TRIANGLE cell with invalid point references. Invalid cell id: [0]\n'
        ' - Mesh has 1 QUAD cell with invalid point references. Invalid cell id: [1]'
    )
    assert expected == message

    message = mixed_dimension_cells_invalid_point_references.validate_mesh(**kwargs).message
    expected = (
        'PolyData mesh is not valid:\n'
        ' - Mesh has 1 POLY_VERTEX cell with invalid point references. Invalid cell id: [0]\n'
        ' - Mesh has 1 TRIANGLE cell with invalid point references. Invalid cell id: [1]'
    )
    assert expected == message


@pytest.mark.parametrize('as_grid', [True, False])
@pytest.mark.parametrize('validate', [True, 'cells'])
def test_init_invalid_mesh(invalid_random_polydata, tmp_path, as_grid, validate):
    if as_grid:
        alg = _vtk.vtkAppendFilter()
        alg.AddInputData(invalid_random_polydata)
        alg.Update()
        vtk_mesh = alg.GetOutput()
        mesh = pv.UnstructuredGrid()
        mesh.ShallowCopy(vtk_mesh)
        array_args = mesh.cells, mesh.celltypes, mesh.points
    else:
        mesh = invalid_random_polydata
        vtk_mesh = _vtk.vtkPolyData()
        vtk_mesh.ShallowCopy(mesh)
        array_args = mesh.points, mesh.faces
    mesh_type = type(mesh)

    filepath = tmp_path / 'invalid.vtk'
    mesh.save(filepath)

    match = 'mesh is not valid'

    # Init from file
    with pytest.raises(pv.InvalidMeshError, match=match):
        mesh_type(filepath, validate=validate)

    # Init from unwrapped VTK mesh
    with pytest.raises(pv.InvalidMeshError, match=match):
        mesh_type(vtk_mesh, validate=validate)

    # Init from PyVista mesh
    with pytest.raises(pv.InvalidMeshError, match=match):
        mesh_type(mesh, validate=validate)

    # Init from arrays
    with pytest.raises(pv.InvalidMeshError, match=match):
        mesh_type(*array_args, validate=validate)


@pytest.mark.parametrize(
    'mesh',
    [
        pv.ImageData(),
        pv.RectilinearGrid(),
        pv.StructuredGrid(),
        pv.PointSet(),
        pv.PolyData(),
        pv.UnstructuredGrid(),
        pv.MultiBlock([pv.PolyData()]),
        # pv.ExplicitStructuredGrid(),  Seg fault with empty mesh. This type is tested separately.
    ],
)
@pytest.mark.parametrize('validate', [True, 'data'])
def test_init_mesh_validate(mesh, validate):
    mesh_type = type(mesh)
    if mesh_type is pv.MultiBlock:
        _add_invalid_arrays(mesh[0])
    else:
        _add_invalid_arrays(mesh)

    match = 'mesh is not valid'
    with pytest.raises(pv.InvalidMeshError, match=match):
        mesh_type(mesh, validate=validate)


def test_validate_mesh_explicit_structured_grid():
    grid = examples.load_explicit_structured()
    valid_grid = pv.ExplicitStructuredGrid(grid, validate=True)
    assert valid_grid == grid


def test_extract_surface_multiblock_no_args(multiblock_all_no_pointset_with_nested_and_none):
    # Get output directly from vtkCompositeDataGeometryFilter
    poly_from_vtk_filter = (
        multiblock_all_no_pointset_with_nested_and_none._composite_geometry_filter()
    )

    # Test branch without any config options, similar to vtkCompositeDataGeometryFilter
    kwargs = dict(
        algorithm='dataset_surface',
        pass_cellid=False,
        pass_pointid=False,
        progress_bar=False,
    )
    poly_no_config = multiblock_all_no_pointset_with_nested_and_none.extract_surface(**kwargs)
    assert poly_no_config == poly_from_vtk_filter


@pytest.mark.parametrize('algorithm', ['geometry', 'dataset_surface', None, _SENTINEL])
@pytest.mark.parametrize('bool_kwargs', [True, False])
def test_extract_surface_datasets(multiblock_all_no_pointset, algorithm, bool_kwargs):
    kwargs = dict(
        algorithm=algorithm,
        progress_bar=bool_kwargs,
        pass_cellid=bool_kwargs,
        pass_pointid=bool_kwargs,
    )
    for dataobj in (*multiblock_all_no_pointset, multiblock_all_no_pointset):
        if algorithm is _SENTINEL:
            with pytest.warns(pv.PyVistaFutureWarning):
                surf = dataobj.extract_surface(**kwargs)
        else:
            surf = dataobj.extract_surface(**kwargs)

        assert surf is not None
        assert isinstance(surf, pv.PolyData)
        assert ('vtkOriginalPointIds' in surf.point_data) == bool_kwargs
        assert ('vtkOriginalCellIds' in surf.cell_data) == bool_kwargs


def test_extract_surface_pointset_raises(multiblock_all):
    with pytest.raises(pv.PointSetCellOperationError):
        pv.PointSet().extract_surface(algorithm=None)
    with pytest.raises(pv.PointSetCellOperationError, match='type PointSet'):
        multiblock_all.extract_surface(algorithm=None)


@pytest.mark.parametrize('as_multiblock', [True, False])
def test_extract_surface_nonlinear(as_multiblock):
    # create a single quadratic hexahedral cell
    lin_pts = np.array(
        [
            [-1, -1, -1],  # node 0
            [1, -1, -1],  # node 1
            [1, 1, -1],  # node 2
            [-1, 1, -1],  # node 3
            [-1, -1, 1],  # node 4
            [1, -1, 1],  # node 5
            [1, 1, 1],  # node 6
            [-1, 1, 1],  # node 7
        ],
        np.double,
    )

    quad_pts = np.array(
        [
            (lin_pts[1] + lin_pts[0]) / 2,  # between point 0 and 1
            (lin_pts[1] + lin_pts[2]) / 2,  # between point 1 and 2
            (lin_pts[2] + lin_pts[3]) / 2,  # and so on...
            (lin_pts[3] + lin_pts[0]) / 2,
            (lin_pts[4] + lin_pts[5]) / 2,
            (lin_pts[5] + lin_pts[6]) / 2,
            (lin_pts[6] + lin_pts[7]) / 2,
            (lin_pts[7] + lin_pts[4]) / 2,
            (lin_pts[0] + lin_pts[4]) / 2,
            (lin_pts[1] + lin_pts[5]) / 2,
            (lin_pts[2] + lin_pts[6]) / 2,
            (lin_pts[3] + lin_pts[7]) / 2,
        ],
    )

    # introduce a minor variation to the location of the mid-side points
    quad_pts += np.random.default_rng().random(quad_pts.shape) * 0.25
    pts = np.vstack((lin_pts, quad_pts))

    cells = np.hstack((20, np.arange(20))).astype(np.int64, copy=False)
    celltypes = np.array([pv.CellType.QUADRATIC_HEXAHEDRON])
    grid = pv.UnstructuredGrid(cells, celltypes, pts)
    grid = grid.cast_to_multiblock() if as_multiblock else grid

    # expect each face to be divided 6 times since it has a midside node
    surf = grid.extract_surface(algorithm=None, progress_bar=True)
    assert surf.n_faces == 36
    surf = grid.extract_surface(algorithm='dataset_surface', progress_bar=True)
    assert surf.n_faces == 36

    # expect each face to be divided several more times than the linear extraction
    surf_subdivided = grid.extract_surface(
        algorithm=None, nonlinear_subdivision=5, progress_bar=True
    )
    assert surf_subdivided.n_faces > surf.n_faces
    match = (
        'geometry algorithm cannot process non-linear cells and therefore '
        'cannot be used to control non-linear subdivision.'
    )
    with pytest.raises(ValueError, match=match):
        grid.extract_surface(algorithm='geometry', nonlinear_subdivision=5)

    match = 'Mesh contains non-linear cells which cannot be processed by the geometry algorithm.'
    if as_multiblock:
        match = '(?s)could not be applied to the block at index 0.*' + match
    with pytest.raises(ValueError, match=match):
        grid.extract_surface(algorithm='geometry')

    # No subdivision, expect one face per cell
    surf_no_subdivide = grid.extract_surface(
        algorithm=None, nonlinear_subdivision=0, progress_bar=True
    )
    assert surf_no_subdivide.n_faces == 6


def test_convex_hull_3d():
    # Exercises whichever backend (vtkConvexHull or the scipy fallback) matches
    # the vtk actually installed; the tox matrix covers vtk<9.7 and vtk>=9.7.
    sphere = pv.Sphere()
    hull = sphere.convex_hull()
    assert isinstance(hull, pv.PolyData)
    assert hull.dimensionality == 3
    assert hull.n_points > 0
    assert hull.n_cells > 1
    assert np.allclose(hull.bounds, sphere.bounds, atol=1e-3)
    assert hull.point_data.keys() == []
    assert hull.cell_data.keys() == []


def test_convex_hull_2d():
    circle = pv.Circle(radius=0.5)
    hull = circle.convex_hull(dimensionality=2)
    assert hull.dimensionality == 2
    assert hull.n_cells == 1
    assert np.allclose(hull.bounds, circle.bounds, atol=1e-3)


def test_convex_hull_2d_tilted_plane():
    # A 2D hull is computed from points projected onto the best-fit plane, even
    # when that plane is not axis-aligned.
    circle = pv.Circle(radius=0.5).rotate_vector((1, 2, 3), 40)
    hull = circle.convex_hull(dimensionality=2)
    assert hull.dimensionality == 2
    assert hull.n_cells == 1
    assert np.allclose(hull.bounds, circle.bounds, atol=1e-3)


def test_convex_hull_multiblock():
    circle1 = pv.Circle(radius=0.5)
    circle2 = pv.Circle(radius=0.25).translate((1.0, 0.0, 0.0))
    mesh = pv.MultiBlock([circle1, circle2])
    hull = mesh.convex_hull(dimensionality=2)
    assert hull.n_cells == 1
    combined_bounds = pv.merge([circle1, circle2]).bounds
    assert np.allclose(hull.bounds, combined_bounds, atol=1e-3)


def test_convex_hull_planar_default_dimensionality():
    # dimensionality defaults to 3, which is degenerate for planar input
    circle = pv.Circle(radius=0.5)
    if pv.vtk_version_info >= (9, 7, 0):
        # vtkConvexHull degrades gracefully for degenerate input
        hull = circle.convex_hull()
        assert hull.n_points > 0
    else:
        # the scipy fallback raises instead of silently returning an empty mesh
        with pytest.raises(ValueError, match='degenerate'):
            circle.convex_hull()


def test_convex_hull_dimensionality_1():
    sphere = pv.Sphere()
    if pv.vtk_version_info >= (9, 7, 0):
        hull = sphere.convex_hull(dimensionality=1)
        assert hull.dimensionality == 1
        assert hull.n_points == 2
        assert hull.n_cells == 1
    else:
        with pytest.raises(pv.VTKVersionError, match='dimensionality=1'):
            sphere.convex_hull(dimensionality=1)


def test_convex_hull_auto_dimensionality():
    circle = pv.Circle(radius=0.5)
    hull2d = circle.convex_hull(dimensionality='auto')
    assert hull2d.dimensionality == 2

    sphere = pv.Sphere()
    hull3d = sphere.convex_hull(dimensionality='auto')
    assert hull3d.dimensionality == 3


def test_convex_hull_scipy_3d():
    points = pv.Sphere().points
    hull = _convex_hull_scipy(points, dimensionality=3)
    assert hull.dimensionality == 3
    assert hull.n_points > 0
    assert hull.n_cells > 1


def test_convex_hull_scipy_2d():
    points = pv.Circle(radius=0.5).points
    hull = _convex_hull_scipy(points, dimensionality=2)
    assert hull.dimensionality == 2
    assert hull.n_cells == 1


def test_convex_hull_scipy_dimensionality_1_raises():
    points = pv.Sphere().points
    with pytest.raises(pv.VTKVersionError, match='dimensionality=1'):
        _convex_hull_scipy(points, dimensionality=1)


def test_convex_hull_scipy_degenerate_raises():
    points = pv.Circle(radius=0.5).points  # planar, degenerate for a 3D hull
    with pytest.raises(ValueError, match='degenerate'):
        _convex_hull_scipy(points, dimensionality=3)


def test_convex_hull_scipy_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, 'scipy', None)
    monkeypatch.setitem(sys.modules, 'scipy.spatial', None)
    with pytest.raises(ImportError, match='scipy'):
        _convex_hull_scipy(pv.Sphere().points, dimensionality=3)


def test_resample_to_image(tetbeam):
    tetbeam['point_scalars'] = tetbeam.points[:, 2]
    tetbeam.cell_data['cell_scalars'] = np.arange(tetbeam.n_cells, dtype=float)
    image = tetbeam.resample_to_image()

    assert isinstance(image, pv.ImageData)
    assert 'point_scalars' in image.point_data
    assert 'cell_scalars' in image.point_data
    assert np.allclose(image.points_to_cells().bounds, tetbeam.bounds)

    # The interior of the beam is sampled and its arrays are interpolated
    valid = image['vtkValidPointMask'].astype(bool)
    assert valid.any()
    assert np.allclose(image['point_scalars'][valid], image.points[valid][:, 2])

    # The spacing follows the input's own cells
    coarse = tetbeam.resample_to_image(cell_length_percentile=0.9)
    assert np.all(np.array(coarse.spacing) > image.spacing)


def test_resample_to_image_dimensions_and_spacing(tetbeam):
    dims = (10, 11, 12)
    assert tetbeam.resample_to_image(dimensions=dims).dimensions == dims

    image = tetbeam.resample_to_image(spacing=tetbeam.length / 20)
    assert np.allclose(image.spacing, tetbeam.length / 20, atol=1e-2)
    assert np.allclose(image.points_to_cells().bounds, tetbeam.bounds)


def test_resample_to_image_geometry_matches_voxelize(sphere):
    # Both filters place their voxels identically, so their outputs can be combined
    mask = sphere.voxelize_binary_mask(dimensions=(20, 21, 22))
    image = sphere.resample_to_image(dimensions=(20, 21, 22))

    assert image.dimensions == mask.dimensions
    assert image.spacing == mask.spacing
    assert image.origin == mask.origin


def test_resample_to_image_reference_volume(tetbeam):
    tetbeam['point_scalars'] = tetbeam.points[:, 2]
    reference = pv.ImageData()
    reference.extent = (2, 6, 3, 8, 4, 10)
    reference.spacing = (0.2, 0.3, 0.4)
    reference.origin = (0.1, 0.2, 0.3)
    reference.direction_matrix = pv.Transform().rotate_z(30).matrix[:3, :3]
    reference['reference_scalars'] = np.arange(reference.n_points)

    image = tetbeam.resample_to_image(reference_volume=reference)

    assert image.extent == reference.extent
    assert image.offset == reference.offset
    assert image.spacing == reference.spacing
    assert image.origin == reference.origin
    assert np.allclose(image.direction_matrix, reference.direction_matrix)
    # The reference only defines the geometry, its arrays are not part of the output
    assert 'reference_scalars' not in image.array_names
    assert 'point_scalars' in image.point_data


def voxel_of_each_point(mesh, image):
    """Return the flat index of the voxel containing each of the mesh's points."""
    dims = np.array(image.dimensions)
    ijk = np.round((mesh.points - np.array(image.origin)) / np.array(image.spacing)).astype(int)
    ijk = np.clip(ijk, 0, dims - 1)
    return ijk[:, 0] + dims[0] * (ijk[:, 1] + dims[1] * ijk[:, 2])


def test_resample_to_image_method_interpolate(sphere):
    sphere['point_scalars'] = sphere.points[:, 0]
    dims = (20, 20, 20)

    # Interpolating from the points fills every voxel the surface crosses
    image = sphere.resample_to_image(dimensions=dims)
    valid = image['vtkValidPointMask'].astype(bool)
    assert valid[voxel_of_each_point(sphere.subdivide(3), image)].all()

    # A surface has no volume for a cell search to land in, so sampling does not
    sampled = sphere.resample_to_image(dimensions=dims, method='sample')
    sampled_valid = sampled['vtkValidPointMask'].astype(bool)
    assert not sampled_valid[voxel_of_each_point(sphere, sampled)].all()
    assert sampled_valid.sum() < valid.sum()

    # Only voxels within the default radius of the surface are filled, not the interior
    lengths = _cell_edge_lengths(sphere)
    radius = np.quantile(lengths[lengths > 0], 0.95)
    distance = image.compute_implicit_distance(sphere)['implicit_distance']
    assert np.all(np.abs(distance[valid]) <= radius)
    assert not valid[voxel_of_each_point(pv.PolyData([sphere.center]), image)].any()

    # A smaller radius fills fewer voxels
    half_diagonal = np.linalg.norm(image.spacing) / 2
    tight = sphere.resample_to_image(dimensions=dims, radius=half_diagonal)
    assert tight['vtkValidPointMask'].sum() < valid.sum()

    # A point cloud has no cells to reach across, so its radius is half a voxel diagonal
    cloud = pv.PolyData(sphere.points)
    cloud['point_scalars'] = sphere['point_scalars']
    assert cloud.resample_to_image(dimensions=dims) == cloud.resample_to_image(
        dimensions=dims, radius=half_diagonal
    )


def test_resample_to_image_multiblock():
    blocks = pv.MultiBlock([pv.Sphere(), pv.Sphere(center=(1.5, 0, 0))])
    for block in blocks:
        block['height'] = block.points[:, 2]

    image = blocks.resample_to_image(target_n_points=20_000, mark_blank=True)
    assert isinstance(image, pv.ImageData)
    assert 'height' in image.point_data
    # Voxels are points, so the cells rather than the points span the input's bounds
    assert np.allclose(np.array(image.bounds_size) + np.array(image.spacing), blocks.bounds_size)

    # Surfaces have no volume, so the blocks are interpolated from their points
    valid = image['vtkValidPointMask'].astype(bool)
    assert valid.any()
    assert image.point_data['vtkGhostType'].size == image.n_points

    # Each block reaches the output
    for block in blocks:
        ijk = np.round((block.points - np.array(image.origin)) / np.array(image.spacing)).astype(
            int
        )
        ijk = np.clip(ijk, 0, np.array(image.dimensions) - 1)
        flat = ijk[:, 0] + image.dimensions[0] * (ijk[:, 1] + image.dimensions[1] * ijk[:, 2])
        assert valid[flat].all()


def test_resample_to_image_multiblock_volumetric():
    blocks = pv.MultiBlock(
        [pv.SolidSphere(outer_radius=0.5), pv.SolidSphere(outer_radius=0.5, center=(1.2, 0, 0))]
    )
    for block in blocks:
        block['height'] = block.points[:, 2]

    image = blocks.resample_to_image(target_n_points=20_000)
    assert 'height' in image.point_data
    # Volumetric blocks are sampled, which fills their interiors
    assert image['vtkValidPointMask'].sum() > 0.3 * image.n_points


def test_resample_to_image_multiblock_matches_combined():
    blocks = pv.MultiBlock([pv.Sphere(), pv.Sphere(center=(0.6, 0, 0))])
    for block in blocks:
        block['height'] = block.points[:, 2]

    from_blocks = blocks.resample_to_image(target_n_points=10_000)
    from_combined = blocks.combine().resample_to_image(target_n_points=10_000)
    assert from_blocks.dimensions == from_combined.dimensions
    assert np.allclose(from_blocks.origin, from_combined.origin)
    assert np.allclose(from_blocks['height'], from_combined['height'])


@pytest.mark.parametrize('target', [1_000, 100_000, 1_000_000])
def test_target_n_points(sphere, target):
    image = sphere.resample_to_image(target_n_points=target)
    assert 0.8 <= image.n_points / target <= 1.2
    assert image.n_points == np.prod(image.dimensions)

    # Both filters place their voxels identically
    mask = sphere.voxelize_binary_mask(target_n_points=target)
    assert mask.dimensions == image.dimensions
    assert np.allclose(mask.origin, image.origin)
    assert np.allclose(mask.spacing, image.spacing)

    # Dimensions follow the bounds, so the spacing is isotropic up to the rounding
    spacing = np.array(image.spacing)
    assert spacing.max() / spacing.min() <= 1 + 1 / min(image.dimensions)


def test_target_n_points_flat_axis():
    plane = pv.Plane(i_size=2, j_size=3, i_resolution=20, j_resolution=20)
    image = plane.resample_to_image(target_n_points=10_000)
    # The flat axis holds one point and takes no part in the count
    assert image.dimensions[2] == 1
    assert 0.8 <= image.n_points / 10_000 <= 1.2


def test_max_n_points_clamps_the_defaults():
    mesh = pv.Sphere(theta_resolution=50, phi_resolution=50)

    # An estimated geometry is coarsened to fit, without raising
    assert mesh.resample_to_image().n_points > 1000
    for cap in [1000, 500, 100, 8, 1]:
        assert mesh.resample_to_image(max_n_points=cap).n_points <= cap


@pytest.mark.parametrize('cap', range(1, 200, 7))
def test_max_n_points_is_a_strict_bound(sphere, cap):
    assert sphere.resample_to_image(max_n_points=cap).n_points <= cap


def test_max_n_points_clamps_a_flat_axis():
    plane = pv.Plane(i_size=2, j_size=3, i_resolution=50, j_resolution=50)
    image = plane.resample_to_image(max_n_points=100)
    assert image.dimensions[2] == 1
    assert image.n_points <= 100


def test_max_n_points_raises_for_a_requested_geometry():
    mesh = pv.Sphere(theta_resolution=50, phi_resolution=50)
    match = 'points, which exceeds `max_n_points=1000`'
    for kwargs in [
        dict(dimensions=(40, 40, 40)),
        dict(spacing=0.02),
        dict(cell_length_percentile=0.01),
        dict(reference_volume=pv.ImageData(dimensions=(40, 40, 40), spacing=(0.03,) * 3)),
    ]:
        with pytest.raises(ValueError, match=re.escape(match)):
            mesh.resample_to_image(max_n_points=1000, **kwargs)

    # A requested geometry inside the limit is left alone
    assert mesh.resample_to_image(dimensions=(5, 5, 5), max_n_points=1000).n_points == 125


def test_max_n_points_bounds_the_target(sphere):
    match = 'Target n points (2000) cannot exceed max n points (1000).'
    with pytest.raises(ValueError, match=re.escape(match)):
        sphere.resample_to_image(target_n_points=2000, max_n_points=1000)

    # The target is approached, the limit is not exceeded
    for target in [500, 999, 1000]:
        assert sphere.resample_to_image(target_n_points=target, max_n_points=1000).n_points <= 1000


def test_max_n_points_raises(sphere):
    with pytest.raises(ValueError, match='greater than or equal to'):
        sphere.resample_to_image(max_n_points=0)

    with pytest.raises(ValueError, match='integer-like'):
        sphere.resample_to_image(max_n_points=2.5)


def test_target_n_points_raises(sphere):
    match = 'Target n points cannot be set with dimensions, spacing or cell length options'
    for kwargs in [
        dict(dimensions=(10, 10, 10)),
        dict(spacing=0.1),
        dict(cell_length_percentile=0.5),
        dict(cell_length_sample_size=100),
    ]:
        with pytest.raises(TypeError, match=match):
            sphere.resample_to_image(target_n_points=1000, **kwargs)

    with pytest.raises(TypeError, match='Cannot specify a reference volume'):
        sphere.resample_to_image(target_n_points=1000, reference_volume=pv.ImageData())

    with pytest.raises(ValueError, match='greater than or equal to'):
        sphere.resample_to_image(target_n_points=0)

    with pytest.raises(ValueError, match='integer-like'):
        sphere.resample_to_image(target_n_points=2.5)


def test_resample_to_image_interpolate_warns_on_cell_data(sphere, tetbeam):
    sphere.clear_data()
    sphere.cell_data['cval'] = np.arange(sphere.n_cells, dtype=float)

    match = r"Cell data \['cval'\] is dropped by `method='interpolate'`, chosen for this input"
    with pytest.warns(UserWarning, match=match):
        image = sphere.resample_to_image(dimensions=(20, 20, 20))
    assert 'cval' not in image.point_data

    def dropped_warnings(recorded):
        """Return only the warnings this filter raises about dropped cell data."""
        return [w for w in recorded if 'is dropped by' in str(w.message)]

    # Converting first keeps it, and warns no more
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        converted = sphere.cell_data_to_point_data().resample_to_image(dimensions=(20, 20, 20))
    assert not dropped_warnings(recorded)
    assert 'cval' in converted.point_data

    # Asking for the method by name says so instead
    with pytest.warns(UserWarning, match=r"`method='interpolate'`, which reads point data"):
        sphere.resample_to_image(dimensions=(20, 20, 20), method='interpolate')

    # `sample` carries cell data, so it does not warn
    tetbeam.clear_data()
    tetbeam.cell_data['cval'] = np.arange(tetbeam.n_cells, dtype=float)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        sampled = tetbeam.resample_to_image(dimensions=(10, 10, 10))
    assert not dropped_warnings(recorded)
    assert 'cval' in sampled.point_data


def test_resample_to_image_blanks_invalid_points(sphere, tetbeam):
    ghost_name = pv._vtk.vtkDataSetAttributes.GhostArrayName()
    hidden = pv._vtk.vtkDataSetAttributes.HIDDENPOINT

    # Both methods hide the voxels their mask flags as empty
    for mesh, kwargs in [(sphere, dict(dimensions=(20, 20, 20))), (tetbeam, {})]:
        for method in ['sample', 'interpolate']:
            image = mesh.resample_to_image(method=method, mark_blank=True, **kwargs)
            invalid = image['vtkValidPointMask'] == 0
            ghosts = image.point_data[ghost_name]
            assert ghosts.dtype == np.uint8
            assert np.array_equal(ghosts, np.where(invalid, hidden, 0))

            # Blanking is off by default, which keeps the mask but hides nothing
            unmarked = mesh.resample_to_image(method=method, **kwargs)
            assert ghost_name not in unmarked.point_data
            assert ghost_name not in unmarked.cell_data
            assert np.array_equal(unmarked['vtkValidPointMask'], image['vtkValidPointMask'])


@pytest.mark.parametrize(('method', 'kwargs'), [('sample', {}), ('interpolate', {'radius': 0.05})])
def test_resample_to_image_null_value(sphere, method, kwargs):
    sphere.clear_data()
    sphere['point_scalars'] = sphere.points[:, 0]
    dims = (20, 20, 20)
    shared = dict(dimensions=dims, method=method, **kwargs)

    plain = sphere.resample_to_image(**shared)
    invalid = plain['vtkValidPointMask'] == 0
    assert invalid.any()
    assert np.array_equal(plain['point_scalars'][invalid], np.zeros(invalid.sum()))

    # Both methods fill the empty voxels, and leave the rest alone
    filled = sphere.resample_to_image(null_value=-99.0, **shared)
    assert np.array_equal(filled['point_scalars'][invalid], np.full(invalid.sum(), -99.0))
    assert np.array_equal(filled['point_scalars'][~invalid], plain['point_scalars'][~invalid])

    # The mask and the blanking flags are not values to fill
    blanked = sphere.resample_to_image(null_value=-99.0, mark_blank=True, **shared)
    hidden = pv._vtk.vtkDataSetAttributes.HIDDENPOINT
    ghost_name = pv._vtk.vtkDataSetAttributes.GhostArrayName()
    assert np.array_equal(blanked['vtkValidPointMask'], plain['vtkValidPointMask'])
    assert np.array_equal(blanked.point_data[ghost_name], np.where(invalid, hidden, 0))


@pytest.mark.parametrize('null_value', [-1.0, 300.0, np.nan, 1.5])
def test_resample_to_image_null_value_dtype_raises(sphere, null_value):
    sphere.clear_data()
    sphere['counts'] = np.arange(sphere.n_points, dtype=np.uint8)
    match = (
        f"`null_value={null_value}` cannot be stored in array 'counts', whose "
        '`uint8` data type holds integers from 0 to 255.'
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        sphere.resample_to_image(dimensions=(20, 20, 20), method='sample', null_value=null_value)


def test_resample_to_image_null_value_float_dtype(sphere):
    sphere.clear_data()
    sphere['counts'] = np.arange(sphere.n_points, dtype=np.float32)
    image = sphere.resample_to_image(dimensions=(20, 20, 20), method='sample', null_value=-1.0)
    assert image['counts'][image['vtkValidPointMask'] == 0].min() == -1.0


def test_resample_to_image_method_default(sphere, mocker: MockerFixture):
    from pyvista.core.filters import data_object
    from pyvista.core.filters import data_set

    sample = mocker.spy(data_object.DataObjectFilters, 'sample')
    interpolate = mocker.spy(data_set.DataSetFilters, 'interpolate')
    dims = (20, 20, 20)

    # A volumetric input is sampled from its cells
    sphere.delaunay_3d().resample_to_image(dimensions=dims)
    assert sample.call_count == 1
    assert interpolate.call_count == 0

    # Anything else is interpolated from its points
    for mesh in (sphere, pv.PointSet(sphere.points), pv.Line()):
        mesh.resample_to_image(dimensions=dims)
    assert sample.call_count == 1
    assert interpolate.call_count == 3

    # An explicit method overrides the default
    sphere.resample_to_image(dimensions=dims, method='sample')
    assert sample.call_count == 2


def test_resample_to_image_interpolate_point_cloud(sphere):
    # A point cloud has no cells at all, so only interpolation can reach it
    cloud = pv.PointSet(sphere.points)
    cloud['point_scalars'] = sphere.points[:, 0]
    image = cloud.resample_to_image(dimensions=(20, 20, 20))

    valid = image['vtkValidPointMask'].astype(bool)
    assert valid[voxel_of_each_point(cloud, image)].all()
    # Each filled voxel takes a value from points no further away than the radius
    radius = np.linalg.norm(image.spacing) / 2
    assert np.allclose(image['point_scalars'][valid], image.points[valid][:, 0], atol=radius)


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_resample_to_image_flat_input(axis):
    direction = np.zeros(3)
    direction[axis] = 1
    other = (axis + 1) % 3
    plane = pv.Plane(direction=direction, i_size=2, j_size=3, i_resolution=9, j_resolution=9)
    plane['point_scalars'] = plane.points[:, other]
    image = plane.resample_to_image(dimensions=np.where(np.eye(3)[axis], 1, 10).astype(int))

    assert image.dimensions[axis] == 1
    assert image.spacing[axis] > 0
    # Every voxel of the flat image takes a value from the plane
    assert image['vtkValidPointMask'].all()

    # The voxels are centered on the plane, so sampling its cells is exact
    exact = plane.resample_to_image(dimensions=image.dimensions, method='sample')
    assert np.allclose(exact['point_scalars'], exact.points[:, other])


def test_resample_to_image_categorical(tetbeam):
    tetbeam.point_data['labels'] = np.where(tetbeam.points[:, 2] > 2.5, 7.0, 3.0)
    dims = (8, 8, 8)

    blended = tetbeam.resample_to_image(dimensions=dims)
    categorical = tetbeam.resample_to_image(dimensions=dims, categorical=True)

    valid = categorical['vtkValidPointMask'].astype(bool)
    # Interpolating labels invents values between them, nearest neighbor does not
    assert np.array_equal(np.unique(categorical['labels'][valid]), [3.0, 7.0])
    assert len(np.unique(blended['labels'][valid])) > 2


def test_resample_to_image_raises(sphere):
    match = 'Spacing and dimensions cannot both be set. Set one or the other.'
    with pytest.raises(TypeError, match=match):
        sphere.resample_to_image(dimensions=(1, 2, 3), spacing=(4, 5, 6))

    match = (
        'Cannot specify a reference volume with other geometry parameters. '
        '`reference_volume` must define the geometry exclusively.'
    )
    with pytest.raises(TypeError, match=re.escape(match)):
        sphere.resample_to_image(reference_volume=pv.ImageData(), dimensions=(1, 2, 3))

    match = (
        'Spacing cannot be estimated from the input cells. '
        'Set `dimensions` or `spacing` explicitly.'
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        pv.PointSet(np.zeros((4, 3))).resample_to_image()

    with pytest.raises(ValueError, match="method 'nonsense' is not valid"):
        sphere.resample_to_image(dimensions=(4, 5, 6), method='nonsense')

    match = 'spacing values must all be greater than 0.0.'
    with pytest.raises(ValueError, match=re.escape(match)):
        sphere.resample_to_image(spacing=0)
    match = 'spacing must have finite values.'
    with pytest.raises(ValueError, match=re.escape(match)):
        sphere.resample_to_image(spacing=np.inf)
    match = 'rounding_func output must have integer-like values.'
    with pytest.raises(ValueError, match=re.escape(match)):
        sphere.resample_to_image(spacing=0.1, rounding_func=lambda d: np.asarray(d) + 0.5)

    for name, value in [('radius', 0.1), ('sharpness', 4.0)]:
        match = f"`{name}` requires `method='interpolate'`, but `method='sample'`."
        with pytest.raises(TypeError, match=re.escape(match)):
            sphere.resample_to_image(dimensions=(4, 5, 6), method='sample', **{name: value})

    for name, value in [('tolerance', 0.1), ('categorical', True), ('categorical', False)]:
        match = f"`{name}` requires `method='sample'`, but `method='interpolate'`."
        with pytest.raises(TypeError, match=re.escape(match)):
            sphere.resample_to_image(dimensions=(4, 5, 6), method='interpolate', **{name: value})

    # The message says when the method was picked for the input rather than requested
    match = "`radius` requires `method='interpolate'`, but `method='sample'`, chosen for this"
    with pytest.raises(TypeError, match=re.escape(match)):
        sphere.delaunay_3d().resample_to_image(dimensions=(4, 5, 6), radius=0.1)


def _surface():
    return pv.Sphere(theta_resolution=8, phi_resolution=8)


def _cloud():
    return pv.PointSet(_surface().points)


def _plane():
    return generate_plane((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def _line():
    return pv.Line((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0), resolution=4)


_COMPOSITE_FILTERS = {
    'clip': lambda mesh: mesh.clip(),
    'clip_box': lambda mesh: mesh.clip_box(),
    'clip_slab': lambda mesh: mesh.clip_slab(thickness=0.5, normal='z'),
    'slice': lambda mesh: mesh.slice(),
    'slice_implicit': lambda mesh: mesh.slice_implicit(_plane()),
    'slice_along_line': lambda mesh: mesh.slice_along_line(_line()),
    'extract_all_edges': lambda mesh: mesh.extract_all_edges(),
    'cell_centers': lambda mesh: mesh.cell_centers(),
    'triangulate': lambda mesh: mesh.triangulate(),
    'outline_corners': lambda mesh: mesh.outline_corners(nested=True),
}

_POINTSET_BLOCK_TYPE = {
    'clip': pv.PointSet,
    'clip_box': pv.PointSet,
    'clip_slab': pv.PointSet,
    'cell_centers': pv.PolyData,
    'outline_corners': pv.PolyData,
}

_POINTSET_RAISES = {
    'slice': PointSetDimensionReductionError,
    'slice_implicit': PointSetDimensionReductionError,
    'slice_along_line': PointSetDimensionReductionError,
    'extract_all_edges': PointSetCellOperationError,
    'triangulate': PointSetCellOperationError,
}


@pytest.mark.parametrize('name', sorted(_COMPOSITE_FILTERS))
def test_composite_filter_pointset_block_type(name):
    composite = pv.MultiBlock([_cloud()])
    if name in _POINTSET_RAISES:
        with pytest.raises(_POINTSET_RAISES[name]):
            _COMPOSITE_FILTERS[name](composite)
    else:
        assert type(_COMPOSITE_FILTERS[name](composite)[0]) is _POINTSET_BLOCK_TYPE[name]


@pytest.mark.parametrize('name', sorted(_COMPOSITE_FILTERS))
def test_composite_filter_keeps_empty_block(name):
    out = _COMPOSITE_FILTERS[name](pv.MultiBlock([_surface(), None]))
    assert out[1] is None
