"""Guard the samples and arguments that measure double-precision support.

The marks on each class's ``Filters`` section come from running the filter, so a filter
that cannot be run is left unmarked. These tests pin which ones those are, so that a
filter added without a sample argument is caught here rather than quietly losing its
mark.
"""

from __future__ import annotations

import pytest

from doc.source import points_dtype as _points_dtype
import pyvista as pv
from pyvista.ext import _autoinherit as autoinherit
from tests.conftest import PYVISTA_ROOT_DIR

# Filters that no sample measures: they render, take a callback, need data the samples do
# not carry, or return something other than a dataset.
UNMEASURED = frozenset(
    {
        'collision',
        'contour_banded',
        'crop',
        'curvature',
        'decimate_polyline',
        'edge_mask',
        'extract_cells_by_type',
        'extract_values',
        'extrude_trim',
        'flip_normals',
        'generic_filter',
        'geodesic_distance',
        'high_pass',
        'intersection',
        'label_connectivity',
        'low_pass',
        'multi_ray_trace',
        'plot_boundaries',
        'plot_curvature',
        'plot_normals',
        'plot_over_circular_arc',
        'plot_over_circular_arc_normal',
        'plot_over_line',
        'protein_ribbon',
        'ray_trace',
        'remove_points',
        'resize',
        'rfft',
        'ribbon',
        'ruled_surface',
        'sample_over_circular_arc',
        'sample_over_circular_arc_normal',
        'sample_over_multiple_lines',
        'select_interior_points',
        'select_values',
        'slice_along_line',
        'slice_implicit',
        'slice_index',
        'streamlines_evenly_spaced_2D',
        'streamlines_from_source',
        'subdivide_tetra',
        'surface_indices',
        'triangulate_contours',
        'tube',
        'validate_mesh',
    }
)

# Filters measured on some dataset types but not on these ones.
UNMEASURED_PER_CLASS: dict[str, frozenset[str]] = {
    'PolyData': frozenset(
        {
            'compute_boundary_mesh_quality',
            'extract_feature_edges',
            'streamlines',
            'tessellate',
            'warp_by_vector',
        }
    ),
    'UnstructuredGrid': frozenset({'clip_scalar', 'clip_slab', 'clip_surface'}),
    'StructuredGrid': frozenset({'clip_scalar'}),
    'PointSet': frozenset(
        {
            'clip_box',
            'clip_scalar',
            'compute_boundary_mesh_quality',
            'connectivity',
            'ctp',
            'extract_cells',
            'extract_feature_edges',
            'extract_largest',
            'extract_points',
            'partition',
            'ptc',
            'remove_nan_cells',
            'sample_over_line',
            'split_bodies',
            'streamlines',
            'voxelize',
            'voxelize_binary_mask',
            'voxelize_rectilinear',
        }
    ),
    'ImageData': frozenset({'clip_scalar', 'delaunay_3d'}),
    'RectilinearGrid': frozenset(
        {'clip_scalar', 'delaunay_3d', 'rotate_vector', 'rotate_x', 'rotate_y', 'rotate_z'}
    ),
    'ExplicitStructuredGrid': frozenset({'clip_scalar'}),
}


@pytest.fixture(scope='module', autouse=True)
def _srcdir():
    """Point the documented-class registry at the real documentation source."""
    autoinherit._srcdir = PYVISTA_ROOT_DIR / 'doc' / 'source'
    yield
    autoinherit._srcdir = None


def _filters(class_name: str) -> set[str]:
    """Return the filter names shown in ``class_name``'s ``Filters`` section."""
    members = [name for name in dir(getattr(pv, class_name)) if not name.startswith('_')]
    rows = autoinherit.filter_member_rows('pyvista', class_name, members)
    return {label.rsplit('.', 1)[-1] for label, _, _ in rows}


@pytest.mark.parametrize('class_name', sorted(_points_dtype.SAMPLES))
def test_sample_builds(class_name):
    """Every dataset type has a sample with points for a filter to act on."""
    mesh = _points_dtype.sample(class_name)
    assert type(mesh).__name__ == class_name
    blocks = mesh.recursive_iterator() if isinstance(mesh, pv.MultiBlock) else [mesh]
    assert all(block.n_points for block in blocks)


@pytest.mark.parametrize('class_name', sorted(_points_dtype.SAMPLES))
def test_every_filter_is_measured(class_name):
    """Each filter is measured, or is one of the known unmeasured ones."""
    expected = UNMEASURED | UNMEASURED_PER_CLASS.get(class_name, frozenset())
    names = _filters(class_name)
    unmeasured = {n for n in names if _points_dtype.delivers_double(class_name, n) is None}

    missing = unmeasured - expected
    assert not missing, (
        f'{class_name} filters are no longer measured: {sorted(missing)}. Add sample '
        f'arguments to `ARGS` in doc/source/points_dtype.py, or list them here.'
    )

    stale = (expected & names) - unmeasured
    assert not stale, (
        f'{class_name} filters are measured now: {sorted(stale)}. Remove them from '
        f'UNMEASURED or UNMEASURED_PER_CLASS.'
    )


@pytest.mark.parametrize('class_name', sorted(_points_dtype.SAMPLES))
def test_mark_is_a_known_symbol(class_name):
    """Every filter renders one of the two marks, or none at all."""
    marks = {_points_dtype.points_dtype_mark(class_name, n) for n in _filters(class_name)}
    assert marks <= {_points_dtype.YES_MARK, _points_dtype.NO_MARK, ''}


def test_mark_is_empty_for_an_undocumented_class():
    """A class with no sample is left unmarked rather than raising."""
    assert _points_dtype.points_dtype_mark('Camera', 'DataSetFilters.contour') == ''
