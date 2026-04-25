"""Tests for pyvista.core.formatting_html module."""

from __future__ import annotations

import re

import numpy as np
import pytest

import pyvista as pv
from pyvista.core.formatting_html import _array_badges
from pyvista.core.formatting_html import _badge
from pyvista.core.formatting_html import _children_section
from pyvista.core.formatting_html import _copy_btn
from pyvista.core.formatting_html import _data_array_section
from pyvista.core.formatting_html import _fmt_memory
from pyvista.core.formatting_html import _load_css
from pyvista.core.formatting_html import _load_mesh_icon
from pyvista.core.formatting_html import _metadata_html
from pyvista.core.formatting_html import _summarize_array
from pyvista.core.formatting_html import build_repr_html
from pyvista.core.formatting_html import collapsible_section


def test_load_css():
    css = _load_css()
    assert 'pv-wrap' in css
    assert 'pv-section-summary' in css


def test_css_cached():
    a = _load_css()
    b = _load_css()
    assert a is b


@pytest.mark.parametrize(
    'mesh_type',
    [
        'PolyData',
        'UnstructuredGrid',
        'ImageData',
        'RectilinearGrid',
        'StructuredGrid',
        'MultiBlock',
    ],
)
def test_load_mesh_icon(mesh_type):
    svg = _load_mesh_icon(mesh_type)
    assert svg is not None
    assert '<svg' in svg


def test_unknown_mesh_type_returns_none():
    assert _load_mesh_icon('NonExistent') is None


def test_collapsible_section_basic():
    html = collapsible_section('Test:', details='<p>body</p>', n_items=3)
    assert 'pv-section-summary' in html
    assert 'Test:' in html
    assert '(3)' in html
    assert '<p>body</p>' in html
    # Must include a checkbox input for expand/collapse
    assert "type='checkbox'" in html


def test_collapsible_section_disabled_when_no_items():
    html = collapsible_section('Empty:', n_items=0)
    assert 'disabled' in html


def test_collapsible_section_disabled_when_enabled_false():
    html = collapsible_section('Label:', enabled=False, n_items=5)
    assert 'disabled' in html


def test_collapsible_section_collapsed():
    html = collapsible_section('Col:', details='d', n_items=1, collapsed=True)
    assert 'checked' not in html


def test_collapsible_section_expanded_by_default():
    html = collapsible_section('Exp:', details='d', n_items=2, collapsed=False)
    assert 'checked' in html


def test_collapsible_section_unique_ids():
    a = collapsible_section('A:', n_items=1)
    b = collapsible_section('B:', n_items=1)
    ids_a = re.findall(r"id='(section-[^']+)'", a)
    ids_b = re.findall(r"id='(section-[^']+)'", b)
    assert ids_a[0] != ids_b[0]


def test_collapsible_section_no_details_div_when_empty():
    html = collapsible_section('X:', n_items=1)
    assert 'pv-section-details' not in html


def test_collapsible_section_inline_details():
    html = collapsible_section('X:', inline_details='inline text', n_items=1)
    assert 'inline text' in html
    assert 'pv-section-inline-details' in html


def test_badge_html():
    html = _badge('active', 'pv-badge-active')
    assert 'pv-badge' in html
    assert 'pv-badge-active' in html
    assert 'active' in html


def test_badge_escapes_html():
    html = _badge('<script>', 'pv-badge-active')
    assert '&lt;script&gt;' in html
    assert '<script>' not in html


def test_copy_btn_uses_data_attribute():
    html = _copy_btn('hello')
    assert "data-copy='hello'" in html
    assert 'this.dataset.copy' in html
    assert 'pv-copy-btn' in html


def test_copy_btn_escapes_quotes():
    """Verify that quotes and HTML chars in the copy text are safely escaped
    in the data-copy attribute, preventing attribute breakout."""
    html = _copy_btn('it\'s a <test> & "value"')
    # Must not contain raw quotes/HTML that could break the attribute
    assert 'it&#x27;s' in html
    assert '&lt;test&gt;' in html
    assert '&amp;' in html
    # The raw dangerous characters must not appear in the attribute
    assert "data-copy='it's" not in html


@pytest.mark.parametrize(
    ('kwarg', 'expected_class'),
    [
        ('active_scalars', 'pv-badge-active'),
        ('active_vectors', 'pv-badge-vectors'),
        ('active_normals', 'pv-badge-normals'),
        ('active_tcoords', 'pv-badge-tcoords'),
    ],
)
def test_array_badges_single(kwarg, expected_class):
    badges = _array_badges('arr', **{kwarg: 'arr'})
    assert expected_class in badges


def test_array_badges_no_match_returns_empty():
    assert _array_badges('foo') == ''


def test_array_badges_multiple():
    badges = _array_badges('arr', active_scalars='arr', active_vectors='arr')
    assert 'pv-badge-active' in badges
    assert 'pv-badge-vectors' in badges


def test_summarize_array_scalar_vs_multicomponent():
    scalar_html = _summarize_array('temperature', 1, 'float64')
    assert 'scalar' in scalar_html
    assert 'temperature' in scalar_html
    assert 'float64' in scalar_html

    vec_html = _summarize_array('velocity', 3, 'float32')
    assert '3 comp' in vec_html
    assert 'scalar' not in vec_html


def test_summarize_array_escaping():
    html = _summarize_array('<b>bad</b>', 1, 'int32')
    assert '&lt;b&gt;bad&lt;/b&gt;' in html


def test_summarize_array_active_class():
    active_html = _summarize_array('temp', 1, 'float64', is_active=True)
    assert 'pv-var-name-active' in active_html

    inactive_html = _summarize_array('temp', 1, 'float64', is_active=False)
    assert 'pv-var-name-active' not in inactive_html


def test_data_array_section_renders_all_arrays():
    arrays = [('temp', 1, 'float64', '', '[-1.0, 1.0]'), ('vel', 3, 'float32', '', '')]
    html = _data_array_section('Point Data', arrays)
    assert 'Point Data' in html
    assert '(2)' in html
    assert 'temp' in html
    assert 'vel' in html
    assert '[-1.0, 1.0]' in html


def test_data_array_section_active_inline_label():
    """When an active scalar exists, its name should appear in the inline
    summary (visible when collapsed) with an 'active' badge."""
    arrays = [('temp', 1, 'float64', '', ''), ('pres', 1, 'float32', '', '')]
    html = _data_array_section('Point Data', arrays, active_scalars='temp')

    # Extract the inline-details div content
    inline_match = re.search(r"class='pv-section-inline-details'>(.*?)</div>", html, re.DOTALL)
    assert inline_match is not None
    inline_content = inline_match.group(1)
    assert 'temp' in inline_content
    assert 'pv-badge-active' in inline_content


def test_data_array_section_no_active_no_inline_badge():
    arrays = [('temp', 1, 'float64', '', '')]
    html = _data_array_section('Point Data', arrays)
    inline_match = re.search(r"class='pv-section-inline-details'>(.*?)</div>", html, re.DOTALL)
    assert inline_match is not None
    assert 'pv-badge-active' not in inline_match.group(1)


def test_data_array_section_empty_disabled():
    html = _data_array_section('Cell Data', [])
    assert '(0)' in html
    assert 'disabled' in html


def test_data_array_section_shape_overrides_comp_label():
    """When shape_str is provided, it replaces the 'scalar'/'N comp' label."""
    arrays = [('info', 1, 'int64', '(42,)', '')]
    html = _data_array_section('Field Data', arrays)
    assert '(42,)' in html
    assert 'scalar' not in html


def test_metadata_html_structure_and_content():
    rows = [('Bounds', [('X', '[0, 1]'), ('Y', '[0, 2]')], '(0, 1, 0, 2)')]
    html = _metadata_html(rows)
    assert 'pv-metadata' in html
    assert 'Bounds' in html
    assert '[0, 1]' in html
    assert 'pv-meta-label' in html
    # copy_text is non-empty, so copy button should appear
    assert 'pv-copy-btn' in html


def test_metadata_html_no_copy_button_when_empty():
    rows = [('Cells', [('faces', '100')], '')]
    html = _metadata_html(rows)
    assert 'pv-copy-btn' not in html


def test_metadata_html_escaping():
    rows = [('<script>', [('<k>', '<v>')], '')]
    html = _metadata_html(rows)
    assert '&lt;script&gt;' in html
    assert '&lt;k&gt;' in html
    assert '&lt;v&gt;' in html


@pytest.mark.parametrize(
    ('kib', 'expected'),
    [
        (512, '512 KiB'),
        (1536, '1.5 MiB'),
        (2048, '2.0 MiB'),
        (2 * 1024 * 1024, '2.00 GiB'),
    ],
)
def test_fmt_memory(kib, expected):
    assert _fmt_memory(kib) == expected


def test_children_section_renders_all_children():
    children = [('block0', 'PolyData', '100 pts'), ('block1', 'UnstructuredGrid', '')]
    html = _children_section('Blocks', children)
    assert 'Blocks' in html
    assert 'block0' in html
    assert 'PolyData' in html
    assert '100 pts' in html
    assert 'pv-child-name' in html
    assert '(2)' in html


def test_children_section_escaping():
    children = [('<script>', 'Type', '')]
    html = _children_section('Blocks', children)
    assert '&lt;script&gt;' in html


def test_build_repr_html_structure():
    html = build_repr_html('PolyData', 'PolyData', text_repr='test fallback')
    assert '<style>' in html
    assert 'pv-wrap' in html
    assert 'pv-text-repr-fallback' in html
    assert 'test fallback' in html
    assert 'PolyData' in html
    # Should include a mesh icon
    assert '<svg' in html
    assert 'pv-logo' in html


def test_build_repr_html_unknown_mesh_type_falls_back():
    html = build_repr_html('CustomType', 'NonExistent')
    # Should fall back to ImageData icon rather than no icon
    assert '<svg' in html


def test_build_repr_html_header_badges():
    html = build_repr_html('PolyData', 'PolyData', header_badges=['1.5 MiB'])
    assert '1.5 MiB' in html
    assert 'pv-header-badge' in html


def test_build_repr_html_no_badges():
    html = build_repr_html('PolyData', 'PolyData')
    body = html.split('</style>')[1]
    assert 'pv-header-badge' not in body


def test_build_repr_html_text_repr_escaped():
    html = build_repr_html('T', 'PolyData', text_repr='<script>alert(1)</script>')
    assert '&lt;script&gt;' in html


def test_build_repr_html_metadata_conditional():
    """Metadata div should only appear when metadata is provided."""
    with_meta = build_repr_html('T', 'PolyData', metadata=[('B', [('X', '1')], '')])
    without_meta = build_repr_html('T', 'PolyData')
    assert 'pv-metadata' in with_meta
    assert 'pv-metadata' not in without_meta.split('</style>')[1]


def test_dataset_repr_polydata():
    """Comprehensive check for the most common mesh type."""
    mesh = pv.Sphere()
    html = mesh._repr_html_()
    assert 'PolyData' in html
    # Header badges: points, cells, memory
    assert 'points' in html
    assert 'cells' in html
    assert 'pv-header-badge' in html
    # Bounds metadata
    assert 'Bounds' in html
    assert 'pv-metadata' in html
    # PolyData cell breakdown
    assert 'faces' in html
    # Point Data with Normals badge
    assert 'Point Data' in html
    assert 'Normals' in html
    assert 'pv-badge-normals' in html
    # Text fallback
    assert 'pv-text-repr-fallback' in html


def test_dataset_repr_active_scalars():
    mesh = pv.Sphere()
    mesh['Z Height'] = mesh.points[:, 2]
    html = mesh._repr_html_()
    assert 'pv-badge-active' in html
    assert 'Z Height' in html
    assert 'pv-var-name-active' in html
    # Should show min/max range for numeric arrays
    assert 'pv-var-range' in html
    assert '-5.000e-01' in html  # z_min of sphere is -0.5


def test_dataset_repr_imagedata_grid_metadata():
    """ImageData should show grid-specific dims and spacing."""
    mesh = pv.ImageData(dimensions=(3, 4, 5))
    html = mesh._repr_html_()
    assert 'ImageData' in html
    assert '3 x 4 x 5' in html
    assert 'spacing' in html


def test_dataset_repr_no_arrays_omits_sections():
    """When a dataset has no data arrays, the data sections should not appear."""
    mesh = pv.PolyData()
    mesh.points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    html = mesh._repr_html_()
    sections_html = html.split('pv-sections')[1] if 'pv-sections' in html else ''
    assert 'Point Data' not in sections_html
    assert 'Cell Data' not in sections_html


def test_dataset_repr_cell_data():
    mesh = pv.Sphere()
    mesh.cell_data['cell_arr'] = np.arange(mesh.n_cells)
    html = mesh._repr_html_()
    assert 'Cell Data' in html
    assert 'cell_arr' in html


def test_dataset_repr_field_data_shows_shape():
    """Field data arrays should display their shape instead of 'scalar'/'N comp'."""
    mesh = pv.Sphere()
    mesh.field_data['info'] = np.array([1, 2, 3])
    html = mesh._repr_html_()
    assert 'Field Data' in html
    assert 'info' in html
    assert '(3,)' in html


def test_dataset_repr_active_vectors_badge():
    mesh = pv.Sphere()
    mesh.point_data['vel'] = np.random.default_rng(0).random((mesh.n_points, 3))
    mesh.active_vectors_name = 'vel'
    html = mesh._repr_html_()
    assert 'pv-badge-vectors' in html


def test_dataset_repr_active_tcoords_badge():
    mesh = pv.Sphere()
    mesh.point_data['UV'] = np.random.default_rng(0).random((mesh.n_points, 2))
    mesh.point_data.active_texture_coordinates_name = 'UV'
    html = mesh._repr_html_()
    assert 'pv-badge-tcoords' in html


def test_dataset_repr_string_field_data():
    """String field data exercises the isinstance(arr, str) coercion path
    and must not crash."""
    mesh = pv.Sphere()
    mesh.field_data['name'] = 'test_string'
    html = mesh._repr_html_()
    assert 'Field Data' in html
    assert 'name' in html


def test_dataset_repr_html_escapes_special_chars():
    mesh = pv.Sphere()
    mesh.point_data['<script>'] = np.zeros(mesh.n_points)
    html = mesh._repr_html_()
    assert '&lt;script&gt;' in html
    assert '<script>' not in html.split('<style>')[0]


def test_dataset_repr_polydata_strips():
    """Stripped PolyData should show a 'strips' count in the cell breakdown."""
    mesh = pv.Sphere().strip()
    assert mesh.n_strips > 0, 'test setup: expected strips after .strip()'
    html = mesh._repr_html_()
    assert 'strips' in html


def test_dataset_repr_cell_scalars_active():
    """Active scalars associated with cells (not points) must still
    get the active badge in the Cell Data section."""
    mesh = pv.Sphere()
    mesh.cell_data['cdata'] = np.arange(mesh.n_cells)
    mesh.set_active_scalars('cdata', preference='cell')
    html = mesh._repr_html_()
    assert 'Cell Data' in html
    assert 'pv-badge-active' in html


def test_multiblock_repr_basic():
    mb = pv.MultiBlock([pv.Sphere(), pv.Cube()])
    html = mb._repr_html_()
    assert 'MultiBlock' in html
    assert 'pv-wrap' in html
    assert 'Blocks' in html
    assert 'Bounds' in html
    assert 'pv-metadata' in html


def test_multiblock_repr_named_blocks():
    mb = pv.MultiBlock()
    mb['sphere'] = pv.Sphere()
    mb['cube'] = pv.Cube()
    html = mb._repr_html_()
    assert 'sphere' in html
    assert 'cube' in html


def test_multiblock_repr_empty():
    mb = pv.MultiBlock()
    html = mb._repr_html_()
    assert 'MultiBlock' in html
    assert 'blocks' in html


def test_multiblock_repr_block_detail():
    """Child blocks should show point/cell counts and memory in detail."""
    mb = pv.MultiBlock()
    mb['sphere'] = pv.Sphere()
    html = mb._repr_html_()
    assert 'pts' in html
    assert 'cells' in html
    assert 'pv-child-detail' in html


def test_multiblock_repr_nested():
    """Nested MultiBlocks should show 'N blocks' in the child detail."""
    inner = pv.MultiBlock([pv.Sphere()])
    outer = pv.MultiBlock()
    outer['inner'] = inner
    html = outer._repr_html_()
    assert '1 blocks' in html


def test_multiblock_repr_none_block():
    """None blocks should render without crashing."""
    mb = pv.MultiBlock()
    mb.append(None)
    mb['valid'] = pv.Sphere()
    html = mb._repr_html_()
    assert 'None' in html
    assert 'pv-child-name' in html
    # The valid block should still render
    assert 'PolyData' in html


def test_partitioned_dataset_repr_basic():
    pd = pv.PartitionedDataSet([pv.Sphere(), pv.Cube()])
    html = pd._repr_html_()
    assert 'PartitionedDataSet' in html
    assert 'partitions' in html
    assert 'Partitions' in html
    assert 'pv-wrap' in html


def test_partitioned_dataset_repr_empty():
    pd = pv.PartitionedDataSet()
    html = pd._repr_html_()
    assert 'PartitionedDataSet' in html


def test_table_repr_basic():
    table = pv.Table({'a': np.array([1, 2, 3]), 'b': np.array([4.0, 5.0, 6.0])})
    html = table._repr_html_()
    assert 'Table' in html
    assert 'rows' in html
    assert 'arrays' in html
    assert 'Row Data' in html
    assert 'pv-wrap' in html


def test_table_repr_empty_omits_section():
    """Empty table should not show a Row Data section."""
    table = pv.Table()
    html = table._repr_html_()
    assert 'Table' in html
    assert 'Row Data' not in html


def test_camera_position_repr():
    cp = pv.CameraPosition((1.0, 2.0, 3.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    html = cp._repr_html_()
    assert 'CameraPosition' in html
    assert 'position' in html
    assert 'focal_point' in html
    assert 'viewup' in html
    assert 'pv-copy-btn' in html
    # Verify actual coordinate values appear formatted
    assert '1.0000' in html
    assert '2.0000' in html
    assert '3.0000' in html
