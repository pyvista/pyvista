from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.core.errors import MissingDataError


def test_contour_banded_raise(sphere):
    sphere.clear_data()

    with pytest.raises(MissingDataError):
        sphere.contour_banded(5)

    sphere['data'] = sphere.points[:, 2]
    with pytest.raises(ValueError):  # noqa: PT011
        sphere.contour_banded(5, scalar_mode='foo')

    sphere.clear_data()
    sphere['data'] = range(sphere.n_cells)
    with pytest.raises(MissingDataError):
        _ = sphere.contour_banded(10)


def test_contour_banded_points(sphere):
    sphere.clear_data()
    sphere['data'] = sphere.points[:, 2]
    out, edges = sphere.contour_banded(10)
    assert out.n_cells
    assert edges.n_cells
    assert 'data' in out.point_data

    out = sphere.contour_banded(10, generate_contour_edges=False)
    assert out.n_cells

    rng = [-100, 100]
    out = sphere.contour_banded(
        10,
        rng=rng,
        generate_contour_edges=False,
        scalar_mode='index',
        clipping=True,
    )
    assert out['data'].min() <= rng[0]
    assert out['data'].max() >= rng[1]


def test_boolean_intersect_edge_case():
    a = pv.Cube(x_length=2, y_length=2, z_length=2).triangulate()
    b = pv.Cube().triangulate()  # smaller cube (x_length=1)

    with pytest.warns(UserWarning, match='contained within another'):
        a.boolean_intersection(b)


def test_identical_boolean(sphere):
    with pytest.raises(ValueError, match='identical points'):
        sphere.boolean_intersection(sphere.copy())


def test_triangulate_contours():
    poly = pv.Polygon(n_sides=4, fill=False)
    filled = poly.triangulate_contours()
    for cell in filled.cell:
        assert cell.type == pv.CellType.TRIANGLE


@pytest.mark.skipif(
    pv.vtk_version_info < (9, 1, 0),
    reason="Requires VTK>=9.1.0 for a vtkIOChemistry.vtkCMLMoleculeReader",
)
def test_protein_ribbon():
    tgqp = examples.download_3gqp()
    ribbon = tgqp.protein_ribbon()
    assert ribbon.n_cells


@pytest.fixture
def frog_tissues_image():
    return examples.load_frog_tissues()


@pytest.fixture
def frog_tissues_contour(frog_tissues_image):
    return frog_tissues_image.contour_labeled(smoothing=False)


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_voxelize_binary_mask(frog_tissues_image, frog_tissues_contour):
    mask = frog_tissues_contour.voxelize_binary_mask(reference_volume=frog_tissues_image)

    expected_voxels = frog_tissues_image.points_to_cells().threshold(0.5)
    actual_voxels = mask.points_to_cells().threshold(0.5)

    assert expected_voxels.bounds == actual_voxels.bounds
    assert expected_voxels.n_cells == actual_voxels.n_cells


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_voxelize_binary_mask_no_reference(frog_tissues_image, frog_tissues_contour):
    mask = frog_tissues_contour.voxelize_binary_mask()
    assert np.allclose(mask.points_to_cells().bounds, frog_tissues_contour.bounds)


def test_voxelize_binary_mask_dimensions(sphere):
    dims = (10, 11, 12)
    mask = sphere.voxelize_binary_mask(dimensions=dims)
    assert np.allclose(mask.points_to_cells().bounds, sphere.bounds)
    assert mask.dimensions == dims


@pytest.mark.parametrize(
    'rounding_func',
    [np.round, np.ceil, np.floor, lambda x: [np.round(x[0]), np.ceil(x[1]), np.floor(x[2])]],
)
def test_voxelize_binary_mask_rounding_func(sphere, rounding_func):
    spacing = np.array((1.1, 1.2, 1.3))
    mask = sphere.voxelize_binary_mask(spacing=spacing, rounding_func=rounding_func)
    assert np.allclose(mask.points_to_cells().bounds, sphere.bounds)
    if rounding_func == np.round:
        assert np.any(mask.spacing > spacing)
        assert np.any(mask.spacing < spacing)
    elif rounding_func == np.ceil:
        assert np.all(mask.spacing < spacing)
    elif rounding_func == np.floor:
        assert np.all(mask.spacing > spacing)
    else:  # rounding_func == lambda x: [np.round(x[0]), np.ceil(x[1]), np.floor(x[2])]]
        assert mask.spacing[1] < spacing[1]
        assert mask.spacing[2] > spacing[2]


@pytest.mark.parametrize('foreground', [1, 2])
@pytest.mark.parametrize('background', [-1, 0])
def test_voxelize_binary_mask_foreground(sphere, foreground, background):
    mask = sphere.voxelize_binary_mask(foreground_value=foreground, background_value=background)
    unique, counts = np.unique(mask['mask'], return_counts=True)
    assert np.array_equal(unique, [background, foreground])
    assert counts[1] > counts[0]


def test_voxelize_binary_mask_raises(sphere):
    match = 'Spacing and dimensions cannot both be set. Set one or the other.'
    with pytest.raises(TypeError, match=match):
        sphere.voxelize_binary_mask(dimensions=(1, 2, 3), spacing=(4, 5, 6))

    match = 'Spacing and cell length percentile cannot both be set. Set one or the other.'
    with pytest.raises(TypeError, match=match):
        sphere.voxelize_binary_mask(spacing=(4, 5, 6), cell_length_percentile=0.2)

    match = (
        'Cell length percentile and mesh length fraction cannot both be set. Set one or the other.'
    )
    with pytest.raises(TypeError, match=match):
        sphere.voxelize_binary_mask(mesh_length_fraction=1 / 100, cell_length_percentile=0.2)
