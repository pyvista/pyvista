from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.core.errors import MissingDataError
from tests.conftest import flaky_test


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
    reason='Requires VTK>=9.1.0 for a vtkIOChemistry.vtkCMLMoleculeReader',
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
    mask = frog_tissues_contour.voxelize_binary_mask(
        reference_volume=frog_tissues_image, progress_bar=True
    )

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


def test_voxelize_binary_mask_auto_spacing(ant):
    # Test default
    mask_no_input = ant.voxelize_binary_mask()
    if pv.vtk_version_info < (9, 2):
        expected_mask = ant.voxelize_binary_mask(mesh_length_fraction=1 / 100)
    else:
        expected_mask = ant.voxelize_binary_mask(cell_length_percentile=0.1)
    assert mask_no_input.spacing == expected_mask.spacing

    # Test cell length
    if pv.vtk_version_info < (9, 2):
        match = 'Cell length percentile and sample size requires VTK 9.2 or greater.'
        with pytest.raises(TypeError, match=match):
            ant.voxelize_binary_mask(cell_length_percentile=0.2)
    else:
        mask_percentile_20 = ant.voxelize_binary_mask(cell_length_percentile=0.2)
        mask_percentile_50 = ant.voxelize_binary_mask(cell_length_percentile=0.5)
        assert np.all(np.array(mask_percentile_20.spacing) < mask_percentile_50.spacing)

    # Test mesh length
    mask_fraction_200 = ant.voxelize_binary_mask(mesh_length_fraction=1 / 200)
    mask_fraction_500 = ant.voxelize_binary_mask(mesh_length_fraction=1 / 500)
    assert np.all(np.array(mask_fraction_200.spacing) > mask_fraction_500.spacing)
    # Check spacing matches mesh length. Use atol since spacing is approximate.
    assert np.allclose(mask_fraction_500.spacing, ant.length / 500, atol=1e-3)


# This test is flaky because of random sampling that cannot be controlled.
# Sometimes the sampling produces the same output.
# https://github.com/pyvista/pyvista/pull/6728
@flaky_test(times=5)
def test_voxelize_binary_mask_cell_length_sample_size(ant):
    if pv.vtk_version_info < (9, 2):
        match = 'Cell length percentile and sample size requires VTK 9.2 or greater.'
        with pytest.raises(TypeError, match=match):
            ant.voxelize_binary_mask(cell_length_percentile=0.2)
    else:
        mask_samples_1 = ant.voxelize_binary_mask(cell_length_sample_size=100)
        mask_samples_2 = ant.voxelize_binary_mask(cell_length_sample_size=200)
        assert mask_samples_1.spacing != mask_samples_2.spacing

        mask_samples_1 = ant.voxelize_binary_mask(cell_length_sample_size=ant.n_cells)
        mask_samples_2 = ant.voxelize_binary_mask(cell_length_sample_size=ant.n_cells)
        assert mask_samples_1.spacing == mask_samples_2.spacing


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


@pytest.mark.parametrize('foreground', [1, 2.1])
@pytest.mark.parametrize('background', [-1, 0])
def test_voxelize_binary_mask_foreground_background(sphere, foreground, background):
    mask = sphere.voxelize_binary_mask(foreground_value=foreground, background_value=background)
    unique, counts = np.unique(mask['mask'], return_counts=True)
    assert np.array_equal(unique, [background, foreground])
    # Test we have more foreground than background (not always true, but is true for a sphere mesh)
    assert counts[1] > counts[0]

    # Test dtype
    if (
        isinstance(foreground, int)
        and isinstance(background, int)
        and foreground >= 0
        and background >= 0
    ):
        assert mask['mask'].dtype == np.uint8
    elif isinstance(foreground, int) and isinstance(background, int):
        assert mask['mask'].dtype == int
    else:
        assert mask['mask'].dtype == float


@pytest.fixture
def oriented_image():
    image = pv.ImageData()
    image.spacing = (1.1, 1.2, 1.3)
    image.dimensions = (10, 11, 12)
    image.direction_matrix = pv.Transform().rotate_vector((4, 5, 6), 30).matrix[:3, :3]
    image['scalars'] = np.ones((image.n_points,))
    return image


@pytest.fixture
def oriented_polydata(oriented_image):
    oriented_poly = oriented_image.pad_image().contour_labeled(smoothing=False)
    assert np.allclose(oriented_poly.bounds, oriented_image.points_to_cells().bounds, atol=0.1)
    return oriented_poly


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_voxelize_binary_mask_orientation(oriented_image, oriented_polydata):
    mask = oriented_polydata.voxelize_binary_mask(reference_volume=oriented_image)
    assert mask.bounds == oriented_image.bounds
    mask_as_surface = mask.pad_image().contour_labeled(smoothing=False)
    assert mask_as_surface.bounds == oriented_polydata.bounds


def test_voxelize_binary_mask_raises(sphere):
    match = 'Spacing and dimensions cannot both be set. Set one or the other.'
    with pytest.raises(TypeError, match=match):
        sphere.voxelize_binary_mask(dimensions=(1, 2, 3), spacing=(4, 5, 6))

    match = 'Spacing and cell length percentile cannot both be set. Set one or the other.'
    with pytest.raises(TypeError, match=match):
        sphere.voxelize_binary_mask(spacing=(4, 5, 6), cell_length_percentile=0.2)

    match = 'Spacing and mesh length fraction cannot both be set. Set one or the other.'
    with pytest.raises(TypeError, match=match):
        sphere.voxelize_binary_mask(spacing=(4, 5, 6), mesh_length_fraction=0.2)

    match = (
        'Cell length percentile and mesh length fraction cannot both be set. Set one or the other.'
    )
    with pytest.raises(TypeError, match=match):
        sphere.voxelize_binary_mask(mesh_length_fraction=1 / 100, cell_length_percentile=0.2)

    match = 'Rounding func cannot be set when dimensions is specified. Set one or the other.'
    with pytest.raises(TypeError, match=match):
        sphere.voxelize_binary_mask(dimensions=(1, 2, 3), rounding_func=np.round)

    for parameter in [
        'dimensions',
        'spacing',
        'rounding_func',
        'cell_length_percentile',
        'cell_length_sample_size',
        'mesh_length_fraction',
    ]:
        kwargs = {parameter: 0}  # Give parameter any value for test
        match = 'Cannot specify a reference volume with other geometry parameters. `reference_volume` must define the geometry exclusively.'
        with pytest.raises(TypeError, match=match):
            sphere.voxelize_binary_mask(reference_volume=pv.ImageData(), **kwargs)


def test_ruled_surface():
    poly = pv.PolyData(
        [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1]],
        lines=[[2, 0, 1], [2, 2, 3]],
        force_float=False,
    )
    ruled = poly.ruled_surface(resolution=(21, 21))
    assert ruled.n_cells
