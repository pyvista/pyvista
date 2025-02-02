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


@pytest.fixture
def poly_circle():
    circle = pv.Circle(resolution=30)
    return pv.PolyData(circle.points, lines=[31, *list(range(30)), 0])


def test_decimate_polyline(poly_circle):
    assert poly_circle.n_points == 30
    decimated = poly_circle.decimate_polyline(0.5)
    # Allow some leeway for approximtely 50%
    assert decimated.n_points >= 14
    assert decimated.n_points <= 16


def test_decimate_polyline_maximum_error(poly_circle):
    assert poly_circle.n_points == 30
    # low maximum error will prevent decimation.
    # Since this is a regular shape, no decimation occurs at all with suitable choice
    decimated = poly_circle.decimate_polyline(0.5, maximum_error=0.0001)
    assert decimated.n_points == 30


def test_decimate_polyline_inplace(poly_circle):
    poly_circle.decimate_polyline(0.5, inplace=True)
    # Allow some leeway for approximtely 50%
    assert poly_circle.n_points >= 14
    assert poly_circle.n_points <= 16


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
def oriented_image():
    image = pv.ImageData()
    image.spacing = (1.1, 1.2, 1.3)
    image.dimensions = (10, 11, 12)
    image.direction_matrix = pv.Transform().rotate_vector((4, 5, 6), 30).matrix[:3, :3]
    image['scalars'] = np.ones((image.n_points,))
    return image


@pytest.fixture
def oriented_polydata(oriented_image):
    oriented_poly = oriented_image.pad_image().contour_labels(smoothing=False)
    assert np.allclose(oriented_poly.bounds, oriented_image.points_to_cells().bounds, atol=0.1)
    return oriented_poly


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_voxelize_binary_mask_orientation(oriented_image, oriented_polydata):
    mask = oriented_polydata.voxelize_binary_mask(reference_volume=oriented_image)
    assert mask.bounds == oriented_image.bounds
    mask_as_surface = mask.pad_image().contour_labels(smoothing=False)
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
