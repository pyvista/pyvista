from __future__ import annotations

import functools
import itertools
from pathlib import Path
import re
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from unittest.mock import Mock
from unittest.mock import patch

from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import composite
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import one_of
import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.core import _vtk_core as _vtk
from pyvista.core.celltype import CellType
from pyvista.core.errors import MissingDataError
from pyvista.core.errors import NotAllTrianglesError
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.filters.data_set import _swap_axes
from tests.conftest import flaky_test

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from pyvista.core.pointset import PolyData

normals = ['x', 'y', '-z', (1, 1, 1), (3.3, 5.4, 0.8)]


HYPOTHESIS_MAX_EXAMPLES = 20


@composite
def n_numbers(draw, n):
    numbers = []
    for _ in range(n):
        number = draw(
            one_of(
                floats(),
                integers(max_value=np.iinfo(int).max, min_value=np.iinfo(int).min),
            )
        )
        numbers.append(number)
    return numbers


def aprox_le(a, b, rtol=1e-5, atol=1e-8):
    """Return if that ``a <= b`` within a tolerance.

    See numpy.isclose for the description of ``rtol`` and ``atol``.

    """
    if a < b:
        return True
    else:
        return np.isclose(a, b, rtol, atol)


class GetOutput:
    # Helper class to patch ``pv.core.filters._get_output`` which captures the raw
    # VTK algorithm objects at the time ``_get_output`` is invoked.

    def __init__(self):
        self._mock = Mock()

    def __call__(self, algorithm, *args, **kwargs):
        self._mock(algorithm, *args, **kwargs)
        return pv.core.filters._get_output(algorithm)

    def reset(self, *args, **kwargs):
        self._mock.reset_mock(*args, **kwargs)

    @property
    def latest_algorithm(self):
        return self._mock.call_args_list[-1][0][0]


@pytest.fixture(scope='module')
def uniform_vec():
    nx, ny, nz = 20, 15, 5
    origin = (-(nx - 1) * 0.1 / 2, -(ny - 1) * 0.1 / 2, -(nz - 1) * 0.1 / 2)
    mesh = pv.ImageData(dimensions=(nx, ny, nz), spacing=(0.1, 0.1, 0.1), origin=origin)
    mesh['vectors'] = mesh.points
    return mesh


def test_threshold_raises(mocker: MockerFixture):
    from pyvista.core.filters import data_set

    m = mocker.patch.object(data_set, 'get_array')
    m.return_value = None
    with pytest.raises(ValueError, match=r'No arrays present to threshold.'):
        pv.Sphere().threshold(1.0)


def test_contour_raises(mocker: MockerFixture):
    from pyvista.core.filters import data_set

    m = mocker.patch.object(data_set, 'set_default_active_scalars')
    m().name = 'foo'

    with pytest.raises(
        ValueError, match=r'Input dataset for the contour filter must have scalar.'
    ):
        pv.PolyData().contour()


def test_wrap_by_vector_raises(mocker: MockerFixture):
    from pyvista.core.filters import data_set

    m = mocker.patch.object(data_set, 'get_array')
    m.return_value = None
    with pytest.raises(ValueError, match=r'No vectors present to warp by vector.'):
        pv.Sphere().warp_by_vector()


@given(
    strategy=st.text().filter(lambda x: x not in ['null_value', 'mark_points', 'closest_point'])
)
def test_interpolate_raises(strategy):
    with pytest.raises(ValueError, match=re.escape(f'strategy `{strategy}` not supported.')):
        pv.Sphere().interpolate(pv.Sphere(), strategy=strategy)


def test_datasetfilters_init():
    with pytest.raises(TypeError):
        pv.core.filters.DataSetFilters()


@pytest.mark.skip_mac('Flaky Mac test')
@pytest.mark.parametrize('both', [False, True])
@pytest.mark.parametrize('invert', [False, True])
def test_clip_scalar_filter(datasets, both, invert):
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
                dataset.clip_scalar(
                    scalars='to_clip', value=clip_value, both=False, invert=invert
                ),
            )
            assert len(clps) == 1
            expect_les = (invert,)

        for clp, expect_le in zip(clps, expect_les, strict=True):
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


def test_clip_scalar_no_active(sphere):
    # test no active scalars case
    sphere.point_data.set_array(sphere.points[:, 2], 'data')
    assert sphere.active_scalars_name is None
    clp = sphere.clip_scalar()
    assert clp.n_points < sphere.n_points


def test_clip_scalar_ranges_imagedata():
    mesh = pv.examples.download_whole_body_ct_male()['ct']
    vol = mesh.clip_scalar(
        value=(150, 3000),
    )
    assert vol.n_points < mesh.n_points
    vol2 = mesh.clip_scalar(
        value=150,
    )
    assert vol.n_points < vol2.n_points


def test_clip_scalar_errors():
    mesh = pv.examples.download_whole_body_ct_male()['ct']
    with pytest.raises(TypeError):
        mesh.clip_scalar(value=(150, 3000), inplace=True)
    with pytest.raises(ValueError, match='Cannot have invert=False for a range clip'):
        mesh.clip_scalar(value=(150, 3000), invert=False)
    with pytest.raises(ValueError, match='Cannot have both=True for a range clip'):
        mesh.clip_scalar(value=(150, 3000), both=True)


def test_clip_scalar_multiple():
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


@pytest.mark.parametrize('crinkle', [True, False])
def test_clip_surface_output_type(datasets, crinkle):
    for dataset in datasets:
        clp = dataset.clip_surface(dataset.extract_surface(algorithm=None), crinkle=crinkle)
        assert clp is not None
        if isinstance(dataset, pv.PointSet):
            assert isinstance(clp, pv.PointSet)
        elif isinstance(dataset, pv.PolyData):
            assert isinstance(clp, pv.PolyData)
        elif isinstance(dataset, pv.MultiBlock):
            assert isinstance(clp, pv.MultiBlock)
        else:
            assert isinstance(clp, pv.UnstructuredGrid)


def test_clip_closed_surface():
    closed_surface = pv.Sphere()
    clipped = closed_surface.clip_closed_surface(progress_bar=True)
    assert clipped.n_open_edges == 0
    open_surface = closed_surface.clip(progress_bar=True)
    with pytest.raises(ValueError):  # noqa: PT011
        _ = open_surface.clip_closed_surface()


def test_implicit_distance():
    surface = pv.Cone(
        direction=(0, 0, -1),
        height=3.0,
        radius=1,
        resolution=50,
    )
    xx = yy = zz = 1 - np.linspace(0, 51, 11) * 2 / 50
    dataset = pv.RectilinearGrid(xx, yy, zz)
    res = dataset.compute_implicit_distance(surface)
    assert 'implicit_distance' in res.point_data
    dataset.compute_implicit_distance(surface, inplace=True)
    assert 'implicit_distance' in dataset.point_data


def test_threshold(datasets):
    for dataset in datasets[0:3]:
        thresh = dataset.threshold(progress_bar=True)
        assert thresh is not None
        assert isinstance(thresh, pv.UnstructuredGrid)
    # Test value ranges
    dataset = examples.load_uniform()  # ImageData
    thresh = dataset.threshold(100, invert=False, progress_bar=True)
    assert thresh is not None
    assert isinstance(thresh, pv.UnstructuredGrid)
    thresh = dataset.threshold([100, 500], invert=False, progress_bar=True)
    assert thresh is not None
    assert isinstance(thresh, pv.UnstructuredGrid)
    thresh = dataset.threshold([100, 500], invert=True, progress_bar=True)
    assert thresh is not None
    assert isinstance(thresh, pv.UnstructuredGrid)
    # allow Sequence but not Iterable
    with pytest.raises(TypeError):
        dataset.threshold({100, 500}, progress_bar=True)

    # Now test DataSet without arrays
    dataset = datasets[3]  # polydata with no arrays
    with pytest.raises(ValueError):  # noqa: PT011
        thresh = dataset.threshold()

    dataset = examples.load_uniform()
    with pytest.raises(ValueError):  # noqa: PT011
        dataset.threshold([10, 100, 300], progress_bar=True)

    with pytest.raises(ValueError):  # noqa: PT011
        dataset.threshold(100, method='between')

    with pytest.raises(ValueError):  # noqa: PT011
        dataset.threshold((2, 1))


def test_threshold_all_scalars():
    mesh = pv.Sphere()
    mesh.clear_data()

    mesh['scalar0'] = np.zeros(mesh.n_points)
    mesh['scalar1'] = np.ones(mesh.n_points)
    mesh.set_active_scalars('scalar1')
    thresh_all = mesh.threshold(value=0.5, all_scalars=True)  # only uses scalar1
    assert thresh_all.n_points == mesh.n_points
    assert thresh_all.n_cells == mesh.n_cells

    mesh['scalar1'][0 : int(mesh.n_points / 2)] = 0.0
    thresh = mesh.threshold(value=0.5, all_scalars=False)
    thresh_all = mesh.threshold(value=0.5, all_scalars=True)
    assert thresh_all.n_points < mesh.n_points
    # removes additional cells/points due to all_scalars
    assert thresh_all.n_points < thresh.n_points
    assert thresh_all.n_cells < mesh.n_cells
    assert thresh_all.n_cells < thresh.n_cells

    mesh.clear_data()
    mesh['scalar0'] = np.zeros(mesh.n_cells)
    mesh['scalar1'] = np.ones(mesh.n_cells)
    mesh['scalar1'][0 : int(mesh.n_cells / 2)] = 0.0
    mesh.set_active_scalars('scalar1')
    thresh = mesh.threshold(value=0.5, all_scalars=False)
    thresh_all = mesh.threshold(value=0.5, all_scalars=True)
    # when thresholding by cell data, all_scalars has no effect since it has 1 value per cell
    assert thresh_all.n_points < mesh.n_points
    assert thresh_all.n_points == thresh.n_points
    assert thresh_all.n_cells < mesh.n_cells
    assert thresh_all.n_cells == thresh.n_cells


def test_threshold_multicomponent():
    mesh = pv.Plane()
    data = np.zeros((mesh.n_cells, 3))
    data[0:3, 0] = 1
    data[2:4, 1] = 2
    data[2, 2] = 3
    mesh['data'] = data

    thresh = mesh.threshold(value=0.5, scalars='data', component_mode='component', component=0)
    assert thresh.n_cells == 3
    thresh = mesh.threshold(value=0.5, scalars='data', component_mode='component', component=1)
    assert thresh.n_cells == 2
    thresh = mesh.threshold(value=0.5, scalars='data', component_mode='all')
    assert thresh.n_cells == 1
    thresh = mesh.threshold(value=0.5, scalars='data', component_mode='any')
    assert thresh.n_cells == 4

    with pytest.raises(ValueError):  # noqa: PT011
        mesh.threshold(value=0.5, scalars='data', component_mode='not a mode')

    with pytest.raises(ValueError):  # noqa: PT011
        mesh.threshold(value=0.5, scalars='data', component_mode='component', component=-1)

    with pytest.raises(ValueError):  # noqa: PT011
        mesh.threshold(value=0.5, scalars='data', component_mode='component', component=3)

    with pytest.raises(TypeError):
        mesh.threshold(value=0.5, scalars='data', component_mode='component', component=0.5)


def test_threshold_percent(datasets):
    percents = [25, 50, [18.0, 85.0], [19.0, 80.0], 0.70]
    inverts = [False, True, False, True, False]
    # Only test data sets that have arrays
    for i, dataset in enumerate(datasets[0:3]):
        thresh = dataset.threshold_percent(
            percent=percents[i],
            invert=inverts[i],
            progress_bar=True,
        )
        assert thresh is not None
        assert isinstance(thresh, pv.UnstructuredGrid)
    dataset = examples.load_uniform()
    _ = dataset.threshold_percent(0.75, scalars='Spatial Cell Data', progress_bar=True)
    with pytest.raises(ValueError):  # noqa: PT011
        dataset.threshold_percent(20000)
    with pytest.raises(ValueError):  # noqa: PT011
        dataset.threshold_percent(0.0)
    # allow Sequence but not Iterable
    with pytest.raises(TypeError):
        dataset.threshold_percent({18.0, 85.0})


def test_threshold_paraview_consistency():
    """Validate expected results that match ParaView."""
    x = np.arange(5, dtype=float)
    y = np.arange(6, dtype=float)
    z = np.arange(2, dtype=float)
    xx, yy, zz = np.meshgrid(x, y, z)
    mesh = pv.StructuredGrid(xx, yy, zz)
    mesh.cell_data.set_scalars(np.repeat(range(5), 4))

    # Input mesh
    #   [[0, 0, 0, 0, 1],
    #    [1, 1, 1, 2, 2],
    #    [2, 2, 3, 3, 3],
    #    [3, 4, 4, 4, 4]]

    # upper(0): extract all
    thresh = mesh.threshold(0, invert=False, method='upper')
    assert thresh.n_cells == mesh.n_cells
    assert np.allclose(thresh.active_scalars, mesh.active_scalars)
    # upper(0),invert: extract none
    thresh = mesh.threshold(0, invert=True, method='upper')
    assert thresh.n_cells == 0

    # lower(0)
    #   [[0, 0, 0, 0   ]]
    thresh = mesh.threshold(0, invert=False, method='lower')
    assert thresh.n_cells == 4
    assert np.allclose(thresh.active_scalars, np.array([0, 0, 0, 0]))
    # lower(0),invert
    #   [[            1],
    #    [1, 1, 1, 2, 2],
    #    [2, 2, 3, 3, 3],
    #    [3, 4, 4, 4, 4]]
    thresh = mesh.threshold(0, invert=True, method='lower')
    assert thresh.n_cells == 16
    assert thresh.get_data_range() == (1, 4)

    # upper(2)
    #   [[         2, 2],
    #    [2, 2, 3, 3, 3],
    #    [3, 4, 4, 4, 4]]
    thresh = mesh.threshold(2, invert=False, method='upper')
    assert thresh.n_cells == 12
    assert thresh.get_data_range() == (2, 4)
    # upper(2),invert
    #   [[0, 0, 0, 0, 1],
    #    [1, 1, 1,     ]]
    thresh = mesh.threshold(2, invert=True, method='upper')
    assert thresh.n_cells == 8
    assert thresh.get_data_range() == (0, 1)

    # lower(2)
    #   [[0, 0, 0, 0, 1],
    #    [1, 1, 1, 2, 2],
    #    [2, 2,        ]]
    thresh = mesh.threshold(2, invert=False, method='lower')
    assert thresh.n_cells == 12
    assert thresh.get_data_range() == (0, 2)
    # lower(2),invert
    #   [[      3, 3, 3],
    #    [3, 4, 4, 4, 4]]
    thresh = mesh.threshold(2, invert=True, method='lower')
    assert thresh.n_cells == 8
    assert thresh.get_data_range() == (3, 4)

    # between(0, 0)
    #   [[0, 0, 0, 0   ]]
    thresh = mesh.threshold((0, 0), invert=False)
    assert thresh.n_cells == 4
    assert np.allclose(thresh.active_scalars, np.array([0, 0, 0, 0]))
    # between(0,0),invert
    #   [[            1],
    #    [1, 1, 1, 2, 2],
    #    [2, 2, 3, 3, 3],
    #    [3, 4, 4, 4, 4]]
    thresh = mesh.threshold((0, 0), invert=True)
    assert thresh.n_cells == 16
    assert thresh.get_data_range() == (1, 4)

    # between(2,3)
    #   [[         2, 2],
    #    [2, 2, 3, 3, 3],
    #    [3,           ]]
    thresh = mesh.threshold((2, 3), invert=False)
    assert thresh.n_cells == 8
    assert thresh.get_data_range() == (2, 3)
    # between(2,3),invert
    #   [[0, 0, 0, 0, 1],
    #    [1, 1, 1,     ],
    #    [             ],
    #    [   4, 4, 4, 4]]
    thresh = mesh.threshold((2, 3), invert=True)
    assert thresh.n_cells == 12
    assert thresh.get_data_range() == (0, 4)


def test_outline(datasets):
    for dataset in datasets:
        outline = dataset.outline(progress_bar=True)
        assert outline is not None
        assert isinstance(outline, pv.PolyData)


def test_outline_composite(multiblock_all):
    # Now test composite data structures
    output = multiblock_all.outline(progress_bar=True)
    assert isinstance(output, pv.PolyData)
    output = multiblock_all.outline(nested=True, progress_bar=True)

    # vtk 9.0.0 returns polydata
    assert isinstance(output, (pv.MultiBlock, pv.PolyData))
    if isinstance(output, pv.MultiBlock):
        assert output.n_blocks == multiblock_all.n_blocks


def test_outline_corners(datasets):
    for dataset in datasets:
        outline = dataset.outline_corners(progress_bar=True)
        assert outline is not None
        assert isinstance(outline, pv.PolyData)


def test_outline_corners_composite(multiblock_all):
    # Now test composite data structures
    output = multiblock_all.outline_corners(progress_bar=True)
    assert isinstance(output, pv.PolyData)
    output = multiblock_all.outline_corners(nested=True)
    assert output.n_blocks == multiblock_all.n_blocks


def test_gaussian_splatting(sphere: PolyData):
    output = sphere.gaussian_splatting(progress_bar=True)
    assert output is not None
    assert isinstance(output, pv.ImageData)
    assert output.dimensions == (50, 50, 50)

    dimensions = (10, 11, 12)
    output = sphere.gaussian_splatting(dimensions=dimensions)
    assert output.dimensions == dimensions


def test_extract_geometry(datasets, multiblock_all):
    for dataset in datasets:
        with pytest.warns(pv.PyVistaDeprecationWarning):
            geom = dataset.extract_geometry(progress_bar=True)
        assert geom is not None
        assert isinstance(geom, pv.PolyData)
    # Now test composite data structures
    with pytest.warns(pv.PyVistaDeprecationWarning):
        output = multiblock_all.extract_geometry()
    assert isinstance(output, pv.PolyData)


def test_extract_geometry_extent(uniform):
    match = '`extract_geometry` is deprecated. Use `extract_surface(algorithm=None)` instead.'
    with pytest.warns(pv.PyVistaDeprecationWarning, match=re.escape(match)):
        geom = uniform.extract_geometry(extent=(0, 5, 0, 100, 0, 100))
    assert isinstance(geom, pv.PolyData)
    assert geom.bounds == (0.0, 5.0, 0.0, 9.0, 0.0, 9.0)


def test_delaunay_2d_unstructured():
    mesh = examples.load_hexbeam().delaunay_2d(progress_bar=True)  # UnstructuredGrid
    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_points
    assert len(mesh.point_data.keys()) > 0


@pytest.mark.parametrize('method', ['contour', 'marching_cubes', 'flying_edges'])
def test_contour(uniform, method):
    iso = uniform.contour(method=method, progress_bar=True)
    assert iso is not None
    iso = uniform.contour(isosurfaces=[100, 300, 500], method=method, progress_bar=True)
    assert iso is not None

    # ensure filter can work with non-string inputs
    iso_new_scalars = uniform.contour(
        isosurfaces=[100, 300, 500],
        scalars=range(uniform.n_points),
        method=method,
    )

    assert 'Contour Data' in iso_new_scalars.point_data


def test_contour_errors(uniform, airplane):
    with pytest.raises(TypeError):
        uniform.contour(scalars='Spatial Cell Data')
    with pytest.raises(TypeError):
        uniform.contour(isosurfaces=pv.PolyData())
    with pytest.raises(TypeError):
        uniform.contour(isosurfaces={100, 300, 500})
    with pytest.raises(TypeError):
        uniform.contour(rng={})
    match = 'rng has shape (1,) which is not allowed. Shape must be 2.'
    with pytest.raises(ValueError, match=re.escape(match)):
        uniform.contour(rng=[1])
    match = 'rng with 2 elements must be sorted in ascending order. Got:\n    array([2, 1])'
    with pytest.raises(ValueError, match=re.escape(match)):
        uniform.contour(rng=[2, 1])

    with pytest.raises(ValueError):  # noqa: PT011
        airplane.contour()
    with pytest.raises(ValueError):  # noqa: PT011
        airplane.contour(method='invalid method')
    with pytest.raises(TypeError, match='Invalid type for `scalars`'):
        airplane.contour(scalars=1)
    match = 'No data available.'
    with pytest.raises(ValueError, match=match):
        airplane.contour(rng={})


def test_texture_map_to_plane():
    dataset = examples.load_airplane()
    # Automatically decide plane
    out = dataset.texture_map_to_plane(inplace=False, progress_bar=True)
    assert isinstance(out, type(dataset))
    # Define the plane explicitly
    bnds = dataset.bounds
    origin = bnds[0::2]
    point_u = (bnds.x_max, bnds.y_max, bnds.z_min)
    point_v = (bnds.x_min, bnds.y_min, bnds.z_min)
    out = dataset.texture_map_to_plane(
        origin=origin,
        point_u=point_u,
        point_v=point_v,
        progress_bar=True,
    )
    assert isinstance(out, type(dataset))
    assert 'Texture Coordinates' in out.array_names
    # FINAL: Test in place modifiacation
    dataset.texture_map_to_plane(inplace=True)
    assert 'Texture Coordinates' in dataset.array_names


def test_texture_map_to_sphere():
    dataset = pv.Sphere(radius=1.0)
    # Automatically decide plane
    out = dataset.texture_map_to_sphere(inplace=False, prevent_seam=False, progress_bar=True)
    assert isinstance(out, type(dataset))
    # Define the center explicitly
    out = dataset.texture_map_to_sphere(
        center=(0.1, 0.0, 0.0),
        prevent_seam=True,
        progress_bar=True,
    )
    assert isinstance(out, type(dataset))
    assert 'Texture Coordinates' in out.array_names
    # FINAL: Test in place modifiacation
    dataset.texture_map_to_sphere(inplace=True, progress_bar=True)
    assert 'Texture Coordinates' in dataset.array_names


def test_glyph(datasets, sphere):
    for dataset in datasets:
        dataset['vectors'] = np.ones_like(dataset.points)
        result = dataset.glyph(progress_bar=True)
        assert result is not None
        assert isinstance(result, pv.PolyData)
    # Test different options for glyph filter
    sphere_sans_arrays = sphere.copy()
    sphere.compute_normals(inplace=True)
    sphere['vectors'] = np.ones([sphere.n_points, 3])
    sphere.set_active_vectors('vectors')
    sphere.point_data['arr'] = np.ones(sphere.n_points)

    assert sphere.glyph(scale=False, progress_bar=True)
    assert sphere.glyph(scale='arr', progress_bar=True)
    assert sphere.glyph(scale='arr', orient='Normals', factor=0.1, progress_bar=True)
    assert sphere.glyph(
        scale='arr', orient='Normals', factor=0.1, tolerance=0.1, progress_bar=True
    )
    assert sphere.glyph(
        scale='arr',
        orient='Normals',
        factor=0.1,
        tolerance=0.1,
        clamping=False,
        rng=[1, 1],
        progress_bar=True,
    )
    # passing one or more custom glyphs; many cases for full coverage
    geoms = [
        pv.Sphere(theta_resolution=5, phi_resolution=5),
        pv.Arrow(tip_resolution=5, shaft_resolution=5),
        pv.ParametricSuperToroid(u_res=10, v_res=10, w_res=10),
    ]
    indices = range(len(geoms))
    assert sphere.glyph(geom=geoms[0], progress_bar=True)
    assert sphere.glyph(geom=geoms, indices=indices, rng=(0, len(geoms)), progress_bar=True)
    assert sphere.glyph(geom=geoms, progress_bar=True)
    assert sphere.glyph(
        geom=geoms,
        scale='arr',
        orient='Normals',
        factor=0.1,
        tolerance=0.1,
        progress_bar=True,
    )
    assert sphere.glyph(geom=geoms[:1], indices=[None], progress_bar=True)

    # tries to orient but no orientation vector available
    with pytest.warns(UserWarning, match=r'No vector-like data to use for orient'):
        assert sphere_sans_arrays.glyph(geom=geoms, scale=False, progress_bar=True)

    sphere_sans_arrays['vec1'] = np.ones((sphere_sans_arrays.n_points, 3))
    sphere_sans_arrays['vec2'] = np.ones((sphere_sans_arrays.n_points, 3))
    # tries to orient but multiple orientation vectors are possible
    with pytest.warns(
        UserWarning, match=r'It is unclear which one to use. orient will be set to False'
    ):
        assert sphere_sans_arrays.glyph(geom=geoms, scale=False, progress_bar=True)

    with pytest.raises(TypeError):
        # wrong type for the glyph
        sphere.glyph(geom=pv.StructuredGrid())
    with pytest.raises(TypeError):
        # wrong type for the indices
        sphere.glyph(geom=geoms, indices=set(indices))
    with pytest.raises(ValueError):  # noqa: PT011
        # wrong length for the indices
        sphere.glyph(geom=geoms, indices=indices[:-1])


def test_glyph_warns_ambiguous_data(sphere):
    sphere.compute_normals(inplace=True)
    with pytest.warns(UserWarning, match='It is unclear which one to use') as warning_info:
        sphere.glyph(scale=True)
    # Check that at least one of the expected warnings is raised
    warning_messages = [str(w.message) for w in warning_info]
    assert any(
        'It is unclear which one to use. scale will be set to False' in msg
        for msg in warning_messages
    )


def test_glyph_cell_point_data(sphere):
    sphere['vectors_cell'] = np.ones([sphere.n_cells, 3])
    sphere['vectors_points'] = np.ones([sphere.n_points, 3])
    sphere['arr_cell'] = np.ones(sphere.n_cells)
    sphere['arr_points'] = np.ones(sphere.n_points)

    assert sphere.glyph(orient='vectors_cell', scale='arr_cell', progress_bar=True)
    assert sphere.glyph(orient='vectors_points', scale='arr_points', progress_bar=True)
    with pytest.raises(ValueError):  # noqa: PT011
        sphere.glyph(orient='vectors_cell', scale='arr_points', progress_bar=True)
    with pytest.raises(ValueError):  # noqa: PT011
        sphere.glyph(orient='vectors_points', scale='arr_cell', progress_bar=True)


class InterrogateVTKGlyph3D:
    def __init__(self, alg: _vtk.vtkGlyph3D):
        self.alg = alg

    @property
    def input_data_object(self):
        return pv.wrap(self.alg.GetInputDataObject(0, 0))

    @property
    def input_active_scalars_info(self):
        return self.input_data_object.active_scalars_info

    @property
    def input_active_vectors_info(self):
        return self.input_data_object.active_vectors_info

    @property
    def scaling(self):
        return self.alg.GetScaling()

    @property
    def scale_mode(self):
        return self.alg.GetScaleModeAsString()

    @property
    def scale_factor(self):
        return self.alg.GetScaleFactor()

    @property
    def clamping(self):
        return self.alg.GetClamping()

    @property
    def vector_mode(self):
        return self.alg.GetVectorModeAsString()


def test_glyph_settings(sphere):
    sphere['vectors_cell'] = np.ones([sphere.n_cells, 3])
    sphere['vectors_points'] = np.ones([sphere.n_points, 3])
    sphere['arr_cell'] = np.ones(sphere.n_cells)
    sphere['arr_points'] = np.ones(sphere.n_points)

    sphere['arr_both'] = np.ones(sphere.n_points)
    sphere['arr_both'] = np.ones(sphere.n_cells)
    sphere['vectors_both'] = np.ones([sphere.n_points, 3])
    sphere['vectors_both'] = np.ones([sphere.n_cells, 3])

    sphere['active_arr_points'] = np.ones(sphere.n_points)
    sphere['active_vectors_points'] = np.ones([sphere.n_points, 3])

    with patch('pyvista.core.filters.data_set._get_output', GetOutput()) as go:
        # no orient with cell scale
        sphere.glyph(scale='arr_cell', orient=False)
        alg = InterrogateVTKGlyph3D(go.latest_algorithm)
        assert alg.input_active_scalars_info.name == 'arr_cell'
        assert alg.scale_mode == 'ScaleByScalar'
        go.reset()

        # cell orient with no scale
        sphere.glyph(scale=False, orient='vectors_cell')
        alg = InterrogateVTKGlyph3D(go.latest_algorithm)
        assert alg.input_active_vectors_info.name == 'vectors_cell'
        assert alg.scale_mode == 'DataScalingOff'
        go.reset()

        # cell orient with cell scale
        sphere.glyph(scale='arr_cell', orient='vectors_cell')
        alg = InterrogateVTKGlyph3D(go.latest_algorithm)
        assert alg.input_active_scalars_info.name == 'arr_cell'
        assert alg.input_active_vectors_info.name == 'vectors_cell'
        assert alg.scale_mode == 'ScaleByScalar'
        go.reset()

        # cell orient with cell scale and tolerance
        sphere.glyph(scale='arr_cell', orient='vectors_cell', tolerance=0.05)
        alg = InterrogateVTKGlyph3D(go.latest_algorithm)
        assert alg.input_active_scalars_info.name == 'arr_cell'
        assert alg.input_active_vectors_info.name == 'vectors_cell'
        assert alg.scale_mode == 'ScaleByScalar'
        go.reset()

        # no orient with point scale
        sphere.glyph(scale='arr_points', orient=False)
        alg = InterrogateVTKGlyph3D(go.latest_algorithm)
        assert alg.input_active_scalars_info.name == 'arr_points'
        assert alg.scale_mode == 'ScaleByScalar'
        go.reset()

        # point orient with no scale
        sphere.glyph(scale=False, orient='vectors_points')
        alg = InterrogateVTKGlyph3D(go.latest_algorithm)
        assert alg.input_active_vectors_info.name == 'vectors_points'
        assert alg.scale_mode == 'DataScalingOff'
        go.reset()

        # point orient with point scale
        sphere.glyph(scale='arr_points', orient='vectors_points')
        alg = InterrogateVTKGlyph3D(go.latest_algorithm)
        assert alg.input_active_scalars_info.name == 'arr_points'
        assert alg.input_active_vectors_info.name == 'vectors_points'
        assert alg.scale_mode == 'ScaleByScalar'
        go.reset()

        # point orient with point scale and tolerance
        sphere.glyph(scale='arr_points', orient='vectors_points', tolerance=0.05)
        alg = InterrogateVTKGlyph3D(go.latest_algorithm)
        assert alg.input_active_scalars_info.name == 'arr_points'
        assert alg.input_active_vectors_info.name == 'vectors_points'
        assert alg.scale_mode == 'ScaleByScalar'
        go.reset()

        # point orient with point scale + factor
        sphere.glyph(scale='arr_points', orient='vectors_points', factor=5)
        alg = InterrogateVTKGlyph3D(go.latest_algorithm)
        assert alg.input_active_scalars_info.name == 'arr_points'
        assert alg.input_active_vectors_info.name == 'vectors_points'
        assert alg.scale_factor == 5

        # ambiguous point/cell prefers points
        sphere.glyph(scale='arr_both', orient='vectors_both')
        alg = InterrogateVTKGlyph3D(go.latest_algorithm)
        assert alg.input_active_scalars_info.name == 'arr_both'
        assert alg.input_active_vectors_info.name == 'vectors_both'
        # Test the length of the field and not the FieldAssociation
        # because the vtkGlyph3D filter takes POINT data
        assert len(alg.input_data_object.active_scalars) == sphere.n_cells
        assert len(alg.input_data_object.active_scalars) == sphere.n_cells

        # no fields selected uses active
        sphere.set_active_scalars('active_arr_points')
        sphere.set_active_vectors('active_vectors_points')
        sphere.glyph(scale=True, orient=True)
        alg = InterrogateVTKGlyph3D(go.latest_algorithm)
        assert alg.input_active_scalars_info.name == 'active_arr_points'
        assert alg.input_active_vectors_info.name == 'active_vectors_points'


def test_glyph_orient_and_scale():
    grid = pv.ImageData(dimensions=(1, 1, 1))
    geom = pv.Line()
    scale = 10.0
    orient = np.array([[0.0, 0.0, 1.0]])
    grid['z_axis'] = orient * scale
    glyph1 = grid.glyph(geom=geom, orient='z_axis', scale='z_axis')
    glyph2 = grid.glyph(geom=geom, orient=False, scale='z_axis')
    glyph3 = grid.glyph(geom=geom, orient='z_axis', scale=False)
    glyph4 = grid.glyph(geom=geom, orient=False, scale=False)
    assert glyph1.bounds.z_min == geom.bounds.x_min * scale
    assert glyph1.bounds.z_max == geom.bounds.x_max * scale
    assert glyph2.bounds.x_min == geom.bounds.x_min * scale
    assert glyph2.bounds.x_max == geom.bounds.x_max * scale
    assert glyph3.bounds.z_min == geom.bounds.x_min
    assert glyph3.bounds.z_max == geom.bounds.x_max
    assert glyph4.bounds.x_min == geom.bounds.x_min
    assert glyph4.bounds.x_max == geom.bounds.x_max


@pytest.mark.parametrize('color_mode', ['scale', 'scalar', 'vector'])
def test_glyph_color_mode(sphere, color_mode):
    # define vector data
    sphere.point_data['velocity'] = sphere.points[:, [1, 0, 2]] * [-1, 1, 0]
    sphere.glyph(color_mode=color_mode)


def test_glyph_raises(sphere):
    with pytest.raises(ValueError, match="Invalid color mode 'foo'"):
        sphere.glyph(color_mode='foo', scale=False, orient=False)


@pytest.fixture
def connected_datasets():
    # This is similar to the datasets fixture, but the PolyData is fully connected
    return [
        examples.load_uniform(),  # ImageData
        examples.load_rectilinear(),  # RectilinearGrid
        examples.load_hexbeam(),  # UnstructuredGrid
        pv.Sphere(),  # PolyData
        examples.load_structured(),  # StructuredGrid
    ]


@pytest.fixture
def foot_bones() -> pv.PolyData:
    return examples.download_foot_bones()


@pytest.fixture
def connected_datasets_single_disconnected_cell(connected_datasets):
    # Create datasets of MultiBlocks with either 'point' or 'cell' scalar
    # data, where a single cell (or its points) has a completely different
    # scalar value to all other cells in the dataset
    for i, dataset in enumerate(connected_datasets):
        dataset.clear_data()
        dataset_composite = pv.MultiBlock()
        single_cell_id = dataset.n_cells - 1
        single_cell_point_ids = dataset.get_cell(single_cell_id).point_ids
        for association in ['point', 'cell']:
            # Add copy as block
            dataset_copy = dataset.copy()
            dataset_composite[association] = dataset_copy

            # Make scalar data such that values for a single cell or its points
            # fall outside of the extraction range

            if association == 'point':
                num_scalars = dataset_copy.n_points
                node_dataset = dataset_copy.point_data
                ids = single_cell_point_ids
            else:
                num_scalars = dataset_copy.n_cells
                node_dataset = dataset_copy.cell_data
                ids = single_cell_id

            # Assign non-zero floating point scalar data with range [-10, 10]
            # which is intended to be extracted later
            scalar_data = np.linspace(-10, 10, num=num_scalars)
            node_dataset['data'] = scalar_data

            # Assign a totally different scalar value to the single node
            node_dataset['data'][ids] = -1000.0

        connected_datasets[i] = dataset_composite
    return connected_datasets


@pytest.mark.parametrize('dataset_index', list(range(5)))
@pytest.mark.parametrize(
    'extraction_mode',
    ['all', 'largest', 'specified', 'cell_seed', 'point_seed', 'closest'],
)
@pytest.mark.parametrize('label_regions', [True, False])
@pytest.mark.parametrize('scalar_range', [True, False])
def test_connectivity_inplace_and_output_type(
    datasets,
    dataset_index,
    extraction_mode,
    label_regions,
    scalar_range,
):
    # parameterize with label_regions and scalar_range as these parameters
    # have branches which may modify input/input type
    dataset = datasets[dataset_index]

    # ensure we have scalars and set a restricted range
    if scalar_range:
        if len(dataset.array_names) == 0:
            dataset.point_data['data'] = np.arange(0, dataset.n_points)
        pv.set_default_active_scalars(dataset)
        scalar_range = [np.mean(dataset.active_scalars), np.max(dataset.active_scalars)]
    else:
        scalar_range = None

    common_args = dict(
        extraction_mode=extraction_mode,
        point_ids=0,
        cell_ids=0,
        region_ids=0,
        closest_point=(0, 0, 0),
        label_regions=label_regions,
        scalar_range=scalar_range,
    )
    conn = dataset.connectivity(inplace=False, **common_args)
    assert conn is not dataset

    conn = dataset.connectivity(inplace=True, **common_args)
    if isinstance(dataset, (pv.UnstructuredGrid, pv.PolyData)):
        assert conn is dataset
    else:
        assert conn is not dataset

    # test correct output type
    if isinstance(dataset, pv.PolyData):
        assert isinstance(conn, pv.PolyData)
    else:
        assert isinstance(conn, pv.UnstructuredGrid)


@pytest.mark.parametrize('dataset_index', list(range(5)))
@pytest.mark.parametrize(
    'extraction_mode',
    ['all', 'largest', 'specified', 'cell_seed', 'point_seed', 'closest'],
)
def test_connectivity_label_regions(datasets, dataset_index, extraction_mode):
    # the connectivity filter is known to output incorrectly sized scalars
    # test all modes and datasets for correct scalar size
    dataset = datasets[dataset_index]
    common_args = dict(
        extraction_mode=extraction_mode,
        point_ids=0,
        cell_ids=0,
        region_ids=0,
        closest_point=(0, 0, 0),
    )
    conn = dataset.connectivity(**common_args, label_regions=True)
    assert 'RegionId' in conn.point_data.keys()
    assert 'RegionId' in conn.cell_data.keys()

    expected_cell_scalars_size = conn.n_cells
    actual_cell_scalars_size = conn.cell_data['RegionId'].size
    assert expected_cell_scalars_size == actual_cell_scalars_size

    expected_point_scalars_size = conn.n_points
    actual_point_scalars_size = conn.point_data['RegionId'].size
    assert expected_point_scalars_size == actual_point_scalars_size

    # test again but without labels
    active_scalars_info = dataset.active_scalars_info
    conn = dataset.connectivity(**common_args, label_regions=False)
    assert 'RegionId' not in conn.point_data.keys()
    assert 'RegionId' not in conn.cell_data.keys()

    assert conn.n_cells == expected_cell_scalars_size
    assert conn.n_points == expected_point_scalars_size

    # test previously active scalars are restored
    assert conn.active_scalars_info[0] == active_scalars_info[0]
    assert conn.active_scalars_info[1] == active_scalars_info[1]


def test_connectivity_raises(
    connected_datasets_single_disconnected_cell,
):
    dataset: pv.DataSet = connected_datasets_single_disconnected_cell[0]['point']

    with pytest.raises(TypeError, match='Scalar range must be'):
        dataset.connectivity(scalar_range=dataset)

    with pytest.raises(ValueError, match='Scalar range must have two elements'):
        dataset.connectivity(scalar_range=[1, 2, 3])

    with pytest.raises(ValueError, match='Scalar range must have two elements'):
        dataset.connectivity(scalar_range=np.array([[1, 2], [3, 4]]))

    with pytest.raises(ValueError, match='Lower value'):
        dataset.connectivity(scalar_range=[1, 0])

    with pytest.raises(ValueError, match='Invalid value for `extraction_mode`'):
        dataset.connectivity(extraction_mode='foo')

    with pytest.raises(ValueError, match='`closest_point` must be specified'):
        dataset.connectivity(extraction_mode='closest')

    with pytest.raises(ValueError, match='`point_ids` must be specified'):
        dataset.connectivity(extraction_mode='point_seed')

    with pytest.raises(ValueError, match='`cell_ids` must be specified'):
        dataset.connectivity(extraction_mode='cell_seed')

    with pytest.raises(ValueError, match='`region_ids` must be specified'):
        dataset.connectivity(extraction_mode='specified')

    with pytest.raises(ValueError, match='positive integer values'):
        dataset.connectivity(extraction_mode='cell_seed', cell_ids=[-1, 2])

    match = re.escape(
        "Invalid `region_assignment_mode` 'bar' . Must be in ['ascending', 'descending', 'unspecified']"  # noqa: E501
    )
    with pytest.raises(ValueError, match=match):
        dataset.connectivity(extraction_mode='all', region_assignment_mode='bar')


@pytest.mark.parametrize('dataset_index', list(range(5)))
@pytest.mark.parametrize(
    'extraction_mode',
    ['all', 'largest', 'specified', 'cell_seed', 'point_seed', 'closest'],
)
@pytest.mark.parametrize('association', ['cell', 'point'])
def test_connectivity_scalar_range(
    connected_datasets_single_disconnected_cell,
    dataset_index,
    extraction_mode,
    association,
):
    dataset = connected_datasets_single_disconnected_cell[dataset_index][association]

    common_args = dict(
        extraction_mode=extraction_mode,
        point_ids=dataset.get_cell(0).point_ids[0],
        cell_ids=0,
        region_ids=0,
        closest_point=dataset.get_cell(0).points[0],
        label_regions=True,
    )

    # test a single cell is removed
    conn_no_range = dataset.connectivity(**common_args)
    conn_with_range = dataset.connectivity(**common_args, scalar_range=[-10, 10])
    assert conn_with_range.n_cells == conn_no_range.n_cells - 1

    # test no cells are removed
    conn_with_full_range = dataset.connectivity(
        **common_args,
        scalar_range=dataset.get_data_range(),
    )
    assert conn_with_full_range.n_cells == dataset.n_cells

    # test input scalars are passed to output
    assert len(conn_no_range.array_names) == 3  # ['data', 'RegionId', 'RegionId']
    assert len(conn_with_range.array_names) == 3
    assert len(conn_with_full_range.array_names) == 3


@pytest.mark.parametrize('region_assignment_mode', ['ascending', 'descending', 'unspecified'])
def test_connectivity_all(foot_bones: pv.PolyData, region_assignment_mode: str):
    conn = foot_bones.connectivity('all')
    assert conn.n_cells == foot_bones.n_cells

    # test correct labels
    conn = foot_bones.connectivity(
        'all',
        label_regions=True,
        region_assignment_mode=region_assignment_mode,
    )
    region_ids, counts = np.unique(conn.cell_data['RegionId'], return_counts=True)
    assert np.array_equal(region_ids, list(range(26)))

    n_cells = [
        598,
        586,
        392,
        360,
        228,
        212,
        154,
        146,
        146,
        146,
        134,
        134,
        134,
        126,
        124,
        74,
        66,
        60,
        60,
        60,
        48,
        46,
        46,
        46,
        46,
        32,
    ]

    if region_assignment_mode != 'unspecified':
        assert counts.tolist() == (
            n_cells[::-1] if region_assignment_mode == 'ascending' else n_cells
        )


def test_connectivity_largest(foot_bones):
    conn = foot_bones.connectivity('largest')
    assert conn.n_cells == 598

    # test correct labels
    conn = foot_bones.connectivity('largest', label_regions=True)
    region_ids, counts = np.unique(conn.cell_data['RegionId'], return_counts=True)
    assert region_ids == [0]
    assert counts == [598]


@pytest.mark.parametrize('region_assignment_mode', ['ascending', 'descending'])
def test_connectivity_specified(foot_bones: pv.PolyData, region_assignment_mode: str):
    # test all regions
    all_regions = list(range(26))
    conn = foot_bones.connectivity(
        'specified',
        region_ids=all_regions,
        region_assignment_mode=region_assignment_mode,
    )
    assert conn.n_cells == foot_bones.n_cells

    # test irrelevant region IDs
    test_regions = [*all_regions, 77, 99]
    conn = foot_bones.connectivity(
        'specified',
        test_regions,
        region_assignment_mode=region_assignment_mode,
    )
    assert conn.n_cells == foot_bones.n_cells

    # test some regions
    some_regions = [1, 2, 4, 5] if region_assignment_mode == 'descending' else [1, 5, 6, 10]
    expected_n_cells = (
        (586 + 392 + 228 + 212) if region_assignment_mode == 'descending' else (46 + 48 + 60 + 74)
    )
    conn = foot_bones.connectivity(
        'specified',
        some_regions,
        region_assignment_mode=region_assignment_mode,
    )
    assert conn.n_cells == expected_n_cells

    # test correct labels
    conn = foot_bones.connectivity(
        'specified',
        some_regions,
        label_regions=True,
        region_assignment_mode=region_assignment_mode,
    )
    region_ids = np.unique(conn.cell_data['RegionId'])
    assert np.array_equal(region_ids, [0, 1, 2, 3])

    n_cells = [
        conn.threshold([i] * 2, scalars='RegionId', preference='cell').n_cells for i in region_ids
    ]
    assert n_cells == (
        [586, 392, 228, 212] if region_assignment_mode == 'descending' else [46, 48, 60, 74]
    )


def test_connectivity_specified_warning(foot_bones: pv.PolyData):
    match = re.escape(
        'Using the `unspecified` region assignment mode with the `specified` extraction mode can be unintuitive. Ignore this warning if this was intentional'  # noqa: E501
    )
    with pytest.warns(UserWarning, match=match):
        foot_bones.connectivity('specified', region_assignment_mode='unspecified', region_ids=[0])


@pytest.mark.parametrize('dataset_index', list(range(5)))
def test_connectivity_specified_returns_empty(connected_datasets, dataset_index):
    dataset = connected_datasets[dataset_index]
    unused_region_id = 1
    conn = dataset.connectivity('specified', unused_region_id)
    assert conn.n_cells == 0
    assert conn.n_points == 0


def test_connectivity_point_seed(foot_bones):
    conn = foot_bones.connectivity('point_seed', point_ids=1598)
    assert conn.n_cells == 598
    conn = foot_bones.connectivity('point_seed', [1326, 1598])
    assert conn.n_cells == 598 + 360
    assert conn.n_points == 301 + 182

    # test correct labels
    conn = foot_bones.connectivity('point_seed', [1326, 1598], label_regions=True)
    region_ids, counts = np.unique(conn.cell_data['RegionId'], return_counts=True)
    assert np.array_equal(region_ids, [0, 1])
    assert np.array_equal(counts, [598, 360])


def test_connectivity_cell_seed(foot_bones):
    conn = foot_bones.connectivity('cell_seed', cell_ids=3122)
    assert conn.n_cells == 598
    assert conn.n_points == 301
    conn = foot_bones.connectivity('cell_seed', [2588, 3122])
    assert conn.n_cells == 598 + 360
    assert conn.n_points == 301 + 182

    # test correct labels
    conn = foot_bones.connectivity('point_seed', [1326, 1598], label_regions=True)
    region_ids, counts = np.unique(conn.cell_data['RegionId'], return_counts=True)
    assert np.array_equal(region_ids, [0, 1])
    assert np.array_equal(counts, [598, 360])


def test_connectivity_closest_point(foot_bones):
    conn = foot_bones.connectivity('closest', closest_point=(-3.5, -0.5, -0.5))
    assert conn.n_cells == 598
    assert conn.n_points == 301
    conn = foot_bones.connectivity('closest', (-1.5, -0.5, 0.05))
    assert conn.n_cells == 360
    assert conn.n_points == 182

    # test correct labels
    conn = foot_bones.connectivity('closest', (-3.5, -0.5, -0.5), label_regions=True)
    region_ids, counts = np.unique(conn.cell_data['RegionId'], return_counts=True)
    assert region_ids == [0]
    assert counts == [598]


def test_split_bodies():
    # Load a simple example mesh
    dataset = examples.load_uniform()
    dataset.set_active_scalars('Spatial Cell Data')
    threshed = dataset.threshold_percent([0.15, 0.50], invert=True, progress_bar=True)

    bodies = threshed.split_bodies(progress_bar=True)

    volumes = [518.0, 35.0]
    assert len(volumes) == bodies.n_blocks
    for i, body in enumerate(bodies):
        assert np.allclose(body.volume, volumes[i], rtol=0.1)


def test_warp_by_scalar():
    data = examples.load_uniform()
    warped = data.warp_by_scalar(progress_bar=True)
    assert data.n_points == warped.n_points
    warped = data.warp_by_scalar(scale_factor=3, progress_bar=True)
    assert data.n_points == warped.n_points
    warped = data.warp_by_scalar(normal=[1, 1, 3], progress_bar=True)
    assert data.n_points == warped.n_points
    # Test in place!
    foo = examples.load_hexbeam()
    foo.point_data.active_scalars_name = 'sample_point_scalars'
    warped = foo.warp_by_scalar(progress_bar=True)
    foo.warp_by_scalar(inplace=True, progress_bar=True)
    assert np.allclose(foo.points, warped.points)


def test_warp_by_vector():
    # Test when inplace=False (default)
    data = examples.load_sphere_vectors()
    warped = data.warp_by_vector(progress_bar=True)
    assert data.n_points == warped.n_points
    assert not np.allclose(data.points, warped.points)
    warped = data.warp_by_vector(factor=3, progress_bar=True)
    assert data.n_points == warped.n_points
    assert not np.allclose(data.points, warped.points)
    # Test when inplace=True
    foo = examples.load_sphere_vectors()
    warped = foo.warp_by_vector(progress_bar=True)
    foo.warp_by_vector(inplace=True, progress_bar=True)
    assert np.allclose(foo.points, warped.points)


def test_invalid_warp_scalar(sphere):
    sphere['cellscalars'] = np.random.default_rng().random(sphere.n_cells)
    sphere.point_data.clear()
    with pytest.raises(TypeError):
        sphere.warp_by_scalar()


def test_invalid_warp_scalar_inplace(uniform):
    with pytest.raises(TypeError):
        uniform.warp_by_scalar(inplace=True, progress_bar=True)


def test_invalid_warp_vector(sphere):
    # bad vectors
    sphere.point_data['Normals'] = np.empty((sphere.n_points, 2))
    with pytest.raises(ValueError):  # noqa: PT011
        sphere.warp_by_vector('Normals')

    # no vectors
    sphere.point_data.clear()
    with pytest.raises(ValueError):  # noqa: PT011
        sphere.warp_by_vector()


def test_delaunay_3d():
    data = examples.load_uniform().threshold_percent(30, progress_bar=True)
    result = data.delaunay_3d()
    assert np.any(result.points)


@pytest.mark.needs_vtk_version(9, 3)
def test_smooth(uniform):
    surf = uniform.extract_surface(algorithm=None).clean()
    smoothed = surf.smooth()

    # expect mesh is smoothed, raising mean curvature since it is more "spherelike"
    assert smoothed.triangulate().curvature().mean() > surf.triangulate().curvature().mean()

    smooth_inplace = surf.smooth(inplace=True)
    assert np.allclose(surf.points, smoothed.points)
    assert np.allclose(smooth_inplace.points, smoothed.points)


@pytest.mark.needs_vtk_version(9, 3)
def test_smooth_taubin(uniform):
    surf = uniform.extract_surface(algorithm=None).clean()
    smoothed = surf.smooth_taubin()

    # expect mesh is smoothed, raising mean curvature since it is more "spherelike"
    assert smoothed.triangulate().curvature().mean() > surf.triangulate().curvature().mean()

    # while volume is maintained
    assert np.isclose(smoothed.volume, surf.volume, rtol=0.01)

    smooth_inplace = surf.smooth_taubin(inplace=True)
    assert np.allclose(surf.points, smoothed.points)
    assert np.allclose(smooth_inplace.points, smoothed.points)


@pytest.mark.parametrize('integration_direction', ['forward', 'backward', 'both'])
def test_streamlines_dir(uniform_vec, integration_direction):
    stream = uniform_vec.streamlines(
        'vectors',
        integration_direction=integration_direction,
        progress_bar=True,
    )
    assert all([stream.n_points, stream.n_cells])


@pytest.mark.parametrize('integrator_type', [2, 4, 45])
def test_streamlines_type(uniform_vec, integrator_type):
    stream = uniform_vec.streamlines('vectors', integrator_type=integrator_type, progress_bar=True)
    assert all([stream.n_points, stream.n_cells])


@pytest.mark.parametrize('interpolator_type', ['point', 'cell'])
def test_streamlines_cell_point(uniform_vec, interpolator_type):
    stream = uniform_vec.streamlines(
        'vectors',
        interpolator_type=interpolator_type,
        progress_bar=True,
    )
    assert all([stream.n_points, stream.n_cells])


def test_streamlines_return_source(uniform_vec):
    stream, src = uniform_vec.streamlines(
        'vectors',
        return_source=True,
        pointa=(0.0, 0.0, 0.0),
        pointb=(1.1, 1.1, 0.1),
        progress_bar=True,
    )
    assert isinstance(src, pv.DataSet)
    assert all([stream.n_points, stream.n_cells, src.n_points])


def test_streamlines_start_position(uniform_vec):
    stream = uniform_vec.streamlines('vectors', start_position=(0.5, 0.0, 0.0), progress_bar=True)

    assert all([stream.n_points, stream.n_cells])


def test_streamlines_max_length():
    # mesh that is 50x50x50 length units in size
    mesh = pv.ImageData(dimensions=(6, 6, 6), spacing=(10, 10, 10))

    vel = np.zeros((mesh.n_points, 3))
    vel[:, 0] = 1
    mesh['vel'] = vel

    for step_unit in ('cl', 'l'):
        # First show that the default integrates to near the end of the domain.
        # Use max_step_length to limit error near end of domain.
        stream = mesh.streamlines(
            vectors='vel',
            start_position=(0, 0, 0),
            integration_direction='forward',
            step_unit=step_unit,
            max_step_length=0.1,
        )
        # It doesn't identify edge of domain, and just stops at last point inside domain.
        # Check that is ended reasonably close to edge of domain.
        assert stream.length > 45

        # Next show that the max_length is satisfied
        stream = mesh.streamlines(
            vectors='vel',
            start_position=(0, 0, 0),
            integration_direction='forward',
            max_length=1,
            step_unit=step_unit,
        )
        assert np.isclose(stream.length, 1)

    def check_deprecation():
        if pv._version.version_info[:2] > (0, 48):
            msg = 'Convert error ``max_time`` parameter in ``streamlines_from_source``'
            raise RuntimeError(msg)
        if pv._version.version_info[:2] > (0, 49):
            msg = 'Remove ``max_time`` parameter in ``streamlines_from_source``'
            raise RuntimeError(msg)

    with pytest.warns(PyVistaDeprecationWarning, match='``max_time`` parameter is deprecated'):
        stream = mesh.streamlines(
            vectors='vel',
            start_position=(0, 0, 0),
            integration_direction='forward',
            max_time=1,
            max_step_length=0.1,
        )
    check_deprecation()
    assert np.isclose(stream.length, 1)

    with pytest.warns(
        PyVistaDeprecationWarning,
        match='``max_length`` and ``max_time`` provided. Ignoring deprecated ``max_time``.',
    ):
        stream = mesh.streamlines(
            vectors='vel',
            start_position=(0, 0, 0),
            integration_direction='forward',
            max_time=5,
            max_length=1,
        )
    check_deprecation()
    assert np.isclose(stream.length, 1)


def test_streamlines_errors(uniform_vec):
    with pytest.raises(ValueError):  # noqa: PT011
        uniform_vec.streamlines('vectors', integration_direction='not valid')

    with pytest.raises(ValueError):  # noqa: PT011
        uniform_vec.streamlines('vectors', integrator_type=42)

    with pytest.raises(ValueError):  # noqa: PT011
        uniform_vec.streamlines('vectors', interpolator_type='not valid')

    with pytest.raises(ValueError):  # noqa: PT011
        uniform_vec.streamlines('vectors', step_unit='not valid')

    with pytest.raises(ValueError):  # noqa: PT011
        uniform_vec.streamlines('vectors', pointa=(0, 0, 0))
    with pytest.raises(ValueError):  # noqa: PT011
        uniform_vec.streamlines('vectors', pointb=(0, 0, 0))


def test_streamlines_from_source(uniform_vec):
    vertices = np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0]])
    source = pv.PolyData(vertices)
    stream = uniform_vec.streamlines_from_source(source, 'vectors', progress_bar=True)
    assert all([stream.n_points, stream.n_cells])

    source = pv.ImageData(dimensions=[5, 5, 5], spacing=[0.1, 0.1, 0.1], origin=[0, 0, 0])
    stream = uniform_vec.streamlines_from_source(source, 'vectors', progress_bar=True)
    assert all([stream.n_points, stream.n_cells])


def test_streamlines_from_source_structured_grids():
    x, y, z = np.meshgrid(
        np.arange(-10, 10, 0.5), np.arange(-10, 10, 0.5), np.arange(-10, 10, 0.5)
    )
    mesh = pv.StructuredGrid(x, y, z)
    x2, y2, z2 = np.meshgrid(np.arange(-1, 1, 0.5), np.arange(-1, 1, 0.5), np.arange(-1, 1, 0.5))
    mesh2 = pv.StructuredGrid(x2, y2, z2)
    mesh['vectors'] = np.ones([mesh.n_points, 3])
    mesh.set_active_vectors('vectors')

    with pv.VtkErrorCatcher(raise_errors=True):
        stream = mesh.streamlines_from_source(mesh2)
    assert all([stream.n_points, stream.n_cells])


def mesh_2d_velocity():
    mesh = pv.Plane(i_resolution=100, j_resolution=100)
    velocity = np.zeros([mesh.n_points, 3])
    velocity[:, 0] = 1
    mesh['velocity'] = velocity
    mesh.set_active_vectors('velocity')
    return mesh


def test_streamlines_evenly_spaced_2d():
    mesh = mesh_2d_velocity()
    streams = mesh.streamlines_evenly_spaced_2D(progress_bar=True)
    assert all([streams.n_points, streams.n_cells])


def test_streamlines_evenly_spaced_2d_sep_dist_ratio():
    mesh = mesh_2d_velocity()
    streams = mesh.streamlines_evenly_spaced_2D(separating_distance_ratio=0.1, progress_bar=True)
    assert all([streams.n_points, streams.n_cells])


def test_streamlines_evenly_spaced_2d_start_position():
    mesh = mesh_2d_velocity()
    streams = mesh.streamlines_evenly_spaced_2D(start_position=(-0.1, 0.1, 0.0), progress_bar=True)
    assert all([streams.n_points, streams.n_cells])


def test_streamlines_evenly_spaced_2d_vectors():
    mesh = mesh_2d_velocity()
    mesh.set_active_vectors(None)
    streams = mesh.streamlines_evenly_spaced_2D(vectors='velocity', progress_bar=True)
    assert all([streams.n_points, streams.n_cells])


def test_streamlines_evenly_spaced_2d_integrator_type():
    mesh = mesh_2d_velocity()
    streams = mesh.streamlines_evenly_spaced_2D(integrator_type=4, progress_bar=True)
    assert all([streams.n_points, streams.n_cells])


def test_streamlines_evenly_spaced_2d_interpolator_type():
    mesh = mesh_2d_velocity()
    streams = mesh.streamlines_evenly_spaced_2D(interpolator_type='cell', progress_bar=True)
    assert all([streams.n_points, streams.n_cells])


def test_streamlines_evenly_spaced_2d_errors():
    mesh = mesh_2d_velocity()

    with pytest.raises(ValueError):  # noqa: PT011
        mesh.streamlines_evenly_spaced_2D(integrator_type=45)

    with pytest.raises(ValueError):  # noqa: PT011
        mesh.streamlines_evenly_spaced_2D(interpolator_type='not valid')

    with pytest.raises(ValueError):  # noqa: PT011
        mesh.streamlines_evenly_spaced_2D(step_unit='not valid')


@pytest.mark.xfail
def test_streamlines_nonxy_plane():
    # streamlines_evenly_spaced_2D only works for xy plane datasets
    # test here so that fixes in vtk can be caught
    mesh = mesh_2d_velocity()
    mesh.translate((0, 0, 1), inplace=True)  # move to z=1, xy plane
    streams = mesh.streamlines_evenly_spaced_2D(progress_bar=True)
    assert all([streams.n_points, streams.n_cells])


def test_sample_over_line():
    """Test that we get a sampled line."""
    name = 'values'

    line = pv.Line([0, 0, 0], [0, 0, 10], resolution=9)
    line[name] = np.linspace(0, 10, 10)

    sampled_line = line.sample_over_line([0, 0, 0.5], [0, 0, 1.5], resolution=2, progress_bar=True)

    expected_result = np.array([0.5, 1, 1.5])
    assert np.allclose(sampled_line[name], expected_result)
    assert name in sampled_line.array_names  # is name in sampled result

    # test no resolution
    sphere = pv.Sphere(center=(4.5, 4.5, 4.5), radius=4.5)
    sampled_from_sphere = sphere.sample_over_line([3, 1, 1], [-3, -1, -1], progress_bar=True)
    assert sampled_from_sphere.n_points == sphere.n_cells + 1
    # is sampled result a polydata object
    assert isinstance(sampled_from_sphere, pv.PolyData)


def test_plot_over_line(tmpdir):
    tmp_dir = tmpdir.mkdir('tmpdir')
    filename = str(tmp_dir.join('tmp.png'))
    mesh = examples.load_uniform()
    # Make two points to construct the line between
    a = [mesh.bounds.x_min, mesh.bounds.y_min, mesh.bounds.z_min]
    b = [mesh.bounds.x_max, mesh.bounds.y_max, mesh.bounds.z_max]
    mesh.plot_over_line(a, b, resolution=1000, show=False, progress_bar=True)
    # Test multicomponent
    mesh['foo'] = np.random.default_rng().random((mesh.n_cells, 3))
    mesh.plot_over_line(
        a,
        b,
        resolution=None,
        scalars='foo',
        title='My Stuff',
        ylabel='3 Values',
        show=False,
        fname=filename,
        progress_bar=True,
    )
    assert Path(filename).is_file()
    # Should fail if scalar name does not exist
    with pytest.raises(KeyError):
        mesh.plot_over_line(
            a,
            b,
            resolution=None,
            scalars='invalid_array_name',
            title='My Stuff',
            ylabel='3 Values',
            show=False,
        )


def test_sample_over_multiple_lines():
    """Test that"""
    name = 'values'

    line = pv.Line([0, 0, 0], [0, 0, 10], resolution=9)
    line[name] = np.linspace(0, 10, 10)

    sampled_multiple_lines = line.sample_over_multiple_lines(
        [[0, 0, 0.5], [0, 0, 1], [0, 0, 1.5]],
        progress_bar=True,
    )

    expected_result = np.array([0.5, 1, 1.5])
    assert np.allclose(sampled_multiple_lines[name], expected_result)
    assert name in sampled_multiple_lines.array_names  # is name in sampled result


def test_sample_over_circular_arc():
    """Test that we get a circular arc."""
    name = 'values'

    uniform = examples.load_uniform()
    uniform[name] = uniform.points[:, 2]

    xmin = uniform.bounds.x_min
    xmax = uniform.bounds.x_max
    ymin = uniform.bounds.y_min
    zmin = uniform.bounds.z_min
    zmax = uniform.bounds.z_max
    pointa = [xmin, ymin, zmax]
    pointb = [xmax, ymin, zmin]
    center = [xmin, ymin, zmin]
    sampled_arc = uniform.sample_over_circular_arc(
        pointa=pointa, pointb=pointb, center=center, resolution=2, progress_bar=True
    )

    expected_result = zmin + (zmax - zmin) * np.sin([np.pi / 2.0, np.pi / 4.0, 0.0])
    assert np.allclose(sampled_arc[name], expected_result)
    assert name in sampled_arc.array_names  # is name in sampled result

    # test no resolution
    sphere = pv.Sphere(center=(4.5, 4.5, 4.5), radius=4.5)
    sampled_from_sphere = sphere.sample_over_circular_arc(
        pointa=[3, 1, 1],
        pointb=[-3, -1, -1],
        center=[0, 0, 0],
        progress_bar=True,
    )
    assert sampled_from_sphere.n_points == sphere.n_cells + 1

    # is sampled result a polydata object
    assert isinstance(sampled_from_sphere, pv.PolyData)


def test_sample_over_circular_arc_normal():
    """Test that we get a circular arc_normal."""
    name = 'values'

    uniform = examples.load_uniform()
    uniform[name] = uniform.points[:, 2]

    xmin = uniform.bounds.x_min
    ymin = uniform.bounds.y_min
    ymax = uniform.bounds.y_max
    zmin = uniform.bounds.z_min
    zmax = uniform.bounds.z_max
    normal = [xmin, ymax, zmin]
    polar = [xmin, ymin, zmax]
    angle = 90.0 * np.random.default_rng().random()
    resolution = np.random.default_rng().integers(10000)
    center = [xmin, ymin, zmin]
    sampled_arc_normal = uniform.sample_over_circular_arc_normal(
        center=center,
        resolution=resolution,
        normal=normal,
        polar=polar,
        angle=angle,
        progress_bar=True,
    )
    angles = np.linspace(np.pi / 2.0, np.pi / 2.0 - np.deg2rad(angle), resolution + 1)

    expected_result = zmin + (zmax - zmin) * np.sin(angles)
    assert np.allclose(sampled_arc_normal[name], expected_result)
    assert name in sampled_arc_normal.array_names  # is name in sampled result

    # test no resolution
    sphere = pv.Sphere(center=(4.5, 4.5, 4.5), radius=4.5)
    sampled_from_sphere = sphere.sample_over_circular_arc_normal(
        center=[0, 0, 0],
        polar=[3, 1, 1],
        angle=180,
        progress_bar=True,
    )
    assert sampled_from_sphere.n_points == sphere.n_cells + 1

    # is sampled result a polydata object
    assert isinstance(sampled_from_sphere, pv.PolyData)


def test_plot_over_circular_arc(tmpdir):
    mesh = examples.load_uniform()
    tmp_dir = tmpdir.mkdir('tmpdir')
    filename = str(tmp_dir.join('tmp.png'))

    # Make two points and center to construct the circular arc between
    a = [mesh.bounds.x_min, mesh.bounds.y_min, mesh.bounds.z_max]
    b = [mesh.bounds.x_max, mesh.bounds.y_min, mesh.bounds.z_min]
    center = [mesh.bounds.x_min, mesh.bounds.y_min, mesh.bounds.z_min]
    mesh.plot_over_circular_arc(
        pointa=a,
        pointb=b,
        center=center,
        resolution=1000,
        show=False,
        fname=filename,
        progress_bar=True,
    )
    assert Path(filename).is_file()

    # Test multicomponent
    mesh['foo'] = np.random.default_rng().random((mesh.n_cells, 3))
    mesh.plot_over_circular_arc(
        pointa=a,
        pointb=b,
        center=center,
        resolution=None,
        scalars='foo',
        title='My Stuff',
        ylabel='3 Values',
        show=False,
        progress_bar=True,
    )

    # Should fail if scalar name does not exist
    with pytest.raises(KeyError):
        mesh.plot_over_circular_arc(
            pointa=a,
            pointb=b,
            center=center,
            resolution=None,
            scalars='invalid_array_name',
            title='My Stuff',
            ylabel='3 Values',
            show=False,
        )


def test_plot_over_circular_arc_normal(tmpdir):
    mesh = examples.load_uniform()
    tmp_dir = tmpdir.mkdir('tmpdir')
    filename = str(tmp_dir.join('tmp.png'))

    # Make center and normal/polar vector to construct the circular arc between
    # normal = [mesh.bounds.x_min, mesh.bounds.y_min, mesh.bounds.z_max]
    polar = [mesh.bounds.x_min, mesh.bounds.y_max, mesh.bounds.z_min]
    angle = 90
    center = [mesh.bounds.x_min, mesh.bounds.y_min, mesh.bounds.z_min]
    mesh.plot_over_circular_arc_normal(
        center=center,
        polar=polar,
        angle=angle,
        show=False,
        fname=filename,
        progress_bar=True,
    )
    assert Path(filename).is_file()

    # Test multicomponent
    mesh['foo'] = np.random.default_rng().random((mesh.n_cells, 3))
    mesh.plot_over_circular_arc_normal(
        center=center,
        polar=polar,
        angle=angle,
        resolution=None,
        scalars='foo',
        title='My Stuff',
        ylabel='3 Values',
        show=False,
        progress_bar=True,
    )

    # Should fail if scalar name does not exist
    with pytest.raises(KeyError):
        mesh.plot_over_circular_arc_normal(
            center=center,
            polar=polar,
            angle=angle,
            resolution=None,
            scalars='invalid_array_name',
            title='My Stuff',
            ylabel='3 Values',
            show=False,
        )


def extract_points_invalid(sphere):
    with pytest.raises(ValueError):  # noqa: PT011
        sphere.extract_points('invalid')

    with pytest.raises(TypeError):
        sphere.extract_points(object)


@pytest.fixture
def grid4x4():
    # mesh points (4x4 regular grid)
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [2, 1, 0],
            [3, 1, 0],
            [0, 2, 0],
            [1, 2, 0],
            [2, 2, 0],
            [3, 2, 0],
            [0, 3, 0],
            [1, 3, 0],
            [2, 3, 0],
            [3, 3, 0],
        ],
        dtype=np.float32,
    )
    # corresponding mesh faces
    faces = np.hstack(
        [
            [4, 0, 1, 5, 4],  # square
            [4, 1, 2, 6, 5],  # square
            [4, 2, 3, 7, 6],  # square
            [4, 4, 5, 9, 8],  # square
            [4, 5, 6, 10, 9],  # square
            [4, 6, 7, 11, 10],  # square
            [4, 8, 9, 13, 12],  # square
            [4, 9, 10, 14, 13],  # square
            [4, 10, 11, 15, 14],  # square
        ],
    )
    surf = pv.PolyData(vertices, faces)
    surf.point_data['labels'] = range(surf.n_points)
    surf.cell_data['labels'] = range(surf.n_cells)
    return surf


@pytest.fixture
def extracted_with_adjacent_false(grid4x4):
    """Return expected output for adjacent_cells=False and include_cells=True"""
    input_point_ids = [0, 1, 4, 5]
    expected_cell_ids = [0]
    expected_point_ids = [0, 1, 4, 5]
    expected_verts = grid4x4.points[expected_point_ids, :]
    expected_faces = [4, 0, 1, 3, 2]
    celltypes = np.full(1, CellType.QUAD, dtype=np.uint8)
    expected_surf = pv.UnstructuredGrid(expected_faces, celltypes, expected_verts)
    expected_surf.point_data['labels'] = expected_point_ids
    expected_surf.cell_data['labels'] = expected_cell_ids
    return grid4x4, input_point_ids, expected_cell_ids, expected_surf


@pytest.fixture
def extracted_with_adjacent_true(grid4x4):
    """Return expected output for adjacent_cells=True and include_cells=True"""
    input_point_ids = [0, 1, 4, 5]
    expected_cell_ids = [0, 1, 3, 4]
    expected_point_ids = [0, 1, 2, 4, 5, 6, 8, 9, 10]
    expected_verts = grid4x4.points[expected_point_ids, :]
    expected_faces = [4, 0, 1, 4, 3, 4, 1, 2, 5, 4, 4, 3, 4, 7, 6, 4, 4, 5, 8, 7]
    celltypes = np.full(4, CellType.QUAD, dtype=np.uint8)
    expected_surf = pv.UnstructuredGrid(expected_faces, celltypes, expected_verts)
    expected_surf.point_data['labels'] = expected_point_ids
    expected_surf.cell_data['labels'] = expected_cell_ids
    return grid4x4, input_point_ids, expected_cell_ids, expected_surf


@pytest.fixture
def extracted_with_include_cells_false(grid4x4):
    """Return expected output for adjacent_cells=True and include_cells=False"""
    input_point_ids = [0, 1, 4, 5]
    expected_cell_ids = [0, 0, 0, 0]
    expected_point_ids = [0, 1, 4, 5]
    expected_verts = grid4x4.points[expected_point_ids, :]
    expected_faces = [1, 0, 1, 1, 1, 2, 1, 3]
    celltypes = np.full(4, CellType.VERTEX, dtype=np.uint8)
    expected_surf = pv.UnstructuredGrid(expected_faces, celltypes, expected_verts)
    expected_surf.point_data['labels'] = expected_point_ids
    expected_surf.cell_data['labels'] = expected_cell_ids
    return grid4x4, input_point_ids, expected_cell_ids, expected_surf


@pytest.mark.parametrize(
    'dataset_filter',
    [pv.DataSetFilters.extract_points, pv.DataSetFilters.extract_values],
)
def test_extract_points_adjacent_cells_true(dataset_filter, extracted_with_adjacent_true):
    input_surf, input_point_ids, _, expected_surf = extracted_with_adjacent_true

    # extract sub-surface with adjacent cells
    sub_surf_adj = dataset_filter(
        input_surf,
        input_point_ids,
        adjacent_cells=True,
        progress_bar=True,
    )

    assert sub_surf_adj.n_points == 9
    assert np.array_equal(sub_surf_adj.points, expected_surf.points)
    assert sub_surf_adj.n_cells == 4
    assert np.array_equal(sub_surf_adj.cells, expected_surf.cells)


@pytest.mark.parametrize(
    'dataset_filter',
    [pv.DataSetFilters.extract_points, pv.DataSetFilters.extract_values],
)
def test_extract_points_adjacent_cells_false(dataset_filter, extracted_with_adjacent_false):
    input_surf, input_point_ids, _, expected_surf = extracted_with_adjacent_false
    # extract sub-surface without adjacent cells
    sub_surf = dataset_filter(input_surf, input_point_ids, adjacent_cells=False, progress_bar=True)

    assert sub_surf.n_points == 4
    assert np.array_equal(sub_surf.points, expected_surf.points)
    assert sub_surf.n_cells == 1
    assert np.array_equal(sub_surf.cells, expected_surf.cells)


@pytest.mark.parametrize(
    'dataset_filter',
    [pv.DataSetFilters.extract_points, pv.DataSetFilters.extract_values],
)
def test_extract_points_include_cells_false(
    dataset_filter,
    extracted_with_include_cells_false,
):
    input_surf, input_point_ids, _, expected_surf = extracted_with_include_cells_false
    # extract sub-surface without cells
    sub_surf_nocells = dataset_filter(
        input_surf,
        input_point_ids,
        adjacent_cells=True,
        include_cells=False,
        progress_bar=True,
    )
    assert np.array_equal(sub_surf_nocells.points, expected_surf.points)
    assert np.array_equal(sub_surf_nocells.cells, expected_surf.cells)
    assert all(celltype == pv.CellType.VERTEX for celltype in sub_surf_nocells.celltypes)


def test_extract_points_default(extracted_with_adjacent_true):
    input_surf, input_point_ids, _, expected_surf = extracted_with_adjacent_true

    assert 'vtkOriginalPointIds' not in input_surf.point_data
    assert 'vtkOriginalCellIds' not in input_surf.cell_data

    # test adjacent_cells=True and include_cells=True by default
    sub_surf_adj = input_surf.extract_points(input_point_ids)

    # should be no side effects
    assert 'vtkOriginalPointIds' not in input_surf.point_data
    assert 'vtkOriginalCellIds' not in input_surf.cell_data

    assert np.array_equal(sub_surf_adj.points, expected_surf.points)
    assert np.array_equal(sub_surf_adj.cells, expected_surf.cells)


@pytest.mark.parametrize('preference', ['point', 'cell'])
@pytest.mark.parametrize('adjacent_fixture', [True, False])
def test_extract_values_preference(
    preference,
    adjacent_fixture,
    extracted_with_adjacent_true,
    extracted_with_adjacent_false,
):
    # test points are extracted with point data (with adjacent = False by default)
    # test cells are extracted with cell data
    fixture = extracted_with_adjacent_true if adjacent_fixture else extracted_with_adjacent_false
    input_surf, input_point_ids, input_cell_ids, expected_surf = fixture

    func = functools.partial(input_surf.extract_values, scalars='labels', preference=preference)
    if preference == 'point':
        sub_surf = func(input_point_ids)
        if not adjacent_fixture:
            pytest.xfail(
                'Will not match expected output since adjacent is True by default for point data ',
            )
    else:
        sub_surf = func(input_cell_ids)

    assert np.array_equal(sub_surf.points, expected_surf.points)
    assert np.array_equal(sub_surf.cells, expected_surf.cells)


def extract_values_values():
    # Define values to extract all 16 points or all 9 cells of the 4x4 grid
    point_values = [
        range(16),
        list(range(16)),
        np.array(range(16)),
    ]
    cell_values = [
        range(10),
        list(range(10)),
        np.array(range(10)),
    ]

    return list(zip(point_values, cell_values, strict=True))


@pytest.mark.parametrize('preference', ['point', 'cell'])
@pytest.mark.parametrize('invert', [True, False])
@pytest.mark.parametrize('values', extract_values_values())
def test_extract_values_input_values_and_invert(preference, values, invert, grid4x4):
    # test extracting all points or cells
    values = values[0] if preference == 'point' else values[1]
    extracted = grid4x4.extract_values(values, preference=preference, invert=invert)
    if invert:
        assert extracted.n_points == 0
        assert extracted.n_cells == 0
        assert extracted.n_arrays == 0
    else:
        assert np.array_equal(extracted.points, grid4x4.points)
        assert np.array_equal(extracted.cells, grid4x4.faces)


def test_extract_values_open_intervals(grid4x4):
    extracted = grid4x4.extract_values(ranges=[float('-inf'), float('inf')])
    assert extracted.n_points == 16
    assert extracted.n_cells == 9

    extracted = grid4x4.extract_values(ranges=[0, np.inf])
    assert extracted.n_points == 16
    assert extracted.n_cells == 9

    extracted = grid4x4.extract_values(ranges=[-np.inf, 16])
    assert extracted.n_points == 16
    assert extracted.n_cells == 9


SMALL_VOLUME = 1**3
BIG_VOLUME = 2**3


@pytest.fixture
def labeled_data():
    def append(mesh1, mesh2):
        filter_ = _vtk.vtkAppendFilter()
        filter_.AddInputData(mesh1)
        filter_.AddInputData(mesh2)
        filter_.Update()
        return pv.wrap(filter_.GetOutput())

    bounds = np.array((-0.5, 0.5, -0.5, 0.5, -0.5, 0.5))
    small_box = pv.Box(bounds=bounds)
    big_box = pv.Box(bounds=bounds * 2)
    labeled = append(big_box, small_box).extract_surface(algorithm=None).connectivity()
    assert isinstance(labeled, pv.PolyData)
    assert labeled.array_names == ['RegionId', 'RegionId']
    assert np.allclose(small_box.volume, SMALL_VOLUME)
    assert np.allclose(big_box.volume, BIG_VOLUME)
    return small_box, big_box, labeled


def add_component_to_labeled_data(labeled_data, offset):
    """Add second component to scalars by duplicating first component and adding offset."""
    if offset is None:
        return labeled_data
    small_box, big_box, labeled_data = labeled_data

    point_data = labeled_data.point_data['RegionId']
    point_data = np.vstack([point_data, point_data + offset]).T
    labeled_data.point_data['RegionId'] = point_data

    cell_data = labeled_data.cell_data['RegionId']
    cell_data = np.vstack([cell_data, cell_data + offset]).T
    labeled_data.cell_data['RegionId'] = cell_data

    return small_box, big_box, labeled_data


class SplitComponentTestCase(NamedTuple):
    component_offset: Any
    component_mode: Any
    expected_n_blocks: Any
    expected_volume: Any


split_component_test_cases = [
    SplitComponentTestCase(
        component_offset=None,
        component_mode=0,
        expected_n_blocks=2,
        expected_volume=[SMALL_VOLUME, BIG_VOLUME],
    ),
    SplitComponentTestCase(
        component_offset=None,
        component_mode='any',
        expected_n_blocks=2,
        expected_volume=[SMALL_VOLUME, BIG_VOLUME],
    ),
    SplitComponentTestCase(
        component_offset=None,
        component_mode='all',
        expected_n_blocks=2,
        expected_volume=[SMALL_VOLUME, BIG_VOLUME],
    ),
    SplitComponentTestCase(
        component_offset=0,
        component_mode='0',
        expected_n_blocks=2,
        expected_volume=[SMALL_VOLUME, BIG_VOLUME],
    ),
    SplitComponentTestCase(
        component_offset=0,
        component_mode='1',
        expected_n_blocks=2,
        expected_volume=[SMALL_VOLUME, BIG_VOLUME],
    ),
    SplitComponentTestCase(
        component_offset=0,
        component_mode='any',
        expected_n_blocks=2,
        expected_volume=[SMALL_VOLUME, BIG_VOLUME],
    ),
    SplitComponentTestCase(
        component_offset=0,
        component_mode='all',
        expected_n_blocks=2,
        expected_volume=[SMALL_VOLUME, BIG_VOLUME],
    ),
    SplitComponentTestCase(
        component_offset=-0.5,
        component_mode=0,
        expected_n_blocks=2,
        expected_volume=[SMALL_VOLUME, BIG_VOLUME],
    ),
    SplitComponentTestCase(
        component_offset=-0.5,
        component_mode=1,
        expected_n_blocks=2,
        expected_volume=[SMALL_VOLUME, BIG_VOLUME],
    ),
    SplitComponentTestCase(
        component_offset=-0.5,
        component_mode='any',
        expected_n_blocks=4,
        expected_volume=[SMALL_VOLUME, SMALL_VOLUME, BIG_VOLUME, BIG_VOLUME],
    ),
    SplitComponentTestCase(
        component_offset=-0.5,
        component_mode='all',
        expected_n_blocks=4,
        expected_volume=[0, 0, 0, 0],
    ),
]


@pytest.mark.parametrize(
    ('component_offset', 'component_mode', 'expected_n_blocks', 'expected_volume'),
    split_component_test_cases,
)
@pytest.mark.parametrize(
    ('dataset_filter', 'kwargs'),
    [
        (pv.DataSetFilters.split_values, {}),
        (pv.DataSetFilters.extract_values, dict(values='_unique', split=True)),
    ],
)
def test_split_values_extract_values_component(
    dataset_filter,
    kwargs,
    labeled_data,
    component_offset,
    component_mode,
    expected_n_blocks,
    expected_volume,
):
    # Add second component to fixture for test as needed
    _small_box, _big_box, labeled_data = add_component_to_labeled_data(
        labeled_data,
        component_offset,
    )
    multiblock = dataset_filter(labeled_data, component_mode=component_mode, **kwargs)
    assert isinstance(multiblock, pv.MultiBlock)
    assert multiblock.n_blocks == expected_n_blocks
    assert all(isinstance(block, pv.UnstructuredGrid) for block in multiblock)

    # Convert to polydata to test volume
    multiblock = multiblock.as_polydata_blocks()
    assert expected_n_blocks == len(expected_volume)
    for block, volume in zip(multiblock, expected_volume, strict=True):
        assert np.isclose(block.volume, volume)


def test_extract_values_split_ranges_values(labeled_data):
    _, _, labeled_data = labeled_data
    extracted = labeled_data.extract_values(values=[0, 1], ranges=[[0, 0], [1, 1]], split=True)
    assert isinstance(extracted, pv.MultiBlock)
    assert extracted.n_blocks == 4
    extracted_value0 = extracted[0]
    extracted_value1 = extracted[1]
    extracted_range00 = extracted[2]
    extracted_range11 = extracted[3]
    assert extracted_value0 == extracted_range00
    assert extracted_value1 == extracted_range11


# Test cases with and/or without dict inputs
# Include swapped [name, value] or [value, name] inputs
values_nodict_ranges_dict = dict(values=0, ranges=dict(rng=[0, 0])), ['Block-00', 'rng']
values_dict_ranges_nodict = dict(values={0: 'val'}, ranges=[0, 0]), ['val', 'Block-01']
values_dict_ranges_dict = (
    dict(values=dict(val=0), ranges={(0, 0): 'rng'}),
    ['val', 'rng'],
)
values_component_dict = (
    dict(values=dict(val0=[0], val1=[1]), component_mode='multi'),
    [
        'val0',
        'val1',
    ],
)


@pytest.mark.parametrize(
    ('dict_inputs', 'block_names'),
    [
        values_nodict_ranges_dict,
        values_dict_ranges_nodict,
        values_dict_ranges_dict,
        values_component_dict,
    ],
)
def test_extract_values_dict_input(labeled_data, dict_inputs, block_names):
    _, _, labeled_data = labeled_data
    extracted = labeled_data.extract_values(**dict_inputs, split=True)
    assert isinstance(extracted, pv.MultiBlock)
    assert extracted.n_blocks == 2
    assert extracted.keys() == block_names


BLACK = [0.0, 0.0, 0.0]
WHITE = [1.0, 1.0, 1.0]
RED = [1.0, 0.0, 0.0]
GREEN = [0.0, 1.0, 0.0]
BLUE = [0.0, 0.0, 1.0]
COLORS_LIST = [BLACK, WHITE, RED, GREEN, BLUE]


@pytest.fixture
def point_cloud_colors():
    # Define point cloud where the points and rgb scalars are the same
    array = COLORS_LIST
    point_cloud = pv.PointSet(array)
    point_cloud['colors'] = array
    return point_cloud


@pytest.fixture
def point_cloud_colors_duplicates(point_cloud_colors):
    # Same fixture as point_cloud_colors but with double the points
    copied = point_cloud_colors.copy()
    copied.points += 0.5
    return point_cloud_colors + copied


class ComponentModeTestCase(NamedTuple):
    values: Any
    component_mode: Any
    expected: Any
    expected_invert: Any


component_mode_test_cases = [
    ComponentModeTestCase(
        values=0,
        component_mode=0,
        expected=[BLACK, GREEN, BLUE],
        expected_invert=[WHITE, RED],
    ),
    ComponentModeTestCase(
        values=0,
        component_mode=1,
        expected=[BLACK, RED, BLUE],
        expected_invert=[WHITE, GREEN],
    ),
    ComponentModeTestCase(
        values=0,
        component_mode=2,
        expected=[BLACK, RED, GREEN],
        expected_invert=[WHITE, BLUE],
    ),
    ComponentModeTestCase(
        values=0,
        component_mode='any',
        expected=[BLACK, RED, GREEN, BLUE],
        expected_invert=[WHITE],
    ),
    ComponentModeTestCase(
        values=1,
        component_mode='any',
        expected=[WHITE, RED, GREEN, BLUE],
        expected_invert=[BLACK],
    ),
    ComponentModeTestCase(
        values=0,
        component_mode='all',
        expected=[BLACK],
        expected_invert=[WHITE, RED, GREEN, BLUE],
    ),
    ComponentModeTestCase(
        values=1,
        component_mode='all',
        expected=[WHITE],
        expected_invert=[BLACK, RED, GREEN, BLUE],
    ),
    ComponentModeTestCase(
        values=BLACK,
        component_mode='multi',
        expected=[BLACK],
        expected_invert=[WHITE, RED, GREEN, BLUE],
    ),
    ComponentModeTestCase(
        values=[WHITE, RED],
        component_mode='multi',
        expected=[WHITE, RED],
        expected_invert=[BLACK, GREEN, BLUE],
    ),
]


@pytest.mark.parametrize('values_as_ranges', [True, False])
@pytest.mark.parametrize(('split', 'invert'), [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize(
    ('values', 'component_mode', 'expected', 'expected_invert'),
    component_mode_test_cases,
)
def test_extract_values_component_mode(
    point_cloud_colors,
    values,
    component_mode,
    expected,
    expected_invert,
    invert,
    split,
    values_as_ranges,
):
    values_kwarg = dict(values=values)
    if values_as_ranges:
        # Get additional test coverage by converting single value inputs into a range
        if component_mode == 'multi':
            pytest.skip("Cannot use ranges with 'multi' mode.")
        values_kwarg = dict(ranges=[values - 0.5, values + 0.5])

    extracted = point_cloud_colors.extract_values(
        **values_kwarg,
        component_mode=component_mode,
        split=split,
        invert=invert,
    )
    single_mesh = extracted.combine() if split else extracted
    actual_points = single_mesh.points
    actual_colors = single_mesh['colors']
    assert np.array_equal(actual_points, expected_invert if invert else expected)
    assert np.array_equal(actual_colors, expected_invert if invert else expected)


@pytest.mark.parametrize(
    ('dataset_filter', 'kwargs'),
    [
        (pv.DataSetFilters.split_values, {}),
        (pv.DataSetFilters.extract_values, dict(values='_unique', split=True)),
    ],
)
def test_extract_values_component_values_split_unique(
    point_cloud_colors_duplicates,
    dataset_filter,
    kwargs,
):
    extracted = dataset_filter(point_cloud_colors_duplicates, component_mode='multi', **kwargs)
    assert isinstance(extracted, pv.MultiBlock)
    assert extracted.n_blocks == len(COLORS_LIST)
    assert (
        np.array_equal(block['colors'], [color, color])
        for block, color in zip(extracted, COLORS_LIST, strict=True)
    )


@pytest.mark.parametrize('pass_point_ids', [True, False])
@pytest.mark.parametrize('pass_cell_ids', [True, False])
def test_extract_values_pass_ids(grid4x4, pass_point_ids, pass_cell_ids):
    POINT_IDS = 'vtkOriginalPointIds'
    CELL_IDS = 'vtkOriginalCellIds'
    extracted = grid4x4.extract_values(ranges=grid4x4.get_data_range())
    assert extracted.point_data.keys() == ['labels', POINT_IDS]
    assert extracted.cell_data.keys() == ['labels', CELL_IDS]

    extracted = grid4x4.extract_values(
        ranges=grid4x4.get_data_range(),
        pass_point_ids=pass_point_ids,
        pass_cell_ids=pass_cell_ids,
    )
    if pass_cell_ids:
        assert CELL_IDS in extracted.cell_data
    if pass_point_ids:
        assert POINT_IDS in extracted.point_data

    extracted = grid4x4.extract_values(
        ranges=grid4x4.get_data_range(preference='point'),
        invert=True,
        pass_point_ids=pass_point_ids,
        pass_cell_ids=pass_cell_ids,
    )
    if pass_cell_ids:
        assert extracted.cell_data.keys() == []
    if pass_point_ids:
        assert extracted.point_data.keys() == []


def test_extract_values_empty():
    empty = pv.PolyData()
    output = pv.PolyData().extract_values()
    assert isinstance(output, pv.UnstructuredGrid)
    assert empty is not output

    output = empty.extract_values(split=True)
    assert isinstance(output, pv.MultiBlock)
    assert output.n_blocks == 0

    output = empty.extract_values([0, 1, 2], ranges=[1, 2], split=True)
    assert isinstance(output, pv.MultiBlock)
    assert output.n_blocks == 4


def test_extract_values_raises(grid4x4):
    match = 'Values must be numeric.'
    with pytest.raises(TypeError, match=match):
        grid4x4.extract_values('abc')

    match = 'Values must be one-dimensional. Got 2d values.'
    with pytest.raises(ValueError, match=match):
        grid4x4.extract_values([[2, 3]])

    match = 'Ranges must be 2 dimensional. Got 3.'
    with pytest.raises(ValueError, match=match):
        grid4x4.extract_values(ranges=[[[0, 1]]])

    match = 'Ranges must be numeric.'
    with pytest.raises(TypeError, match=match):
        grid4x4.extract_values(ranges='abc')

    match = 'Invalid range [1 0] specified. Lower value cannot be greater than upper value.'
    with pytest.raises(ValueError, match=re.escape(match)):
        grid4x4.extract_values(ranges=[1, 0])

    match = 'No ranges or values were specified. At least one must be specified.'
    with pytest.raises(TypeError, match=match):
        grid4x4.extract_values()

    match = "Invalid dict mapping. The dict's keys or values must contain strings."
    with pytest.raises(TypeError, match=match):
        grid4x4.extract_values({0: 1})

    match = (
        "Invalid component index '1' specified for scalars with 1 component(s). "
        'Value must be one of: (0,).'
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        grid4x4.extract_values(component_mode=1)

    match = "Invalid component index '-1' specified"
    with pytest.raises(ValueError, match=match):
        grid4x4.extract_values(component_mode=-1)

    match = "Invalid component 'foo'. Must be an integer, 'any', 'all', or 'multi'."
    with pytest.raises(ValueError, match=match):
        grid4x4.extract_values(component_mode='foo')

    match = "Ranges cannot be extracted using component mode 'multi'. Expected None, got [0, 1]."
    with pytest.raises(TypeError, match=re.escape(match)):
        grid4x4.extract_values(ranges=[0, 1], component_mode='multi')

    match = 'Component values cannot be more than 2 dimensions. Got 3.'
    with pytest.raises(ValueError, match=match):
        grid4x4.extract_values(values=[[[0]]], component_mode='multi')

    match = (
        "Invalid component index '2' specified for scalars with 1 component(s). "
        'Value must be one of: (0,).'
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        grid4x4.extract_values(component_mode=2)

    match = 'Num components in values array (2) must match num components in data array (1).'
    with pytest.raises(ValueError, match=re.escape(match)):
        grid4x4.extract_values(values=[0, 1], component_mode='multi')


def test_interpolate():
    pdata = pv.PolyData()
    pdata.points = np.random.default_rng().random((10, 3))
    pdata['scalars'] = np.random.default_rng().random(10)
    surf = pv.Sphere(theta_resolution=10, phi_resolution=10)
    interp = surf.interpolate(pdata, radius=0.01, progress_bar=True)
    assert interp.n_points
    assert interp.n_arrays


def test_select_enclosed_points(uniform, hexbeam):
    surf = pv.Sphere(center=uniform.center, radius=uniform.length / 2.0)
    with pytest.warns(pv.PyVistaDeprecationWarning):
        result = uniform.select_enclosed_points(surf, progress_bar=True)
    assert isinstance(result, type(uniform))
    assert 'SelectedPoints' in result.array_names
    assert result['SelectedPoints'].any()
    assert result.n_arrays == uniform.n_arrays + 1

    # Now check non-closed surface
    mesh = pv.Sphere(end_theta=270)
    surf = mesh.rotate_x(90, inplace=False)
    with pytest.warns(pv.PyVistaDeprecationWarning):
        result = mesh.select_enclosed_points(surf, check_surface=False, progress_bar=True)
    assert isinstance(result, type(mesh))
    assert 'SelectedPoints' in result.array_names
    assert result.n_arrays == mesh.n_arrays + 1
    with pytest.raises(RuntimeError):
        with pytest.warns(pv.PyVistaDeprecationWarning):
            result = mesh.select_enclosed_points(surf, check_surface=True, progress_bar=True)
    with pytest.raises(TypeError):
        with pytest.warns(pv.PyVistaDeprecationWarning):
            result = mesh.select_enclosed_points(hexbeam, check_surface=True, progress_bar=True)


def test_select_interior_points(uniform, hexbeam):
    surf = pv.Sphere(center=uniform.center, radius=uniform.length / 2.0)
    assert uniform.active_scalars_name is not None
    result = uniform.select_interior_points(surf)
    assert isinstance(result, type(uniform))
    assert 'selected_points' in result.array_names
    assert result['selected_points'].any()
    assert result.n_arrays == uniform.n_arrays + 1
    assert result.active_scalars_name == 'selected_points'

    # Now check non-closed surface
    mesh = pv.Sphere(end_theta=270)
    surf = mesh.rotate_x(90, inplace=False)
    result = mesh.select_interior_points(surf, check_surface=False)
    assert isinstance(result, type(mesh))
    assert 'selected_points' in result.array_names
    assert result.n_arrays == mesh.n_arrays + 1

    match = (
        'Surface is not closed. Please read the warning in the documentation for\n'
        'this function and either pass `check_surface=False` or repair the surface.'
    )
    with pytest.raises(RuntimeError, match=match):
        mesh.select_interior_points(surf, check_surface=True)
    with pytest.raises(TypeError):
        mesh.select_interior_points(hexbeam, check_surface=True)


@pytest.mark.parametrize('inside_out', [True, False])
def test_select_interior_points_method(sphere, plane, inside_out):
    def _extract_points(method):
        selected_locator = plane.select_interior_points(
            sphere, method=method, inside_out=inside_out
        )
        return plane.extract_points(selected_locator['selected_points'], include_cells=False)

    pts_locator = _extract_points('cell_locator')
    pts_distance = _extract_points('signed_distance')
    assert pts_locator == pts_distance


def test_select_interior_points_raises(sphere, plane):
    match = 'locator_tolerance cannot be used with the signed_distance method.'
    with pytest.raises(ValueError, match=match):
        plane.select_interior_points(sphere, locator_tolerance=0.1)

    plane.select_interior_points(sphere, method='cell_locator', locator_tolerance=0.1)


@pytest.mark.parametrize('method', ['cell_locator', 'signed_distance'])
def test_select_interior_points_empty_mesh(method):
    out = pv.PolyData().select_interior_points(pv.PolyData(), method=method)
    assert isinstance(out, pv.PolyData)
    assert out.array_names == ['selected_points']
    assert out['selected_points'].size == 0


def test_decimate_boundary():
    mesh = examples.load_uniform()
    boundary = mesh.decimate_boundary(progress_bar=True)
    assert boundary.n_points


def test_merge_general(uniform):
    thresh = uniform.threshold_percent([0.2, 0.5], progress_bar=True)  # unstructured grid
    con = uniform.contour()  # poly data
    merged = thresh + con
    assert isinstance(merged, pv.UnstructuredGrid)
    merged = con + thresh
    assert isinstance(merged, pv.UnstructuredGrid)
    # Pure PolyData inputs should yield poly data output
    merged = uniform.extract_surface(algorithm=None) + con
    assert isinstance(merged, pv.PolyData)


def test_merge_active_normals():
    plane = pv.Plane()

    # Check default normals
    default_normal = np.array([0, 0, 1])
    assert np.array_equal(plane['Normals'][0], default_normal)
    assert np.array_equal(plane.active_normals[0], default_normal)
    assert np.array_equal(plane.point_normals[0], default_normal)

    # Customize the normals
    plane['Normals'] *= -1
    negative_normal = -default_normal
    assert np.array_equal(plane['Normals'][0], negative_normal)
    assert np.array_equal(plane.active_normals[0], negative_normal)
    assert np.array_equal(plane.point_normals[0], negative_normal)

    # Now test merge
    merged = pv.merge([plane])
    assert np.array_equal(merged['Normals'][0], negative_normal)
    assert np.array_equal(merged.active_normals[0], negative_normal)
    assert np.array_equal(merged.point_normals[0], negative_normal)


def test_iadd_general(uniform, hexbeam, sphere):
    unstructured = hexbeam
    sphere_shifted = sphere.copy()
    sphere_shifted.points += [1, 1, 1]
    # successful case: poly += poly
    merged = sphere
    merged += sphere_shifted
    assert merged is sphere

    # successful case: unstructured += anything
    merged = unstructured
    merged += uniform
    assert merged is unstructured
    merged += unstructured
    assert merged is unstructured
    merged += sphere
    assert merged is unstructured

    # failing case: poly += non-poly
    merged = sphere
    with pytest.raises(TypeError):
        merged += uniform

    # failing case: uniform += anything
    merged = uniform
    with pytest.raises(TypeError):
        merged += uniform
    with pytest.raises(TypeError):
        merged += unstructured
    with pytest.raises(TypeError):
        merged += sphere


@pytest.mark.needs_vtk_version(9, 3, 0)
def test_compute_boundary_mesh_quality():
    mesh = examples.download_can_crushed_vtu()
    qual = mesh.compute_boundary_mesh_quality()
    assert 'DistanceFromCellCenterToFaceCenter' in qual.array_names
    assert 'DistanceFromCellCenterToFacePlane' in qual.array_names
    assert 'AngleFaceNormalAndCellCenterToFaceCenterVector' in qual.array_names


def test_compute_derivatives(random_hills):
    mesh = random_hills
    vector = np.zeros((mesh.n_points, 3))
    vector[:, 1] = np.ones(mesh.n_points)
    mesh['vector'] = vector
    derv = mesh.compute_derivative(
        scalars='vector',
        gradient=True,
        divergence=True,
        vorticity=True,
        qcriterion=True,
        progress_bar=True,
    )
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

    derv = mesh.compute_derivative(
        scalars='vector',
        gradient='gradienttest',
        divergence='divergencetest',
        vorticity='vorticitytest',
        qcriterion='qcriteriontest',
        progress_bar=True,
    )
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

    grad = mesh.compute_derivative(scalars='Elevation', gradient=True, progress_bar=True)
    assert 'gradient' in grad.array_names
    assert np.shape(grad['gradient'])[0] == mesh.n_points
    assert np.shape(grad['gradient'])[1] == 3

    grad = mesh.compute_derivative(
        scalars='Elevation',
        gradient=True,
        faster=True,
        progress_bar=True,
    )
    assert 'gradient' in grad.array_names
    assert np.shape(grad['gradient'])[0] == mesh.n_points
    assert np.shape(grad['gradient'])[1] == 3

    grad = mesh.compute_derivative(scalars='vector', gradient=True, faster=True, progress_bar=True)
    assert 'gradient' in grad.array_names
    assert np.shape(grad['gradient'])[0] == mesh.n_points
    assert np.shape(grad['gradient'])[1] == 9

    with pytest.raises(ValueError):  # noqa: PT011
        grad = mesh.compute_derivative(scalars='Elevation', gradient=False, progress_bar=True)

    with pytest.raises(TypeError):
        derv = mesh.compute_derivative(object)

    mesh.point_data.clear()
    with pytest.raises(MissingDataError):
        derv = mesh.compute_derivative()


@pytest.mark.parametrize('rebase_coordinates', [True, False])
def test_extract_subset(uniform, rebase_coordinates):
    offset = (1, 2, 3)
    dict_ = {'foo': 'bar'}
    origin = (1.1, 2.2, 3.3)
    uniform.user_dict = dict_
    uniform.offset = offset
    uniform.origin = origin

    new_offset = (2, 4, 6)
    new_dims = 4, 3, 2
    extent = pv.ImageData(dimensions=new_dims, offset=new_offset).extent
    voi = uniform.extract_subset(extent, progress_bar=True, rebase_coordinates=rebase_coordinates)
    assert isinstance(voi, pv.ImageData)
    assert voi is not uniform
    assert voi.spacing == uniform.spacing
    assert voi.dimensions == new_dims
    assert voi.array_names == uniform.array_names
    assert voi.user_dict == dict_

    if rebase_coordinates:
        # Test that we fix the confusing issue from extents in
        #   https://gitlab.kitware.com/vtk/vtk/-/issues/17938
        assert voi.origin != origin
        assert voi.origin == voi.bounds[::2]
        assert voi.offset != new_offset
        assert voi.offset == (0, 0, 0)
    else:
        assert voi.origin == origin
        assert voi.offset == new_offset

    # Test same output as using slice_index
    cropped = uniform.slice_index(
        [extent[0], extent[1] + 1],
        [extent[2], extent[3] + 1],
        [extent[4], extent[5] + 1],
        index_mode='extent',
        rebase_coordinates=rebase_coordinates,
    )
    assert cropped == voi

    # Test same output as using crop
    cropped = uniform.crop(extent=extent, rebase_coordinates=rebase_coordinates)
    assert cropped == voi


def test_gaussian_smooth_output_type():
    volume = examples.load_uniform()
    volume_smooth = volume.gaussian_smooth()
    assert isinstance(volume_smooth, pv.ImageData)
    volume_smooth = volume.gaussian_smooth(scalars='Spatial Point Data')
    assert isinstance(volume_smooth, pv.ImageData)


def test_gaussian_smooth_constant_data():
    point_data = np.ones((10, 10, 10))
    volume = pv.ImageData(dimensions=(10, 10, 10))
    volume.point_data['point_data'] = point_data.flatten(order='F')
    volume_smoothed = volume.gaussian_smooth()
    assert np.allclose(volume.point_data['point_data'], volume_smoothed.point_data['point_data'])


def test_gaussian_smooth_outlier():
    point_data = np.ones((10, 10, 10))
    point_data[4, 4, 4] = 100
    volume = pv.ImageData(dimensions=(10, 10, 10))
    volume.point_data['point_data'] = point_data.flatten(order='F')
    volume_smoothed = volume.gaussian_smooth()
    assert volume_smoothed.get_data_range()[1] < volume.get_data_range()[1]


def test_gaussian_smooth_cell_data_specified():
    point_data = np.zeros((10, 10, 10))
    cell_data = np.zeros((9, 9, 9))
    volume = pv.ImageData(dimensions=(10, 10, 10))
    volume.point_data['point_data'] = point_data.flatten(order='F')
    volume.cell_data['cell_data'] = cell_data.flatten(order='F')
    with pytest.raises(ValueError):  # noqa: PT011
        volume.gaussian_smooth(scalars='cell_data')


def test_gaussian_smooth_cell_data_active():
    point_data = np.zeros((10, 10, 10))
    cell_data = np.zeros((9, 9, 9))
    volume = pv.ImageData(dimensions=(10, 10, 10))
    volume.point_data['point_data'] = point_data.flatten(order='F')
    volume.cell_data['cell_data'] = cell_data.flatten(order='F')
    volume.set_active_scalars('cell_data')
    with pytest.raises(ValueError):  # noqa: PT011
        volume.gaussian_smooth()


def test_median_smooth_output_type():
    volume = examples.load_uniform()
    volume_smooth = volume.median_smooth()
    assert isinstance(volume_smooth, pv.ImageData)
    volume_smooth = volume.median_smooth(scalars='Spatial Point Data')
    assert isinstance(volume_smooth, pv.ImageData)


def test_median_smooth_constant_data():
    point_data = np.ones((10, 10, 10))
    volume = pv.ImageData(dimensions=(10, 10, 10))
    volume.point_data['point_data'] = point_data.flatten(order='F')
    volume_smoothed = volume.median_smooth()
    assert np.array_equal(
        volume.point_data['point_data'], volume_smoothed.point_data['point_data']
    )


def test_median_smooth_outlier():
    point_data = np.ones((10, 10, 10))
    point_data_outlier = point_data.copy()
    point_data_outlier[4, 4, 4] = 100
    volume = pv.ImageData(dimensions=(10, 10, 10))
    volume.point_data['point_data'] = point_data.flatten(order='F')
    volume_outlier = pv.ImageData(dimensions=(10, 10, 10))
    volume_outlier.point_data['point_data'] = point_data_outlier.flatten(order='F')
    volume_outlier_smoothed = volume_outlier.median_smooth()
    assert np.array_equal(
        volume.point_data['point_data'],
        volume_outlier_smoothed.point_data['point_data'],
    )


def test_image_dilate_erode_output_type():
    point_data = np.zeros((10, 10, 10))
    point_data[4, 4, 4] = 1
    volume = pv.ImageData(dimensions=(10, 10, 10))
    volume.point_data['point_data'] = point_data.flatten(order='F')
    with pytest.warns(PyVistaDeprecationWarning, match='image_dilate_erode is deprecated'):
        volume_dilate_erode = volume.image_dilate_erode()
    assert isinstance(volume_dilate_erode, pv.ImageData)
    with pytest.warns(PyVistaDeprecationWarning, match='image_dilate_erode is deprecated'):
        volume_dilate_erode = volume.image_dilate_erode(scalars='point_data')
    assert isinstance(volume_dilate_erode, pv.ImageData)


def test_image_dilate_erode_dilation():
    point_data = np.zeros((10, 10, 10))
    point_data[4, 4, 4] = 1
    point_data_dilated = point_data.copy()
    point_data_dilated[3:6, 3:6, 4] = 1  # "activate" all voxels within diameter 3 around (4,4,4)
    point_data_dilated[3:6, 4, 3:6] = 1
    point_data_dilated[4, 3:6, 3:6] = 1
    volume = pv.ImageData(dimensions=(10, 10, 10))
    volume.point_data['point_data'] = point_data.flatten(order='F')
    with pytest.warns(PyVistaDeprecationWarning, match='image_dilate_erode is deprecated'):
        volume_dilated = volume.image_dilate_erode()  # default is binary dilation
    assert np.array_equal(
        volume_dilated.point_data['point_data'],
        point_data_dilated.flatten(order='F'),
    )


def test_image_dilate_erode_erosion():
    point_data = np.zeros((10, 10, 10))
    point_data[4, 4, 4] = 1
    point_data_eroded = np.zeros((10, 10, 10))
    volume = pv.ImageData(dimensions=(10, 10, 10))
    volume.point_data['point_data'] = point_data.flatten(order='F')
    with pytest.warns(PyVistaDeprecationWarning, match='image_dilate_erode is deprecated'):
        volume_eroded = volume.image_dilate_erode(0, 1)  # binary erosion
    assert np.array_equal(
        volume_eroded.point_data['point_data'],
        point_data_eroded.flatten(order='F'),
    )


def test_image_dilate_erode_cell_data_specified():
    point_data = np.zeros((10, 10, 10))
    cell_data = np.zeros((9, 9, 9))
    volume = pv.ImageData(dimensions=(10, 10, 10))
    volume.point_data['point_data'] = point_data.flatten(order='F')
    volume.cell_data['cell_data'] = cell_data.flatten(order='F')
    with pytest.warns(PyVistaDeprecationWarning, match='image_dilate_erode is deprecated'):
        with pytest.raises(ValueError):  # noqa: PT011
            volume.image_dilate_erode(scalars='cell_data')


def test_image_dilate_erode_cell_data_active():
    point_data = np.zeros((10, 10, 10))
    cell_data = np.zeros((9, 9, 9))
    volume = pv.ImageData(dimensions=(10, 10, 10))
    volume.point_data['point_data'] = point_data.flatten(order='F')
    volume.cell_data['cell_data'] = cell_data.flatten(order='F')
    volume.set_active_scalars('cell_data')
    with pytest.warns(PyVistaDeprecationWarning, match='image_dilate_erode is deprecated'):
        with pytest.raises(ValueError):  # noqa: PT011
            volume.image_dilate_erode()


def test_image_threshold_output_type(uniform):
    threshold = 10  # 'random' value
    volume_thresholded = uniform.image_threshold(threshold)
    assert isinstance(volume_thresholded, pv.ImageData)
    volume_thresholded = uniform.image_threshold(threshold, scalars='Spatial Point Data')
    assert isinstance(volume_thresholded, pv.ImageData)


def test_image_threshold_raises(uniform):
    match = 'Threshold must have one or two values, got 3.'
    with pytest.raises(ValueError, match=match):
        uniform.image_threshold([1, 2, 3])


@pytest.mark.parametrize('value_dtype', [float, int])
@pytest.mark.parametrize('array_dtype', [float, int, np.uint8])
def test_image_threshold_dtype(value_dtype, array_dtype):
    image = pv.ImageData(dimensions=(2, 2, 2))
    thresh_value = value_dtype(4)
    assert type(thresh_value) is value_dtype

    data_array = np.array(range(8), dtype=array_dtype)
    image['Data'] = data_array

    thresh = image.image_threshold(thresh_value)
    assert thresh['Data'].dtype == np.dtype(array_dtype)

    expected_array = [0, 0, 0, 0, 1, 1, 1, 1]
    actual_array = thresh['Data']
    assert np.array_equal(actual_array, expected_array)

    assert image['Data'].dtype == thresh['Data'].dtype


def test_image_threshold_wrong_threshold_length():
    threshold = (10, 10, 10)  # tuple with too many values
    volume = examples.load_uniform()
    with pytest.raises(ValueError):  # noqa: PT011
        volume.image_threshold(threshold)


def test_image_threshold_wrong_threshold_type():
    threshold = {'min': 0, 'max': 10}  # dict thresh
    volume = examples.load_uniform()
    with pytest.raises(TypeError):
        volume.image_threshold(threshold)


@pytest.mark.parametrize('in_value', [1, None])
@pytest.mark.parametrize('out_value', [0, None])
def test_image_threshold_upper(in_value, out_value):
    threshold = 0  # 'random' value
    array_shape = (3, 3, 3)
    in_value_location = (1, 1, 1)
    point_data = np.ones(array_shape)
    in_value_mask = np.zeros(array_shape, dtype=bool)
    in_value_mask[in_value_location] = True
    point_data[in_value_mask] = 100  # the only 'in' value
    point_data[~in_value_mask] = -100  # out values
    volume = pv.ImageData(dimensions=array_shape)
    volume.point_data['point_data'] = point_data.flatten(order='F')
    point_data_thresholded = point_data.copy()
    if in_value is not None:
        point_data_thresholded[in_value_mask] = in_value
    if out_value is not None:
        point_data_thresholded[~in_value_mask] = out_value
    volume_thresholded = volume.image_threshold(threshold, in_value=in_value, out_value=out_value)
    assert np.array_equal(
        volume_thresholded.point_data['point_data'],
        point_data_thresholded.flatten(order='F'),
    )


@pytest.mark.parametrize('in_value', [1, None])
@pytest.mark.parametrize('out_value', [0, None])
def test_image_threshold_between(in_value, out_value):
    threshold = [-10, 10]  # 'random' values
    array_shape = (3, 3, 3)
    in_value_location = (1, 1, 1)
    low_value_location = (1, 1, 2)
    point_data = np.ones(array_shape)
    in_value_mask = np.zeros(array_shape, dtype=bool)
    in_value_mask[in_value_location] = True
    point_data[in_value_mask] = 0  # the only 'in' value
    point_data[~in_value_mask] = 100  # out values
    point_data[low_value_location] = -100  # add a value below the threshold also
    volume = pv.ImageData(dimensions=array_shape)
    volume.point_data['point_data'] = point_data.flatten(order='F')
    point_data_thresholded = point_data.copy()
    if in_value is not None:
        point_data_thresholded[in_value_mask] = in_value
    if out_value is not None:
        point_data_thresholded[~in_value_mask] = out_value
    volume_thresholded = volume.image_threshold(threshold, in_value=in_value, out_value=out_value)
    assert np.array_equal(
        volume_thresholded.point_data['point_data'],
        point_data_thresholded.flatten(order='F'),
    )


def test_extract_subset_structured():
    structured = examples.load_structured()
    voi = structured.extract_subset([0, 3, 1, 4, 0, 1])
    assert isinstance(voi, pv.StructuredGrid)
    assert voi.dimensions == (4, 4, 1)


@pytest.fixture
def structured_grids_split_coincident():
    """Two structured grids which are coincident along second axis (axis=1), and
    the grid from which they were extracted.
    """
    structured = examples.load_structured()
    point_data = (np.ones((80, 80)) * np.arange(0, 80)).ravel(order='F')
    cell_data = (np.ones((79, 79)) * np.arange(0, 79)).T.ravel(order='F')
    structured.point_data['point_data'] = point_data
    structured.cell_data['cell_data'] = cell_data
    voi_1 = structured.extract_subset([0, 80, 0, 40, 0, 1])
    voi_2 = structured.extract_subset([0, 80, 40, 80, 0, 1])
    return voi_1, voi_2, structured


@pytest.fixture
def structured_grids_split_disconnected():
    """Two structured grids which are disconnected."""
    structured = examples.load_structured()
    point_data = (np.ones((80, 80)) * np.arange(0, 80)).ravel(order='F')
    cell_data = (np.ones((79, 79)) * np.arange(0, 79)).T.ravel(order='F')
    structured.point_data['point_data'] = point_data
    structured.cell_data['cell_data'] = cell_data
    voi_1 = structured.extract_subset([0, 80, 0, 40, 0, 1])
    voi_2 = structured.extract_subset([0, 80, 45, 80, 0, 1])
    return voi_1, voi_2


def test_concatenate_structured(structured_grids_split_coincident):
    voi_1, voi_2, structured = structured_grids_split_coincident
    joined = voi_1.concatenate(voi_2, axis=1)
    assert structured.points == pytest.approx(joined.points)
    assert structured.volume == pytest.approx(joined.volume)
    assert structured.point_data['point_data'] == pytest.approx(joined.point_data['point_data'])
    assert structured.cell_data['cell_data'] == pytest.approx(joined.cell_data['cell_data'])


def test_concatenate_structured_bad_dimensions(structured_grids_split_coincident):
    voi_1, voi_2, _structured = structured_grids_split_coincident

    # test invalid dimensions
    with pytest.raises(ValueError):  # noqa: PT011
        voi_1.concatenate(voi_2, axis=0)

    with pytest.raises(ValueError):  # noqa: PT011
        voi_1.concatenate(voi_2, axis=2)


def test_concatenate_structured_bad_inputs(structured_grids_split_coincident):
    voi_1, voi_2, _structured = structured_grids_split_coincident
    with pytest.raises(RuntimeError):
        voi_1.concatenate(voi_2, axis=3)


def test_concatenate_structured_bad_point_data(structured_grids_split_coincident):
    voi_1, voi_2, _structured = structured_grids_split_coincident
    voi_1['point_data'] = voi_1['point_data'] * 2.0
    with pytest.raises(RuntimeError):
        voi_1.concatenate(voi_2, axis=1)


def test_concatenate_structured_disconnected(structured_grids_split_disconnected):
    voi_1, voi_2 = structured_grids_split_disconnected
    with pytest.raises(RuntimeError):
        voi_1.concatenate(voi_2, axis=1)


def test_concatenate_structured_different_arrays(structured_grids_split_coincident):
    voi_1, voi_2, _structured = structured_grids_split_coincident
    point_data = voi_1.point_data.pop('point_data')
    with pytest.raises(RuntimeError):
        voi_1.concatenate(voi_2, axis=1)

    voi_1.point_data['point_data'] = point_data
    voi_1.cell_data.remove('cell_data')
    with pytest.raises(RuntimeError):
        voi_1.concatenate(voi_2, axis=1)


def test_structured_add_non_grid():
    grid = examples.load_structured()
    merged = grid + examples.load_hexbeam()
    assert isinstance(merged, pv.UnstructuredGrid)


def test_poly_data_strip():
    mesh = examples.load_airplane()
    slc = mesh.slice(normal='z', origin=(0, 0, -10))
    stripped = slc.strip(progress_bar=True)
    assert stripped.n_cells == 1


def test_shrink():
    mesh = pv.Sphere()
    shrunk = mesh.shrink(shrink_factor=0.8, progress_bar=True)
    assert shrunk.n_cells == mesh.n_cells
    assert shrunk.area < mesh.area
    mesh = examples.load_uniform()
    shrunk = mesh.shrink(shrink_factor=0.8, progress_bar=True)
    assert shrunk.n_cells == mesh.n_cells
    assert shrunk.volume < mesh.volume


def test_tessellate():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 0.5, 0.0],
            [1.5, 1.5, 0.0],
            [0.5, 1.5, 0.0],
        ],
    )
    cells = np.array([6, 0, 1, 2, 3, 4, 5])
    cell_types = np.array([CellType.QUADRATIC_TRIANGLE])
    ugrid = pv.UnstructuredGrid(cells, cell_types, points)
    tessellated = ugrid.tessellate(progress_bar=True)
    assert tessellated.n_cells > ugrid.n_cells
    assert tessellated.n_points > ugrid.n_points
    assert ugrid.tessellate(max_n_subdivide=6).n_cells > tessellated.n_cells
    assert ugrid.tessellate(merge_points=False).n_points > tessellated.n_points
    pdata = pv.PolyData()
    with pytest.raises(TypeError):
        tessellated = pdata.tessellate(progress_bar=True)


def test_extrude_rotate():
    resolution = 4
    line = pv.Line(pointa=(0, 0, 0), pointb=(1, 0, 0))

    with pytest.raises(ValueError):  # noqa: PT011
        line.extrude_rotate(resolution=0, capping=True)

    poly = line.extrude_rotate(resolution=resolution, progress_bar=True, capping=True)
    assert poly.n_cells == line.n_points - 1
    assert poly.n_points == (resolution + 1) * line.n_points

    translation = 10.0
    dradius = 1.0
    poly = line.extrude_rotate(
        translation=translation,
        dradius=dradius,
        progress_bar=True,
        capping=True,
    )
    zmax = poly.bounds.z_max
    assert zmax == translation
    xmax = poly.bounds.x_max
    assert xmax == line.bounds.x_max + dradius

    poly = line.extrude_rotate(angle=90.0, progress_bar=True, capping=True)
    xmin = poly.bounds.x_min
    xmax = poly.bounds.x_max
    ymin = poly.bounds.y_min
    ymax = poly.bounds.y_max
    assert xmin == line.bounds.x_min
    assert xmax == line.bounds.x_max
    assert ymin == line.bounds.x_min
    assert ymax == line.bounds.x_max

    rotation_axis = (0, 1, 0)
    poly = line.extrude_rotate(
        rotation_axis=rotation_axis,
        resolution=resolution,
        progress_bar=True,
        capping=True,
    )
    assert poly.n_cells == line.n_points - 1
    assert poly.n_points == (resolution + 1) * line.n_points

    with pytest.raises(ValueError):  # noqa: PT011
        line.extrude_rotate(rotation_axis=[1, 2], capping=True)


def test_extrude_rotate_inplace():
    resolution = 4
    poly = pv.Line(pointa=(0, 0, 0), pointb=(1, 0, 0))
    old_line = poly.copy()
    poly.extrude_rotate(resolution=resolution, inplace=True, progress_bar=True, capping=True)
    assert poly.n_cells == old_line.n_points - 1
    assert poly.n_points == (resolution + 1) * old_line.n_points


def test_extrude_trim():
    direction = (0, 0, 1)
    mesh = pv.Plane(
        center=(0, 0, 0),
        direction=direction,
        i_size=1,
        j_size=1,
        i_resolution=10,
        j_resolution=10,
    )
    trim_surface = pv.Plane(
        center=(0, 0, 1),
        direction=direction,
        i_size=2,
        j_size=2,
        i_resolution=20,
        j_resolution=20,
    )
    poly = mesh.extrude_trim(direction, trim_surface)
    assert np.isclose(poly.volume, 1.0)


@pytest.mark.parametrize('extrusion', ['boundary_edges', 'all_edges'])
@pytest.mark.parametrize(
    'capping',
    ['intersection', 'minimum_distance', 'maximum_distance', 'average_distance'],
)
def test_extrude_trim_strategy(extrusion, capping):
    direction = (0, 0, 1)
    mesh = pv.Plane(
        center=(0, 0, 0),
        direction=direction,
        i_size=1,
        j_size=1,
        i_resolution=10,
        j_resolution=10,
    )
    trim_surface = pv.Plane(
        center=(0, 0, 1),
        direction=direction,
        i_size=2,
        j_size=2,
        i_resolution=20,
        j_resolution=20,
    )
    poly = mesh.extrude_trim(direction, trim_surface, extrusion=extrusion, capping=capping)
    assert isinstance(poly, pv.PolyData)
    assert poly.n_cells
    assert poly.n_points


def test_extrude_trim_catch():
    direction = (0, 0, 1)
    mesh = pv.Plane()
    trim_surface = pv.Plane()
    with pytest.raises(ValueError):  # noqa: PT011
        _ = mesh.extrude_trim(direction, trim_surface, extrusion='Invalid strategy')
    with pytest.raises(TypeError, match='Invalid type'):
        _ = mesh.extrude_trim(direction, trim_surface, extrusion=0)
    with pytest.raises(ValueError):  # noqa: PT011
        _ = mesh.extrude_trim(direction, trim_surface, capping='Invalid strategy')
    with pytest.raises(TypeError, match='Invalid type'):
        _ = mesh.extrude_trim(direction, trim_surface, capping=0)
    with pytest.raises(TypeError):
        _ = mesh.extrude_trim('foobar', trim_surface)
    with pytest.raises(TypeError):
        _ = mesh.extrude_trim([1, 2], trim_surface)


def test_extrude_trim_inplace():
    direction = (0, 0, 1)
    mesh = pv.Plane(
        center=(0, 0, 0),
        direction=direction,
        i_size=1,
        j_size=1,
        i_resolution=10,
        j_resolution=10,
    )
    trim_surface = pv.Plane(
        center=(0, 0, 1),
        direction=direction,
        i_size=2,
        j_size=2,
        i_resolution=20,
        j_resolution=20,
    )
    mesh.extrude_trim(direction, trim_surface, inplace=True, progress_bar=True)
    assert np.isclose(mesh.volume, 1.0)


@pytest.mark.parametrize('inplace', [True, False])
def test_subdivide_adaptive(sphere, inplace):
    orig_n_faces = sphere.n_faces_strict
    sub = sphere.subdivide_adaptive(
        max_edge_len=0.01,
        max_tri_area=0.001,
        max_n_tris=100000,
        max_n_passes=2,
        inplace=inplace,
        progress_bar=True,
    )
    assert sub.n_faces_strict > orig_n_faces
    if inplace:
        assert sphere.n_faces_strict == sub.n_faces_strict


def test_invalid_subdivide_adaptive(cube):
    # check non-triangulated
    with pytest.raises(NotAllTrianglesError):
        cube.subdivide_adaptive()


def test_collision(sphere):
    moved_sphere = sphere.translate((0.5, 0, 0), inplace=False)
    output, n_collision = sphere.collision(moved_sphere)
    assert isinstance(output, pv.PolyData)
    assert n_collision > 40
    assert 'ContactCells' in output.field_data

    # test no collision
    moved_sphere.translate((1000, 0, 0), inplace=True)
    _, n_collision = sphere.collision(moved_sphere)
    assert not n_collision


def test_collision_solid_non_triangle(hexbeam):
    # test non-triangular mesh with a unstructured grid
    cube = pv.Cube()
    output, n_collision = cube.collision(hexbeam)
    assert isinstance(output, pv.PolyData)
    assert n_collision > 40
    assert 'ContactCells' in output.field_data
    assert output.is_all_triangles


def test_reconstruct_surface_poly(sphere):
    pc = pv.wrap(sphere.points)
    surf = pc.reconstruct_surface(nbr_sz=10, sample_spacing=50)
    assert surf.is_all_triangles


def test_is_manifold(sphere, plane):
    assert sphere.is_manifold
    assert not plane.is_manifold


def test_reconstruct_surface_unstructured():
    mesh = examples.load_hexbeam().reconstruct_surface()
    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_points


def test_integrate_data_datasets(datasets):
    """Test multiple dataset types."""
    for dataset in datasets:
        integrated = dataset.integrate_data()
        if 'Area' in integrated.array_names:
            assert integrated['Area'] > 0
        elif 'Volume' in integrated.array_names:
            assert integrated['Volume'] > 0
        else:
            msg = 'Unexpected integration'
            raise ValueError(msg)


def test_integrate_data():
    """Test specific case."""
    # sphere with radius = 0.5, area = pi
    # increase resolution to increase precision
    sphere = pv.Sphere(theta_resolution=100, phi_resolution=100)
    sphere.cell_data['cdata'] = 2 * np.ones(sphere.n_cells)
    sphere.point_data['pdata'] = 3 * np.ones(sphere.n_points)

    integrated = sphere.integrate_data()
    assert np.isclose(integrated['Area'], np.pi, rtol=1e-3)
    assert np.isclose(integrated['cdata'], 2 * np.pi, rtol=1e-3)
    assert np.isclose(integrated['pdata'], 3 * np.pi, rtol=1e-3)


def test_align():
    # Create a simple mesh
    source = pv.Cylinder(resolution=30).triangulate().subdivide(1)
    transformed = source.rotate_y(20).rotate_z(25).translate([-0.75, -0.5, 0.5])

    # Perform ICP registration
    aligned = transformed.align(source)

    _, matrix = transformed.align(source, return_matrix=True)
    assert isinstance(matrix, np.ndarray)

    # Check if the number of points in the aligned mesh is the same as the original mesh
    assert source.n_points == aligned.n_points

    _, closest_points = aligned.find_closest_cell(source.points, return_closest_point=True)
    dist = np.linalg.norm(source.points - closest_points, axis=1)
    assert np.abs(dist).mean() < 1e-3


def test_align_xyz():
    mesh = examples.download_oblique_cone()
    aligned = mesh.align_xyz()
    assert np.allclose(aligned.center, (0, 0, 0))

    aligned = mesh.align_xyz(centered=False)
    assert np.allclose(aligned.center, mesh.center)


def test_align_xyz_return_matrix():
    mesh = examples.download_oblique_cone()
    initial_bounds = mesh.bounds

    aligned, matrix = mesh.align_xyz(return_matrix=True)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (4, 4)

    inverse_matrix = pv.Transform(matrix).inverse_matrix
    inverted_mesh = aligned.transform(inverse_matrix, inplace=False)
    inverted_bounds = inverted_mesh.bounds

    assert np.allclose(inverted_bounds, initial_bounds)


@pytest.mark.parametrize(
    ('as_composite', 'mesh_type'), [(True, pv.MultiBlock), (False, pv.PolyData)]
)
def test_bounding_box_as_composite(sphere, as_composite, mesh_type):
    box = sphere.bounding_box(as_composite=as_composite)
    assert isinstance(box, mesh_type)
    assert box.bounds == sphere.bounds


def test_oriented_bounding_box():
    rotation = pv.transformations.axis_angle_rotation((1, 2, 3), 30)
    box_mesh = pv.Cube(x_length=1, y_length=2, z_length=3)
    box_mesh.transform(rotation, inplace=True)
    obb = box_mesh.oriented_bounding_box()
    assert obb.bounds == box_mesh.bounds


@pytest.mark.parametrize('oriented', [True, False])
@pytest.mark.parametrize('as_composite', [True, False])
def test_bounding_box_return_meta(oriented, as_composite):
    # Generate a random rotation matrix
    vector = np.random.default_rng().random((3,))
    angle = np.random.default_rng().random((1,)) * 360
    rotation = pv.transformations.axis_angle_rotation(vector, angle)

    # Transform a box manually and get its OBB
    box_mesh = pv.Cube(x_length=1, y_length=2, z_length=3)
    box_mesh.transform(rotation, inplace=True)
    obb, point, axes = box_mesh.bounding_box(
        oriented=oriented, return_meta=True, as_composite=as_composite
    )
    ATOL = 1e-6  # Needed for numerical error from calculating the principal axes
    if oriented:
        # Test axes are equal (up to a difference in sign)
        expected_axes = pv.principal_axes(box_mesh.points)
        identity = np.abs(expected_axes @ axes.T)
        assert np.allclose(identity, np.eye(3), atol=ATOL)
    else:
        # Test identity always returned for non-oriented box
        assert np.array_equal(axes, np.eye(3))
        bnds = box_mesh.bounds
        assert np.array_equal(point, (bnds.x_min, bnds.y_min, bnds.z_min))

    # Test the returned point is one of the box's points
    if as_composite:
        assert any(point in face.points for face in obb)

        # Also test that box's normals are aligned with the axes directions
        assert np.allclose(axes[0], obb['+X'].cell_normals[0], atol=ATOL)
        assert np.allclose(axes[1], obb['+Y'].cell_normals[0], atol=ATOL)
        assert np.allclose(axes[2], obb['+Z'].cell_normals[0], atol=ATOL)
    else:
        assert point in obb.points


DELTA = 0.1


@pytest.fixture
def planar_mesh():
    # Define planar data with largest variation in x, then y
    # Use a delta to make data slightly asymmetric so the principal axes
    # are not computed as the identity matrix (we want some negative axes for the tests)
    points = np.array([[2 + DELTA, 1 + DELTA, 0], [2, -1, 0], [-2, 1, 0], [-2, -1, 0]])
    axes = pv.principal_axes(points)
    assert np.allclose(axes, [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], atol=DELTA)
    return pv.PolyData(points)


@pytest.mark.parametrize(
    ('name', 'value'),
    [
        ('axis_0_direction', [1, 0, 0]),
        ('axis_0_direction', [-1, 0, 0]),
        ('axis_1_direction', [0, 1, 0]),
        ('axis_1_direction', [0, -1, 0]),
        ('axis_2_direction', [0, 0, 1]),
        ('axis_2_direction', [0, 0, -1]),
    ],
)
def test_align_xyz_single_axis_direction(planar_mesh, name, value):
    _, matrix = planar_mesh.align_xyz(**{name: value}, return_matrix=True)
    axes = matrix[:3, :3]

    # Test that the axis has the right direction
    axis = np.flatnonzero(value)
    assert np.allclose(axes[axis], value, atol=DELTA)


def test_align_xyz_no_axis_direction(planar_mesh):
    # Test that axis-aligned principal axes with negative directions are "converted"
    # into the identity matrix (i.e. the negative directions are flipped to be positive)
    axes_in = pv.principal_axes(planar_mesh.points)
    _, matrix = planar_mesh.align_xyz(return_matrix=True)
    axes_out = matrix[:3, :3]

    identity = np.eye(3)
    assert not np.allclose(axes_in, identity, atol=DELTA)
    assert np.allclose(axes_out, identity, atol=DELTA)


def test_align_xyz_two_axis_directions(planar_mesh):
    axis_0_direction = [-1, 0, 0]
    axis_1_direction = [0, -1, 0]
    _, matrix = planar_mesh.align_xyz(
        axis_0_direction=axis_0_direction,
        axis_1_direction=axis_1_direction,
        return_matrix=True,
    )
    axes = matrix[:3, :3]
    assert np.allclose(axes, [axis_0_direction, axis_1_direction, [0, 0, 1]], atol=DELTA)

    axis_1_direction = [0, -1, 0]
    axis_2_direction = [0, 0, -1]
    _, matrix = planar_mesh.align_xyz(
        axis_1_direction=axis_1_direction,
        axis_2_direction=axis_2_direction,
        return_matrix=True,
    )
    axes = matrix[:3, :3]
    assert np.allclose(axes, [[1, 0, 0], axis_1_direction, axis_2_direction], atol=DELTA)


def test_align_xyz_three_axis_directions(planar_mesh):
    axis_2_direction = np.array((0.0, 0.0, -1.0))
    _, matrix = planar_mesh.align_xyz(
        axis_0_direction='x',
        axis_1_direction='-y',
        axis_2_direction=axis_2_direction,  # test has no effect
        return_matrix=True,
    )
    axes = matrix[:3, :3]
    assert np.allclose(axes, [[1, 0, 0], [0, -1, 0], [0, 0, -1]], atol=DELTA)

    match = (
        'Invalid `axis_2_direction` [-0. -0.  1.]. This direction results in a '
        'left-handed transformation.'
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        _ = planar_mesh.align_xyz(
            axis_0_direction='x',
            axis_1_direction='-y',
            axis_2_direction=axis_2_direction * -1,
            return_matrix=True,
        )


@pytest.mark.filterwarnings('ignore:Points is not a float type.*:UserWarning')
def test_align_xyz_swap_axes():
    # create planar data with equal variance in x and z
    points = np.array([[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]])
    _, matrix = pv.PolyData(points).align_xyz(return_matrix=True)
    axes = matrix[:3, :3]
    assert np.array_equal(axes, [[1, 0, 0], [0, 0, 1], [0, -1, 0]])  # XZY (instead of ZXY)

    # create planar data with equal variance in x and y
    points = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    _, matrix = pv.PolyData(points).align_xyz(return_matrix=True)
    axes = matrix[:3, :3]
    assert np.array_equal(axes, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # XYZ (instead of YXZ)


@pytest.mark.parametrize('x', [(1, 0, 0), (-1, 0, 0)])
@pytest.mark.parametrize('y', [(0, 1, 0), (0, -1, 0)])
@pytest.mark.parametrize('z', [(0, 0, 1), (0, 0, -1)])
@pytest.mark.parametrize('order', itertools.permutations([0, 1, 2]))
@pytest.mark.parametrize(
    ('test_case', 'values'),
    [
        ('swap_all', [1, 1, 1]),
        ('swap_none', [3, 2, 1]),
        ('swap_0_1', [2, 2, 1]),
        ('swap_1_2', [2, 1, 1]),
    ],
)
def test_swap_axes(x, y, z, order, test_case, values):
    axes = np.array((x, y, z))[list(order)]
    swapped = _swap_axes(axes, values)
    if test_case == 'swap_all':
        # All axes have the same weight, expect swap to have x-y-z order
        assert np.array_equal(np.abs(swapped), np.eye(3))
    elif test_case == 'swap_none':
        # Expect no swapping, output is input
        assert np.array_equal(axes, swapped)
    elif test_case == 'swap_0_1':
        first_index = np.flatnonzero(swapped[0])[0]
        second_index = np.flatnonzero(swapped[1])[0]
        assert first_index < second_index
    elif test_case == 'swap_1_2':
        first_index = np.flatnonzero(swapped[1])[0]
        second_index = np.flatnonzero(swapped[2])[0]
        assert first_index < second_index


def test_subdivide_tetra(tetbeam):
    grid = tetbeam.subdivide_tetra()
    assert grid.n_cells == tetbeam.n_cells * 12


def test_extract_cells_by_type(tetbeam, hexbeam):
    combined = tetbeam + hexbeam

    assert 'vtkOriginalPointIds' not in hexbeam.point_data
    assert 'vtkOriginalCellIds' not in hexbeam.cell_data

    hex_cells = combined.extract_cells_by_type(
        [
            pv.CellType.HEXAHEDRON,
            pv.CellType.BEZIER_PYRAMID,
        ]
    )
    assert np.all(hex_cells.celltypes == pv.CellType.HEXAHEDRON)

    # should be no side effects
    assert 'vtkOriginalPointIds' not in hexbeam.point_data
    assert 'vtkOriginalCellIds' not in hexbeam.cell_data

    tet_cells = combined.extract_cells_by_type(pv.CellType.TETRA)
    assert np.all(tet_cells.celltypes == pv.CellType.TETRA)

    int_array = np.array([int(pv.CellType.TETRA), int(pv.CellType.HEXAHEDRON)])
    tet_hex_cells = combined.extract_cells_by_type(int_array)
    assert pv.CellType.TETRA in tet_hex_cells.celltypes
    assert pv.CellType.HEXAHEDRON in tet_hex_cells.celltypes

    should_be_empty = combined.extract_cells_by_type(pv.CellType.BEZIER_CURVE)
    assert should_be_empty.n_cells == 0

    combined.extract_cells_by_type(1.0)
    match = 'cell_types must have integer-like values.'
    with pytest.raises(ValueError, match=re.escape(match)):
        combined.extract_cells_by_type(1.1)


def test_merge_points():
    cells = [2, 0, 1]
    celltypes = [pv.CellType.LINE]
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    pdata = pv.UnstructuredGrid(cells, celltypes, points)
    assert pdata.merge(pdata, merge_points=True, tolerance=1.0).n_points == 1
    assert pdata.merge(pdata, merge_points=True, tolerance=0.1).n_points == 2


@pytest.mark.parametrize('inplace', [True, False])
def test_merge_points_filter(inplace):
    # Set up
    cells = [2, 0, 1]
    celltypes = [pv.CellType.LINE]
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    mesh = pv.UnstructuredGrid(cells, celltypes, points)
    assert mesh.n_points == 2

    # Do test
    output = mesh.merge_points(inplace=inplace, tolerance=1.0)
    assert output.n_points == 1
    assert isinstance(mesh, pv.UnstructuredGrid)
    assert (mesh is output) == inplace


@pytest.fixture
def labeled_image():
    image = pv.ImageData(dimensions=(2, 2, 2))
    image['labels'] = [0, 3, 3, 3, 3, 0, 2, 2]
    return image


def test_sort_labels(labeled_image):
    sorted_ = labeled_image.sort_labels()
    assert np.array_equal(sorted_['packed_labels'], [2, 0, 0, 0, 0, 2, 1, 1])

    # test no data
    with pytest.raises(ValueError):  # noqa: PT011
        pv.ImageData(dimensions=(2, 2, 2)).sort_labels()

    # test single label
    labeled_image['labels'] = [0, 0, 0, 0, 0, 0, 0, 0]
    sorted_ = labeled_image.sort_labels(scalars='labels')
    assert np.array_equal(sorted_['packed_labels'], [0, 0, 0, 0, 0, 0, 0, 0])


def test_pack_labels(labeled_image):
    labeled_image['misc'] = [0, 0, 0, 0, 0, 0, 0, 0]
    packed = labeled_image.pack_labels(progress_bar=True)
    assert np.array_equal(packed['packed_labels'], [0, 2, 2, 2, 2, 0, 1, 1])
    assert 'labels' in packed.array_names
    assert 'packed_labels' in packed.array_names


def test_pack_labels_inplace(uniform):
    assert uniform.pack_labels() is not uniform  # default False
    assert uniform.pack_labels(inplace=False) is not uniform
    assert uniform.pack_labels(inplace=True) is uniform


def test_pack_labels_output_scalars(labeled_image):
    packed = labeled_image.pack_labels(output_scalars='foo')
    assert np.array_equal(packed['foo'], [0, 2, 2, 2, 2, 0, 1, 1])
    assert 'labels' in packed.array_names
    assert packed.active_scalars_name == 'foo'

    with pytest.raises(TypeError):
        labeled_image.pack_labels(output_scalars=1)


def test_pack_labels_preference(uniform):
    uniform.rename_array('Spatial Point Data', 'labels_in')
    uniform.rename_array('Spatial Cell Data', 'labels_in')

    mesh = uniform.copy()
    packed = mesh.pack_labels(preference='point')
    expected_shape = mesh.point_data['labels_in'].shape
    actual_shape = packed.point_data['packed_labels'].shape
    assert np.array_equal(actual_shape, expected_shape)

    mesh = uniform.copy()
    packed = mesh.pack_labels(preference='cell')
    expected_shape = mesh.cell_data['labels_in'].shape
    actual_shape = packed.cell_data['packed_labels'].shape
    assert np.array_equal(actual_shape, expected_shape)

    # test point preference without point data
    mesh = uniform.copy()
    mesh.point_data.remove('labels_in')
    packed = mesh.pack_labels(preference='point')
    expected_shape = mesh.cell_data['labels_in'].shape
    actual_shape = packed.cell_data['packed_labels'].shape
    assert np.array_equal(actual_shape, expected_shape)


@pytest.mark.parametrize('coloring_mode', ['index', 'cycle', None])
def test_color_labels(uniform, coloring_mode):
    default_cmap = pv.get_cmap_safe('glasbey_category10')
    original_scalars_name = uniform.active_scalars_name

    if coloring_mode == 'index':
        # Test invalid input
        match = (
            "Index coloring mode cannot be used with scalars 'Spatial Point Data'. "
            'Scalars must be positive integers \n'
            'and the max value (729.0) must be less than the number of colors (256).'
        )
        with pytest.raises(ValueError, match=re.escape(match)):
            uniform.color_labels(coloring_mode=coloring_mode)

        # Use pack labels so that index mapping can be used
        uniform = uniform.pack_labels(output_scalars=original_scalars_name)

    colored_mesh = uniform.color_labels(coloring_mode=coloring_mode)
    assert colored_mesh is not uniform
    assert uniform.active_scalars_name == original_scalars_name
    colors_name = original_scalars_name + '_rgb'
    assert [0, 0, 0] not in np.unique(colored_mesh[colors_name], axis=0).tolist()

    label_ids = np.unique(uniform.active_scalars)
    for i, label_id in enumerate(label_ids):
        data_ids = np.where(uniform[original_scalars_name] == label_id)[0]
        expected_color_rgb = pv.Color(default_cmap.colors[i]).int_rgb
        for data_id in data_ids:
            actual_rgba = colored_mesh[colors_name][data_id]
            assert np.allclose(actual_rgba, expected_color_rgb)

    # Test in place
    colored_mesh = uniform.color_labels(coloring_mode=coloring_mode, inplace=True)
    assert colored_mesh is uniform


VIRIDIS_RGB = [pv.Color(c).int_rgb for c in pv.get_cmap_safe('viridis').colors]
COLORS_DICT = {
    0: 'red',
    1: (0, 0, 0),
    2: 'blue',
    3: (1.0, 1.0, 1.0),
    4: 'orange',
    5: 'green',
}
COLORS_DICT_RGB = [pv.Color(c).int_rgb for c in COLORS_DICT.values()]
RED_RGB = pv.Color('red').int_rgb


@pytest.mark.parametrize(
    ('color_input', 'expected_rgb'),
    [
        ('viridis', VIRIDIS_RGB),
        (COLORS_DICT, COLORS_DICT_RGB),
        (COLORS_DICT_RGB, COLORS_DICT_RGB),
        ('red', [RED_RGB, RED_RGB, RED_RGB, RED_RGB]),
    ],
    ids=['cmap', 'dict', 'sequence', 'named_color'],
)
def test_color_labels_inputs(labeled_image, color_input, expected_rgb):
    label_scalars = labeled_image.active_scalars
    colored = labeled_image.color_labels(color_input)
    color_scalars = colored.active_scalars
    for id_ in np.unique(label_scalars):
        assert np.allclose(color_scalars[label_scalars == id_], expected_rgb[id_])


@pytest.mark.parametrize('color_type', ['int_rgb', 'int_rgba', 'float_rgb', 'float_rgba'])
def test_color_labels_color_type_partial_dict(labeled_image, color_type):
    input_scalars_name = labeled_image.active_scalars_name
    colored = labeled_image.color_labels({0: RED_RGB}, color_type=color_type)
    color_scalars = colored.active_scalars
    color_scalars_name = colored.active_scalars_name
    unique = np.unique(color_scalars, axis=0)

    expected_color = getattr(pv.Color(RED_RGB), color_type)
    if 'float' in color_type:
        assert np.array_equal(expected_color, unique[0])
        assert np.array_equal([np.nan] * len(expected_color), unique[1], equal_nan=True)
        assert color_scalars.dtype == float
    else:
        assert np.array_equal(expected_color, unique[1])
        assert np.array_equal([0] * len(expected_color), unique[0])
        assert color_scalars.dtype == np.uint8
    if 'rgba' in color_type:
        assert color_scalars_name == input_scalars_name + '_rgba'
    else:
        assert color_scalars_name == input_scalars_name + '_rgb'


@pytest.mark.parametrize('color_type', ['float_rgb', 'float_rgba'])
def test_color_labels_color_type_cmap(color_type):
    labels = pv.ImageData(dimensions=(256, 1, 1))
    labels['256'] = range(256)
    colored = labels.color_labels('viridis', color_type=color_type)
    cmap_colors = pv.get_cmap_safe('viridis').colors
    for i, color in enumerate(colored.active_scalars):
        expected_color = cmap_colors[i]
        if 'rgba' in color_type:
            expected_color.append(1.0)
        assert np.array_equal(color, expected_color)


LABEL_DATA = [-1, -2, 1]


@pytest.mark.parametrize(
    ('negative_indexing', 'cmap_index'),
    [(True, LABEL_DATA), (False, np.argsort(LABEL_DATA))],
)
def test_color_labels_negative_index(negative_indexing, cmap_index):
    labels = pv.ImageData(dimensions=(3, 1, 1))
    labels['data'] = LABEL_DATA
    colored = labels.color_labels('viridis', negative_indexing=negative_indexing)
    color_array = colored.active_scalars

    assert np.array_equal(color_array[0], VIRIDIS_RGB[cmap_index[0]])
    assert np.array_equal(color_array[1], VIRIDIS_RGB[cmap_index[1]])
    assert np.array_equal(color_array[2], VIRIDIS_RGB[cmap_index[2]])


def test_color_labels_scalars(uniform):
    # Test active scalars
    active_before = uniform.active_scalars_name
    for name in uniform.array_names:
        colored = uniform.color_labels(scalars=name)
        assert name in colored.active_scalars_name
    assert uniform.active_scalars_name == active_before

    # Give cell data and point data the same name
    GENERIC = 'generic'
    for name in uniform.array_names:
        uniform.rename_array(name, GENERIC)
    assert all(name == GENERIC for name in uniform.array_names)

    # Test preference
    for name in uniform.array_names:
        colored = uniform.color_labels(scalars=name, preference='point')
        assert GENERIC + '_rgb' in colored.point_data

        colored = uniform.color_labels(scalars=name, preference='cell')
        assert GENERIC + '_rgb' in colored.cell_data

    # Test output scalars
    CUSTOM = 'custom'
    colored = uniform.color_labels(output_scalars=CUSTOM)
    assert CUSTOM in colored.array_names


def test_color_labels_invalid_input(uniform):
    match = 'Coloring mode cannot be set when a color dictionary is specified.'
    with pytest.raises(TypeError, match=match):
        uniform.color_labels({}, coloring_mode='index')

    match = "Colormap 'bwr' must be a ListedColormap, got LinearSegmentedColormap instead."
    with pytest.raises(TypeError, match=match):
        uniform.color_labels('bwr')

    match = 'color must be an instance of'
    with pytest.raises(TypeError, match=match):
        uniform.color_labels([[1]])
    match = (
        'Invalid colors. Colors must be one of:\n'
        '  - sequence of color-like values,\n'
        '  - dict with color-like values,\n'
        '  - named colormap string.'
    )
    with pytest.raises(ValueError, match=match):
        uniform.color_labels('fake')

    match = "Negative indexing is not supported with 'cycle' mode enabled."
    with pytest.raises(ValueError, match=match):
        uniform.color_labels(coloring_mode='cycle', negative_indexing=True)

    match = (
        'Multi-component scalars are not supported for coloring. '
        'Scalar array Normals must be one-dimensional.'
    )
    with pytest.raises(ValueError, match=match):
        pv.Sphere().color_labels(scalars='Normals')


@pytest.mark.parametrize('color_type', ['float_rgb', 'int_rgba'])
def test_color_labels_return_dict(labeled_image, color_type):
    expected_keys = np.unique(labeled_image.active_scalars)

    input_colors = pv.plotting.colors.matplotlib_default_colors
    input_keys = list(range(len(input_colors)))
    assert set(expected_keys) < set(input_keys)

    mapping_in = dict(zip(input_keys, input_colors, strict=True))
    colored_mesh, mapping_out = labeled_image.color_labels(
        mapping_in, return_dict=True, color_type=color_type
    )
    assert isinstance(colored_mesh, type(labeled_image))
    assert isinstance(mapping_out, dict)
    assert set(mapping_out.keys()) < set(mapping_in.keys())

    for key in expected_keys:
        expected_color = getattr(pv.Color(mapping_in[key]), color_type)
        actual_color = mapping_out[key]
        assert actual_color == expected_color


@pytest.fixture
def frog_tissues_image():
    return examples.load_frog_tissues()


@pytest.fixture
def frog_tissues_contour(frog_tissues_image):
    return frog_tissues_image.contour_labels(smoothing=False)


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
def test_voxelize_binary_mask_no_reference(frog_tissues_contour):
    mask = frog_tissues_contour.voxelize_binary_mask()
    assert np.allclose(mask.points_to_cells().bounds, frog_tissues_contour.bounds)


def test_voxelize_binary_mask_dimensions(sphere):
    dims = (10, 11, 12)
    mask = sphere.voxelize_binary_mask(dimensions=dims)
    assert np.allclose(mask.points_to_cells().bounds, sphere.bounds)
    assert mask.dimensions == dims


def test_voxelize_binary_mask_spacing(ant):
    # Test default
    mask_no_input = ant.voxelize_binary_mask()
    expected_mask = ant.voxelize_binary_mask(cell_length_percentile=0.1)
    assert mask_no_input.spacing == expected_mask.spacing

    # Test cell length
    mask_percentile_20 = ant.voxelize_binary_mask(cell_length_percentile=0.2)
    mask_percentile_50 = ant.voxelize_binary_mask(cell_length_percentile=0.5)
    assert np.all(np.array(mask_percentile_20.spacing) < mask_percentile_50.spacing)

    # Test mesh length
    mask_fraction_200 = ant.voxelize_binary_mask(spacing=ant.length / 200)
    mask_fraction_500 = ant.voxelize_binary_mask(spacing=ant.length / 500)
    assert np.all(np.array(mask_fraction_200.spacing) > mask_fraction_500.spacing)
    # Check spacing matches mesh length. Use atol since spacing is approximate.
    assert np.allclose(mask_fraction_500.spacing, ant.length / 500, atol=1e-3)

    match = 'Spacing and cell length options cannot both be set. Set one or the other.'
    with pytest.raises(TypeError, match=match):
        ant.voxelize_binary_mask(spacing=0.1, cell_length_percentile=0.1)
    with pytest.raises(TypeError, match=match):
        ant.voxelize_binary_mask(spacing=0.1, cell_length_sample_size=ant.n_cells)


# This test is flaky because of random sampling that cannot be controlled.
# Sometimes the sampling produces the same output.
# https://github.com/pyvista/pyvista/pull/6728
@flaky_test(times=5)
def test_voxelize_binary_mask_cell_length_sample_size(ant):
    mask_samples_1 = ant.voxelize_binary_mask(cell_length_sample_size=100)
    mask_samples_2 = ant.voxelize_binary_mask(cell_length_sample_size=200)
    assert mask_samples_1.spacing != mask_samples_2.spacing

    mask_samples_1 = ant.voxelize_binary_mask(cell_length_sample_size=ant.n_cells)
    mask_samples_2 = ant.voxelize_binary_mask(cell_length_sample_size=ant.n_cells)
    assert mask_samples_1.spacing == mask_samples_2.spacing


@pytest.mark.parametrize(
    'rounding_func',
    [
        np.round,
        np.ceil,
        np.floor,
        lambda x: [np.round(x[0]), np.ceil(x[1]), np.floor(x[2])],
    ],
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


def test_voxelize_binary_mask_input(hexbeam):
    # Test unstructured grid works
    assert isinstance(hexbeam, pv.UnstructuredGrid)
    mask = hexbeam.voxelize_binary_mask()
    assert mask.n_points

    # Test point cloud does not
    mesh = pv.PolyData(hexbeam.points)
    with pytest.raises(ValueError, match='Input mesh must have faces for voxelization'):
        mesh.voxelize_binary_mask()


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

    match = 'Spacing and cell length options cannot both be set. Set one or the other.'
    with pytest.raises(TypeError, match=match):
        sphere.voxelize_binary_mask(spacing=(4, 5, 6), cell_length_percentile=0.2)
    with pytest.raises(TypeError, match=match):
        sphere.voxelize_binary_mask(spacing=0.1, cell_length_sample_size=sphere.n_cells)

    match = 'Rounding func cannot be set when dimensions is specified. Set one or the other.'
    with pytest.raises(TypeError, match=match):
        sphere.voxelize_binary_mask(dimensions=(1, 2, 3), rounding_func=np.round)

    for parameter in [
        'dimensions',
        'spacing',
        'rounding_func',
        'cell_length_percentile',
        'cell_length_sample_size',
    ]:
        kwargs = {parameter: 0}  # Give parameter any value for test
        match = (
            'Cannot specify a reference volume with other geometry parameters. '
            '`reference_volume` must define the geometry exclusively.'
        )
        with pytest.raises(TypeError, match=match):
            sphere.voxelize_binary_mask(reference_volume=pv.ImageData(), **kwargs)


def test_voxelize_rectilinear(ant):
    vox = ant.voxelize_rectilinear()
    assert isinstance(vox, pv.RectilinearGrid)

    # Test dimensions
    dims = np.array((10, 20, 30))
    vox = ant.voxelize_rectilinear(dimensions=dims)
    assert np.array_equal(vox.dimensions, dims)

    # Test spacing by voxelizing as a single cell
    bnds = ant.bounds
    size = bnds.x_max - bnds.x_min, bnds.y_max - bnds.y_min, bnds.z_max - bnds.z_min
    single_cell_dims = np.array((2, 2, 2))
    vox = ant.voxelize_rectilinear(spacing=size)
    assert np.allclose(vox.dimensions, single_cell_dims)
    assert np.allclose(vox.bounds, ant.bounds)

    # Test reference volume - specify both dimensions and spacing
    reference_volume = pv.ImageData(dimensions=single_cell_dims - 1, spacing=size)
    vox = ant.voxelize_rectilinear(reference_volume=reference_volume)
    assert np.allclose(vox.dimensions, single_cell_dims)
    assert np.allclose(vox.bounds, ant.bounds)

    # Test scalar values
    foreground = 2
    background = 3
    vox = ant.voxelize_rectilinear(foreground_value=foreground, background_value=background)
    assert 'mask' in vox.cell_data
    assert vox.array_names == ['mask']
    values = np.unique(vox['mask'])
    assert np.array_equal(values, [foreground, background])

    # Test other keywords
    vox = ant.voxelize(cell_length_percentile=0.5)
    assert vox.n_cells
    vox = ant.voxelize(cell_length_sample_size=ant.n_cells)
    assert vox.n_cells
    assert vox.n_cells
    vox = ant.voxelize_rectilinear(progress_bar=True)
    assert vox.n_cells
    vox = ant.voxelize_rectilinear(spacing=(0.1, 0.2, 0.3), rounding_func=np.ceil)
    assert vox.n_cells

    # Test invalid input
    with pytest.raises(TypeError, match='Object arrays are not supported'):
        ant.voxelize_rectilinear(spacing={0.5, 0.3})


def test_voxelize(ant):
    vox = ant.voxelize()
    assert isinstance(vox, pv.UnstructuredGrid)
    assert vox.array_names == []

    # Test dimensions
    dims = np.array((10, 20, 30))
    vox = pv.Cube().voxelize(dimensions=dims)
    assert np.array_equal(vox.n_points, np.prod(dims))

    # Test spacing by voxelizing as a single cell
    bnds = ant.bounds
    size = bnds.x_max - bnds.x_min, bnds.y_max - bnds.y_min, bnds.z_max - bnds.z_min
    vox = ant.voxelize(spacing=size)
    assert np.allclose(vox.n_cells, 1)
    assert np.allclose(vox.bounds, ant.bounds)

    # Test other keywords
    vox = ant.voxelize(cell_length_percentile=0.5)
    assert vox.n_cells
    vox = ant.voxelize(cell_length_sample_size=ant.n_cells)
    assert vox.n_cells
    vox = ant.voxelize(progress_bar=True)
    assert vox.n_cells
    vox = ant.voxelize(spacing=(0.1, 0.2, 0.3), rounding_func=np.ceil)
    assert vox.n_cells

    # Test invalid input
    with pytest.raises(TypeError, match='Object arrays are not supported'):
        ant.voxelize(spacing={0.5, 0.3})
