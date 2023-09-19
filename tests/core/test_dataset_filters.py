import itertools
import os
import platform
from unittest.mock import Mock, patch

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.core import _vtk_core
from pyvista.core.celltype import CellType
from pyvista.core.errors import (
    MissingDataError,
    NotAllTrianglesError,
    PyVistaDeprecationWarning,
    VTKVersionError,
)

normals = ['x', 'y', '-z', (1, 1, 1), (3.3, 5.4, 0.8)]

skip_mac = pytest.mark.skipif(platform.system() == 'Darwin', reason="Flaky Mac tests")


def aprox_le(a, b, rtol=1e-5, atol=1e-8):
    """Return if that ``a <= b`` within a tolerance.

    See numpy.isclose for the description of ``rtol`` and ``atol``.

    """
    if a < b:
        return True
    else:
        return np.isclose(a, b, rtol, atol)


class GetOutput:
    """Helper class to patch ``pv.core.filters._get_output`` which captures the raw VTK algorithm objects at the time
    ``_get_output`` is invoked.
    """

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


@pytest.fixture
def composite(datasets):
    return pv.MultiBlock(datasets)


@pytest.fixture(scope='module')
def uniform_vec():
    nx, ny, nz = 20, 15, 5
    origin = (-(nx - 1) * 0.1 / 2, -(ny - 1) * 0.1 / 2, -(nz - 1) * 0.1 / 2)
    mesh = pv.ImageData(dimensions=(nx, ny, nz), spacing=(0.1, 0.1, 0.1), origin=origin)
    mesh['vectors'] = mesh.points
    return mesh


def test_datasetfilters_init():
    with pytest.raises(TypeError):
        pv.core.filters.DataSetFilters()


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
    clp = pv.Wavelet().clip(normal=(1, 1, 1), crinkle=True)
    assert clp is not None
    clp1, clp2 = pv.Wavelet().clip(normal=(1, 1, 1), return_clipped=True, crinkle=True)
    assert clp1 is not None
    assert clp2 is not None


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
                scalars='to_clip', value=clip_value, both=True, invert=invert
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
    with pytest.raises(ValueError):
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

    with pytest.raises(ValueError):
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
    with pytest.raises(ValueError):
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
    assert "implicit_distance" in res.point_data
    dataset.compute_implicit_distance(surface, inplace=True)
    assert "implicit_distance" in dataset.point_data


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
    with pytest.raises(ValueError):
        dataset.slice_along_axis(axis='u')


def test_slice_along_axis_composite(composite):
    # Now test composite data structures
    output = composite.slice_along_axis(progress_bar=True)
    assert output.n_blocks == composite.n_blocks


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
    with pytest.raises(ValueError):
        dataset = datasets[3]  # polydata with no arrays
        thresh = dataset.threshold()

    dataset = examples.load_uniform()
    with pytest.raises(ValueError):
        dataset.threshold([10, 100, 300], progress_bar=True)

    with pytest.raises(ValueError):
        dataset.threshold(100, method='between')

    with pytest.raises(ValueError):
        dataset.threshold((2, 1))


def test_threshold_all_scalars():
    mesh = pv.Sphere()
    mesh.clear_data()

    mesh["scalar0"] = np.zeros(mesh.n_points)
    mesh["scalar1"] = np.ones(mesh.n_points)
    mesh.set_active_scalars("scalar1")
    thresh_all = mesh.threshold(value=0.5, all_scalars=True)  # only uses scalar1
    assert thresh_all.n_points == mesh.n_points
    assert thresh_all.n_cells == mesh.n_cells

    mesh["scalar1"][0 : int(mesh.n_points / 2)] = 0.0
    thresh = mesh.threshold(value=0.5, all_scalars=False)
    thresh_all = mesh.threshold(value=0.5, all_scalars=True)
    assert thresh_all.n_points < mesh.n_points
    # removes additional cells/points due to all_scalars
    assert thresh_all.n_points < thresh.n_points
    assert thresh_all.n_cells < mesh.n_cells
    assert thresh_all.n_cells < thresh.n_cells

    mesh.clear_data()
    mesh["scalar0"] = np.zeros(mesh.n_cells)
    mesh["scalar1"] = np.ones(mesh.n_cells)
    mesh["scalar1"][0 : int(mesh.n_cells / 2)] = 0.0
    mesh.set_active_scalars("scalar1")
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
    mesh["data"] = data

    thresh = mesh.threshold(value=0.5, scalars="data", component_mode="component", component=0)
    assert thresh.n_cells == 3
    thresh = mesh.threshold(value=0.5, scalars="data", component_mode="component", component=1)
    assert thresh.n_cells == 2
    thresh = mesh.threshold(value=0.5, scalars="data", component_mode="all")
    assert thresh.n_cells == 1
    thresh = mesh.threshold(value=0.5, scalars="data", component_mode="any")
    assert thresh.n_cells == 4

    with pytest.raises(ValueError):
        mesh.threshold(value=0.5, scalars="data", component_mode="not a mode")

    with pytest.raises(ValueError):
        mesh.threshold(value=0.5, scalars="data", component_mode="component", component=-1)

    with pytest.raises(ValueError):
        mesh.threshold(value=0.5, scalars="data", component_mode="component", component=3)

    with pytest.raises(TypeError):
        mesh.threshold(value=0.5, scalars="data", component_mode="component", component=0.5)


def test_threshold_percent(datasets):
    percents = [25, 50, [18.0, 85.0], [19.0, 80.0], 0.70]
    inverts = [False, True, False, True, False]
    # Only test data sets that have arrays
    for i, dataset in enumerate(datasets[0:3]):
        thresh = dataset.threshold_percent(
            percent=percents[i], invert=inverts[i], progress_bar=True
        )
        assert thresh is not None
        assert isinstance(thresh, pv.UnstructuredGrid)
    dataset = examples.load_uniform()
    _ = dataset.threshold_percent(0.75, scalars='Spatial Cell Data', progress_bar=True)
    with pytest.raises(ValueError):
        dataset.threshold_percent(20000)
    with pytest.raises(ValueError):
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


def test_extract_geometry(datasets, composite):
    for dataset in datasets:
        geom = dataset.extract_geometry(progress_bar=True)
        assert geom is not None
        assert isinstance(geom, pv.PolyData)
    # Now test composite data structures
    output = composite.extract_geometry()
    assert isinstance(output, pv.PolyData)


def test_extract_geometry_extent(uniform):
    geom = uniform.extract_geometry(extent=(0, 5, 0, 100, 0, 100))
    assert isinstance(geom, pv.PolyData)
    assert geom.bounds == (0.0, 5.0, 0.0, 9.0, 0.0, 9.0)


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


def test_delaunay_2d_unstructured(datasets):
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


def test_contour_errors(uniform):
    with pytest.raises(TypeError):
        uniform.contour(scalars='Spatial Cell Data')
    with pytest.raises(TypeError):
        uniform.contour(isosurfaces=pv.PolyData())
    with pytest.raises(TypeError):
        uniform.contour(isosurfaces={100, 300, 500})
    uniform = examples.load_airplane()
    with pytest.raises(ValueError):
        uniform.contour()
    with pytest.raises(ValueError):
        uniform.contour(method='invalid method')
    with pytest.raises(TypeError, match='Invalid type for `scalars`'):
        uniform.contour(scalars=1)
    with pytest.raises(TypeError):
        uniform.contour(rng={})
    with pytest.raises(ValueError, match='rng must be a two-length'):
        uniform.contour(rng=[1])
    with pytest.raises(ValueError, match='rng must be a sorted'):
        uniform.contour(rng=[2, 1])


def test_elevation():
    dataset = examples.load_uniform()
    # Test default params
    elev = dataset.elevation(progress_bar=True)
    assert 'Elevation' in elev.array_names
    assert 'Elevation' == elev.active_scalars_name
    assert elev.get_data_range() == (dataset.bounds[4], dataset.bounds[5])
    # test vector args
    c = list(dataset.center)
    t = list(c)  # cast so it does not point to `c`
    t[2] = dataset.bounds[-1]
    elev = dataset.elevation(low_point=c, high_point=t, progress_bar=True)
    assert 'Elevation' in elev.array_names
    assert 'Elevation' == elev.active_scalars_name
    assert elev.get_data_range() == (dataset.center[2], dataset.bounds[5])
    # Test not setting active
    elev = dataset.elevation(set_active=False, progress_bar=True)
    assert 'Elevation' in elev.array_names
    assert 'Elevation' != elev.active_scalars_name
    # Set use a range by scalar name
    elev = dataset.elevation(scalar_range='Spatial Point Data', progress_bar=True)
    assert 'Elevation' in elev.array_names
    assert 'Elevation' == elev.active_scalars_name
    assert dataset.get_data_range('Spatial Point Data') == (elev.get_data_range('Elevation'))
    # Set use a user defined range
    elev = dataset.elevation(scalar_range=[1.0, 100.0], progress_bar=True)
    assert 'Elevation' in elev.array_names
    assert 'Elevation' == elev.active_scalars_name
    assert elev.get_data_range('Elevation') == (1.0, 100.0)
    # test errors
    with pytest.raises(TypeError):
        elev = dataset.elevation(scalar_range=0.5, progress_bar=True)
    with pytest.raises(ValueError):
        elev = dataset.elevation(scalar_range=[1, 2, 3], progress_bar=True)
    with pytest.raises(TypeError):
        elev = dataset.elevation(scalar_range={1, 2}, progress_bar=True)


def test_elevation_composite(composite):
    # Now test composite data structures
    output = composite.elevation(progress_bar=True)
    assert output.n_blocks == composite.n_blocks


def test_texture_map_to_plane():
    dataset = examples.load_airplane()
    # Automatically decide plane
    out = dataset.texture_map_to_plane(inplace=False, progress_bar=True)
    assert isinstance(out, type(dataset))
    # Define the plane explicitly
    bnds = dataset.bounds
    origin = bnds[0::2]
    point_u = (bnds[1], bnds[2], bnds[4])
    point_v = (bnds[0], bnds[3], bnds[4])
    out = dataset.texture_map_to_plane(
        origin=origin, point_u=point_u, point_v=point_v, progress_bar=True
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
        center=(0.1, 0.0, 0.0), prevent_seam=True, progress_bar=True
    )
    assert isinstance(out, type(dataset))
    assert 'Texture Coordinates' in out.array_names
    # FINAL: Test in place modifiacation
    dataset.texture_map_to_sphere(inplace=True, progress_bar=True)
    assert 'Texture Coordinates' in dataset.array_names


def test_compute_cell_sizes(datasets):
    for dataset in datasets:
        result = dataset.compute_cell_sizes(progress_bar=True)
        assert result is not None
        assert isinstance(result, type(dataset))
        assert 'Area' in result.array_names
        assert 'Volume' in result.array_names
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


def test_cell_centers_composite(composite):
    # Now test composite data structures
    output = composite.cell_centers(progress_bar=True)
    assert output.n_blocks == composite.n_blocks


def test_glyph(datasets, sphere):
    for dataset in datasets:
        dataset["vectors"] = np.ones_like(dataset.points)
        result = dataset.glyph(progress_bar=True)
        assert result is not None
        assert isinstance(result, pv.PolyData)
    # Test different options for glyph filter
    sphere_sans_arrays = sphere.copy()
    sphere.compute_normals(inplace=True)
    sphere["vectors"] = np.ones([sphere.n_points, 3])
    sphere.set_active_vectors("vectors")
    sphere.point_data['arr'] = np.ones(sphere.n_points)

    assert sphere.glyph(scale=False, progress_bar=True)
    assert sphere.glyph(scale='arr', progress_bar=True)
    assert sphere.glyph(scale='arr', orient='Normals', factor=0.1, progress_bar=True)
    assert sphere.glyph(scale='arr', orient='Normals', factor=0.1, tolerance=0.1, progress_bar=True)
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
        geom=geoms, scale='arr', orient='Normals', factor=0.1, tolerance=0.1, progress_bar=True
    )
    assert sphere.glyph(geom=geoms[:1], indices=[None], progress_bar=True)

    with pytest.warns(Warning):  # tries to orient but no orientation vector available
        assert sphere_sans_arrays.glyph(geom=geoms, progress_bar=True)

    sphere_sans_arrays["vec1"] = np.ones((sphere_sans_arrays.n_points, 3))
    sphere_sans_arrays["vec2"] = np.ones((sphere_sans_arrays.n_points, 3))
    with pytest.warns(Warning):  # tries to orient but multiple orientation vectors are possible
        assert sphere_sans_arrays.glyph(geom=geoms, progress_bar=True)

    with pytest.raises(TypeError):
        # wrong type for the glyph
        sphere.glyph(geom=pv.StructuredGrid())
    with pytest.raises(TypeError):
        # wrong type for the indices
        sphere.glyph(geom=geoms, indices=set(indices))
    with pytest.raises(ValueError):
        # wrong length for the indices
        sphere.glyph(geom=geoms, indices=indices[:-1])


def test_glyph_cell_point_data(sphere):
    sphere['vectors_cell'] = np.ones([sphere.n_cells, 3])
    sphere['vectors_points'] = np.ones([sphere.n_points, 3])
    sphere['arr_cell'] = np.ones(sphere.n_cells)
    sphere['arr_points'] = np.ones(sphere.n_points)

    assert sphere.glyph(orient='vectors_cell', scale='arr_cell', progress_bar=True)
    assert sphere.glyph(orient='vectors_points', scale='arr_points', progress_bar=True)
    with pytest.raises(ValueError):
        sphere.glyph(orient='vectors_cell', scale='arr_points', progress_bar=True)
    with pytest.raises(ValueError):
        sphere.glyph(orient='vectors_points', scale='arr_cell', progress_bar=True)


class InterrogateVTKGlyph3D:
    def __init__(self, alg: _vtk_core.vtkGlyph3D):
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
        ## Test the length of the field and not the FieldAssociation, because the vtkGlyph3D filter takes POINT data
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
    grid["z_axis"] = orient * scale
    glyph1 = grid.glyph(geom=geom, orient="z_axis", scale="z_axis")
    glyph2 = grid.glyph(geom=geom, orient=False, scale="z_axis")
    glyph3 = grid.glyph(geom=geom, orient="z_axis", scale=False)
    glyph4 = grid.glyph(geom=geom, orient=False, scale=False)
    assert glyph1.bounds[4] == geom.bounds[0] * scale and glyph1.bounds[5] == geom.bounds[1] * scale
    assert glyph2.bounds[0] == geom.bounds[0] * scale and glyph2.bounds[1] == geom.bounds[1] * scale
    assert glyph3.bounds[4] == geom.bounds[0] and glyph3.bounds[5] == geom.bounds[1]
    assert glyph4.bounds[0] == geom.bounds[0] and glyph4.bounds[1] == geom.bounds[1]


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
def foot_bones():
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
    'extraction_mode', ['all', 'largest', 'specified', 'cell_seed', 'point_seed', 'closest']
)
@pytest.mark.parametrize('label_regions', [True, False])
@pytest.mark.parametrize('scalar_range', [True, False])
@pytest.mark.needs_vtk_version(9, 1, 0)
def test_connectivity_inplace_and_output_type(
    datasets, dataset_index, extraction_mode, label_regions, scalar_range
):
    # paramaterize with label_regions and scalar_range as these parameters
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
    'extraction_mode', ['all', 'largest', 'specified', 'cell_seed', 'point_seed', 'closest']
)
@pytest.mark.needs_vtk_version(9, 1, 0)
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


@pytest.mark.needs_vtk_version(9, 1, 0)
def test_connectivity_raises(
    connected_datasets_single_disconnected_cell,
):
    dataset = connected_datasets_single_disconnected_cell[0]['point']

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


@pytest.mark.parametrize('dataset_index', list(range(5)))
@pytest.mark.parametrize(
    'extraction_mode', ['all', 'largest', 'specified', 'cell_seed', 'point_seed', 'closest']
)
@pytest.mark.parametrize('association', ['cell', 'point'])
@pytest.mark.needs_vtk_version(9, 1, 0)
def test_connectivity_scalar_range(
    connected_datasets_single_disconnected_cell, dataset_index, extraction_mode, association
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
        **common_args, scalar_range=dataset.get_data_range()
    )
    assert conn_with_full_range.n_cells == dataset.n_cells

    # test input scalars are passed to output
    assert len(conn_no_range.array_names) == 3  # ['data', 'RegionId', 'RegionId']
    assert len(conn_with_range.array_names) == 3
    assert len(conn_with_full_range.array_names) == 3


@pytest.mark.needs_vtk_version(9, 1, 0)
def test_connectivity_all(foot_bones):
    conn = foot_bones.connectivity('all')
    assert conn.n_cells == foot_bones.n_cells

    # test correct labels
    conn = foot_bones.connectivity('all', label_regions=True)
    region_ids, counts = np.unique(conn.cell_data['RegionId'], return_counts=True)
    assert np.array_equal(region_ids, list(range(26)))
    assert np.array_equal(
        counts,
        [
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
        ],
    )


@pytest.mark.needs_vtk_version(9, 1, 0)
def test_connectivity_largest(foot_bones):
    conn = foot_bones.connectivity('largest')
    assert conn.n_cells == 598

    # test correct labels
    conn = foot_bones.connectivity('largest', label_regions=True)
    region_ids, counts = np.unique(conn.cell_data['RegionId'], return_counts=True)
    assert region_ids == [0]
    assert counts == [598]


@pytest.mark.needs_vtk_version(9, 1, 0)
def test_connectivity_specified(foot_bones):
    # test all regions
    all_regions = list(range(26))
    conn = foot_bones.connectivity('specified', region_ids=all_regions)
    assert conn.n_cells == foot_bones.n_cells

    # test irrelevant region IDs
    test_regions = all_regions + [77, 99]
    conn = foot_bones.connectivity('specified', test_regions)
    assert conn.n_cells == foot_bones.n_cells

    # test some regions
    some_regions = [1, 2, 4, 5]
    expected_n_cells = 586 + 392 + 228 + 212
    conn = foot_bones.connectivity('specified', some_regions)
    assert conn.n_cells == expected_n_cells

    # test correct labels
    conn = foot_bones.connectivity('specified', some_regions, label_regions=True)
    region_ids, counts = np.unique(conn.cell_data['RegionId'], return_counts=True)
    assert np.array_equal(region_ids, [0, 1, 2, 3])
    assert np.array_equal(counts, [586, 392, 228, 212])


@pytest.mark.parametrize('dataset_index', list(range(5)))
@pytest.mark.needs_vtk_version(9, 1, 0)
def test_connectivity_specified_returns_empty(connected_datasets, dataset_index):
    dataset = connected_datasets[dataset_index]
    unused_region_id = 1
    conn = dataset.connectivity('specified', unused_region_id)
    assert conn.n_cells == 0
    assert conn.n_points == 0


@pytest.mark.needs_vtk_version(9, 1, 0)
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


@pytest.mark.needs_vtk_version(9, 1, 0)
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


@pytest.mark.needs_vtk_version(9, 1, 0)
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
    sphere['cellscalars'] = np.random.random(sphere.n_cells)
    sphere.point_data.clear()
    with pytest.raises(TypeError):
        sphere.warp_by_scalar()


def test_invalid_warp_scalar_inplace(uniform):
    with pytest.raises(TypeError):
        uniform.warp_by_scalar(inplace=True, progress_bar=True)


def test_invalid_warp_vector(sphere):
    # bad vectors
    sphere.point_data['Normals'] = np.empty((sphere.n_points, 2))
    with pytest.raises(ValueError):
        sphere.warp_by_vector('Normals')

    # no vectors
    sphere.point_data.clear()
    with pytest.raises(ValueError):
        sphere.warp_by_vector()


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


def test_delaunay_3d():
    data = examples.load_uniform().threshold_percent(30, progress_bar=True)
    result = data.delaunay_3d()
    assert np.any(result.points)


def test_smooth(uniform):
    surf = uniform.extract_surface().clean()
    smoothed = surf.smooth()

    # expect mesh is smoothed, raising mean curvature since it is more "spherelike"
    assert smoothed.triangulate().curvature().mean() > surf.triangulate().curvature().mean()

    smooth_inplace = surf.smooth(inplace=True)
    assert np.allclose(surf.points, smoothed.points)
    assert np.allclose(smooth_inplace.points, smoothed.points)


def test_smooth_taubin(uniform):
    surf = uniform.extract_surface().clean()
    smoothed = surf.smooth_taubin()

    # expect mesh is smoothed, raising mean curvature since it is more "spherelike"
    assert smoothed.triangulate().curvature().mean() > surf.triangulate().curvature().mean()

    # while volume is maintained
    assert np.isclose(smoothed.volume, surf.volume, rtol=0.01)

    smooth_inplace = surf.smooth_taubin(inplace=True)
    assert np.allclose(surf.points, smoothed.points)
    assert np.allclose(smooth_inplace.points, smoothed.points)


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
    with pytest.raises(ValueError):
        sample_test(locator="invalid")
    sample_test(pass_cell_data=False)
    sample_test(pass_point_data=False)
    sample_test(pass_field_data=False)


def test_sample_composite():
    mesh0 = pv.ImageData(dimensions=(11, 11, 1), origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    mesh1 = pv.ImageData(dimensions=(11, 11, 1), origin=(10.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    mesh0["common_data"] = np.zeros(mesh0.n_points)
    mesh1["common_data"] = np.ones(mesh1.n_points)
    mesh0["partial_data"] = np.zeros(mesh0.n_points)

    composite = pv.MultiBlock([mesh0, mesh1])

    probe_points = pv.PolyData(
        [
            [5.0, 5.0, 0.0],
            [15.0, 5.0, 0.0],
            [25.0, 5.0, 0.0],  # outside domain
        ]
    )

    result = probe_points.sample(composite)
    assert "common_data" in result.point_data
    # Need pass partial arrays?
    assert "partial_data" not in result.point_data
    assert "vtkValidPointMask" in result.point_data
    assert "vtkGhostType" in result.point_data
    # data outside domain is 0
    assert np.array_equal(result["common_data"], [0.0, 1.0, 0.0])
    assert np.array_equal(result["vtkValidPointMask"], [1, 1, 0])

    result = probe_points.sample(composite, mark_blank=False)
    assert "vtkGhostType" not in result.point_data

    small_mesh_0 = pv.ImageData(
        dimensions=(6, 6, 1), origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)
    )
    small_mesh_1 = pv.ImageData(
        dimensions=(6, 6, 1), origin=(10.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)
    )

    probe_composite = pv.MultiBlock([small_mesh_0, small_mesh_1])
    result = probe_composite.sample(composite)
    assert "common_data" in result[0].point_data
    # Need pass partial arrays?
    assert "partial_data" not in result[0].point_data
    assert "vtkValidPointMask" in result[0].point_data
    assert "vtkGhostType" in result[0].point_data


@pytest.mark.parametrize('use_points', [True, False])
@pytest.mark.parametrize('categorical', [True, False])
@pytest.mark.parametrize('locator', [None, _vtk_core.vtkStaticCellLocator()])
def test_probe(categorical, use_points, locator):
    mesh = pv.Sphere(center=(4.5, 4.5, 4.5), radius=4.5)
    data_to_probe = examples.load_uniform()
    if use_points:
        dataset = np.array(mesh.points)
    else:
        dataset = mesh
    with pytest.warns(PyVistaDeprecationWarning):
        result = data_to_probe.probe(
            dataset, tolerance=1e-5, categorical=categorical, progress_bar=True, locator=locator
        )
    name = 'Spatial Point Data'
    assert name in result.array_names
    assert isinstance(result, type(mesh))
    result = mesh.sample(data_to_probe, tolerance=1.0, progress_bar=True)
    name = 'Spatial Point Data'
    assert name in result.array_names
    assert isinstance(result, type(mesh))


@pytest.mark.parametrize('integration_direction', ['forward', 'backward', 'both'])
def test_streamlines_dir(uniform_vec, integration_direction):
    stream = uniform_vec.streamlines(
        'vectors', integration_direction=integration_direction, progress_bar=True
    )
    assert all([stream.n_points, stream.n_cells])


@pytest.mark.parametrize('integrator_type', [2, 4, 45])
def test_streamlines_type(uniform_vec, integrator_type):
    stream = uniform_vec.streamlines('vectors', integrator_type=integrator_type, progress_bar=True)
    assert all([stream.n_points, stream.n_cells])


@pytest.mark.parametrize('interpolator_type', ['point', 'cell'])
def test_streamlines_cell_point(uniform_vec, interpolator_type):
    stream = uniform_vec.streamlines(
        'vectors', interpolator_type=interpolator_type, progress_bar=True
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


def test_streamlines_errors(uniform_vec):
    with pytest.raises(ValueError):
        uniform_vec.streamlines('vectors', integration_direction='not valid')

    with pytest.raises(ValueError):
        uniform_vec.streamlines('vectors', integrator_type=42)

    with pytest.raises(ValueError):
        uniform_vec.streamlines('vectors', interpolator_type='not valid')

    with pytest.raises(ValueError):
        uniform_vec.streamlines('vectors', step_unit='not valid')

    with pytest.raises(ValueError):
        uniform_vec.streamlines('vectors', pointa=(0, 0, 0))
    with pytest.raises(ValueError):
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
    x, y, z = np.meshgrid(np.arange(-10, 10, 0.5), np.arange(-10, 10, 0.5), np.arange(-10, 10, 0.5))
    mesh = pv.StructuredGrid(x, y, z)
    x2, y2, z2 = np.meshgrid(np.arange(-1, 1, 0.5), np.arange(-1, 1, 0.5), np.arange(-1, 1, 0.5))
    mesh2 = pv.StructuredGrid(x2, y2, z2)
    mesh["vectors"] = np.ones([mesh.n_points, 3])
    mesh.set_active_vectors("vectors")

    with pv.VtkErrorCatcher(raise_errors=True):
        stream = mesh.streamlines_from_source(mesh2)
    assert all([stream.n_points, stream.n_cells])


def mesh_2D_velocity():
    mesh = pv.Plane(i_resolution=100, j_resolution=100)
    velocity = np.zeros([mesh.n_points, 3])
    velocity[:, 0] = 1
    mesh["velocity"] = velocity
    mesh.set_active_vectors("velocity")
    return mesh


def test_streamlines_evenly_spaced_2D():
    mesh = mesh_2D_velocity()
    streams = mesh.streamlines_evenly_spaced_2D(progress_bar=True)
    assert all([streams.n_points, streams.n_cells])


def test_streamlines_evenly_spaced_2D_sep_dist_ratio():
    mesh = mesh_2D_velocity()
    streams = mesh.streamlines_evenly_spaced_2D(separating_distance_ratio=0.1, progress_bar=True)
    assert all([streams.n_points, streams.n_cells])


def test_streamlines_evenly_spaced_2D_start_position():
    mesh = mesh_2D_velocity()
    streams = mesh.streamlines_evenly_spaced_2D(start_position=(-0.1, 0.1, 0.0), progress_bar=True)
    assert all([streams.n_points, streams.n_cells])


def test_streamlines_evenly_spaced_2D_vectors():
    mesh = mesh_2D_velocity()
    mesh.set_active_vectors(None)
    streams = mesh.streamlines_evenly_spaced_2D(vectors="velocity", progress_bar=True)
    assert all([streams.n_points, streams.n_cells])


def test_streamlines_evenly_spaced_2D_integrator_type():
    mesh = mesh_2D_velocity()
    streams = mesh.streamlines_evenly_spaced_2D(integrator_type=4, progress_bar=True)
    assert all([streams.n_points, streams.n_cells])


def test_streamlines_evenly_spaced_2D_interpolator_type():
    mesh = mesh_2D_velocity()
    streams = mesh.streamlines_evenly_spaced_2D(interpolator_type='cell', progress_bar=True)
    assert all([streams.n_points, streams.n_cells])


def test_streamlines_evenly_spaced_2D_errors():
    mesh = mesh_2D_velocity()

    with pytest.raises(ValueError):
        mesh.streamlines_evenly_spaced_2D(integrator_type=45)

    with pytest.raises(ValueError):
        mesh.streamlines_evenly_spaced_2D(interpolator_type="not valid")

    with pytest.raises(ValueError):
        mesh.streamlines_evenly_spaced_2D(step_unit="not valid")


@pytest.mark.xfail
def test_streamlines_nonxy_plane():
    # streamlines_evenly_spaced_2D only works for xy plane datasets
    # test here so that fixes in vtk can be caught
    mesh = mesh_2D_velocity()
    mesh.translate((0, 0, 1), inplace=True)  # move to z=1, xy plane
    streams = mesh.streamlines_evenly_spaced_2D(progress_bar=True)
    assert all([streams.n_points, streams.n_cells])


def test_sample_over_line():
    """Test that we get a sampled line."""
    name = 'values'

    line = pv.Line([0, 0, 0], [0, 0, 10], 9)
    line[name] = np.linspace(0, 10, 10)

    sampled_line = line.sample_over_line([0, 0, 0.5], [0, 0, 1.5], 2, progress_bar=True)

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
    tmp_dir = tmpdir.mkdir("tmpdir")
    filename = str(tmp_dir.join('tmp.png'))
    mesh = examples.load_uniform()
    # Make two points to construct the line between
    a = [mesh.bounds[0], mesh.bounds[2], mesh.bounds[4]]
    b = [mesh.bounds[1], mesh.bounds[3], mesh.bounds[5]]
    mesh.plot_over_line(a, b, resolution=1000, show=False, progress_bar=True)
    # Test multicomponent
    mesh['foo'] = np.random.rand(mesh.n_cells, 3)
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
    assert os.path.isfile(filename)
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

    line = pv.Line([0, 0, 0], [0, 0, 10], 9)
    line[name] = np.linspace(0, 10, 10)

    sampled_multiple_lines = line.sample_over_multiple_lines(
        [[0, 0, 0.5], [0, 0, 1], [0, 0, 1.5]], progress_bar=True
    )

    expected_result = np.array([0.5, 1, 1.5])
    assert np.allclose(sampled_multiple_lines[name], expected_result)
    assert name in sampled_multiple_lines.array_names  # is name in sampled result


def test_sample_over_circular_arc():
    """Test that we get a circular arc."""

    name = 'values'

    uniform = examples.load_uniform()
    uniform[name] = uniform.points[:, 2]

    xmin = uniform.bounds[0]
    xmax = uniform.bounds[1]
    ymin = uniform.bounds[2]
    zmin = uniform.bounds[4]
    zmax = uniform.bounds[5]
    pointa = [xmin, ymin, zmax]
    pointb = [xmax, ymin, zmin]
    center = [xmin, ymin, zmin]
    sampled_arc = uniform.sample_over_circular_arc(pointa, pointb, center, 2, progress_bar=True)

    expected_result = zmin + (zmax - zmin) * np.sin([np.pi / 2.0, np.pi / 4.0, 0.0])
    assert np.allclose(sampled_arc[name], expected_result)
    assert name in sampled_arc.array_names  # is name in sampled result

    # test no resolution
    sphere = pv.Sphere(center=(4.5, 4.5, 4.5), radius=4.5)
    sampled_from_sphere = sphere.sample_over_circular_arc(
        [3, 1, 1], [-3, -1, -1], [0, 0, 0], progress_bar=True
    )
    assert sampled_from_sphere.n_points == sphere.n_cells + 1

    # is sampled result a polydata object
    assert isinstance(sampled_from_sphere, pv.PolyData)


def test_sample_over_circular_arc_normal():
    """Test that we get a circular arc_normal."""

    name = 'values'

    uniform = examples.load_uniform()
    uniform[name] = uniform.points[:, 2]

    xmin = uniform.bounds[0]
    ymin = uniform.bounds[2]
    ymax = uniform.bounds[3]
    zmin = uniform.bounds[4]
    zmax = uniform.bounds[5]
    normal = [xmin, ymax, zmin]
    polar = [xmin, ymin, zmax]
    angle = 90.0 * np.random.rand()
    resolution = np.random.randint(10000)
    center = [xmin, ymin, zmin]
    sampled_arc_normal = uniform.sample_over_circular_arc_normal(
        center, resolution=resolution, normal=normal, polar=polar, angle=angle, progress_bar=True
    )
    angles = np.linspace(np.pi / 2.0, np.pi / 2.0 - np.deg2rad(angle), resolution + 1)

    expected_result = zmin + (zmax - zmin) * np.sin(angles)
    assert np.allclose(sampled_arc_normal[name], expected_result)
    assert name in sampled_arc_normal.array_names  # is name in sampled result

    # test no resolution
    sphere = pv.Sphere(center=(4.5, 4.5, 4.5), radius=4.5)
    sampled_from_sphere = sphere.sample_over_circular_arc_normal(
        [0, 0, 0], polar=[3, 1, 1], angle=180, progress_bar=True
    )
    assert sampled_from_sphere.n_points == sphere.n_cells + 1

    # is sampled result a polydata object
    assert isinstance(sampled_from_sphere, pv.PolyData)


def test_plot_over_circular_arc(tmpdir):
    mesh = examples.load_uniform()
    tmp_dir = tmpdir.mkdir("tmpdir")
    filename = str(tmp_dir.join('tmp.png'))

    # Make two points and center to construct the circular arc between
    a = [mesh.bounds[0], mesh.bounds[2], mesh.bounds[5]]
    b = [mesh.bounds[1], mesh.bounds[2], mesh.bounds[4]]
    center = [mesh.bounds[0], mesh.bounds[2], mesh.bounds[4]]
    mesh.plot_over_circular_arc(
        a, b, center, resolution=1000, show=False, fname=filename, progress_bar=True
    )
    assert os.path.isfile(filename)

    # Test multicomponent
    mesh['foo'] = np.random.rand(mesh.n_cells, 3)
    mesh.plot_over_circular_arc(
        a,
        b,
        center,
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
            a,
            b,
            center,
            resolution=None,
            scalars='invalid_array_name',
            title='My Stuff',
            ylabel='3 Values',
            show=False,
        )


def test_plot_over_circular_arc_normal(tmpdir):
    mesh = examples.load_uniform()
    tmp_dir = tmpdir.mkdir("tmpdir")
    filename = str(tmp_dir.join('tmp.png'))

    # Make center and normal/polar vector to construct the circular arc between
    # normal = [mesh.bounds[0], mesh.bounds[2], mesh.bounds[5]]
    polar = [mesh.bounds[0], mesh.bounds[3], mesh.bounds[4]]
    angle = 90
    center = [mesh.bounds[0], mesh.bounds[2], mesh.bounds[4]]
    mesh.plot_over_circular_arc_normal(
        center, polar=polar, angle=angle, show=False, fname=filename, progress_bar=True
    )
    assert os.path.isfile(filename)

    # Test multicomponent
    mesh['foo'] = np.random.rand(mesh.n_cells, 3)
    mesh.plot_over_circular_arc_normal(
        center,
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
            center,
            polar=polar,
            angle=angle,
            resolution=None,
            scalars='invalid_array_name',
            title='My Stuff',
            ylabel='3 Values',
            show=False,
        )


def test_slice_along_line():
    model = examples.load_uniform()
    n = 5
    x = y = z = np.linspace(model.bounds[0], model.bounds[1], num=n)
    points = np.c_[x, y, z]
    spline = pv.Spline(points, n)
    slc = model.slice_along_line(spline, progress_bar=True)
    assert slc.n_points > 0
    slc = model.slice_along_line(spline, contour=True, progress_bar=True)
    assert slc.n_points > 0
    # Now check a simple line
    a = [model.bounds[0], model.bounds[2], model.bounds[4]]
    b = [model.bounds[1], model.bounds[3], model.bounds[5]]
    line = pv.Line(a, b, resolution=10)
    slc = model.slice_along_line(line, progress_bar=True)
    assert slc.n_points > 0
    # Now check a bad input
    a = [model.bounds[0], model.bounds[2], model.bounds[4]]
    b = [model.bounds[1], model.bounds[2], model.bounds[5]]
    line2 = pv.Line(a, b, resolution=10)
    line = line2.cast_to_unstructured_grid().merge(line.cast_to_unstructured_grid())
    with pytest.raises(ValueError):
        slc = model.slice_along_line(line, progress_bar=True)

    with pytest.raises(TypeError):
        one_cell = model.extract_cells(0, progress_bar=True)
        model.slice_along_line(one_cell, progress_bar=True)


def extract_points_invalid(sphere):
    with pytest.raises(ValueError):
        sphere.extract_points('invalid')

    with pytest.raises(TypeError):
        sphere.extract_points(object)


def test_extract_points():
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
        ]
    )
    # create pv object
    surf = pv.PolyData(vertices, faces)
    # extract sub-surface with adjacent cells
    sub_surf_adj = surf.extract_points(np.array([0, 1, 4, 5]), progress_bar=True)
    # extract sub-surface without adjacent cells
    sub_surf = surf.extract_points(np.array([0, 1, 4, 5]), adjacent_cells=False, progress_bar=True)
    # extract sub-surface without cells
    sub_surf_nocells = surf.extract_points(
        np.array([0, 1, 4, 5]), include_cells=False, progress_bar=True
    )
    # check sub-surface size
    assert sub_surf.n_points == 4
    assert sub_surf.n_cells == 1
    assert sub_surf_adj.n_points == 9
    assert sub_surf_adj.n_cells == 4
    assert sub_surf_nocells.cells[0] == 1


def test_slice_along_line_composite(composite):
    # Now test composite data structures
    a = [composite.bounds[0], composite.bounds[2], composite.bounds[4]]
    b = [composite.bounds[1], composite.bounds[3], composite.bounds[5]]
    line = pv.Line(a, b, resolution=10)
    output = composite.slice_along_line(line, progress_bar=True)
    assert output.n_blocks == composite.n_blocks


def test_interpolate():
    pdata = pv.PolyData()
    pdata.points = np.random.random((10, 3))
    pdata['scalars'] = np.random.random(10)
    surf = pv.Sphere(theta_resolution=10, phi_resolution=10)
    interp = surf.interpolate(pdata, radius=0.01, progress_bar=True)
    assert interp.n_points
    assert interp.n_arrays


def test_select_enclosed_points(uniform, hexbeam):
    surf = pv.Sphere(center=uniform.center, radius=uniform.length / 2.0)
    result = uniform.select_enclosed_points(surf, progress_bar=True)
    assert isinstance(result, type(uniform))
    assert 'SelectedPoints' in result.array_names
    assert result['SelectedPoints'].any()
    assert result.n_arrays == uniform.n_arrays + 1

    # Now check non-closed surface
    mesh = pv.Sphere(end_theta=270)
    surf = mesh.rotate_x(90, inplace=False)
    result = mesh.select_enclosed_points(surf, check_surface=False, progress_bar=True)
    assert isinstance(result, type(mesh))
    assert 'SelectedPoints' in result.array_names
    assert result.n_arrays == mesh.n_arrays + 1
    with pytest.raises(RuntimeError):
        result = mesh.select_enclosed_points(surf, check_surface=True, progress_bar=True)
    with pytest.raises(TypeError):
        result = mesh.select_enclosed_points(hexbeam, check_surface=True, progress_bar=True)


def test_decimate_boundary():
    mesh = examples.load_uniform()
    boundary = mesh.decimate_boundary(progress_bar=True)
    assert boundary.n_points


def test_extract_surface():
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
        ]
    )

    # introduce a minor variation to the location of the mid-side points
    quad_pts += np.random.random(quad_pts.shape) * 0.25
    pts = np.vstack((lin_pts, quad_pts))

    cells = np.hstack((20, np.arange(20))).astype(np.int64, copy=False)
    celltypes = np.array([CellType.QUADRATIC_HEXAHEDRON])
    grid = pv.UnstructuredGrid(cells, celltypes, pts)

    # expect each face to be divided 6 times since it has a midside node
    surf = grid.extract_surface(progress_bar=True)
    assert surf.n_faces == 36

    # expect each face to be divided several more times than the linear extraction
    surf_subdivided = grid.extract_surface(nonlinear_subdivision=5, progress_bar=True)
    assert surf_subdivided.n_faces > surf.n_faces

    # No subdivision, expect one face per cell
    surf_no_subdivide = grid.extract_surface(nonlinear_subdivision=0, progress_bar=True)
    assert surf_no_subdivide.n_faces == 6


def test_merge_general(uniform):
    thresh = uniform.threshold_percent([0.2, 0.5], progress_bar=True)  # unstructured grid
    con = uniform.contour()  # poly data
    merged = thresh + con
    assert isinstance(merged, pv.UnstructuredGrid)
    merged = con + thresh
    assert isinstance(merged, pv.UnstructuredGrid)
    # Pure PolyData inputs should yield poly data output
    merged = uniform.extract_surface() + con
    assert isinstance(merged, pv.PolyData)


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


def test_compute_cell_quality():
    mesh = pv.ParametricEllipsoid().triangulate().decimate(0.8)
    qual = mesh.compute_cell_quality(progress_bar=True)
    assert 'CellQuality' in qual.array_names
    with pytest.raises(KeyError):
        qual = mesh.compute_cell_quality(quality_measure='foo', progress_bar=True)


def test_compute_derivatives():
    mesh = examples.load_random_hills()
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
        scalars='Elevation', gradient=True, faster=True, progress_bar=True
    )
    assert 'gradient' in grad.array_names
    assert np.shape(grad['gradient'])[0] == mesh.n_points
    assert np.shape(grad['gradient'])[1] == 3

    grad = mesh.compute_derivative(scalars='vector', gradient=True, faster=True, progress_bar=True)
    assert 'gradient' in grad.array_names
    assert np.shape(grad['gradient'])[0] == mesh.n_points
    assert np.shape(grad['gradient'])[1] == 9

    with pytest.raises(ValueError):
        grad = mesh.compute_derivative(scalars='Elevation', gradient=False, progress_bar=True)

    with pytest.raises(TypeError):
        derv = mesh.compute_derivative(object)

    mesh.point_data.clear()
    with pytest.raises(MissingDataError):
        derv = mesh.compute_derivative()


def test_extract_subset():
    volume = examples.load_uniform()
    voi = volume.extract_subset([0, 3, 1, 4, 5, 7], progress_bar=True)
    assert isinstance(voi, pv.ImageData)
    # Test that we fix the confusing issue from extents in
    #   https://gitlab.kitware.com/vtk/vtk/-/issues/17938
    assert voi.origin == voi.bounds[::2]


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
    with pytest.raises(ValueError):
        volume.gaussian_smooth(scalars='cell_data')


def test_gaussian_smooth_cell_data_active():
    point_data = np.zeros((10, 10, 10))
    cell_data = np.zeros((9, 9, 9))
    volume = pv.ImageData(dimensions=(10, 10, 10))
    volume.point_data['point_data'] = point_data.flatten(order='F')
    volume.cell_data['cell_data'] = cell_data.flatten(order='F')
    volume.set_active_scalars('cell_data')
    with pytest.raises(ValueError):
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
    assert np.array_equal(volume.point_data['point_data'], volume_smoothed.point_data['point_data'])


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
        volume.point_data['point_data'], volume_outlier_smoothed.point_data['point_data']
    )


def test_image_dilate_erode_output_type():
    point_data = np.zeros((10, 10, 10))
    point_data[4, 4, 4] = 1
    volume = pv.ImageData(dimensions=(10, 10, 10))
    volume.point_data['point_data'] = point_data.flatten(order='F')
    volume_dilate_erode = volume.image_dilate_erode()
    assert isinstance(volume_dilate_erode, pv.ImageData)
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
    volume_dilated = volume.image_dilate_erode()  # default is binary dilation
    assert np.array_equal(
        volume_dilated.point_data['point_data'], point_data_dilated.flatten(order='F')
    )


def test_image_dilate_erode_erosion():
    point_data = np.zeros((10, 10, 10))
    point_data[4, 4, 4] = 1
    point_data_eroded = np.zeros((10, 10, 10))
    volume = pv.ImageData(dimensions=(10, 10, 10))
    volume.point_data['point_data'] = point_data.flatten(order='F')
    volume_eroded = volume.image_dilate_erode(0, 1)  # binary erosion
    assert np.array_equal(
        volume_eroded.point_data['point_data'], point_data_eroded.flatten(order='F')
    )


def test_image_dilate_erode_cell_data_specified():
    point_data = np.zeros((10, 10, 10))
    cell_data = np.zeros((9, 9, 9))
    volume = pv.ImageData(dimensions=(10, 10, 10))
    volume.point_data['point_data'] = point_data.flatten(order='F')
    volume.cell_data['cell_data'] = cell_data.flatten(order='F')
    with pytest.raises(ValueError):
        volume.image_dilate_erode(scalars='cell_data')


def test_image_dilate_erode_cell_data_active():
    point_data = np.zeros((10, 10, 10))
    cell_data = np.zeros((9, 9, 9))
    volume = pv.ImageData(dimensions=(10, 10, 10))
    volume.point_data['point_data'] = point_data.flatten(order='F')
    volume.cell_data['cell_data'] = cell_data.flatten(order='F')
    volume.set_active_scalars('cell_data')
    with pytest.raises(ValueError):
        volume.image_dilate_erode()


def test_image_threshold_output_type():
    threshold = 10  # 'random' value
    volume = examples.load_uniform()
    volume_thresholded = volume.image_threshold(threshold)
    assert isinstance(volume_thresholded, pv.ImageData)
    volume_thresholded = volume.image_threshold(threshold, scalars='Spatial Point Data')
    assert isinstance(volume_thresholded, pv.ImageData)


def test_image_threshold_wrong_threshold_length():
    threshold = (10, 10, 10)  # tuple with too many values
    volume = examples.load_uniform()
    with pytest.raises(ValueError):
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
        volume_thresholded.point_data['point_data'], point_data_thresholded.flatten(order='F')
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
        volume_thresholded.point_data['point_data'], point_data_thresholded.flatten(order='F')
    )


def test_extract_subset_structured():
    structured = examples.load_structured()
    voi = structured.extract_subset([0, 3, 1, 4, 0, 1])
    assert isinstance(voi, pv.StructuredGrid)
    assert voi.dimensions == (4, 4, 1)


@pytest.fixture
def structured_grids_split_coincident():
    """Two structured grids which are coincident along second axis (axis=1), and
    the grid from which they were extracted."""
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


def test_concatenate_structured(
    structured_grids_split_coincident, structured_grids_split_disconnected
):
    voi_1, voi_2, structured = structured_grids_split_coincident
    joined = voi_1.concatenate(voi_2, axis=1)
    assert structured.points == pytest.approx(joined.points)
    assert structured.volume == pytest.approx(joined.volume)
    assert structured.point_data['point_data'] == pytest.approx(joined.point_data['point_data'])
    assert structured.cell_data['cell_data'] == pytest.approx(joined.cell_data['cell_data'])


def test_concatenate_structured_bad_dimensions(structured_grids_split_coincident):
    voi_1, voi_2, structured = structured_grids_split_coincident

    # test invalid dimensions
    with pytest.raises(ValueError):
        voi_1.concatenate(voi_2, axis=0)

    with pytest.raises(ValueError):
        voi_1.concatenate(voi_2, axis=2)


def test_concatenate_structured_bad_inputs(structured_grids_split_coincident):
    voi_1, voi_2, structured = structured_grids_split_coincident
    with pytest.raises(RuntimeError):
        voi_1.concatenate(voi_2, axis=3)


def test_concatenate_structured_bad_point_data(structured_grids_split_coincident):
    voi_1, voi_2, structured = structured_grids_split_coincident
    voi_1['point_data'] = voi_1['point_data'] * 2.0
    with pytest.raises(RuntimeError):
        voi_1.concatenate(voi_2, axis=1)


def test_concatenate_structured_disconnected(structured_grids_split_disconnected):
    voi_1, voi_2 = structured_grids_split_disconnected
    with pytest.raises(RuntimeError):
        voi_1.concatenate(voi_2, axis=1)


def test_concatenate_structured_different_arrays(structured_grids_split_coincident):
    voi_1, voi_2, structured = structured_grids_split_coincident
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
        ]
    )
    cells = np.array([6, 0, 1, 2, 3, 4, 5])
    cell_types = np.array([CellType.QUADRATIC_TRIANGLE])
    ugrid = pv.UnstructuredGrid(cells, cell_types, points)
    tessellated = ugrid.tessellate(progress_bar=True)
    assert tessellated.n_cells > ugrid.n_cells
    assert tessellated.n_points > ugrid.n_points
    assert ugrid.tessellate(max_n_subdivide=6).n_cells > tessellated.n_cells
    assert ugrid.tessellate(merge_points=False).n_points > tessellated.n_points
    with pytest.raises(TypeError):
        pdata = pv.PolyData()
        tessellated = pdata.tessellate(progress_bar=True)


@pytest.mark.parametrize('num_cell_arrays,num_point_data', itertools.product([0, 1, 2], [0, 1, 2]))
def test_transform_mesh(datasets, num_cell_arrays, num_point_data):
    # rotate about x-axis by 90 degrees
    for dataset in datasets:
        tf = pv.core.utilities.transformations.axis_angle_rotation((1, 0, 0), 90)

        for i in range(num_cell_arrays):
            dataset.cell_data[f'C{i}'] = np.random.rand(dataset.n_cells, 3)

        for i in range(num_point_data):
            dataset.point_data[f'P{i}'] = np.random.rand(dataset.n_points, 3)

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


@pytest.mark.parametrize('num_cell_arrays,num_point_data', itertools.product([0, 1, 2], [0, 1, 2]))
def test_transform_mesh_and_vectors(datasets, num_cell_arrays, num_point_data):
    for dataset in datasets:
        # rotate about x-axis by 90 degrees
        tf = pv.core.utilities.transformations.axis_angle_rotation((1, 0, 0), 90)

        for i in range(num_cell_arrays):
            dataset.cell_data[f'C{i}'] = np.random.rand(dataset.n_cells, 3)

        for i in range(num_point_data):
            dataset.point_data[f'P{i}'] = np.random.rand(dataset.n_points, 3)

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
                transformed.cell_data[f'C{i}'][:, 0]
            )
            assert dataset.cell_data[f'C{i}'][:, 2] == pytest.approx(
                -transformed.cell_data[f'C{i}'][:, 1]
            )
            assert dataset.cell_data[f'C{i}'][:, 1] == pytest.approx(
                transformed.cell_data[f'C{i}'][:, 2]
            )

        for i in range(num_point_data):
            assert dataset.point_data[f'P{i}'][:, 0] == pytest.approx(
                transformed.point_data[f'P{i}'][:, 0]
            )
            assert dataset.point_data[f'P{i}'][:, 2] == pytest.approx(
                -transformed.point_data[f'P{i}'][:, 1]
            )
            assert dataset.point_data[f'P{i}'][:, 1] == pytest.approx(
                transformed.point_data[f'P{i}'][:, 2]
            )


@pytest.mark.parametrize("num_cell_arrays,num_point_data", itertools.product([0, 1, 2], [0, 1, 2]))
def test_transform_int_vectors_warning(datasets, num_cell_arrays, num_point_data):
    for dataset in datasets:
        tf = pv.core.utilities.transformations.axis_angle_rotation((1, 0, 0), 90)
        dataset.clear_data()
        for i in range(num_cell_arrays):
            dataset.cell_data[f"C{i}"] = np.random.randint(
                np.iinfo(int).max, size=(dataset.n_cells, 3)
            )
        for i in range(num_point_data):
            dataset.point_data[f"P{i}"] = np.random.randint(
                np.iinfo(int).max, size=(dataset.n_points, 3)
            )
        if not (num_cell_arrays == 0 and num_point_data == 0):
            with pytest.warns(UserWarning, match="Integer"):
                _ = dataset.transform(tf, transform_all_input_vectors=True, inplace=False)


@pytest.mark.parametrize(
    'dataset',
    [
        examples.load_uniform(),  # ImageData
        examples.load_rectilinear(),  # RectilinearGrid
    ],
)
def test_transform_inplace_bad_types(dataset):
    # assert that transformations of these types throw the correct error
    tf = pv.core.utilities.transformations.axis_angle_rotation(
        (1, 0, 0), 90
    )  # rotate about x-axis by 90 degrees
    with pytest.raises(TypeError):
        dataset.transform(tf, inplace=True)


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
            [1, 2, 3], dtype=float
        ).reshape((1, 3))
        dataset.point_data['P'] = np.arange(dataset.n_points)[:, np.newaxis] * np.array(
            [1, 2, 3], dtype=float
        ).reshape((1, 3))

        reflected = dataset.reflect(
            (1, 0, 0), transform_all_input_vectors=True, inplace=False, progress_bar=True
        )

        # assert isinstance(reflected, type(dataset))
        assert reflected.n_cells == dataset.n_cells
        assert reflected.n_points == dataset.n_points
        assert np.allclose(dataset.points[:, 0], -reflected.points[:, 0])
        assert np.allclose(dataset.points[:, 1:], reflected.points[:, 1:])

        # assert normals are reflected
        if hasattr(dataset, 'compute_normals'):
            assert np.allclose(
                dataset.cell_data['Normals'][:, 0], -reflected.cell_data['Normals'][:, 0]
            )
            assert np.allclose(
                dataset.cell_data['Normals'][:, 1:], reflected.cell_data['Normals'][:, 1:]
            )
            assert np.allclose(
                dataset.point_data['Normals'][:, 0], -reflected.point_data['Normals'][:, 0]
            )
            assert np.allclose(
                dataset.point_data['Normals'][:, 1:], reflected.point_data['Normals'][:, 1:]
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


@pytest.mark.parametrize(
    'dataset',
    [
        examples.load_uniform(),  # ImageData
        examples.load_rectilinear(),  # RectilinearGrid
    ],
)
def test_transform_inplace_bad_types_2(dataset):
    # assert that transformations of these types throw the correct error
    with pytest.raises(TypeError):
        dataset.reflect((1, 0, 0), inplace=True)


def test_extrude_rotate():
    resolution = 4
    line = pv.Line(pointa=(0, 0, 0), pointb=(1, 0, 0))

    with pytest.raises(ValueError):
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
    zmax = poly.bounds[5]
    assert zmax == translation
    xmax = poly.bounds[1]
    assert xmax == line.bounds[1] + dradius

    poly = line.extrude_rotate(angle=90.0, progress_bar=True, capping=True)
    xmin = poly.bounds[0]
    xmax = poly.bounds[1]
    ymin = poly.bounds[2]
    ymax = poly.bounds[3]
    assert (
        (xmin == line.bounds[0])
        and (xmax == line.bounds[1])
        and (ymin == line.bounds[0])
        and (ymax == line.bounds[1])
    )

    rotation_axis = (0, 1, 0)
    if not pv.vtk_version_info >= (9, 1, 0):
        with pytest.raises(VTKVersionError):
            poly = line.extrude_rotate(rotation_axis=rotation_axis, capping=True)
    else:
        poly = line.extrude_rotate(
            rotation_axis=rotation_axis, resolution=resolution, progress_bar=True, capping=True
        )
        assert poly.n_cells == line.n_points - 1
        assert poly.n_points == (resolution + 1) * line.n_points

    with pytest.raises(ValueError):
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
        center=(0, 0, 0), direction=direction, i_size=1, j_size=1, i_resolution=10, j_resolution=10
    )
    trim_surface = pv.Plane(
        center=(0, 0, 1), direction=direction, i_size=2, j_size=2, i_resolution=20, j_resolution=20
    )
    poly = mesh.extrude_trim(direction, trim_surface)
    assert np.isclose(poly.volume, 1.0)


@pytest.mark.parametrize('extrusion', ["boundary_edges", "all_edges"])
@pytest.mark.parametrize(
    'capping', ["intersection", "minimum_distance", "maximum_distance", "average_distance"]
)
def test_extrude_trim_strategy(extrusion, capping):
    direction = (0, 0, 1)
    mesh = pv.Plane(
        center=(0, 0, 0), direction=direction, i_size=1, j_size=1, i_resolution=10, j_resolution=10
    )
    trim_surface = pv.Plane(
        center=(0, 0, 1), direction=direction, i_size=2, j_size=2, i_resolution=20, j_resolution=20
    )
    poly = mesh.extrude_trim(direction, trim_surface, extrusion=extrusion, capping=capping)
    assert isinstance(poly, pv.PolyData)
    assert poly.n_cells
    assert poly.n_points


def test_extrude_trim_catch():
    direction = (0, 0, 1)
    mesh = pv.Plane()
    trim_surface = pv.Plane()
    with pytest.raises(ValueError):
        _ = mesh.extrude_trim(direction, trim_surface, extrusion="Invalid strategy")
    with pytest.raises(TypeError, match='Invalid type'):
        _ = mesh.extrude_trim(direction, trim_surface, extrusion=0)
    with pytest.raises(ValueError):
        _ = mesh.extrude_trim(direction, trim_surface, capping="Invalid strategy")
    with pytest.raises(TypeError, match='Invalid type'):
        _ = mesh.extrude_trim(direction, trim_surface, capping=0)
    with pytest.raises(TypeError):
        _ = mesh.extrude_trim('foobar', trim_surface)
    with pytest.raises(TypeError):
        _ = mesh.extrude_trim([1, 2], trim_surface)


def test_extrude_trim_inplace():
    direction = (0, 0, 1)
    mesh = pv.Plane(
        center=(0, 0, 0), direction=direction, i_size=1, j_size=1, i_resolution=10, j_resolution=10
    )
    trim_surface = pv.Plane(
        center=(0, 0, 1), direction=direction, i_size=2, j_size=2, i_resolution=20, j_resolution=20
    )
    mesh.extrude_trim(direction, trim_surface, inplace=True, progress_bar=True)
    assert np.isclose(mesh.volume, 1.0)


@pytest.mark.parametrize('inplace', [True, False])
def test_subdivide_adaptive(sphere, inplace):
    orig_n_faces = sphere.n_faces
    sub = sphere.subdivide_adaptive(0.01, 0.001, 100000, 2, inplace=inplace, progress_bar=True)
    assert sub.n_faces > orig_n_faces
    if inplace:
        assert sphere.n_faces == sub.n_faces


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
        if "Area" in integrated.array_names:
            assert integrated["Area"] > 0
        elif "Volume" in integrated.array_names:
            assert integrated["Volume"] > 0
        else:
            raise ValueError("Unexpected integration")


def test_integrate_data():
    """Test specific case."""
    # sphere with radius = 0.5, area = pi
    # increase resolution to increase precision
    sphere = pv.Sphere(theta_resolution=100, phi_resolution=100)
    sphere.cell_data["cdata"] = 2 * np.ones(sphere.n_cells)
    sphere.point_data["pdata"] = 3 * np.ones(sphere.n_points)

    integrated = sphere.integrate_data()
    assert np.isclose(integrated["Area"], np.pi, rtol=1e-3)
    assert np.isclose(integrated["cdata"], 2 * np.pi, rtol=1e-3)
    assert np.isclose(integrated["pdata"], 3 * np.pi, rtol=1e-3)


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


def test_subdivide_tetra(tetbeam):
    grid = tetbeam.subdivide_tetra()
    assert grid.n_cells == tetbeam.n_cells * 12


def test_extract_cells_by_type(tetbeam, hexbeam):
    combined = tetbeam + hexbeam

    hex_cells = combined.extract_cells_by_type([pv.CellType.HEXAHEDRON, pv.CellType.BEZIER_PYRAMID])
    assert np.alltrue(hex_cells.celltypes == pv.CellType.HEXAHEDRON)

    tet_cells = combined.extract_cells_by_type(pv.CellType.TETRA)
    assert np.alltrue(tet_cells.celltypes == pv.CellType.TETRA)

    should_be_empty = combined.extract_cells_by_type(pv.CellType.BEZIER_CURVE)
    assert should_be_empty.n_cells == 0

    with pytest.raises(TypeError, match='Invalid type'):
        combined.extract_cells_by_type(1.0)


def test_merge_points():
    cells = [2, 0, 1]
    celltypes = [pv.CellType.LINE]
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    pdata = pv.UnstructuredGrid(cells, celltypes, points)
    assert (
        pdata.merge(pdata, main_has_priority=True, merge_points=True, tolerance=1.0).n_points == 1
    )
    assert (
        pdata.merge(pdata, main_has_priority=True, merge_points=True, tolerance=0.1).n_points == 2
    )
