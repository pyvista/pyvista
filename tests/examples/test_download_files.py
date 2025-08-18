"""Test downloading files.

Enable these tests with:

pytest --test_downloads

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
import warnings

import numpy as np
import pytest
from pytest_cases import parametrize
import requests

import pyvista as pv
from pyvista import examples

if TYPE_CHECKING:
    import pytest_mock

if 'TEST_DOWNLOADS' in os.environ:
    warnings.warn('"TEST_DOWNLOADS" has been deprecated. Use `pytest --test_downloads`')

pytestmark = pytest.mark.needs_download
skip_9_1_0 = pytest.mark.needs_vtk_version(9, 1, 0)


def _on_ci():
    return os.environ.get('CI', 'false').lower() == 'true'


@pytest.fixture(scope='module', autouse=True)
def check_cache_on_ci():
    if not _on_ci():
        return

    assert examples.downloads._FILE_CACHE, (
        f'Expected `_FILE_CACHE` to be True on CI. Source is set to {examples.downloads.SOURCE}'
    )


@pytest.fixture(autouse=True)
def requests_fixture(mocker: pytest_mock.MockerFixture):
    """Mock the requests.get method to make sure HTTP requests are not emitted on CI,
    since can cause flakiness dut to GH rate limits.
    """
    if not _on_ci():
        yield
        return

    spy = mocker.spy(requests, 'get')
    yield
    assert spy.call_count == 0, spy.mock_calls


def test_download_single_sphere_animation():
    filename = examples.download_single_sphere_animation(load=False)
    assert Path(filename).is_file()

    data = examples.download_single_sphere_animation()
    assert data.n_blocks


def test_download_masonry_texture():
    data = examples.download_masonry_texture()
    assert isinstance(data, pv.Texture)


def test_download_usa_texture():
    data = examples.download_usa_texture()
    assert isinstance(data, pv.Texture)


def test_download_usa():
    data = examples.download_usa()
    assert np.any(data.points)


def test_download_st_helens():
    data = examples.download_st_helens()
    assert data.n_points


def test_download_bunny():
    data = examples.download_bunny()
    assert data.n_points


def test_download_cow():
    data = examples.download_cow()
    assert data.n_points


def test_download_faults():
    data = examples.download_faults()
    assert data.n_points


def test_download_tensors():
    data = examples.download_tensors()
    assert data.n_points


def test_download_head():
    data = examples.download_head()
    assert data.n_points


def test_download_bolt_nut():
    filenames = examples.download_bolt_nut(load=False)
    assert Path(filenames[0]).is_file()
    assert Path(filenames[1]).is_file()

    data = examples.download_bolt_nut()
    assert isinstance(data, pv.MultiBlock)


def test_download_clown():
    data = examples.download_clown()
    assert data.n_points


def test_download_exodus():
    data = examples.download_exodus()
    assert data.n_blocks


def test_download_fea_hertzian_contact_cylinder():
    filename = examples.download_fea_hertzian_contact_cylinder(load=False)
    assert Path(filename).is_file()

    data = examples.download_fea_hertzian_contact_cylinder()
    assert data.n_cells


def test_download_nefertiti():
    filename = examples.download_nefertiti(load=False)
    assert Path(filename).is_file()

    data = examples.download_nefertiti()
    assert data.n_cells


def test_download_blood_vessels():
    """Tests the parallel VTU reader"""
    filename = examples.download_blood_vessels(load=False)
    assert Path(filename).is_file()

    data = examples.download_blood_vessels()
    assert isinstance(data, pv.UnstructuredGrid)
    assert data.active_vectors_name == 'velocity'


def test_download_bunny_coarse():
    data = examples.download_bunny_coarse()
    assert data.n_cells


def test_download_cow_head():
    data = examples.download_cow_head()
    assert data.n_cells


def test_download_knee_full():
    data = examples.download_knee_full()
    assert data.n_cells


def test_download_iron_protein():
    data = examples.download_iron_protein()
    assert data.n_cells


def test_download_tetra_dc_mesh():
    data = examples.download_tetra_dc_mesh()
    assert data.n_blocks == 2
    assert data.keys() == ['forward', 'inverse']
    assert data['forward'].active_scalars_name == 'Resistivity(log10)-fwd'
    assert data['inverse'].active_scalars_name == 'Resistivity(log10)'


def test_download_tetrahedron():
    data = examples.download_tetrahedron()
    assert data.n_cells


def test_download_saddle_surface():
    data = examples.download_saddle_surface()
    assert data.n_cells


def test_download_foot_bones():
    data = examples.download_foot_bones()
    assert data.n_cells


def test_download_guitar():
    data = examples.download_guitar()
    assert data.n_cells


def test_download_quadratic_pyramid():
    data = examples.download_quadratic_pyramid()
    assert data.n_cells


def test_download_bird():
    data = examples.download_bird()
    assert data.n_cells


def test_download_bird_texture():
    data = examples.download_bird_texture()
    assert isinstance(data, pv.Texture)


def test_download_office():
    data = examples.download_office()
    assert data.n_cells


def test_download_horse_points():
    data = examples.download_horse_points()
    assert data.n_points


def test_download_horse():
    data = examples.download_horse()
    assert data.n_cells


def test_download_cake_easy():
    data = examples.download_cake_easy()
    assert data.n_cells


def test_download_cake_easy_texture():
    data = examples.download_cake_easy_texture()
    assert isinstance(data, pv.Texture)


@skip_9_1_0
def test_download_can_crushed_hdf():
    path = examples.download_can_crushed_hdf(load=False)
    assert Path(path).is_file()
    dataset = examples.download_can_crushed_hdf()
    assert isinstance(dataset, pv.UnstructuredGrid)


def test_download_can_crushed_vtu():
    path = examples.download_can_crushed_vtu(load=False)
    assert Path(path).is_file()
    dataset = examples.download_can_crushed_vtu()
    assert isinstance(dataset, pv.UnstructuredGrid)


def test_download_rectilinear_grid():
    data = examples.download_rectilinear_grid()
    assert data.n_cells


def test_download_gourds():
    data = examples.download_gourds()
    assert data.n_cells
    data = examples.download_gourds(zoom=True)
    assert data.n_cells


def test_download_gourds_texture():
    data = examples.download_gourds_texture()
    assert isinstance(data, pv.Texture)
    data = examples.download_gourds_texture(zoom=True)
    assert isinstance(data, pv.Texture)


def test_download_unstructured_grid():
    data = examples.download_unstructured_grid()
    assert data.n_cells


def test_download_letter_k():
    data = examples.download_letter_k()
    assert data.n_cells


def test_download_letter_a():
    data = examples.download_letter_a()
    assert data.n_cells


def test_download_poly_line():
    data = examples.download_poly_line()
    assert data.n_cells


def test_download_cad_model():
    data = examples.download_cad_model()
    assert data.n_cells


def test_download_frog():
    data = examples.download_frog()
    assert data.n_cells


def test_download_chest():
    data = examples.download_chest()
    assert data.n_cells


def test_download_prostate():
    data = examples.download_prostate()
    assert data.n_cells


def test_download_filled_contours():
    data = examples.download_filled_contours()
    assert data.n_cells


def test_download_doorman():
    data = examples.download_doorman()
    assert data.n_cells


def test_download_mug():
    data = examples.download_mug()
    assert data.n_blocks


def test_download_oblique_cone():
    data = examples.download_oblique_cone()
    assert data.n_cells


def test_download_emoji():
    data = examples.download_emoji()
    assert data.n_cells


def test_download_emoji_texture():
    data = examples.download_emoji_texture()
    assert isinstance(data, pv.Texture)


def test_download_teapot():
    data = examples.download_teapot()
    assert data.n_cells


def test_download_brain():
    data = examples.download_brain()
    assert data.n_cells


def test_download_structured_grid():
    data = examples.download_structured_grid()
    assert data.n_cells


def test_download_structured_grid_two():
    data = examples.download_structured_grid_two()
    assert data.n_cells


def test_download_trumpet():
    data = examples.download_trumpet()
    assert data.n_cells


def test_download_face():
    data = examples.download_face()
    assert data.n_cells


def test_download_sky_box_nz():
    data = examples.download_sky_box_nz()
    assert data.n_cells


def test_download_sky_box_nz_texture():
    data = examples.download_sky_box_nz_texture()
    assert isinstance(data, pv.Texture)


def test_download_disc_quads():
    data = examples.download_disc_quads()
    assert data.n_cells


def test_download_honolulu():
    data = examples.download_honolulu()
    assert data.n_cells


def test_download_motor():
    data = examples.download_motor()
    assert data.n_cells


def test_download_tri_quadratic_hexahedron():
    data = examples.download_tri_quadratic_hexahedron()
    path = examples.download_tri_quadratic_hexahedron(load=False)
    assert data.n_cells
    assert data.n_arrays == 0
    assert pv.read(path).n_arrays != 0


def test_download_human():
    data = examples.download_human()
    assert data.n_cells


def test_download_vtk():
    data = examples.download_vtk()
    assert data.n_cells


def test_download_spider():
    data = examples.download_spider()
    assert data.n_cells


def test_download_carotid():
    filename = examples.download_carotid(load=False)
    assert Path(filename).is_file()

    data = examples.download_carotid()
    assert data.n_cells


def test_download_blow():
    data = examples.download_blow()
    assert data.n_cells


def test_download_shark():
    data = examples.download_shark()
    assert data.n_cells


def test_download_dragon():
    data = examples.download_dragon()
    assert data.n_cells


def test_download_armadillo():
    data = examples.download_armadillo()
    assert data.n_cells


def test_download_gears():
    data = examples.download_gears()
    assert data.n_cells


def test_download_torso():
    data = examples.download_torso()
    assert data.n_cells


def test_download_kitchen():
    filename = examples.download_kitchen(load=False)
    assert Path(filename).is_file()

    data = examples.download_kitchen()
    assert data.n_cells


def test_download_kitchen_split():
    data = examples.download_kitchen(split=True)
    assert data.n_blocks


def test_download_backward_facing_step():
    filename = examples.download_backward_facing_step(load=False)
    assert Path(filename).is_file()

    data = examples.download_backward_facing_step()
    assert data.n_blocks


def test_download_topo_global():
    data = examples.download_topo_global()
    assert isinstance(data, pv.PolyData)
    assert data.n_cells


def test_download_topo_land():
    data = examples.download_topo_land()
    assert isinstance(data, pv.PolyData)
    assert data.n_cells


def test_download_coastlines():
    data = examples.download_coastlines()
    assert data.n_cells


def test_download_knee():
    data = examples.download_knee()
    assert data.n_cells


def test_download_lidar():
    data = examples.download_lidar()
    assert data.n_cells


def test_download_pine_roots():
    data = examples.download_pine_roots()
    assert data.n_points


def test_download_dicom_stack():
    filename = examples.download_dicom_stack(load=False)
    assert Path(filename).is_dir()

    data = examples.download_dicom_stack()
    assert isinstance(data, pv.ImageData)
    assert all([data.n_points, data.n_cells])


def test_download_teapot_vrml():
    filename = examples.vrml.download_teapot()
    assert Path(filename).is_file()


def test_download_sextant_vrml():
    filename = examples.vrml.download_sextant()
    assert Path(filename).is_file()


def test_download_cavity():
    filename = examples.download_cavity(load=False)
    assert Path(filename).is_file()

    dataset = examples.download_cavity(load=True)
    assert isinstance(dataset, pv.MultiBlock)


def test_download_openfoam_tubes():
    filename = examples.download_openfoam_tubes(load=False)
    assert Path(filename).is_file()

    dataset = examples.download_openfoam_tubes(load=True)
    assert isinstance(dataset, pv.MultiBlock)


def test_download_lucy():
    filename = examples.download_lucy(load=False)
    assert Path(filename).is_file()

    dataset = examples.download_lucy(load=True)
    assert isinstance(dataset, pv.PolyData)


def test_download_pump_bracket():
    filename = examples.download_pump_bracket(load=False)
    assert Path(filename).is_file()

    dataset = examples.download_pump_bracket(load=True)
    assert isinstance(dataset, pv.UnstructuredGrid)
    assert len(dataset.point_data) == 10


def test_download_electronics_cooling():
    filenames = examples.download_electronics_cooling(load=False)
    for filename in filenames:
        assert Path(filename).is_file()

    structure, air = examples.download_electronics_cooling(load=True)
    assert isinstance(structure, pv.PolyData)
    assert isinstance(air, pv.UnstructuredGrid)


def test_download_angular_sector():
    filename = examples.download_angular_sector(load=False)
    assert Path(filename).is_file()

    dataset = examples.download_angular_sector(load=True)
    assert isinstance(dataset, pv.UnstructuredGrid)


def test_download_mount_damavand():
    filename = examples.download_mount_damavand(load=False)
    assert Path(filename).is_file()

    dataset = examples.download_mount_damavand(load=True)
    assert isinstance(dataset, pv.PolyData)


def test_download_cubemap_space_4k():
    dataset = examples.download_cubemap_space_4k()
    assert isinstance(dataset, pv.Texture)


def test_download_cubemap_space_16k():
    dataset = examples.download_cubemap_space_16k()
    assert isinstance(dataset, pv.Texture)


def test_download_particles_lethe():
    filename = examples.download_particles_lethe(load=False)
    assert Path(filename).is_file()

    dataset = examples.download_particles_lethe(load=True)
    assert isinstance(dataset, pv.UnstructuredGrid)


def test_download_cubemap_park():
    dataset = examples.download_cubemap_park()
    assert isinstance(dataset, pv.Texture)


def test_download_gif_simple():
    filename = examples.download_gif_simple(load=False)
    assert Path(filename).is_file()
    assert filename.endswith('gif')

    dataset = examples.download_gif_simple(load=True)
    assert isinstance(dataset, pv.ImageData)
    assert 'frame0' in dataset.point_data


@parametrize(high_resolution=[True, False])
def test_download_black_vase(high_resolution: bool):
    filename = examples.download_black_vase(load=False, high_resolution=high_resolution)
    assert Path(filename).is_file()
    assert filename.endswith('vtp')

    dataset = examples.download_black_vase(load=True, high_resolution=high_resolution)
    assert isinstance(dataset, pv.PolyData)
    assert dataset.n_points == 17_337 if not high_resolution else 1_611_789


@parametrize(high_resolution=[True, False])
def test_download_ivan_angel(high_resolution: bool):
    filename = examples.download_ivan_angel(load=False, high_resolution=high_resolution)
    assert Path(filename).is_file()
    assert filename.endswith('vtp')

    dataset = examples.download_ivan_angel(load=True, high_resolution=high_resolution)
    assert isinstance(dataset, pv.PolyData)
    assert dataset.n_points == 18_412 if not high_resolution else 1_811_531


@parametrize(high_resolution=[True, False])
def test_download_bird_bath(high_resolution: bool):
    filename = examples.download_bird_bath(load=False, high_resolution=high_resolution)
    assert Path(filename).is_file()
    assert filename.endswith('vtp')

    dataset = examples.download_bird_bath(load=True, high_resolution=high_resolution)
    assert isinstance(dataset, pv.PolyData)
    assert dataset.n_points == 18_796 if not high_resolution else 1_831_383


@parametrize(high_resolution=[True, False])
def test_download_owl(high_resolution: bool):
    filename = examples.download_owl(load=False, high_resolution=high_resolution)
    assert Path(filename).is_file()
    assert filename.endswith('vtp')

    dataset = examples.download_owl(load=True, high_resolution=high_resolution)
    assert isinstance(dataset, pv.PolyData)
    assert dataset.n_points == 12442 if not high_resolution else 1_221_756


@parametrize(high_resolution=[True, False])
def test_download_plastic_vase(high_resolution: bool):
    filename = examples.download_plastic_vase(load=False, high_resolution=high_resolution)
    assert Path(filename).is_file()
    assert filename.endswith('vtp')

    dataset = examples.download_plastic_vase(load=True, high_resolution=high_resolution)
    assert isinstance(dataset, pv.PolyData)
    assert dataset.n_points == 18238 if not high_resolution else 1_796_805


@parametrize(high_resolution=[True, False])
def test_download_sea_vase(high_resolution: bool):
    filename = examples.download_sea_vase(load=False, high_resolution=high_resolution)
    assert Path(filename).is_file()
    assert filename.endswith('vtp')

    dataset = examples.download_sea_vase(load=True, high_resolution=high_resolution)
    assert isinstance(dataset, pv.PolyData)
    assert dataset.n_points == 18_063 if not high_resolution else 1_810_012


def test_download_sparse_points():
    filename = examples.download_sparse_points(load=False)
    assert Path(filename).is_file()
    assert filename.endswith('txt')

    dataset = examples.download_sparse_points(load=True)
    assert isinstance(dataset, pv.PolyData)
    assert dataset.n_points == 16


def test_download_puppy():
    dataset = examples.download_puppy()
    assert isinstance(dataset, pv.ImageData)
    assert dataset.n_points > 1_000_000


def test_download_puppy_texture():
    dataset = examples.download_puppy_texture()
    assert isinstance(dataset, pv.Texture)


def test_download_gourds_pnm():
    dataset = examples.download_gourds_pnm()
    assert isinstance(dataset, pv.ImageData)


def test_download_model_with_variance():
    dataset = examples.download_model_with_variance()
    assert isinstance(dataset, pv.UnstructuredGrid)


def test_download_thermal_probes():
    dataset = examples.download_thermal_probes()
    assert isinstance(dataset, pv.PolyData)


def test_download_turbine_blade():
    dataset = examples.download_turbine_blade()
    assert isinstance(dataset, pv.PolyData)


def test_download_crater_topo():
    dataset = examples.download_crater_topo()
    assert isinstance(dataset, pv.ImageData)


def test_download_crater_imagery():
    dataset = examples.download_crater_imagery()
    assert isinstance(dataset, pv.Texture)


def test_download_dolfin():
    dataset = examples.download_dolfin()
    assert isinstance(dataset, pv.UnstructuredGrid)


def test_download_meshio_xdmf():
    dataset = examples.download_meshio_xdmf()
    assert isinstance(dataset, pv.MultiBlock)


def test_download_damavand_volcano():
    filename = examples.download_damavand_volcano(load=False)
    assert Path(filename).is_file()

    dataset = examples.download_damavand_volcano()
    assert isinstance(dataset, pv.ImageData)


def test_download_delaunay_example():
    dataset = examples.download_delaunay_example()
    assert isinstance(dataset, pv.PolyData)


def test_download_embryo():
    filename = examples.download_embryo(load=False)
    assert Path(filename).is_file()

    dataset = examples.download_embryo()
    assert isinstance(dataset, pv.ImageData)
    assert not np.any(dataset['SLCImage'] == 255)


def test_download_antarctica_velocity():
    dataset = examples.download_antarctica_velocity()
    assert isinstance(dataset, pv.PolyData)


def test_download_room_surface_mesh():
    dataset = examples.download_room_surface_mesh()
    assert isinstance(dataset, pv.PolyData)


def test_download_beach():
    dataset = examples.download_beach()
    assert isinstance(dataset, pv.ImageData)


def test_download_rgba_texture():
    dataset = examples.download_rgba_texture()
    assert isinstance(dataset, pv.Texture)


def test_download_vtk_logo():
    dataset = examples.download_vtk_logo()
    assert isinstance(dataset, pv.Texture)


def test_download_gpr_data_array():
    filename = examples.download_gpr_data_array(load=False)
    assert Path(filename).is_file()

    dataset = examples.download_gpr_data_array()
    assert isinstance(dataset, np.ndarray)


def test_download_gpr_path():
    filename = examples.download_gpr_path(load=False)
    assert Path(filename).is_file()

    dataset = examples.download_gpr_path()
    assert isinstance(dataset, pv.PolyData)


def test_download_woman():
    dataset = examples.download_woman()
    assert isinstance(dataset, pv.PolyData)


def test_download_lobster():
    dataset = examples.download_lobster()
    assert isinstance(dataset, pv.PolyData)


def test_download_face2():
    dataset = examples.download_face2()
    assert isinstance(dataset, pv.PolyData)


def test_download_urn():
    dataset = examples.download_urn()
    assert isinstance(dataset, pv.PolyData)


def test_download_pepper():
    dataset = examples.download_pepper()
    assert isinstance(dataset, pv.PolyData)


def test_download_drill():
    dataset = examples.download_drill()
    assert isinstance(dataset, pv.PolyData)


@parametrize(high_resolution=[True, False])
def test_download_action_figure(high_resolution: bool):
    dataset = examples.download_action_figure(high_resolution=high_resolution)
    assert isinstance(dataset, pv.PolyData)


def test_download_notch_stress():
    dataset = examples.download_notch_stress()
    assert isinstance(dataset, pv.UnstructuredGrid)


def test_download_notch_displacement():
    dataset = examples.download_notch_displacement()
    assert isinstance(dataset, pv.UnstructuredGrid)


def test_download_louis_louvre():
    dataset = examples.download_louis_louvre()
    assert isinstance(dataset, pv.PolyData)


def test_download_cylinder_crossflow():
    filename = examples.download_cylinder_crossflow(load=False)
    assert Path(filename).is_file()

    dataset = examples.download_cylinder_crossflow()
    assert isinstance(dataset, pv.MultiBlock)


def test_download_naca():
    filename = examples.download_naca(load=False)
    assert Path(filename).is_file()

    dataset = examples.download_naca()
    assert isinstance(dataset, pv.MultiBlock)


def test_download_lshape():
    filename = examples.download_lshape(load=False)
    assert Path(filename).is_file()

    dataset = examples.download_lshape()
    assert isinstance(dataset, pv.MultiBlock)


def test_download_wavy():
    filename = examples.download_wavy(load=False)
    assert Path(filename).is_file()

    dataset = examples.download_wavy()
    assert isinstance(dataset, pv.MultiBlock)


def test_download_dual_sphere_animation():
    filename = examples.download_dual_sphere_animation(load=False)
    assert Path(filename).is_file()

    dataset = examples.download_dual_sphere_animation()
    assert isinstance(dataset, pv.MultiBlock)


@skip_9_1_0
def test_download_cgns_structured():
    filename = examples.download_cgns_structured(load=False)
    assert Path(filename).is_file()

    dataset = examples.download_cgns_structured()
    assert isinstance(dataset, pv.MultiBlock)


def test_download_tecplot_ascii():
    filename = examples.download_tecplot_ascii(load=False)
    assert Path(filename).is_file()

    dataset = examples.download_tecplot_ascii()
    assert isinstance(dataset, pv.MultiBlock)


@skip_9_1_0
def test_download_cgns_multi():
    filename = examples.download_cgns_multi(load=False)
    assert Path(filename).is_file()

    dataset = examples.download_cgns_multi()
    assert isinstance(dataset, pv.MultiBlock)


def test_download_parched_canal_4k():
    dataset = examples.download_parched_canal_4k()
    assert isinstance(dataset, pv.Texture)


def test_download_cells_nd():
    dataset = examples.download_cells_nd()
    assert isinstance(dataset, pv.UnstructuredGrid)


def test_download_moonlanding_image():
    dataset = examples.download_moonlanding_image()
    assert isinstance(dataset, pv.ImageData)


def test_download_gltf_milk_truck():
    filename = examples.gltf.download_milk_truck()
    assert Path(filename).is_file()
    pl = pv.Plotter()
    pl.import_gltf(filename)


def test_download_gltf_damaged_helmet():
    filename = examples.gltf.download_damaged_helmet()
    assert Path(filename).is_file()
    pl = pv.Plotter()
    pl.import_gltf(filename)


@pytest.mark.needs_vtk_version(
    less_than=(9, 1),
    reason='Skip until glTF extension KHR_texture_transform is supported.',
)
def test_download_gltf_sheen_chair():
    filename = examples.gltf.download_sheen_chair()
    assert Path(filename).is_file()
    pl = pv.Plotter()
    pl.import_gltf(filename)


def test_download_gltf_gearbox():
    filename = examples.gltf.download_gearbox()
    assert Path(filename).is_file()
    pl = pv.Plotter()
    pl.import_gltf(filename)


def test_download_gltf_avocado():
    filename = examples.gltf.download_avocado()
    assert Path(filename).is_file()
    pl = pv.Plotter()
    pl.import_gltf(filename)


@skip_9_1_0
def test_download_cloud_dark_matter():
    filename = examples.download_cloud_dark_matter(load=False)
    assert Path(filename).is_file()
    assert filename.endswith('npy')

    dataset = examples.download_cloud_dark_matter(load=True)
    assert isinstance(dataset, pv.DataSet)
    assert dataset.n_points == 32314


@skip_9_1_0
def test_download_cloud_dark_matter_dense():
    filename = examples.download_cloud_dark_matter_dense(load=False)
    assert Path(filename).is_file()
    assert filename.endswith('npy')

    dataset = examples.download_cloud_dark_matter_dense(load=True)
    assert isinstance(dataset, pv.DataSet)
    assert dataset.n_points == 2062256


def test_download_stars_cloud_hyg():
    filename = examples.download_stars_cloud_hyg(load=False)
    assert Path(filename).is_file()
    assert filename.endswith('vtp')

    dataset = examples.download_stars_cloud_hyg(load=True)
    assert isinstance(dataset, pv.PolyData)
    assert dataset.n_points == 107857


def test_download_cad_model_case():
    filename = examples.download_cad_model_case(load=False)
    assert Path(filename).is_file()
    assert filename.endswith('vtp')

    dataset = examples.download_cad_model_case(load=True)
    assert isinstance(dataset, pv.PolyData)
    assert dataset.n_points == 7677


def test_download_aero_bracket():
    filename = examples.download_aero_bracket(load=False)
    assert Path(filename).is_file()
    assert filename.endswith('vtu')

    dataset = examples.download_aero_bracket(load=True)
    assert isinstance(dataset, pv.UnstructuredGrid)
    assert len(dataset.point_data.keys()) == 3


def test_download_coil_magnetic_field():
    filename = examples.download_coil_magnetic_field(load=False)
    assert Path(filename).is_file()
    assert filename.endswith('vti')

    dataset = examples.download_coil_magnetic_field(load=True)
    assert isinstance(dataset, pv.ImageData)
    assert dataset.n_points == 531441


def test_load_sun():
    mesh = examples.planets.load_sun()
    assert mesh.n_cells


def test_load_moon():
    mesh = examples.planets.load_moon()
    assert mesh.n_cells


def test_load_mercury():
    mesh = examples.planets.load_mercury()
    assert mesh.n_cells


def test_load_venus():
    mesh = examples.planets.load_venus()
    assert mesh.n_cells


def test_load_mars():
    mesh = examples.planets.load_mars()
    assert mesh.n_cells


def test_load_jupiter():
    mesh = examples.planets.load_jupiter()
    assert mesh.n_cells


def test_load_saturn():
    mesh = examples.planets.load_saturn()
    assert mesh.n_cells


def test_load_saturn_rings():
    mesh = examples.planets.load_saturn_rings()
    assert mesh.n_cells


def test_load_uranus():
    mesh = examples.planets.load_uranus()
    assert mesh.n_cells


def test_load_neptune():
    mesh = examples.planets.load_neptune()
    assert mesh.n_cells


def test_load_pluto():
    mesh = examples.planets.load_pluto()
    assert mesh.n_cells


def test_download_nek5000():
    filename = examples.download_nek5000(load=False)
    assert Path(filename).is_file()
    assert filename.endswith('nek5000')

    # nek5000 reader can only be used with vtk >= 9.3
    if pv.vtk_version_info >= (9, 3):
        nek_reader = pv.get_reader(filename)
        assert nek_reader.number_time_points == 11

        nek_data = examples.download_nek5000(load=True)
        assert isinstance(nek_data, pv.UnstructuredGrid)


@pytest.mark.skip_windows
def test_download_biplane():
    filename = examples.download_biplane(load=False)
    assert Path(filename).is_file()
    assert filename.endswith('exo')

    biplane = examples.download_biplane()
    assert isinstance(biplane, pv.MultiBlock)


def test_download_head2():
    filename = examples.download_head_2(load=False)
    assert Path(filename).is_file()
    assert filename.endswith('vti')

    biplane = examples.download_head_2()
    assert isinstance(biplane, pv.ImageData)


def test_download_great_white_shark():
    filename = examples.download_great_white_shark(load=False)
    assert (p := Path(filename)).is_file()
    assert p.suffix == '.stl'

    shark = examples.download_great_white_shark()
    assert isinstance(shark, pv.PolyData)


def test_download_grey_nurse_shark():
    filename = examples.download_grey_nurse_shark(load=False)
    assert (p := Path(filename)).is_file()
    assert p.suffix == '.stl'

    shark = examples.download_grey_nurse_shark()
    assert isinstance(shark, pv.PolyData)


def test_download_carburetor():
    filename = examples.download_carburetor(load=False)
    assert (p := Path(filename)).is_file()
    assert p.suffix == '.ply'

    carburetor = examples.download_carburetor()
    assert isinstance(carburetor, pv.PolyData)


def test_download_dikhololo_night():
    filename = examples.download_dikhololo_night(load=False)
    assert (p := Path(filename)).is_file()
    assert p.suffix == '.hdr'

    mesh = examples.download_dikhololo_night()
    assert isinstance(mesh, pv.Texture)


def test_download_victorian_goblet_face_illusion():
    filename = examples.download_victorian_goblet_face_illusion(load=False)
    assert (p := Path(filename)).is_file()
    assert p.suffix == '.stl'

    mesh = examples.download_victorian_goblet_face_illusion()
    assert isinstance(mesh, pv.PolyData)


def test_download_reservoir():
    filename = examples.download_reservoir(load=False)
    assert (p := Path(filename)).is_file()
    assert p.suffix == '.vtu'

    mesh = examples.download_reservoir()
    assert isinstance(mesh, pv.ExplicitStructuredGrid)


@parametrize(high_resolution=[True, False])
def test_download_whole_body_ct_male(high_resolution: bool):
    filename = examples.download_whole_body_ct_male(load=False, high_resolution=high_resolution)

    if not high_resolution:
        assert (p := (Path(filename))).is_file()
        assert p.suffix == '.vtm'

    dataset: pv.MultiBlock = examples.download_whole_body_ct_male(
        load=True, high_resolution=high_resolution
    )
    assert isinstance(dataset, pv.MultiBlock)
    npoints = max(b.n_points for b in dataset.recursive_iterator())
    assert npoints == 6_988_800 if not high_resolution else 56_012_800


@parametrize(high_resolution=[True, False])
def test_download_whole_body_ct_female(high_resolution: bool):
    filename = examples.download_whole_body_ct_female(load=False, high_resolution=high_resolution)

    if not high_resolution:
        assert (p := (Path(filename))).is_file()
        assert p.suffix == '.vtm'

    dataset = examples.download_whole_body_ct_female(load=True, high_resolution=high_resolution)
    assert isinstance(dataset, pv.MultiBlock)
    npoints = max(b.n_points for b in dataset.recursive_iterator())
    assert npoints == 6_937_600 if not high_resolution else 55_603_200


def test_download_headsq():
    filename = examples.download_headsq(load=False)
    assert (p := Path(filename)).is_file()
    assert p.suffix == '.nhdr'

    mesh = examples.download_headsq()
    assert isinstance(mesh, pv.ImageData)


def test_download_t3_grid_0():
    filename = examples.download_t3_grid_0(load=False)
    assert (p := Path(filename)).is_file()
    assert p.suffix == '.mnc'

    mesh = examples.download_t3_grid_0()
    assert isinstance(mesh, pv.ImageData)


def test_download_full_head():
    filename = examples.download_full_head(load=False)
    assert (p := Path(filename)).is_file()
    assert p.suffix == '.mhd'

    mesh = examples.download_full_head()
    assert isinstance(mesh, pv.ImageData)


@parametrize(partial=[True, False])
@pytest.mark.needs_vtk_version(9, 1, less_than=(9, 2))  # 9.1 for HDFReader, 9.2 for example
def test_download_can(partial: bool):
    filename = examples.download_can(load=False, partial=partial)

    if partial:
        assert (p := (Path(filename))).is_file()
        assert p.suffix == '.hdf'
    else:
        assert all(Path(f).is_file() for f in filename)
        assert all(Path(f).suffix == '.hdf' for f in filename)

    dataset: pv.UnstructuredGrid = examples.download_can(load=True, partial=partial)
    assert isinstance(dataset, pv.UnstructuredGrid)
    assert dataset.n_points == 6724 if partial else 20_172


@parametrize(partial=[True, False])
@pytest.mark.needs_vtk_version(9, 2)
def test_download_can_raises(partial: bool):
    with pytest.raises(pv.VTKVersionError):
        examples.download_can(partial=partial)


def test_download_fea_bracket():
    filename = examples.download_fea_bracket(load=False)
    assert (p := Path(filename)).is_file()
    assert p.suffix == '.vtu'

    mesh = examples.download_fea_bracket()
    assert isinstance(mesh, pv.UnstructuredGrid)


def test_download_yinyang():
    filename = examples.download_yinyang(load=False)
    assert (p := Path(filename)).is_file()
    assert p.suffix == '.png'

    mesh = examples.download_yinyang()
    assert isinstance(mesh, pv.ImageData)
