import os

import numpy as np
import vtk

import pyvista
from pyvista import examples

ffmpeg_failed = False
try:
    try:
        import imageio_ffmpeg

        imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        import imageio

        imageio.plugins.ffmpeg.download()
except:  # noqa: E722
    ffmpeg_failed = True

TEST_DOWNLOADS = False
try:
    if os.environ['TEST_DOWNLOADS'].lower() == 'true':
        TEST_DOWNLOADS = True
except KeyError:
    pass


def test_load_nut():
    mesh = examples.load_nut()
    assert mesh.n_points


def test_load_ant():
    """Load ply ant mesh"""
    mesh = examples.load_ant()
    assert mesh.n_points


def test_load_airplane():
    """Load ply airplane mesh"""
    mesh = examples.load_airplane()
    assert mesh.n_points


def test_load_sphere():
    """Loads sphere ply mesh"""
    mesh = examples.load_sphere()
    assert mesh.n_points


def test_load_channels():
    """Loads geostat training image"""
    mesh = examples.load_channels()
    assert mesh.n_points


def test_load_spline():
    mesh = examples.load_spline()
    assert mesh.n_points


def test_load_random_hills():
    mesh = examples.load_random_hills()
    assert mesh.n_cells


if TEST_DOWNLOADS:

    def test_download_single_sphere_animation():
        data = examples.download_single_sphere_animation()
        assert data.n_blocks

    def test_download_masonry_texture():
        data = examples.download_masonry_texture()
        assert isinstance(data, vtk.vtkTexture)

    def test_download_usa_texture():
        data = examples.download_usa_texture()
        assert isinstance(data, vtk.vtkTexture)

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
        data = examples.download_bolt_nut()
        assert isinstance(data, pyvista.MultiBlock)

    def test_download_clown():
        data = examples.download_clown()
        assert data.n_points

    def test_download_exodus():
        data = examples.download_exodus()
        assert data.n_blocks

    def test_download_nefertiti():
        data = examples.download_nefertiti()
        assert data.n_cells

    def test_download_blood_vessels():
        """Tests the parallel VTU reader"""
        data = examples.download_blood_vessels()
        assert isinstance(data, pyvista.UnstructuredGrid)

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
        assert data.n_blocks

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
        assert isinstance(data, vtk.vtkTexture)

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
        assert isinstance(data, vtk.vtkTexture)

    def test_download_can_crushed_hdf():
        path = examples.download_can_crushed_hdf(load=False)
        assert os.path.isfile(path)
        dataset = examples.download_can_crushed_hdf()
        assert isinstance(dataset, pyvista.UnstructuredGrid)

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
        assert isinstance(data, vtk.vtkTexture)
        data = examples.download_gourds_texture(zoom=True)
        assert isinstance(data, vtk.vtkTexture)

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
        assert isinstance(data, vtk.vtkTexture)

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
        assert isinstance(data, vtk.vtkTexture)

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
        assert data.n_cells

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
        data = examples.download_kitchen()
        assert data.n_cells

    def test_download_kitchen_split():
        data = examples.download_kitchen(split=True)
        assert data.n_blocks

    def test_download_backward_facing_step():
        data = examples.download_backward_facing_step()
        assert data.n_blocks

    # def test_download_topo_global():
    #     data = examples.download_topo_global()
    #     assert data.n_cells
    #
    # def test_download_topo_land():
    #     data = examples.download_topo_land()
    #     assert data.n_cells

    def test_download_coastlines():
        data = examples.download_coastlines()
        assert data.n_cells

    def test_download_knee():
        data = examples.download_knee()
        assert data.n_cells

    def test_download_lidar():
        data = examples.download_lidar()
        assert data.n_cells

    def test_pine_roots():
        data = examples.download_pine_roots()
        assert data.n_points

    def test_download_dicom_stack():
        data = examples.download_dicom_stack()
        assert isinstance(data, pyvista.UniformGrid)
        assert all([data.n_points, data.n_cells])

    def test_vrml_download_teapot():
        filename = examples.vrml.download_teapot()
        assert os.path.isfile(filename)

    def test_vrml_download_sextant():
        filename = examples.vrml.download_sextant()
        assert os.path.isfile(filename)

    def test_download_cavity():
        filename = examples.download_cavity(load=False)
        assert os.path.isfile(filename)

        dataset = examples.download_cavity(load=True)
        assert isinstance(dataset, pyvista.MultiBlock)

    def test_download_lucy():
        filename = examples.download_lucy(load=False)
        assert os.path.isfile(filename)

        dataset = examples.download_lucy(load=True)
        assert isinstance(dataset, pyvista.PolyData)

    def test_angular_sector():
        filename = examples.download_angular_sector(load=False)
        assert os.path.isfile(filename)

        dataset = examples.download_angular_sector(load=True)
        assert isinstance(dataset, pyvista.UnstructuredGrid)

    def test_mount_damavand():
        filename = examples.download_mount_damavand(load=False)
        assert os.path.isfile(filename)

        dataset = examples.download_mount_damavand(load=True)
        assert isinstance(dataset, pyvista.PolyData)

    def test_download_cubemap_space_4k():
        dataset = examples.download_cubemap_space_4k()
        assert isinstance(dataset, pyvista.Texture)

    def test_download_cubemap_space_16k():
        dataset = examples.download_cubemap_space_16k()
        assert isinstance(dataset, pyvista.Texture)

    def test_particles_lethe():
        filename = examples.download_particles_lethe(load=False)
        assert os.path.isfile(filename)

        dataset = examples.download_particles_lethe(load=True)
        assert isinstance(dataset, pyvista.UnstructuredGrid)

    def test_gif_simple():
        filename = examples.download_gif_simple(load=False)
        assert os.path.isfile(filename)
        assert filename.endswith('gif')

        dataset = examples.download_gif_simple(load=True)
        assert isinstance(dataset, pyvista.UniformGrid)
        assert 'frame0' in dataset.point_data
