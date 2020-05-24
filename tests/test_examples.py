import os

import numpy as np
import pytest
import vtk

import pyvista
from pyvista import examples
from pyvista.plotting import system_supports_plotting

ffmpeg_failed = False
try:
    try:
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        import imageio
        imageio.plugins.ffmpeg.download()
except:
    ffmpeg_failed = True

TEST_DOWNLOADS = False
try:
    if os.environ['TEST_DOWNLOADS'].lower() == 'true':
        TEST_DOWNLOADS = True
except KeyError:
    pass


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
def test_plot_wave():
    points = examples.plot_wave(wavetime=0.1, off_screen=True)
    assert isinstance(points, np.ndarray)


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
def test_beam_example():
    examples.beam_example(off_screen=True)


@pytest.mark.skipif(not system_supports_plotting(), reason="Requires system to support plotting")
def test_plot_ants_plane():
    examples.plot_ants_plane(off_screen=True)


def test_load_ant():
    """ Load ply ant mesh """
    mesh = examples.load_ant()
    assert mesh.n_points


def test_load_airplane():
    """ Load ply airplane mesh """
    mesh = examples.load_airplane()
    assert mesh.n_points


def test_load_sphere():
    """ Loads sphere ply mesh """
    mesh = examples.load_sphere()
    assert mesh.n_points


def test_load_channels():
    """ Loads geostat training image """
    mesh = examples.load_channels()
    assert mesh.n_points


def test_load_spline():
    mesh = examples.load_spline()
    assert mesh.n_points


def test_load_random_hills():
    mesh = examples.load_random_hills()
    assert mesh.n_cells


if TEST_DOWNLOADS:
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


    def test_download_iron_pot():
        data = examples.download_iron_pot()
        assert data.n_cells


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

# End of download tests
