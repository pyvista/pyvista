"""Test examples that do not require downloading."""
import numpy as np
import pytest

import pyvista as pv
from pyvista import examples


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


def test_load_tetbeam():
    mesh = examples.load_tetbeam()
    assert mesh.n_cells
    assert (mesh.celltypes == 10).all()


def test_sphere_with_texture_map():
    sphere = pv.examples.planets._sphere_with_texture_map()
    assert isinstance(sphere, pv.PolyData)
    assert 'Texture Coordinates' in sphere.point_data
    assert sphere['Texture Coordinates'].shape == (sphere.n_points, 2)


def test_load_earth():
    mesh = pv.examples.planets.load_earth()
    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_cells
    assert mesh.textures["surface"]


def test_load_hydrogen_orbital():
    with pytest.raises(ValueError, match='`n` must be'):
        pv.examples.load_hydrogen_orbital(-1, 1, 0)
    with pytest.raises(ValueError, match='`l` must be'):
        pv.examples.load_hydrogen_orbital(1, 1, 0)
    with pytest.raises(ValueError, match='`m` must be'):
        pv.examples.load_hydrogen_orbital(1, 0, 1)

    orbital = pv.examples.load_hydrogen_orbital(3, 2, 1)
    assert isinstance(orbital, pv.UniformGrid)
    assert 'wf' in orbital.point_data
    assert orbital.point_data['wf'].dtype == np.complex128
    assert 'real_wf' in orbital.point_data
    assert orbital.point_data['real_wf'].dtype == np.float64
