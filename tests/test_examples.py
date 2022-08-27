"""Test examples that do not require downloading."""
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
