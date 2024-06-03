"""Test examples that do not require downloading."""

from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
from tests.examples.test_dataset_loader import DatasetLoaderTestCase
from tests.examples.test_dataset_loader import _generate_dataset_loader_test_cases_from_module
from tests.examples.test_dataset_loader import _get_mismatch_fail_msg


def pytest_generate_tests(metafunc):
    """Generate parametrized tests."""
    if 'test_case' in metafunc.fixturenames:
        # Generate a separate test case for each loadable dataset
        test_cases = _generate_dataset_loader_test_cases_from_module(pv.examples.examples)
        ids = [case.dataset_name for case in test_cases]
        metafunc.parametrize('test_case', test_cases, ids=ids)


def test_dataset_loader_name_matches_function_name(test_case: DatasetLoaderTestCase):
    if (msg := _get_mismatch_fail_msg(test_case)) is not None:
        pytest.fail(msg)


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


def test_load_random_hills(random_hills):
    assert random_hills.n_cells


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


def test_load_hydrogen_orbital():
    with pytest.raises(ValueError, match='`n` must be'):
        pv.examples.load_hydrogen_orbital(-1, 1, 0)
    with pytest.raises(ValueError, match='`l` must be'):
        pv.examples.load_hydrogen_orbital(1, 1, 0)
    with pytest.raises(ValueError, match='`m` must be'):
        pv.examples.load_hydrogen_orbital(1, 0, 1)

    orbital = pv.examples.load_hydrogen_orbital(3, 2, 1)
    assert isinstance(orbital, pv.ImageData)
    assert 'wf' in orbital.point_data
    assert orbital.point_data['wf'].dtype == np.complex128
    assert 'real_wf' in orbital.point_data
    assert orbital.point_data['real_wf'].dtype == np.float64


def test_load_logo():
    mesh = examples.load_logo()
    assert mesh.n_points


def test_load_frog_tissue():
    data = examples.load_frog_tissues()
    assert data.n_points
    assert data.get_data_range() == (0, 29)
