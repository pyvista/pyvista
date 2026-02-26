from __future__ import annotations

import pytest

import pyvista as pv
from pyvista.core.utilities.frd import FRDReader


@pytest.fixture
def mock_frd_file(tmp_path):
    """Fixture to create a temporary FRD file with basic ASCII structure."""
    content = """2C
 -1 1 0.0 0.0 0.0
 -1 2 1.0 0.0 0.0
 -1 3 1.0 1.0 0.0
 -1 4 0.0 1.0 0.0
 -1 5 0.0 0.0 1.0
 -1 6 1.0 0.0 1.0
 -1 7 1.0 1.0 1.0
 -1 8 0.0 1.0 1.0
 -3
 3C
 -1 1 1
 -2 1 2 3 4 5 6 7 8
 -3
 100CL 1 0.1
 -4 STRESS 6
 -1 1 10.0 20.0 30.0 0.0 0.0 0.0
 -1 2 10.0 20.0 30.0 0.0 0.0 0.0
 -1 3 10.0 20.0 30.0 0.0 0.0 0.0
 -1 4 10.0 20.0 30.0 0.0 0.0 0.0
 -1 5 10.0 20.0 30.0 0.0 0.0 0.0
 -1 6 10.0 20.0 30.0 0.0 0.0 0.0
 -1 7 10.0 20.0 30.0 0.0 0.0 0.0
 -1 8 10.0 20.0 30.0 0.0 0.0 0.0
 -3
 100CL 2 0.2
 -4 STRAIN 6
 -1 1 0.1 0.2 0.3 0.0 0.0 0.0
 -1 2 0.1 0.2 0.3 0.0 0.0 0.0
 -1 3 0.1 0.2 0.3 0.0 0.0 0.0
 -1 4 0.1 0.2 0.3 0.0 0.0 0.0
 -1 5 0.1 0.2 0.3 0.0 0.0 0.0
 -1 6 0.1 0.2 0.3 0.0 0.0 0.0
 -1 7 0.1 0.2 0.3 0.0 0.0 0.0
 -1 8 0.1 0.2 0.3 0.0 0.0 0.0
 -3
 100CL 3 0.3
 -4 DISP 3
 -1 1 1.0 2.0 3.0
 -1 2 1.0 2.0 3.0
 -1 3 1.0 2.0 3.0
 -1 4 1.0 2.0 3.0
 -1 5 1.0 2.0 3.0
 -1 6 1.0 2.0 3.0
 -1 7 1.0 2.0 3.0
 -1 8 1.0 2.0 3.0
 -3
"""
    file_path = tmp_path / 'test_model.frd'
    file_path.write_text(content, encoding='utf-8')
    return str(file_path)


@pytest.fixture
def comprehensive_frd_file(tmp_path):
    """Fixture generating a mock FRD file covering all parsing branches and errors."""
    content = """1C file header
2C
 -1 1 0.0 0.0 0.0
 -1 2 1.0 0.0 0.0
 -1 3 0.0 1.0 0.0
 -1 4 1.0 1.0 0.0
 -1 bad_node_id 0.0 0.0 0.0
 -3
3C
 -1 1 7
 -2 1 3 2
 -1 2 9
 -2 1 3 4 2
 -1 3 bad_type
 -1 4 999
 -1 5 1
 -2 1 2 3 4 5 6 7 8
 -3
100CL 1 0.1
 -4 STRESS 6
 -5 skip component info
 -1 bad_first_line 1.0
 -1 1 1.0 2.0 3.0 4.0 5.0 6.0
 -1 2 bad_float 3.0 4.0 5.0 6.0
 -3
100CL 2 0.2
 -4 EMPTY_BLOCK 1
 -5 some info
 -3
"""
    file_path = tmp_path / 'comprehensive.frd'
    file_path.write_text(content, encoding='utf-8')
    return str(file_path)


@pytest.fixture
def no_steps_frd_file(tmp_path):
    """Fixture to trigger empty step checks."""
    content = '1C\n2C\n -1 1 0.0 0.0 0.0\n -3\n'
    file_path = tmp_path / 'nosteps.frd'
    file_path.write_text(content, encoding='utf-8')
    return str(file_path)


@pytest.fixture
def empty_frd_file(tmp_path):
    """Fixture to trigger empty file checks."""
    content = '1C Empty File\n'
    file_path = tmp_path / 'empty.frd'
    file_path.write_text(content, encoding='utf-8')
    return str(file_path)


def test_frd_reader_init_and_read(mock_frd_file):
    reader = FRDReader(mock_frd_file)
    mesh = reader.read()

    assert mesh.n_points == 8
    assert mesh.n_cells == 1
    assert 'Original_Node_ID' in mesh.point_data


def test_frd_reader_time_steps(mock_frd_file):
    reader = FRDReader(mock_frd_file)

    assert reader.number_time_points == 3
    assert reader.time_values == [0.1, 0.2, 0.3]
    assert reader.time_point_value(0) == 0.1
    assert reader.active_time_value == 0.1

    reader.set_active_time_point(2)
    assert reader.active_time_value == 0.3

    mesh = reader.read()
    assert 'DISP' in mesh.point_data
    assert mesh.point_data['DISP'].shape == (8, 3)


def test_frd_reader_set_active_time_value(mock_frd_file):
    reader = FRDReader(mock_frd_file)

    reader.set_active_time_value(0.18)
    assert reader.active_time_value == 0.2

    reader.active_time_value = 0.29
    assert reader.active_time_value == 0.3


def test_frd_reader_derived_stress(mock_frd_file):
    reader = FRDReader(mock_frd_file)
    reader.set_active_time_point(0)
    mesh = reader.read()

    assert 'STRESS' in mesh.point_data
    assert 'STRESS_vMises' in mesh.point_data
    assert 'STRESS_sgMises' in mesh.point_data
    assert 'STRESS_PS1' in mesh.point_data
    assert 'STRESS_PS2' in mesh.point_data
    assert 'STRESS_PS3' in mesh.point_data


def test_frd_reader_derived_strain(mock_frd_file):
    reader = FRDReader(mock_frd_file)
    reader.set_active_time_point(1)
    mesh = reader.read()

    assert 'STRAIN' in mesh.point_data
    assert 'STRAIN_vMises' in mesh.point_data
    assert 'STRAIN_sgMises' in mesh.point_data
    assert 'STRAIN_PS1' in mesh.point_data


def test_frd_reader_comprehensive(comprehensive_frd_file):
    # This directly hits the logic in pyvista/core/utilities/fileio.py
    mesh_from_pv = pv.read(comprehensive_frd_file)
    assert isinstance(mesh_from_pv, pv.UnstructuredGrid)

    reader = FRDReader(comprehensive_frd_file)
    mesh = reader.read()

    assert mesh.n_points == 4
    assert reader.number_time_points == 2

    # Check if derived stress fields were computed correctly
    assert 'STRESS_vMises' in mesh.point_data
    assert 'STRESS_sgMises' in mesh.point_data


def test_no_time_steps(no_steps_frd_file):
    reader = FRDReader(no_steps_frd_file)

    # This triggers the 'if not steps: return 0.0' lines
    assert reader.active_time_value == 0.0

    with pytest.raises(RuntimeError, match='No time steps found'):
        reader.set_active_time_value(0.5)


def test_frd_reader_errors(mock_frd_file, empty_frd_file):
    reader = FRDReader(mock_frd_file)

    with pytest.raises(IndexError, match='out of range'):
        reader.set_active_time_point(99)

    with pytest.raises(IndexError, match='out of range'):
        reader.set_active_time_point(-1)

    empty_reader = FRDReader(empty_frd_file)
    with pytest.raises(ValueError, match='No nodes found'):
        empty_reader.read()
