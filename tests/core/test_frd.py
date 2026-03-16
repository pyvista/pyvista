from __future__ import annotations

import re

import numpy as np
import pytest

import pyvista as pv
from pyvista.core.utilities._frd import CCX_TO_VTK_TYPE
from pyvista.core.utilities._frd import FRDElementType


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
 -2 99 99 99 99 99 99 99 99
 -1 6 5
 -2 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
 -1 7 2
 -2 1 2 3
 -1 8 7
 -2 1 2 3 4 5
 -3
100CL BAD_STEP BAD_TIME
 -4 STRESS 6
 -5 skip component info
 -1 bad_first_line 1.0
 -1 1 1.0 2.0 3.0 4.0 5.0 6.0
 -1 2 bad_float 3.0 4.0 5.0 6.0
 -1 99 1.0 2.0 3.0 4.0 5.0 6.0
 -3
100CL 2 0.2
 -4 EMPTY_BLOCK 1
 -5 some info
 -3
100CL 3 0.3
 -4 SCALAR 1
 -1 1 10.0
 -1 2 20.0
 -1 3 30.0
 -1 4 40.0
 -1 99 99.0
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
    # Test reading via the common pyvista endpoint
    mesh_from_pv = pv.read(mock_frd_file)
    assert mesh_from_pv.n_points == 8
    assert mesh_from_pv.n_cells == 1
    assert 'original_node_ids' in mesh_from_pv.point_data

    # Test reading directly with the reader class
    reader = pv.FRDReader(mock_frd_file)
    mesh = reader.read()

    assert mesh.n_points == 8
    assert mesh.n_cells == 1
    assert 'original_node_ids' in mesh.point_data


def test_frd_reader_time_steps(mock_frd_file):
    reader = pv.FRDReader(mock_frd_file)

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
    reader = pv.FRDReader(mock_frd_file)

    # Exact time match is required.
    # The mock_frd_file has time steps: 0.1, 0.2, 0.3.

    # 1. Check setting an invalid time (e.g., 0.18 instead of 0.2)
    with pytest.raises(ValueError, match='Not a valid time'):
        reader.set_active_time_value(0.18)

    # 2. Check setting a valid time using the method
    reader.set_active_time_value(0.2)
    assert reader.active_time_value == 0.2

    # 3. Check setting an invalid time via the setter property
    with pytest.raises(ValueError, match='Not a valid time'):
        reader.active_time_value = 0.29

    # 4. Check setting a valid time via the setter property
    reader.active_time_value = 0.3
    assert reader.active_time_value == 0.3


def test_frd_reader_derived_stress(mock_frd_file):
    reader = pv.FRDReader(mock_frd_file)
    reader.set_active_time_point(0)
    mesh = reader.read()

    assert 'STRESS' in mesh.point_data

    # Check calculated stress derived fields for mock_frd_file
    # values: xx=10, yy=20, zz=30, xy=yz=zx=0
    # vMises for this state is ~17.3205
    expected_vmises = np.sqrt(300.0)

    np.testing.assert_allclose(mesh.point_data['STRESS_Mises'], expected_vmises)
    np.testing.assert_allclose(mesh.point_data['STRESS_sgMises'], expected_vmises)

    # Principal stresses should match diagonal entries since shear is 0
    np.testing.assert_allclose(mesh.point_data['STRESS_PS3'], 10.0)
    np.testing.assert_allclose(mesh.point_data['STRESS_PS2'], 20.0)
    np.testing.assert_allclose(mesh.point_data['STRESS_PS1'], 30.0)


def test_frd_reader_derived_strain(mock_frd_file):
    reader = pv.FRDReader(mock_frd_file)
    reader.set_active_time_point(1)
    mesh = reader.read()

    assert 'STRAIN' in mesh.point_data

    # Check calculated strain derived fields for mock_frd_file
    # values: xx=0.1, yy=0.2, zz=0.3, xy=yz=zx=0
    # Mises strain for this state is sqrt(3)/15 = ~0.11547
    expected_vmises = np.sqrt(3.0) / 15.0

    np.testing.assert_allclose(mesh.point_data['STRAIN_Mises'], expected_vmises)
    np.testing.assert_allclose(mesh.point_data['STRAIN_sgMises'], expected_vmises)

    np.testing.assert_allclose(mesh.point_data['STRAIN_PS3'], 0.1)
    np.testing.assert_allclose(mesh.point_data['STRAIN_PS2'], 0.2)
    np.testing.assert_allclose(mesh.point_data['STRAIN_PS1'], 0.3)


def test_frd_reader_comprehensive(comprehensive_frd_file):
    match1 = (
        '1 cell with too many points detected:\n'
        '  line 22, element type 7 (TR3), num nodes 5 (expected 3)'
    )
    match2 = (
        '1 cell with too few points detected. These elements are skipped:\n'
        '  line 20, element type 2 (PE6), num nodes 3 (expected 6)'
    )
    match3 = (
        '1 cell with unknown element type encountered. These elements are skipped.:\n'
        '  line 15, element type 999'
    )
    with pytest.warns(pv.InvalidMeshWarning, match=re.escape(match1)):  # noqa: PT031
        with pytest.warns(pv.InvalidMeshWarning, match=re.escape(match2)):
            with pytest.warns(pv.InvalidMeshWarning, match=re.escape(match3)):
                mesh_from_pv = pv.read(comprehensive_frd_file)
    assert isinstance(mesh_from_pv, pv.UnstructuredGrid)

    with pytest.warns(pv.InvalidMeshWarning):
        reader = pv.FRDReader(comprehensive_frd_file)

    # Inject empty dictionary to trigger the defensive "if not data: continue" branch
    reader.reader._frd_data.results_by_step[0.2]['FAKE_EMPTY'] = {}

    mesh = reader.read()

    assert mesh.n_points == 4
    assert reader.number_time_points == 3

    # Check if derived stress fields were computed correctly
    assert 'STRESS_Mises' in mesh.point_data
    assert 'STRESS_sgMises' in mesh.point_data

    # Check if scalar values (n_components == 1) were correctly added
    reader.set_active_time_point(2)
    mesh_scalar = reader.read()
    assert 'SCALAR' in mesh_scalar.point_data


def test_no_time_steps(no_steps_frd_file):
    reader = pv.FRDReader(no_steps_frd_file)

    # This triggers the 'if not steps: return 0.0' lines
    assert reader.active_time_value == 0.0

    with pytest.raises(RuntimeError, match='No time steps found'):
        reader.set_active_time_value(0.5)


def test_frd_reader_errors(mock_frd_file, empty_frd_file):
    reader = pv.FRDReader(mock_frd_file)

    with pytest.raises(IndexError, match='out of range'):
        reader.set_active_time_point(99)

    with pytest.raises(IndexError, match='out of range'):
        reader.set_active_time_point(-1)

    empty_reader = pv.FRDReader(empty_frd_file)
    with pytest.raises(ValueError, match='No nodes found'):
        empty_reader.read()


# =============================================================================
# Element node ordering validation (Volume / Area checks)
# =============================================================================

# A dictionary mapping FRD element types to a valid 3D nodal coordinate list.
# These nodes are carefully ordered according to the CalculiX manual so that,
# when parsed and converted to VTK cells, the resulting volume or area is strictly positive.
VALID_ELEMENT_DEFINITIONS = {
    # 3D Elements (Expect volume > 0)
    'HE8': (
        1,
        '1 0.0 0.0 0.0\n -1 2 1.0 0.0 0.0\n -1 3 1.0 1.0 0.0\n -1 4 0.0 1.0 0.0\n'
        ' -1 5 0.0 0.0 1.0\n -1 6 1.0 0.0 1.0\n -1 7 1.0 1.0 1.0\n -1 8 0.0 1.0 1.0',
        '1 2 3 4 5 6 7 8',
    ),
    'PE6': (
        2,
        '1 0.0 0.0 0.0\n -1 2 1.0 0.0 0.0\n -1 3 0.0 1.0 0.0\n -1 4 0.0 0.0 1.0\n'
        ' -1 5 1.0 0.0 1.0\n -1 6 0.0 1.0 1.0',
        '1 2 3 4 5 6',
    ),
    'TE4': (
        3,
        '1 0.0 0.0 0.0\n -1 2 1.0 0.0 0.0\n -1 3 0.0 1.0 0.0\n -1 4 0.0 0.0 1.0',
        '1 2 3 4',
    ),
    'HE20': (
        4,
        '1 0.0 0.0 0.0\n -1 2 1.0 0.0 0.0\n -1 3 1.0 1.0 0.0\n -1 4 0.0 1.0 0.0\n'
        ' -1 5 0.0 0.0 1.0\n -1 6 1.0 0.0 1.0\n -1 7 1.0 1.0 1.0\n -1 8 0.0 1.0 1.0\n'
        ' -1 9 0.5 0.0 0.0\n -1 10 1.0 0.5 0.0\n -1 11 0.5 1.0 0.0\n -1 12 0.0 0.5 0.0\n'
        ' -1 13 0.5 0.0 1.0\n -1 14 1.0 0.5 1.0\n -1 15 0.5 1.0 1.0\n -1 16 0.0 0.5 1.0\n'
        ' -1 17 0.0 0.0 0.5\n -1 18 1.0 0.0 0.5\n -1 19 1.0 1.0 0.5\n -1 20 0.0 1.0 0.5',
        '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20',
    ),
    'PE15': (
        5,
        '1 0.0 0.0 0.0\n -1 2 1.0 0.0 0.0\n -1 3 0.0 1.0 0.0\n -1 4 0.0 0.0 1.0\n'
        ' -1 5 1.0 0.0 1.0\n -1 6 0.0 1.0 1.0\n -1 7 0.5 0.0 0.0\n -1 8 0.5 0.5 0.0\n'
        ' -1 9 0.0 0.5 0.0\n -1 10 0.0 0.0 0.5\n -1 11 1.0 0.0 0.5\n -1 12 0.0 1.0 0.5\n'
        ' -1 13 0.5 0.0 1.0\n -1 14 0.5 0.5 1.0\n -1 15 0.0 0.5 1.0',
        '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15',
    ),
    'TE10': (
        6,
        '1 0.0 0.0 0.0\n -1 2 1.0 0.0 0.0\n -1 3 0.0 1.0 0.0\n -1 4 0.0 0.0 1.0\n'
        ' -1 5 0.5 0.0 0.0\n -1 6 0.5 0.5 0.0\n -1 7 0.0 0.5 0.0\n -1 8 0.0 0.0 0.5\n'
        ' -1 9 0.5 0.0 0.5\n -1 10 0.0 0.5 0.5',
        '1 2 3 4 5 6 7 8 9 10',
    ),
    # 2D Elements (Expect area > 0)
    'TR3': (7, '1 0.0 0.0 0.0\n -1 2 1.0 0.0 0.0\n -1 3 0.0 1.0 0.0', '1 2 3'),
    'TR6': (
        8,
        '1 0.0 0.0 0.0\n -1 2 1.0 0.0 0.0\n -1 3 0.0 1.0 0.0\n -1 4 0.5 0.0 0.0\n'
        ' -1 5 0.5 0.5 0.0\n -1 6 0.0 0.5 0.0',
        '1 2 3 4 5 6',
    ),
    'QU4': (
        9,
        '1 0.0 0.0 0.0\n -1 2 1.0 0.0 0.0\n -1 3 1.0 1.0 0.0\n -1 4 0.0 1.0 0.0',
        '1 2 3 4',
    ),
    'QU8': (
        10,
        '1 0.0 0.0 0.0\n -1 2 1.0 0.0 0.0\n -1 3 1.0 1.0 0.0\n -1 4 0.0 1.0 0.0\n'
        ' -1 5 0.5 0.0 0.0\n -1 6 1.0 0.5 0.0\n -1 7 0.5 1.0 0.0\n -1 8 0.0 0.5 0.0',
        '1 2 3 4 5 6 7 8',
    ),
    # 1D Elements (Expect length > 0)
    'BE2': (11, '1 0.0 0.0 0.0\n -1 2 1.0 0.0 0.0', '1 2'),
    'BE3': (12, '1 0.0 0.0 0.0\n -1 2 1.0 0.0 0.0\n -1 3 0.5 0.0 0.0', '1 2 3'),
}


@pytest.fixture
def generic_element_frd(tmp_path, request):
    """Generates an FRD file on the fly for a specific element type."""
    elem_name = request.param
    etype_id, nodes, connectivity = VALID_ELEMENT_DEFINITIONS[elem_name]

    content = f"""1C Element test
2C
 -1 {nodes}
 -3
3C
 -1 1 {etype_id}
 -2 {connectivity}
 -3
"""
    file_path = tmp_path / f'{elem_name}_test.frd'
    file_path.write_text(content, encoding='utf-8')
    return str(file_path), elem_name


@pytest.mark.parametrize(
    'generic_element_frd', list(VALID_ELEMENT_DEFINITIONS.keys()), indirect=True
)
def test_frd_element_sizes(generic_element_frd):
    """Condense 1D/2D/3D element size testing into a single dynamic check."""
    filepath, elem_name = generic_element_frd
    mesh = pv.FRDReader(filepath).read()

    # 1. Get the FRD type from the enum
    frd_enum = FRDElementType[elem_name]

    # 2. Map to the VTK cell type
    vtk_type = CCX_TO_VTK_TYPE[frd_enum]

    # 3. Compute cell sizes
    sizes = mesh.compute_cell_sizes().cell_data

    # 4. Check the dimension dynamically and make the appropriate assertion
    if vtk_type.dimension == 1:
        val = sizes['Length'][0]
        assert val > 0.0, (
            f'Element {elem_name} generated non-positive length ({val}). Bad node ordering!'
        )
    elif vtk_type.dimension == 2:
        val = sizes['Area'][0]
        assert val > 0.0, (
            f'Element {elem_name} generated non-positive area ({val}). Bad node ordering!'
        )
    elif vtk_type.dimension == 3:
        val = sizes['Volume'][0]
        assert val > 0.0, (
            f'Element {elem_name} generated non-positive volume ({val}). Bad node ordering!'
        )
    else:
        pytest.fail(f'Unhandled cell dimension for element {elem_name} with VTK type {vtk_type}.')
