import os
import sys

developer_note = """
vtk has been directly imported in vtk>=9
Please see:
https://github.com/pyvista/pyvista/pull/1163
"""


def test_vtk_not_loaded():
    """This test verifies that the vtk module isn't loaded when using vtk>=9

    We use ``os.system`` because we need to test the import of pyvista
    outside of the pytest unit test framework as pytest loads vtk.

    """
    exe_str = "import pyvista; import sys; assert 'vtk' not in sys.modules"

    # anything other than 0 indicates an error
    assert not os.system(f'{sys.executable} -c "{exe_str}"'), developer_note
