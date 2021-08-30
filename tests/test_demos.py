import os
import platform
import pytest

from pyvista import demos
from pyvista.plotting import system_supports_plotting

skip_no_plotting = pytest.mark.skipif(not system_supports_plotting(),
                                      reason="Test requires system to support plotting")

# this will have to be modified once VTK finalizes how they release
# dev wheels
try:
    from vtkmodules.vtkCommonCore import vtkVersion
    vtk_dev = len(str(vtkVersion().GetVTKBuildVersion())) > 2
except:
    vtk_dev = False


# These tests fail with mesa opengl on windows
skip_windows_dev_whl = pytest.mark.skipif(os.name == 'nt' and vtk_dev,
                                          reason='Test fails on Windows with VTK dev wheels')

@skip_no_plotting
def test_plot_glyphs():
    demos.plot_glyphs(2)


def test_atomized():
    grid = demos.logo_atomized(density=0.2, scale=0.6)
    assert grid.n_cells


def test_logo_basic():
    pd = demos.logo_basic()
    assert pd.n_faces


def test_logo_voxel():
    grid = demos.logo_voxel()
    assert grid.n_cells


@pytest.mark.skipif(platform.system() == 'Darwin',
                    reason='MacOS testing on Azure fails when downloading')
@skip_no_plotting
@skip_windows_dev_whl
def test_plot_logo():
    # simply should not fail
    demos.plot_logo()
