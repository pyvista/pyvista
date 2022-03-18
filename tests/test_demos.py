import os
import platform

import numpy as np
import pytest

import pyvista
from pyvista import demos
from pyvista.plotting import system_supports_plotting

skip_no_plotting = pytest.mark.skipif(
    not system_supports_plotting(), reason="Test requires system to support plotting"
)


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


@pytest.mark.skipif(
    platform.system() == 'Darwin', reason='MacOS testing on Azure fails when downloading'
)
@skip_no_plotting
@pytest.mark.skipif(os.name == 'nt', reason='Test fails on Windows')
def test_plot_logo():
    # simply should not fail
    demos.plot_logo()


@skip_no_plotting
def test_plot_datasets():
    # simply should not fail
    demos.plot_datasets()


@skip_no_plotting
def test_plot_wave():
    points = demos.plot_wave(wavetime=0.1)
    assert isinstance(points, np.ndarray)


@skip_no_plotting
def test_beam_example():
    demos.plot_beam()


@skip_no_plotting
def test_plot_ants_plane():
    demos.plot_ants_plane()


@skip_no_plotting
def test_orientation_cube():
    pl = demos.orientation_plotter()
    assert isinstance(pl, pyvista.Plotter)
    pl.show()
