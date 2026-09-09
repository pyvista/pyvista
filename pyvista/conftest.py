"""Close all plotters to help control memory usage for our doctests."""

from __future__ import annotations

import os

import matplotlib as mpl
import pytest

import pyvista as pv
from pyvista import _vtk

# Need to import all vtk modules eagerly to avoid issues with parallel lazy imports
_vtk.import_all()

collect_ignore = [  # Avoid importing deprecated modules
    'examples/download_3ds.py',
    'examples/gltf.py',
    'examples/vrml.py',
]


@pytest.fixture(autouse=True)
def fail_on_vtk_output():
    """Fail the test when VTK logs an error or warning while it runs.

    Defined here rather than in ``tests`` so that it also applies to the doctests run
    from the installed package, which collect no ``conftest.py`` from the repository.
    """
    with pv.VtkErrorCatcher(send_to_logging=False) as catcher:
        yield
    if events := catcher.events:
        logged = '\n'.join(str(event) for event in events)
        msg = f'VTK logged {len(events)} error(s) or warning(s):\n{logged}'
        pytest.fail(msg)


@pytest.fixture(autouse=True, scope='session')
def matplotlib_headless():
    """Use a non-interactive Matplotlib backend to avoid Tk issues on Windows CI."""
    if 'CI' in os.environ:
        mpl.use('Agg')


@pytest.fixture(autouse=True)
def autoclose_plotters():
    """Close all plotters."""
    yield
    pv.close_all()


@pytest.fixture(autouse=True)
def reset_global_theme():
    """Reset ``global_theme``."""
    # this stops any doctest-module tests from overriding the global theme and
    # creating test side effects
    pv.set_plot_theme('document_build')
    yield
    pv.set_plot_theme('document_build')
