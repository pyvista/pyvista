"""Close all plotters to help control memory usage for our doctests."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib as mpl
import pytest

import pyvista as pv
from pyvista import _vtk

if TYPE_CHECKING:
    from collections.abc import Generator

# Need to import all vtk modules eagerly to avoid issues with parallel lazy imports
_vtk.import_all()

collect_ignore = [  # Avoid importing deprecated modules
    'examples/download_3ds.py',
    'examples/gltf.py',
    'examples/vrml.py',
]


@pytest.fixture(autouse=True)
def fail_on_vtk_output() -> Generator[None, None, None]:
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
def matplotlib_headless() -> None:
    """Use a non-interactive Matplotlib backend to avoid Tk issues on Windows CI."""
    if 'CI' in os.environ:
        mpl.use('Agg')


@pytest.fixture(autouse=True)
def autoclose_plotters() -> Generator[None, None, None]:
    """Close all plotters."""
    yield
    pv.close_all()


@pytest.fixture(autouse=True)
def reset_global_theme() -> Generator[None, None, None]:
    """Reset ``global_theme``."""
    # this stops any doctest-module tests from overriding the global theme and
    # creating test side effects
    pv.set_plot_theme('document_build')
    yield
    pv.set_plot_theme('document_build')
