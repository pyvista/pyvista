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


def _require_reason(marker: pytest.Mark) -> None:
    """Fail unless the marker says why the test provokes VTK.

    Parameters
    ----------
    marker : pytest.Mark
        Marker to check for a ``reason`` keyword.

    """
    if not marker.kwargs.get('reason'):
        msg = f'@pytest.mark.{marker.name} needs reason=... saying why VTK logs here'
        pytest.fail(msg)


@pytest.fixture(autouse=True)
def fail_on_vtk_output(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Fail the test when VTK logs an error or warning while it runs.

    - ``expect_vtk_output(*messages, reason=...)`` allows the errors and warnings whose
      text it names, matched as substrings. Anything else VTK logs still fails the test.
    - ``skip_vtk_output_check(reason=...)`` turns the check off for that test.

    Both markers require a ``reason``, so a test never silences VTK without saying why.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Fixture request for the test, used to look up its markers.

    """
    if marker := request.node.get_closest_marker('skip_vtk_output_check'):
        _require_reason(marker)
        yield
        return
    markers = list(request.node.iter_markers('expect_vtk_output'))
    for marker in markers:
        _require_reason(marker)
    expected = [pattern for marker in markers for pattern in marker.args]
    with pv.VtkErrorCatcher(send_to_logging=False) as catcher:
        yield
    events = catcher.events
    # The traceback of a failure raised here keeps this frame alive, and with it the
    # catcher's own output window, which the leak check would then report instead.
    del catcher
    if unexpected := [
        event for event in events if not any(text in event.alert for text in expected)
    ]:
        logged = '\n'.join(str(event) for event in unexpected)
        msg = f'VTK logged {len(unexpected)} error(s) or warning(s):\n{logged}'
        reasons = '; '.join(marker.kwargs['reason'] for marker in markers)
        msg += f'\n\nThis test expects VTK output because {reasons}'
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
