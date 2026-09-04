"""Pytest plugin failing a test when VTK logs an error or warning.

Registered with ``-p tests.vtk_noise_guard`` so it also applies to runs whose
collection root is outside this repository, where ``conftest.py`` is not loaded.
"""

from __future__ import annotations

import pytest

import pyvista as pv


@pytest.fixture(autouse=True)
def _fail_on_vtk_output():
    """Fail the test when VTK logs an error or warning while it runs."""
    with pv.VtkErrorCatcher(send_to_logging=False) as catcher:
        yield
    if events := catcher.events:
        logged = '\n'.join(str(event) for event in events)
        msg = f'VTK logged {len(events)} error(s) or warning(s):\n{logged}'
        pytest.fail(msg)
