"""Session-scoped setup for jupyter backend / trame integration tests.

The repo-wide ``reset_global_state`` autouse fixture in
``tests/conftest.py`` snapshots the plotter component registry at the
start of every test and restores it afterwards. Components that are
imported lazily (via the ``pyvista.plotter_components`` entry-point
group) attach themselves through a decorator side-effect on first
import; once a test has triggered that import, restoring the snapshot
removes the descriptor from ``BasePlotter`` and the import-module cache
prevents the decorator from firing again.

To keep the ``trame`` plotter component (provided by ``trame-pyvista``)
attached throughout the test session, resolve it once at session start
so the snapshot taken by ``reset_global_state`` already includes it.
"""

from __future__ import annotations

import importlib.util

import pytest


@pytest.fixture(scope='session', autouse=True)
def _eager_resolve_trame_component():
    """Resolve the trame plotter component before any per-test snapshots.

    Also normalize the jupyter backend registry: importing
    ``trame_pyvista`` registers ``trame``/``server``/``client``/``html``
    via :func:`register_jupyter_backend` (an "explicit" registration
    with no ``:`` in its source). Subsequent jupyter-backend tests
    expect the canonical entry-point-discovered shape (``:`` in source),
    so clear and reload the registry once now.
    """
    if importlib.util.find_spec('trame_pyvista') is None:
        return
    import pyvista as pv
    from pyvista import jupyter as jupyter_mod

    pv.OFF_SCREEN = True
    pl = pv.Plotter()
    try:
        _ = pl.trame  # triggers entry-point import + descriptor install
    finally:
        pl.close()

    jupyter_mod._custom_backends.clear()
    jupyter_mod._custom_backend_sources.clear()
    jupyter_mod._entry_points_loaded = False
    jupyter_mod._ensure_entry_points()
