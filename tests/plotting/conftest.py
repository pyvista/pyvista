"""This conftest is here to allow for checking garbage collection and
memory leaks for all plotting tests
"""

from __future__ import annotations

import gc
import platform
from types import SimpleNamespace

import pytest
from refleak.testing import Snapshot
from refleak.testing import gc_collect_once

import pyvista as pv
from pyvista import _vtk
from pyvista.plotting import system_supports_plotting

# these are set here because we only need them for plotting tests
pv.OFF_SCREEN = True
SKIP_PLOTTING = not system_supports_plotting()
APPLE_SILICON = platform.system() == 'Darwin' and platform.machine() == 'arm64'


# Configure skip_plotting marker
def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'skip_plotting: skip the test if system does not support plotting',
    )


def pytest_runtest_setup(item):
    skip = any(mark.name == 'skip_plotting' for mark in item.iter_markers())
    if skip and SKIP_PLOTTING:
        pytest.skip('Test requires system to support plotting')


@pytest.fixture(autouse=True)
def _clean_trame_env(monkeypatch):
    # Isolate trame/jupyter-hub env vars so tests don't inherit developer
    # machine state (e.g. PYVISTA_TRAME_SERVER_PROXY_PREFIX set by a tailnet
    # proxy). Tests that need these set should call monkeypatch.setenv.
    for var in (
        'PYVISTA_TRAME_SERVER_PROXY_PREFIX',
        'JUPYTERHUB_SERVICE_PREFIX',
        'TRAME_JUPYTER_WWW',
        'PYVISTA_TRAME_JUPYTER_MODE',
    ):
        monkeypatch.delenv(var, raising=False)


if APPLE_SILICON:

    @pytest.fixture(autouse=True)
    def macos_memory_leak(request):  # noqa: ARG001
        # Without this, only 500 render windows can be created in a single Python
        # process on MacOS using Apple silicon
        # See https://gitlab.kitware.com/vtk/vtk/-/issues/18713
        from Foundation import NSAutoreleasePool  # for macOS

        pool = NSAutoreleasePool.alloc().init()
        yield

        # pool goes out of scope and resources get collected
        del pool


_phase_report_key = pytest.StashKey()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):  # noqa: ARG001
    """Stash per-phase reports so check_gc can skip the leak check on failure."""
    outcome = yield
    rep = outcome.get_result()
    item.stash.setdefault(_phase_report_key, {})[rep.when] = rep


def _test_passed(item) -> bool:
    report = item.stash.get(_phase_report_key, {})
    return 'call' in report and report['call'].outcome == 'passed'


def _flush_vtk_ghosts() -> None:
    """Sweep dead entries out of VTK's ghost map.

    When a wrapper with attributes dies while its C++ object is still
    referenced, VTK "ghosts" the attribute dict so it can be restored should
    the C++ object resurface in Python. The map is only swept when a new
    ghost is added, so the dict of a wrapper that died during this test can
    linger after its C++ object dies -- and anything it holds (e.g. a
    composite mapper's ``_dataset``) then looks like a leak. Adding one
    throwaway ghost forces the sweep.
    """
    holder = _vtk.vtkPolyData()
    bait = _vtk.vtkPoints()
    bait._pyvista_ghost_bait = True
    holder.SetPoints(bait)
    # bait's wrapper dies while its C++ object is still held by holder, so it
    # is added to the ghost map, sweeping out stale ghosts; deleting holder
    # then kills the C++ object, letting a later sweep remove the bait itself.
    del bait
    del holder


_check_gc_key = pytest.StashKey()


@pytest.fixture(autouse=True)
def check_gc(request):
    """Snapshot live objects so leaks from this test can be detected.

    The check itself runs in the ``pytest_runtest_teardown`` hookwrapper
    below, not in this fixture's teardown: several fixtures are set up
    before this one (``monkeypatch`` via other autouse fixtures, the
    registry save/restore in ``tests/conftest.py::reset_global_state``),
    so their finalizers run *after* an autouse fixture's teardown -- and
    anything they still held (a patched-in mock, a test class left in a
    global registry) would be misreported here as a leak.
    """
    if request.node.get_closest_marker('skip_check_gc'):
        yield
        return

    # Snapshots so that leftovers of earlier tests that legitimately skip
    # this check are not blamed on this test. Matching vtkObjectBase by
    # isinstance rather than by class-name prefix also covers pyvista's own
    # vtk subclasses (PolyData, ...) and the pythonic override subclasses
    # VTK >= 9.6 instantiates, whose names lack the 'vtk' prefix.
    gc.collect()
    objs = gc.get_objects()  # scan the heap once, share across snapshots
    request.node.stash[_check_gc_key] = (
        Snapshot(pv.plotting.plotter.BasePlotter, objs=objs),
        Snapshot(_vtk.vtkObjectBase, label='VTK', objs=objs),
    )
    del objs
    yield


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_teardown(item):
    """Ensure that all VTK objects created during a test are garbage-collected.

    A hookwrapper so the check runs after every fixture finalizer has run
    (see ``check_gc``, which takes the snapshots this checks against).
    """
    yield
    snaps = item.stash.get(_check_gc_key, None)
    if snaps is None:
        return
    del item.stash[_check_gc_key]
    snap_plotter, snap_vtk = snaps

    pv.close_all()

    # Skip GC check if test failed (or was skipped during call)
    if not _test_passed(item):
        return

    # pytest holds every fixture value in item.funcargs until after all
    # teardown hooks have run (its runner only then sets it to None), so a
    # VTK-typed fixture value (sphere, texture, ...) would always be flagged
    # as a leak. The test passed and its fixtures are already finalized, so
    # release them a moment early.
    item.funcargs.clear()

    when = f'teardown of {item.name}'
    # gc_collect_once deduplicates on request.node; it only needs .node
    request = SimpleNamespace(node=item)
    gc_collect_once(request)

    def _assert_no_new():
        objs = gc.get_objects()
        try:
            # No plotter created during a test may survive it ...
            snap_plotter.assert_no_new(when, request=request, objs=objs)
            # ... and neither may any VTK object created during the test.
            snap_vtk.assert_no_new(when, request=request, objs=objs)
        except AssertionError:
            # A stale VTK ghost is deferred bookkeeping, not a leak: flush
            # the ghost map and re-check before reporting a failure.
            del objs
            _flush_vtk_ghosts()
            gc.collect()
            objs = gc.get_objects()
            snap_plotter.assert_no_new(when, request=request, objs=objs)
            snap_vtk.assert_no_new(when, request=request, objs=objs)
        del objs

    if item.get_closest_marker('expect_check_gc_fail'):
        with pytest.raises(AssertionError, match='Found '):
            _assert_no_new()
        return

    _assert_no_new()


@pytest.fixture
def colorful_tetrahedron():
    mesh = pv.Tetrahedron()
    mesh.cell_data['colors'] = [[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    return mesh


@pytest.fixture(autouse=True)
def set_default_theme():
    """Reset the testing theme for every test."""
    pv.global_theme.load_theme(pv.plotting.themes._TestingTheme())
    yield
    pv.global_theme.load_theme(pv.plotting.themes._TestingTheme())


def make_two_char_img(text):
    """Turn text into an image.

    This is really only here to make a two character black and white image.

    """
    # create a basic texture by plotting a sphere and converting the image
    # buffer to a texture
    pl = pv.Plotter(window_size=(300, 300), lighting=None, off_screen=True)
    pl.add_text(text, color='w', font_size=100, position=(0.1, 0.1), viewport=True, font='courier')
    pl.background_color = 'k'
    pl.camera.zoom = 'tight'
    return pv.Texture(pl.screenshot()).to_image()


def get_actor_mapper_input(actor):
    """Return a detached deep copy of the mapper's current pipeline input.

    The deep copy detaches the returned dataset from the live VTK
    pipeline so ``check_gc`` teardown doesn't race with test assertions
    that inspect its arrays.
    """
    actor.mapper.update()
    return pv.wrap(actor.mapper.GetInputDataObject(0, 0)).copy(deep=True)


class AlgorithmExecutionTracker:
    """Callable filter body that records whether it was invoked.

    Used to assert that mapper configuration is lazy, i.e. does not
    force the pipeline to run before ``show()`` or ``render()``.
    """

    def __init__(self) -> None:
        self.executed = False

    def __call__(self, mesh: pv.DataSet) -> pv.DataSet:
        self.executed = True
        return mesh


@pytest.fixture
def cubemap():
    """Sample texture as a cubemap."""
    return pv.Texture(
        [
            make_two_char_img('X+'),
            make_two_char_img('X-'),
            make_two_char_img('Y+'),
            make_two_char_img('Y-'),
            make_two_char_img('Z+'),
            make_two_char_img('Z-'),
        ],
    )
