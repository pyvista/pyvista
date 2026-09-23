"""This conftest is here to allow for checking garbage collection and
memory leaks for all plotting tests
"""

from __future__ import annotations

import importlib.util
import platform

import pytest

import pyvista as pv
from pyvista import _vtk
from pyvista.plotting import system_supports_plotting
from tests.gc_check import assert_no_leaks
from tests.gc_check import check_enabled
from tests.gc_check import stash_phase_report
from tests.gc_check import take_snapshot

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


@pytest.fixture(autouse=True)
def _trame_array_cache():
    """Clear trame's serializer cache before ``check_gc``'s teardown check.

    trame's session-lifetime ``SynchronizationContext`` caches exported arrays on a
    20-second window, so they outlive the exporting test and are reported as its leak.
    Any scene-exporting test hits this (not just jupyter), hence the suite-wide scope.
    """
    yield
    if importlib.util.find_spec('trame_vtk') is None:
        return
    from trame_vtk.modules.vtk import HELPERS_PER_SERVER

    for helper in HELPERS_PER_SERVER.values():
        protocol = helper._root_protocol
        if protocol is None:
            continue
        for link_protocol in protocol.getLinkProtocols():
            context = getattr(link_protocol, 'context', None)
            if context is not None:
                context.data_array_cache.clear()


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


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):  # noqa: ARG001
    """Stash per-phase reports so check_gc can skip the leak check on failure."""
    outcome = yield
    stash_phase_report(item, outcome.get_result())


@pytest.fixture(autouse=True)
def check_gc(request):
    """Snapshot live plotters and VTK objects so leaks from this test can be detected."""
    if not check_enabled(request.node):
        yield
        return
    # BasePlotter is not a vtkObjectBase, so both types are needed here
    take_snapshot(
        request.node,
        (pv.plotting.plotter.BasePlotter, _vtk.vtkObjectBase),
        'VTK/plotter',
        owner=__name__,
    )
    yield


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_teardown(item):
    """Close every plotter, and check that nothing the test created outlived it.

    A hookwrapper so both run after every fixture finalizer has run (see ``check_gc``,
    which takes the snapshot this checks against).
    """
    yield
    # Unconditional, and before the check rather than from it: a test that leaves a
    # plotter open leaves a render window open for every test after it, whether or not
    # this run is checking for leaks.
    pv.close_all()
    assert_no_leaks(item, owner=__name__, flush_ghosts=True)


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
