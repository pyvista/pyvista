"""This conftest is here to allow for checking garbage collection and
memory leaks for all plotting tests
"""

from __future__ import annotations

import gc
import inspect
import platform

import pytest
from vtk import vtkObjectBase

import pyvista as pv
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


@pytest.fixture(autouse=True)
def check_gc(request):
    """Ensure that all VTK objects are garbage-collected by Python."""
    if request.node.get_closest_marker('skip_check_gc'):
        yield
        return

    # Get all VTK objects before calling the test
    gc.collect()
    before = set()
    for obj in gc.get_objects():
        # Micro-optimized for performance as this is called millions of times
        try:
            if isinstance(obj, vtkObjectBase) and obj.__class__.__name__.startswith('vtk'):
                before.add(id(obj))
        except ReferenceError:
            pass

    yield

    pv.close_all()

    # Skip GC check if test failed
    if hasattr(request.node, 'rep_call') and request.node.rep_call.failed:
        return

    # get all vtk objects after the test
    gc.collect()
    after = []
    for obj in gc.get_objects():
        # Micro-optimized for performance as this is called millions of times
        try:
            if (
                isinstance(obj, vtkObjectBase)
                and obj.__class__.__name__.startswith('vtk')
                and id(obj) not in before
            ):
                after.append(obj)
        except ReferenceError:
            pass

    msg = 'Not all objects GCed:\n'
    for obj in after:
        cn = obj.__class__.__name__
        cf = inspect.currentframe()
        referrers = [v for v in gc.get_referrers(obj) if v is not after and v is not cf]
        del cf
        for ri, referrer in enumerate(referrers):
            if isinstance(referrer, dict):
                for k, v in referrer.items():
                    if k is obj:
                        referrers[ri] = 'dict: d key'
                        del k, v
                        break
                    elif v is obj:
                        referrers[ri] = f'dict: d[{k!r}]'
                        del k, v
                        break
                    del k, v
                else:
                    referrers[ri] = f'dict: len={len(referrer)}'
            else:
                referrers[ri] = repr(referrer)
            del ri, referrer
        msg += f'{cn} at {hex(id(obj))}: {referrers}\n'
        del cn, referrers

    if request.node.get_closest_marker('expect_check_gc_fail'):
        assert after
        return

    assert len(after) == 0, msg


@pytest.fixture
def colorful_tetrahedron():
    mesh = pv.Tetrahedron()
    mesh.cell_data['colors'] = [[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    return mesh


@pytest.fixture(autouse=True)
def set_default_theme(request: pytest.FixtureRequest):
    """Reset the testing theme for every test.
    Use @pytest.mark.no_default_theme to skip this autouse fixture
    """
    if 'no_default_theme' in request.keywords:
        yield
        return
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
