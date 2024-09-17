"""This conftest is here to allow for checking garbage collection and
memory leaks for all plotting tests
"""

from __future__ import annotations

import gc
import inspect

import pytest

import pyvista as pv
from pyvista.plotting import system_supports_plotting

# these are set here because we only need them for plotting tests
pv.global_theme.load_theme(pv.plotting.themes._TestingTheme())
pv.OFF_SCREEN = True
SKIP_PLOTTING = not system_supports_plotting()


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


def _is_vtk(obj):
    try:
        return obj.__class__.__name__.startswith('vtk')
    except Exception:  # old Python sometimes no __class__.__name__
        return False


@pytest.fixture
def skip_check_gc(check_gc):  # noqa: PT004
    """Skip check_gc fixture."""
    check_gc.skip = True


@pytest.fixture(autouse=True)
def check_gc():
    """Ensure that all VTK objects are garbage-collected by Python."""
    gc.collect()
    before = {id(o) for o in gc.get_objects() if _is_vtk(o)}

    class GcHandler:
        def __init__(self) -> None:
            # if set to True, will entirely skip checking in this fixture
            self.skip = False

    gc_handler = GcHandler()

    yield gc_handler

    if gc_handler.skip:
        return

    pv.close_all()

    gc.collect()
    after = [o for o in gc.get_objects() if _is_vtk(o) and id(o) not in before]
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
    assert len(after) == 0, msg


@pytest.fixture
def colorful_tetrahedron():
    mesh = pv.Tetrahedron()
    mesh.cell_data["colors"] = [[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    return mesh


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
def cubemap(texture):
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
