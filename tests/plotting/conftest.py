"""This conftest is here to allow for checking garbage collection and
memory leaks for all plotting tests
"""
import gc

import pytest

import pyvista

# these are set here because we only need them for plotting tests
pyvista.global_theme.load_theme(pyvista.themes._TestingTheme())
pyvista.OFF_SCREEN = True


def _is_vtk(obj):
    try:
        return obj.__class__.__name__.startswith('vtk')
    except Exception:  # old Python sometimes no __class__.__name__
        return False


@pytest.fixture(autouse=True)
def check_gc(request):
    """Ensure that all VTK objects are garbage-collected by Python."""
    before = set(id(o) for o in gc.get_objects() if _is_vtk(o))
    yield

    # Do not check for collection if the test session failed. Tests that fail
    # also fail to cleanup their resources and this makes reading the unit test
    # output more difficult.
    #
    # This applies to the entire session, so it's going to be the most useful
    # when debugging tests with `pytest -x`
    if request.session.testsfailed:
        return

    pyvista.close_all()

    gc.collect()
    after = [o for o in gc.get_objects() if _is_vtk(o) and id(o) not in before]
    assert len(after) == 0, 'Not all objects GCed:\n' + '\n'.join(
        sorted(o.__class__.__name__ for o in after)
    )
