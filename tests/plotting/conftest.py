"""This conftest is here to allow for checking garbage collection and
memory leaks for all plotting tests
"""
import gc
import inspect

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
    gc.collect()
    before = {id(o) for o in gc.get_objects() if _is_vtk(o)}
    yield

    # Do not check for collection if the test session failed. Tests that fail
    # also fail to cleanup their resources and this makes reading the unit test
    # output more difficult.
    #
    # This applies to the entire session, so it's going to be the most useful
    # when debugging tests with `pytest -x`
    pyvista.close_all()
    if request.session.testsfailed:
        return

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
        msg += f'{cn}: {referrers}\n'
        del cn, referrers
    assert len(after) == 0, msg


@pytest.fixture()
def colorful_tetrahedron():
    mesh = pyvista.Tetrahedron()
    mesh.cell_data["colors"] = [[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    return mesh
