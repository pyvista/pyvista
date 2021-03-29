import pytest

import pyvista as pv
from pyvista import jupyter

ALLOWED_BACKENDS = list(jupyter.ALLOWED_BACKENDS)
ALLOWED_BACKENDS.append(None)


@pytest.mark.parametrize('backend', ALLOWED_BACKENDS)
def test_set_jupyter_backend_ipygany(backend):
    pv.set_jupyter_backend(backend)
    if backend == 'none':
        backend = None
    assert pv.rcParams['jupyter_backend'] == backend


def test_set_jupyter_backend_ipygany_fail():
    with pytest.raises(ValueError, match='Invalid Jupyter notebook plotting backend'):
        pv.set_jupyter_backend('not a backend')
