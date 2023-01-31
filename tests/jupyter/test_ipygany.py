import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.utilities.misc import PyVistaDeprecationWarning

has_ipygany = True
try:
    from ipygany.ipygany import Scene
    from ipywidgets import AppLayout

    from pyvista.jupyter.pv_ipygany import check_colormap
except:  # noqa: E722
    has_ipygany = False

skip_no_ipygany = pytest.mark.skipif(not has_ipygany, reason="Requires ipygany package")


@skip_no_ipygany
def test_set_jupyter_backend_ipygany():
    try:
        with pytest.warns(PyVistaDeprecationWarning):
            pv.global_theme.jupyter_backend = 'ipygany'
        assert pv.global_theme.jupyter_backend == 'ipygany'
    finally:
        pv.global_theme.jupyter_backend = None


@skip_no_ipygany
@pytest.mark.parametrize(
    'dataset',
    [
        examples.load_uniform(),  # UniformGrid
        examples.load_rectilinear(),  # RectilinearGrid
        examples.load_airplane(),  # PolyData
        examples.load_hexbeam(),  # UnstructuredGrid
        np.random.random((10, 3)),
    ],
)
def test_ipygany_from_plotter(dataset):
    pl = pv.Plotter(notebook=True)
    pl.add_mesh(dataset)
    viewer = pl.show(jupyter_backend='ipygany', return_viewer=True)
    assert isinstance(viewer, (AppLayout, Scene))


@skip_no_ipygany
def test_ipygany_from_show(sphere):
    jupyter_kwargs = {'height': 100, 'width': 100}
    viewer = sphere.plot(
        notebook=True, jupyter_backend='ipygany', return_viewer=True, jupyter_kwargs=jupyter_kwargs
    )
    assert isinstance(viewer, Scene)
    assert viewer.children


@skip_no_ipygany
def test_check_colormap_fail():
    with pytest.raises(ValueError, match='is not supported'):
        check_colormap('notacolormap')


@skip_no_ipygany
@pytest.mark.filterwarnings("ignore")
def test_wireframe(sphere):
    # this is expected to warn and plot nothing as it's unsupported
    with pytest.warns(UserWarning, match='not supported'):
        viewer = sphere.plot(
            style='wireframe', notebook=True, jupyter_backend='ipygany', return_viewer=True
        )
    assert isinstance(viewer, Scene)
    assert not viewer.children


@skip_no_ipygany
def test_ipygany_scalar_bar(sphere):
    sphere['my_values'] = sphere.points[:, 2]
    viewer = sphere.plot(
        notebook=True, jupyter_backend='ipygany', return_viewer=False, show_scalar_bar=True
    )
    assert viewer is None
