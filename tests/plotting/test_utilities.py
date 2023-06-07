import numpy as np
import pytest

import pyvista
from pyvista.plotting import system_supports_plotting
from pyvista.report import GPUInfo

skip_no_plotting = pytest.mark.skipif(
    not system_supports_plotting(), reason="Requires system to support plotting"
)


@skip_no_plotting
def test_gpuinfo():
    gpuinfo = GPUInfo()
    _repr = gpuinfo.__repr__()
    _repr_html = gpuinfo._repr_html_()
    assert isinstance(_repr, str) and len(_repr) > 1
    assert isinstance(_repr_html, str) and len(_repr_html) > 1

    # test corrupted internal infos
    gpuinfo._gpu_info = 'foo'
    for func_name in ['renderer', 'version', 'vendor']:
        with pytest.raises(RuntimeError, match=func_name):
            getattr(gpuinfo, func_name)()


@skip_no_plotting
def test_ray_trace_plot(sphere):
    points, ind = sphere.ray_trace(
        [0, 0, 0], [1, 1, 1], plot=True, first_point=True, off_screen=True
    )
    assert np.any(points)
    assert np.any(ind)


@skip_no_plotting
def test_plot_curvature(sphere):
    sphere.plot_curvature(off_screen=True)


@skip_no_plotting
def test_plot_boundaries():
    # make sure to plot an object that has boundaries
    pyvista.Cube().plot_boundaries(off_screen=True)


@skip_no_plotting
@pytest.mark.parametrize('flip', [True, False])
@pytest.mark.parametrize('faces', [True, False])
def test_plot_normals(sphere, flip, faces):
    sphere.plot_normals(off_screen=True, flip=flip, faces=faces)


def test_get_sg_image_scraper():
    scraper = pyvista._get_sg_image_scraper()
    assert isinstance(scraper, pyvista.Scraper)
    assert callable(scraper)
