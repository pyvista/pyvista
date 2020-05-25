import numpy as np
import pytest

import pyvista
from pyvista.plotting import system_supports_plotting

# for azure testing and itkwidgets
# import matplotlib
# matplotlib.use("agg")

NO_PLOTTING = not system_supports_plotting()

HAS_ITK = False
try:
    import itkwidgets
    HAS_ITK = True
except ImportError:
    pass

SPHERE = pyvista.Sphere()
SPHERE['z'] = SPHERE.points[:, 2]


@pytest.mark.skipif(NO_PLOTTING or not HAS_ITK, reason="Requires system to support plotting and have itkwidgets.")
def test_itk_plotting():
    viewer = pyvista.plot_itk(SPHERE)
    assert isinstance(viewer, itkwidgets.Viewer)


@pytest.mark.skipif(NO_PLOTTING or not HAS_ITK, reason="Requires system to support plotting and have itkwidgets.")
def test_itk_plotting_points():
    viewer = pyvista.plot_itk(np.random.random((100, 3)))
    assert isinstance(viewer, itkwidgets.Viewer)


@pytest.mark.skipif(NO_PLOTTING or not HAS_ITK, reason="Requires system to support plotting and have itkwidgets.")
def test_itk_plotting_class():
    pl = pyvista.PlotterITK()
    pl.add_mesh(SPHERE, scalars='z')
    viewer = pl.show()
    assert isinstance(viewer, itkwidgets.Viewer)


@pytest.mark.skipif(NO_PLOTTING or not HAS_ITK, reason="Requires system to support plotting and have itkwidgets.")
def test_itk_plotting_class_no_scalars():
    pl = pyvista.PlotterITK()
    pl.add_mesh(SPHERE, color='w')
    viewer = pl.show()
    assert isinstance(viewer, itkwidgets.Viewer)


@pytest.mark.skipif(NO_PLOTTING or not HAS_ITK, reason="Requires system to support plotting and have itkwidgets.")
def test_itk_plotting_class_npndarray_scalars():
    pl = pyvista.PlotterITK()
    pl.add_mesh(SPHERE, scalars=SPHERE.points[:, 0])
    viewer = pl.show()
    assert isinstance(viewer, itkwidgets.Viewer)
