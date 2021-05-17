import numpy as np
import pytest

import pyvista
from pyvista.plotting import system_supports_plotting

NO_PLOTTING = not system_supports_plotting()

HAS_ITK = False
try:
    import itkwidgets
    HAS_ITK = True
except ImportError:
    pass

SPHERE = pyvista.Sphere()
SPHERE['z'] = SPHERE.points[:, 2]

no_itk = pytest.mark.skipif(NO_PLOTTING or not HAS_ITK, reason="Requires system to support plotting and have itkwidgets.")


@no_itk
def test_itk_plotting():
    viewer = pyvista.plot_itk(SPHERE)
    assert isinstance(viewer, itkwidgets.Viewer)


@no_itk
def test_itk_plotting_points():
    with pytest.raises(TypeError):
        viewer = pyvista.plot_itk([1, 2, 3], point_size='foo')

    viewer = pyvista.plot_itk(np.random.random((100, 3)))
    assert isinstance(viewer, itkwidgets.Viewer)


@no_itk
def test_itk_plotting_points_polydata():
    points = pyvista.PolyData(np.random.random((100, 3)))
    pl = pyvista.PlotterITK()
    pl.add_points(points)

    with pytest.raises(TypeError):
        viewer = pl.add_points([1, 2, 3], point_size='foo')

    viewer = pl.show()
    assert isinstance(viewer, itkwidgets.Viewer)


@no_itk
def test_itk_plotting_class():
    pl = pyvista.PlotterITK()
    pl.add_mesh(SPHERE, scalars='z')
    viewer = pl.show()
    assert isinstance(viewer, itkwidgets.Viewer)


@no_itk
def test_itk_plotting_class_no_scalars():
    pl = pyvista.PlotterITK()
    pl.add_mesh(SPHERE, color='w')
    with pytest.raises(ValueError):
        pl.camera_position = 'xy'
    with pytest.raises(ValueError):
        pl.camera_position = [[1, 0, 0], [1, 0, 0]]
    with pytest.raises(ValueError):
        pl.camera_position = [[1, 0, 0], [1, 0, 0], [1]]
    pl.camera_position = [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
    assert isinstance(pl.camera_position, list)
    pl.background_color = 'k'
    assert pl.background_color == (0.0, 0.0, 0.0)
    viewer = pl.show()
    assert isinstance(viewer, itkwidgets.Viewer)


@no_itk
def test_itk_plotting_class_npndarray_scalars():
    pl = pyvista.PlotterITK()
    pl.add_mesh(SPHERE, scalars=SPHERE.points[:, 0], smooth_shading=True)
    viewer = pl.show()
    assert isinstance(viewer, itkwidgets.Viewer)


@no_itk
def test_itk_plotting_class_unstructured(hexbeam):
    hexbeam.clear_arrays()
    pl = pyvista.PlotterITK()
    pl.add_mesh(hexbeam, smooth_shading=True)
    viewer = pl.show()
    assert isinstance(viewer, itkwidgets.Viewer)
