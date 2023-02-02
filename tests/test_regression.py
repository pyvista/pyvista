import pytest

import pyvista as pv
from pyvista.plotting import system_supports_plotting

# skip all tests if unable to render
if not system_supports_plotting():
    pytestmark = pytest.mark.skip


def test_compare_images_two_plotters_same(sphere, tmpdir):
    filename = str(tmpdir.mkdir("tmpdir").join('tmp.png'))
    pl1 = pv.Plotter()
    pl1.add_mesh(sphere)
    arr1 = pl1.screenshot(filename)
    im1 = pv.read(filename)

    pl2 = pv.Plotter()
    pl2.add_mesh(sphere)

    assert not pv.compare_images(pl1, pl2)
    assert not pv.compare_images(arr1, pl2)
    assert not pv.compare_images(im1, pl2)
    assert not pv.compare_images(filename, pl2)
    assert not pv.compare_images(arr1, pl2, use_vtk=True)

    with pytest.raises(TypeError):
        pv.compare_images(im1, pl1.render_window)

    # test that this fails when the plotter is closed
    pl1.close()
    with pytest.raises(RuntimeError, match='already been closed'):
        pv.compare_images(pl1, pl2)


def test_compare_images_two_plotter_different(sphere, airplane, tmpdir):
    tmppath = tmpdir.mkdir("tmpdir")
    filename = str(tmppath.join('tmp.png'))
    filename2 = str(tmppath.join('tmp2.png'))
    pl1 = pv.Plotter()
    pl1.add_mesh(sphere)
    arr1 = pl1.screenshot(filename)
    im1 = pv.read(filename)

    pl2 = pv.Plotter()
    pl2.add_mesh(airplane)
    arr2 = pl2.screenshot(filename2)
    im2 = pv.read(filename2)

    assert pv.compare_images(arr1, pl2) > 10000
    assert pv.compare_images(arr1, arr2) > 10000

    assert pv.compare_images(pl1, pl2) > 10000

    assert pv.compare_images(im1, pl2) > 10000
    assert pv.compare_images(im1, im2) > 10000

    assert pv.compare_images(filename, pl2) > 10000
    assert pv.compare_images(filename, filename2) > 10000

    assert pv.compare_images(arr1, pl2, use_vtk=True) > 10000

    with pytest.raises(TypeError):
        pv.compare_images(im1, pl1.render_window)
