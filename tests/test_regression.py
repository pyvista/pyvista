import pytest

import pyvista as pv

def test_compare_images_two_plotters(sphere, tmpdir):
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
        pv.compare_images(im1, pl1.ren_win)
