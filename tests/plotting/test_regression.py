from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING

from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.plotting.utilities import regression

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# skip all tests if unable to render
pytestmark = pytest.mark.skip_plotting


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(ndim=st.integers().filter(lambda x: x != 3))
def test_wrap_image_array_raises_ndim(mocker: MockerFixture, ndim):
    m = mocker.MagicMock()
    m.ndim = ndim
    with pytest.raises(
        ValueError,
        match=re.escape('Expecting a X by Y by (3 or 4) array'),
    ):
        pv.wrap_image_array(m)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(shape=st.integers().filter(lambda x: x not in (3, 4)))
def test_wrap_image_array_raises_shape(mocker: MockerFixture, shape):
    m = mocker.MagicMock()
    m.ndim = 3
    m.shape.__getitem__.return_value = shape
    with pytest.raises(
        ValueError,
        match=re.escape('Expecting a X by Y by (3 or 4) array'),
    ):
        pv.wrap_image_array(m)


def test_wrap_image_array_raises_dtype(mocker: MockerFixture):
    m = mocker.MagicMock()
    m.ndim = 3
    m.shape = [0, 0, 3]
    m.dtype = float
    with pytest.raises(
        ValueError,
        match=re.escape('Expecting a np.uint8 array'),
    ):
        pv.wrap_image_array(m)


def test_compare_images_raises(mocker: MockerFixture):
    @dataclass
    class Foo:
        n_calls: int = 0

        def __call__(self, v):  # noqa: ARG002
            self.n_calls += 1
            return Bar(self.n_calls)

    @dataclass
    class Bar:
        n_calls: int

        def GetDimensions(self) -> int:  # noqa: N802
            return self.n_calls

        @property
        def dimensions(self) -> int:
            return self.n_calls

    mocker.patch.object(regression, 'remove_alpha', new=Foo())

    with pytest.raises(RuntimeError, match=r'Input images are not the same size.'):
        pv.compare_images(pv.ImageData(), pv.ImageData())

    with pytest.raises(TypeError, match='may not be an image'):
        pv.compare_images(pv.ImageData(), examples.antfile)


def test_compare_images_two_plotters_same(sphere, tmpdir):
    filename = str(tmpdir.mkdir('tmpdir').join('tmp.png'))
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
    assert not pv.compare_images(arr1, pl2, use_vtk=False)

    with pytest.raises(TypeError):
        pv.compare_images(im1, pl1.render_window)

    # test that this fails when the plotter is closed
    pl1.close()
    with pytest.raises(RuntimeError, match='already been closed'):
        pv.compare_images(pl1, pl2)


def test_compare_images_two_plotter_different(sphere, airplane, tmpdir):
    tmppath = tmpdir.mkdir('tmpdir')
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

    assert pv.compare_images(im1, im2, use_vtk=False) > 50
