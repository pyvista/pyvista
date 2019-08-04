import os

import pytest
import numpy as np
from PIL import Image

import sphinx_gallery
from sphinx_gallery.gen_gallery import _complete_gallery_conf
from sphinx_gallery.scrapers import (figure_rst, mayavi_scraper, SINGLE_IMAGE,
                                     matplotlib_scraper, ImagePathIterator,
                                     save_figures, _KNOWN_IMG_EXTS)
from sphinx_gallery.utils import _TempDir


@pytest.fixture(scope='function')
def gallery_conf(tmpdir):
    """Sets up a test sphinx-gallery configuration"""
    gallery_conf = _complete_gallery_conf({}, str(tmpdir), True, False)
    gallery_conf.update(examples_dir=_TempDir(), gallery_dir=str(tmpdir))
    return gallery_conf


class matplotlib_svg_scraper():

    def __repr__(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        return matplotlib_scraper(*args, format='svg', **kwargs)


@pytest.mark.parametrize('ext', ('png', 'svg'))
def test_save_matplotlib_figures(gallery_conf, ext):
    """Test matplotlib figure save."""
    if ext == 'svg':
        gallery_conf['image_scrapers'] = (matplotlib_svg_scraper(),)
    import matplotlib.pyplot as plt  # nest these so that Agg can be set
    plt.plot(1, 1)
    fname_template = os.path.join(gallery_conf['gallery_dir'], 'image{0}.png')
    image_path_iterator = ImagePathIterator(fname_template)
    block = ('',) * 3
    block_vars = dict(image_path_iterator=image_path_iterator)
    image_rst = save_figures(block, block_vars, gallery_conf)
    assert len(image_path_iterator) == 1
    fname = '/image1.{0}'.format(ext)
    assert fname in image_rst
    fname = gallery_conf['gallery_dir'] + fname
    assert os.path.isfile(fname)

    # Test capturing 2 images with shifted start number
    image_path_iterator.next()
    image_path_iterator.next()
    plt.plot(1, 1)
    plt.figure()
    plt.plot(1, 1)
    image_rst = save_figures(block, block_vars, gallery_conf)
    assert len(image_path_iterator) == 5
    for ii in range(4, 6):
        fname = '/image{0}.{1}'.format(ii, ext)
        assert fname in image_rst
        fname = gallery_conf['gallery_dir'] + fname
        assert os.path.isfile(fname)


def test_save_mayavi_figures(gallery_conf):
    """Test file naming when saving figures. Requires mayavi."""
    try:
        from mayavi import mlab
    except ImportError:
        raise pytest.skip('Mayavi not installed')
    import matplotlib.pyplot as plt
    mlab.options.offscreen = True

    gallery_conf.update(
        image_scrapers=(matplotlib_scraper, mayavi_scraper))
    fname_template = os.path.join(gallery_conf['gallery_dir'], 'image{0}.png')
    image_path_iterator = ImagePathIterator(fname_template)
    block = ('',) * 3
    block_vars = dict(image_path_iterator=image_path_iterator)

    plt.axes([-0.1, -0.1, 1.2, 1.2])
    plt.pcolor([[0]], cmap='Greens')
    mlab.test_plot3d()
    image_rst = save_figures(block, block_vars, gallery_conf)
    assert len(plt.get_fignums()) == 0
    assert len(image_path_iterator) == 2
    assert '/image0.png' not in image_rst
    assert '/image1.png' in image_rst
    assert '/image2.png' in image_rst
    assert '/image3.png' not in image_rst
    assert not os.path.isfile(fname_template.format(0))
    assert os.path.isfile(fname_template.format(1))
    assert os.path.isfile(fname_template.format(2))
    assert not os.path.isfile(fname_template.format(0))
    with Image.open(fname_template.format(1)) as img:
        pixels = np.asarray(img.convert("RGB"))
    assert (pixels == [247, 252, 245]).all()  # plt first

    # Test next-value handling, plus image_scrapers modification
    gallery_conf.update(image_scrapers=(matplotlib_scraper,))
    mlab.test_plot3d()
    plt.axes([-0.1, -0.1, 1.2, 1.2])
    plt.pcolor([[0]], cmap='Reds')
    image_rst = save_figures(block, block_vars, gallery_conf)
    assert len(plt.get_fignums()) == 0
    assert len(image_path_iterator) == 3
    assert '/image1.png' not in image_rst
    assert '/image2.png' not in image_rst
    assert '/image3.png' in image_rst
    assert '/image4.png' not in image_rst
    assert not os.path.isfile(fname_template.format(0))
    for ii in range(3):
        assert os.path.isfile(fname_template.format(ii + 1))
    assert not os.path.isfile(fname_template.format(4))
    with Image.open(fname_template.format(3)) as img:
        pixels = np.asarray(img.convert("RGB"))
    assert (pixels == [255, 245, 240]).all()


def _custom_func(x, y, z):
    return ''


def test_custom_scraper(gallery_conf, monkeypatch):
    """Test custom scrapers."""
    # custom finders
    with monkeypatch.context() as m:
        m.setattr(sphinx_gallery, '_get_sg_image_scraper',
                  lambda: _custom_func, raising=False)
        for cust in (_custom_func, 'sphinx_gallery'):
            gallery_conf.update(image_scrapers=[cust])
            fname_template = os.path.join(gallery_conf['gallery_dir'],
                                          'image{0}.png')
            image_path_iterator = ImagePathIterator(fname_template)
            block = ('',) * 3
            block_vars = dict(image_path_iterator=image_path_iterator)

    # degenerate
    gallery_conf.update(image_scrapers=['foo'])
    complete_args = (gallery_conf, gallery_conf['gallery_dir'], True, False)
    with pytest.raises(ValueError, match='Unknown image scraper'):
        _complete_gallery_conf(*complete_args)
    gallery_conf.update(
        image_scrapers=[lambda x, y, z: y['image_path_iterator'].next()])
    with pytest.raises(RuntimeError, match='did not produce expected image'):
        save_figures(block, block_vars, gallery_conf)
    gallery_conf.update(image_scrapers=[lambda x, y, z: 1.])
    with pytest.raises(TypeError, match='was not a string'):
        save_figures(block, block_vars, gallery_conf)
    # degenerate string interface
    gallery_conf.update(image_scrapers=['sphinx_gallery'])
    with monkeypatch.context() as m:
        m.setattr(sphinx_gallery, '_get_sg_image_scraper', 'foo',
                  raising=False)
        with pytest.raises(ValueError, match='^Unknown image.*\n.*callable'):
            _complete_gallery_conf(*complete_args)
    with monkeypatch.context() as m:
        m.setattr(sphinx_gallery, '_get_sg_image_scraper', lambda: 'foo',
                  raising=False)
        with pytest.raises(ValueError, match='^Scraper.*was not callable'):
            _complete_gallery_conf(*complete_args)


@pytest.mark.parametrize('ext', _KNOWN_IMG_EXTS)
def test_figure_rst(ext):
    """Testing rst of images"""
    figure_list = ['sphx_glr_plot_1.' + ext]
    image_rst = figure_rst(figure_list, '.')
    single_image = """
.. image:: /sphx_glr_plot_1.{ext}
    :class: sphx-glr-single-img
""".format(ext=ext)
    assert image_rst == single_image

    image_rst = figure_rst(figure_list + ['second.' + ext], '.')

    image_list_rst = """
.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /sphx_glr_plot_1.{ext}
            :class: sphx-glr-multi-img

    *

      .. image:: /second.{ext}
            :class: sphx-glr-multi-img
""".format(ext=ext)
    assert image_rst == image_list_rst

    # test issue #229
    local_img = [os.path.join(os.getcwd(), 'third.' + ext)]
    image_rst = figure_rst(local_img, '.')

    single_image = SINGLE_IMAGE % ("third." + ext)
    assert image_rst == single_image


def test_iterator():
    """Test ImagePathIterator."""
    ipi = ImagePathIterator('foo{0}')
    ipi._stop = 10
    with pytest.raises(RuntimeError, match='10 images'):
        for ii in ipi:
            pass
