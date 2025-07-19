from __future__ import annotations

from pathlib import Path

from matplotlib.pyplot import imread
import pytest

import pyvista as pv
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper
from pyvista.plotting.utilities.sphinx_gallery import Scraper

# skip all tests if unable to render
pytestmark = pytest.mark.skip_plotting


class QApplication:
    def __init__(self, *args):
        pass

    def processEvents(self):  # noqa: N802
        pass


def test_scraper_with_app(tmpdir, monkeypatch):
    n_win = 2
    pytest.importorskip('sphinx_gallery')
    monkeypatch.setattr(pv, 'BUILDING_GALLERY', True)
    pv.close_all()

    scraper = Scraper()

    plotters = [pv.Plotter(off_screen=True) for _ in range(n_win)]

    # add cone, change view to test that it takes effect
    plotters[0].iren.initialize()
    pv.set_new_attribute(plotters[0], 'app', QApplication([]))  # fake QApplication
    plotters[0].add_mesh(pv.Cone())
    plotters[0].camera_position = 'xy'

    plotters[1].add_mesh(pv.Cone())

    src_dir = str(tmpdir)
    out_dir = str(Path(str(tmpdir)) / '_build' / 'html')
    img_fnames = [
        str(Path(src_dir) / 'auto_examples' / 'images' / f'sg_img_{n}.png') for n in range(n_win)
    ]

    gallery_conf = {'src_dir': src_dir, 'builder_name': 'html'}
    target_file = str(Path(src_dir) / 'auto_examples' / 'sg.py')
    block = None
    block_vars = dict(
        image_path_iterator=iter(img_fnames),
        example_globals=dict(a=1),
        target_file=target_file,
    )

    Path(img_fnames[0]).parent.mkdir(parents=True)
    for img_fname in img_fnames:
        assert not Path(img_fname).is_file()

    Path(out_dir).mkdir(parents=True)
    scraper(block, block_vars, gallery_conf)
    for img_fname in img_fnames:
        assert Path(img_fname).is_file()

    # test that the plot has the camera position updated with a checksum
    # when the Plotter has an app instance
    assert imread(img_fnames[0]).sum() != imread(img_fnames[1]).sum()

    for plotter in plotters:
        plotter.close()


@pytest.mark.parametrize('scraper_type', ['static', 'dynamic'])
@pytest.mark.parametrize('n_win', [1, 2])
def test_scraper(tmpdir, monkeypatch, n_win, scraper_type):
    pytest.importorskip('sphinx_gallery')
    monkeypatch.setattr(pv, 'BUILDING_GALLERY', True)
    pv.close_all()
    plotters = [pv.Plotter(off_screen=True) for _ in range(n_win)]
    plotter_gif = pv.Plotter()

    # Initialize scraper and check stable representation
    if scraper_type == 'static':
        scraper = Scraper()
        assert repr(scraper) == '<Scraper object>'
    elif scraper_type == 'dynamic':
        scraper = DynamicScraper()
        assert repr(scraper) == '<DynamicScraper object>'
    else:
        msg = f'Invalid scraper type: {scraper}'
        raise ValueError(msg)

    src_dir = str(tmpdir)
    out_dir = str(Path(str(tmpdir)) / '_build' / 'html')
    img_fnames = [
        str(Path(src_dir) / 'auto_examples' / 'images' / f'sg_img_{n}.png') for n in range(n_win)
    ]

    # create and save GIF to tmpdir
    gif_path = str(Path(tmpdir + 'sg_img_0.gif').resolve())
    plotter_gif.open_gif(gif_path)
    plotter_gif.write_frame()
    plotter_gif.close()

    gallery_conf = {'src_dir': src_dir, 'builder_name': 'html'}
    target_file = str(Path(src_dir) / 'auto_examples' / 'sg.py')
    block = ('empty_block', '', 0)
    block_vars = dict(
        image_path_iterator=iter(img_fnames),
        example_globals=dict(a=1, PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT=True),
        target_file=target_file,
    )

    Path(img_fnames[0]).parent.mkdir(parents=True)
    for img_fname in img_fnames:
        assert not Path(img_fname).is_file()

    # add gif to list after checking other filenames are empty
    img_fnames.append(gif_path)
    Path(out_dir).mkdir(parents=True)
    scraper(block, block_vars, gallery_conf)
    for img_fname in img_fnames:
        assert Path(img_fname).is_file()
    for plotter in plotters:
        plotter.close()


def test_scraper_raise(tmpdir):
    pytest.importorskip('sphinx_gallery')
    pv.close_all()
    plotter = pv.Plotter(off_screen=True)
    scraper = Scraper()
    src_dir = str(tmpdir)
    out_dir = str(Path(tmpdir) / '_build' / 'html')
    img_fname = str(Path(src_dir) / 'auto_examples' / 'images' / 'sg_img.png')
    gallery_conf = {'src_dir': src_dir, 'builder_name': 'html'}
    target_file = str(Path(src_dir) / 'auto_examples' / 'sg.py')
    block = None
    block_vars = dict(
        image_path_iterator=(img for img in [img_fname]),
        example_globals=dict(a=1),
        target_file=target_file,
    )
    Path(img_fname).parent.mkdir(parents=True)
    assert not Path(img_fname).is_file()
    Path(out_dir).mkdir(parents=True)

    with pytest.raises(RuntimeError, match='pyvista.BUILDING_GALLERY'):
        scraper(block, block_vars, gallery_conf)

    plotter.close()


def test_namespace_contract():
    assert hasattr(pv, '_get_sg_image_scraper')
