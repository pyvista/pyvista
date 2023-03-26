import os
import os.path as op

import pytest

import pyvista
from pyvista.plotting import system_supports_plotting
from pyvista.utilities import Scraper

# skip all tests if unable to render
if not system_supports_plotting():
    pytestmark = pytest.mark.skip(reason='Requires system to support plotting')


@pytest.mark.parametrize('n_win', [1, 2])
def test_scraper(tmpdir, monkeypatch, n_win):
    pytest.importorskip('sphinx_gallery')
    monkeypatch.setattr(pyvista, 'BUILDING_GALLERY', True)
    pyvista.close_all()
    plotters = [pyvista.Plotter(off_screen=True) for _ in range(n_win)]
    scraper = Scraper()
    src_dir = str(tmpdir)
    out_dir = op.join(str(tmpdir), '_build', 'html')
    img_fnames = [
        op.join(src_dir, 'auto_examples', 'images', f'sg_img_{n}.png') for n in range(n_win)
    ]
    gallery_conf = {"src_dir": src_dir, "builder_name": "html"}
    target_file = op.join(src_dir, 'auto_examples', 'sg.py')
    block = None
    block_vars = dict(
        image_path_iterator=iter(img_fnames),
        example_globals=dict(a=1),
        target_file=target_file,
    )
    os.makedirs(op.dirname(img_fnames[0]))
    for img_fname in img_fnames:
        assert not os.path.isfile(img_fname)
    os.makedirs(out_dir)
    scraper(block, block_vars, gallery_conf)
    for img_fname in img_fnames:
        assert os.path.isfile(img_fname)
    for plotter in plotters:
        plotter.close()


def test_scraper_raise(tmpdir):
    pytest.importorskip('sphinx_gallery')
    pyvista.close_all()
    plotter = pyvista.Plotter(off_screen=True)
    scraper = Scraper()
    src_dir = str(tmpdir)
    out_dir = op.join(str(tmpdir), '_build', 'html')
    img_fname = op.join(src_dir, 'auto_examples', 'images', 'sg_img.png')
    gallery_conf = {"src_dir": src_dir, "builder_name": "html"}
    target_file = op.join(src_dir, 'auto_examples', 'sg.py')
    block = None
    block_vars = dict(
        image_path_iterator=(img for img in [img_fname]),
        example_globals=dict(a=1),
        target_file=target_file,
    )
    os.makedirs(op.dirname(img_fname))
    assert not os.path.isfile(img_fname)
    os.makedirs(out_dir)

    with pytest.raises(RuntimeError, match="pyvista.BUILDING_GALLERY"):
        scraper(block, block_vars, gallery_conf)

    plotter.close()
