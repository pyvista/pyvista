import os
import os.path as op

import pytest

import pyvista
from pyvista.utilities import Scraper


def test_scraper(tmpdir):
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
    scraper(block, block_vars, gallery_conf)
    assert os.path.isfile(img_fname)
    plotter.close()
