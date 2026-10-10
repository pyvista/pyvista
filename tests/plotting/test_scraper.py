from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from matplotlib.pyplot import imread
import pytest

import pyvista as pv
from pyvista.plotting.plotter import BasePlotter
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
    else:  # pragma: no cover -- parametrize covers every case
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
    pv.close_all()
    pl = pv.Plotter(off_screen=True)
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

    with pytest.raises(RuntimeError, match=r'pyvista.BUILDING_GALLERY'):
        scraper(block, block_vars, gallery_conf)

    pl.close()


def test_namespace_contract():
    assert hasattr(pv, '_get_sg_image_scraper')


@pytest.mark.parametrize(
    ('example_globals', 'exported'),
    [
        ({}, True),
        ({'PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT': True}, False),
        ({'PYVISTA_GALLERY_FORCE_STATIC': True}, False),
        (
            {
                'PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT': True,
                'PYVISTA_GALLERY_FORCE_STATIC': False,
            },
            True,
        ),
    ],
)
def test_show_skips_scene_export_for_static_example(monkeypatch, example_globals, exported):
    monkeypatch.setattr(pv, 'BUILDING_GALLERY', True)
    fake = SimpleNamespace(export_vtksz=lambda filename: b'scene')  # noqa: ARG005
    monkeypatch.setattr(BasePlotter, '_trame_component', lambda self: fake)  # noqa: ARG005
    pv.close_all()
    exec('pv.Sphere().plot(off_screen=True)', {'pv': pv, **example_globals})  # noqa: S102
    (pl,) = pv.plotting.plotter._ALL_PLOTTERS.values()
    assert pl.last_image is not None
    assert pl.last_vtksz == (b'scene' if exported else None)
    del pl
    pv.close_all()


def test_dynamic_scraper_clears_block_force_static(tmpdir, monkeypatch):
    monkeypatch.setattr(pv, 'BUILDING_GALLERY', True)
    pv.close_all()
    example_globals = {'PYVISTA_GALLERY_FORCE_STATIC': True}
    block_vars = dict(image_path_iterator=iter([]), example_globals=example_globals)
    gallery_conf = {'src_dir': str(tmpdir), 'builder_name': 'html'}
    DynamicScraper()(('code', 'PYVISTA_GALLERY_FORCE_STATIC = True', 0), block_vars, gallery_conf)
    assert 'PYVISTA_GALLERY_FORCE_STATIC' not in example_globals


@pytest.mark.parametrize(
    ('make_scraper', 'exported'),
    [
        (Scraper, False),
        (pv._get_sg_image_scraper, False),
        (DynamicScraper, True),
        (lambda: (Scraper(), DynamicScraper()), True),
    ],
)
def test_show_skips_scene_export_for_static_scraper(monkeypatch, make_scraper, exported):
    monkeypatch.setattr(pv, 'BUILDING_GALLERY', True)
    fake = SimpleNamespace(export_vtksz=lambda filename: b'scene')  # noqa: ARG005
    monkeypatch.setattr(BasePlotter, '_trame_component', lambda self: fake)  # noqa: ARG005
    pv.close_all()
    make_scraper()
    pv.Sphere().plot(off_screen=True)
    (pl,) = pv.plotting.plotter._ALL_PLOTTERS.values()
    assert pl.last_vtksz == (b'scene' if exported else None)
    del pl
    pv.close_all()


@pytest.mark.parametrize(
    ('make_scraper', 'code', 'n_vtksz'),
    [
        (Scraper, 'pv.Sphere().plot()', 0),
        (pv._get_sg_image_scraper, 'pv.Sphere().plot()', 0),
        (DynamicScraper, 'PYVISTA_GALLERY_FORCE_STATIC = True\npv.Sphere().plot()', 0),
        (
            DynamicScraper,
            'PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True\npv.Sphere().plot()',
            0,
        ),
        (DynamicScraper, 'pv.Sphere().plot()', 1),
    ],
)
def test_scraper_writes_no_vtksz_for_static_plot(
    tmp_path, monkeypatch, make_scraper, code, n_vtksz
):
    monkeypatch.setattr(pv, 'BUILDING_GALLERY', True)
    exports = []
    fake = SimpleNamespace(export_vtksz=lambda filename: exports.append(filename) or b'scene')
    monkeypatch.setattr(BasePlotter, '_trame_component', lambda self: fake)  # noqa: ARG005
    pv.close_all()
    scraper = make_scraper()
    example_globals = {'pv': pv}
    exec(code, example_globals)  # noqa: S102

    images = tmp_path / 'images'
    images.mkdir()
    block_vars = dict(
        image_path_iterator=iter([str(images / 'sg_img.png')]),
        example_globals=example_globals,
    )
    scraper(('code', code, 0), block_vars, {'src_dir': str(tmp_path), 'builder_name': 'html'})

    assert len(list(images.glob('*.vtksz'))) == len(exports) == n_vtksz
    assert len(list(images.glob('*.png'))) == 1
