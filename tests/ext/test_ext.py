"""Test functions from plotting extension."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import pyvista as pv
from pyvista.ext import plot_directive
from pyvista.ext import viewer_directive
from pyvista.ext.plot_directive import hash_plot_code


def test_hash_plot_code_consistency():
    code = 'import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])'
    options = {}

    hash1 = hash_plot_code(code, options)
    hash2 = hash_plot_code(code, options)
    assert hash1 == hash2
    assert len(hash1) == 16

    different_code = 'plt.plot([4, 5, 6])'
    hash3 = hash_plot_code(different_code, options)
    assert hash1 != hash3


def test_hash_plot_code_normalization():
    code_with_noise = (
        'import matplotlib.pyplot as plt  # plotting lib\n\nplt.plot([1, 2, 3])  # make plot\n\n'
    )
    code_clean = 'import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])'
    doctest_code = '>>> import matplotlib.pyplot as plt\n>>> plt.plot([1, 2, 3])'
    options = {}

    hash1 = hash_plot_code(code_with_noise, options)
    hash2 = hash_plot_code(code_clean, options)
    hash3 = hash_plot_code(doctest_code, options)
    assert hash1 == hash2 == hash3


def test_hash_plot_code_context_option():
    code = 'plt.plot([1, 2, 3])'

    hash_no_context = hash_plot_code(code, {})
    hash_with_context = hash_plot_code(code, {'context': True})
    hash_other_option = hash_plot_code(code, {'other': True})

    assert hash_no_context != hash_with_context
    assert hash_no_context == hash_other_option


class _Builder:
    def __init__(self, target_uri):
        self.target_uri = target_uri

    def get_target_uri(self, docname):
        assert docname == 'guide/example'
        return self.target_uri


@pytest.mark.parametrize(
    ('target_uri', 'expected_viewer_uri'),
    [
        ('guide/example.html', '../_static/viewer.html'),
        ('guide/example/', '../../_static/viewer.html'),
    ],
)
def test_offline_viewer_paths_use_builder_target_uri(
    tmp_path, monkeypatch, target_uri, expected_viewer_uri
):
    monkeypatch.setattr(viewer_directive, 'HTML_VIEWER_PATH', '/tmp/viewer.html')
    out_dir = tmp_path / '_build' / 'html'
    dest_file = out_dir / '_images' / 'plot_directive' / 'guide' / 'scene.vtksz'
    dest_file.parent.mkdir(parents=True)
    dest_file.touch()
    env = SimpleNamespace(
        docname='guide/example',
        app=SimpleNamespace(outdir=out_dir, builder=_Builder(target_uri)),
    )

    viewer_uri, asset_uri = viewer_directive._offline_viewer_paths(env, dest_file)

    assert viewer_uri == expected_viewer_uri
    assert asset_uri == '../_images/plot_directive/guide/scene.vtksz'


def test_offline_viewer_paths_warns_for_asset_outside_images(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(viewer_directive, 'HTML_VIEWER_PATH', '/tmp/viewer.html')
    out_dir = tmp_path / '_build' / 'html'
    dest_file = out_dir / 'plot_directive' / 'guide' / 'scene.vtksz'
    dest_file.parent.mkdir(parents=True)
    dest_file.touch()
    env = SimpleNamespace(
        docname='guide/example',
        app=SimpleNamespace(outdir=out_dir, builder=_Builder('guide/example.html')),
    )

    with caplog.at_level('WARNING', logger=viewer_directive.__name__):
        viewer_uri, asset_uri = viewer_directive._offline_viewer_paths(env, dest_file)

    assert viewer_uri is None
    assert asset_uri is None
    assert 'is not under outdir/_images; cannot compute asset URI' in caplog.text


def test_import_does_not_enable_gallery_mode():
    """Importing the extension must not change how plotters behave process-wide.

    ``BUILDING_GALLERY`` makes ``_ALL_PLOTTERS`` hold every plotter strongly rather
    than through a weak proxy, so setting it here at import time meant that merely
    importing this module -- which collecting this file does -- leaked a plotter for
    every test that ran afterwards in the same session. A documentation build gets
    the flag from :func:`~pyvista.ext.plot_directive.setup` instead.
    """
    code = (
        'import pyvista as pv;'
        'import pyvista.ext.plot_directive;'
        'print(pv.BUILDING_GALLERY, pv.OFF_SCREEN)'
    )
    # A subprocess because this process imported the module long ago, and with the
    # environment pinned because both flags also have an environment default.
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, 'PYVISTA_BUILDING_GALLERY': 'false', 'PYVISTA_OFF_SCREEN': 'false'},
    )
    assert result.stdout.split() == ['False', 'False']


def test_setup_enables_gallery_mode(monkeypatch):
    """Loading the extension is what turns gallery mode on."""
    monkeypatch.setattr(pv, 'BUILDING_GALLERY', False)
    monkeypatch.setattr(pv, 'OFF_SCREEN', False)

    plot_directive.setup(MagicMock())

    assert pv.BUILDING_GALLERY
    assert pv.OFF_SCREEN
