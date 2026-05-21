"""Test functions from plotting extension."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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
