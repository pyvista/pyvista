from __future__ import annotations

import cmocean
import matplotlib as mpl

from doc.source.make_tables import _COLORMAP_INFO


def test_matplotlib_colormaps():
    matplotlib_cmaps = [cmap for cmap in mpl.colormaps._builtin_cmaps if not cmap.endswith('_r')]
    documented_colormaps = [info.name for info in _COLORMAP_INFO if info.package == 'matplotlib']
    assert documented_colormaps == matplotlib_cmaps


def test_colorcet_colormaps():
    cmocean_cmaps = cmocean.cm.cmapnames
    documented_colormaps = [info.name for info in _COLORMAP_INFO if info.package == 'cmocean']
    assert documented_colormaps == cmocean_cmaps
