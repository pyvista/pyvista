from __future__ import annotations

import cmocean
from colorcet.plotting import all_original_names
from colorcet.plotting import get_aliases
import matplotlib as mpl
import pytest

from doc.source.make_tables import _COLORMAP_INFO
from tests.conftest import MATPLOTLIB_VERSION_INFO

MISSING_COLORMAPS_MSG = (
    'Documentation is missing named colormaps. The colormap table should be updated.'
)


@pytest.mark.skipif(MATPLOTLIB_VERSION_INFO < (3, 6))
def test_colormap_table_matplotlib():
    # Need to access private var here because non-default cmaps are added
    # to the public `mpl.colormaps` registry
    matplotlib_cmaps = [cmap for cmap in mpl.colormaps._builtin_cmaps if not cmap.endswith('_r')]
    documented_colormaps = [info.name for info in _COLORMAP_INFO if info.package == 'matplotlib']
    assert documented_colormaps == matplotlib_cmaps, MISSING_COLORMAPS_MSG


def test_colormap_table_cmocean():
    cmocean_cmaps = cmocean.cm.cmapnames
    documented_colormaps = [info.name for info in _COLORMAP_INFO if info.package == 'cmocean']
    assert documented_colormaps == cmocean_cmaps, MISSING_COLORMAPS_MSG


@pytest.fixture
def colorcet_named_continuous_cmaps():
    # Get cmaps with alias and return the first alias
    cmaps = all_original_names(only_aliased=True, not_group='glasbey')
    return [get_aliases(name).split(',')[0] for name in cmaps]


@pytest.fixture
def colorcet_named_categorical_cmaps():
    # Get all glasbey cmaps and only keep ones with aliases or with
    # non-technical names
    cmaps = []
    alL_categorical_cmaps = all_original_names(group='glasbey')
    for original_name in alL_categorical_cmaps:
        if 'minc' in original_name:
            name = get_aliases(original_name).split(',')[0]
            if name == original_name:
                # No aliases, skip
                continue
        else:
            name = original_name
        cmaps.append(name)
    return cmaps


def test_colormap_table_colorcet_continuous(colorcet_named_continuous_cmaps):
    documented_colormaps = [
        info.name
        for info in _COLORMAP_INFO
        if info.package == 'colorcet' and info.kind != 'categorical'
    ]
    assert documented_colormaps == colorcet_named_continuous_cmaps, MISSING_COLORMAPS_MSG


def test_colormap_table_colorcet_categorical(colorcet_named_categorical_cmaps):
    documented_colormaps = [
        info.name
        for info in _COLORMAP_INFO
        if info.package == 'colorcet' and info.kind == 'categorical'
    ]
    assert documented_colormaps == colorcet_named_categorical_cmaps, MISSING_COLORMAPS_MSG
