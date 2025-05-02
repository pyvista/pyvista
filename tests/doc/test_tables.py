from __future__ import annotations

import cmocean
from colorcet.plotting import all_original_names
from colorcet.plotting import get_aliases
import matplotlib as mpl
import pytest

from doc.source.make_tables import _COLORMAP_INFO


def assert_cmaps_equal(actual, expected):
    msg = 'Documentation is missing named colormaps. The colormap table should be updated.'
    # Test same cmaps
    assert set(actual) == set(expected), msg
    # Test order
    assert actual == expected


@pytest.fixture
def matplotlib_default_cmaps():
    # Need to unregister all 3rd-party cmaps
    for cmap in list(mpl.colormaps):
        try:
            mpl.colormaps.unregister(cmap)
        except ValueError:
            continue
    return [cmap for cmap in mpl.colormaps if not cmap.endswith('_r')]


def test_colormap_table_matplotlib(matplotlib_default_cmaps):
    documented_cmaps = [info.name for info in _COLORMAP_INFO if info.package == 'matplotlib']
    assert_cmaps_equal(documented_cmaps, matplotlib_default_cmaps)


def test_colormap_table_cmocean():
    cmocean_cmaps = cmocean.cm.cmapnames
    documented_cmaps = [info.name for info in _COLORMAP_INFO if info.package == 'cmocean']
    assert_cmaps_equal(documented_cmaps, cmocean_cmaps)


@pytest.fixture
def colorcet_continuous_cmaps():
    # Get cmaps with alias and return the first alias
    cmaps = all_original_names(only_aliased=True, not_group='glasbey')
    return [get_aliases(name).split(',')[0] for name in cmaps]


@pytest.fixture
def colorcet_categorical_cmaps():
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


def test_colormap_table_colorcet_continuous(colorcet_continuous_cmaps):
    documented_cmaps = [
        info.name
        for info in _COLORMAP_INFO
        if info.package == 'colorcet' and info.kind != 'categorical'
    ]
    assert_cmaps_equal(documented_cmaps, colorcet_continuous_cmaps)


def test_colormap_table_colorcet_categorical(colorcet_categorical_cmaps):
    documented_cmaps = [
        info.name
        for info in _COLORMAP_INFO
        if info.package == 'colorcet' and info.kind == 'categorical'
    ]
    assert_cmaps_equal(documented_cmaps, colorcet_categorical_cmaps)
