from __future__ import annotations

import cmcrameri
import cmocean
from colorcet import all_original_names
from colorcet import get_aliases
import matplotlib as mpl
import pytest

from doc.source.make_tables import _COLORMAP_INFO

CMAP_SET_MISMATCH_ERROR_MSG = (
    'Colormaps in documentation differ from colormaps available. '
    'The colormap table should be updated.'
)
DUPLICATE_CMAP_ERROR_MSG = 'Duplicate colormaps exist in the documentation.'


@pytest.fixture
def matplotlib_named_cmaps():
    # Need to unregister all 3rd-party cmaps
    for cmap in list(mpl.colormaps):
        try:
            mpl.colormaps.unregister(cmap)
        except (ValueError, AttributeError):
            continue

    is_reversed = lambda x: x.endswith('_r')
    is_synonym = lambda x: 'Grey' in x or 'grey' in x or 'yerg' in x
    return [cmap for cmap in mpl.colormaps if not is_synonym(cmap) and not is_reversed(cmap)]


def test_colormap_table_matplotlib(matplotlib_named_cmaps):
    if (
        'berlin' not in matplotlib_named_cmaps
        and 'vanimo' not in matplotlib_named_cmaps
        and 'managua' not in matplotlib_named_cmaps
    ):
        pytest.xfail('Older Matplotlib is missing a few colormaps.')
    documented_cmaps = [info.name for info in _COLORMAP_INFO if info.package == 'matplotlib']
    assert set(documented_cmaps) == set(matplotlib_named_cmaps), CMAP_SET_MISMATCH_ERROR_MSG
    assert sorted(documented_cmaps) == sorted(matplotlib_named_cmaps), DUPLICATE_CMAP_ERROR_MSG


def test_colormap_table_cmocean():
    cmocean_cmaps = cmocean.cm.cmapnames
    documented_cmaps = [info.name for info in _COLORMAP_INFO if info.package == 'cmocean']
    assert set(documented_cmaps) == set(cmocean_cmaps), CMAP_SET_MISMATCH_ERROR_MSG
    assert sorted(documented_cmaps) == sorted(cmocean_cmaps), DUPLICATE_CMAP_ERROR_MSG


def test_colormap_table_cmcrameri():
    cmcrameri_cmaps = [cmap for cmap in cmcrameri.cm.cmaps if not cmap.endswith('_r')]
    documented_cmaps = [info.name for info in _COLORMAP_INFO if info.package == 'cmcrameri']
    assert set(documented_cmaps) == set(cmcrameri_cmaps), CMAP_SET_MISMATCH_ERROR_MSG
    assert sorted(documented_cmaps) == sorted(cmcrameri_cmaps), DUPLICATE_CMAP_ERROR_MSG


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
        if (info.package == 'colorcet') and (info.kind.name != 'CATEGORICAL')
    ]
    assert set(documented_cmaps) == set(colorcet_continuous_cmaps), CMAP_SET_MISMATCH_ERROR_MSG
    assert sorted(documented_cmaps) == sorted(colorcet_continuous_cmaps), DUPLICATE_CMAP_ERROR_MSG


def test_colormap_table_colorcet_categorical(colorcet_categorical_cmaps):
    documented_cmaps = [
        info.name
        for info in _COLORMAP_INFO
        if info.package == 'colorcet' and info.kind.name == 'CATEGORICAL'
    ]
    assert set(documented_cmaps) == set(colorcet_categorical_cmaps), CMAP_SET_MISMATCH_ERROR_MSG
    assert sorted(documented_cmaps) == sorted(colorcet_categorical_cmaps), DUPLICATE_CMAP_ERROR_MSG
