from __future__ import annotations

import cmcrameri
import cmocean
from colorcet import all_original_names
from colorcet import get_aliases
import docutils.nodes
from docutils.parsers.rst.directives import class_option
import matplotlib as mpl
import pytest

from doc.source import make_tables
import pyvista as pv
from pyvista.examples._dataset_loader import _DatasetLoader
from pyvista.examples._dataset_loader import _MultiFileDatasetLoader
from pyvista.examples._dataset_loader import _SingleFileDatasetLoader

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


@pytest.mark.skipif(
    'dev' in mpl.__version__, reason='Only run documentation tests for matplotlib releases.'
)
def test_colormap_table_matplotlib(matplotlib_named_cmaps):
    if (
        'berlin' not in matplotlib_named_cmaps
        or 'vanimo' not in matplotlib_named_cmaps
        or 'managua' not in matplotlib_named_cmaps
        or 'okabe_ito' not in matplotlib_named_cmaps
    ):
        pytest.xfail('Older Matplotlib is missing a few colormaps.')
    documented_cmaps = [
        info.name for info in make_tables._COLORMAP_INFO if info.package == 'matplotlib'
    ]
    assert set(documented_cmaps) == set(matplotlib_named_cmaps), CMAP_SET_MISMATCH_ERROR_MSG
    assert sorted(documented_cmaps) == sorted(matplotlib_named_cmaps), DUPLICATE_CMAP_ERROR_MSG


def test_colormap_table_cmocean():
    cmocean_cmaps = cmocean.cm.cmapnames
    documented_cmaps = [
        info.name for info in make_tables._COLORMAP_INFO if info.package == 'cmocean'
    ]
    assert set(documented_cmaps) == set(cmocean_cmaps), CMAP_SET_MISMATCH_ERROR_MSG
    assert sorted(documented_cmaps) == sorted(cmocean_cmaps), DUPLICATE_CMAP_ERROR_MSG


def test_colormap_table_cmcrameri():
    cmcrameri_cmaps = [cmap for cmap in cmcrameri.cm.cmaps if not cmap.endswith('_r')]
    documented_cmaps = [
        info.name for info in make_tables._COLORMAP_INFO if info.package == 'cmcrameri'
    ]
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
        for info in make_tables._COLORMAP_INFO
        if (info.package == 'colorcet') and (info.kind.name != 'CATEGORICAL')
    ]
    assert set(documented_cmaps) == set(colorcet_continuous_cmaps), CMAP_SET_MISMATCH_ERROR_MSG
    assert sorted(documented_cmaps) == sorted(colorcet_continuous_cmaps), DUPLICATE_CMAP_ERROR_MSG


def test_colormap_table_colorcet_categorical(colorcet_categorical_cmaps):
    documented_cmaps = [
        info.name
        for info in make_tables._COLORMAP_INFO
        if info.package == 'colorcet' and info.kind.name == 'CATEGORICAL'
    ]
    assert set(documented_cmaps) == set(colorcet_categorical_cmaps), CMAP_SET_MISMATCH_ERROR_MSG
    assert sorted(documented_cmaps) == sorted(colorcet_categorical_cmaps), DUPLICATE_CMAP_ERROR_MSG


def test_update_image_placeholders_missing_dataset(monkeypatch, caplog, tmp_path):
    """Test missing dataset images suggest adding to the extension dict."""
    monkeypatch.chdir(tmp_path)

    gallery = tmp_path / '_build' / 'pyvista_plot_directive' / 'api' / 'examples' / '_autosummary'
    gallery.mkdir(parents=True)

    monkeypatch.setattr(
        make_tables,
        'DATASET_GALLERY_IMAGE_NOT_AVAILABLE_PATH',
        str(tmp_path / 'not_available.png'),
    )

    node = docutils.nodes.image(
        uri='../_build/pyvista_plot_directive/api/examples/_autosummary/'
        'pyvista-examples-download_sheen_chair-IMAGE-HASH-PLACEHOLDER_00_00.png'
    )

    with caplog.at_level('WARNING'):
        make_tables._update_image_placeholders(node)

    assert node['uri'].endswith('not_available.png')
    assert 'sheen_chair' in caplog.text
    assert "add `'sheen_chair': None` to DATASET_GALLERY_IMAGE_EXT_DICT" in caplog.text


def test_update_image_placeholders_missing_non_dataset(monkeypatch, caplog, tmp_path):
    """Test missing non-dataset images do not suggest dictionary entries."""
    monkeypatch.chdir(tmp_path)

    gallery = tmp_path / '_build' / 'pyvista_plot_directive' / 'api' / 'examples' / '_autosummary'
    gallery.mkdir(parents=True)

    monkeypatch.setattr(
        make_tables,
        'DATASET_GALLERY_IMAGE_NOT_AVAILABLE_PATH',
        str(tmp_path / 'not_available.png'),
    )

    node = docutils.nodes.image(uri='../_build/some-other-image-IMAGE-HASH-PLACEHOLDER_00_00.png')

    with caplog.at_level('WARNING'):
        make_tables._update_image_placeholders(node)

    assert node['uri'].endswith('not_available.png')
    assert 'DATASET_GALLERY_IMAGE_EXT_DICT' not in caplog.text


def test_update_image_placeholders_existing(monkeypatch, tmp_path):
    """Test resolving a placeholder to an existing generated image."""
    monkeypatch.chdir(tmp_path)

    gallery = tmp_path / '_build' / 'pyvista_plot_directive' / 'api' / 'examples' / '_autosummary'
    gallery.mkdir(parents=True)

    expected = gallery / 'pyvista-examples-download_sheen_chair-abc123_00_00.png'
    expected.touch()

    node = docutils.nodes.image(
        uri='../_build/pyvista_plot_directive/api/examples/_autosummary/'
        'pyvista-examples-download_sheen_chair-IMAGE-HASH-PLACEHOLDER_00_00.png'
    )

    make_tables._update_image_placeholders(node)

    assert node['uri'].endswith(expected.name)


@pytest.mark.parametrize(
    ('filename', 'field', 'slug', 'label'),
    [
        (
            'mesh.vtp',
            ':class:`~pyvista.core.utilities.reader.XMLPolyDataReader`',
            'reader-xml-poly-data-reader',
            'XMLPolyDataReader',
        ),
        (
            'mesh.frd',
            '``pyvista_frd.FRDReader``',
            'reader-pyvista-frd-frd-reader',
            'pyvista_frd.FRDReader',
        ),
        ('mesh.npy', '``N/A (read in code)``', 'reader-na-read-in-code', 'N/A (read in code)'),
        ('cubemap/', '``N/A (read in code)``', 'reader-na-read-in-code', 'N/A (read in code)'),
        (None, '``N/A (generated in code)``', 'reader-na', 'N/A (generated in code)'),
    ],
)
def test_dataset_card_reader_field(tmp_path, filename, field, slug, label):
    """Each reader state gets its own card field, facet slug and facet label."""
    if filename is None:
        loader = _DatasetLoader(pv.Sphere)
    else:
        path = tmp_path / filename
        if filename.endswith('/'):
            path.mkdir()
            (path / 'posx.jpg').touch()
        else:
            path.touch()
        loader = _SingleFileDatasetLoader(str(path))

    assert make_tables.DatasetPropsGenerator.generate_reader_type(loader) == field

    classes, labels = make_tables.DatasetCard._generate_facet_classes(loader, pv.examples.examples)
    reader_classes = [cls for cls in classes.split() if cls.startswith('reader-')]
    assert reader_classes == [slug]
    assert labels[slug] == label
    assert class_option(slug) == [slug]


def test_dataset_card_reader_field_mixed(tmp_path):
    """A loader with both kinds of file lists both readers rather than one N/A."""
    paths = [tmp_path / 'mesh.vtp', tmp_path / 'mesh.frd']
    for path in paths:
        path.touch()

    def _files_func():
        return tuple(_SingleFileDatasetLoader(str(path)) for path in paths)

    loader = _MultiFileDatasetLoader(_files_func)
    assert make_tables.DatasetPropsGenerator.generate_reader_type(loader) == (
        ':class:`~pyvista.core.utilities.reader.XMLPolyDataReader`\n``pyvista_frd.FRDReader``'
    )

    classes, _ = make_tables.DatasetCard._generate_facet_classes(loader, pv.examples.examples)
    reader_classes = [cls for cls in classes.split() if cls.startswith('reader-')]
    assert reader_classes == ['reader-xml-poly-data-reader', 'reader-pyvista-frd-frd-reader']
