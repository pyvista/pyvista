from __future__ import annotations

from dataclasses import replace
from io import StringIO
import json
import re
from typing import get_args

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
from pyvista.examples._dataset_loader import _SingleFileDownloadableDatasetLoader
from pyvista.examples._get_example import _get_dataset_loader

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


def test_usage_badges_cover_every_usage_value():
    """A `use` value missing from the manifest order is dropped from the filter panel.

    `collectFacetValues` in dataset_gallery_filter.js keeps only the values named in
    `order`, so a slug the cards emit but the manifest omits disappears silently.
    """
    from pyvista.examples._dataset_metadata import Usage

    badges = make_tables.DATASET_GALLERY_USAGE_BADGES
    assert set(get_args(Usage)) == set(badges)
    # The facet slug is derived from the value, so the manifest order must agree.
    assert [make_tables._facet_slugify(usage) for usage in badges] == [
        'unrestricted',
        'attribution',
        'share-alike',
        'non-commercial',
        'undetermined',
    ]
    colours = [colour for _, colour in badges.values()]
    assert len(set(colours)) == len(colours)
    # Solid marks usage and outlined marks provenance; the module badge is not checked.
    assert not any(colour.endswith('-line') for colour in colours)
    assert all(
        colour.endswith('-line')
        for colour in make_tables.DATASET_GALLERY_PROVENANCE_COLORS.values()
    )


@pytest.mark.parametrize(
    ('prose', 'expected'),
    [
        ('plain text', 'plain text'),
        ('a `code` span', 'a ``code`` span'),
        ('already ``literal``', 'already ``literal``'),
        ('a_b and *star*', r'a\_b and \*star\*'),
        ('see https://example.org/a_b', 'see ``https://example.org/a_b``'),
        # A trailing underscore in a URL is a reStructuredText reference otherwise.
        (
            'see https://web.archive.org/web/2024id_/https://x.org/y',
            'see ``https://web.archive.org/web/2024id_/https://x.org/y``',
        ),
        ('(https://example.org/x), next', '(``https://example.org/x``), next'),
        ('ends https://example.org/x.', 'ends ``https://example.org/x``.'),
        ('wiki https://e.org/Foo_(bar) here', 'wiki ``https://e.org/Foo_(bar)`` here'),
    ],
)
def test_rst_from_prose(prose, expected):
    assert make_tables._rst_from_prose(prose) == expected


def test_rst_from_prose_output_parses_as_rst():
    """The escaping exists to keep the docs build green, so parse the result."""
    from docutils.core import publish_doctree

    hostile = 'Trailing https://web.archive.org/web/2020id_/http://x.org/y and a_b *and* `c`.'
    errors = StringIO()
    publish_doctree(
        make_tables._rst_from_prose(hostile),
        settings_overrides={
            'report_level': 2,
            'halt_level': 5,
            'warning_stream': errors,
            'file_insertion_enabled': False,
        },
    )

    assert 'ERROR' not in errors.getvalue()
    assert 'WARNING' not in errors.getvalue()


@pytest.fixture
def metadata():
    """A record covering every field the gallery renders."""
    from pyvista.examples._dataset_metadata import ExampleMetadata
    from pyvista.examples._dataset_metadata import License

    return ExampleMetadata(
        name='thing',
        title='Thing',
        description='A thing.',
        paths=('thing.vtk',),
        license_expression='CC-BY-3.0',
        licenses=(
            License(
                spdx_id='CC-BY-3.0',
                title='Creative Commons Attribution 3.0 Unported',
                url='https://creativecommons.org/licenses/by/3.0/',
                commercial_use=True,
                attribution_required=True,
                share_alike=False,
                text_url='https://example.org/LICENSES/CC-BY-3.0.txt',
            ),
        ),
        provenance='verified',
        origin_url='https://humus.name/',
        origin_title='Humus texture library',
        redistributed_from='https://gitlab.kitware.com/vtk/vtk-examples/-/tree/master/x',
    )


def test_license_field_links_the_text_and_the_issuer(metadata):
    field = make_tables.DatasetPropsGenerator.generate_license_field(metadata)

    assert ':bdg-link-primary:`CC-BY-3.0 <https://example.org/LICENSES/CC-BY-3.0.txt>`' in field
    assert '`Creative Commons Attribution 3.0 Unported <https://creativecommons.org/' in field


def test_usage_badge_links_the_legend(metadata):
    gen = make_tables.DatasetPropsGenerator
    sa = replace(metadata.licenses[0], share_alike=True)

    assert gen.generate_usage_badge(metadata) == (
        ':bdg-ref-info:`Credit required <dataset_gallery_usage>`'
    )
    assert gen.generate_usage_badge(replace(metadata, licenses=(sa,))) == (
        ':bdg-ref-warning:`Share alike <dataset_gallery_usage>`'
    )
    assert gen.generate_provenance_field(metadata) == ':bdg-success-line:`verified`'


def test_usage_badge_marks_an_uncatalogued_file_but_not_generated_data():
    gen = make_tables.DatasetPropsGenerator

    assert gen.generate_usage_badge(None, _SingleFileDownloadableDatasetLoader('mesh.vtp')) == (
        ':bdg-ref-muted-line:`Not recorded <dataset_gallery_usage>`'
    )
    assert gen.generate_usage_badge(None, _DatasetLoader(pv.Sphere)) == ''
    assert gen.generate_usage_badge(None) == ''


def test_card_header_carries_the_module_and_usage_badges(monkeypatch, metadata, tmp_path):
    path = tmp_path / 'mesh.vtp'
    path.touch()
    loader = _SingleFileDatasetLoader(str(path))
    metadata = replace(metadata, licenses=(replace(metadata.licenses[0], share_alike=True),))
    monkeypatch.setattr(
        make_tables.DatasetPropsGenerator, '_dataset_metadata', staticmethod(lambda _: metadata)
    )
    card = make_tables.DatasetCard(
        'thing', loader, module=pv.examples.downloads, function=pv.examples.download_bunny
    )
    monkeypatch.setattr(card, '_generate_cross_references', lambda *_: '')

    rst = card.generate()

    header = rst.split('^^^')[0]
    assert ':bdg-ref-secondary:`Downloads' in header
    assert ':bdg-ref-warning:`Share alike <dataset_gallery_usage>`' in header
    # The facet slug is what docutils makes of the value, and the manifest lists it.
    assert 'use-share-alike' in header
    assert make_tables.DatasetCardFetcher.FACET_LABELS['use-share-alike'] == 'Share alike'
    footer = rst.split('+++')[1]
    assert '**Usage**' in footer
    assert '**Commercial use**' in footer
    assert '**Attribution required**' in footer
    assert '**Share alike**' in footer
    assert re.findall(r'^\s*(Yes|No)$', footer, re.MULTILINE) == ['Yes', 'Yes', 'Yes']
    plain = make_tables.DatasetCard._create_footer_block('', replace(metadata, licenses=()))
    assert re.findall(r'^\s*(Yes|No)$', plain, re.MULTILINE) == ['No', 'Yes', 'No']


def test_filter_manifest_lists_the_usage_slugs_the_cards_emit():
    html = make_tables.DatasetCardFetcher.generate_filter_toolbar()
    start = html.index('>', html.index('id="facet-manifest"')) + 1
    manifest = json.loads(html[start : html.index('</script>', start)])

    assert manifest['order']['use'] == [
        'unrestricted',
        'attribution',
        'share-alike',
        'non-commercial',
        'undetermined',
    ]


def test_unrecorded_facets_tell_a_missing_record_from_generated_data(monkeypatch):
    monkeypatch.setattr(
        make_tables.DatasetPropsGenerator, '_dataset_metadata', staticmethod(lambda _: None)
    )
    packaged, _, _ = _get_dataset_loader(pv.examples.load_ant)
    classes, labels = make_tables.DatasetCard._generate_facet_classes(
        packaged, pv.examples.examples
    )
    assert 'use-na' in classes.split()
    assert labels['use-na'] == labels['license-na'] == 'N/A (not recorded)'

    _, labels = make_tables.DatasetCard._generate_facet_classes(
        _DatasetLoader(pv.Sphere), pv.examples.examples
    )
    assert labels['use-na'] == 'N/A (no file)'


def test_license_field_falls_back_to_the_issuer_page(metadata):
    lic = replace(metadata.licenses[0], text_url=None)
    field = make_tables.DatasetPropsGenerator.generate_license_field(
        replace(metadata, licenses=(lic,))
    )

    assert ':bdg-link-primary:`CC-BY-3.0 <https://creativecommons.org/licenses/by/3.0/>`' in field


def test_redistributor_shows_the_host_not_the_whole_url(metadata):
    field = make_tables.DatasetPropsGenerator.generate_redistributor_field(metadata)
    bare = make_tables.DatasetPropsGenerator.generate_redistributor_field(
        replace(metadata, redistributed_from='a private archive')
    )
    missing = make_tables.DatasetPropsGenerator.generate_redistributor_field(
        replace(metadata, redistributed_from=None)
    )

    assert field.startswith('`gitlab.kitware.com <https://gitlab.kitware.com/')
    assert bare == '``a private archive``'
    assert missing is None


def test_origin_field_falls_back_to_a_literal_for_a_non_url(metadata):
    linked = make_tables.DatasetPropsGenerator.generate_origin_field(metadata)
    bare = make_tables.DatasetPropsGenerator.generate_origin_field(
        replace(metadata, origin_url='OnScale Solve')
    )
    missing = make_tables.DatasetPropsGenerator.generate_origin_field(
        replace(metadata, origin_url=None)
    )

    assert linked == '`Humus texture library <https://humus.name/>`__'
    assert bare == '``Humus texture library``'
    assert missing is None


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
