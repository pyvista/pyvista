from __future__ import annotations

from dataclasses import replace
from io import StringIO

import cmcrameri
import cmocean
from colorcet import all_original_names
from colorcet import get_aliases
import docutils.nodes
import matplotlib as mpl
import pytest

from doc.source import make_tables

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


def test_usage_facet_order_lists_every_value_the_cards_emit():
    """A `use` value missing from the manifest order is dropped from the filter panel.

    `collectFacetValues` in dataset_gallery_filter.js keeps only the values named in
    `order`, so a slug the cards emit but the manifest omits disappears silently.
    """
    emitted = {
        make_tables._facet_slugify(label)
        for label in (
            'Commercial use',
            'Not for commercial use',
            'ShareAlike',
            'Attribution required',
        )
    }
    listed = set(make_tables.DATASET_GALLERY_USE_ORDER)

    # `N/A (not recorded)` is added with an explicit `na` slug, not through the slugifier.
    assert emitted <= listed


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


def test_usage_and_provenance_badges_do_not_share_a_colour(metadata):
    usage = make_tables.DatasetPropsGenerator.generate_usage_field(metadata)
    provenance = make_tables.DatasetPropsGenerator.generate_provenance_field(metadata)

    sa = replace(metadata.licenses[0], share_alike=True)
    restricted = make_tables.DatasetPropsGenerator.generate_usage_field(
        replace(metadata, licenses=(sa,))
    )

    assert ':bdg-success:`Commercial use`' in usage
    assert ':bdg-info:`Attribution required`' in usage
    assert ':bdg-warning:`ShareAlike`' in restricted
    # Solid marks an obligation, outlined marks confidence; they must stay distinct.
    assert provenance == ':bdg-success-line:`verified`'


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

    assert linked == '`Humus texture library <https://humus.name/>`_'
    assert bare == '``Humus texture library``'
    assert missing is None
