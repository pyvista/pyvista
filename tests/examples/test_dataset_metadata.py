from __future__ import annotations

from pathlib import Path

import pytest

from pyvista.examples._dataset_metadata import ExampleMetadata
from pyvista.examples._dataset_metadata import License
from pyvista.examples._dataset_metadata import _build_index
from pyvista.examples._dataset_metadata import _license_terms
from pyvista.examples._dataset_metadata import _load_toml
from pyvista.examples._dataset_metadata import _matches

DOCUMENT = """
schema_version = 1

[license."CC-BY-4.0"]
title = "Creative Commons Attribution 4.0 International"
url = "https://creativecommons.org/licenses/by/4.0/"
commercial_use = true
attribution_required = true
share_alike = false
file = "LICENSES/CC-BY-4.0.txt"

[license."CC-BY-SA-3.0"]
title = "Creative Commons Attribution Share Alike 3.0 Unported"
url = "https://creativecommons.org/licenses/by-sa/3.0/"
commercial_use = true
attribution_required = true
share_alike = true
file = "LICENSES/CC-BY-SA-3.0.txt"

[license."LicenseRef-Unknown"]
title = "Undetermined license"
url = "https://example.org/unknown"
commercial_use = false
attribution_required = false
share_alike = false
file = "LICENSES/LicenseRef-Unknown.txt"

[collection."thingiverse"]
title = "Thingiverse"
url = "https://www.thingiverse.com/"
description = "User-contributed models."

[[dataset]]
name = "shark"
title = "Shark"
description = "A shark."
path = ["shark/**"]
SPDX-License-Identifier = "CC-BY-SA-3.0"
SPDX-FileCopyrightText = ["2013 someone"]
provenance = "inferred"
origin_url = "https://www.thingiverse.com/thing:1"
collection = "thingiverse"
authors = ["Someone"]
attribution = "Shark by Someone."
modified = true
modification = "Converted to STL."
notes = "The page states no version."
references = [{ citation = "A paper.", doi = "10.1/2" }]

[[dataset]]
name = "both"
title = "Two licences"
description = "Covered by two licences at once."
path = ["both.vtk"]
SPDX-License-Identifier = "CC-BY-4.0 AND LicenseRef-Unknown"
provenance = "verified"
origin_url = "https://example.org/both"

[[dataset]]
name = "plain"
title = "Plain"
description = "One file, one licence."
path = ["plain.vtk", "nested/*.vtk"]
SPDX-License-Identifier = "CC-BY-4.0"
provenance = "verified"
origin_url = "https://example.org/plain"
attribution = "Plain by Someone."
"""


@pytest.fixture
def index():
    return _build_index(_load_toml(DOCUMENT.encode()))


@pytest.mark.parametrize(
    ('pattern', 'path', 'expected'),
    [
        ('bunny.ply', 'bunny.ply', True),
        ('bunny.ply', 'other.ply', False),
        # `*` stops at a separator and `**` crosses one.
        ('skybox/*', 'skybox/a.jpg', True),
        ('skybox/*', 'skybox/nested/a.jpg', False),
        ('skybox/**', 'skybox/nested/a.jpg', True),
        ('skybox/**', 'skybox/a.jpg', True),
        ('skybox/**', 'skyboxes/a.jpg', False),
        ('*', 'a.vtk', True),
        ('*', 'dir/a.vtk', False),
        ('skybox/*.jpg', 'skybox/a.jpg', True),
        ('skybox/*.jpg', 'skybox/a.png', False),
        ('sim_?.vtu', 'sim_1.vtu', True),
        ('sim_?.vtu', 'sim_12.vtu', False),
        # `**` crosses separators where `*` stops at one.
        ('froggy/**', 'froggy/frog.mhd', True),
        ('froggy/**', 'froggy/sub/frog.mhd', True),
        ('froggy/*', 'froggy/sub/frog.mhd', False),
        ('**/frog.mhd', 'a/b/frog.mhd', True),
        # A dot in the pattern is literal, not a wildcard.
        ('a.vtk', 'axvtk', False),
    ],
)
def test_matches(pattern, path, expected):
    assert _matches(pattern, path) is expected


@pytest.mark.parametrize(
    ('expression', 'expected'),
    [
        ('MIT', ['MIT']),
        ('CC-BY-4.0 AND LicenseRef-Unknown', ['CC-BY-4.0', 'LicenseRef-Unknown']),
        ('MIT OR Apache-2.0', ['MIT', 'Apache-2.0']),
        # The exception after `WITH` is not itself a licence.
        ('Apache-2.0 WITH LLVM-exception', ['Apache-2.0']),
        ('GPL-2.0-only WITH Classpath-exception-2.0 OR MIT', ['GPL-2.0-only', 'MIT']),
        ('(MIT AND Apache-2.0)', ['MIT', 'Apache-2.0']),
    ],
)
def test_license_terms(expression, expected):
    assert _license_terms(expression) == expected


def test_build_index_reads_every_field(index):
    entry = index.match('shark/shark.stl')
    assert entry == ExampleMetadata(
        name='shark',
        title='Shark',
        description='A shark.',
        license_expression='CC-BY-SA-3.0',
        licenses=(index.licenses['CC-BY-SA-3.0'],),
        provenance='inferred',
        paths=('shark/**',),
        origin_url='https://www.thingiverse.com/thing:1',
        collection='thingiverse',
        authors=('Someone',),
        copyright=('2013 someone',),
        attribution='Shark by Someone.',
        modified=True,
        modification='Converted to STL.',
        notes='The page states no version.',
        references=entry.references,
    )
    assert entry.references[0].citation == 'A paper.'
    assert entry.references[0].doi == '10.1/2'
    assert entry.references[0].url is None


def test_license_table_is_resolved(index):
    assert index.licenses['CC-BY-4.0'] == License(
        spdx_id='CC-BY-4.0',
        title='Creative Commons Attribution 4.0 International',
        url='https://creativecommons.org/licenses/by/4.0/',
        commercial_use=True,
        attribution_required=True,
        share_alike=False,
    )


def test_obligations_combine_across_an_expression(index):
    plain = index.match('plain.vtk')
    assert plain.commercial_use
    assert plain.attribution_required
    assert not plain.share_alike

    shark = index.match('shark/shark.stl')
    assert shark.share_alike

    # An undetermined licence makes the whole expression unusable commercially.
    both = index.match('both.vtk')
    assert not both.commercial_use


def test_match_uses_every_pattern(index):
    assert index.match('nested/a.vtk').name == 'plain'
    assert index.match('nested/deep/a.vtk') is None


def test_match_returns_none_for_unclaimed_path(index):
    assert index.match('not/in/the/table.vtk') is None


def test_schema_version_must_match():
    document = _load_toml(b'schema_version = 99')
    match = 'declares schema_version 99'
    with pytest.raises(ValueError, match=match):
        _build_index(document)


def test_metadata_for_source_names(index, monkeypatch):
    from pyvista.examples import _dataset_metadata

    monkeypatch.setattr(_dataset_metadata, '_metadata_index', lambda: index)

    assert _dataset_metadata._metadata_for_source_names(['shark/a.stl']).name == 'shark'
    # Several files of one example resolve to the single entry claiming them.
    assert _dataset_metadata._metadata_for_source_names(['shark/a.stl', 'shark/b.stl']).name == (
        'shark'
    )
    assert _dataset_metadata._metadata_for_source_names(['nothing.vtk']) is None

    match = 'span more than one dataset entry: plain, shark'
    with pytest.raises(ValueError, match=match):
        _dataset_metadata._metadata_for_source_names(['shark/a.stl', 'plain.vtk'])


def test_override_reads_a_local_file(tmp_path, monkeypatch):
    from pyvista.examples import _dataset_metadata

    path = tmp_path / 'DATASETS.toml'
    path.write_text(DOCUMENT)
    monkeypatch.setenv(_dataset_metadata._METADATA_VARNAME, str(path))
    _dataset_metadata._metadata_index.cache_clear()

    def fail() -> str:
        msg = 'the override must not download'
        raise AssertionError(msg)

    monkeypatch.setattr(_dataset_metadata, '_download_metadata_file', fail)
    assert _dataset_metadata._metadata_index().match('plain.vtk').name == 'plain'
    _dataset_metadata._metadata_index.cache_clear()


def test_override_reports_a_missing_file(tmp_path, monkeypatch):
    from pyvista.examples import _dataset_metadata

    missing = tmp_path / 'nope.toml'
    monkeypatch.setenv(_dataset_metadata._METADATA_VARNAME, str(missing))
    _dataset_metadata._metadata_index.cache_clear()
    with pytest.raises(FileNotFoundError, match='which is not a file'):
        _dataset_metadata._metadata_index()
    _dataset_metadata._metadata_index.cache_clear()


def test_example_exposes_the_record_directly(monkeypatch, index):
    from pyvista import examples
    from pyvista.examples import _get_example

    monkeypatch.setattr(
        _get_example, '_metadata_for_source_names', lambda _names: index.match('shark/a.stl')
    )
    example = examples.get_example('sphere', download=False)
    assert example.license == 'CC-BY-SA-3.0'
    assert example.share_alike is True
    assert example.commercial_use is True
    assert example.provenance == 'inferred'
    assert example.origin_url == 'https://www.thingiverse.com/thing:1'
    assert example.authors == ('Someone',)
    assert example.copyright == ('2013 someone',)
    assert example.modified is True
    assert example.references[0].doi == '10.1/2'
    assert not hasattr(example, 'metadata')


def test_example_without_a_record_is_empty(monkeypatch):
    from pyvista import examples
    from pyvista.examples import _get_example

    monkeypatch.setattr(_get_example, '_metadata_for_source_names', lambda _names: None)
    example = examples.get_example('sphere', download=False)
    assert example.license is None
    assert example.commercial_use is None
    assert example.licenses == ()
    assert example.authors == ()
    assert example.modified is False


def test_bundled_table_covers_every_packaged_file():
    import pyvista as pv
    from pyvista.examples._dataset_metadata import _bundled_index
    from pyvista.examples._dataset_metadata import _matches

    directory = Path(pv.examples.__file__).parent
    packaged = sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix not in {'.py', '.pyi', '.toml', '.typed'}
    )
    entries = _bundled_index().entries
    for name in packaged:
        owners = [e.name for e in entries if any(_matches(p, name) for p in e.paths)]
        assert len(owners) == 1, f'{name} is claimed by {owners}'


def test_metadata_base_url_sits_beside_the_data_directory():
    """The table is published next to `Data/`, not inside it."""
    from pyvista.examples import _dataset_metadata
    from pyvista.examples.downloads import SOURCE

    base = _dataset_metadata._metadata_base_url()

    assert SOURCE.endswith('Data/')
    assert base == SOURCE.removesuffix('Data/')
    assert not base.endswith('Data/')


def test_download_metadata_file_fetches_the_published_table(monkeypatch, tmp_path):
    """Without an override the table is fetched from where pyvista/data publishes it."""
    import pooch

    from pyvista.examples import _dataset_metadata

    fetched = tmp_path / _dataset_metadata._METADATA_FILENAME
    fetched.write_text(DOCUMENT)
    created = {}

    def fake_create(**kwargs):
        created.update(kwargs)
        return 'fetcher'

    def fake_locked_fetch(fetcher, filename, downloader=None):  # noqa: ARG001
        assert fetcher == 'fetcher'
        assert filename == _dataset_metadata._METADATA_FILENAME
        return str(fetched)

    monkeypatch.setattr(pooch, 'create', fake_create)
    monkeypatch.setattr('pyvista.examples.downloads._locked_fetch', fake_locked_fetch)

    assert _dataset_metadata._download_metadata_file() == str(fetched)
    assert created['base_url'] == _dataset_metadata._metadata_base_url()
    assert _dataset_metadata._METADATA_FILENAME in created['registry']


def test_metadata_index_reads_the_downloaded_table(monkeypatch, tmp_path):
    """With no override set, the index is built from the downloaded file."""
    from pyvista.examples import _dataset_metadata

    path = tmp_path / 'DATASETS.toml'
    path.write_text(DOCUMENT)
    monkeypatch.delenv(_dataset_metadata._METADATA_VARNAME, raising=False)
    monkeypatch.setattr(_dataset_metadata, '_download_metadata_file', lambda: str(path))
    _dataset_metadata._metadata_index.cache_clear()

    assert _dataset_metadata._metadata_index().match('plain.vtk').name == 'plain'

    _dataset_metadata._metadata_index.cache_clear()
