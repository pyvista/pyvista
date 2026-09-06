"""Provenance and licensing metadata, read from ``pyvista/data``'s ``DATASETS.toml``."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import functools
import re
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

SCHEMA_VERSION = 1

_METADATA_FILENAME = 'DATASETS.toml'
_METADATA_VARNAME = 'PYVISTA_DATASETS_TOML'

Provenance = Literal['verified', 'inferred', 'unknown']


def _load_toml(data: bytes) -> dict[str, Any]:
    """Parse TOML bytes, using ``tomli`` before Python 3.11."""
    try:
        import tomllib  # noqa: PLC0415
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]  # noqa: PLC0415
    return tomllib.loads(data.decode('utf-8'))


@dataclass(frozen=True)
class License:
    """A licence one or more example datasets are distributed under.

    .. versionadded:: 0.49

    """

    spdx_id: str
    """SPDX identifier, or a ``LicenseRef-`` identifier for terms SPDX does not list."""

    title: str
    """Full name of the licence, such as ``'Creative Commons Attribution 4.0 International'``."""

    url: str
    """Canonical URL of the licence text."""

    commercial_use: bool
    """Whether the licence permits use in a product for sale."""

    attribution_required: bool
    """Whether the licence requires the work to be credited."""

    share_alike: bool
    """Whether derivative works must carry the same licence."""


@dataclass(frozen=True)
class Reference:
    """A work an example dataset asks to be cited.

    .. versionadded:: 0.49

    """

    citation: str
    """Full citation text."""

    doi: str | None = None
    """Digital Object Identifier, without the ``https://doi.org/`` prefix."""

    url: str | None = None
    """URL of the work, when it has no DOI."""


@dataclass(frozen=True)
class ExampleMetadata:
    """Where an example dataset came from and how it may be used.

    .. versionadded:: 0.49

    Access this through :attr:`~pyvista.examples.Example.metadata`; this class is not
    meant to be constructed directly. It mirrors one ``[[dataset]]`` block of
    ``DATASETS.toml`` in the `pyvista/data <https://github.com/pyvista/data>`_
    repository, which is the source of truth for every field here.

    Examples
    --------
    >>> from pyvista import examples
    >>> shark = examples.get_example('grey_nurse_shark')  # doctest:+SKIP
    >>> shark.metadata.license_expression  # doctest:+SKIP
    'CC-BY-SA-3.0'

    A licence can attach obligations that outlive the download.

    >>> shark.metadata.share_alike  # doctest:+SKIP
    True

    """

    name: str
    """Name of the dataset entry in ``DATASETS.toml``, which may differ from the example name."""

    title: str
    """Short human-readable name of the data."""

    description: str
    """What the data is, in one or two sentences."""

    license_expression: str
    """SPDX licence expression, such as ``'CC-BY-4.0'`` or ``'MIT AND CC0-1.0'``."""

    licenses: tuple[License, ...]
    """Every licence the expression names, resolved to its full terms."""

    provenance: Provenance
    """Confidence in the origin, not the licence: ``'verified'``, ``'inferred'``, ``'unknown'``."""

    paths: tuple[str, ...]
    """Path patterns the entry claims, relative to the ``Data/`` directory."""

    source_url: str | None = None
    """Where the data came from."""

    source_title: str | None = None
    """Human-readable name of the source."""

    collection: str | None = None
    """Upstream collection the data belongs to, when several datasets share one."""

    authors: tuple[str, ...] = ()
    """Who made the data."""

    copyright: tuple[str, ...] = ()
    """Copyright notices, in ``SPDX-FileCopyrightText`` form."""

    attribution: str | None = None
    """Credit line the licence requires, when it requires one."""

    redistributed_from: str | None = None
    """Intermediate redistributor the file reached this project through."""

    modified: bool = False
    """Whether the file differs from what the source published."""

    modification: str | None = None
    """What was done to the file after it left its source."""

    notes: str | None = None
    """What was and was not established about the origin and the terms."""

    references: tuple[Reference, ...] = ()
    """Works the dataset asks to be cited."""

    @property
    def commercial_use(self) -> bool:
        """Return whether every licence named permits use in a product for sale.

        Undetermined terms count as not permitted.

        Returns
        -------
        bool
            ``True`` when the data is cleared for commercial use.

        """
        return all(licence.commercial_use for licence in self.licenses)

    @property
    def attribution_required(self) -> bool:
        """Return whether any licence named requires the work to be credited.

        Returns
        -------
        bool
            ``True`` when credit is required.

        """
        return any(licence.attribution_required for licence in self.licenses)

    @property
    def share_alike(self) -> bool:
        """Return whether any licence named requires derivatives to carry it too.

        Returns
        -------
        bool
            ``True`` when the ShareAlike obligation propagates.

        """
        return any(licence.share_alike for licence in self.licenses)


@dataclass(frozen=True)
class _MetadataIndex:
    """Every dataset entry, with the licence table needed to resolve them."""

    licenses: Mapping[str, License]
    collections: Mapping[str, Mapping[str, str]]
    entries: tuple[ExampleMetadata, ...]
    _by_path: dict[str, ExampleMetadata] = field(default_factory=dict, repr=False)

    def match(self, path: str) -> ExampleMetadata | None:
        """Return the entry claiming a path relative to ``Data/``.

        Parameters
        ----------
        path : str
            Path of one file, relative to the ``Data/`` directory.

        Returns
        -------
        ExampleMetadata | None
            The entry claiming the path, or ``None`` when no entry does.

        """
        if path in self._by_path:
            return self._by_path[path]
        for entry in self.entries:
            if any(_matches(pattern, path) for pattern in entry.paths):
                self._by_path[path] = entry
                return entry
        return None


def _pattern_regex(pattern: str) -> re.Pattern[str]:
    """Compile a path pattern, where ``*`` stops at a separator and ``**`` crosses one."""
    out: list[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == '*':
            if pattern[index + 1 : index + 2] == '*':
                out.append('.*')
                index += 2
                continue
            out.append('[^/]*')
        elif char == '?':
            out.append('[^/]')
        else:
            out.append(re.escape(char))
        index += 1
    return re.compile('^' + ''.join(out) + '$')


@functools.cache
def _compiled(pattern: str) -> re.Pattern[str]:
    """Return the compiled form of a path pattern."""
    return _pattern_regex(pattern)


def _matches(pattern: str, path: str) -> bool:
    """Match a path pattern against a path relative to ``Data/``."""
    if pattern.endswith('/**'):
        return path.startswith(pattern[:-2])
    return bool(_compiled(pattern).match(path))


def _license_terms(expression: str) -> list[str]:
    """Split an SPDX expression into the licence identifiers it names."""
    tokens = [token for token in re.split(r'[()\s]+', expression) if token]
    terms: list[str] = []
    skip = False
    for token in tokens:
        if token.upper() == 'WITH':
            skip = True
        elif token.upper() in {'AND', 'OR'} or skip:
            skip = False
        else:
            terms.append(token)
    return terms


def _build_index(document: Mapping[str, Any]) -> _MetadataIndex:
    """Turn a parsed ``DATASETS.toml`` document into a lookup index."""
    version = document.get('schema_version')
    if version != SCHEMA_VERSION:
        msg = (
            f'DATASETS.toml declares schema_version {version!r}, but this version of '
            f'PyVista understands {SCHEMA_VERSION}. Upgrade PyVista to read it.'
        )
        raise ValueError(msg)

    licenses = {
        key: License(
            spdx_id=key,
            title=table['title'],
            url=table['url'],
            commercial_use=table['commercial_use'],
            attribution_required=table['attribution_required'],
            share_alike=table['share_alike'],
        )
        for key, table in document.get('license', {}).items()
    }
    entries = tuple(
        ExampleMetadata(
            name=entry['name'],
            title=entry['title'],
            description=entry['description'],
            license_expression=entry['SPDX-License-Identifier'],
            licenses=tuple(
                licenses[term]
                for term in _license_terms(entry['SPDX-License-Identifier'])
                if term in licenses
            ),
            provenance=entry['provenance'],
            paths=tuple(entry['path']),
            source_url=entry.get('source_url'),
            source_title=entry.get('source_title'),
            collection=entry.get('collection'),
            authors=tuple(entry.get('authors', ())),
            copyright=tuple(entry.get('SPDX-FileCopyrightText', ())),
            attribution=entry.get('attribution'),
            redistributed_from=entry.get('redistributed_from'),
            modified=entry.get('modified', False),
            modification=entry.get('modification'),
            notes=entry.get('notes'),
            references=tuple(
                Reference(
                    citation=reference['citation'],
                    doi=reference.get('doi'),
                    url=reference.get('url'),
                )
                for reference in entry.get('references', ())
            ),
        )
        for entry in document.get('dataset', ())
    )
    return _MetadataIndex(
        licenses=licenses,
        collections=document.get('collection', {}),
        entries=entries,
    )


def _metadata_base_url() -> str:
    """Return the location of ``DATASETS.toml``, alongside the ``Data/`` directory."""
    from pyvista.examples.downloads import SOURCE  # noqa: PLC0415

    return SOURCE.removesuffix('Data/')


def _download_metadata_file() -> str:
    """Download ``DATASETS.toml`` and return its local path."""
    import pooch  # noqa: PLC0415

    from pyvista.examples.downloads import _FILE_CACHE  # noqa: PLC0415
    from pyvista.examples.downloads import USER_DATA_PATH  # noqa: PLC0415
    from pyvista.examples.downloads import _file_copier  # noqa: PLC0415
    from pyvista.examples.downloads import _locked_fetch  # noqa: PLC0415

    fetcher = pooch.create(  # type: ignore[attr-defined]
        path=USER_DATA_PATH,
        base_url=_metadata_base_url(),
        registry={_METADATA_FILENAME: None},
        retry_if_failed=3,
    )
    return _locked_fetch(
        fetcher, _METADATA_FILENAME, downloader=_file_copier if _FILE_CACHE else None
    )


@functools.lru_cache(maxsize=1)
def _bundled_index() -> _MetadataIndex:
    """Return the index for the example files that ship inside this package."""
    from pathlib import Path  # noqa: PLC0415

    path = Path(__file__).parent / _METADATA_FILENAME
    return _build_index(_load_toml(path.read_bytes()))


@functools.lru_cache(maxsize=1)
def _metadata_index() -> _MetadataIndex:
    """Return the dataset metadata index, reading and parsing it once."""
    import os  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    override = os.environ.get(_METADATA_VARNAME)
    if override:
        path = Path(override)
        if not path.is_file():
            msg = f'{_METADATA_VARNAME} is set to {override!r}, which is not a file.'
            raise FileNotFoundError(msg)
    else:
        path = Path(_download_metadata_file())
    return _build_index(_load_toml(path.read_bytes()))


def _metadata_for_source_names(source_names: Iterable[str]) -> ExampleMetadata | None:
    """Return the entry claiming these files, or ``None`` when they are not in ``pyvista/data``."""
    bundled = _bundled_index()
    resolved = [bundled.match(name) or _metadata_index().match(name) for name in source_names]
    matched = {entry.name: entry for entry in resolved if entry is not None}
    if not matched:
        return None
    if len(matched) > 1:
        spanned = ', '.join(sorted(matched))
        msg = f'Example files span more than one dataset entry: {spanned}.'
        raise ValueError(msg)
    return next(iter(matched.values()))
