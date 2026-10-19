"""Known VTK backends and where their manifests come from."""

from __future__ import annotations

from pathlib import Path

from pyvista.core._vtk_utilities import _MAX_SUPPORTED_VTK_VERSION
from pyvista.core._vtk_utilities import VersionInfo

from .manifest import MANIFEST_FILENAME
from .manifest import Manifest
from .manifest import ManifestError
from .manifest import load_manifest

MANIFESTS_DIR = Path(__file__).parent / 'manifests'

#: ``default_ref`` of each shipped manifest: the newest release PyVista supports.
DEFAULT_REFS: dict[str, str] = {
    'vtk': f'v{VersionInfo._format(_MAX_SUPPORTED_VTK_VERSION)}',
}

#: Git URL of each backend whose manifest lives in its own tree.
BACKENDS: dict[str, str] = {
    'cvista': 'https://github.com/pyvista/cvista.git',
}


def known_backends() -> list[str]:
    """Return every backend name the command knows without ``--url`` or ``--source``.

    Returns
    -------
    list[str]
        Sorted names from the registry and the shipped manifests.

    """
    return sorted(set(BACKENDS) | {p.stem for p in MANIFESTS_DIR.glob('*.toml')})


def shipped_manifest(backend: str) -> Path | None:
    """Return the manifest PyVista ships for a backend.

    Parameters
    ----------
    backend : str
        Backend name.

    Returns
    -------
    Path | None
        Manifest path, or ``None`` when PyVista ships none for it.

    """
    path = MANIFESTS_DIR / f'{backend}.toml'
    return path if path.is_file() else None


def find_manifest(backend: str, source: Path | None) -> Manifest:
    """Load a backend's manifest: the checkout's own, else the shipped one.

    Parameters
    ----------
    backend : str
        Backend name.

    source : Path, optional
        Source checkout to look in first.

    Returns
    -------
    Manifest
        Loaded manifest.

    """
    if source is not None:
        in_tree = source / MANIFEST_FILENAME
        if in_tree.is_file():
            return load_manifest(in_tree)
    shipped = shipped_manifest(backend)
    if shipped is not None:
        return load_manifest(shipped, default_ref=DEFAULT_REFS.get(backend))
    where = f'{source / MANIFEST_FILENAME}' if source is not None else 'the checkout'
    msg = (
        f'No {MANIFEST_FILENAME} found in {where} and PyVista ships no manifest for '
        f'{backend!r}. Known backends: {known_backends()}.'
    )
    raise ManifestError(msg)
