"""Security regression tests: PyVista refuses pickle as a file format.

These tests pin the refusal contract so that ``.pkl`` / ``.pickle``
support cannot silently re-open through ``pv.read``, ``DataObject.save``,
or the top-level ``pv.read_pickle`` / ``pv.save_pickle`` API shims.

Background: pickle is a Python serialization protocol, not a mesh file
format. Unpickling untrusted data is arbitrary code execution (CWE-502).
The in-memory pickle protocol via ``__getstate__`` / ``__setstate__``
(used by ``multiprocessing``, ``dask``, ``joblib``) is unaffected and
covered by ``tests/core/test_dataobject.py``.

Tracks:
- ``security-audit/INVENTORY.md`` P-1a (remote-URI pickle RCE)
- ``security-audit/INVENTORY.md`` P-1b (local-pickle footgun)
- ``security-audit/INVENTORY.md`` T-1 (security regression suite)
"""

from __future__ import annotations

import pickle
from unittest.mock import patch

import pytest

import pyvista as pv

pytestmark = pytest.mark.security

_REFUSAL_MATCH = 'pickle is a Python serialization protocol, not a mesh'

# Minimal valid pickle stream that decodes to ``None``. Used to prove
# refusal happens *before* any ``pickle.load`` call.
_PICKLE_NONE = b'\x80\x04N.'


@pytest.mark.parametrize('ext', ['.pkl', '.pickle', '.PKL', '.Pickle'])
def test_local_pv_read_refuses_pickle(tmp_path, ext):
    """``pv.read('local.pkl')`` must refuse — covers P-1b."""
    p = tmp_path / f'x{ext}'
    p.write_bytes(_PICKLE_NONE)
    with pytest.raises(ValueError, match=_REFUSAL_MATCH):
        pv.read(p)


@pytest.mark.parametrize('ext', ['.pkl', '.pickle'])
def test_dataobject_save_refuses_pickle(sphere, tmp_path, ext):
    """``DataObject.save('x.pkl')`` must refuse — covers P-1b symmetry."""
    target = tmp_path / f'x{ext}'
    with pytest.raises(ValueError, match=_REFUSAL_MATCH):
        sphere.save(target)
    assert not target.exists(), 'save must not produce a file when refusing'


@pytest.mark.parametrize(
    'uri',
    [
        'https://attacker.example/evil.pkl',
        'http://attacker.example/evil.pickle',
        'https://attacker.example/EVIL.PKL',
        's3://attacker-bucket/evil.pkl',
    ],
)
def test_remote_pickle_uri_refused_before_download(uri):
    """Remote ``.pkl`` URIs must refuse before any network call — covers P-1a."""
    downloaded = False

    def fake_retrieve(*_args, **_kwargs):
        nonlocal downloaded
        downloaded = True
        return '/tmp/should-not-be-used'

    with patch('pooch.retrieve', side_effect=fake_retrieve):
        with pytest.raises(ValueError, match=_REFUSAL_MATCH):
            pv.read(uri)

    assert downloaded is False, 'remote pickle URI must refuse before download'


def test_force_ext_pickle_refused(tmp_path):
    """``force_ext='.pkl'`` must not bypass the refusal."""
    p = tmp_path / 'innocuous.vtp'
    p.write_bytes(_PICKLE_NONE)
    with pytest.raises(ValueError, match=_REFUSAL_MATCH):
        pv.read(p, force_ext='.pkl')


def test_top_level_read_pickle_stub_refuses(sphere):
    """``pv.read_pickle`` remains importable for back-compat but refuses."""
    with pytest.raises(ValueError, match=_REFUSAL_MATCH):
        pv.read_pickle('anything.pkl')
    with pytest.raises(ValueError, match=_REFUSAL_MATCH):
        pv.save_pickle('anything.pkl', sphere)


def test_in_memory_pickle_protocol_still_works(sphere):
    """The in-memory pickle protocol (multiprocessing/dask) must NOT break.

    This is the legitimate use of pickle — only the file-format API was
    removed. If this ever fails, the removal went too far.
    """
    if pv.vtk_version_info < (9, 3):
        pytest.skip('VTK < 9.3 has limited pickle support.')
    unpickled = pickle.loads(pickle.dumps(sphere))
    assert unpickled == sphere
