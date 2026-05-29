"""Test the tox plugin hooks in ``toxfile.py``."""

from __future__ import annotations

import pytest

pytest.importorskip('tox_uv')

from packaging.version import Version

import toxfile


def test_get_freezed_requirements_skips_non_pep508_lines() -> None:
    """`uv pip freeze` emits lines that are not PEP 508 requirements.

    Editable installs (``-e file:///path``) are used for integration packages
    installed from source (eg. mne). These have no ``==`` pin and previously
    crashed the freeze check with ``InvalidRequirement``. Regression for #8690.
    """
    lines = [
        'numpy==2.3.5',
        '-e file:///home/runner/work/pyvista/pyvista/.tox/.tmp/mne',
        'pyvista @ file:///home/runner/work/pyvista/pyvista',
        'pyvistaqt @ git+https://github.com/pyvista/pyvistaqt.git@79ac29ea',
        '',
        '   ',
        '# a comment',
        'vtk==9.6.2',
    ]

    parsed = dict(toxfile._get_freezed_requirements(lines))

    assert parsed == {'numpy': Version('2.3.5'), 'vtk': Version('9.6.2')}
    # The editable install must be skipped, not raise and not appear pinned.
    assert 'mne' not in parsed


def test_get_freezed_requirements_normalizes_names() -> None:
    parsed = dict(toxfile._get_freezed_requirements(['Foo_Bar.Baz==1.2.3']))
    assert parsed == {'foo-bar-baz': Version('1.2.3')}
