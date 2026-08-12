"""Minimal Sphinx project for building docs pages via pyvista.ext._autoenum."""

from __future__ import annotations

from pathlib import Path

import pyvista as pv
from pyvista.ext._autoenum import instance_property_names
from pyvista.ext._autoenum import metaclass_property_names

project = 'tinypages_autoenum'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'pyvista.ext._autoenum',
]
autosummary_generate = True

# Reuse the real templates so this exercises what actually ships, not a copy of it. Resolved
# via the installed pyvista package, not `__file__`: this file gets copied to a tmp dir
# before each build (see test_autoenum.py), where a `__file__`-relative path would be wrong.
_repo_root = Path(pv.__file__).resolve().parents[1]
templates_path = [str(_repo_root / 'doc' / 'source' / '_templates')]

autosummary_context = {
    'instance_property_names': instance_property_names,
    'metaclass_property_names': metaclass_property_names,
    'skipmethods': [],
}
