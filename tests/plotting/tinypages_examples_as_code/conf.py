from __future__ import annotations

import os
from pathlib import Path
import re
import sys

import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent))

extensions = [
    'numpydoc',
    'pyvista.ext.plot_directive',
    'pyvista.ext.viewer_directive',
    'pyvista.ext.examples_as_code',
    'sphinx.ext.autodoc',
    'sphinx_design',
]

root_doc = 'index'
project = 'tinypages_examples_as_code'
exclude_patterns = ['_build']

pv.BUILDING_GALLERY = True
pv.OFF_SCREEN = True

numpydoc_show_class_members = False

plot_setup = plot_cleanup = 'import pyvista as pv'

# Same numpydoc "Examples" -> pyvista-plot auto-wrap monkeypatch used by the
# real tinypages/conf.py, so docstring behavior matches the real docs build.
from numpydoc.docscrape_sphinx import SphinxDocString

IMPORT_PYVISTA_RE = r'\b(import +pyvista|from +pyvista +import)\b'


def _str_examples(self):
    examples_str = '\n'.join(self['Examples'])
    if re.search(IMPORT_PYVISTA_RE, examples_str) and 'pyvista-plot::' not in examples_str:
        out = []
        out += self._str_header('Examples')
        out += ['.. pyvista-plot::', '']
        out += self._str_indent(self['Examples'])
        out += ['']
        return out
    return self._str_section('Examples')


SphinxDocString._str_examples = _str_examples

pyvista_plot_use_counter = os.environ.get('PYVISTA_PLOT_USE_COUNTER', 'true').lower() == 'true'
