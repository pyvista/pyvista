"""Minimal Sphinx site for testing ``pyvista.ext._autolink`` in isolation.

Kept separate from ``tests/plotting/tinypages``: that site's
``test_tinypages`` asserts an exact set of generated
``pyvista_plot_directive`` output filenames, keyed in part on a counter
that increments globally, in document order, across the whole build.
Adding pages here would shift those filenames.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import pyvista as pv

sys.path.append(str(Path(__file__).parent))

# -- General configuration ------------------------------------------------

source_suffix = '.rst'
root_doc = 'index'
project = 'tinypages-autolink'
exclude_patterns = ['_build']
pygments_style = 'sphinx'

extensions = [
    'numpydoc',
    'pyvista.ext.plot_directive',
]

# Otherwise numpydoc appends an autosummary Methods table to autoclass'd classes,
# expecting toctree-generated stub pages that don't exist here.
numpydoc_show_class_members = False

# -- pyvista configuration ------------------------------------------------
pv.BUILDING_GALLERY = True
pyvista_plot_autolink = True

# -- .. pyvista-plot:: directive, wrapping numpydoc's Examples sections ---
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
    else:
        return self._str_section('Examples')


SphinxDocString._str_examples = _str_examples
