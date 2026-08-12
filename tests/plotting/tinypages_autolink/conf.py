"""Minimal Sphinx site for testing :mod:`pyvista.ext._autolink` in isolation.

Deliberately kept separate from ``tests/plotting/tinypages`` -- that site's
own tests (``test_tinypages``) assert an *exact* set of generated
``pyvista_plot_directive`` output files, including, for the default
(non-parallel) case, filenames derived from a counter that increments
globally across every ``.. pyvista-plot::`` block in the whole build, in
document-processing order. Adding another page with its own such blocks
there would shift that counter for every page processed after it, breaking
those assertions for reasons unrelated to what they're actually testing.
Building this as its own tiny site avoids that coupling entirely.
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

# Without this, numpydoc auto-appends a Methods/Attributes autosummary table to
# every autoclass'd class, expecting toctree-generated stub pages for each member.
# ``autolink_samples`` documents members individually with .. automethod:: instead,
# so that table has nothing to link to and just warns. Matches doc/source/conf.py.
numpydoc_show_class_members = False

# -- pyvista configuration ------------------------------------------------
pv.BUILDING_GALLERY = True

# Opt-in, like sphinx-gallery's own backreferences_dir -- this site exists to test it.
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
