"""Minimal Sphinx site for testing ``pyvista_plot_autocodelink`` in isolation.

Kept separate from ``tests/plotting/tinypages``: that site's ``test_tinypages``
asserts an exact set of output filenames keyed on a global counter; another
page here would shift them.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING

from docutils import nodes
from jinja2.sandbox import SandboxedEnvironment
from numpydoc.docscrape import NumpyDocString
from sphinx import addnodes

import pyvista as pv

if TYPE_CHECKING:
    from docutils.nodes import Element
    from sphinx.application import Sphinx

sys.path.append(str(Path(__file__).parent))

# -- General configuration ------------------------------------------------

source_suffix = '.rst'
root_doc = 'index'
project = 'tinypages-autocodelink'
exclude_patterns = ['_build']
pygments_style = 'sphinx'

extensions = [
    'numpydoc',
    'pyvista.ext.plot_directive',
]

# Otherwise numpydoc appends an autosummary Methods table with no stub pages to link to.
numpydoc_show_class_members = False

# -- pyvista configuration ------------------------------------------------
pv.BUILDING_GALLERY = True
pyvista_plot_autocodelink = True
autocodelink_autodoc_backrefs = True

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


# -- "See Also" as a real section, not a `.. seealso::` admonition ------------
# Mirrors doc/source/conf.py's own override of the same name: SphinxDocString's own
# _str_see_also always wraps the base rendering in `.. seealso::`, which isn't a real
# docutils section. The un-wrapped base rendering already goes through self._str_header,
# so skipping the wrap turns it into a real heading for free.
def _str_see_also(self, func_role):
    return NumpyDocString._str_see_also(self, func_role)


SphinxDocString._str_see_also = _str_see_also


# -- docstring section order: Parameters, ..., Examples, See Also -------------
# Mirrors doc/source/conf.py's own override of the same name: moves "See Also" to the
# very end, after "Examples" -- and so directly before sphinx-autocodelink's "Used In",
# appended after the whole docstring renders. Identical to numpydoc's own template
# otherwise, just with {{see_also}} moved down.
_DOCSTRING_TEMPLATE = SandboxedEnvironment().from_string(
    '{{index}}\n'
    '{{summary}}\n'
    '{{extended_summary}}\n'
    '{{parameters}}\n'
    '{{attributes}}\n'
    '{{methods}}\n'
    '{{returns}}\n'
    '{{yields}}\n'
    '{{receives}}\n'
    '{{other_parameters}}\n'
    '{{raises}}\n'
    '{{warns}}\n'
    '{{warnings}}\n'
    '{{notes}}\n'
    '{{references}}\n'
    '{{examples}}\n'
    '{{see_also}}\n'
)

_original_load_config = SphinxDocString.load_config


def _load_config(self, config):
    _original_load_config(self, config)
    self.template = _DOCSTRING_TEMPLATE


SphinxDocString.load_config = _load_config


# -- headings instead of rubrics for docstring sections -----------------------
# Mirrors doc/source/conf.py's own override of the same name: numpydoc renders section
# headers (Notes, References, Examples) as `.. rubric::` by default, which are invisible
# to a heading-built "on this page" navbar. Duplicated here (not imported) since this
# fixture is deliberately self-contained -- see the module docstring.
def _str_header(self, name):  # noqa: ARG001
    return [name, '-' * len(name), '']


SphinxDocString._str_header = _str_header


def _is_nested_desc(node: Element) -> bool:
    parent = node.parent
    while parent is not None:
        if isinstance(parent, addnodes.desc):
            return True
        parent = parent.parent
    return False


def hoist_docstring_sections(app: Sphinx, doctree: Element) -> None:  # noqa: ARG001
    """Move docstring sections out of their ``desc`` node to page level.

    Finds sections at any depth inside ``desc_content``, not just its direct
    children -- see doc/source/conf.py's own copy of this function for why.
    """
    for desc in list(doctree.findall(addnodes.desc)):
        if _is_nested_desc(desc):
            continue
        parent = desc.parent
        if parent is None:
            continue
        if len([node for node in parent if isinstance(node, addnodes.desc)]) != 1:
            continue
        content = next((node for node in desc if isinstance(node, addnodes.desc_content)), None)
        if content is None:
            continue
        sections = list(content.findall(nodes.section))
        index = parent.index(desc)
        for offset, section in enumerate(sections):
            section.parent.remove(section)
            parent.insert(index + 1 + offset, section)


def setup(app: Sphinx) -> None:
    """Wire up the same doctree-read priority ordering as doc/source/conf.py."""
    # priority < 500 so this runs before Sphinx's TocTreeCollector builds the toc
    app.connect('doctree-read', hoist_docstring_sections, priority=400)
