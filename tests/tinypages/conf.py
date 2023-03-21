import datetime
import os
import re
import sys

from packaging.version import parse as parse_version
import sphinx

import pyvista

sys.path.append(os.path.join(os.path.dirname(__file__)))

# -- General configuration ------------------------------------------------

extensions = ['pyvista.ext.plot_directive']
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = 'tinypages'
year = datetime.date.today().year
copyright = f"2021-{year}, PyVista developers"
version = '0.1'
release = '0.1'
exclude_patterns = ['_build']
pygments_style = 'sphinx'

extensions = [
    "numpydoc",
    "pyvista.ext.plot_directive",
    "sphinx.ext.autosummary",
]

# -- Plot directive specific configuration --------------------------------
plot_setup = plot_cleanup = 'import pyvista'

# -- Options for HTML output ----------------------------------------------

if parse_version(sphinx.__version__) >= parse_version('1.3'):
    html_theme = 'classic'
else:
    html_theme = 'default'

html_static_path = ['_static']

# -- pyvista configuration ------------------------------------------------
pyvista.BUILDING_GALLERY = True

# -- .. pyvista-plot:: directive ----------------------------------------------
from numpydoc.docscrape_sphinx import SphinxDocString

IMPORT_PYVISTA_RE = r'\b(import +pyvista|from +pyvista +import)\b'
IMPORT_MATPLOTLIB_RE = r'\b(import +matplotlib|from +matplotlib +import)\b'

plot_setup = """
from pyvista import set_plot_theme as __s_p_t
__s_p_t('document')
del __s_p_t
"""
plot_cleanup = plot_setup


def _str_examples(self):
    examples_str = "\n".join(self['Examples'])

    if (
        self.use_plots
        and re.search(IMPORT_MATPLOTLIB_RE, examples_str)
        and 'plot::' not in examples_str
    ):
        out = []
        out += self._str_header('Examples')
        out += ['.. plot::', '']
        out += self._str_indent(self['Examples'])
        out += ['']
        return out
    elif re.search(IMPORT_PYVISTA_RE, examples_str) and 'plot-pyvista::' not in examples_str:
        out = []
        out += self._str_header('Examples')
        out += ['.. pyvista-plot::', '']
        out += self._str_indent(self['Examples'])
        out += ['']
        return out
    else:
        return self._str_section('Examples')


SphinxDocString._str_examples = _str_examples
