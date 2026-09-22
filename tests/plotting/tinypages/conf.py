from __future__ import annotations

import datetime
import os
from pathlib import Path
import re
import sys
import warnings

import pyvista as pv

sys.path.append(str(Path(__file__).parent))

# Suppress invalid user data path when there is an issue with restoring the cache for CI
# https://github.com/pyvista/pyvista/pull/7747
warnings.filterwarnings(
    'ignore',
    message=r'(?s).*PYVISTA(_VTK)?_DATA is not a valid directory.*',
    category=UserWarning,
)

# -- General configuration ------------------------------------------------

templates_path = ['_templates']
source_suffix = '.rst'
root_doc = 'index'
project = 'tinypages'
year = datetime.datetime.now(tz=datetime.timezone.utc).date().year
copyright = f'2021-{year}, PyVista developers'  # noqa: A001
version = '0.1'
release = '0.1'
exclude_patterns = ['_build', 'gallery_src']
pygments_style = 'sphinx'

extensions = [
    'numpydoc',
    'matplotlib.sphinxext.plot_directive',
    'pyvista.ext.plot_directive',
    'pyvista.ext.viewer_directive',
    'sphinx_autoopengraph',
    'sphinx_examples_as_code',
    'sphinx.ext.autosummary',
    'sphinx_design',
    'sphinx_gallery.gen_gallery',
    'sphinxext.opengraph',
]

sphinx_examples_as_code_conf = {}

# -- Sphinx-Gallery -------------------------------------------------------
# ``gallery_dirs`` is generated into the source tree, so tests build from a copy.
#
# The examples deliberately render with matplotlib rather than PyVista. Gallery
# examples run in the main process before Sphinx forks its parallel readers, and on
# macOS forking after VTK has initialised a render window kills the workers. What is
# under test here is which image the gallery selects, which does not depend on the
# scraper that produced it.
sphinx_gallery_conf = {
    'examples_dirs': ['gallery_src'],
    'gallery_dirs': ['gallery'],
    'filename_pattern': r'\.py',
    'image_scrapers': ('matplotlib',),
    'download_all_examples': False,
    'remove_config_comments': True,
}

# -- Open Graph -----------------------------------------------------------
# No configuration of ``sphinx_autoopengraph`` itself is required; ``ogp_site_url``
# is the only thing ``sphinxext-opengraph`` requires to emit absolute URLs.
ogp_site_url = 'https://docs.example.org/'
ogp_image = 'https://docs.example.org/_static/fallback.png'

# -- Plot directive specific configuration --------------------------------
plot_setup = plot_cleanup = 'import pyvista as pv'

# -- Options for HTML output ----------------------------------------------

html_theme = 'sphinx_book_theme'

html_static_path = ['_static']

# -- pyvista configuration ------------------------------------------------
pv.BUILDING_GALLERY = True
# Small renders: nothing in these builds asserts on image size.
pv.global_theme.window_size = [256, 192]

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

if value := os.environ.get('PYVISTA_PLOT_SKIP'):
    pyvista_plot_skip = value.lower() == 'true'

if value := os.environ.get('PYVISTA_PLOT_SKIP_OPTIONAL'):
    pyvista_plot_skip_optional = value.lower() == 'true'


def _str_examples(self):
    examples_str = '\n'.join(self['Examples'])

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
    elif re.search(IMPORT_PYVISTA_RE, examples_str) and 'pyvista-plot::' not in examples_str:
        out = []
        out += self._str_header('Examples')
        out += ['.. pyvista-plot::', '']
        out += self._str_indent(self['Examples'])
        out += ['']
        return out
    else:
        return self._str_section('Examples')


SphinxDocString._str_examples = _str_examples

# required for testing
pyvista_plot_use_counter = os.environ.get('PYVISTA_PLOT_USE_COUNTER', 'true').lower() == 'true'
