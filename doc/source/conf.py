from __future__ import annotations

import datetime
import faulthandler
import locale
import os
from pathlib import Path
import sys

# Otherwise VTK reader issues on some systems, causing sphinx to crash. See also #226.
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

faulthandler.enable()

sys.path.insert(0, str(Path().resolve()))
import make_external_gallery
import make_tables

make_external_gallery.make_example_gallery()
make_tables.make_all_tables()

# -- pyvista configuration ---------------------------------------------------
import pyvista
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.docs import linkcode_resolve  # noqa: F401
from pyvista.core.utilities.docs import pv_html_page_context
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper

# Manage errors
pyvista.set_error_output_file("errors.txt")
# Ensure that offscreen rendering is used for docs generation
pyvista.OFF_SCREEN = True  # Not necessary - simply an insurance policy
# Preferred plotting style for documentation
pyvista.set_plot_theme("document")
pyvista.global_theme.window_size = [1024, 768]
pyvista.global_theme.font.size = 22
pyvista.global_theme.font.label_size = 22
pyvista.global_theme.font.title_size = 22
pyvista.global_theme.return_cpos = False
pyvista.set_jupyter_backend(None)
# Save figures in specified directory
pyvista.FIGURE_PATH = str(Path("./images/").resolve() / "auto-generated/")
if not Path(pyvista.FIGURE_PATH).exists():
    Path(pyvista.FIGURE_PATH).mkdir()

# necessary when building the sphinx gallery
pyvista.BUILDING_GALLERY = True
os.environ['PYVISTA_BUILDING_GALLERY'] = 'true'

# SG warnings
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.",
)

# Prevent deprecated features from being used in examples
warnings.filterwarnings(
    "error",
    category=PyVistaDeprecationWarning,
)

# -- General configuration ------------------------------------------------
numfig = False
html_logo = "./_static/pyvista_logo_sm.png"

sys.path.append(str(Path("./_ext").resolve()))

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "enum_tools.autoenum",
    "jupyter_sphinx",
    "notfound.extension",
    "numpydoc",
    "pyvista.ext.coverage",
    "pyvista.ext.plot_directive",
    "pyvista.ext.viewer_directive",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.linkcode",  # This adds the button ``[Source]`` to each Python API site by calling ``linkcode_resolve``
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.asciinema",
    "sphinx_tags",
    "sphinx_toolbox.more_autodoc.overloads",
    "sphinx_toolbox.more_autodoc.typevars",
    "sphinx_toolbox.more_autodoc.autonamedtuple",
    "sphinxext.opengraph",
    "sphinx_sitemap",
]

# Configuration of pyvista.ext.coverage
coverage_additional_modules = [
    'pyvista',
    'pyvista.errors',
    'pyvista.report',
    # core
    'pyvista.core.cell',
    'pyvista.core.celltype',
    'pyvista.core.composite',
    'pyvista.core.dataobject',
    'pyvista.core.dataset',
    'pyvista.core.datasetattributes',
    'pyvista.core.errors',
    'pyvista.core.grid',
    'pyvista.core.objects',
    'pyvista.core.pointset',
    'pyvista.core.pyvista_ndarray',
    'pyvista.core.filters.composite',
    'pyvista.core.filters.data_set',
    'pyvista.core.filters.poly_data',
    'pyvista.core.filters.structured_grid',
    'pyvista.core.filters.image_data',
    'pyvista.core.filters.unstructured_grid',
    'pyvista.core.utilitis.arrays',
    'pyvista.core.utilitis.cell_type_helper',
    'pyvista.core.utilitis.cells',
    'pyvista.core.utilitis.features',
    'pyvista.core.utilitis.fileio',
    'pyvista.core.utilitis.geometric_objects',
    'pyvista.core.utilitis.helpers',
    'pyvista.core.utilitis.misc',
    'pyvista.core.utilitis.observers',
    'pyvista.core.utilitis.parametric_objects',
    'pyvista.core.utilitis.points',
    'pyvista.core.utilitis.reader',
    'pyvista.core.utilitis.transformations',
    # demos
    'pyvista.demos.demos',
    'pyvista.demos.logo',
    # examples
    'pyvista.examples.cells',
    'pyvista.examples.downloads',
    'pyvista.examples.examples',
    'pyvista.examples.gltf',
    'pyvista.examples.planets',
    'pyvista.examples.vrml',
    # plotting
    'pyvista.plotting.actor_properties',
    'pyvista.plotting.actor',
    'pyvista.plotting.axes_actor',
    'pyvista.plotting.axes',
    'pyvista.plotting.background_renderer',
    'pyvista.plotting.camera',
    'pyvista.plotting.charts',
    'pyvista.plotting.colors',
    'pyvista.plotting.composite_mapper',
    'pyvista.plotting.cube_axes_actor',
    'pyvista.plotting.errors',
    'pyvista.plotting.helpers',
    'pyvista.plotting.lights',
    'pyvista.plotting.lookup_table',
    'pyvista.plotting.mapper',
    'pyvista.plotting.opts',
    'pyvista.plotting.picking',
    'pyvista.plotting.plotter',
    'pyvista.plotting.prop3d',
    'pyvista.plotting.render_passes',
    'pyvista.plotting.render_window_interactor',
    'pyvista.plotting.renderer',
    'pyvista.plotting.renderers',
    'pyvista.plotting.scalar_bars',
    'pyvista.plotting.texture',
    'pyvista.plotting.themes',
    'pyvista.plotting.tools',
    'pyvista.plotting.volume_property',
    'pyvista.plotting.volume',
    'pyvista.plotting.widgets',
]
coverage_ignore_modules = [
    r'\.plot_directive$',  # Issue with class parameter documentation
]


# Configuration for sphinx.ext.autodoc
# Do not expand following type aliases when generating the docs
autodoc_type_aliases = {
    "Chart": "pyvista.Chart",
    "ColorLike": "pyvista.ColorLike",
    "ArrayLike": "pyvista.ArrayLike",
    "VectorLike": "pyvista.VectorLike",
    "MatrixLike": "pyvista.MatrixLike",
    "BoundsLike": "pyvista.BoundsLike",
    "CellsLike": "pyvista.CellsLike",
    "CellArrayLike": "pyvista.CellArrayLike",
    "TransformLike": "pyvista.TransformLike",
    "RotationLike": "pyvista.RotationLike",
}

# Hide overload type signatures (from "sphinx_toolbox.more_autodoc.overload")
overloads_location = ["bottom"]

# Display long function signatures with each param on a new line.
# Helps make annotated signatures more readable.
maximum_signature_line_length = 88

# See https://numpydoc.readthedocs.io/en/latest/install.html
numpydoc_use_plots = True
numpydoc_show_class_members = False
numpydoc_xref_param_type = True

# linkcheck ignore entries
nitpick_ignore_regex = [
    (r'py:.*', '.*ColorLike'),
    (r'py:.*', '.*lookup_table_ndarray'),
    (r'py:.*', 'ActiveArrayInfo'),
    (r'py:.*', 'FieldAssociation'),
    (r'py:.*', 'VTK'),
    (r'py:.*', 'colors.Colormap'),
    (r'py:.*', 'cycler.Cycler'),
    (r'py:.*', 'ipywidgets.Widget'),
    (r'py:.*', 'meshio.*'),
    (r'py:.*', 'networkx.*'),
    (r'py:.*', 'of'),
    (r'py:.*', 'optional'),
    (r'py:.*', 'or'),
    (r'py:.*', 'pyvista.LookupTable.n_values'),
    (r'py:.*', 'pyvista.PVDDataSet'),
    (r'py:.*', 'sys.float_info.max'),
    (r'py:.*', 'various'),
    (r'py:.*', 'vtk.*'),
]


add_module_names = False

# Intersphinx mapping
# NOTE: if these are changed, then doc/intersphinx/update.sh
# must be changed accordingly to keep auto-updated mappings working
intersphinx_mapping = {
    'python': (
        'https://docs.python.org/3.11',
        (None, '../intersphinx/python-objects.inv'),
    ),  # Pin Python 3.11. See https://github.com/pyvista/pyvista/pull/5018 .
    'scipy': (
        'https://docs.scipy.org/doc/scipy/',
        (None, '../intersphinx/scipy-objects.inv'),
    ),
    'numpy': ('https://numpy.org/doc/stable', (None, '../intersphinx/numpy-objects.inv')),
    'matplotlib': (
        'https://matplotlib.org/stable',
        (None, '../intersphinx/matplotlib-objects.inv'),
    ),
    'imageio': (
        'https://imageio.readthedocs.io/en/stable',
        (None, '../intersphinx/imageio-objects.inv'),
    ),
    'pandas': (
        'https://pandas.pydata.org/pandas-docs/stable',
        (None, '../intersphinx/pandas-objects.inv'),
    ),
    'pytest': ('https://docs.pytest.org/en/stable', (None, '../intersphinx/pytest-objects.inv')),
    'pyvistaqt': ('https://qtdocs.pyvista.org/', (None, '../intersphinx/pyvistaqt-objects.inv')),
    'trimesh': ('https://trimesh.org', (None, '../intersphinx/trimesh-objects.inv')),
}
intersphinx_timeout = 10

linkcheck_retries = 3
linkcheck_timeout = 500

# Select if we want to generate production or dev documentation
#
# Generate class table auto-summary when enabled. This generates one page per
# class method or attribute and should be used with the production
# documentation, but local builds and PR commits can get away without this as
# it takes ~4x as long to generate the documentation.
templates_path = ["_templates"]

# Autosummary configuration
autosummary_context = {
    # Methods that should be skipped when generating the docs
    # __init__ should be documented in the class docstring
    # override is a VTK method
    "skipmethods": ["__init__", "override"],
}

# The suffix(es) of source filenames.
source_suffix = ".rst"

# The main toctree document.
root_doc = "index"


# General information about the project.
project = "PyVista"
year = datetime.datetime.now(tz=datetime.timezone.utc).date().year
copyright = f"2017-{year}, The PyVista Developers"  # noqa: A001
author = "Alex Kaszynski and Bane Sullivan"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = pyvista.__version__

# The full version, including alpha/beta/rc tags.
release = pyvista.__version__


# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints", "_templates*"]
html_extra_path = ["_extra"]

# Pages are not detected correct by ``make linkcheck``
linkcheck_ignore = [
    'https://data.kitware.com/#collection/55f17f758d777f6ddc7895b7/folder/5afd932e8d777f15ebe1b183',
    'https://www.sciencedirect.com/science/article/abs/pii/S0309170812002564',
    'https://www.researchgate.net/publication/2926068_LightKit_A_lighting_system_for_effective_visualization',
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Sphinx Gallery Options
from sphinx_gallery.sorting import FileNameSortKey


class ResetPyVista:
    """Reset pyvista module to default settings."""

    def __call__(self, gallery_conf, fname):
        """Reset pyvista module to default settings

        If default documentation settings are modified in any example, reset here.
        """
        import pyvista

        pyvista._wrappers['vtkPolyData'] = pyvista.PolyData
        pyvista.set_plot_theme('document')

    def __repr__(self):
        return 'ResetPyVista'


reset_pyvista = ResetPyVista()


# skip building the osmnx example if osmnx is not installed
has_osmnx = False
try:
    import fiona  # noqa: F401
    import osmnx  # noqa: F401

    has_osmnx = True
except:
    pass


sphinx_gallery_conf = {
    # convert rst to md for ipynb
    "pypandoc": True,
    # path to your examples scripts
    "examples_dirs": ["../../examples/"],
    # path where to save gallery generated examples
    "gallery_dirs": ["examples"],
    # Pattern to search for example files
    "filename_pattern": r"\.py" if has_osmnx else r"(?!osmnx-example)\.py",
    # Remove the "Download all examples" button from the top level gallery
    "download_all_examples": False,
    # Remove sphinx configuration comments from code blocks
    "remove_config_comments": True,
    # Sort gallery example by file name instead of number of lines (default)
    "within_subsection_order": FileNameSortKey,
    # directory where function granular galleries are stored
    "backreferences_dir": None,
    # Modules for which function level galleries are created.  In
    "doc_module": "pyvista",
    "image_scrapers": (DynamicScraper(), "matplotlib"),
    "first_notebook_cell": "%matplotlib inline",
    "reset_modules": (reset_pyvista,),
    "reset_modules_order": "both",
}

suppress_warnings = ["config.cache"]

import re

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


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
import sphinx_book_theme  # noqa: F401

html_theme = "sphinx_book_theme"
html_context = {
    "github_user": "pyvista",
    "github_repo": "pyvista",
    "github_version": "main",
    "doc_path": "doc/source",
    "examples_path": "examples",
}
html_show_sourcelink = False
html_copy_source = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False


def get_version_match(semver):
    """Evaluate the version match for the multi-documentation."""
    if semver.endswith("dev0"):
        return "dev"
    major, minor, _ = semver.split(".")
    return f"{major}.{minor}"


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "analytics": {"google_analytics_id": "UA-140243896-1"},
    "show_prev_next": False,
    "github_url": "https://github.com/pyvista/pyvista",
    "collapse_navigation": True,
    "use_edit_page_button": True,
    "navigation_with_keys": False,
    "show_navbar_depth": 1,
    "max_navbar_depth": 3,
    "icon_links": [
        {
            "name": "Slack Community",
            "url": "https://communityinviter.com/apps/pyvista/pyvista",
            "icon": "fab fa-slack",
        },
        {
            "name": "Support",
            "url": "https://github.com/pyvista/pyvista/discussions",
            "icon": "fa fa-comment fa-fw",
        },
        {
            "name": "Contributing",
            "url": "https://github.com/pyvista/pyvista/blob/main/CONTRIBUTING.rst",
            "icon": "fa fa-gavel fa-fw",
        },
        {
            "name": "The Paper",
            "url": "https://doi.org/10.21105/joss.01450",
            "icon": "fa fa-file-text fa-fw",
        },
    ],
}

# sphinx-panels shouldn't add bootstrap css since the pydata-sphinx-theme
# already loads it
panels_add_bootstrap_css = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    'cards.css',  # used in card CSS
    'no_italic.css',  # disable italic for span classes
]

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "pyvistadoc"


# -- Options for LaTeX output ---------------------------------------------

latex_elements: dict[str, str] = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'point_size': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (root_doc, "pyvista.tex", "pyvista Documentation", author, "manual"),
]

# -- Options for gettext output -------------------------------------------

# To specify names to enable gettext extracting and translation applying for i18n additionally. You can specify below names:
gettext_additional_targets = ["raw"]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(root_doc, "pyvista", "pyvista Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        root_doc,
        "pyvista",
        "pyvista Documentation",
        author,
        "pyvista",
        "A Streamlined Python Interface for the Visualization Toolkit",
        "Miscellaneous",
    ),
]

# -- Custom 404 page

notfound_context = {
    "body": '<h1>Page not found.</h1>\n\nPerhaps try the <a href="http://docs.pyvista.org/examples/index.html">examples page</a>.',
}
notfound_urls_prefix = None


# Copy button customization ---------------------------------------------------
# exclude traditional Python prompts from the copied code
copybutton_prompt_text = r'>>> ?|\.\.\. '
copybutton_prompt_is_regexp = True

# sphinx-tags options ---------------------------------------------------------
# See https://sphinx-tags.readthedocs.io/en/latest/index.html

tags_badge_colors = {
    "load": "primary",
    "filter": "secondary",
    "plot": "dark",
    "widgets": "success",
    "lights": "primary",
}
tags_create_tags = True
tags_create_badges = True
tags_index_head = "Gallery examples categorised by tag:"  # tags landing page intro text
tags_intro_text = "Tags:"  # prefix text for a tags list
tags_overview_title = "Tags"  # title for the tags landing page
tags_output_dir = "tags"
tags_page_header = "Gallery examples contain this tag:"  # tag sub-page, header text
tags_page_title = "Tag"  # tag sub-page, title appended with the tag name

# sphinxext.opengraph ---------------------------------------------------------

ogp_site_url = "https://docs.pyvista.org/"
ogp_image = "https://docs.pyvista.org/_static/pyvista_banner_small.png"

# sphinx-sitemap options ---------------------------------------------------------
html_baseurl = 'https://docs.pyvista.org/'


def setup(app):
    app.connect("html-page-context", pv_html_page_context)
    app.add_css_file("copybutton.css")
    app.add_css_file("no_search_highlight.css")
