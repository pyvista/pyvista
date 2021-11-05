import os
import sys
import datetime

if sys.version_info >= (3, 0):
    import faulthandler

    faulthandler.enable()

sys.path.insert(0, os.path.abspath("."))
import make_external_gallery

make_external_gallery.make_example_gallery()


# -- pyvista configuration ---------------------------------------------------
import pyvista
import numpy as np

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
pyvista.FIGURE_PATH = os.path.join(os.path.abspath("./images/"), "auto-generated/")
if not os.path.exists(pyvista.FIGURE_PATH):
    os.makedirs(pyvista.FIGURE_PATH)

# necessary when building the sphinx gallery
pyvista.BUILDING_GALLERY = True

# SG warnings
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.",
)

# -- General configuration ------------------------------------------------
numfig = False
html_logo = "./_static/pyvista_logo_sm.png"

sys.path.append(os.path.abspath("./_ext"))

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "notfound.extension",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.extlinks",
    "jupyter_sphinx",
    "sphinx_panels",
    "pyvista.ext.plot_directive",
    "pyvista.ext.coverage",
    "numpydoc"
]

# Configuration of pyvista.ext.coverage
coverage_additional_modules = [
    'pyvista',
    'pyvista.themes',

    'pyvista.plotting.axes',
    'pyvista.plotting.camera',
    'pyvista.plotting.helpers',
    'pyvista.plotting.lights',
    'pyvista.plotting.picking',
    'pyvista.plotting.plotting',
    'pyvista.plotting.renderer',
    'pyvista.plotting.renderers',
    'pyvista.plotting.scalar_bars',
    'pyvista.plotting.theme',
    'pyvista.plotting.tools',
    'pyvista.plotting.widgets',

    'pyvista.core.common_data',
    'pyvista.core.common_data',
    'pyvista.core.composite',
    'pyvista.core.dataobject',
    'pyvista.core.datasetattributes',
    'pyvista.core.dataset',
    'pyvista.core.errors',
    'pyvista.core.grid',
    'pyvista.core.imaging',
    'pyvista.core.objects',
    'pyvista.core.pointset',
    'pyvista.core.pyvista_ndarray',

    'pyvista.core.filters.composite',
    'pyvista.core.filters.data_set',
    'pyvista.core.filters.poly_data',
    'pyvista.core.filters.structured_grid',
    'pyvista.core.filters.uniform_grid',
    'pyvista.core.filters.unstructured_grid',

    'pyvista.demos',
    'pyvista.examples.examples',

    'pyvista.utilities.features',
    'pyvista.utilities.fileio',
    'pyvista.utilities.geometric_objects',
    'pyvista.utilities.parametric_objects',
    'pyvista.utilities.reader',
    'pyvista.utilities.xvfb',
    'pyvista.utilities.transformations',
    'pyvista.utilities.regression',

]
coverage_ignore_modules = [
    r'\.plot_directive$',  # Issue with class parameter documentation
]


# See https://numpydoc.readthedocs.io/en/latest/install.html
numpydoc_use_plots = True
numpydoc_show_class_members = False
numpydoc_xref_param_type = True

# see https://github.com/pyvista/pyvista/pull/1612
numpydoc_validate = True
numpydoc_validation_checks = {
    "all",  # all but the following:
    "GL01",  # Contradicts numpydoc examples
    "GL02",  # Permit a blank line after the end of our docstring
    "GL03",  # Considering enforcing
    "SA01",  # Not all docstrings need a see also
    "SA04",  # See also section does not need descriptions
    "SS05",  # Appears to be broken.
    "ES01",  # Not all docstrings need an extend summary.
    "EX01",  # Examples: Will eventually enforce
    "YD01",  # Yields: No plan to enforce
}
numpydoc_validation_exclude = {  # set of regex
    r'\.Plotter$',  # Issue with class parameter documentation

    r'\.from_dict$',
    r'\.__init__$',

    # parm of abstract classes
    r'\.CompositeFilters$',
    r'\.PolyDataFilters$',
    r'\.CompositeFilters$',
    r'\.PolyDataFilters$',
    r'\.UniformGridFilters$',
    r'\.DataSetFilters$',
    r'\.UnstructuredGridFilters$',
    r'\.MultiBlock$',
    r'\.DataSet$',
    r'\.DataSetFilters$',
    r'\.PolyDataFilters$',
    r'\.UnstructuredGridFilters$',
    r'\.UniformGridFilters$',
    r'\.CompositeFilters$',
    r'\.Grid$',
    r'\.RectilinearGrid$',
    r'\.UniformGrid$',
    r'\.DataObject$',
    r'\.Table.save$',
    r'\.Table$',
    r'\.Table.save$',
    r'\.UnstructuredGrid$',
    r'\.ExplicitStructuredGrid$',
    r'\.StructuredGrid$',
    r'\.PointGrid$',

    # classes inherit from BaseReader
    r'\.*Reader$',
    r'\.*Reader\.*',

    # internal
    r'\.Renderer$',

    # deprecated
    r'\.*boolean_add$',
    r'\.*boolean_cut$',
    r'\.*add_field_array$',
    r'\.*add_field_array$',

    # methods we probably should make private
    r'\.store_click_position$',
    r'\.store_mouse_position$',
    r'\.*fly_to_mouse_position$',
    r'\.*key_press_event$',
    r'\.*left_button_down$',

    # MISC
    r'\.*PlotterITK$',
    r'\.*MultiBlock.copy_meta_from$',
    r'\.DataObject.copy_meta_from$',

    # wraps
    r'\.*Plotter.enable_depth_peeling$',
    r'\.*add_scalar_bar$',

    # pending refactor
    r'\.*MultiBlock.next$',

    # called from inherited
    r'\.*Table.copy_meta_from$',

}


add_module_names = False

# Intersphinx mapping
# NOTE: if these are changed, then doc/intersphinx/update.sh
# must be changed accordingly to keep auto-updated mappings working
intersphinx_mapping = {
    'python': ('https://docs.python.org/dev', (None, 'intersphinx/python-objects.inv')),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', (None, 'intersphinx/scipy-objects.inv')),
    'numpy': ('https://numpy.org/doc/stable', (None, 'intersphinx/numpy-objects.inv')),
    'matplotlib': ('https://matplotlib.org/stable', (None, 'intersphinx/matplotlib-objects.inv')),
    'imageio': ('https://imageio.readthedocs.io/en/stable', (None, 'intersphinx/imageio-objects.inv')),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', (None, 'intersphinx/pandas-objects.inv')),
    'pytest': ('https://docs.pytest.org/en/stable', (None, 'intersphinx/pytest-objects.inv')),
    'pyvistaqt': ('https://qtdocs.pyvista.org/', (None, 'intersphinx/pyvistaqt-objects.inv')),
}

linkcheck_retries = 3
linkcheck_timeout = 500

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
source_suffix = ".rst"

# The main toctree document.
master_doc = "index"


# General information about the project.
project = "PyVista"
year = datetime.date.today().year
copyright = f"2017-{year}, The PyVista Developers"
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
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Sphinx Gallery Options
from sphinx_gallery.sorting import FileNameSortKey


def reset_pyvista(gallery_conf, fname):
    """Reset pyvista module to default settings

    If default documentation settings are modified in any example, reset here.
    """
    import pyvista
    pyvista._wrappers['vtkPolyData'] = pyvista.PolyData


sphinx_gallery_conf = {
    # convert rst to md for ipynb
    "pypandoc": True,
    # path to your examples scripts
    "examples_dirs": ["../examples/"],
    # path where to save gallery generated examples
    "gallery_dirs": ["examples"],
    # Patter to search for example files
    "filename_pattern": r"\.py",
    # Remove the "Download all examples" button from the top level gallery
    "download_all_examples": False,
    # Sort gallery example by file name instead of number of lines (default)
    "within_subsection_order": FileNameSortKey,
    # directory where function granular galleries are stored
    "backreferences_dir": None,
    # Modules for which function level galleries are created.  In
    "doc_module": "pyvista",
    "image_scrapers": ("pyvista", "matplotlib"),
    "first_notebook_cell": (
        "%matplotlib inline\n"
        "from pyvista import set_plot_theme\n"
        "set_plot_theme('document')\n"
    ),
    "reset_modules": (reset_pyvista, ),
}

# -- .. pyvista-plot:: directive ----------------------------------------------
from numpydoc.docscrape_sphinx import SphinxDocString
import re

IMPORT_PYVISTA_RE = r'\b(import +pyvista|from +pyvista +import)\b'
IMPORT_MATPLOTLIB_RE = r'\b(import +matplotlib|from +matplotlib +import)\b'


def _str_examples(self):
    examples_str = "\n".join(self['Examples'])

    if (self.use_plots and re.search(IMPORT_MATPLOTLIB_RE, examples_str)
            and 'plot::' not in examples_str):
        out = []
        out += self._str_header('Examples')
        out += ['.. plot::', '']
        out += self._str_indent(self['Examples'])
        out += ['']
        return out
    elif (re.search(IMPORT_PYVISTA_RE, examples_str)
          and 'plot-pyvista::' not in examples_str):
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
import pydata_sphinx_theme

html_theme = "pydata_sphinx_theme"
# html_theme_path = [pydata_sphinx_theme.get_html_theme_path()]
html_context = {
    "github_user": "pyvista",
    "github_repo": "pyvista",
    "github_version": "main",
    "doc_path": "doc",
}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "google_analytics_id": "UA-140243896-1",
    "show_prev_next": False,
    "github_url": "https://github.com/pyvista/pyvista",
    "icon_links": [
        {
            "name": "Slack Community",
            "url": "http://slack.pyvista.org",
            "icon": "fab fa-slack",
        },
        {
            "name": "Support",
            "url": "https://github.com/pyvista/pyvista/discussions",
            "icon": "fa fa-comment fa-fw",
        },
        {
            "name": "Contributing",
            "url": "https://github.com/pyvista/pyvista/blob/main/CONTRIBUTING.md",
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


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "pyvistadoc"


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
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
    (master_doc, "pyvista.tex", "pyvista Documentation", author, "manual"),
]

# -- Options for gettext output -------------------------------------------

# To specify names to enable gettext extracting and translation applying for i18n additionally. You can specify below names:
gettext_additional_targets = ["raw"]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "pyvista", "pyvista Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
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
notfound_no_urls_prefix = True


# Copy button customization ---------------------------------------------------
# exclude traditional Python prompts from the copied code
copybutton_prompt_text = r'>>> ?|\.\.\. '
copybutton_prompt_is_regexp = True


# -- Autosummary options
from sphinx.ext.autosummary import Autosummary
from sphinx.ext.autosummary import get_documenter
from docutils.parsers.rst import directives
from sphinx.util.inspect import safe_getattr


class AutoAutoSummary(Autosummary):

    option_spec = {
        "methods": directives.unchanged,
        "attributes": directives.unchanged,
    }

    required_arguments = 1
    app = None

    @staticmethod
    def get_members(obj, typ, include_public=None):
        if not include_public:
            include_public = []
        items = []
        for name in sorted(obj.__dict__.keys()):  # dir(obj):
            try:
                documenter = get_documenter(
                    AutoAutoSummary.app, safe_getattr(obj, name), obj
                )
            except AttributeError:
                continue
            if documenter.objtype in typ:
                items.append(name)
        public = [x for x in items if x in include_public or not x.startswith("_")]
        return public, items

    def run(self):
        clazz = str(self.arguments[0])
        try:
            (module_name, class_name) = clazz.rsplit(".", 1)
            m = __import__(module_name, globals(), locals(), [class_name])
            c = getattr(m, class_name)
            if "methods" in self.options:
                _, methods = self.get_members(c, ["method"], ["__init__"])
                self.content = [
                    f"~{clazz}.{method}"
                    for method in methods
                    if not method.startswith("_")
                ]
            if "attributes" in self.options:
                _, attribs = self.get_members(c, ["attribute", "property"])
                self.content = [
                    f"~{clazz}.{attrib}"
                    for attrib in attribs
                    if not attrib.startswith("_")
                ]
        except:
            print(f"Something went wrong when autodocumenting {clazz}")
        finally:
            return super().run()


def setup(app):
    AutoAutoSummary.app = app
    app.add_directive("autoautosummary", AutoAutoSummary)
    app.add_css_file("copybutton.css")
