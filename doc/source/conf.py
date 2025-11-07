"""Documentation configuration."""

from __future__ import annotations

import datetime
import faulthandler
import locale
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING
import warnings

from atsphinx.mini18n import get_template_dir
from docutils.parsers.rst.directives.images import Image

if TYPE_CHECKING:
    from sphinx.application import Sphinx

# Otherwise VTK reader issues on some systems, causing sphinx to crash. See also #226.
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

faulthandler.enable()

# ignore joblib warnings from sphinx-gallery parallel build:
# .../site-packages/joblib/externals/loky/process_executor.py:782: UserWarning:
# A worker stopped while some jobs were given to the executor. This can be
# caused by a too short worker timeout or by a memory leak.
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message='A worker stopped while some jobs were given to the executor',
)

# This flag is set *before* any pyvista import. It allows `pyvista.core._typing_core._aliases` to
# import things like `scipy` or `matplotlib` that would be unnecessarily bulky to import by default
# during normal operation. See https://github.com/pyvista/pyvista/pull/7023.
# Note that `import make_tables` below imports pyvista.
os.environ['PYVISTA_DOCUMENTATION_BULKY_IMPORTS_ALLOWED'] = 'true'

sys.path.insert(0, str(Path().cwd()))
import make_tables

# -- pyvista configuration ---------------------------------------------------
import pyvista
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.docs import linkcode_resolve  # noqa: F401
from pyvista.core.utilities.docs import pv_html_page_context
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper

# Manage errors
pyvista.set_error_output_file('errors.txt')
# Ensure that offscreen rendering is used for docs generation
pyvista.OFF_SCREEN = True  # Not necessary - simply an insurance policy
# Preferred plotting style for documentation
pyvista.set_plot_theme('document_build')
pyvista.set_jupyter_backend(None)
# Save figures in specified directory
pyvista.FIGURE_PATH = str(Path('./images/').resolve() / 'auto-generated/')
if not Path(pyvista.FIGURE_PATH).exists():
    Path(pyvista.FIGURE_PATH).mkdir()

# necessary when building the sphinx gallery
pyvista.BUILDING_GALLERY = True
os.environ['PYVISTA_BUILDING_GALLERY'] = 'true'

# SG warnings
import warnings

warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message='Matplotlib is currently using agg, which is a non-GUI backend, '
    'so cannot show the figure.',
)

# Prevent deprecated features from being used in examples
warnings.filterwarnings(
    'error',
    category=PyVistaDeprecationWarning,
)
warnings.filterwarnings(
    'always',
    category=PyVistaDeprecationWarning,
    message='Assigning a theme for a plotter instance is deprecated',
)

# -- General configuration ------------------------------------------------
numfig = False
html_logo = './_static/pyvista_logo.svg'
html_favicon = './_static/pyvista_logo.svg'

sys.path.append(str(Path('./_ext').resolve()))

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'atsphinx.mini18n',
    'enum_tools.autoenum',
    'jupyter_sphinx',
    'notfound.extension',
    'numpydoc',
    'pyvista.ext.plot_directive',
    'pyvista.ext.viewer_directive',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.linkcode',  # Adds [Source] button to each API site by calling ``linkcode_resolve``
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.duration',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx_gallery.gen_gallery',
    'sphinxcontrib.asciinema',
    'sphinx_togglebutton',
    'sphinx_tags',
    'sphinx_toolbox.more_autodoc.overloads',
    'sphinx_toolbox.more_autodoc.typevars',
    'sphinx_toolbox.more_autodoc.autonamedtuple',
    'sphinxext.opengraph',
    'sphinx_sitemap',
    'vtk_xref',
]


# Configuration for sphinx.ext.autodoc
# Do not expand following type aliases when generating the docs
autodoc_type_aliases = {
    'CameraPositionOptions': 'pyvista.CameraPositionOptions',
    'JupyterBackendOptions': 'pyvista.JupyterBackendOptions',
    'Chart': 'pyvista.Chart',
    'ColorLike': 'pyvista.ColorLike',
    'ArrayLike': 'pyvista.ArrayLike',
    'VectorLike': 'pyvista.VectorLike',
    'MatrixLike': 'pyvista.MatrixLike',
    'BoundsLike': 'pyvista.BoundsLike',
    'CellsLike': 'pyvista.CellsLike',
    'CellArrayLike': 'pyvista.CellArrayLike',
    'TransformLike': 'pyvista.TransformLike',
    'RotationLike': 'pyvista.RotationLike',
    'InteractionEventType': 'pyvista.InteractionEventType',
}

# Needed to address a code-block parsing error by sphinx for an example
autodoc_mock_imports = ['example']

# Hide overload type signatures (from "sphinx_toolbox.more_autodoc.overload")
overloads_location = ['bottom']

# Display long function signatures with each param on a new line.
# Helps make annotated signatures more readable.
maximum_signature_line_length = 88

# See https://numpydoc.readthedocs.io/en/latest/install.html
numpydoc_use_plots = True
numpydoc_show_class_members = False
numpydoc_xref_param_type = True

# Warn if target links or references cannot be found
nitpicky = True
# Except ignore these entries
nitpick_ignore_regex = [
    # NOTE: We need to ignore any/all pyvista objects which are used as type hints
    # in function signatures since these are not linked by sphinx (bug).
    # See https://github.com/pyvista/pyvista/pull/6206#issuecomment-2149138086
    #
    # PyVista TypeVars and TypeAliases
    (r'py:.*', '.*ColorLike'),
    (r'py:.*', '.*ImageCompareType'),
    (r'py:.*', '.*ColormapOptions'),
    (r'py:.*', '.*ArrayLike'),
    (r'py:.*', '.*MatrixLike'),
    (r'py:.*', '.*VectorLike'),
    (r'py:.*', '.*TransformLike'),
    (r'py:.*', '.*InteractionEventType'),
    (r'py:.*', '.*BoundsLike'),
    (r'py:.*', '.*RotationLike'),
    (r'py:.*', '.*CellsLike'),
    (r'py:.*', '.*ShapeLike'),
    (r'py:.*', '.*NumpyArray'),
    (r'py:.*', '.*_ArrayLikeOrScalar'),
    (r'py:.*', '.*NumberType'),
    (r'py:.*', '.*_PolyDataType'),
    (r'py:.*', '.*_UnstructuredGridType'),
    (r'py:.*', '.*_GridType'),
    (r'py:.*', '.*_PointGridType'),
    (r'py:.*', '.*_PointSetType'),
    (r'py:.*', '.*_DataSetType'),
    (r'py:.*', '.*_DataSetOrMultiBlockType'),
    (r'py:.*', '.*_DataObjectType'),
    (r'py:.*', '.*_MeshType_co'),
    (r'py:.*', '.*_WrappableVTKDataObjectType'),
    (r'py:.*', '.*_VTKWriterType'),
    (r'py:.*', '.*NormalsLiteral'),
    (r'py:.*', '.*_CellQualityLiteral'),
    (r'py:.*', '.*_CompressionOptions'),
    (r'py:.*', '.*T'),
    (r'py:.*', '.*Options'),
    #
    # Dataset-related types
    (r'py:.*', '.*DataSet'),
    (r'py:.*', '.*DataObject'),
    (r'py:.*', '.*PolyData'),
    (r'py:.*', '.*UnstructuredGrid'),
    (r'py:.*', '.*_TypeMultiBlockLeaf'),
    (r'py:.*', '.*Grid'),
    (r'py:.*', '.*PointGrid'),
    (r'py:.*', '.*_PointSet'),
    #
    # PyVista array-related types
    (r'py:.*', 'ActiveArrayInfo'),
    (r'py:.*', 'FieldAssociation'),
    (r'py:.*', '.*CellLiteral'),
    (r'py:.*', '.*PointLiteral'),
    (r'py:.*', '.*FieldLiteral'),
    (r'py:.*', '.*RowLiteral'),
    (r'py:.*', '.*_SerializedDictArray'),
    (r'py:.*', '.*_FiveArrays'),
    #
    # PyVista AxesAssembly-related types
    (r'py:.*', '.*GeometryTypes'),
    (r'py:.*', '.*ShaftType'),
    (r'py:.*', '.*TipType'),
    (r'py:.*', '.*_AxesGeometryKwargs'),
    (r'py:.*', '.*_OrthogonalPlanesKwargs'),
    #
    # PyVista Widget enums
    (r'py:.*', '.*PickerType'),
    (r'py:.*', '.*ElementType'),
    #
    # PyVista Texture enum
    (r'py:.*', '.*WrapType'),
    #
    # PyVista plotting-related classes
    (r'py:.*', '.*BasePlotter'),
    (r'py:.*', '.*ScalarBars'),
    (r'py:.*', '.*Theme'),
    #
    # Misc pyvista ignores
    (r'py:.*', 'principal_axes'),  # Valid ref, but is not linked correctly in some wrapped cases
    (r'py:.*', 'axes_enabled'),  # Valid ref, but is not linked correctly in some wrapped cases
    (r'py:.*', '.*lookup_table_ndarray'),
    (r'py:.*', '.*colors.Colormap'),
    (r'py:.*', 'colors.ListedColormap'),
    (r'py:.*', '.*CellQualityInfo'),
    (r'py:.*', 'cycler.Cycler'),
    (r'py:.*', 'pyvista.PVDDataSet'),
    (r'py:.*', 'ScalarBarArgs'),
    (r'py:.*', 'SilhouetteArgs'),
    (r'py:.*', 'BackfaceArgs'),
    (r'py:.*', 'CullingOptions'),
    (r'py:.*', 'OpacityOptions'),
    (r'py:.*', 'CameraPositionOptions'),
    (r'py:.*', 'StyleOptions'),
    (r'py:.*', 'FontFamilyOptions'),
    (r'py:.*', 'HorizontalOptions'),
    (r'py:.*', 'VerticalOptions'),
    (r'py:.*', '.*JupyterBackendOptions'),
    (r'py:.*', '_InterpolationOptions'),
    (r'py:.*', 'PlottableType'),
    #
    # Built-in python types. TODO: Fix links (intersphinx?)
    (r'py:.*', '.*StringIO'),
    (r'py:.*', '.*Path'),
    (r'py:.*', '.*UserDict'),
    (r'py:.*', 'sys.float_info.max'),
    (r'py:.*', '.*NoneType'),
    (r'py:.*', 'collections.*'),
    (r'py:.*', '.*PathStrSeq'),
    (r'py:.*', 'ModuleType'),
    (r'py:.*', 'typing.Union'),
    #
    # NumPy types. TODO: Fix links (intersphinx?)
    (r'py:.*', '.*DTypeLike'),
    (r'py:.*', 'np.*'),
    (r'py:.*', 'npt.*'),
    (r'py:.*', 'numpy.*'),
    (r'py:.*', '.*NDArray'),
    #
    # Third party ignores. TODO: Can these be linked with intersphinx?
    (r'py:.*', 'ipywidgets.Widget'),
    (r'py:.*', 'EmbeddableWidget'),
    (r'py:.*', 'Widget'),
    (r'py:.*', 'IFrame'),
    (r'py:.*', 'Image'),
    (r'py:.*', 'meshio.*'),
    (r'py:.*', '.*Mesh'),
    (r'py:.*', '.*Trimesh'),
    (r'py:.*', 'networkx.*'),
    (r'py:.*', 'Rotation'),
    (r'py:.*', 'vtk.*'),
    (r'py:.*', '_vtk.*'),
    (r'py:.*', 'VTK'),
    #
    # Misc general ignores
    (r'py:.*', 'optional'),
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

# Select if we want to generate production or dev documentation
#
# Generate class table auto-summary when enabled. This generates one page per
# class method or attribute and should be used with the production
# documentation, but local builds and PR commits can get away without this as
# it takes ~4x as long to generate the documentation.
templates_path = ['_templates', get_template_dir()]

# Autosummary configuration
autosummary_context = {
    # Methods that should be skipped when generating the docs
    # __init__ should be documented in the class docstring
    # override is a VTK method
    'skipmethods': ['__init__', 'override'],
}

# The suffix(es) of source filenames.
source_suffix = '.rst'

# The main toctree document.
root_doc = 'index'


# General information about the project.
project = 'PyVista'
year = datetime.datetime.now(tz=datetime.timezone.utc).date().year
copyright = f'2017-{year}, The PyVista Developers'  # noqa: A001
author = 'Alex Kaszynski and Bane Sullivan'

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
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints', '_templates*']
html_extra_path = ['_extra']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'friendly'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Sphinx Gallery Options
from sphinx_gallery.sorting import FileNameSortKey


class ResetPyVista:
    """Reset pyvista module to default settings."""

    def __call__(self, gallery_conf, fname):  # noqa: ARG002
        """Reset pyvista module to default settings.

        If default documentation settings are modified in any example, reset here.
        """
        import matplotlib as mpl  # must import before pyvista

        # clear all mpl figures, force non-interactive backend, and reset defaults
        mpl.use('Agg', force=True)
        mpl.pyplot.close('all')
        mpl.rcdefaults()
        mpl.pyplot.figure().clear()
        mpl.pyplot.close()

        import pyvista

        pyvista._wrappers['vtkPolyData'] = pyvista.PolyData
        pyvista.set_plot_theme('document_build')

    def __repr__(self):
        return 'ResetPyVista'


reset_pyvista = ResetPyVista()


sphinx_gallery_conf = {
    'abort_on_example_error': True,  # Fail early
    # convert rst to md for ipynb
    'pypandoc': True,
    # path to your examples scripts
    'examples_dirs': ['../../examples/'],
    # path where to save gallery generated examples
    'gallery_dirs': ['examples'],
    # Pattern to search for example files
    'filename_pattern': r'\.py',
    # Remove the "Download all examples" button from the top level gallery
    'download_all_examples': False,
    # Remove sphinx configuration comments from code blocks
    'remove_config_comments': True,
    # Sort gallery example by file name instead of number of lines (default)
    'within_subsection_order': FileNameSortKey,
    # directory where function granular galleries are stored
    'backreferences_dir': None,
    # Modules for which function level galleries are created.  In
    'doc_module': 'pyvista',
    'reference_url': {'pyvista': None},  # Add hyperlinks inside code blocks to pyvista methods
    'image_scrapers': (DynamicScraper(), 'matplotlib'),
    'first_notebook_cell': '%matplotlib inline',
    'reset_modules': (reset_pyvista,),
    'reset_modules_order': 'both',
    'junit': str(Path('sphinx-gallery') / 'junit-results.xml'),
    'parallel': True,  # use the same number of workers as "-j" in sphinx
}

suppress_warnings = ['config.cache', 'image.not_readable']

import re

# -- .. pyvista-plot:: directive ----------------------------------------------
from numpydoc.docscrape_sphinx import SphinxDocString

IMPORT_PYVISTA_RE = r'\b(import +pyvista|from +pyvista +import)\b'
IMPORT_MATPLOTLIB_RE = r'\b(import +matplotlib|from +matplotlib +import)\b'

pyvista_plot_setup = """
from pyvista import set_plot_theme as __s_p_t
__s_p_t('document_build')
del __s_p_t
"""
pyvista_plot_cleanup = pyvista_plot_setup


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

html_theme = 'sphinx_book_theme'
html_context = {
    'github_user': 'pyvista',
    'github_repo': 'pyvista',
    'github_version': 'main',
    'doc_path': 'doc/source',
    'examples_path': 'examples',
}
html_show_sourcelink = False
html_copy_source = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False


def get_version_match(semver):
    """Evaluate the version match for the multi-documentation."""
    if semver.endswith('dev0'):
        return 'dev'
    major, minor, _ = semver.split('.')
    return f'{major}.{minor}'


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'analytics': {'google_analytics_id': 'UA-140243896-1'},
    'show_prev_next': False,
    'github_url': 'https://github.com/pyvista/pyvista',
    'collapse_navigation': True,
    'use_edit_page_button': True,
    'navigation_with_keys': False,
    'show_navbar_depth': 1,
    'max_navbar_depth': 3,
    'icon_links': [
        {
            'name': 'Slack Community',
            'url': 'https://communityinviter.com/apps/pyvista/pyvista',
            'icon': 'fab fa-slack',
        },
        {
            'name': 'Support',
            'url': 'https://github.com/pyvista/pyvista/discussions',
            'icon': 'fa fa-comment fa-fw',
        },
        {
            'name': 'Contributing',
            'url': 'https://github.com/pyvista/pyvista/blob/main/CONTRIBUTING.rst',
            'icon': 'fa fa-gavel fa-fw',
        },
        {
            'name': 'The Paper',
            'url': 'https://doi.org/10.21105/joss.01450',
            'icon': 'fa fa-file-text fa-fw',
        },
    ],
}

if 'dev' in pyvista.__version__:
    stable_base = 'https://docs.pyvista.org'
    announcement_html = f"""
    <div class="pv-announcement">
        This is documentation for an <strong>unstable development version</strong>
        <span style="white-space: nowrap;">.</span> 
        <a id="stable-link" class="pv-announcement-button">
            Switch to stable version
        </a>
    </div>
    <script>
        const link = document.getElementById('stable-link');
        const stableBase = "{stable_base}";
        const path = window.location.pathname + window.location.hash + window.location.search;
        link.href = stableBase + path;
    </script>
    """

    html_theme_options['announcement'] = announcement_html


# sphinx-panels shouldn't add bootstrap css since the pydata-sphinx-theme
# already loads it
panels_add_bootstrap_css = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'cards.css',  # used in card CSS
    'no_italic.css',  # disable italic for span classes
    'announcement.css',  # override banner color
]

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'pyvistadoc'


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
    (root_doc, 'pyvista.tex', 'pyvista Documentation', author, 'manual'),
]

# -- Options for gettext output -------------------------------------------

# To specify names to enable gettext extracting and translation applying for i18n additionally.
# You can specify below names:
gettext_additional_targets = ['raw']

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(root_doc, 'pyvista', 'pyvista Documentation', [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        root_doc,
        'pyvista',
        'pyvista Documentation',
        author,
        'pyvista',
        'A Streamlined Python Interface for the Visualization Toolkit',
        'Miscellaneous',
    ),
]

# -- Custom 404 page

notfound_context = {
    'body': (
        '<h1>Page not found.</h1>\n\n'
        'Perhaps try the <a href="http://docs.pyvista.org/examples/index.html">examples page</a>.'
    ),
}
notfound_urls_prefix = None


# Copy button customization ---------------------------------------------------
# exclude traditional Python prompts from the copied code
copybutton_prompt_text = r'>>> ?|\.\.\. '
copybutton_prompt_is_regexp = True

# sphinx-tags options ---------------------------------------------------------
# See https://sphinx-tags.readthedocs.io/en/latest/index.html

tags_badge_colors = {
    'load': 'primary',
    'filter': 'secondary',
    'plot': 'dark',
    'widgets': 'success',
    'lights': 'primary',
}
tags_create_tags = True
tags_create_badges = True
tags_index_head = 'Gallery examples categorised by tag:'  # tags landing page intro text
tags_intro_text = 'Tags:'  # prefix text for a tags list
tags_overview_title = 'Tags'  # title for the tags landing page
tags_output_dir = 'tags'
tags_page_header = 'Gallery examples contain this tag:'  # tag sub-page, header text
tags_page_title = 'Tag'  # tag sub-page, title appended with the tag name

# sphinxext.opengraph ---------------------------------------------------------

ogp_site_url = 'https://docs.pyvista.org/'
ogp_image = 'https://docs.pyvista.org/_static/pyvista_banner_small.png'

# sphinx-sitemap options ---------------------------------------------------------
html_baseurl = 'https://docs.pyvista.org/'

# atsphinx.mini18n options ---------------------------------------------------------
html_sidebars = {
    '**': [
        'navbar-logo.html',
        'icon-links.html',
        'mini18n/snippets/select-lang.html',
        'search-button-field.html',
        'sbt-sidebar-nav.html',
    ],
}
mini18n_default_language = language
mini18n_support_languages = ['en', 'ja']
locale_dirs = ['../../pyvista-doc-translations/locale']


class PlaceHolderImage(Image):
    """A custom Image directive that checks for placeholders in an image path."""

    gen_image_path = Path(make_tables.DATASET_GALLERY_IMAGE_DIR).relative_to('..')

    def run(self):  # noqa: D102
        image_path_str = self.arguments[0]

        if make_tables.PLACEHOLDER in image_path_str:
            image_path = Path(image_path_str)
            # Fill in the placeholder with the first matching image. This will
            # not respect order of generation.
            basename = image_path.name.replace('PLACEHOLDER', '*')
            actual_image = next(self.gen_image_path.glob(basename), None)
            if actual_image:
                self.arguments[0] = str(actual_image)

        return super().run()


def report_parallel_safety(app: Sphinx, *_) -> None:
    """Raise an error if an extension is blocking a parallel build."""
    if app.parallel > 1:
        for name, ext in sorted(app.extensions.items()):
            read_safe = getattr(ext, 'parallel_read_safe', None)
            write_safe = getattr(ext, 'parallel_write_safe', None)
            if read_safe is not True or write_safe is not True:
                msg = (
                    f'Parallel build enabled but extension "{name}" is not fully parallel '
                    f'safe (read_safe={read_safe}, write_safe={write_safe})'
                )
                raise RuntimeError(msg)


def configure_backend(app: Sphinx) -> None:  # noqa: D103
    app.add_directive('image', PlaceHolderImage)


def setup(app: Sphinx) -> None:  # noqa: D103
    app.connect('config-inited', report_parallel_safety)
    app.connect('builder-inited', configure_backend)
    app.connect('html-page-context', pv_html_page_context)

    # right before writing, patch the gallery placeholders
    app.connect('doctree-resolved', make_tables.patch_gallery_placeholders)

    app.add_css_file('copybutton.css')
    app.add_css_file('no_search_highlight.css')
