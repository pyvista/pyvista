"""Documentation configuration."""

from __future__ import annotations

import datetime
import faulthandler
import json
import locale
import os
from pathlib import Path
import shutil
import sys
from typing import TYPE_CHECKING
import warnings

from docutils import nodes
from docutils.parsers.rst.directives.images import Image
from sphinx import addnodes
from sphinx_autocodelink.gallery import AutoCodeLinkScraper

if TYPE_CHECKING:
    from docutils.nodes import Element
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
os.environ['_PYVISTA_DOCUMENTATION_BULKY_IMPORTS_ALLOWED'] = 'true'

sys.path.insert(0, str(Path().cwd()))
import make_search_summaries
import make_tables

# -- pyvista configuration ---------------------------------------------------
import pyvista as pv
from pyvista import _vtk
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.docs import linkcode_resolve  # noqa: F401
from pyvista.core.utilities.docs import pv_html_page_context
from pyvista.ext._autoenum import instance_property_names
from pyvista.ext._autoenum import metaclass_property_descriptions
from pyvista.ext._autoenum import metaclass_property_names
from pyvista.ext._autoinherit import filter_member_rows
from pyvista.ext._autoinherit import inherited_member_rows
from pyvista.ext._autoinherit import own_members
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper

# Need to import all vtk modules eagerly to avoid issues with parallel lazy imports
_vtk.import_all()

# Manage errors
pv.set_error_output_file('errors.txt')
# Ensure that offscreen rendering is used for docs generation
pv.OFF_SCREEN = True  # Not necessary - simply an insurance policy
# Preferred plotting style for documentation
pv.set_plot_theme('document_build')
pv.set_jupyter_backend(None)
# Save figures in specified directory
pv.FIGURE_PATH = str(Path('./images/').resolve() / 'auto-generated/')
if not Path(pv.FIGURE_PATH).exists():
    Path(pv.FIGURE_PATH).mkdir()

# necessary when building the sphinx gallery
pv.BUILDING_GALLERY = True
os.environ['PYVISTA_BUILDING_GALLERY'] = 'true'

# Copy contents of `pyvista/examples` dir so that we have actual mesh files
# we can run CLI commands on locally without polluting the source dir
HERE = Path(__file__).parent
src = HERE.parent.parent / 'pyvista' / 'examples'
dst = HERE / '_local_examples'
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(src, dst)

# SG warnings
import warnings

warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message=(
        'Matplotlib is currently using agg, which is a non-GUI backend, '
        'so cannot show the figure.|'
        'FigureCanvasAgg is non-interactive, and thus cannot be shown'
    ),
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
    'erbsland.sphinx.ansi',
    'jupyter_sphinx',
    'notfound.extension',
    'numpydoc',
    'pyvista.ext._autoenum',
    'pyvista.ext._autoinherit',
    'pyvista.ext.plot_directive',
    'sphinx_autoopengraph',
    'sphinx_examples_as_code',
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
    'sphinxcontrib.programoutput',
    'sphinx_togglebutton',
    'sphinx_tags',
    'sphinx_toolbox.more_autodoc.overloads',
    'sphinx_toolbox.more_autodoc.typevars',
    'sphinx_toolbox.more_autodoc.autonamedtuple',
    'sphinxext.opengraph',
    'sphinx_sitemap',
    'sphinx_vtk_xref',
]


# Configuration for sphinx.ext.duration: report in the build log, skip the JSON file
duration_n_slowest = 50
duration_write_json = None


# Configuration for sphinx.ext.autodoc
# Do not expand following type aliases when generating the docs
autodoc_type_aliases = {
    'CameraPositionOptions': 'pyvista.CameraPositionOptions',
    'JupyterBackendOptions': 'pyvista.JupyterBackendOptions',
    'MeshValidationFields': 'pyvista.MeshValidationFields',
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

# Enable ANSI coloring for programoutput, using erbsland.sphinx.ansi
programoutput_use_ansi = True

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

sphinx_examples_as_code_conf = {
    # Replace sphinx-gallery's own per-example download footer/note with
    # this extension's nicer, cross-reference-aware .py/.ipynb downloads.
    'gallery_downloads': True,
}

# Disable checking if vtk links resolve correctly, web checks can be unstable
vtk_xref_nitpicky = False

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
    (r'py:.*', '.*InteractorStyleHandler'),
    (r'py:.*', '.*WriterHandler'),
    (r'py:.*', '.*ReaderHandler'),
    (r'py:.*', '.*ReaderProvider'),
    (r'py:.*', '.*_T_Provider'),
    (r'py:.*', '.*BoundsLike'),
    (r'py:.*', '.*RotationLike'),
    (r'py:.*', '.*CellsLike'),
    (r'py:.*', '.*ShapeLike'),
    (r'py:.*', '.*NumpyArray'),
    (r'py:.*', '.*MeshValidationFields'),
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
    (r'py:.*', '.*_T_Output_co'),
    (r'py:.*', '.*_WrappableVTKDataObjectType'),
    (r'py:.*', '.*_VTKWriterType'),
    (r'py:.*', '.*NormalsLiteral'),
    (r'py:.*', '.*_CellQualityLiteral'),
    (r'py:.*', '.*_CompressionOptions'),
    (r'py:.*', '.*_SENTINEL'),
    (r'py:.*', '.*T'),
    (r'py:.*', '.*Options'),
    # Python 3.14 typing internals leaked through get_type_hints() on
    # forward-refs inside Union aliases (e.g., 'Color' inside ColorLike).
    (r'py:.*', 'TypeAliasForwardRef'),
    #
    # Dataset-related types
    (r'py:.*', '.*DataSet'),
    (r'py:.*', '.*DataObject'),
    (r'py:.*', '.*PolyData'),
    (r'py:.*', '.*UnstructuredGrid'),
    (r'py:.*', '.*_TypeMultiBlockLeaf'),
    (r'py:.*', '.*Grid'),
    (r'py:.*', '.*PointGrid'),
    (r'py:.*', '.*_PointSetBase'),
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
    # PyVista shader/plotting enums
    (r'py:.*', '.*ShaderType'),
    (r'py:.*', '.*PointSpriteShape'),
    (r'py:.*', '.*StereoType'),
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
    (r'py:.*', '.*MeshValidationReport'),
    (r'py:.*', '.*CellQualityInfo'),
    (r'py:.*', 'cycler.Cycler'),
    (r'py:.*', 'pyvista.PVDDataSet'),
    (r'py:.*', 'pyvista.SeriesDataSet'),
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
    (r'py:.*', '_Dimensionality'),
    #
    # Built-in python types. TODO: Fix links (intersphinx?)
    (r'py:.*', '.*BytesIO'),
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
    (r'py:.*', 'ndarray'),
    #
    # pyarrow does not register a py:module entry in its intersphinx
    # inventory, so ``:mod:`pyarrow``` cannot be resolved even when the
    # inventory is loaded. ``pyarrow.Table`` is registered as a py:class
    # and resolves normally.
    (r'py:mod', 'pyarrow'),
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
    (r'py:.*', '.*VtkEvent'),
    (r'py:.*', 'vtk.*'),
    (r'py:.*', '_vtk.*'),
    (r'py:.*', 'VTK'),
    #
    # Misc general ignores
    (r'py:.*', 'optional'),
    #
    # Private implementation types used in signatures
    (r'py:.*', r'.*_SMPToolsContext'),
    (r'py:.*', r'.*_ActiveArrayExistsInfoTuple'),
    #
    # Private algorithm classes returned by plotting utility functions
    (r'py:.*', r'.*ActiveScalarsAlgorithm'),
    (r'py:.*', r'.*AddIDsAlgorithm'),
    (r'py:.*', r'.*CallbackFilterAlgorithm'),
    (r'py:.*', r'.*CrinkleAlgorithm'),
    (r'py:.*', r'.*PointSetToPolyDataAlgorithm'),
    (r'py:.*', r'.*SmoothShadingAlgorithm'),
    (r'py:.*', r'.*SourceAlgorithm'),
    (r'py:.*', r'pyvista\.Common'),
    #
    # Long-form function paths used in some docstrings/examples
    (r'py:.*', r'pyvista\.core\.utilities\.features\.perlin_noise'),
    (r'py:.*', r'pyvista\.core\.utilities\.features\.sample_function'),
]


add_module_names = False
toc_object_entries_show_parents = 'hide'

# Intersphinx mapping
# NOTE: if these are changed, then doc/intersphinx/update.sh
# must be changed accordingly to keep auto-updated mappings working
intersphinx_mapping = {
    'python': (
        'https://docs.python.org/3.11/',
        ('../intersphinx/python-objects.inv',),
    ),  # Pin Python 3.11. See https://github.com/pyvista/pyvista/pull/5018 .
    'scipy': (
        'https://docs.scipy.org/doc/scipy/',
        ('../intersphinx/scipy-objects.inv',),
    ),
    'numpy': ('https://numpy.org/doc/stable/', ('../intersphinx/numpy-objects.inv',)),
    'matplotlib': (
        'https://matplotlib.org/stable/',
        ('../intersphinx/matplotlib-objects.inv',),
    ),
    'imageio': (
        'https://imageio.readthedocs.io/en/stable/',
        ('../intersphinx/imageio-objects.inv',),
    ),
    'pandas': (
        'https://pandas.pydata.org/pandas-docs/stable',
        ('../intersphinx/pandas-objects.inv',),
    ),
    'pyarrow': (
        'https://arrow.apache.org/docs/',
        ('../intersphinx/pyarrow-objects.inv',),
    ),
    'pytest': ('https://docs.pytest.org/en/stable/', ('../intersphinx/pytest-objects.inv',)),
    'pyvistaqt': ('https://qt.pyvista.org/', ('../intersphinx/pyvistaqt-objects.inv',)),
    'pyvista_validation': (
        'https://validation.pyvista.org/',
        ('../intersphinx/pyvista-validation-objects.inv',),
    ),
    'trimesh': ('https://trimesh.org', ('../intersphinx/trimesh-objects.inv',)),
}
intersphinx_timeout = 5

# Select if we want to generate production or dev documentation
#
# Generate class table auto-summary when enabled. This generates one page per
# class method or attribute and should be used with the production
# documentation, but local builds and PR commits can get away without this as
# it takes ~4x as long to generate the documentation.
templates_path = ['_templates']

# Autosummary configuration
autosummary_context = {
    # Methods that should be skipped when generating the docs
    # __init__ should be documented in the class docstring
    # override is a VTK method
    # check_attribute is an undocumented hook used by DisableVtkSnakeCase
    'skipmethods': ['__init__', 'override', 'check_attribute'],
    # Used by _templates/autosummary/class.rst: see pyvista/ext/_autoinherit.py for how
    # each member is routed to exactly one class page.
    'own_members': own_members,
    'inherited_member_rows': inherited_member_rows,
    'filter_member_rows': filter_member_rows,
    # Used by _templates/autosummary/enum.rst: autosummary does not populate `attributes`
    # for the `enum` objtype the way it does for `class`, so enum.rst asks these directly.
    'instance_property_names': instance_property_names,
    'metaclass_property_names': metaclass_property_names,
    'metaclass_property_descriptions': metaclass_property_descriptions,
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
version = pv.__version__

# The full version, including alpha/beta/rc tags.
release = pv.__version__


# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
    '_templates*',
    # Fragments that only ever get ``.. include::``-ed into another page or into a
    # docstring. Sphinx exempts included files from the "isn't included in any
    # toctree" warning but still builds them as standalone documents, which puts 70
    # title-less pages in the search index, e.g. searching `bunny` returns four
    # `<no title>` dataset-gallery carousels. Excluding them keeps the text
    # searchable through the page that includes it.
    'api/core/cell_quality/*.rst',
    'api/examples/dataset-gallery/*.rst',
    'api/plotting/charts/pen_line_styles.rst',
    'api/plotting/charts/plot_color_schemes.rst',
    'api/plotting/charts/scatter_marker_styles.rst',
    'api/readers/readers_table.rst',
    'api/utilities/color_table/*.rst',
    'api/utilities/colormap_table/*.rst',
    'api/utilities/io_table/*.rst',
    'api/utilities/mesh_io.rst',
]
_repo_context7 = Path(__file__).resolve().parents[2] / 'context7.json'
_docs_context7 = Path(__file__).parent / '_extra' / 'context7.json'
_context7_data = json.loads(_repo_context7.read_text())
_context7_data['url'] = 'https://context7.com/websites/pyvista'
_docs_context7.write_text(json.dumps(_context7_data, indent=2) + '\n')

html_extra_path = ['_extra']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'friendly'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Sphinx Gallery Options
from sphinx_gallery.sorting import FileNameSortKey


def _filter_sphinx_gallery_warnings():
    import warnings

    # Ignore specific warnings
    warnings.filterwarnings(
        'ignore',
        message='Call to deprecated method GetData',  # emitted by trame-vtk
        category=DeprecationWarning,
    )
    # Matplotlib >=3.10 emits this when plt.show() runs under a non-interactive
    # backend inside sphinx-gallery workers.
    warnings.filterwarnings(
        'ignore',
        message='FigureCanvasAgg is non-interactive, and thus cannot be shown',
        category=UserWarning,
    )

    # Treat all remaining warnings as errors
    warnings.simplefilter('error', append=True)


class ResetPyVista:
    """Reset pyvista module to default settings."""

    def __call__(self, gallery_conf, fname):  # noqa: ARG002
        """Reset pyvista module to default settings.

        If default documentation settings are modified in any example, reset here.
        """
        _filter_sphinx_gallery_warnings()
        import matplotlib as mpl  # must import before pyvista

        # clear all mpl figures, force non-interactive backend, and reset defaults
        mpl.use('Agg', force=True)
        mpl.pyplot.close('all')
        mpl.rcdefaults()
        mpl.pyplot.figure().clear()
        mpl.pyplot.close()

        import pyvista as pv

        pv._wrappers['vtkPolyData'] = pv.PolyData
        pv.set_plot_theme('document_build')

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
    # AutoCodeLinkScraper adds hyperlinks inside code blocks to pyvista methods.
    'image_scrapers': (DynamicScraper(), AutoCodeLinkScraper(), 'matplotlib'),
    'first_notebook_cell': '%matplotlib inline',
    'reset_modules': (reset_pyvista,),
    'reset_modules_order': 'both',
    'junit': str(Path('sphinx-gallery') / 'junit-results.xml'),
    'parallel': True,  # use the same number of workers as "-j" in sphinx
}

suppress_warnings = [
    'config.cache',
    'image.not_readable',
]

import re

# -- .. pyvista-plot:: directive ----------------------------------------------
from jinja2.sandbox import SandboxedEnvironment
from numpydoc.docscrape import NumpyDocString
from numpydoc.docscrape_sphinx import SphinxDocString

# Also matches submodule imports, e.g. ``from pyvista.examples.cells import ...``.
IMPORT_PYVISTA_RE = r'\b(import +pyvista|from +pyvista(\.[\w.]+)? +import)\b'
IMPORT_MATPLOTLIB_RE = r'\b(import +matplotlib|from +matplotlib +import)\b'

pyvista_plot_setup = """
from pyvista import set_plot_theme as __s_p_t
__s_p_t('document_build')
del __s_p_t
"""
pyvista_plot_cleanup = pyvista_plot_setup

# Hyperlink identifiers in ``.. pyvista-plot::`` output to their documented targets.
pyvista_plot_autocodelink = True

# Append a "Used In" backreferences section to every autodoc-documented object's own
# docstring (empty ones get nothing appended, not "No references found.").
autocodelink_autodoc_backrefs = True

# Rename backreferences group headings.
autocodelink_category_labels = {
    'Sphinx Gallery': 'Gallery Examples',
    'Docstring Examples': 'Docstring Examples',
    'Documentation': 'Guides',
}

# show gallery examples last, not alphabetically by heading
autocodelink_category_order = ['Documentation', 'Docstring Examples', 'Sphinx Gallery']

# rank "Used In" entries by usage frequency
autocodelink_sort = 'frequency'

# show each entry's usage count alongside it
autocodelink_show_usage_count = True

# render gallery backreferences as thumbnail cards
autocodelink_gallery_cards = True

# execute and record ``.. jupyter-execute::`` cells so their identifiers link too
autocodelink_jupyter_blocks = True


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


# -- "See Also" as a real section, not a `.. seealso::` admonition ------------
# SphinxDocString's own _str_see_also always wraps the base rendering in
# `.. seealso::`, which isn't a real docutils section: no heading, unlinkable,
# invisible to the "on this page" navbar. The un-wrapped base rendering already
# goes through self._str_header, so skipping the wrap turns it into a real
# heading for free, the same way _str_header below already does for Notes,
# References, and Examples.
def _str_see_also(self, func_role):
    return NumpyDocString._str_see_also(self, func_role)


SphinxDocString._str_see_also = _str_see_also


# -- docstring section order: Parameters, ..., Examples, See Also -------------
# numpydoc's own template puts "See Also" right after the parameter-ish sections and
# well before Notes/Examples. Move it to the very end instead -- sphinx-autocodelink's
# "Used In" (appended after the whole docstring renders, via autodoc-process-docstring)
# always lands after whatever numpydoc itself renders last, so this also puts "See
# Also" directly before "Used In".  Identical to numpydoc's own template otherwise --
# see numpydoc/templates/numpydoc_docstring.rst -- just with {{see_also}} moved down.
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
# numpydoc renders section headers (Notes, References, Examples) as
# `.. rubric::` by default. Rubrics aren't real docutils sections, so they're
# invisible to the "on this page" navbar, which is built from actual heading
# structure. Since pyvista generates one dedicated page per function/method/
# class (see doc/source/_templates/autosummary/*.rst), each page has at most
# one docstring, so promoting these to real headings doesn't create the
# duplicate-heading clutter it would on a page combining many docstrings.
def _str_header(self, name):  # noqa: ARG001
    return [name, '-' * len(name), '']


SphinxDocString._str_header = _str_header


# Making the sections above real headings isn't enough on its own: autodoc wraps
# the whole docstring in a `desc` node, and Sphinx's TocTreeCollector explicitly
# skips any section it finds inside one. Lift them out to page level so they show
# up in the navbar.
def _is_nested_desc(node: Element) -> bool:
    parent = node.parent
    while parent is not None:
        if isinstance(parent, addnodes.desc):
            return True
        parent = parent.parent
    return False


def promote_seealso_admonitions(app: Sphinx, doctree: Element) -> None:  # noqa: ARG001
    """Turn a literal ``.. seealso::`` admonition in a docstring into a real section.

    Some docstrings (e.g. ``pyvista.examples.downloads``) write ``.. seealso::``
    directly rather than using numpydoc's own "See Also" section syntax. Left as
    an admonition it has no heading, isn't linkable, and is invisible to the "on
    this page" navbar -- the same problem fixed above for numpydoc's own "See
    Also" section by not wrapping it in one. Converting it to a section here lets
    hoist_docstring_sections below lift it to page level the same way.

    A literal ``.. seealso::`` is written wherever the docstring author put it --
    usually right before References/Examples -- unlike numpydoc's own "See Also",
    which _DOCSTRING_TEMPLATE above always renders last. Reposition the promoted
    section to match: directly before "Used In" if present, otherwise at the end.

    "Used In" isn't necessarily a sibling: nothing closes off the docstring's last
    heading before its directive runs, so it lands nested inside that heading's
    section instead, and only reads as a sibling once hoist_docstring_sections
    below extracts every section it finds. Search at any depth to still find it.
    """
    for admonition in list(doctree.findall(addnodes.seealso)):
        if not _is_nested_desc(admonition):
            continue
        section = nodes.section()
        section += nodes.title(text='See Also')
        section.extend(admonition.children)
        doctree.note_implicit_target(section, section)

        container = admonition.parent
        while container is not None and not isinstance(container, addnodes.desc_content):
            container = container.parent

        admonition.replace_self(section)
        section.parent.remove(section)

        used_in = (
            next(
                (s for s in container.findall(nodes.section) if s[0].astext() == 'Used In'),
                None,
            )
            if container is not None
            else None
        )
        if used_in is not None:
            used_in.parent.insert(used_in.parent.index(used_in), section)
        else:
            (container if container is not None else admonition.parent).append(section)


def hoist_docstring_sections(app: Sphinx, doctree: Element) -> None:  # noqa: ARG001
    """Move docstring sections out of their ``desc`` node to page level.

    Finds sections at any depth inside ``desc_content``, not just its direct
    children: a section appended after numpydoc's own Examples section (e.g.
    sphinx-autocodelink's "Used In", via ``autodoc-process-docstring``) lands
    *inside* Examples' own section rather than beside it, since nothing closed
    Examples' heading first. Hoisting it from wherever it actually is keeps it
    a sibling of Notes/References/Examples/etc., not a subsection of one of
    them.
    """
    for desc in list(doctree.findall(addnodes.desc)):
        if _is_nested_desc(desc):
            continue
        parent = desc.parent
        if parent is None:
            continue
        # Only hoist when this object owns the page, otherwise sections from
        # several objects would collide at page level.
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
    # Capping at depth 4 keeps classes nested under their section pages while
    # avoiding an O(N^2) sidebar render across ~2,700 method-level entries.
    'max_navbar_depth': 4,
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
            'name': 'The Paper',
            'url': 'https://doi.org/10.21105/joss.01450',
            'icon': 'fa fa-file-text fa-fw',
        },
        {
            'name': 'PyPI',
            'url': 'https://pypi.org/project/pyvista',
            'icon': 'fa-brands fa-python',
        },
    ],
}

if 'dev' in pv.__version__:
    stable_base = 'https://docs.pyvista.org'
    announcement_html = f"""
    <div class="pv-announcement">
        This is documentation for an <strong>unstable development version</strong>
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
    'codimensional.css',  # pin partner card to bottom of right sidebar
    'jupyter_sphinx_theme.css',  # make jupyter-sphinx containers follow the dark mode toggle
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
        'Perhaps try the <a href="https://docs.pyvista.org/examples/index.html">examples page</a>.'
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

html_sidebars = {
    '**': [
        'navbar-logo.html',
        'icon-links.html',
        'search-button-field.html',
        'sbt-sidebar-nav.html',
    ],
}

# Pin the CoDimensional PBC partner card to the bottom of the right
# (secondary) sidebar, below the page table of contents, on every page.
html_theme_options['secondary_sidebar_items'] = [
    'page-toc.html',
    'codimensional.html',
]


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
    # Priority must stay above the 501 used by sphinx-book-theme's
    # ``add_source_buttons``, which is what builds the "suggest edit" button.
    app.connect('html-page-context', pv_html_page_context, priority=502)

    # priority < 500 so this runs before Sphinx's TocTreeCollector builds the toc, and
    # before priority 400 so hoist_docstring_sections below sees the promoted sections
    app.connect('doctree-read', promote_seealso_admonitions, priority=300)
    app.connect('doctree-read', hoist_docstring_sections, priority=400)

    # right before writing, patch the gallery placeholders
    app.connect('doctree-resolved', make_tables.patch_gallery_placeholders)

    # feeds the search result snippets rendered by search_summaries.js
    app.connect('build-finished', make_search_summaries.dump_search_summaries)

    app.add_css_file('copybutton.css')
    app.add_css_file('no_search_highlight.css')
    app.add_css_file('dataset_gallery_filter.css')
    app.add_js_file('redirect_fragments.js')
    app.add_js_file('dataset_gallery_filter.js')
