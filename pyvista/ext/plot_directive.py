"""Plot directive module.

A directive for including a PyVista plot in a Sphinx document.

The ``.. pyvista-plot::`` sphinx directive will include an inline
``.png`` image.

The source code for the plot may be included in one of two ways:

1. Using **doctest** syntax::

    .. pyvista-plot::

       >>> import pyvista as pv
       >>> sphere = pv.Sphere()
       >>> out = sphere.plot()

2. **A path to a source file** as the argument to the directive::

     .. pyvista-plot:: path/to/plot.py

   When a path to a source file is given, the content of the
   directive may optionally contain a caption for the plot::

     .. pyvista-plot:: path/to/plot.py

        The plot's caption.

   Additionally, one may specify the name of a function to call (with
   no arguments) immediately after importing the module::

     .. pyvista-plot:: path/to/plot.py plot_function1

.. note::
   Code blocks containing ``doctest:+SKIP`` will be skipped.

.. note::
   Animations will not be saved, only the last frame will be shown.


**Options**
The ``pyvista-plot`` directive supports the following options:

    include-source : bool
        Whether to display the source code. The default can be changed
        using the ``pyvista_plot_include_source`` variable in :file:`conf.py`.

    encoding : str
        If this source file is in a non-UTF8 or non-ASCII encoding, the
        encoding must be specified using the ``:encoding:`` option.  The
        encoding will not be inferred using the ``-*- coding -*-`` metacomment.

    context : None
        If provided, the code will be run in the context of all previous plot
        directives for which the ``:context:`` option was specified.  This only
        applies to inline code plot directives, not those run from files.

    nofigs : None
        When setting this flag, the code block will be run but no figures will be
        inserted.  This is usually useful with the ``:context:`` option.

    caption : str
        If specified, the option's argument will be used as a caption for the
        figure. This overwrites the caption given in the content, when the plot
        is generated from a file.

    force_static : None
        When setting this flag, static images will be used instead of an
        interactive scene.

    skip : bool, default: True
        Whether to skip execution of this directive. If no argument is provided
        i.e., ``:skip:``, then it defaults to ``:skip: true``.  Default
        behaviour is controlled by the ``pyvista_plot_skip`` boolean variable in
        :file:`conf.py`.  Note that, if specified, this option overrides the
        ``pyvista_plot_skip`` configuration.

    optional : None
        This flag marks the directive for *conditional* execution. Whether the
        directive is executed is controlled by the ``pyvista_plot_skip_optional``
        boolean variable in :file:`conf.py`.

Additionally, this directive supports all the options of the `image`
directive, except for *target* (since plot will add its own target).  These
include *alt*, *height*, *width*, *scale*, *align*.


**Open Graph previews**

When `sphinxext-opengraph <https://github.com/wpilibsuite/sphinxext-opengraph>`_
is enabled, every page that renders at least one ``pyvista-plot`` image also gets
that image as its ``og:image`` link preview. This requires no configuration beyond
the usual ``ogp_site_url``; see ``pyvista_plot_opengraph`` below to opt out.

By default the *first* image rendered on the page is used. Use the
``pyvista-plot-thumbnail`` directive to pick a different one::

    .. pyvista-plot-thumbnail:: 2

The argument is the one-based position of the image among *all* plot images on
the page, counting in document order and ignoring how the underlying files
happen to be named. Negative values count backwards from the last image. Unlike
``sphinxext-opengraph``'s own ``:og:image:`` field, this directive is not
restricted to the top of the page, so it can be written next to the code it
refers to. In a docstring the natural place is the start of the ``Examples``
section::

    .. pyvista-plot-thumbnail:: 2

    Create a sphere.

    >>> import pyvista as pv
    >>> pv.Sphere().plot()

    And a cube, which is the image used for link previews.

    >>> pv.Cube().plot()

The directive renders nothing. Open Graph metadata is per-document, so a page can
only have one link preview: using the directive more than once on a page warns and
keeps the first selection. This can happen without either docstring being wrong,
on pages that document several objects at once via ``:members:``.

Sphinx-Gallery examples are handled separately: their ``og:image`` always follows
the gallery's own thumbnail selection -- the full resolution version of it -- so
that link previews match the gallery. Using ``pyvista-plot-thumbnail`` in a gallery
example is an error; use ``# sphinx_gallery_thumbnail_number = <number>`` instead.


**Configuration options**

.. versionchanged:: 0.45
   Prior to v0.45, these directives conflicted with ``matplotlib``. All
   directives have been prepended with ``pyvista_``.

The plot directive has the following configuration options:

    pyvista_plot_include_source : bool, default: True
        Default value for the ``include-source`` directive option.
        Default is ``True``.

    pyvista_plot_basedir : str
        Base directory, to which ``plot::`` file names are relative
        to.  If ``None`` or unset, file names are relative to the
        directory where the file containing the directive is.

    pyvista_plot_html_show_formats : bool, default: True
        Whether to show links to the files in HTML. Default ``True``.

    pyvista_plot_template : str
        Provide a customized Jinja2 template for preparing restructured text.

    pyvista_plot_setup : str
        Python code to be run before every plot directive block.

    pyvista_plot_cleanup : str
        Python code to be run after every plot directive block.

    pyvista_plot_skip : bool, default: False
        Default value for the ``skip`` directive option.

    pyvista_plot_skip_optional : bool, default: False
        Whether to skip execution of ``optional`` directives.

    pyvista_plot_opengraph : bool or None, default: None
        Set each page's Open Graph ``og:image`` from the images rendered on that
        page. When ``None``, this is enabled automatically if ``sphinxext.opengraph``
        is enabled. Set to ``True`` or ``False`` to explicitly opt in or out.

These options can be set by defining global variables of the same name in
:file:`conf.py`.


**Directive Configuration Settings**

Globally, you can set if the file names should be either:

* Deterministic, based on directive source hash:
  ``<BASENAME>-<HASH>_<INDEX>_<SUBINDEX>.<EXT>`` (Default)
* Indexed, based on location in document:
  ``<BASENAME>-<DOC-INDEX>_<INDEX>_<SUBINDEX>.<EXT>``

Enable indexed naming this by setting ``pyvista_plot_use_counter=True``. Note
that indexed is incompatible with parallel builds due to race conditions.

.. versionchanged:: 0.47
    Hash-based image naming is now used by default.

.. versionchanged:: 0.49
    Generated source code is now wrapped in a ``.. container:: pyvista-plot-source``
    node. This has no effect on rendering, but allows other independent tooling
    (e.g. the https://github.com/pyvista/sphinx-examples-as-code Sphinx extension)
    to reliably locate this directive's generated code within a page.

.. versionadded:: 0.49
    Open Graph link previews. See ``pyvista_plot_opengraph`` and the
    ``pyvista-plot-thumbnail`` directive.

"""

from __future__ import annotations

import doctest
import hashlib
import os
from pathlib import Path
import posixpath
import re
import shutil
import textwrap
import traceback
from typing import TYPE_CHECKING
from typing import ClassVar
import urllib.parse

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
from docutils.parsers.rst.directives.images import Image
import jinja2  # Sphinx dependency.
from sphinx.util import logging

# must enable BUILDING_GALLERY to keep windows active
# enable offscreen to hide figures when generating them.
import pyvista as pv
from pyvista.ext import _opengraph

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from sphinx.application import Sphinx
    from sphinx.config import Config


pv.BUILDING_GALLERY = True
pv.OFF_SCREEN = True

logger = logging.getLogger(__name__)

# CSS class marking the ``.. container::`` node that wraps this directive's generated source code
_PLOT_SOURCE_CLASS = 'pyvista-plot-source'

# Name of the directory this directive renders its images into. Sphinx rewrites image
# URIs at write time, but keeps the original path in ``image['candidates']``, which is
# how the Open Graph integration tells this directive's images apart from any others.
_PLOT_BUILD_DIRNAME = 'pyvista_plot_directive'

# Document attribute holding the ``pyvista-plot-thumbnail`` argument for the page
_THUMBNAIL_NUMBER = '_pyvista_plot_thumbnail_number'


# -----------------------------------------------------------------------------
# Registration hook
# -----------------------------------------------------------------------------


def _option_boolean(arg) -> bool:
    if not arg or not arg.strip():
        # no argument given, assume used as a flag
        return True
    elif arg.strip().lower() in ('no', '0', 'false'):
        return False
    elif arg.strip().lower() in ('yes', '1', 'true'):
        return True
    else:  # pragma: no cover
        msg = f'"{arg}" unknown boolean'
        raise ValueError(msg)


def _option_context(arg):
    if arg is not None:  # pragma: no cover
        msg = 'No arguments allowed for ``:context:``'
        raise ValueError(msg)


def _option_format(arg):
    return directives.choice(arg, ('python', 'doctest'))


class PlotDirective(Directive):
    """The ``.. pyvista-plot::`` directive, as documented in the module's docstring."""

    has_content = True
    required_arguments = 0
    optional_arguments = 2
    final_argument_whitespace = False
    option_spec: ClassVar[dict[str, Callable]] = {
        'alt': directives.unchanged,
        'height': directives.length_or_unitless,
        'width': directives.length_or_percentage_or_unitless,
        'scale': directives.nonnegative_int,
        'align': Image.align,
        'include-source': _option_boolean,
        'format': _option_format,
        'context': _option_context,
        'nofigs': directives.flag,
        'encoding': directives.encoding,
        'caption': directives.unchanged,
        'force_static': directives.flag,
        'skip': _option_boolean,
        'optional': directives.flag,
    }

    def run(self):
        """Run the plot directive."""
        try:
            return run(
                self.arguments,
                self.content,
                self.options,
                self.state_machine,
                self.state,
                self.lineno,
            )
        except Exception as e:  # noqa: BLE001  # pragma: no cover
            raise self.error(str(e))


class PlotThumbnailDirective(Directive):
    """The ``.. pyvista-plot-thumbnail::`` directive.

    Selects which of the page's plot images is used as its Open Graph image. See
    the module's docstring.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False

    def run(self):
        """Record the page's thumbnail number and render nothing."""
        document = self.state_machine.document
        env = document.settings.env
        argument = self.arguments[0].strip()
        try:
            number = int(argument)
        except ValueError:
            msg = f"'pyvista-plot-thumbnail' expects an integer, got {argument!r}."
            raise self.error(msg)
        if number == 0:
            msg = (
                "'pyvista-plot-thumbnail' is one-based, so 0 is not a valid image number. "
                'Use 1 for the first image, or -1 for the last one.'
            )
            raise self.error(msg)
        if _is_sphinx_gallery_document(env.app, env.docname):
            raise self.error(_gallery_thumbnail_error(number))
        if _THUMBNAIL_NUMBER in document.attributes:
            # A warning rather than an error: Open Graph metadata is per-document, so a
            # page documenting several objects collides even when each of their
            # docstrings is correct on its own generated page.
            logger.warning(
                'this page already selects plot image %d as its Open Graph image, and a '
                'page can only have one. Ignoring this selection of image %d.',
                document.attributes[_THUMBNAIL_NUMBER],
                number,
                location=(env.docname, self.lineno),
                type='pyvista',
                subtype='plot_thumbnail',
            )
            return []
        document.attributes[_THUMBNAIL_NUMBER] = number
        return []


def _gallery_thumbnail_error(number: int) -> str:
    """Return guidance for choosing a Sphinx-Gallery example's thumbnail."""
    return (
        "'pyvista-plot-thumbnail' cannot be used in a Sphinx-Gallery example, because its "
        'Open Graph image always follows the gallery thumbnail. '
        f"Use '# sphinx_gallery_thumbnail_number = {number}' instead."
    )


def setup(app: Sphinx):
    """Set up the plot directive."""
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir
    app.add_directive('pyvista-plot', PlotDirective)
    app.add_directive('pyvista-plot-thumbnail', PlotThumbnailDirective)

    legacy_keys = [
        'plot_include_source',
        'plot_basedir',
        'plot_html_show_formats',
        'plot_template',
        'plot_setup',
        'plot_cleanup',
        'plot_skip',
        'plot_skip_optional',
    ]

    def raise_on_legacy_config(app: Sphinx, config: Config) -> None:
        """Raise a RuntimeError when using legacy configuration parameters.

        These parameters conflict with matplotlib's ``plot_directive``.

        """
        uses_matplotlib = 'matplotlib.sphinxext.plot_directive' in app.extensions

        if not uses_matplotlib:  # pragma: no cover
            for key in legacy_keys:
                if getattr(config, key, None) is not None:
                    msg = (
                        f"Sphinx config uses deprecated '{key}' without 'pyvista_' prefix. "
                        f"Rename it to 'pyvista_{key}"
                    )
                    raise RuntimeError(msg)

    app.connect('config-inited', raise_on_legacy_config)

    def check_counter_for_parallel_build(app: Sphinx, config: Config) -> None:
        if config.pyvista_plot_use_counter and app.parallel > 1:
            msg = (
                "The 'pyvista_plot_use_counter' option cannot be enabled for parallel builds."
                " Set 'pyvista_plot_use_counter = False' in your conf.py"
                ' or disable parallel builds.'
            )
            raise RuntimeError(msg)

    # Connect the new function to the 'config-inited' event
    app.connect('config-inited', check_counter_for_parallel_build)

    _opengraph.add_auto_config_value(app, 'pyvista_plot_opengraph')
    # Must run before ``sphinxext.opengraph`` renders its tags at the default priority
    app.connect('html-page-context', _set_opengraph_image, priority=400)

    app.add_config_value('pyvista_plot_use_counter', False, 'env')
    app.add_config_value('pyvista_plot_include_source', True, False)
    app.add_config_value('pyvista_plot_basedir', None, True)
    app.add_config_value('pyvista_plot_html_show_formats', True, True)
    app.add_config_value('pyvista_plot_template', None, True)
    app.add_config_value('pyvista_plot_setup', None, True)
    app.add_config_value('pyvista_plot_cleanup', None, True)
    app.add_config_value(name='pyvista_plot_skip', default=False, rebuild='html')
    app.add_config_value(name='pyvista_plot_skip_optional', default=False, rebuild='html')
    return {
        'parallel_read_safe': True,
        'parallel_write_safe': True,
        'version': pv.__version__,
    }


# -----------------------------------------------------------------------------
# Open Graph images
# -----------------------------------------------------------------------------


def _set_opengraph_image(  # noqa: PLR0917
    app: Sphinx,
    pagename: str,
    templatename: str,  # noqa: ARG001
    context: dict,
    doctree: nodes.document | None,
) -> None:
    """Point the page's ``og:image`` at the plot it renders.

    This runs at write time rather than while reading, because that is the first
    point at which an image's final ``_images`` filename is known: hash-based
    naming, parallel reads and Sphinx's own de-duplication all mean a directive
    cannot predict where its output ends up.
    """
    if doctree is None or not app.config.pyvista_plot_opengraph:
        return
    fields = _opengraph.page_fields(app, context)
    if fields is None or 'og:image' in fields:
        return

    if _is_sphinx_gallery_document(app, pagename):
        image = _gallery_opengraph_image(app, pagename, doctree)
    else:
        image = _plot_opengraph_image(app, pagename, doctree)
    if image is not None:
        fields['og:image'] = image


def _plot_opengraph_image(app: Sphinx, docname: str, doctree: nodes.document) -> str | None:
    """Return the URL of the plot image selected by a page."""
    images = [image for image in _image_nodes(doctree) if _is_plot_directive_image(image)]
    if not images:
        return None

    number = doctree.get(_THUMBNAIL_NUMBER, 1)
    index = number - 1 if number > 0 else number
    if not -len(images) <= index < len(images):
        # Not fatal: a build that skips plots legitimately renders fewer images than
        # the page selects from, so fall back to the first image either way
        if not (app.config.pyvista_plot_skip or app.config.pyvista_plot_skip_optional):
            logger.warning(
                "'pyvista-plot-thumbnail' selects image %d, but this page only renders "
                '%d plot image(s). Using the first one.',
                number,
                len(images),
                location=docname,
                type='pyvista',
                subtype='plot_thumbnail',
            )
        index = 0
    # Sphinx has already rewritten the URI to the image's path relative to this page
    return _absolute_url(app, docname, images[index]['uri'])


def _gallery_opengraph_image(app: Sphinx, docname: str, doctree: nodes.document) -> str | None:
    """Return the URL of the image Sphinx-Gallery uses as an example's thumbnail.

    The full resolution image is preferred over the gallery's own thumbnail file,
    which is too small to make a good link preview, but it is always the same image
    the gallery shows.
    """
    source = Path(app.env.doc2path(docname))
    number, path = _gallery_thumbnail_selection(source.with_suffix('.py'))
    if path is None:
        prefix = f'sphx_glr_{source.stem}_'
        images = [
            image
            for image in _image_nodes(doctree)
            if posixpath.basename(image['uri']).startswith(prefix)
        ]
        index = number - 1 if number > 0 else number
        if -len(images) <= index < len(images):
            # Sphinx-Gallery copies its images into the output verbatim
            return _absolute_url(app, docname, _output_image_path(app, images[index]['uri']))

    # ``sphinx_gallery_thumbnail_path`` and failed examples both leave a thumbnail with
    # no full resolution counterpart on the page
    thumbnails = (source.parent / 'images' / 'thumb').glob(f'sphx_glr_{source.stem}_thumb.*')
    thumbnail = next(thumbnails, None)
    if thumbnail is None:
        return None
    return _absolute_url(app, docname, _output_image_path(app, thumbnail.name))


def _gallery_thumbnail_selection(source: Path) -> tuple[int, str | None]:
    """Return the ``sphinx_gallery_thumbnail_{number,path}`` chosen by an example."""
    try:
        from sphinx_gallery.py_source_parser import extract_file_config  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        return 1, None
    try:
        file_conf = extract_file_config(source.read_text(encoding='utf-8'))
    except OSError:
        # Gallery index pages and ``sg_execution_times`` have no example source
        return 1, None
    # A number always wins over a path, matching ``sphinx_gallery.gen_rst.save_thumbnail``
    number = file_conf.get('thumbnail_number')
    if number is None:
        path = file_conf.get('thumbnail_path')
        return 1, None if path is None else str(path)
    return int(number), None


def _image_nodes(doctree: nodes.document) -> Iterator[nodes.Element]:
    """Yield every image-bearing node of a page, in document order.

    Sphinx-Gallery renders its images as ``imgsgnode`` rather than
    :class:`docutils.nodes.image`, so nodes are matched on carrying a ``uri``.
    """
    for node in doctree.findall(nodes.Element):
        if node.get('uri'):
            yield node


def _is_plot_directive_image(node: nodes.Element) -> bool:
    """Return whether an image node was rendered by the plot directive."""
    # ``candidates`` keeps the pre-rewrite path, which still names the build directory
    original = (node.get('candidates') or {}).get('*') or node.get('uri', '')
    return _PLOT_BUILD_DIRNAME in original.replace(os.sep, '/').split('/')


def _output_image_path(app: Sphinx, name: str) -> str:
    """Return the path of an output image, relative to the page being written."""
    return posixpath.join(app.builder.imgpath, posixpath.basename(name))


def _absolute_url(app: Sphinx, docname: str, path: str) -> str:
    """Return the public URL of *path*, which is relative to *docname*."""
    site_url = app.config.ogp_canonical_url or app.config.ogp_site_url
    page_url = urllib.parse.urljoin(site_url, app.builder.get_target_uri(docname))
    return urllib.parse.urljoin(page_url, path)


def _is_sphinx_gallery_document(app: Sphinx, docname: str) -> bool:
    """Return whether *docname* is a generated Sphinx-Gallery document."""
    gallery_conf = getattr(app.config, 'sphinx_gallery_conf', None)
    if not gallery_conf:
        return False

    gallery_dirs = gallery_conf.get('gallery_dirs', ())
    if isinstance(gallery_dirs, str):
        gallery_dirs = (gallery_dirs,)
    directories = [Path(directory).as_posix().strip('/') for directory in gallery_dirs]
    return any(
        docname == directory or docname.startswith(f'{directory}/') for directory in directories
    )


# -----------------------------------------------------------------------------
# Doctest handling
# -----------------------------------------------------------------------------
def _contains_doctest(text):
    try:
        # check if it's valid Python as-is
        compile(text, '<string>', 'exec')
    except SyntaxError:
        pass
    else:
        return False
    r = re.compile(r'^\s*>>>', re.MULTILINE)
    m = r.search(text)
    return bool(m)


def _contains_pyvista_plot(text) -> bool:
    return '.. pyvista-plot::' in text


def _strip_comments(code):
    """Remove comments from a line of python code."""
    return re.sub(r'(?m)^ *#.*\n?', '', code)


def _split_code_at_show(text):
    """Split code at plt.show() or plt.plot().

    Includes logic to deal with edge cases like:

    >>> import pyvista as pv
    >>> pv.Sphere().plot(color='blue', cpos='xy')

    >>> pv.Sphere().plot(color='red', cpos='xy')

    """
    parts = []
    is_doctest = _contains_doctest(text)
    part = []

    within_plot = False
    for line in text.split('\n'):
        part.append(line)

        # check if show(...) or plot(...) is within the line
        line_no_comments = _strip_comments(line)
        if within_plot:  # allow for multi-line plot(...
            if line_no_comments.endswith(')'):
                parts.append('\n'.join(part))
                part = []
                within_plot = False

        elif _show_or_plot_in_string(line_no_comments):
            if line_no_comments.endswith(')'):
                parts.append('\n'.join(part))
                part = []
            else:  # allow for multi-line plot(...
                within_plot = True

    if '\n'.join(part).strip():
        parts.append('\n'.join(part))
    return is_doctest, parts


def _show_or_plot_in_string(string):
    # string contains `.show(`, `.plot(`, or `plot_xyz(` where `xyz` is one
    # or more lower-case letters or underscore, e.g. `plot_cell(`, `plot_datasets(`
    pattern = r'(?:\.plot\(|\.show\(|(?:[ \t\n.]plot_[a-z_]+?)\()'
    return bool(re.search(pattern, string))


# -----------------------------------------------------------------------------
# Template
# -----------------------------------------------------------------------------

TEMPLATE = (
    """
{% if source_code %}
.. container:: """
    + _PLOT_SOURCE_CLASS
    + """

   {{ source_code | indent(3, first=True) }}
{% else %}
{{ source_code }}
{% endif %}

.. only:: html

   {% for img in images %}
   {% if img.extension == 'vtksz' %}

   .. tab-set::

       .. tab-item:: Static Scene

           .. figure:: {{ build_dir }}/{{ img.stem }}.png
              {% for option in options -%}
              {{ option }}
              {% endfor %}


       .. tab-item:: Interactive Scene

           .. offlineviewer:: {{ build_dir }}/{{ img.stem }}.vtksz

   {{ caption }}  {# appropriate leading whitespace added beforehand #}
   {% else %}
   .. figure:: {{ build_dir }}/{{ img.basename }}
      {% for option in options -%}
      {{ option }}
      {% endfor %}

   {{ caption }}  {# appropriate leading whitespace added beforehand #}
   {% endif %}
   {% endfor %}

"""
)

exception_template = """
.. only:: html

   [`source code <%(linkdir)s/%(basename)s.py>`__]

Exception occurred rendering plot.

"""

# the context of the plot for all directives specified with the
# :context: option
plot_context = {}


class ImageFile:
    """Simple representation of an image file path."""

    def __init__(self, dirname, basename):
        """Construct ImageFile."""
        self.basename = basename
        self.dirname = dirname
        self.extension = Path(basename).suffix[1:]

    @property
    def filename(self):
        """Return the filename of this image."""
        return str(Path(self.dirname) / self.basename)

    @property
    def stem(self):
        """Return the basename without the suffix."""
        return Path(self.basename).stem

    def __repr__(self) -> str:  # pragma no cover
        return self.filename


class PlotError(RuntimeError):
    """More descriptive plot error."""


def _run_code(*, code, code_path, ns=None, function_name=None):
    """Run a docstring example.

    Run the example if it does not contain ``'doctest:+SKIP'``, or a
    ```pyvista-plot::`` directive.  In the later case, the doctest parser will
    present the code-block again with the ```pyvista-plot::`` directive
    and its options removed.

    Import a Python module from a path, and run the function given by
    name, if function_name is not None.
    """
    # do not execute code containing any SKIP directives
    if 'doctest:+SKIP' in code:
        return ns

    if 'pyvista-plot::' in code:
        return ns

    try:
        if pv.PLOT_DIRECTIVE_THEME is not None:
            pv.set_plot_theme(pv.PLOT_DIRECTIVE_THEME)  # pragma: no cover
        exec(code, ns)  # noqa: S102
        if function_name is not None:
            ns[function_name]()
    except (Exception, SystemExit) as err:  # pragma: no cover
        # Annotate traceback with source file and line
        tb = traceback.format_exc()
        msg = f'Error in {code_path}:\n{tb}'
        raise PlotError(msg) from err

    return ns


def render_figures(
    *,
    code,
    code_path,
    output_dir,
    output_base,
    context,
    function_name,
    config,
    force_static,
):
    """Run a pyplot script and save the images in *output_dir*.

    Save the images under *output_dir* with file names derived from
    *output_base*. Closed plotters are ignored if they were never
    rendered.
    """
    # We skip snippets that contain the ```pyvista-plot::`` directive as part of their code.
    # The doctest parser will present the code-block once again with the ```pyvista-plot::``
    # directive and its options properly parsed.
    if _contains_pyvista_plot(code):
        is_doctest = True
        code_pieces = [code]
    else:
        # Try to determine if all images already exist
        is_doctest, code_pieces = _split_code_at_show(code)

    # Otherwise, we didn't find the files, so build them
    results = []
    ns = plot_context if context else {}

    # Check for setup and teardown code for plots
    code_setup = config.pyvista_plot_setup
    code_cleanup = config.pyvista_plot_cleanup

    if code_setup:
        _run_code(code=code_setup, code_path=code_path, ns=ns, function_name=function_name)

    try:
        for i, code_piece in enumerate(code_pieces):
            # generate the plot
            _run_code(
                code=doctest.script_from_examples(code_piece) if is_doctest else code_piece,
                code_path=code_path,
                ns=ns,
                function_name=function_name,
            )

            images = []

            if (
                _show_or_plot_in_string(code_piece)
                or '.open_gif' in code_piece
                or 'plot=True' in code_piece
            ):
                figures = pv.plotting.plotter._ALL_PLOTTERS

                for j, (_, plotter) in enumerate(figures.items()):
                    if plotter._gif_filename is not None:
                        image_file = ImageFile(output_dir, f'{output_base}_{i:02d}_{j:02d}.gif')
                        images.append(image_file)
                        shutil.move(plotter._gif_filename, image_file.filename)
                        continue
                    if not plotter._show_called:
                        continue
                    image_file = ImageFile(output_dir, f'{output_base}_{i:02d}_{j:02d}.png')
                    try:
                        plotter.screenshot(image_file.filename)
                    except RuntimeError:  # pragma no cover
                        # ignore closed, unrendered plotters
                        continue
                    if force_static or (plotter.last_vtksz is None):
                        images.append(image_file)
                        continue
                    image_file = ImageFile(output_dir, f'{output_base}_{i:02d}_{j:02d}.vtksz')
                    with Path(image_file.filename).open('wb') as f:
                        f.write(plotter.last_vtksz)

                    images.append(image_file)

            pv.close_all()  # close and clear all plotters

            results.append((code_piece, images))
    finally:
        if code_cleanup:
            _run_code(code=code_cleanup, code_path=code_path, ns=ns, function_name=function_name)

    return results


def _contains_doctest(text: str) -> bool:
    """Check if the text contains doctest markers."""
    r = re.compile(r'^\s*>>>', re.MULTILINE)
    m = r.search(text)
    return bool(m)


def hash_plot_code(code: str, options: dict) -> str:
    """Generate a hash of the plot code."""
    # convert to plain script if doctest code
    script = doctest.script_from_examples(code) if _contains_doctest(code) else code

    lines = []
    for line in script.splitlines():
        line_without_comments = re.sub(r'(?<!["\'])#.*', '', line).strip()
        if line_without_comments:
            lines.append(line_without_comments)
    clean_script = textwrap.dedent('\n'.join(lines))

    parts = [
        'ctx=' + str('context' in options),
        clean_script,
    ]

    # first 16 char should be sufficient
    return hashlib.sha256(''.join(parts).encode('utf-8')).hexdigest()[:16]


def run(arguments, content, options, state_machine, state, lineno):  # noqa: PLR0917
    """Run the plot directive."""
    document = state_machine.document
    config = document.settings.env.config
    nofigs = 'nofigs' in options
    optional = 'optional' in options
    force_static = 'force_static' in options
    use_counter = config.pyvista_plot_use_counter

    default_fmt = 'png'

    options.setdefault('include-source', config.pyvista_plot_include_source)
    options.setdefault('skip', config.pyvista_plot_skip)

    skip = options['skip'] or (optional and config.pyvista_plot_skip_optional)

    keep_context = 'context' in options
    _ = None if not keep_context else options['context']

    rst_file = document.attributes['source']
    rst_dir = str(Path(rst_file).parent)

    if len(arguments):
        if not config.pyvista_plot_basedir:
            source_file_name = str(Path(setup.app.builder.srcdir) / directives.uri(arguments[0]))
        else:
            source_file_name = str(
                Path(setup.confdir) / config.pyvista_plot_basedir / directives.uri(arguments[0]),
            )

        # If there is content, it will be passed as a caption.
        caption = '\n'.join(content)

        # Enforce unambiguous use of captions.
        if 'caption' in options:
            if caption:  # pragma: no cover
                msg = 'Caption specified in both content and options. Please remove ambiguity.'
                raise ValueError(msg)
            # Use caption option
            caption = options['caption']

        # If the optional function name is provided, use it
        function_name = arguments[1] if len(arguments) == 2 else None

        code = Path(source_file_name).read_text(encoding='utf-8')
        output_base = Path(source_file_name).name
    else:
        source_file_name = rst_file
        code = textwrap.dedent('\n'.join(map(str, content)))

        base = Path(source_file_name).stem
        ext = Path(source_file_name).suffix
        function_name = None
        caption = options.get('caption', '')

        if use_counter:
            counter = document.attributes.get('_plot_counter', 0) + 1
            document.attributes['_plot_counter'] = counter
            output_base = f'{base}-{counter}{ext}'
        else:
            code_hash = hash_plot_code(code, options)
            output_base = f'{base}-{code_hash}{ext}'

    base = Path(output_base).stem
    if Path(output_base).suffix in ('.py', '.rst', '.txt'):  # pragma: no branch
        # Python code is extracted from these inputs
        ext_out = '.py'
        output_base = base
    else:
        ext_out = ''

    # ensure that LaTeX includegraphics doesn't choke in foo.bar.pdf filenames
    output_base = output_base.replace('.', '-')

    # is it in doctest format?
    is_doctest = _contains_doctest(code)
    if 'format' in options:
        is_doctest = options['format'] != 'python'

    # determine output directory name fragment
    source_rel_name = os.path.relpath(source_file_name, setup.confdir)
    source_rel_dir = str(Path(source_rel_name).parent).lstrip(os.path.sep)

    # build_dir: where to place output files (temporarily)
    build_dir = str(Path(setup.app.doctreedir).parent / 'pyvista_plot_directive' / source_rel_dir)
    # get rid of .. in paths, also changes pathsep
    # see note in Python docs for warning about symbolic links on Windows.
    # need to compare source and dest paths at end
    build_dir = os.path.normpath(build_dir)
    Path(build_dir).mkdir(parents=True, exist_ok=True)

    # output_dir: final location in the builder's directory
    dest_dir = str((Path(setup.app.builder.outdir) / source_rel_dir).resolve())
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    # how to link to files from the RST file
    dest_dir_link = Path(os.path.relpath(setup.confdir, rst_dir), source_rel_dir).as_posix()
    try:
        build_dir_link = os.path.relpath(build_dir, rst_dir)
    except ValueError:  # pragma: no cover
        # on Windows, relpath raises ValueError when path and start are on
        # different mounts/drives
        build_dir_link = build_dir
    build_dir_link = Path(build_dir_link).as_posix()

    # make figures
    errors = []
    if skip:
        results = [(code, [])]
    else:
        try:
            results = render_figures(
                code=code,
                code_path=source_file_name,
                output_dir=build_dir,
                output_base=output_base,
                context=keep_context,
                function_name=function_name,
                config=config,
                force_static=force_static,
            )
        except PlotError as err:  # pragma: no cover
            reporter = state.memo.reporter
            sm = reporter.system_message(
                2,
                f'Exception occurred in plotting {output_base}\n from {source_file_name}:\n{err}',
                line=lineno,
            )
            results = [(code, [])]
            errors.append(sm)

    # Properly indent the caption
    caption = (
        '' if skip else '\n' + '\n'.join('   ' + line.strip() for line in caption.split('\n'))
    )

    # generate output restructuredtext
    total_lines = []
    for _, (code_piece, images) in enumerate(results):
        if options['include-source']:
            if is_doctest:
                lines = ['', *code_piece.splitlines()]
            else:
                lines = [
                    '.. code-block:: python',
                    '',
                    *textwrap.indent(code_piece, '    ').splitlines(),
                ]
            source_code = '\n'.join(lines)
        else:
            source_code = ''

        images_input = [] if nofigs else images

        opts = [
            f':{key}: {val}'
            for key, val in options.items()
            if key in ('alt', 'height', 'width', 'scale', 'align')
        ]

        result = jinja2.Template(config.pyvista_plot_template or TEMPLATE).render(
            default_fmt=default_fmt,
            dest_dir=dest_dir_link,
            build_dir=build_dir_link,
            source_link=None,
            multi_image=len(images_input) > 1,
            options=opts,
            images=images_input,
            source_code=source_code,
            html_show_formats=config.pyvista_plot_html_show_formats and len(images_input),
            caption=caption,
        )

        total_lines.extend(result.split('\n'))
        total_lines.extend('\n')

        # If there were errors, return the Node objects to Sphinx now.
        if errors:  # pragma: no cover
            return errors

    if total_lines:
        state_machine.insert_input(total_lines, source=source_file_name)

    # copy script (if necessary)
    Path(build_dir, output_base + ext_out).write_text(
        doctest.script_from_examples(code)
        if source_file_name == rst_file and is_doctest
        else code,
        encoding='utf-8',
    )

    return errors
