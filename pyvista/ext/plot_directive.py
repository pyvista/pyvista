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

    include-source : ``bool``
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

    ``nofigs`` : None
        When setting this flag, the code block will be run but no figures will be
        inserted.  This is usually useful with the ``:context:`` option.

    caption : str
        If specified, the option's argument will be used as a caption for the
        figure. This overwrites the caption given in the content, when the plot
        is generated from a file.

    ``force_static`` : None
        When setting this flag, static images will be used instead of an
        interactive scene.

    skip : ``bool``, default: True
        Whether to skip execution of this directive. If no argument is provided
        that is, ``:skip:``, then it defaults to ``:skip: true``.  Default
        behaviour is controlled by the ``pyvista_plot_skip`` boolean variable in
        :file:`conf.py`.  Note that, if specified, this option overrides the
        ``pyvista_plot_skip`` configuration.

    optional : None
        This flag marks the directive for *conditional* execution. Whether the
        directive is executed is controlled by the ``pyvista_plot_skip_optional``
        boolean variable in :file:`conf.py`.

Additionally, this directive supports all the options of the ``image``
directive, except for ``target`` (since plot will add its own target).  These
include ``alt``, ``height``, ``width``, ``scale``, ``align``.


**Open Graph previews**

Add ``sphinx_autoopengraph`` to ``extensions`` for a sensible Open Graph link
preview on every page. It is a separate, independent extension -- nothing about
it is specific to plotting, and it is not enabled by enabling this directive.
See :ref:`opengraph_docs`.


**Configuration options**

.. versionchanged:: 0.45
   Prior to v0.45, these directives conflicted with ``matplotlib``. All
   directives have been prepended with ``pyvista_``.

The plot directive has the following configuration options:

    ``pyvista_plot_include_source`` : bool, default: True
        Default value for the ``include-source`` directive option.
        Default is ``True``.

    ``pyvista_plot_basedir`` : str
        Base directory, to which ``plot::`` file names are relative
        to.  If ``None`` or unset, file names are relative to the
        directory where the file containing the directive is.

    ``pyvista_plot_html_show_formats`` : bool, default: True
        Whether to show links to the files in HTML. Default ``True``.

    ``pyvista_plot_template`` : str
        Provide a customized Jinja2 template for preparing restructured text.

    ``pyvista_plot_setup`` : str
        Python code to be run before every plot directive block.

    ``pyvista_plot_cleanup`` : str
        Python code to be run after every plot directive block.

    ``pyvista_plot_skip`` : bool, default: False
        Default value for the ``skip`` directive option.

    ``pyvista_plot_skip_optional`` : bool, default: False
        Whether to skip execution of ``optional`` directives.

    ``pyvista_plot_autocodelink`` : bool, default: False
        Hyperlink identifiers in the rendered output to their documented
        targets. Requires the `sphinx-autocodelink
        <https://github.com/user27182/sphinx-autocodelink>`_ package to be
        installed (``pip install sphinx-autocodelink``); raises at build time
        if enabled without it. Only applies to directives with
        ``include-source`` on -- otherwise the code being linked from isn't
        actually shown to the reader. Recorded under sphinx-autocodelink's
        own default (uncategorized) bucket; rename its displayed label with
        sphinx-autocodelink's own ``autocodelink_category_labels``.

        .. versionadded:: 0.49

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

Within one build, directives with the same source hash share their rendered
output: the first one executes and every later one copies its figures instead
of executing again. ``:context:`` directives always execute.

.. versionadded:: 0.49

"""

from __future__ import annotations

import doctest
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import textwrap
import traceback
from typing import TYPE_CHECKING
from typing import ClassVar

from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
from docutils.parsers.rst.directives.images import Image
import jinja2  # Sphinx dependency.

import pyvista as pv

try:
    # Optional: only required when `pyvista_plot_autocodelink` is enabled.
    from sphinx_autocodelink import record_namespace
except ImportError:
    record_namespace = None

try:
    # Record capture/replay for the figure cache (see ``run`` when missing).
    from sphinx_autocodelink import DEFAULT_DOCSTRING_EXAMPLE_CATEGORY
    from sphinx_autocodelink import is_inside_autodoc_desc
    from sphinx_autocodelink import record_namespace_to_disk

    try:
        from sphinx_autocodelink import capture_records

        _records_from_jsonable = list
    except ImportError:  # older sphinx-autocodelink keeps the round trip private
        from sphinx_autocodelink import _from_jsonable
        from sphinx_autocodelink import _records_for
        from sphinx_autocodelink import _to_jsonable

        def capture_records(*, source, namespace):
            """Resolve ``source``'s records as JSON-able dicts."""
            return [_to_jsonable(record) for record in _records_for(source, namespace)]

        def _records_from_jsonable(entries):
            """Rebuild record objects for ``record_namespace_to_disk``'s ``extra``."""
            return [_from_jsonable(entry) for entry in entries]
except ImportError:
    _RECORDS_CACHE_OK = False
else:
    _RECORDS_CACHE_OK = True

if TYPE_CHECKING:
    from collections.abc import Callable

    from docutils.parsers.rst.states import RSTState
    from sphinx.application import Sphinx
    from sphinx.config import Config
    from sphinx.environment import BuildEnvironment


# CSS class marking the ``.. container::`` node that wraps this directive's generated source code
_PLOT_SOURCE_CLASS = 'pyvista-plot-source'

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


def setup(app: Sphinx):
    """Set up the plot directive.

    Sphinx calls this when it loads the extension, which is where the two globals
    below belong: setting them at import time made merely importing this module
    change how plotters behave for the rest of the process. ``BUILDING_GALLERY``
    keeps render windows active (and holds each plotter strongly rather than
    through a weak proxy), and ``OFF_SCREEN`` hides the figures while they are
    generated -- both wanted for a documentation build, both surprising anywhere
    else.
    """
    pv.BUILDING_GALLERY = True
    pv.OFF_SCREEN = True

    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir
    app.add_directive('pyvista-plot', PlotDirective)
    app.connect('builder-inited', _clear_figure_cache)
    if record_namespace is not None:
        app.setup_extension('sphinx_autocodelink')

    def check_autocodelink_available(_app: Sphinx, config: Config) -> None:
        if config.pyvista_plot_autocodelink and record_namespace is None:
            msg = (
                "'pyvista_plot_autocodelink' is enabled, but the 'sphinx-autocodelink' "
                'package is not installed. Install it with `pip install sphinx-autocodelink`.'
            )
            raise RuntimeError(msg)

    app.connect('config-inited', check_autocodelink_available)

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

    app.add_config_value('pyvista_plot_use_counter', False, 'env')
    app.add_config_value('pyvista_plot_include_source', True, False)
    app.add_config_value('pyvista_plot_basedir', None, True)
    app.add_config_value('pyvista_plot_html_show_formats', True, True)
    app.add_config_value('pyvista_plot_template', None, True)
    app.add_config_value('pyvista_plot_setup', None, True)
    app.add_config_value('pyvista_plot_cleanup', None, True)
    app.add_config_value(name='pyvista_plot_skip', default=False, rebuild='html')
    app.add_config_value(name='pyvista_plot_skip_optional', default=False, rebuild='html')
    app.add_config_value(name='pyvista_plot_autocodelink', default=False, rebuild='html')
    return {
        'parallel_read_safe': True,
        'parallel_write_safe': True,
        'version': pv.__version__,
    }


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
        """Return the ``basename`` without the suffix."""
        return Path(self.basename).stem

    def __repr__(self) -> str:  # pragma no cover
        return self.filename


class PlotError(RuntimeError):
    """More descriptive plot error."""


def _run_code(*, code, code_path, ns=None, function_name=None):
    """Run a docstring example.

    Run the example if it does not contain ``'doctest:+SKIP'``, or a
    ``pyvista-plot::`` directive.  In the later case, the doctest parser will
    present the code-block again with the ``pyvista-plot::`` directive
    and its options removed.

    Import a Python module from a path, and run the function given by
    name, if ``function_name`` is not None.
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


def _figure_cache_dir(app: Sphinx) -> Path:
    """Return the build-wide directory of results shared between identical directives."""
    return Path(app.doctreedir).parent / 'pyvista_plot_directive' / '_cache'


def _clear_figure_cache(app: Sphinx) -> None:
    """Drop the figure cache so no entry outlives the build that rendered it."""
    shutil.rmtree(_figure_cache_dir(app), ignore_errors=True)


def _load_cached_figures(cache_entry: Path, *, code, output_dir, output_base):
    """Copy a cached directive's files in as this directive's output.

    Returns ``(results, records)`` for ``render_figures``, or ``None`` when
    there is no usable entry and the code has to be executed.
    """
    try:
        manifest = json.loads((cache_entry / 'manifest.json').read_text(encoding='utf-8'))
        image_names = manifest['images']
    except (OSError, ValueError, KeyError, TypeError):
        return None
    _, code_pieces = _split_code_at_show(code)
    if len(image_names) != len(code_pieces):
        return None
    try:
        results = []
        for code_piece, names in zip(code_pieces, image_names, strict=True):
            images = []
            for name in names:
                image_file = ImageFile(output_dir, f'{output_base}_{name}')
                if image_file.extension == 'vtksz':
                    # the template pairs every interactive scene with a static .png
                    png = str(Path(name).with_suffix('.png'))
                    shutil.copyfile(cache_entry / png, Path(output_dir) / f'{output_base}_{png}')
                shutil.copyfile(cache_entry / name, image_file.filename)
                images.append(image_file)
            results.append((code_piece, images))
    except OSError:
        return None
    return results, manifest.get('records')


def _store_cached_figures(cache_entry: Path, *, results, output_base, records) -> None:
    """Copy this directive's rendered files into the build-wide figure cache.

    Staged in a temporary sibling and renamed into place, so a parallel reader
    never sees a partial entry and a losing concurrent writer is discarded.
    """
    prefix = f'{output_base}_'
    image_names = []
    files = []
    for _, images in results:
        names = []
        for img in images:
            names.append(img.basename.removeprefix(prefix))
            files.append((img.filename, names[-1]))
            if img.extension == 'vtksz':
                png = f'{img.stem}.png'
                files.append((str(Path(img.dirname) / png), png.removeprefix(prefix)))
        image_names.append(names)

    staging = cache_entry.parent / f'{cache_entry.name}.{os.getpid()}.tmp'
    try:
        staging.mkdir(parents=True, exist_ok=True)
        for source, name in files:
            shutil.copyfile(source, staging / name)
        manifest = {'images': image_names, 'records': records}
        (staging / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        staging.rename(cache_entry)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)


def _replay_cached_records(records, *, docname, state) -> None:
    """Write cache-captured autocodelink records to the package's on-disk records."""
    category = (
        DEFAULT_DOCSTRING_EXAMPLE_CATEGORY
        if state is not None and is_inside_autodoc_desc(state)
        else ''
    )
    record_namespace_to_disk(
        directory=Path(setup.app.srcdir) / setup.app.config.autocodelink_records_dir,
        docname=docname,
        source='',
        namespace={},
        extra=_records_from_jsonable(records),
        category=category,
    )


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
    env: BuildEnvironment | None = None,
    include_source: bool = True,
    state: RSTState | None = None,
    cache_entry: Path | None = None,
):
    """Run a pyplot script and save the images in *``output_dir``*.

    Save the images under *``output_dir``* with file names derived from
    *``output_base``*. Closed plotters are ignored if they were never
    rendered.

    If ``env`` is given and ``include_source`` is true, also records the code's identifiers
    to hyperlink -- skipped when the source isn't shown, since there would be nothing on the
    page for a reader to click through to. ``state`` is the calling directive's own
    ``self.state``, passed through to sphinx-autocodelink for its own categorization.

    ``cache_entry`` names this code's slot in the build-wide figure cache: a usable
    entry is copied instead of executing the code, and a rendered result is stored.
    """
    if cache_entry is not None:
        cached = _load_cached_figures(
            cache_entry, code=code, output_dir=output_dir, output_base=output_base
        )
        if cached is not None:
            results, cached_records = cached
            if (
                cached_records
                and env is not None
                and config.pyvista_plot_autocodelink
                and include_source
            ):
                _replay_cached_records(cached_records, docname=env.docname, state=state)
            return results

    # We skip snippets that contain the ``pyvista-plot::`` directive as part of their code.
    # The doctest parser will present the code-block once again with the ``pyvista-plot::``
    # directive and its options properly parsed.
    if _contains_pyvista_plot(code):
        is_doctest = True
        code_pieces = [code]
    else:
        # Try to determine if all images already exist
        is_doctest, code_pieces = _split_code_at_show(code)

    # Otherwise, we didn't find the files, so build them
    results = []
    records = None
    ns = plot_context if context else {}
    clean_pieces = []

    # Check for setup and teardown code for plots
    code_setup = config.pyvista_plot_setup
    code_cleanup = config.pyvista_plot_cleanup

    if code_setup:
        _run_code(code=code_setup, code_path=code_path, ns=ns, function_name=function_name)

    try:
        for i, code_piece in enumerate(code_pieces):
            # generate the plot
            clean_piece = doctest.script_from_examples(code_piece) if is_doctest else code_piece
            clean_pieces.append(clean_piece)
            _run_code(
                code=clean_piece,
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

        if env is not None and clean_pieces and config.pyvista_plot_autocodelink:
            source = '\n'.join(clean_pieces)
            if include_source:
                record_namespace(
                    env=env,
                    docname=env.docname,
                    source=source,
                    namespace=ns,
                    state=state,
                )
            if cache_entry is not None:
                records = capture_records(source=source, namespace=ns)
    finally:
        if code_cleanup:
            _run_code(code=code_cleanup, code_path=code_path, ns=ns, function_name=function_name)

    if cache_entry is not None:
        _store_cached_figures(
            cache_entry, results=results, output_base=output_base, records=records
        )

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

    # directives whose results are not a pure function of their source always execute
    cache_entry = None
    if (
        not keep_context
        and function_name is None
        and not _contains_pyvista_plot(code)
        and (not config.pyvista_plot_autocodelink or _RECORDS_CACHE_OK)
    ):
        static_tag = '-static' if force_static else ''
        cache_entry = _figure_cache_dir(setup.app) / f'{hash_plot_code(code, options)}{static_tag}'

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
                env=document.settings.env,
                include_source=options['include-source'],
                state=state,
                cache_entry=cache_entry,
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
