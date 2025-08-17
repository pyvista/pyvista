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
        behaviour is controlled by the ``plot_skip`` boolean variable in
        :file:`conf.py`.  Note that, if specified, this option overrides the
        ``plot_skip`` configuration.

    optional : None
        This flag marks the directive for *conditional* execution. Whether the
        directive is executed is controlled by the ``plot_skip_optional``
        boolean variable in :file:`conf.py`.

Additionally, this directive supports all the options of the `image`
directive, except for *target* (since plot will add its own target).  These
include *alt*, *height*, *width*, *scale*, *align*.


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
"""

from __future__ import annotations

import doctest
import hashlib
import os
from os.path import relpath
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

# must enable BUILDING_GALLERY to keep windows active
# enable offscreen to hide figures when generating them.
import pyvista

if TYPE_CHECKING:
    from collections.abc import Callable

    from sphinx.application import Sphinx
    from sphinx.config import Config


pyvista.BUILDING_GALLERY = True
pyvista.OFF_SCREEN = True

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
    """Set up the plot directive."""
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir
    app.add_directive('pyvista-plot', PlotDirective)

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
    return {
        'parallel_read_safe': True,
        'parallel_write_safe': True,
        'version': pyvista.__version__,
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

TEMPLATE = """
{{ source_code }}

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


def _run_code(*, code, code_path, ns=None, function_name=None):  # noqa: ARG001
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
        if pyvista.PLOT_DIRECTIVE_THEME is not None:
            pyvista.set_plot_theme(pyvista.PLOT_DIRECTIVE_THEME)  # pragma: no cover
        exec(code, ns)
    except (Exception, SystemExit) as err:  # pragma: no cover
        raise PlotError(traceback.format_exc()) from err

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
                figures = pyvista.plotting.plotter._ALL_PLOTTERS

                for j, (_, plotter) in enumerate(figures.items()):
                    if plotter._gif_filename is not None:
                        image_file = ImageFile(output_dir, f'{output_base}_{i:02d}_{j:02d}.gif')
                        shutil.move(plotter._gif_filename, image_file.filename)
                    else:
                        image_file = ImageFile(output_dir, f'{output_base}_{i:02d}_{j:02d}.png')
                        try:
                            plotter.screenshot(image_file.filename)
                        except RuntimeError:  # pragma no cover
                            # ignore closed, unrendered plotters
                            continue
                        if force_static or (plotter.last_vtksz is None):
                            images.append(image_file)
                            continue
                        else:
                            image_file = ImageFile(
                                output_dir, f'{output_base}_{i:02d}_{j:02d}.vtksz'
                            )
                            with Path(image_file.filename).open('wb') as f:
                                f.write(plotter.last_vtksz)
                    images.append(image_file)

            pyvista.close_all()  # close and clear all plotters

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
    source_ext = Path(output_base).suffix
    if source_ext in ('.py', '.rst', '.txt'):
        output_base = base
    else:
        source_ext = ''

    # ensure that LaTeX includegraphics doesn't choke in foo.bar.pdf filenames
    output_base = output_base.replace('.', '-')

    # is it in doctest format?
    is_doctest = _contains_doctest(code)
    if 'format' in options:
        is_doctest = options['format'] != 'python'

    # determine output directory name fragment
    source_rel_name = relpath(source_file_name, setup.confdir)
    source_rel_dir = str(Path(source_rel_name).parent).lstrip(os.path.sep)

    # build_dir: where to place output files (temporarily)
    build_dir = str(Path(setup.app.doctreedir).parent / 'plot_directive' / source_rel_dir)
    # get rid of .. in paths, also changes pathsep
    # see note in Python docs for warning about symbolic links on Windows.
    # need to compare source and dest paths at end
    build_dir = os.path.normpath(build_dir)
    Path(build_dir).mkdir(parents=True, exist_ok=True)

    # output_dir: final location in the builder's directory
    dest_dir = str((Path(setup.app.builder.outdir) / source_rel_dir).resolve())
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    # how to link to files from the RST file
    dest_dir_link = os.path.join(  # noqa: PTH118
        relpath(setup.confdir, rst_dir),
        source_rel_dir,
    ).replace(os.path.sep, '/')
    try:
        build_dir_link = relpath(build_dir, rst_dir).replace(os.path.sep, '/')
    except ValueError:  # pragma: no cover
        # on Windows, relpath raises ValueError when path and start are on
        # different mounts/drives
        build_dir_link = build_dir
    _ = dest_dir_link + '/' + output_base + source_ext

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
            errors.append([sm])

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

    if total_lines:
        state_machine.insert_input(total_lines, source=source_file_name)

    # copy image files to builder's output directory, if necessary
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    for _, images in results:
        for image in images:
            destimg = str(Path(dest_dir) / image.basename)
            if image.filename != destimg:
                shutil.copyfile(image.filename, destimg)

    # copy script (if necessary)
    Path(dest_dir, output_base + source_ext).write_text(
        doctest.script_from_examples(code)
        if source_file_name == rst_file and is_doctest
        else code,
        encoding='utf-8',
    )

    return errors
