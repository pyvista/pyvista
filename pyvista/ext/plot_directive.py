"""Plot directive module.

A directive for including a PyVista plot in a Sphinx document

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
        using the `plot_include_source` variable in :file:`conf.py`.

    encoding : str
        If this source file is in a non-UTF8 or non-ASCII encoding, the
        encoding must be specified using the ``:encoding:`` option.  The
        encoding will not be inferred using the ``-*- coding -*-`` metacomment.

    context : None
        If provided, the code will be run in the context of all previous plot
        directives for which the ``:context:`` option was specified.  This only
        applies to inline code plot directives, not those run from files.

    nofigs : bool
        If specified, the code block will be run, but no figures will be
        inserted.  This is usually useful with the ``:context:`` option.

    caption : str
        If specified, the option's argument will be used as a caption for the
        figure. This overwrites the caption given in the content, when the plot
        is generated from a file.

    force_static : bool
        If specified, static images will be used instead of an interactive scene.

Additionally, this directive supports all of the options of the `image`
directive, except for *target* (since plot will add its own target).  These
include *alt*, *height*, *width*, *scale*, *align*.


**Configuration options**
The plot directive has the following configuration options:

    plot_include_source : bool
        Default value for the include-source option. Default is ``True``.

    plot_basedir : str
        Base directory, to which ``plot::`` file names are relative
        to.  If ``None`` or unset, file names are relative to the
        directory where the file containing the directive is.

    plot_html_show_formats : bool
        Whether to show links to the files in HTML. Default ``True``.

    plot_template : str
        Provide a customized Jinja2 template for preparing restructured text.

    plot_setup : str
        Python code to be run before every plot directive block.

    plot_cleanup : str
        Python code to be run after every plot directive block.

These options can be set by defining global variables of the same name in
:file:`conf.py`.

"""

from __future__ import annotations

import doctest
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

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

pyvista.BUILDING_GALLERY = True
pyvista.OFF_SCREEN = True

# -----------------------------------------------------------------------------
# Registration hook
# -----------------------------------------------------------------------------


def _option_boolean(arg):
    if not arg or not arg.strip():
        # no argument given, assume used as a flag
        return True
    elif arg.strip().lower() in ('no', '0', 'false'):
        return False
    elif arg.strip().lower() in ('yes', '1', 'true'):
        return True
    else:  # pragma: no cover
        raise ValueError(f'"{arg}" unknown boolean')


def _option_context(arg):
    if arg is not None:  # pragma: no cover
        raise ValueError("No arguments allowed for ``:context:``")


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
        except Exception as e:  # pragma: no cover
            raise self.error(str(e))


def setup(app):
    """Set up the plot directive."""
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir
    app.add_directive('pyvista-plot', PlotDirective)
    app.add_config_value('plot_include_source', True, False)
    app.add_config_value('plot_basedir', None, True)
    app.add_config_value('plot_html_show_formats', True, True)
    app.add_config_value('plot_template', None, True)
    app.add_config_value('plot_setup', None, True)
    app.add_config_value('plot_cleanup', None, True)
    return {'parallel_read_safe': True, 'parallel_write_safe': True, 'version': pyvista.__version__}


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
    r = re.compile(r'^\s*>>>', re.M)
    m = r.search(text)
    return bool(m)


def _contains_pyvista_plot(text):
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
    for line in text.split("\n"):
        part.append(line)

        # check if show(...) or plot(...) is within the line
        line = _strip_comments(line)
        if within_plot:  # allow for multi-line plot(...
            if _strip_comments(line).endswith(')'):
                parts.append("\n".join(part))
                part = []
                within_plot = False

        elif '.show(' in line or '.plot(' in line:
            if _strip_comments(line).endswith(')'):
                parts.append("\n".join(part))
                part = []
            else:  # allow for multi-line plot(...
                within_plot = True

    if "\n".join(part).strip():
        parts.append("\n".join(part))
    return is_doctest, parts


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


def _run_code(code, code_path, ns=None, function_name=None):
    """Run a docstring example if it does not contain ``'doctest:+SKIP'``, or a
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
    # The doctest parser will present the code-block once again with the ```pyvista-plot::`` directive
    # and its options properly parsed.
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
    code_setup = config.plot_setup
    code_cleanup = config.plot_cleanup

    if code_setup:
        _run_code(code_setup, code_path, ns, function_name)

    try:
        for i, code_piece in enumerate(code_pieces):
            # generate the plot
            _run_code(
                doctest.script_from_examples(code_piece) if is_doctest else code_piece,
                code_path,
                ns,
                function_name,
            )

            images = []
            figures = pyvista.plotting.plotter._ALL_PLOTTERS

            for j, (_, plotter) in enumerate(figures.items()):
                if hasattr(plotter, '_gif_filename'):
                    image_file = ImageFile(output_dir, f"{output_base}_{i:02d}_{j:02d}.gif")
                    shutil.move(plotter._gif_filename, image_file.filename)
                else:
                    image_file = ImageFile(output_dir, f"{output_base}_{i:02d}_{j:02d}.png")
                    try:
                        plotter.screenshot(image_file.filename)
                    except RuntimeError:  # pragma no cover
                        # ignore closed, unrendered plotters
                        continue
                    if force_static or (plotter.last_vtksz is None):
                        images.append(image_file)
                        continue
                    else:
                        image_file = ImageFile(output_dir, f"{output_base}_{i:02d}_{j:02d}.vtksz")
                        with Path(image_file.filename).open("wb") as f:
                            f.write(plotter.last_vtksz)
                images.append(image_file)

            pyvista.close_all()  # close and clear all plotters

            results.append((code_piece, images))
    finally:
        if code_cleanup:
            _run_code(code_cleanup, code_path, ns, function_name)

    return results


def run(arguments, content, options, state_machine, state, lineno):
    """Run the plot directive."""
    document = state_machine.document
    config = document.settings.env.config
    nofigs = 'nofigs' in options
    force_static = 'force_static' in options

    default_fmt = 'png'

    options.setdefault('include-source', config.plot_include_source)
    keep_context = 'context' in options
    _ = None if not keep_context else options['context']

    rst_file = document.attributes['source']
    rst_dir = str(Path(rst_file).parent)

    if len(arguments):
        if not config.plot_basedir:
            source_file_name = str(Path(setup.app.builder.srcdir) / directives.uri(arguments[0]))
        else:
            source_file_name = str(
                Path(setup.confdir) / config.plot_basedir / directives.uri(arguments[0]),
            )

        # If there is content, it will be passed as a caption.
        caption = '\n'.join(content)

        # Enforce unambiguous use of captions.
        if "caption" in options:
            if caption:  # pragma: no cover
                raise ValueError(
                    'Caption specified in both content and options. Please remove ambiguity.',
                )
            # Use caption option
            caption = options["caption"]

        # If the optional function name is provided, use it
        function_name = arguments[1] if len(arguments) == 2 else None

        code = Path(source_file_name).read_text(encoding='utf-8')
        output_base = Path(source_file_name).name
    else:
        source_file_name = rst_file
        code = textwrap.dedent("\n".join(map(str, content)))
        counter = document.attributes.get('_plot_counter', 0) + 1
        document.attributes['_plot_counter'] = counter
        base, ext = os.path.splitext(os.path.basename(source_file_name))  # noqa: PTH119, PTH122
        output_base = '%s-%d.py' % (base, counter)
        function_name = None
        caption = options.get('caption', '')

    base, source_ext = os.path.splitext(output_base)  # noqa: PTH122
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
    try:
        results = render_figures(
            code,
            source_file_name,
            build_dir,
            output_base,
            keep_context,
            function_name,
            config,
            force_static,
        )
        errors = []
    except PlotError as err:  # pragma: no cover
        reporter = state.memo.reporter
        sm = reporter.system_message(
            2,
            f"Exception occurred in plotting {output_base}\n from {source_file_name}:\n{err}",
            line=lineno,
        )
        results = [(code, [])]
        errors = [sm]

    # Properly indent the caption
    caption = '\n' + '\n'.join('   ' + line.strip() for line in caption.split('\n'))

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
            source_code = "\n".join(lines)
        else:
            source_code = ''

        if nofigs:
            images = []

        opts = [
            f':{key}: {val}'
            for key, val in options.items()
            if key in ('alt', 'height', 'width', 'scale', 'align')
        ]

        result = jinja2.Template(config.plot_template or TEMPLATE).render(
            default_fmt=default_fmt,
            dest_dir=dest_dir_link,
            build_dir=build_dir_link,
            source_link=None,
            multi_image=len(images) > 1,
            options=opts,
            images=images,
            source_code=source_code,
            html_show_formats=config.plot_html_show_formats and len(images),
            caption=caption,
        )

        total_lines.extend(result.split("\n"))
        total_lines.extend("\n")

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
        doctest.script_from_examples(code) if source_file_name == rst_file and is_doctest else code,
        encoding='utf-8',
    )

    return errors
