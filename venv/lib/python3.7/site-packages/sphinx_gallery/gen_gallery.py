# -*- coding: utf-8 -*-
# Author: Óscar Nájera
# License: 3-clause BSD
"""
Sphinx-Gallery Generator
========================

Attaches Sphinx-Gallery to Sphinx in order to generate the galleries
when building the documentation.
"""


from __future__ import division, print_function, absolute_import
import codecs
import copy
from datetime import timedelta, datetime
from distutils.version import LooseVersion
from importlib import import_module
import re
import os
from xml.sax.saxutils import quoteattr, escape

import sphinx
from sphinx.util.console import red
from . import sphinx_compatibility, glr_path_static, __version__ as _sg_version
from .utils import _replace_md5, Bunch
from .backreferences import finalize_backreferences
from .gen_rst import (generate_dir_rst, SPHX_GLR_SIG, _get_memory_base,
                      extract_intro_and_title, get_docstring_and_rest,
                      _get_readme)
from .scrapers import _scraper_dict, _reset_dict
from .docs_resolv import embed_code_links
from .downloads import generate_zipfiles
from .sorting import NumberOfCodeLinesSortKey
from .binder import copy_binder_files

try:
    basestring
except NameError:
    basestring = str
    unicode = str

DEFAULT_GALLERY_CONF = {
    'filename_pattern': re.escape(os.sep) + 'plot',
    'ignore_pattern': r'__init__\.py',
    'examples_dirs': os.path.join('..', 'examples'),
    'subsection_order': None,
    'within_subsection_order': NumberOfCodeLinesSortKey,
    'gallery_dirs': 'auto_examples',
    'backreferences_dir': None,
    'doc_module': (),
    'reference_url': {},
    # Build options
    # -------------
    # We use a string for 'plot_gallery' rather than simply the Python boolean
    # `True` as it avoids a warning about unicode when controlling this value
    # via the command line switches of sphinx-build
    'plot_gallery': 'True',
    'download_all_examples': True,
    'abort_on_example_error': False,
    'failing_examples': {},
    'passing_examples': [],
    'stale_examples': [],  # ones that did not need to be run due to md5sum
    'expected_failing_examples': set(),
    'thumbnail_size': (400, 280),  # Default CSS does 0.4 scaling (160, 112)
    'min_reported_time': 0,
    'binder': {},
    'image_scrapers': ('matplotlib',),
    'reset_modules': ('matplotlib', 'seaborn'),
    'first_notebook_cell': '%matplotlib inline',
    'remove_config_comments': False,
    'show_memory': False,
    'junit': '',
    'log_level': {'backreference_missing': 'warning'},
}

logger = sphinx_compatibility.getLogger('sphinx-gallery')


def parse_config(app):
    """Process the Sphinx Gallery configuration"""
    try:
        plot_gallery = eval(app.builder.config.plot_gallery)
    except TypeError:
        plot_gallery = bool(app.builder.config.plot_gallery)
    src_dir = app.builder.srcdir
    abort_on_example_error = app.builder.config.abort_on_example_error
    lang = app.builder.config.highlight_language
    gallery_conf = _complete_gallery_conf(
        app.config.sphinx_gallery_conf, src_dir, plot_gallery,
        abort_on_example_error, lang, app.builder.name, app)

    # this assures I can call the config in other places
    app.config.sphinx_gallery_conf = gallery_conf
    app.config.html_static_path.append(glr_path_static())
    return gallery_conf


def _complete_gallery_conf(sphinx_gallery_conf, src_dir, plot_gallery,
                           abort_on_example_error, lang='python',
                           builder_name='html', app=None):
    gallery_conf = copy.deepcopy(DEFAULT_GALLERY_CONF)
    gallery_conf.update(sphinx_gallery_conf)
    if sphinx_gallery_conf.get('find_mayavi_figures', False):
        logger.warning(
            "Deprecated image scraping variable `find_mayavi_figures`\n"
            "detected, use `image_scrapers` instead as:\n\n"
            "   image_scrapers=('matplotlib', 'mayavi')",
            type=DeprecationWarning)
        gallery_conf['image_scrapers'] += ('mayavi',)
    gallery_conf.update(plot_gallery=plot_gallery)
    gallery_conf.update(abort_on_example_error=abort_on_example_error)
    gallery_conf['src_dir'] = src_dir
    # Old Sphinx can't handle pickling app, so let's just expose the one
    # thing we need internally
    if LooseVersion(sphinx.__version__) < LooseVersion('1.8'):
        app = Bunch(config=app.config) if app is not None else app
    gallery_conf['app'] = app

    if gallery_conf.get("mod_example_dir", False):
        backreferences_warning = """\n========
        Sphinx-Gallery found the configuration key 'mod_example_dir'. This
        is deprecated, and you should now use the key 'backreferences_dir'
        instead. Support for 'mod_example_dir' will be removed in a subsequent
        version of Sphinx-Gallery. For more details, see the backreferences
        documentation:

        https://sphinx-gallery.github.io/configuration.html#references-to-examples"""  # noqa: E501
        gallery_conf['backreferences_dir'] = gallery_conf['mod_example_dir']
        logger.warning(
            backreferences_warning,
            type=DeprecationWarning)

    # deal with show_memory
    if gallery_conf['show_memory']:
        try:
            from memory_profiler import memory_usage  # noqa, analysis:ignore
        except ImportError:
            logger.warning("Please install 'memory_profiler' to enable peak "
                           "memory measurements.")
            gallery_conf['show_memory'] = False
    gallery_conf['memory_base'] = _get_memory_base(gallery_conf)

    # deal with scrapers
    scrapers = gallery_conf['image_scrapers']
    if not isinstance(scrapers, (tuple, list)):
        scrapers = [scrapers]
    scrapers = list(scrapers)
    for si, scraper in enumerate(scrapers):
        if isinstance(scraper, basestring):
            if scraper in _scraper_dict:
                scraper = _scraper_dict[scraper]
            else:
                orig_scraper = scraper
                try:
                    scraper = import_module(scraper)
                    scraper = getattr(scraper, '_get_sg_image_scraper')
                    scraper = scraper()
                except Exception as exp:
                    raise ValueError('Unknown image scraper %r, got:\n%s'
                                     % (orig_scraper, exp))
            scrapers[si] = scraper
        if not callable(scraper):
            raise ValueError('Scraper %r was not callable' % (scraper,))
    gallery_conf['image_scrapers'] = tuple(scrapers)
    del scrapers

    # deal with resetters
    resetters = gallery_conf['reset_modules']
    if not isinstance(resetters, (tuple, list)):
        resetters = [resetters]
    resetters = list(resetters)
    for ri, resetter in enumerate(resetters):
        if isinstance(resetter, basestring):
            if resetter not in _reset_dict:
                raise ValueError('Unknown module resetter named %r'
                                 % (resetter,))
            resetters[ri] = _reset_dict[resetter]
        elif not callable(resetter):
            raise ValueError('Module resetter %r was not callable'
                             % (resetter,))
    gallery_conf['reset_modules'] = tuple(resetters)

    lang = lang if lang in ('python', 'python3', 'default') else 'python'
    gallery_conf['lang'] = lang
    del resetters

    # Ensure the first cell text is a string if we have it
    first_cell = gallery_conf.get("first_notebook_cell")
    if (not isinstance(first_cell, basestring)) and (first_cell is not None):
        raise ValueError("The 'first_notebook_cell' parameter must be type str"
                         "or None, found type %s" % type(first_cell))
    gallery_conf['first_notebook_cell'] = first_cell
    # Make it easy to know which builder we're in
    gallery_conf['builder_name'] = builder_name
    return gallery_conf


def get_subsections(srcdir, examples_dir, gallery_conf):
    """Return the list of subsections of a gallery.

    Parameters
    ----------
    srcdir : str
        absolute path to directory containing conf.py
    examples_dir : str
        path to the examples directory relative to conf.py
    gallery_conf : dict
        The gallery configuration.

    Returns
    -------
    out : list
        sorted list of gallery subsection folder names
    """
    sortkey = gallery_conf['subsection_order']
    subfolders = [subfolder for subfolder in os.listdir(examples_dir)
                  if _get_readme(os.path.join(examples_dir, subfolder),
                                 gallery_conf, raise_error=False) is not None]
    base_examples_dir_path = os.path.relpath(examples_dir, srcdir)
    subfolders_with_path = [os.path.join(base_examples_dir_path, item)
                            for item in subfolders]
    sorted_subfolders = sorted(subfolders_with_path, key=sortkey)

    return [subfolders[i] for i in [subfolders_with_path.index(item)
                                    for item in sorted_subfolders]]


def _prepare_sphx_glr_dirs(gallery_conf, srcdir):
    """Creates necessary folders for sphinx_gallery files """
    examples_dirs = gallery_conf['examples_dirs']
    gallery_dirs = gallery_conf['gallery_dirs']

    if not isinstance(examples_dirs, list):
        examples_dirs = [examples_dirs]

    if not isinstance(gallery_dirs, list):
        gallery_dirs = [gallery_dirs]

    if bool(gallery_conf['backreferences_dir']):
        backreferences_dir = os.path.join(
            srcdir, gallery_conf['backreferences_dir'])
        if not os.path.exists(backreferences_dir):
            os.makedirs(backreferences_dir)

    return list(zip(examples_dirs, gallery_dirs))


def generate_gallery_rst(app):
    """Generate the Main examples gallery reStructuredText

    Start the sphinx-gallery configuration and recursively scan the examples
    directories in order to populate the examples gallery
    """
    logger.info('generating gallery...', color='white')
    gallery_conf = parse_config(app)

    seen_backrefs = set()

    computation_times = []
    workdirs = _prepare_sphx_glr_dirs(gallery_conf,
                                      app.builder.srcdir)

    # Check for duplicate filenames to make sure linking works as expected
    examples_dirs = [ex_dir for ex_dir, _ in workdirs]
    files = collect_gallery_files(examples_dirs)
    check_duplicate_filenames(files)

    for examples_dir, gallery_dir in workdirs:

        examples_dir = os.path.join(app.builder.srcdir, examples_dir)
        gallery_dir = os.path.join(app.builder.srcdir, gallery_dir)

        # Here we don't use an os.walk, but we recurse only twice: flat is
        # better than nested.
        this_fhindex, this_computation_times = generate_dir_rst(
            examples_dir, gallery_dir, gallery_conf, seen_backrefs)

        computation_times += this_computation_times
        write_computation_times(gallery_conf, gallery_dir,
                                this_computation_times)

        # we create an index.rst with all examples
        index_rst_new = os.path.join(gallery_dir, 'index.rst.new')
        with codecs.open(index_rst_new, 'w', encoding='utf-8') as fhindex:
            # :orphan: to suppress "not included in TOCTREE" sphinx warnings
            fhindex.write(":orphan:\n\n" + this_fhindex)

            for subsection in get_subsections(
                    app.builder.srcdir, examples_dir, gallery_conf):
                src_dir = os.path.join(examples_dir, subsection)
                target_dir = os.path.join(gallery_dir, subsection)
                this_fhindex, this_computation_times = \
                    generate_dir_rst(src_dir, target_dir, gallery_conf,
                                     seen_backrefs)
                fhindex.write(this_fhindex)
                computation_times += this_computation_times
                write_computation_times(gallery_conf, target_dir,
                                        this_computation_times)

            if gallery_conf['download_all_examples']:
                download_fhindex = generate_zipfiles(gallery_dir)
                fhindex.write(download_fhindex)

            fhindex.write(SPHX_GLR_SIG)
        _replace_md5(index_rst_new)
    finalize_backreferences(seen_backrefs, gallery_conf)

    if gallery_conf['plot_gallery']:
        logger.info("computation time summary:", color='white')
        for time_elapsed, fname in sorted(computation_times, reverse=True):
            fname = os.path.relpath(fname,
                                    os.path.normpath(gallery_conf['src_dir']))
            if time_elapsed is not None:
                if time_elapsed >= gallery_conf['min_reported_time']:
                    logger.info("    - %s: %.2g sec", fname, time_elapsed)
            else:
                logger.info("    - %s: not run", fname)
        # Also create a junit.xml file, useful e.g. on CircleCI
        write_junit_xml(gallery_conf, app.builder.outdir, computation_times)


SPHX_GLR_COMP_TIMES = """
:orphan:

.. _{0}:

Computation times
=================
"""


def _sec_to_readable(t):
    """Convert a number of seconds to a more readable representation."""
    # This will only work for < 1 day execution time
    # And we reserve 2 digits for minutes because presumably
    # there aren't many > 99 minute scripts, but occasionally some
    # > 9 minute ones
    t = datetime(1, 1, 1) + timedelta(seconds=t)
    t = '{0:02d}:{1:02d}.{2:03d}'.format(
        t.hour * 60 + t.minute, t.second,
        int(round(t.microsecond / 1000.)))
    return t


def write_computation_times(gallery_conf, target_dir, computation_times):
    if all(time[0] == 0 for time in computation_times):
        return
    target_dir_clean = os.path.relpath(
        target_dir, gallery_conf['src_dir']).replace(os.path.sep, '_')
    new_ref = 'sphx_glr_%s_sg_execution_times' % target_dir_clean
    with codecs.open(os.path.join(target_dir, 'sg_execution_times.rst'), 'w',
                     encoding='utf-8') as fid:
        fid.write(SPHX_GLR_COMP_TIMES.format(new_ref))
        total_time = sum(ct[0] for ct in computation_times)
        fid.write('**{0}** total execution time for **{1}** files:\n\n'
                  .format(_sec_to_readable(total_time), target_dir_clean))
        # sort by time (descending) then alphabetical
        for ct in sorted(computation_times, key=lambda x: (-x[0], x[1])):
            name = os.path.basename(ct[1])
            example_link = 'sphx_glr_%s_%s' % (target_dir_clean, name)
            fid.write(u'- **{0}**: :ref:`{2}` (``{1}``)\n'.format(
                _sec_to_readable(ct[0]), name, example_link))


def write_junit_xml(gallery_conf, target_dir, computation_times):
    if not gallery_conf['junit'] or not gallery_conf['plot_gallery']:
        return
    failing_as_expected, failing_unexpectedly, passing_unexpectedly = \
        _parse_failures(gallery_conf)
    n_tests = 0
    n_failures = 0
    n_skips = 0
    elapsed = 0.
    src_dir = gallery_conf['src_dir']
    output = ''
    for ct in computation_times:
        t, fname = ct
        if not any(fname in x for x in (gallery_conf['passing_examples'],
                                        failing_unexpectedly,
                                        failing_as_expected,
                                        passing_unexpectedly)):
            continue  # not subselected by our regex
        _, title = extract_intro_and_title(
            fname, get_docstring_and_rest(fname)[0])
        output += (
            u'<testcase classname={0!s} file={1!s} line="1" '
            u'name={2!s} time="{3!r}">'
            .format(quoteattr(os.path.splitext(os.path.basename(fname))[0]),
                    quoteattr(os.path.relpath(fname, src_dir)),
                    quoteattr(title), t))
        if fname in failing_as_expected:
            output += u'<skipped message="expected example failure"></skipped>'
            n_skips += 1
        elif fname in failing_unexpectedly or fname in passing_unexpectedly:
            if fname in failing_unexpectedly:
                traceback = gallery_conf['failing_examples'][fname]
            else:  # fname in passing_unexpectedly
                traceback = 'Passed even though it was marked to fail'
            n_failures += 1
            output += (u'<failure message={0!s}>{1!s}</failure>'
                       .format(quoteattr(traceback.splitlines()[-1].strip()),
                               escape(traceback)))
        output += u'</testcase>'
        n_tests += 1
        elapsed += ct[0]
    output += u'</testsuite>'
    output = (u'<?xml version="1.0" encoding="utf-8"?>'
              u'<testsuite errors="0" failures="{0}" name="sphinx-gallery" '
              u'skipped="{1}" tests="{2}" time="{3}">'
              .format(n_failures, n_skips, n_tests, elapsed)) + output
    # Actually write it
    fname = os.path.normpath(os.path.join(target_dir, gallery_conf['junit']))
    junit_dir = os.path.dirname(fname)
    if not os.path.isdir(junit_dir):
        os.makedirs(junit_dir)
    with codecs.open(fname, 'w', encoding='utf-8') as fid:
        fid.write(output)


def touch_empty_backreferences(app, what, name, obj, options, lines):
    """Generate empty back-reference example files.

    This avoids inclusion errors/warnings if there are no gallery
    examples for a class / module that is being parsed by autodoc"""

    if not bool(app.config.sphinx_gallery_conf['backreferences_dir']):
        return

    examples_path = os.path.join(app.srcdir,
                                 app.config.sphinx_gallery_conf[
                                     "backreferences_dir"],
                                 "%s.examples" % name)

    if not os.path.exists(examples_path):
        # touch file
        open(examples_path, 'w').close()


def _expected_failing_examples(gallery_conf):
    return set(
        os.path.normpath(os.path.join(gallery_conf['src_dir'], path))
        for path in gallery_conf['expected_failing_examples'])


def _parse_failures(gallery_conf):
    """Split the failures."""
    failing_examples = set(gallery_conf['failing_examples'].keys())
    expected_failing_examples = _expected_failing_examples(gallery_conf)
    failing_as_expected = failing_examples.intersection(
        expected_failing_examples)
    failing_unexpectedly = failing_examples.difference(
        expected_failing_examples)
    passing_unexpectedly = expected_failing_examples.difference(
        failing_examples)
    # filter from examples actually run
    passing_unexpectedly = [
        src_file for src_file in passing_unexpectedly
        if re.search(gallery_conf.get('filename_pattern'), src_file)]
    return failing_as_expected, failing_unexpectedly, passing_unexpectedly


def summarize_failing_examples(app, exception):
    """Collects the list of falling examples and prints them with a traceback.

    Raises ValueError if there where failing examples.
    """
    if exception is not None:
        return

    # Under no-plot Examples are not run so nothing to summarize
    if not app.config.sphinx_gallery_conf['plot_gallery']:
        logger.info('Sphinx-gallery gallery_conf["plot_gallery"] was '
                    'False, so no examples were executed.', color='brown')
        return

    gallery_conf = app.config.sphinx_gallery_conf
    failing_as_expected, failing_unexpectedly, passing_unexpectedly = \
        _parse_failures(gallery_conf)

    if failing_as_expected:
        logger.info("Examples failing as expected:", color='brown')
        for fail_example in failing_as_expected:
            logger.info('%s failed leaving traceback:', fail_example,
                        color='brown')
            logger.info(gallery_conf['failing_examples'][fail_example],
                        color='brown')

    fail_msgs = []
    if failing_unexpectedly:
        fail_msgs.append(red("Unexpected failing examples:"))
        for fail_example in failing_unexpectedly:
            fail_msgs.append(fail_example + ' failed leaving traceback:\n' +
                             gallery_conf['failing_examples'][fail_example] +
                             '\n')

    if passing_unexpectedly:
        fail_msgs.append(red("Examples expected to fail, but not failing:\n") +
                         "Please remove these examples from\n" +
                         "sphinx_gallery_conf['expected_failing_examples']\n" +
                         "in your conf.py file"
                         "\n".join(passing_unexpectedly))

    # standard message
    n_good = len(gallery_conf['passing_examples'])
    n_tot = len(gallery_conf['failing_examples']) + n_good
    n_stale = len(gallery_conf['stale_examples'])
    logger.info('\nSphinx-gallery successfully executed %d out of %d '
                'file%s subselected by:\n\n'
                '    gallery_conf["filename_pattern"] = %r\n'
                '    gallery_conf["ignore_pattern"]   = %r\n'
                '\nafter excluding %d file%s that had previously been run '
                '(based on MD5).\n'
                % (n_good, n_tot, 's' if n_tot != 1 else '',
                   gallery_conf['filename_pattern'],
                   gallery_conf['ignore_pattern'],
                   n_stale, 's' if n_stale != 1 else '',
                   ),
                color='brown')

    if fail_msgs:
        raise ValueError("Here is a summary of the problems encountered when "
                         "running the examples\n\n" + "\n".join(fail_msgs) +
                         "\n" + "-" * 79)


def collect_gallery_files(examples_dirs):
    """Collect python files from the gallery example directories."""
    files = []
    for example_dir in examples_dirs:
        for root, dirnames, filenames in os.walk(example_dir):
            for filename in filenames:
                if filename.endswith('.py'):
                    files.append(os.path.join(root, filename))
    return files


def check_duplicate_filenames(files):
    """Check for duplicate filenames across gallery directories."""
    # Check whether we'll have duplicates
    used_names = set()
    dup_names = list()

    for this_file in files:
        this_fname = os.path.basename(this_file)
        if this_fname in used_names:
            dup_names.append(this_file)
        else:
            used_names.add(this_fname)

    if len(dup_names) > 0:
        logger.warning(
            'Duplicate file name(s) found. Having duplicate file names will '
            'break some links. List of files: {}'.format(sorted(dup_names),))


def get_default_config_value(key):
    def default_getter(conf):
        return conf['sphinx_gallery_conf'].get(key, DEFAULT_GALLERY_CONF[key])
    return default_getter


def setup(app):
    """Setup sphinx-gallery sphinx extension"""
    sphinx_compatibility._app = app

    app.add_config_value('sphinx_gallery_conf', DEFAULT_GALLERY_CONF, 'html')
    for key in ['plot_gallery', 'abort_on_example_error']:
        app.add_config_value(key, get_default_config_value(key), 'html')

    try:
        app.add_css_file('gallery.css')
    except AttributeError:  # Sphinx < 1.8
        app.add_stylesheet('gallery.css')

    # Sphinx < 1.6 calls it `_extensions`, >= 1.6 is `extensions`.
    extensions_attr = '_extensions' if hasattr(
        app, '_extensions') else 'extensions'
    if 'sphinx.ext.autodoc' in getattr(app, extensions_attr):
        app.connect('autodoc-process-docstring', touch_empty_backreferences)

    app.connect('builder-inited', generate_gallery_rst)
    app.connect('build-finished', copy_binder_files)
    app.connect('build-finished', summarize_failing_examples)
    app.connect('build-finished', embed_code_links)
    metadata = {'parallel_read_safe': True,
                'parallel_write_safe': False,
                'version': _sg_version}
    return metadata


def setup_module():
    # HACK: Stop nosetests running setup() above
    pass
