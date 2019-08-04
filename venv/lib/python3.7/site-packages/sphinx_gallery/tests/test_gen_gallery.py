# -*- coding: utf-8 -*-
# Author: Óscar Nájera
# License: 3-clause BSD
r"""
Test Sphinx-Gallery
"""

from __future__ import (division, absolute_import, print_function,
                        unicode_literals)
import codecs
from contextlib import contextmanager
import os
import sys
import re
import shutil

import pytest

from sphinx.application import Sphinx
from sphinx.errors import ExtensionError
from sphinx.util.docutils import docutils_namespace

from sphinx_gallery.gen_rst import MixedEncodingStringIO
from sphinx_gallery import sphinx_compatibility
from sphinx_gallery.gen_gallery import (check_duplicate_filenames,
                                        collect_gallery_files)
from sphinx_gallery.utils import _TempDir


@pytest.fixture
def conf_file(request):
    try:
        env = request.node.get_closest_marker('conf_file')
    except AttributeError:  # old pytest
        env = request.node.get_marker('conf_file')
    kwargs = env.kwargs if env else {}
    result = {
        'content': "",
    }
    result.update(kwargs)

    return result


@pytest.fixture
def tempdir():
    """
    temporary directory that wrapped with `path` class.
    this fixture is for compat with old test implementation.
    """
    return _TempDir()


class SphinxAppWrapper(object):
    """Wrapper for sphinx.application.Application.

    This allows to control when the sphinx application is initialized, since
    part of the sphinx-gallery build is done in
    sphinx.application.Application.__init__ and the remainder is done in
    sphinx.application.Application.build.

    """

    def __init__(self, srcdir, confdir, outdir, doctreedir, buildername,
                 **kwargs):
        self.srcdir = srcdir
        self.confdir = confdir
        self.outdir = outdir
        self.doctreedir = doctreedir
        self.buildername = buildername
        self.kwargs = kwargs

    def create_sphinx_app(self):
        # Avoid warnings about re-registration, see:
        # https://github.com/sphinx-doc/sphinx/issues/5038
        with self.create_sphinx_app_context() as app:
            pass
        return app

    @contextmanager
    def create_sphinx_app_context(self):
        with docutils_namespace():
            app = Sphinx(self.srcdir, self.confdir, self.outdir,
                         self.doctreedir, self.buildername, **self.kwargs)
            sphinx_compatibility._app = app
            yield app

    def build_sphinx_app(self, *args, **kwargs):
        with self.create_sphinx_app_context() as app:
            # building should be done in the same docutils_namespace context
            app.build(*args, **kwargs)
        return app


@pytest.fixture
def sphinx_app_wrapper(tempdir, conf_file):
    _fixturedir = os.path.join(os.path.dirname(__file__), 'testconfs')
    srcdir = os.path.join(tempdir, "config_test")
    shutil.copytree(_fixturedir, srcdir)
    shutil.copytree(os.path.join(_fixturedir, "src"),
                    os.path.join(tempdir, "examples"))

    base_config = """
import os
import sphinx_gallery
extensions = ['sphinx_gallery.gen_gallery']
exclude_patterns = ['_build']
source_suffix = '.rst'
master_doc = 'index'
# General information about the project.
project = u'Sphinx-Gallery <Tests>'\n\n
"""
    with open(os.path.join(srcdir, "conf.py"), "w") as conffile:
        conffile.write(base_config + conf_file['content'])

    return SphinxAppWrapper(
        srcdir, srcdir, os.path.join(srcdir, "_build"),
        os.path.join(srcdir, "_build", "toctree"),
        "html", warning=MixedEncodingStringIO())


def test_default_config(sphinx_app_wrapper):
    """Test the default Sphinx-Gallery configuration is loaded

    if only the extension is added to Sphinx"""
    sphinx_app = sphinx_app_wrapper.create_sphinx_app()
    cfg = sphinx_app.config
    assert cfg.project == "Sphinx-Gallery <Tests>"
    # no duplicate values allowed The config is present already
    with pytest.raises(ExtensionError) as excinfo:
        sphinx_app.add_config_value('sphinx_gallery_conf', 'x', True)
    assert 'already present' in str(excinfo.value)


@pytest.mark.conf_file(content="""
sphinx_gallery_conf = {
    'examples_dirs': 'src',
    'gallery_dirs': 'ex',
}""")
def test_no_warning_simple_config(sphinx_app_wrapper):
    """Testing that no warning is issued with a simple config.

    The simple config only specifies input (examples_dirs) and output
    (gallery_dirs) directories.
    """
    sphinx_app = sphinx_app_wrapper.create_sphinx_app()
    cfg = sphinx_app.config
    assert cfg.project == "Sphinx-Gallery <Tests>"
    build_warn = sphinx_app._warning.getvalue()
    assert build_warn == ''


@pytest.mark.conf_file(content="""
sphinx_gallery_conf = {
    'mod_example_dir' : os.path.join('modules', 'gen'),
    'examples_dirs': 'src',
    'gallery_dirs': 'ex',
}""")
def test_config_old_backreferences_conf(sphinx_app_wrapper):
    """Testing Deprecation warning message against old backreference config

    In this case the user is required to update the mod_example_dir config
    variable Sphinx-Gallery should notify the user and also silently update
    the old config to the new one. """
    sphinx_app = sphinx_app_wrapper.create_sphinx_app()
    cfg = sphinx_app.config
    assert cfg.project == "Sphinx-Gallery <Tests>"
    assert cfg.sphinx_gallery_conf['backreferences_dir'] == os.path.join(
        'modules', 'gen')
    build_warn = sphinx_app._warning.getvalue()

    assert "WARNING:" in build_warn
    assert "deprecated" in build_warn
    assert "Support for 'mod_example_dir' will be removed" in build_warn


@pytest.mark.conf_file(content="""
sphinx_gallery_conf = {
    'backreferences_dir': os.path.join('gen_modules', 'backreferences'),
    'examples_dirs': 'src',
    'gallery_dirs': 'ex',
}""")
def test_config_backreferences(sphinx_app_wrapper):
    """Test no warning is issued under the new configuration"""
    sphinx_app = sphinx_app_wrapper.create_sphinx_app()
    cfg = sphinx_app.config
    assert cfg.project == "Sphinx-Gallery <Tests>"
    assert cfg.sphinx_gallery_conf['backreferences_dir'] == os.path.join(
        'gen_modules', 'backreferences')
    build_warn = sphinx_app._warning.getvalue()
    assert build_warn == ''


def test_duplicate_files_warn(sphinx_app_wrapper):
    """Test for a warning when two files with the same filename exist."""
    sphinx_app = sphinx_app_wrapper.create_sphinx_app()

    files = ['./a/file1.py', './a/file2.py', 'a/file3.py', './b/file1.py']
    msg = ("Duplicate file name(s) found. Having duplicate file names "
           "will break some links. List of files: {}")
    m = "['./b/file1.py']" if sys.version_info[0] >= 3 else "[u'./b/file1.py']"

    # No warning because no overlapping names
    check_duplicate_filenames(files[:-1])
    build_warn = sphinx_app._warning.getvalue()
    assert build_warn == ''

    # Warning because last file is named the same
    check_duplicate_filenames(files)
    build_warn = sphinx_app._warning.getvalue()
    assert msg.format(m) in build_warn


def _check_order(sphinx_app, key):
    index_fname = os.path.join(sphinx_app.outdir, '..', 'ex', 'index.rst')
    order = list()
    regex = '.*:%s=(.):.*' % key
    with codecs.open(index_fname, 'r', 'utf-8') as fid:
        for line in fid:
            if 'sphx-glr-thumbcontainer' in line:
                order.append(int(re.match(regex, line).group(1)))
    assert len(order) == 3
    assert order == [1, 2, 3]


@pytest.mark.conf_file(content="""
sphinx_gallery_conf = {
    'examples_dirs': 'src',
    'gallery_dirs': 'ex',
}""")
def test_example_sorting_default(sphinx_app_wrapper):
    """Test sorting of examples by default key (number of code lines)."""
    sphinx_app = sphinx_app_wrapper.create_sphinx_app()
    _check_order(sphinx_app, 'lines')


@pytest.mark.conf_file(content="""
from sphinx_gallery.sorting import FileSizeSortKey
sphinx_gallery_conf = {
    'examples_dirs': 'src',
    'gallery_dirs': 'ex',
    'within_subsection_order': FileSizeSortKey,
}""")
def test_example_sorting_filesize(sphinx_app_wrapper):
    """Test sorting of examples by filesize."""
    sphinx_app = sphinx_app_wrapper.create_sphinx_app()
    _check_order(sphinx_app, 'filesize')


@pytest.mark.conf_file(content="""
from sphinx_gallery.sorting import FileNameSortKey
sphinx_gallery_conf = {
    'examples_dirs': 'src',
    'gallery_dirs': 'ex',
    'within_subsection_order': FileNameSortKey,
}""")
def test_example_sorting_filename(sphinx_app_wrapper):
    """Test sorting of examples by filename."""
    sphinx_app = sphinx_app_wrapper.create_sphinx_app()
    _check_order(sphinx_app, 'filename')


@pytest.mark.conf_file(content="""
from sphinx_gallery.sorting import ExampleTitleSortKey
sphinx_gallery_conf = {
    'examples_dirs': 'src',
    'gallery_dirs': 'ex',
    'within_subsection_order': ExampleTitleSortKey,
}""")
def test_example_sorting_title(sphinx_app_wrapper):
    """Test sorting of examples by title."""
    sphinx_app = sphinx_app_wrapper.create_sphinx_app()
    _check_order(sphinx_app, 'title')


def test_collect_gallery_files(tmpdir):
    """Test that example files are collected properly."""
    rel_filepaths = ['examples/file1.py',
                     'examples/test.rst',
                     'examples/README.txt',
                     'examples/folder1/file1.py',
                     'examples/folder1/file2.py',
                     'examples/folder2/file1.py',
                     'tutorials/folder1/subfolder/file1.py',
                     'tutorials/folder2/subfolder/subsubfolder/file1.py']

    abs_paths = [tmpdir.join(rp) for rp in rel_filepaths]
    for ap in abs_paths:
        ap.ensure()

    examples_path = tmpdir.join('examples')
    dirs = [examples_path.strpath]
    collected_files = set(collect_gallery_files(dirs))
    expected_files = set(
        [ap.strpath for ap in abs_paths
         if re.search(r'examples.*\.py$', ap.strpath)])

    assert collected_files == expected_files

    tutorials_path = tmpdir.join('tutorials')
    dirs = [examples_path.strpath, tutorials_path.strpath]
    collected_files = set(collect_gallery_files(dirs))
    expected_files = set(
        [ap.strpath for ap in abs_paths if re.search(r'.*\.py$', ap.strpath)])

    assert collected_files == expected_files


@pytest.mark.conf_file(content="""
sphinx_gallery_conf = {
    'backreferences_dir' : os.path.join('modules', 'gen'),
    'examples_dirs': 'src',
    'gallery_dirs': ['ex'],
    'binder': {'binderhub_url': 'http://test1.com', 'org': 'org',
               'repo': 'repo', 'branch': 'branch',
               'notebooks_dir': 'ntbk_folder',
               'dependencies': 'requirements.txt'}
}""")
def test_binder_copy_files(sphinx_app_wrapper, tmpdir):
    """Test that notebooks are copied properly."""
    from sphinx_gallery.binder import copy_binder_files
    sphinx_app = sphinx_app_wrapper.create_sphinx_app()
    gallery_conf = sphinx_app.config.sphinx_gallery_conf
    # Create requirements file
    with open(os.path.join(sphinx_app.srcdir, 'requirements.txt'), 'w'):
        pass
    copy_binder_files(sphinx_app, None)

    for i_file in ['plot_1', 'plot_2', 'plot_3']:
        assert os.path.exists(os.path.join(
            sphinx_app.outdir, 'ntbk_folder', gallery_conf['gallery_dirs'][0],
            i_file+'.ipynb'))


@pytest.mark.conf_file(content="""
sphinx_gallery_conf = {
    'examples_dirs': 'src',
    'gallery_dirs': 'ex',
}""")
def test_failing_examples_raise_exception(sphinx_app_wrapper):
    example_dir = os.path.join(sphinx_app_wrapper.srcdir,
                               'src')
    with codecs.open(os.path.join(example_dir, 'plot_3.py'), 'a',
                     encoding='utf-8') as fid:
        fid.write('raise SyntaxError')
    with pytest.raises(ValueError) as excinfo:
        sphinx_app_wrapper.build_sphinx_app()
    assert "Unexpected failing examples" in str(excinfo.value)


@pytest.mark.conf_file(content="""
sphinx_gallery_conf = {
    'examples_dirs': 'src',
    'gallery_dirs': 'ex',
    'filename_pattern': 'plot_1.py',
}""")
def test_expected_failing_examples_were_executed(sphinx_app_wrapper):
    """Testing that no exception is issued when broken example is not built

    See #335 for more details.
    """
    sphinx_app_wrapper.build_sphinx_app()


@pytest.mark.conf_file(content="""
sphinx_gallery_conf = {
    'examples_dirs': 'src',
    'gallery_dirs': 'ex',
    'expected_failing_examples' :['src/plot_2.py'],
}""")
def test_examples_not_expected_to_pass(sphinx_app_wrapper):
    with pytest.raises(ValueError) as excinfo:
        sphinx_app_wrapper.build_sphinx_app()
    assert "expected to fail, but not failing" in str(excinfo.value)


@pytest.mark.conf_file(content="""
sphinx_gallery_conf = {
    'first_notebook_cell': 2,
}""")
def test_first_notebook_cell_config(sphinx_app_wrapper):
    from sphinx_gallery.gen_gallery import parse_config
    # First cell must be str
    with pytest.raises(ValueError):
        parse_config(sphinx_app_wrapper.create_sphinx_app())
