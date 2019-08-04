# -*- coding: utf-8 -*-
# Author: Óscar Nájera
# License: 3-clause BSD
"""
Testing the rst files generator
"""
from __future__ import (division, absolute_import, print_function,
                        unicode_literals)
import ast
import codecs
import io
import tempfile
import re
import os
import shutil
import zipfile
import codeop

import pytest

import sphinx_gallery.gen_rst as sg
from sphinx_gallery import downloads
from sphinx_gallery.gen_gallery import generate_dir_rst, _complete_gallery_conf
from sphinx_gallery.utils import _TempDir, Bunch
from sphinx_gallery.scrapers import ImagePathIterator

try:
    FileNotFoundError
except NameError:
    # Python2
    FileNotFoundError = IOError

CONTENT = [
    '"""',
    '================',
    'Docstring header',
    '================',
    '',
    'This is the description of the example',
    'which goes on and on, Óscar',
    '',
    '',
    'And this is a second paragraph',
    '"""',
    '',
    '# sphinx_gallery_thumbnail_number = 1'
    '# and now comes the module code',
    'import logging',
    'import sys',
    'from warnings import warn',
    'x, y = 1, 2',
    'print(u"Óscar output") # need some code output',
    'logger = logging.getLogger()',
    'logger.setLevel(logging.INFO)',
    'lh = logging.StreamHandler(sys.stdout)',
    'lh.setFormatter(logging.Formatter("log:%(message)s"))',
    'logger.addHandler(lh)',
    'logger.info(u"Óscar")',
    'print(r"$\\langle n_\\uparrow n_\\downarrow \\rangle$")',
    'warn("WarningsAbound", RuntimeWarning)',
]


def test_split_code_and_text_blocks():
    """Test if a known example file gets properly split"""

    file_conf, blocks = sg.split_code_and_text_blocks(
        'examples/no_output/just_code.py')

    assert file_conf == {}
    assert blocks[0][0] == 'text'
    assert blocks[1][0] == 'code'


def test_bug_cases_of_notebook_syntax():
    """Test over the known requirements of supported syntax in the
    notebook styled comments"""

    with open('sphinx_gallery/tests/reference_parse.txt') as reference:
        ref_blocks = ast.literal_eval(reference.read())
        file_conf, blocks = sg.split_code_and_text_blocks(
            'tutorials/plot_parse.py')

        assert file_conf == {}
        assert blocks == ref_blocks


def test_direct_comment_after_docstring():
    # For more details see
    # https://github.com/sphinx-gallery/sphinx-gallery/pull/49
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        f.write('\n'.join(['"Docstring"',
                           '# and now comes the module code',
                           '# with a second line of comment',
                           'x, y = 1, 2',
                           '']))
    try:
        file_conf, result = sg.split_code_and_text_blocks(f.name)
    finally:
        os.remove(f.name)

    assert file_conf == {}
    expected_result = [
        ('text', 'Docstring', 1),
        ('code', '\n'.join(['# and now comes the module code',
                            '# with a second line of comment',
                            'x, y = 1, 2',
                            '']), 2)]
    assert result == expected_result


def test_rst_block_after_docstring(gallery_conf, tmpdir):
    """Assert there is a blank line between the docstring and rst blocks."""
    filename = str(tmpdir.join('temp.py'))
    with open(filename, 'w') as f:
        f.write('\n'.join(['"Docstring"',
                           '####################',
                           '# Paragraph 1',
                           '',
                           '####################',
                           '# Paragraph 2',
                           '']))
    file_conf, blocks = sg.split_code_and_text_blocks(filename)

    assert file_conf == {}
    assert blocks[0][0] == 'text'
    assert blocks[1][0] == 'text'
    assert blocks[2][0] == 'text'

    script_vars = {'execute_script': ''}

    output_blocks, time_elapsed = sg.execute_script(blocks,
                                                    script_vars,
                                                    gallery_conf)

    example_rst = sg.rst_blocks(blocks, output_blocks, file_conf, gallery_conf)
    assert example_rst == '\n'.join([
        'Docstring',
        '',
        'Paragraph 1',
        '',
        'Paragraph 2',
        '',
        ''])


def test_script_vars_globals(gallery_conf, tmpdir):
    """Assert the global vars get stored."""
    filename = str(tmpdir.join('temp.py'))
    with open(filename, 'w') as f:
        f.write("""
'''
My example
----------

This is it.
'''
a = 1.
b = 'foo'
""")
    file_conf, blocks = sg.split_code_and_text_blocks(filename)
    assert len(blocks) == 2
    assert blocks[0][0] == 'text'
    assert blocks[1][0] == 'code'
    assert file_conf == {}
    script_vars = {'execute_script': True, 'src_file': filename,
                   'image_path_iterator': [],
                   'target_file': filename}
    output_blocks, time_elapsed = sg.execute_script(blocks,
                                                    script_vars,
                                                    gallery_conf)
    assert 'example_globals' in script_vars
    assert script_vars['example_globals']['a'] == 1.
    assert script_vars['example_globals']['b'] == 'foo'


def test_codestr2rst():
    """Test the correct translation of a code block into rst."""
    output = sg.codestr2rst('print("hello world")')
    reference = """
.. code-block:: python

    print("hello world")"""
    assert reference == output


def test_extract_intro_and_title():
    intro, title = sg.extract_intro_and_title('<string>',
                                              '\n'.join(CONTENT[1:10]))
    assert title == 'Docstring header'
    assert 'Docstring' not in intro
    assert intro == 'This is the description of the example which goes on and on, Óscar'  # noqa
    assert 'second paragraph' not in intro

    # SG incorrectly grabbing description when a label is defined (gh-232)
    intro_label, title_label = sg.extract_intro_and_title(
        '<string>', '\n'.join(['.. my_label', ''] + CONTENT[1:10]))
    assert intro_label == intro
    assert title_label == title

    intro_whitespace, title_whitespace = sg.extract_intro_and_title(
        '<string>', '\n'.join(CONTENT[1:4] + [''] + CONTENT[5:10]))
    assert intro_whitespace == intro
    assert title_whitespace == title

    # Make example title optional (gh-222)
    intro, title = sg.extract_intro_and_title('<string>', 'Title\n-----')
    assert intro == title == 'Title'

    # Title beginning with a space (gh-356)
    intro, title = sg.extract_intro_and_title('filename',
                                              '^^^^^\n   Title  two  \n^^^^^')
    assert intro == title == 'Title  two'

    # Long intro paragraph gets shortened
    intro_paragraph = '\n'.join(['this is one line' for _ in range(10)])
    intro, _ = sg.extract_intro_and_title(
        'filename',
        'Title\n-----\n\n' + intro_paragraph)
    assert len(intro_paragraph) > 100
    assert len(intro) < 100
    assert intro.endswith('...')
    assert intro_paragraph.replace('\n', ' ')[:95] == intro[:95]

    # Errors
    with pytest.raises(ValueError, match='should have a header'):
        sg.extract_intro_and_title('<string>', '')  # no title
    with pytest.raises(ValueError, match='Could not find a title'):
        sg.extract_intro_and_title('<string>', '-')  # no real title


def test_md5sums():
    """Test md5sum check functions work on know file content."""
    with tempfile.NamedTemporaryFile('wb', delete=False) as f:
        f.write(b'Local test\n')
    try:
        file_md5 = sg.get_md5sum(f.name)
        # verify correct md5sum
        assert 'ea8a570e9f3afc0a7c3f2a17a48b8047' == file_md5
        # False because is a new file
        assert not sg.md5sum_is_current(f.name)
        # Write md5sum to file to check is current
        with open(f.name + '.md5', 'w') as file_checksum:
            file_checksum.write(file_md5)
        try:
            assert sg.md5sum_is_current(f.name)
        finally:
            os.remove(f.name + '.md5')
    finally:
        os.remove(f.name)


@pytest.fixture
def gallery_conf(tmpdir):
    """Set up a test sphinx-gallery configuration."""
    app = Bunch()
    app.config = dict(source_suffix={'.rst': None})
    gallery_conf = _complete_gallery_conf({}, str(tmpdir), True, False,
                                          app=app)
    gallery_conf.update(examples_dir=_TempDir(), gallery_dir=str(tmpdir))
    return gallery_conf


def test_fail_example(gallery_conf, log_collector):
    """Test that failing examples are only executed until failing block."""
    gallery_conf.update(filename_pattern='raise.py')

    failing_code = CONTENT + ['#' * 79,
                              'First_test_fail', '#' * 79, 'second_fail']

    with codecs.open(os.path.join(gallery_conf['examples_dir'], 'raise.py'),
                     mode='w', encoding='utf-8') as f:
        f.write('\n'.join(failing_code))

    sg.generate_file_rst('raise.py', gallery_conf['gallery_dir'],
                         gallery_conf['examples_dir'], gallery_conf)
    assert len(log_collector.calls['warning']) == 1
    assert 'not defined' in log_collector.calls['warning'][0].args[2]

    # read rst file and check if it contains traceback output

    with codecs.open(os.path.join(gallery_conf['gallery_dir'], 'raise.rst'),
                     mode='r', encoding='utf-8') as f:
        ex_failing_blocks = f.read().count('pytb')
        if ex_failing_blocks == 0:
            raise ValueError('Did not run into errors in bad code')
        elif ex_failing_blocks > 1:
            raise ValueError('Did not stop executing script after error')


def _generate_rst(gallery_conf, fname, content):
    """Return the rST text of a given example content.

    This writes a file gallery_conf['examples_dir']/fname with *content*,
    creates the corresponding rst file by running generate_file_rst() and
    returns the generated rST code.

    Parameters
    ----------
    gallery_conf
        A gallery_conf as created by the gallery_conf fixture.
    fname : str
        A filename; e.g. 'test.py'. This is relative to
        gallery_conf['examples_dir']
    content : str
        The content of fname.

    Returns
    -------
    rst : str
        The generated rST code.
    """
    with codecs.open(os.path.join(gallery_conf['examples_dir'], fname),
                     mode='w', encoding='utf-8') as f:
        f.write('\n'.join(content))
    # generate rst file
    sg.generate_file_rst(fname, gallery_conf['gallery_dir'],
                         gallery_conf['examples_dir'], gallery_conf)
    # read rst file and check if it contains code output
    rst_fname = os.path.splitext(fname)[0] + '.rst'
    with codecs.open(os.path.join(gallery_conf['gallery_dir'], rst_fname),
                     mode='r', encoding='utf-8') as f:
        rst = f.read()
    return rst


def test_remove_config_comments(gallery_conf):
    """Test the gallery_conf['remove_config_comments'] setting."""
    rst = _generate_rst(gallery_conf, 'test.py', CONTENT)
    assert '# sphinx_gallery_thumbnail_number = 1' in rst

    gallery_conf['remove_config_comments'] = True
    rst = _generate_rst(gallery_conf, 'test.py', CONTENT)
    assert '# sphinx_gallery_thumbnail_number = 1' not in rst


@pytest.mark.parametrize('ext', ('.txt', '.rst', '.bad'))
def test_gen_dir_rst(gallery_conf, fakesphinxapp, ext):
    """Test gen_dir_rst."""
    print(os.listdir(gallery_conf['examples_dir']))
    fname_readme = os.path.join(gallery_conf['src_dir'], 'README.txt')
    with open(fname_readme, 'wb') as fid:
        fid.write(u"Testing\n=======\n\nÓscar here.".encode('utf-8'))
    fname_out = os.path.splitext(fname_readme)[0] + ext
    if fname_readme != fname_out:
        shutil.move(fname_readme, fname_out)
    args = (gallery_conf['src_dir'], gallery_conf['gallery_dir'],
            gallery_conf, [])
    if ext == '.bad':  # not found with correct ext
        with pytest.raises(FileNotFoundError, match='does not have a README'):
            generate_dir_rst(*args)
    else:
        out = generate_dir_rst(*args)
        assert u"Óscar here" in out[0]


def test_pattern_matching(gallery_conf, log_collector):
    """Test if only examples matching pattern are executed."""
    gallery_conf.update(filename_pattern=re.escape(os.sep) + 'plot_0')

    code_output = ('\n Out:\n\n .. code-block:: none\n'
                   '\n'
                   '    Óscar output\n'
                   '    log:Óscar\n'
                   '    $\\langle n_\\uparrow n_\\downarrow \\rangle$'
                   )
    warn_output = 'RuntimeWarning: WarningsAbound'
    # create three files in tempdir (only one matches the pattern)
    fnames = ['plot_0.py', 'plot_1.py', 'plot_2.py']
    for fname in fnames:
        rst = _generate_rst(gallery_conf, fname, CONTENT)
        rst_fname = os.path.splitext(fname)[0] + '.rst'
        if re.search(gallery_conf['filename_pattern'],
                     os.path.join(gallery_conf['gallery_dir'], rst_fname)):
            assert code_output in rst
            assert warn_output in rst
        else:
            assert code_output not in rst
            assert warn_output not in rst


@pytest.mark.parametrize('test_str', [
    '# sphinx_gallery_thumbnail_number= 2',
    '# sphinx_gallery_thumbnail_number=2',
    '#sphinx_gallery_thumbnail_number = 2',
    '    # sphinx_gallery_thumbnail_number=2'])
def test_thumbnail_number(test_str):
    # which plot to show as the thumbnail image
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        f.write('\n'.join(['"Docstring"',
                           test_str]))
    try:
        file_conf, blocks = sg.split_code_and_text_blocks(f.name)
    finally:
        os.remove(f.name)
    assert file_conf == {'thumbnail_number': 2}


def test_zip_notebooks(gallery_conf):
    """Test generated zipfiles are not corrupt."""
    gallery_conf.update(examples_dir='examples')
    examples = downloads.list_downloadable_sources(
        gallery_conf['examples_dir'])
    zipfilepath = downloads.python_zip(examples, gallery_conf['gallery_dir'])
    zipf = zipfile.ZipFile(zipfilepath)
    check = zipf.testzip()
    if check:
        raise OSError("Bad file in zipfile: {0}".format(check))


def test_rst_example(gallery_conf):
    """Test generated rst file includes the correct paths for binder."""
    gallery_conf.update(binder={'org': 'sphinx-gallery',
                                'repo': 'sphinx-gallery.github.io',
                                'binderhub_url': 'https://mybinder.org',
                                'branch': 'master',
                                'dependencies': './binder/requirements.txt',
                                'notebooks_dir': 'notebooks',
                                'use_jupyter_lab': True,
                                })

    example_file = os.path.join(gallery_conf['gallery_dir'], "plot.py")
    sg.save_rst_example("example_rst", example_file, 0, 0, gallery_conf)

    test_file = re.sub(r'\.py$', '.rst', example_file)
    with codecs.open(test_file) as f:
        rst = f.read()

    assert "lab/tree/notebooks/plot.ipy" in rst

    # CSS classes
    assert "rst-class:: sphx-glr-signature" in rst
    assert "rst-class:: sphx-glr-timing" in rst


def test_output_indentation(gallery_conf):
    """Test whether indentation of code output is retained."""
    compiler = codeop.Compile()

    test_string = r"\n".join([
        "  A B",
        "A 1 2",
        "B 3 4"
    ])
    code = "print('" + test_string + "')"
    code_block = ("code", code, 1)

    script_vars = {
        "execute_script": True,
        "image_path_iterator": ImagePathIterator("temp.png"),
        "src_file": __file__,
        "memory_delta": [],
    }

    output = sg.execute_code_block(
        compiler, code_block, {}, script_vars, gallery_conf
    )
    output_test_string = "\n".join(
        [line[4:] for line in output.strip().split("\n")[-3:]]
    )
    assert output_test_string == test_string.replace(r"\n", "\n")


class TestLoggingTee:
    def setup(self):
        self.output_file = io.StringIO()
        self.src_filename = 'source file name'
        self.tee = sg.LoggingTee(self.output_file, sg.logger,
                                 self.src_filename)

    def test_full_line(self, log_collector):
        # A full line is output immediately.
        self.tee.write('Output\n')
        self.tee.flush()
        assert self.output_file.getvalue() == 'Output\n'
        assert len(log_collector.calls['verbose']) == 2
        assert self.src_filename in log_collector.calls['verbose'][0].args
        assert 'Output' in log_collector.calls['verbose'][1].args

    def test_incomplete_line_with_flush(self, log_collector):
        # An incomplete line ...
        self.tee.write('Output')
        assert self.output_file.getvalue() == 'Output'
        assert len(log_collector.calls['verbose']) == 1
        assert self.src_filename in log_collector.calls['verbose'][0].args

        # ... should appear when flushed.
        self.tee.flush()
        assert len(log_collector.calls['verbose']) == 2
        assert 'Output' in log_collector.calls['verbose'][1].args

    def test_incomplete_line_with_more_output(self, log_collector):
        # An incomplete line ...
        self.tee.write('Output')
        assert self.output_file.getvalue() == 'Output'
        assert len(log_collector.calls['verbose']) == 1
        assert self.src_filename in log_collector.calls['verbose'][0].args

        # ... should appear when more data is written.
        self.tee.write('\nMore output\n')
        assert self.output_file.getvalue() == 'Output\nMore output\n'
        assert len(log_collector.calls['verbose']) == 3
        assert 'Output' in log_collector.calls['verbose'][1].args
        assert 'More output' in log_collector.calls['verbose'][2].args

    def test_multi_line(self, log_collector):
        self.tee.write('first line\rsecond line\nthird line')
        assert (self.output_file.getvalue() ==
                'first line\rsecond line\nthird line')
        verbose_calls = log_collector.calls['verbose']
        assert len(verbose_calls) == 3
        assert self.src_filename in verbose_calls[0].args
        assert 'first line' in verbose_calls[1].args
        assert 'second line' in verbose_calls[2].args
        assert self.tee.logger_buffer == 'third line'

    def test_isatty(self, monkeypatch):
        assert not self.tee.isatty()

        monkeypatch.setattr(self.tee.output_file, 'isatty', lambda: True)
        assert self.tee.isatty()


# TODO: test that broken thumbnail does appear when needed
# TODO: test that examples are executed after a no-plot and produce
#       the correct image in the thumbnail
