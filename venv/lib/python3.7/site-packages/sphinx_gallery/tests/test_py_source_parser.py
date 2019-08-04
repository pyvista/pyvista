# -*- coding: utf-8 -*-
r"""
Test source parser
==================


"""
# Author: Óscar Nájera
# License: 3-clause BSD

from __future__ import division, absolute_import, print_function

import os.path as op
import pytest
import sphinx_gallery.py_source_parser as sg


def test_get_docstring_and_rest(unicode_sample, tmpdir):
    docstring, rest, lineno = sg.get_docstring_and_rest(unicode_sample)
    assert u'Únicode' in docstring
    assert u'heiß' in rest
    # degenerate
    fname = op.join(str(tmpdir), 'temp')
    with open(fname, 'w') as fid:
        fid.write('print("hello")\n')
    with pytest.raises(ValueError, match='Could not find docstring'):
        sg.get_docstring_and_rest(fname)


@pytest.mark.parametrize('content, file_conf', [
    ("No config\nin here.",
     {}),
    ("# sphinx_gallery_line_numbers = True",
     {'line_numbers': True}),
    ("  #   sphinx_gallery_line_numbers   =   True   ",
     {'line_numbers': True}),
    ("#sphinx_gallery_line_numbers=True",
     {'line_numbers': True}),
    ("#sphinx_gallery_thumbnail_number\n=\n5",
     {'thumbnail_number': 5}),
])
def test_extract_file_config(content, file_conf):
    assert sg.extract_file_config(content) == file_conf


@pytest.mark.parametrize('contents, result', [
    ("No config\nin here.",
     "No config\nin here."),
    ("# sphinx_gallery_line_numbers = True",
     ""),
    ("  #   sphinx_gallery_line_numbers   =   True   ",
     ""),
    ("#sphinx_gallery_line_numbers=True",
     ""),
    ("#sphinx_gallery_thumbnail_number\n=\n5",
     ""),
    ("a = 1\n# sphinx_gallery_line_numbers = True\nb = 1",
     "a = 1\nb = 1"),
    ("a = 1\n\n# sphinx_gallery_line_numbers = True\n\nb = 1",
     "a = 1\n\n\nb = 1"),
    ("# comment\n# sphinx_gallery_line_numbers = True\n# commment 2",
     "# comment\n# commment 2"),
])
def test_remove_config_comments(contents, result):
    assert sg.remove_config_comments(contents) == result
