# -*- coding: utf-8 -*-
r"""
Test utility functions
==================


"""
# Author: Nicholas Cain
# License: 3-clause BSD

from __future__ import division, absolute_import, print_function
import sphinx_gallery.utils as utils
import pytest

def test_replace_py_ipynb():
    # Test behavior of function with expected input:
    for file_name in ['some/file/name', '/corner.pycase']:
        assert utils.replace_py_ipynb(file_name+'.py') == file_name+'.ipynb'

    # Test behavior of function with unexpected input:
    with pytest.raises(ValueError) as expected_exception:
        utils.replace_py_ipynb(file_name+'.txt')

    assert 'Unrecognized file extension' in str(expected_exception.value)
