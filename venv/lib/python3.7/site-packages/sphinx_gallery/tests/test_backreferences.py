# -*- coding: utf-8 -*-
# Author: Óscar Nájera
# License: 3-clause BSD
"""
Testing the rst files generator
"""
from __future__ import division, absolute_import, print_function

import pytest
import sphinx_gallery.backreferences as sg
from sphinx_gallery.gen_rst import _sanitize_rst


REFERENCE = r"""
.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="{0}">

.. only:: html

    .. figure:: /fake_dir/images/thumb/sphx_glr_test_file_thumb.png

        :ref:`sphx_glr_fake_dir_test_file.py`

.. raw:: html

    </div>{1}
"""


@pytest.mark.parametrize('content, tooltip, is_backref', [
    # HTML sanitizing
    ('<"test">', '&lt;&quot;test&quot;&gt;', False),
    # backref support
    ('test formating', 'test formating', True),
    # RST sanitizing
    ('1 :class:`~a.b`. 2 :class:`a.b` 3 :ref:`whatever <better name>`',
     '1 b. 2 a.b 3 better name', False),
    ('use :meth:`mne.io.Raw.plot_psd` to',
     'use mne.io.Raw.plot_psd to', False),
    ('`this` and ``that``; and `these things` and ``those things``',
     'this and that; and these things and those things', False),
])
def test_thumbnail_div(content, tooltip, is_backref):
    """Test if the thumbnail div generates the correct string."""
    with pytest.raises(RuntimeError, match='internal sphinx-gallery thumb'):
        html_div = sg._thumbnail_div('fake_dir', '', 'test_file.py',
                                     '<"test">')
    content = _sanitize_rst(content)
    html_div = sg._thumbnail_div('fake_dir', '', 'test_file.py',
                                 content, is_backref=is_backref, check=False)
    if is_backref:
        extra = """

.. only:: not html

    * :ref:`sphx_glr_fake_dir_test_file.py`"""
    else:
        extra = ''
    reference = REFERENCE.format(tooltip, extra)
    assert html_div == reference


def test_identify_names(unicode_sample):
    """Test name identification."""
    expected = {
        'os.path.join':
            {'name': 'join', 'module': 'os.path', 'module_short': 'os.path'},
        'br.identify_names':
            {'name': 'identify_names',
             'module': 'sphinx_gallery.back_references',
             'module_short': 'sphinx_gallery.back_references'},
        'identify_names':
            {'name': 'identify_names',
             'module': 'sphinx_gallery.back_references',
             'module_short': 'sphinx_gallery.back_references'}
    }

    res = sg.identify_names(unicode_sample)
    assert expected == res


def test_identify_names2(tmpdir):
    """Test more name identification."""
    code_str = b"""
'''
Title
-----

This is an example.
'''
# -*- coding: utf-8 -*-
# \xc3\x9f
from a.b import c
import d as e
print(c)
e.HelloWorld().f.g
"""
    expected = {'c': {'name': 'c', 'module': 'a.b', 'module_short': 'a.b'},
                'e.HelloWorld': {'name': 'HelloWorld', 'module': 'd',
                                 'module_short': 'd'}}

    fname = tmpdir.join("indentify_names.py")
    fname.write(code_str, 'wb')

    res = sg.identify_names(fname.strpath)

    assert expected == res

    code_str = b"""
'''
Title
-----

This example uses :func:`h.i`.
'''
""" + code_str.split(b"'''")[-1]
    expected['h.i'] = {u'module': u'h', u'module_short': u'h', u'name': u'i'}

    fname = tmpdir.join("indentify_names.py")
    fname.write(code_str, 'wb')

    res = sg.identify_names(fname.strpath)

    assert expected == res
