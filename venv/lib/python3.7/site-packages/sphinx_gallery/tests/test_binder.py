# -*- coding: utf-8 -*-
# Author: Chris Holdgraf
# License: 3-clause BSD
"""
Testing the binder badge functionality
"""
from __future__ import division, absolute_import, print_function

from copy import deepcopy

import pytest

from sphinx_gallery.binder import (gen_binder_url, check_binder_conf,
                                   _copy_binder_reqs)


def test_binder():
    """Testing binder URL generation and checks."""
    file_path = 'blahblah/mydir/myfile.py'
    conf_base = {'binderhub_url': 'http://test1.com', 'org': 'org',
                 'repo': 'repo', 'branch': 'branch',
                 'dependencies': '../requirements.txt'}
    conf_base = check_binder_conf(conf_base)
    gallery_conf_base = {'gallery_dirs': 'mydir', 'src_dir': 'blahblah'}

    url = gen_binder_url(file_path, conf_base, gallery_conf_base)
    expected = ('http://test1.com/v2/gh/org/repo/'
                'branch?filepath=notebooks/mydir/myfile.ipynb')
    assert url == expected

    # Assert filepath prefix is added
    prefix = 'my_prefix/foo'
    conf1 = deepcopy(conf_base)
    conf1['filepath_prefix'] = prefix
    url = gen_binder_url(file_path, conf1, gallery_conf_base)
    expected = ('http://test1.com/v2/gh/org/repo/'
                'branch?filepath={}/notebooks/'
                'mydir/myfile.ipynb').format(prefix)

    assert url == expected
    conf1.pop('filepath_prefix')

    # URL must have http
    conf2 = deepcopy(conf1)
    conf2['binderhub_url'] = 'test1.com'
    with pytest.raises(ValueError, match='did not supply a valid url'):
        url = check_binder_conf(conf2)

    # Assert missing params
    for key in conf1.keys():
        if key == 'notebooks_dir':
            continue
        conf3 = deepcopy(conf1)
        conf3.pop(key)
        with pytest.raises(ValueError, match='binder_conf is missing values'):
            url = check_binder_conf(conf3)

    # Dependencies file
    dependency_file_tests = ['requirements_not.txt', 'doc-requirements.txt']
    for ifile in dependency_file_tests:
        conf3 = deepcopy(conf1)
        conf3['dependencies'] = ifile
        with pytest.raises(ValueError,
                           match=r"Did not find one of `requirements.txt` "
                           "or `environment.yml`"):
            url = check_binder_conf(conf3)

    conf6 = deepcopy(conf1)
    conf6['dependencies'] = {'test': 'test'}
    with pytest.raises(ValueError, match='`dependencies` value should be a '
                       'list of strings'):
        url = check_binder_conf(conf6)

    # Missing requirements file should raise an error
    conf7 = deepcopy(conf1)
    conf7['dependencies'] = ['requirements.txt', 'this_doesntexist.txt']
    # Hack to test the case when dependencies point to a non-existing file
    # w/o needing to do a full build

    def apptmp():
        pass
    apptmp.srcdir = '/'
    with pytest.raises(ValueError, match="Couldn't find the Binder "
                       "requirements file"):
        url = _copy_binder_reqs(apptmp, conf7)

    # Check returns the correct object
    conf4 = check_binder_conf({})
    conf5 = check_binder_conf(None)
    for iconf in [conf4, conf5]:
        assert iconf == {}

    # Assert extra unkonwn params
    conf7 = deepcopy(conf1)
    conf7['foo'] = 'blah'
    with pytest.raises(ValueError, match='Unknown Binder config key'):
        url = check_binder_conf(conf7)

    # Assert using lab correctly changes URL
    conf_lab = deepcopy(conf_base)
    conf_lab['use_jupyter_lab'] = True
    url = gen_binder_url(file_path, conf_lab, gallery_conf_base)
    expected = ('http://test1.com/v2/gh/org/repo/'
                'branch?urlpath=lab/tree/notebooks/mydir/myfile.ipynb')
    assert url == expected

    # Assert using static folder correctly changes URL
    conf_static = deepcopy(conf_base)
    file_path = 'blahblah/mydir/myfolder/myfile.py'
    conf_static['notebooks_dir'] = 'ntbk_folder'
    url = gen_binder_url(file_path, conf_static, gallery_conf_base)
    expected = ('http://test1.com/v2/gh/org/repo/'
                'branch?filepath=ntbk_folder/mydir/myfolder/myfile.ipynb')
    assert url == expected
