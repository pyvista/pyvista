# -*- coding: utf-8 -*-
# Author: Chris Holdgraf
# License: 3-clause BSD
"""
Binder utility functions
========================

Integration with Binder is on an experimental stage. Note that this API may
change in the future.

.. warning::

   Binder is still beta technology, so there may be instability in the
   experience of users who click Binder links.

"""

import shutil
import os

try:
    basestring
except NameError:
    basestring = str
    unicode = str

from .utils import replace_py_ipynb
from . import sphinx_compatibility


logger = sphinx_compatibility.getLogger('sphinx-gallery')


def gen_binder_url(fpath, binder_conf, gallery_conf):
    """Generate a Binder URL according to the configuration in conf.py.

    Parameters
    ----------
    fpath: str
        The path to the `.py` file for which a Binder badge will be generated.
    binder_conf: dict or None
        The Binder configuration dictionary. See `gen_binder_rst` for details.

    Returns
    -------
    binder_url : str
        A URL that can be used to direct the user to the live Binder
        environment.
    """
    # Build the URL
    fpath_prefix = binder_conf.get('filepath_prefix')
    link_base = binder_conf.get('notebooks_dir')

    # We want to keep the relative path to sub-folders
    relative_link = os.path.relpath(fpath, gallery_conf['src_dir'])
    path_link = os.path.join(
        link_base, replace_py_ipynb(relative_link))

    # In case our website is hosted in a sub-folder
    if fpath_prefix is not None:
        path_link = '/'.join([fpath_prefix.strip('/'), path_link])

    # Make sure we have the right slashes (in case we're on Windows)
    path_link = path_link.replace(os.path.sep, '/')

    # Create the URL
    binder_url = binder_conf['binderhub_url']
    binder_url = '/'.join([binder_conf['binderhub_url'],
                           'v2', 'gh',
                           binder_conf['org'],
                           binder_conf['repo'],
                           binder_conf['branch']])

    if binder_conf.get('use_jupyter_lab', False) is True:
        binder_url += '?urlpath=lab/tree/{}'.format(path_link)
    else:
        binder_url += '?filepath={}'.format(path_link)
    return binder_url


def gen_binder_rst(fpath, binder_conf, gallery_conf):
    """Generate the RST + link for the Binder badge.

    Parameters
    ----------
    fpath: str
        The path to the `.py` file for which a Binder badge will be generated.
    binder_conf: dict or None
        If a dictionary it must have the following keys:

        'binderhub_url': The URL of the BinderHub instance that's running a Binder
            service.
        'org': The GitHub organization to which the documentation will be
            pushed.
        'repo': The GitHub repository to which the documentation will be
            pushed.
        'branch': The Git branch on which the documentation exists (e.g.,
            gh-pages).
        'dependencies': A list of paths to dependency files that match the
            Binderspec.

    Returns
    -------
    rst : str
        The reStructuredText for the Binder badge that links to this file.
    """
    binder_conf = check_binder_conf(binder_conf)
    binder_url = gen_binder_url(fpath, binder_conf, gallery_conf)

    rst = (
        "\n"
        "  .. container:: binder-badge\n\n"
        "    .. image:: https://mybinder.org/badge_logo.svg\n"
        "      :target: {}\n"
        "      :width: 150 px\n").format(binder_url)
    return rst


def copy_binder_files(app, exception):
    """Copy all Binder requirements and notebooks files."""
    if exception is not None:
        return

    if app.builder.name not in ['html', 'readthedocs']:
        return

    gallery_conf = app.config.sphinx_gallery_conf
    binder_conf = check_binder_conf(gallery_conf.get('binder'))

    if not len(binder_conf) > 0:
        return

    logger.info('copying binder requirements...', color='white')
    _copy_binder_reqs(app, binder_conf)
    _copy_binder_notebooks(app)


def _copy_binder_reqs(app, binder_conf):
    """Copy Binder requirements files to a "binder" folder in the docs."""
    path_reqs = binder_conf.get('dependencies')
    for path in path_reqs:
        if not os.path.exists(os.path.join(app.srcdir, path)):
            raise ValueError(("Couldn't find the Binder requirements file: {}, "
                              "did you specify the path correctly?".format(path)))

    binder_folder = os.path.join(app.outdir, 'binder')
    if not os.path.isdir(binder_folder):
        os.makedirs(binder_folder)

    # Copy over the requirements to the output directory
    for path in path_reqs:
        shutil.copy(os.path.join(app.srcdir, path), binder_folder)


def _remove_ipynb_files(path, contents):
    """Given a list of files in `contents`, remove all files named `ipynb` or
    directories named `images` and return the result.

    Used with the `shutil` "ignore" keyword to filter out non-ipynb files."""
    contents_return = []
    for entry in contents:
        if entry.endswith('.ipynb'):
            # Don't include ipynb files
            pass
        elif (entry != "images") and os.path.isdir(os.path.join(path, entry)):
            # Don't include folders not called "images"
            pass
        else:
            # Keep everything else
            contents_return.append(entry)
    return contents_return


def _copy_binder_notebooks(app):
    """Copy Jupyter notebooks to the binder notebooks directory.

    Copy each output gallery directory structure but only including the
    Jupyter notebook files."""

    gallery_conf = app.config.sphinx_gallery_conf
    gallery_dirs = gallery_conf.get('gallery_dirs')
    binder_conf = gallery_conf.get('binder')
    notebooks_dir = os.path.join(app.outdir, binder_conf.get('notebooks_dir'))
    shutil.rmtree(notebooks_dir, ignore_errors=True)
    os.makedirs(notebooks_dir)

    if not isinstance(gallery_dirs, (list, tuple)):
        gallery_dirs = [gallery_dirs]

    iterator = sphinx_compatibility.status_iterator(
        gallery_dirs, 'copying binder notebooks...', length=len(gallery_dirs))

    for i_folder in iterator:
        shutil.copytree(os.path.join(app.srcdir, i_folder),
                        os.path.join(notebooks_dir, i_folder),
                        ignore=_remove_ipynb_files)


def check_binder_conf(binder_conf):
    """Check to make sure that the Binder configuration is correct."""
    # Grab the configuration and return None if it's not configured
    binder_conf = {} if binder_conf is None else binder_conf
    if not isinstance(binder_conf, dict):
        raise ValueError('`binder_conf` must be a dictionary or None.')
    if len(binder_conf) == 0:
        return binder_conf

    if binder_conf.get('url') and not binder_conf.get('binderhub_url'):
        logger.warning(
            'Found old BinderHub URL keyword ("url"). Please update your '
            'configuration to use the new keyword ("binderhub_url"). "url" will be '
            'deprecated in sphinx-gallery v0.4')
        binder_conf['binderhub_url'] = binderhub_conf.get('url')

    # Ensure all fields are populated
    req_values = ['binderhub_url', 'org', 'repo', 'branch', 'dependencies']
    optional_values = ['filepath_prefix', 'notebooks_dir', 'use_jupyter_lab']
    missing_values = []
    for val in req_values:
        if binder_conf.get(val) is None:
            missing_values.append(val)

    if len(missing_values) > 0:
        raise ValueError('binder_conf is missing values for: {}'.format(
            missing_values))

    for key in binder_conf.keys():
        if key not in (req_values + optional_values):
            raise ValueError("Unknown Binder config key: {}".format(key))

    # Ensure we have http in the URL
    if not any(binder_conf['binderhub_url'].startswith(ii)
               for ii in ['http://', 'https://']):
        raise ValueError('did not supply a valid url, '
                         'gave binderhub_url: {}'.format(binder_conf['binderhub_url']))

    # Ensure we have at least one dependency file
    # Need at least one of these three files
    required_reqs_files = ['requirements.txt', 'environment.yml', 'Dockerfile']
    path_reqs = binder_conf['dependencies']
    if isinstance(path_reqs, basestring):
        path_reqs = [path_reqs]
        binder_conf['dependencies'] = path_reqs
    elif not isinstance(path_reqs, (list, tuple)):
        raise ValueError("`dependencies` value should be a list of strings. "
                         "Got type {}.".format(type(path_reqs)))

    binder_conf['notebooks_dir'] = binder_conf.get('notebooks_dir',
                                                   'notebooks')
    path_reqs_filenames = [os.path.basename(ii) for ii in path_reqs]
    if not any(ii in path_reqs_filenames for ii in required_reqs_files):
        raise ValueError(
            'Did not find one of `requirements.txt` or `environment.yml` '
            'in the "dependencies" section of the binder configuration '
            'for sphinx-gallery. A path to at least one of these files '
            'must exist in your Binder dependencies.')
    return binder_conf
