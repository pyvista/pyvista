# -*- coding: utf-8 -*-
"""
    sphinxcontrib.napoleon._upstream
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Functions to help compatibility with upstream sphinx.ext.napoleon.

    :copyright: Copyright 2013-2018 by Rob Ruana, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""


def _(message, *args):
    """
    NOOP implementation of sphinx.locale.get_translation shortcut.
    """
    return message
