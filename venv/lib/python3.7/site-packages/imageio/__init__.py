# -*- coding: utf-8 -*-
# Copyright (c) 2014-2018, imageio contributors
# imageio is distributed under the terms of the (new) BSD License.

# This docstring is used at the index of the documentation pages, and
# gets inserted into a slightly larger description (in setup.py) for
# the page on Pypi:
""" 
Imageio is a Python library that provides an easy interface to read and
write a wide range of image data, including animated images, volumetric
data, and scientific formats. It is cross-platform, runs on Python 2.7
and 3.4+, and is easy to install.

Main website: http://imageio.github.io
"""

# flake8: noqa

__version__ = "2.5.0"

# Load some bits from core
from .core import FormatManager, RETURN_BYTES

# Instantiate format manager
formats = FormatManager()

# Load the functions
from .core.functions import help
from .core.functions import get_reader, get_writer
from .core.functions import imread, mimread, volread, mvolread
from .core.functions import imwrite, mimwrite, volwrite, mvolwrite

# Load function aliases
from .core.functions import read, save
from .core.functions import imsave, mimsave, volsave, mvolsave

# Load all the plugins
from . import plugins

# expose the show method of formats
show_formats = formats.show

# Clean up some names
del FormatManager
