"""Utilities routines."""

from .errors import (Observer, Report, assert_empty_kwargs,
                     send_errors_to_logging, set_error_output_file)
from .features import *
from .fileio import *
from .geometric_objects import *
from .parametric_objects import *
from .sphinx_gallery import Scraper, _get_sg_image_scraper
from .helpers import *
