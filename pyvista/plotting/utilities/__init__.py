"""Plotting utilities."""
# flake8: noqa: F401

from . import algorithms
from .algorithms import algorithm_to_mesh_handler, set_algorithm_input
from .cubemap import cubemap, cubemap_from_filenames
from .gl_checks import check_depth_peeling
from .regression import (
    compare_images,
    image_from_window,
    remove_alpha,
    run_image_filter,
    wrap_image_array,
)
from .sphinx_gallery import Scraper, _get_sg_image_scraper
from .xvfb import start_xvfb
