"""Plotting utilities."""

# ruff: noqa: F401
from __future__ import annotations

from .algorithms import active_scalars_algorithm
from .algorithms import add_ids_algorithm
from .algorithms import algorithm_to_mesh_handler
from .algorithms import cell_data_to_point_data_algorithm
from .algorithms import crinkle_algorithm
from .algorithms import decimation_algorithm
from .algorithms import extract_surface_algorithm
from .algorithms import outline_algorithm
from .algorithms import point_data_to_cell_data_algorithm
from .algorithms import pointset_to_polydata_algorithm
from .algorithms import set_algorithm_input
from .algorithms import triangulate_algorithm
from .cubemap import cubemap
from .cubemap import cubemap_from_filenames
from .gl_checks import check_depth_peeling
from .gl_checks import uses_egl
from .regression import compare_images
from .regression import image_from_window
from .regression import remove_alpha
from .regression import run_image_filter
from .regression import wrap_image_array
from .sphinx_gallery import Scraper
from .sphinx_gallery import _get_sg_image_scraper
from .xvfb import start_xvfb
