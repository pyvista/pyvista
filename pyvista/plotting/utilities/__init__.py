"""Plotting utilities."""

from __future__ import annotations

from .algorithms import active_scalars_algorithm as active_scalars_algorithm
from .algorithms import add_ids_algorithm as add_ids_algorithm
from .algorithms import algorithm_to_mesh_handler as algorithm_to_mesh_handler
from .algorithms import cell_data_to_point_data_algorithm as cell_data_to_point_data_algorithm
from .algorithms import crinkle_algorithm as crinkle_algorithm
from .algorithms import decimation_algorithm as decimation_algorithm
from .algorithms import extract_surface_algorithm as extract_surface_algorithm
from .algorithms import outline_algorithm as outline_algorithm
from .algorithms import point_data_to_cell_data_algorithm as point_data_to_cell_data_algorithm
from .algorithms import pointset_to_polydata_algorithm as pointset_to_polydata_algorithm
from .algorithms import set_algorithm_input as set_algorithm_input
from .algorithms import triangulate_algorithm as triangulate_algorithm
from .cubemap import cubemap as cubemap
from .cubemap import cubemap_from_filenames as cubemap_from_filenames
from .gl_checks import check_depth_peeling as check_depth_peeling
from .gl_checks import uses_egl as uses_egl
from .regression import compare_images as compare_images
from .regression import image_from_window as image_from_window
from .regression import remove_alpha as remove_alpha
from .regression import run_image_filter as run_image_filter
from .regression import wrap_image_array as wrap_image_array
from .sphinx_gallery import Scraper as Scraper
from .sphinx_gallery import _get_sg_image_scraper as _get_sg_image_scraper
from .xvfb import start_xvfb as start_xvfb
