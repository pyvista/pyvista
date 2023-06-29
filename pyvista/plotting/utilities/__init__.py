"""Plotting utilities."""
# flake8: noqa: F401

from .algorithms import (
    active_scalars_algorithm,
    add_ids_algorithm,
    algorithm_to_mesh_handler,
    cell_data_to_point_data_algorithm,
    crinkle_algorithm,
    decimation_algorithm,
    extract_surface_algorithm,
    outline_algorithm,
    point_data_to_cell_data_algorithm,
    pointset_to_polydata_algorithm,
    set_algorithm_input,
    triangulate_algorithm,
)
from .cubemap import cubemap, cubemap_from_filenames
from .gl_checks import check_depth_peeling, uses_egl
from .regression import (
    compare_images,
    image_from_window,
    remove_alpha,
    run_image_filter,
    wrap_image_array,
)
from .sphinx_gallery import Scraper, _get_sg_image_scraper
from .xvfb import start_xvfb
