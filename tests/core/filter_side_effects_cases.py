"""The filters, arguments and meshes swept by ``test_filter_side_effects.py``.

The sweep finds every public filter and keyword by itself; this module only supplies what
it cannot work out. When it is missing something the sweep raises ``SweepSetupError``,
which is a problem with the test rather than with the filter, and says which table to edit:

* A new filter with required arguments raises on every mesh. Add the arguments to
  ``POSITIONAL_ARGS`` or ``REQUIRED_KWARGS``, or a mesh it accepts to ``MESH_OVERRIDES``.
* A new keyword which is neither ``bool`` nor a ``Literal`` has no values to try. Add
  values to ``KWARG_VALUES``, or the name to ``SKIP_KWARGS`` if it cannot affect the input.

A filter which does modify its input fails with ``Failed`` instead, listing each call.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import pyvista as pv
from pyvista import _vtk

# Filters that open a plot rather than return a mesh.
PLOTTING_FILTERS = frozenset(
    {
        'plot_curvature',
        'plot_normals',
        'plot_over_circular_arc',
        'plot_over_circular_arc_normal',
        'plot_over_line',
        'plot_boundaries',
    }
)

# Filters which raise unconditionally, so a call cannot reach the input.
DEPRECATED_FILTERS = frozenset({'flip_normals'})

# Keywords which modify the input by design, or which cannot affect it.
SKIP_KWARGS = frozenset(
    {'figsize', 'figure', 'fname', 'inplace', 'progress_bar', 'show', 'title', 'ylabel'}
)


#: Keywords the sweep does not vary yet. Move one to ``KWARG_VALUES`` to cover it.
UNVARIED_KWARGS = frozenset(
    {
        'action',
        'algorithm',
        'attribute_error',
        'axis',
        'binary',
        'border_mode',
        'boundary_weight',
        'box_tolerance',
        'capping',
        'categorical',
        'cell_tolerance',
        'clip_tolerance',
        'component_policy',
        'constant_value',
        'contact_mode',
        'convergence',
        'curv_type',
        'dilate_value',
        'dradius',
        'dtype_policy',
        'edge_angle',
        'edge_source',
        'erode_value',
        'extend_border',
        'extrusion',
        'fill_value',
        'i',
        'in_value',
        'include_cells',
        'interpolation',
        'invert',
        'j',
        'k',
        'keep_scalars',
        'kernel_size',
        'length',
        'main_has_priority',
        'margin',
        'max_degree',
        'max_edge_len',
        'max_n_passes',
        'max_n_points',
        'max_n_tris',
        'max_time',
        'max_tri_area',
        'maximum_error',
        'merging_array_name',
        'method',
        'mode',
        'n_cells_per_node',
        'n_iter',
        'n_sides',
        'nbr_sz',
        'normalized_bounds',
        'normals',
        'normals_weight',
        'off_screen',
        'order',
        'orient_faces',
        'out_value',
        'output_scalars_name',
        'pad_size',
        'padding',
        'pass_band',
        'pass_cell_ids',
        'pass_point_ids',
        'point_seeds',
        'preference',
        'preserve_aspect_ratio',
        'radius_factor',
        'rate',
        'reference_image',
        'relaxation_factor',
        'remove',
        'replacement_value',
        'report_body',
        'resample_kwargs',
        'rotation_axis',
        'sample_rate',
        'sample_spacing',
        'scalar_mode',
        'scalars_weight',
        'select_inputs',
        'select_outputs',
        'simplify_output',
        'smoothing_distance',
        'smoothing_iterations',
        'smoothing_relaxation',
        'smoothing_scale',
        'split_angle',
        'std_dev',
        'target_n_points',
        'tcoords',
        'tcoords_weight',
        'tensors',
        'tensors_weight',
        'tetra_per_cell',
        'translation',
        'use_all_points',
        'vectors_weight',
        'width',
    }
)


#: These re-mesh the whole attribute table, carrying the input's active scalars to the output.
ACTIVE_SCALARS_PASSTHROUGH = frozenset({'cells_to_points', 'points_to_cells'})


def _frequency_image(add_arrays):
    """Return an image carrying complex point scalars, as the frequency filters need."""
    return add_arrays(pv.ImageData(dimensions=(4, 5, 6), spacing=(0.5, 0.5, 0.5))).fft()


def _line_mesh(add_arrays):
    """Return a PolyData made of lines, which the contour filters need."""
    return add_arrays(
        pv.MultipleLines(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]))
    )


def _seam_grid(shift=(0.0, 0.0, 0.0)):
    """Return a structured grid whose arrays are constant, so a seam always matches."""
    x, y, z = np.meshgrid(np.arange(4.0), np.arange(3.0), np.arange(5.0), indexing='ij')
    grid = pv.StructuredGrid(x + shift[0], y + shift[1], z + shift[2])
    grid.point_data['constant'] = np.ones(grid.n_points)
    grid.cell_data['constant'] = np.ones(grid.n_cells)
    return grid


def _planar_mesh(add_arrays):
    """Return a mesh in the XY plane, which the evenly spaced streamlines filter needs."""
    return add_arrays(pv.Plane(i_resolution=4, j_resolution=4))


def _triangulated(add_arrays):
    """Return an all-triangle PolyData."""
    return add_arrays(pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate())


#: Filters which only apply to an input the shared mesh kinds do not cover. Each builder
#: takes ``add_arrays``, which puts the current data arrangement on a mesh and returns it.
MESH_OVERRIDES = {
    'concatenate': lambda add_arrays: _seam_grid(),  # noqa: ARG005
    'dash_lines': _line_mesh,
    'high_pass': _frequency_image,
    'low_pass': _frequency_image,
    'rfft': _frequency_image,
    'triangulate_contours': _line_mesh,
    'decimate': _triangulated,
    'decimate_polyline': _line_mesh,
    'streamlines_evenly_spaced_2D': _planar_mesh,
}


def _closed_surface():
    """Return a closed surface enclosing part of every test mesh."""
    return pv.Sphere(radius=0.4, theta_resolution=10, phi_resolution=10)


def _sample_target():
    """Return a volume carrying arrays for the sampling filters to pull from."""
    target = pv.ImageData(dimensions=(4, 4, 4), spacing=(0.5, 0.5, 0.5), origin=(-0.75,) * 3)
    target.point_data['t_point'] = np.arange(target.n_points, dtype=float)
    target.cell_data['t_cell'] = np.arange(target.n_cells, dtype=float)
    return target


def _implicit_plane():
    """Return an implicit function for :meth:`~pyvista.DataObjectFilters.slice_implicit`."""
    plane = _vtk.vtkPlane()
    plane.SetOrigin(0.0, 0.0, 0.0)
    plane.SetNormal(0.0, 0.0, 1.0)
    return plane


class Fresh:
    """An argument value rebuilt for every call."""

    def __init__(self, factory):
        self._factory = factory

    def build(self):
        """Return a new value."""
        return self._factory()


#: Positional arguments for the filters which require them.
POSITIONAL_ARGS = {
    'align': lambda: (_closed_surface().translate((0.01, 0.01, 0.01)),),
    'boolean_difference': lambda: (_closed_surface().translate((0.1, 0.0, 0.0)),),
    'boolean_intersection': lambda: (_closed_surface().translate((0.1, 0.0, 0.0)),),
    'boolean_union': lambda: (_closed_surface().translate((0.1, 0.0, 0.0)),),
    'collision': lambda: (_closed_surface().translate((0.1, 0.0, 0.0)),),
    'concatenate': lambda: (_seam_grid((3.0, 0.0, 0.0)), 0),
    'contour_banded': lambda: (3,),
    'decimate': lambda: (0.5,),
    'decimate_polyline': lambda: (0.5,),
    'decimate_pro': lambda: (0.5,),
    'edge_mask': lambda: (30.0,),
    'extract_subset': lambda: ((0, 2, 0, 2, 0, 2),),
    'extrude': lambda: ((0.0, 0.0, 1.0),),
    'extrude_trim': lambda: ((0.0, 0.0, 1.0), pv.Plane(center=(0, 0, 1), i_size=10, j_size=10)),
    'fill_holes': lambda: (1.0,),
    'generic_filter': lambda: ('triangulate',),
    'geodesic': lambda: (0, 5),
    'geodesic_distance': lambda: (0, 5),
    'high_pass': lambda: (1.0, 1.0, 1.0),
    'image_threshold': lambda: (1.0,),
    'intersection': lambda: (_closed_surface().translate((0.1, 0.0, 0.0)),),
    'low_pass': lambda: (1.0, 1.0, 1.0),
    'multi_ray_trace': lambda: (
        np.array([[0.0, 0.0, -5.0]]),
        np.array([[0.0, 0.0, 1.0]]),
    ),
    'ray_trace': lambda: ((0.0, 0.0, -5.0), (0.0, 0.0, 5.0)),
    'subdivide': lambda: (1,),
    'clip_slab': lambda: (0.4,),
    'clip_surface': lambda: (_closed_surface(),),
    'compute_implicit_distance': lambda: (_closed_surface(),),
    'extract_cells': lambda: ([0, 1, 2],),
    'extract_cells_by_type': lambda: (pv.CellType.TRIANGLE,),
    'extract_points': lambda: ([0, 1, 2],),
    'flip_normal': lambda: ((1.0, 0.0, 0.0),),
    'interpolate': lambda: (_sample_target(),),
    'partition': lambda: (2,),
    'reflect': lambda: ((1.0, 0.0, 0.0),),
    'remove_cells': lambda: ([0, 1],),
    'remove_points': lambda: ([0, 1],),
    'rotate': lambda: (np.eye(3),),
    'rotate_vector': lambda: ((1.0, 1.0, 1.0), 30.0),
    'rotate_x': lambda: (30.0,),
    'rotate_y': lambda: (30.0,),
    'rotate_z': lambda: (30.0,),
    'sample': lambda: (_sample_target(),),
    'sample_over_circular_arc': lambda: (),
    'sample_over_line': lambda: ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
    'sample_over_multiple_lines': lambda: (
        np.array([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    ),
    'scale': lambda: (2.0,),
    'select_enclosed_points': lambda: (_closed_surface(),),
    'select_interior_points': lambda: (_closed_surface(),),
    'slice_along_line': lambda: (pv.Line((-1, -1, -1), (1, 1, 1), resolution=4),),
    'slice_implicit': lambda: (_implicit_plane(),),
    'streamlines_from_source': lambda: (pv.PointSet(np.array([[0.0, 0.0, 0.0]])),),
    'transform': lambda: (np.eye(4),),
    'translate': lambda: ((1.0, 2.0, 3.0),),
}

#: Keyword arguments required alongside the positional ones.
REQUIRED_KWARGS = {
    'extract_values': dict(values=0.0),
    'sample_over_circular_arc': dict(pointa=(-1, 0, 0), pointb=(1, 0, 0), center=(0, 0, 0)),
    'sample_over_circular_arc_normal': dict(center=(0, 0, 0)),
    'select_values': dict(values=0.0),
    'slice_index': dict(i=0),
    'validate_mesh': dict(action='warn'),
}

#: Values to try for keywords whose type is neither ``bool`` nor a ``Literal``.
KWARG_VALUES: dict[str, list[Any]] = {
    'alpha': [0.5],
    'angle': [90.0],
    'axis_0_direction': ['-x'],
    'axis_1_direction': [(0.0, 1.0, 0.0)],
    'axis_2_direction': [(0.0, 0.0, -1.0)],
    'background_value': [2],
    'bounds': [(-0.2, 0.2, -0.2, 0.2, -0.2, 0.2)],
    'bounds_size': [2.0],
    'cell_ids': [0],
    'cell_length_percentile': [0.5],
    'cell_length_sample_size': [100],
    'cell_types': [pv.CellType.TRIANGLE],
    'center': [(0.0, 0.0, 0.0)],
    'closed_loop_maximum_distance': [0.2],
    'closest_point': [(0.0, 0.0, 0.0)],
    'colors': ['glasbey'],
    'component': [1],
    'dimensions': [(6, 6, 6)],
    'divergence': ['div'],
    'exclude_fields': ['nonmanifold_edges'],
    'extent': [(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)],
    'factor': [0.2],
    'feature_angle': [10.0],
    'foreground_value': [9],
    'frame_width': [0.2],
    'geom': [Fresh(pv.Cube)],
    'gradient': ['grad'],
    'grid': [Fresh(lambda: pv.Sphere(radius=0.3, theta_resolution=6, phi_resolution=6))],
    'high_point': [(0.0, 0.0, 1.0)],
    'ind': [[0, 1]],
    'indices': [(0,)],
    'initial_step_length': [0.2],
    'isosurfaces': [3, [0.5, 1.5]],
    'locator_tolerance': [1e-3],
    'loop_angle': [30.0],
    'low_point': [(0.0, 0.0, -1.0)],
    'max_error': [1e-5],
    'max_iterations': [10],
    'max_landmarks': [20],
    'max_length': [1.0],
    'max_mean_distance': [1e-3],
    'max_n_subdivide': [2],
    'max_step_length': [0.5],
    'max_steps': [10],
    'min_step_length': [0.05],
    'minimum_number_of_loop_points': [2],
    'n': [3],
    'n_partitions': [2],
    'n_points': [10],
    'name': ['custom_name'],
    'nonlinear_subdivision': [2],
    'normal': [(1.0, 1.0, 0.0)],
    'null_value': [-1.0],
    'offset': [3.0],
    'orient': ['p_vectors', 'c_vectors'],
    'pattern': [(0.2, 0.1)],
    'origin': [(0.0, 0.0, 0.0)],
    'output_scalars': ['renamed'],
    'percent': [0.3],
    'planarity_tolerance': [1e-3],
    'plane': [Fresh(lambda: pv.Plane(i_size=5, j_size=5))],
    'point': [(1.0, 0.0, 0.0)],
    'point_ids': [0],
    'point_u': [(1.0, 0.0, 0.0)],
    'point_v': [(0.0, 1.0, 0.0)],
    'pointa': [(-1.0, 0.0, 0.0)],
    'pointb': [(1.0, 0.0, 0.0)],
    'points': [np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])],
    'polar': [(1.0, 0.0, 0.0)],
    'quality_measure': ['area', 'all_valid', ['area', 'aspect_ratio']],
    'radius': [0.2],
    'ranges': [(0.0, 3.0)],
    'reference_volume': [Fresh(lambda: pv.ImageData(dimensions=(5, 5, 5)))],
    'region_ids': [0],
    'resolution': [5],
    'rng': [(0.0, 1.0)],
    'rotation': [np.eye(3)],
    'rotation_scale': [2.0],
    'rounding_func': [np.ceil],
    'scalar_range': [(0.0, 3.0), 'p_scalars'],
    'scalars': ['p_scalars', 'p_other', 'p_vectors', 'c_scalars', 'c_other', 'solo'],
    'scale': ['p_scalars', 'p_other', 'c_scalars'],
    'separating_distance': [5.0],
    'separating_distance_ratio': [0.3],
    'sharpness': [3.0],
    'shrink_factor': [0.5],
    'size_tolerance': [1e-3],
    'source': [Fresh(lambda: pv.PointSet(np.array([[0.0, 0.0, 0.0]])))],
    'source_center': [(0.0, 0.0, 0.0)],
    'source_radius': [0.5],
    'spacing': [0.3],
    'start_position': [(0.0, 0.0, 0.0)],
    'step_length': [0.2],
    'style': [':', '-.'],
    'surface': [Fresh(_closed_surface)],
    'target': [Fresh(_sample_target)],
    'target_reduction': [0.2],
    'terminal_speed': [1e-9],
    'thickness': [0.4],
    'tol': [1e-2],
    'tolerance': [1e-2],
    'trans': [np.eye(4)],
    'validation_fields': ['nonmanifold_edges'],
    'value': [1.0, (0.5, 2.0)],
    'values': [0.0, {'a': 0.0}],
    'variable_input': [0],
    'vector': [(1.0, 1.0, 1.0)],
    'vectors': ['p_vectors', 'c_vectors'],
    'vorticity': ['vort'],
    'x': [0.0],
    'xyz': [2.0],
    'y': [0.0],
    'z': [0.0],
}
