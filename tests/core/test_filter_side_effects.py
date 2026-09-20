"""Check that filters never modify their input, and never read its active arrays.

A filter returns a new dataset, so calling one must leave its input exactly as it was.
The usual way to break that is to activate an array on the input before handing it to
VTK, which nothing notices until someone reads the input afterwards.

Each parametrized test takes one filter and calls it on every mesh type in ``MESH_KINDS``
under several arrangements of data arrays (see ``DATA_MODES``), and the first also tries
every keyword the filter accepts (see ``_call_variants``) for the arrangements in
``KEYWORD_DATA_MODES``. Calls which do not apply to a mesh raise and do not count as runs,
so a filter runs far fewer times than the loops suggest.

A failure lists every call which broke the property, what changed, and the expression
which rebuilds that input, so one failing call can be reproduced on its own::

    _make_mesh('image', 'both').threshold(scalars='c_scalars')

The tables below the mesh builders supply the arguments each filter needs. A filter whose
arguments are missing never runs, and the ``never ran`` assertion reports that.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import re
from typing import Any
import warnings

import numpy as np
import pytest

import pyvista as pv
from pyvista import _vtk
from pyvista.core.filters.composite import CompositeFilters
from pyvista.core.filters.data_object import DataObjectFilters
from pyvista.core.filters.data_set import DataSetFilters
from pyvista.core.filters.image_data import ImageDataFilters
from pyvista.core.filters.poly_data import PolyDataFilters
from pyvista.core.filters.rectilinear_grid import RectilinearGridFilters
from pyvista.core.filters.structured_grid import StructuredGridFilters
from pyvista.core.filters.unstructured_grid import UnstructuredGridFilters
from pyvista.core.utilities.arrays import set_default_active_scalars

# Filters that open a plot rather than return a mesh.
_PLOTTING_FILTERS = frozenset(
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
_DEPRECATED_FILTERS = frozenset({'flip_normals'})

# Keywords which modify the input by design, or which cannot affect it.
_SKIP_KWARGS = frozenset(
    {'figsize', 'figure', 'fname', 'inplace', 'progress_bar', 'show', 'title', 'ylabel'}
)


def _crashes_vtk(kind, name, keyword):
    """Return whether a call segfaults VTK, for reasons unrelated to side effects."""
    # vtkCellLocatorInterpolatedVelocityField dereferences the cells a PointSet lacks
    return (
        kind == 'pointset' and name == 'streamlines_from_source' and keyword == 'interpolator_type'
    )


_LITERAL_PATTERN = re.compile(r'Literal\[([^]]*)]')


def _mesh_arrays(mesh, mode):
    """Add ``mode``'s data arrays to ``mesh`` and return it."""
    rng = np.random.default_rng(0)
    if mode in ('single_point', 'single_cell', 'single_vector'):
        mesh.clear_data()
        cell_only = mode == 'single_cell'
        attributes = mesh.cell_data if cell_only else mesh.point_data
        n = mesh.n_cells if cell_only else mesh.n_points
        solo = rng.random((n, 3)) if mode == 'single_vector' else np.arange(n, dtype=float) % 5
        attributes['solo'] = solo
        for association in (mesh.point_data, mesh.cell_data):
            association.active_scalars_name = None
            association.active_vectors_name = None
            association.active_normals_name = None
        return mesh
    if mode in ('point', 'both'):
        n = mesh.n_points
        mesh.point_data['p_scalars'] = np.arange(n, dtype=float) % 7
        mesh.point_data['p_other'] = np.linspace(-1.0, 1.0, n)
        mesh.point_data['p_vectors'] = rng.random((n, 3))
        mesh.point_data['p_labels'] = (np.arange(n) % 3).astype(np.int32) * 10
        mesh.point_data['p_bool'] = np.arange(n) % 2 == 0
        mesh.point_data.active_scalars_name = 'p_scalars'
        mesh.point_data.active_vectors_name = 'p_vectors'
    if mode in ('cell', 'both'):
        n = mesh.n_cells
        mesh.cell_data['c_scalars'] = np.arange(n, dtype=float) % 5
        mesh.cell_data['c_other'] = np.linspace(-2.0, 2.0, n)
        mesh.cell_data['c_vectors'] = rng.random((n, 3))
        mesh.cell_data['c_labels'] = (np.arange(n) % 3).astype(np.int32) * 10
        mesh.cell_data['c_bool'] = np.arange(n) % 2 == 0
        mesh.cell_data.active_scalars_name = 'c_scalars'
        mesh.cell_data.active_vectors_name = 'c_vectors'
    mesh.field_data['f_data'] = np.array([1.0, 2.0, 3.0])
    return mesh


def _make_mesh(kind, mode):
    """Build a test mesh of ``kind`` carrying ``mode``'s data arrays."""
    if kind == 'poly':
        mesh: pv.DataSet = pv.Sphere(theta_resolution=8, phi_resolution=8)
    elif kind == 'unstructured':
        mesh = pv.ImageData(dimensions=(4, 4, 4)).cast_to_unstructured_grid()
    elif kind == 'image':
        mesh = pv.ImageData(dimensions=(4, 5, 6), spacing=(0.5, 0.5, 0.5), origin=(-1, -1, -1))
    elif kind == 'rectilinear':
        mesh = pv.RectilinearGrid(
            np.array([0.0, 0.5, 1.5, 3.0]),
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.7, 2.0, 2.5]),
        )
    elif kind == 'structured':
        x, y, z = np.meshgrid(np.arange(4.0), np.arange(3.0), np.arange(5.0), indexing='ij')
        mesh = pv.StructuredGrid(x, y, z)
    elif kind == 'multiblock':
        return pv.MultiBlock(
            {
                'poly': _make_mesh('poly', mode),
                'unstructured': _make_mesh('unstructured', mode),
            }
        )
    else:
        mesh = pv.PointSet(pv.Sphere(theta_resolution=8, phi_resolution=8).points)
        # PointSet has no cells, so cell arrays cannot exist
        mode = {'cell': 'point', 'both': 'point', 'single_cell': 'single_point'}.get(mode, mode)
    return _mesh_arrays(mesh, mode)


def _array_summary(array):
    """Summarize a VTK array as ``name:type[components]xtuples#hash``."""
    contents = np.ascontiguousarray(pv.convert_array(array)).tobytes()
    return (
        f'{array.GetName()}:{array.GetDataTypeAsString()}'
        f'[{array.GetNumberOfComponents()}]x{array.GetNumberOfTuples()}'
        f'#{hashlib.sha256(contents).hexdigest()[:8]}'
    )


def _cells_summary(cell_array):
    """Summarize a :vtk:`vtkCellArray`'s offsets and connectivity."""
    return (
        _array_summary(cell_array.GetOffsetsArray()),
        _array_summary(cell_array.GetConnectivityArray()),
    )


def _attribute_entries(attributes, label):
    """Return the arrays and active-array slots of a :vtk:`vtkDataSetAttributes`."""
    entries = {
        f'{label} arrays': tuple(
            _array_summary(attributes.GetAbstractArray(index))
            for index in range(attributes.GetNumberOfArrays())
        )
    }
    for index in range(_vtk.vtkDataSetAttributes.NUM_ATTRIBUTES):
        attribute = attributes.GetAbstractAttribute(index)
        slot = _vtk.vtkDataSetAttributes.GetAttributeTypeAsString(index).lower()
        entries[f'{label} active {slot}'] = None if attribute is None else attribute.GetName()
    return entries


def _fingerprint(mesh, prefix=''):
    """Return a flat dict, keyed in English, of everything a filter could modify."""
    if isinstance(mesh, pv.MultiBlock):
        entries = {f'{prefix}block names': tuple(mesh.keys())}
        for index, block in enumerate(mesh):
            entries.update(_fingerprint(block, f'{prefix}block {index} '))
        return entries

    field_data = mesh.GetFieldData()
    entries: dict[str, Any] = {
        f'{prefix}type': type(mesh).__name__,
        f'{prefix}n_points': mesh.n_points,
        f'{prefix}n_cells': mesh.n_cells,
        f'{prefix}bounds': tuple(mesh.bounds),
        **_attribute_entries(mesh.GetPointData(), f'{prefix}point data'),
        **_attribute_entries(mesh.GetCellData(), f'{prefix}cell data'),
        f'{prefix}field data arrays': tuple(
            _array_summary(field_data.GetAbstractArray(index))
            for index in range(field_data.GetNumberOfArrays())
        ),
    }
    if isinstance(mesh, _vtk.vtkImageData):
        entries[f'{prefix}extent'] = mesh.GetExtent()
    if isinstance(mesh, _vtk.vtkPointSet):
        points = mesh.GetPoints()
        entries[f'{prefix}points'] = None if points is None else _array_summary(points.GetData())
    if isinstance(mesh, _vtk.vtkPolyData):
        for name in ('Verts', 'Lines', 'Polys', 'Strips'):
            entries[f'{prefix}{name.lower()}'] = _cells_summary(getattr(mesh, f'Get{name}')())
    if isinstance(mesh, _vtk.vtkUnstructuredGrid):
        entries[f'{prefix}cells'] = _cells_summary(mesh.GetCells())
    if isinstance(mesh, _vtk.vtkRectilinearGrid):
        for axis in ('X', 'Y', 'Z'):
            coordinates = getattr(mesh, f'Get{axis}Coordinates')()
            entries[f'{prefix}{axis.lower()} coordinates'] = _array_summary(coordinates)
    # Names of the arrays which are read back as bool or complex live on the dataset
    for label, names in (
        ('bool', mesh._association_bitarray_names),
        ('complex', mesh._association_complex_names),
    ):
        entries[f'{prefix}{label} array names'] = tuple(
            sorted((key, tuple(sorted(value))) for key, value in names.items() if value)
        )
    return entries


def _short(value, limit=150):
    """Return a one-line ``repr`` of ``value``, shortened to ``limit`` characters."""
    if isinstance(value, pv.DataObject) or type(value).__name__.startswith('vtk'):
        return f'<{type(value).__name__}>'
    text = repr(value).replace('\n', ' ')
    return text if len(text) <= limit else f'{text[: limit - 3]}...'


def _describe_change(key, before, after):
    """Return a line naming what changed for one fingerprint entry."""
    summaries = (*before, *after) if isinstance(before, tuple) else ()
    if isinstance(after, tuple) and summaries and all(isinstance(s, str) for s in summaries):
        gone = [item for item in before if item not in after]
        arrived = [item for item in after if item not in before]
        if gone or arrived:
            entries = [f'-{_short(item, 60)}' for item in gone]
            entries += [f'+{_short(item, 60)}' for item in arrived]
            return f'{key}: {", ".join(entries)}'
        return f'{key}: reordered'
    return f'{key}: {_short(before)} -> {_short(after)}'


def _changes(before, after):
    """Return one line for each fingerprint entry which differs."""
    lines = [
        _describe_change(key, before[key], after[key])
        if key in after
        else f'{key}: {_short(before[key])} -> <entry gone>'
        for key in before
        if key not in after or before[key] != after[key]
    ]
    lines += [f'{key}: <no entry> -> {_short(after[key])}' for key in after if key not in before]
    return lines


def _call_expression(kind, mode, name, args, kwargs):
    """Return the expression which rebuilds an input and makes one call on it."""
    mesh = (
        f'_MESH_OVERRIDES[{name!r}]({mode!r})'
        if name in _MESH_OVERRIDES
        else f'_make_mesh({kind!r}, {mode!r})'
    )
    shown = [_short(arg, 40) for arg in args]
    shown += [f'{key}={_short(value, 40)}' for key, value in kwargs.items()]
    return f'{mesh}.{name}({", ".join(shown)})'


def _report(kind, mode, name, args, kwargs, changes):
    """Return a readable block naming one call and what it changed."""
    return '\n'.join(
        [
            f'  {kind} mesh, {DATA_MODES[mode]}',
            f'    {_call_expression(kind, mode, name, args, kwargs)}',
            *(f'      {change}' for change in changes),
        ]
    )


def _frequency_image(mode):
    """Return an image carrying complex point scalars, as the frequency filters need."""
    return _make_mesh('image', mode).fft()


def _line_mesh(mode):
    """Return a PolyData made of lines, which the contour filters need."""
    return _mesh_arrays(
        pv.MultipleLines(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])), mode
    )


def _seam_grid(shift=(0.0, 0.0, 0.0)):
    """Return a structured grid whose arrays are constant, so a seam always matches."""
    x, y, z = np.meshgrid(np.arange(4.0), np.arange(3.0), np.arange(5.0), indexing='ij')
    grid = pv.StructuredGrid(x + shift[0], y + shift[1], z + shift[2])
    grid.point_data['constant'] = np.ones(grid.n_points)
    grid.cell_data['constant'] = np.ones(grid.n_cells)
    return grid


def _planar_mesh(mode):
    """Return a mesh in the XY plane, which the evenly spaced streamlines filter needs."""
    return _mesh_arrays(pv.Plane(i_resolution=4, j_resolution=4), mode)


def _triangulated(mode):
    """Return an all-triangle PolyData."""
    return _mesh_arrays(pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate(), mode)


#: Filters which only apply to an input the shared mesh kinds do not cover.
_MESH_OVERRIDES = {
    'concatenate': lambda mode: _seam_grid(),  # noqa: ARG005
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


class _Fresh:
    """An argument value rebuilt for every call."""

    def __init__(self, factory):
        self._factory = factory

    def build(self):
        """Return a new value."""
        return self._factory()


#: Positional arguments for the filters which require them.
_POSITIONAL_ARGS = {
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
_REQUIRED_KWARGS = {
    'extract_values': dict(values=0.0),
    'sample_over_circular_arc': dict(pointa=(-1, 0, 0), pointb=(1, 0, 0), center=(0, 0, 0)),
    'sample_over_circular_arc_normal': dict(center=(0, 0, 0)),
    'select_values': dict(values=0.0),
    'slice_index': dict(i=0),
    'validate_mesh': dict(action='warn'),
}

#: Values to try for keywords whose type is neither ``bool`` nor a ``Literal``.
_KWARG_VALUES: dict[str, list[Any]] = {
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
    'geom': [_Fresh(pv.Cube)],
    'gradient': ['grad'],
    'grid': [_Fresh(lambda: pv.Sphere(radius=0.3, theta_resolution=6, phi_resolution=6))],
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
    'origin': [(0.0, 0.0, 0.0)],
    'output_scalars': ['renamed'],
    'percent': [0.3],
    'planarity_tolerance': [1e-3],
    'plane': [_Fresh(lambda: pv.Plane(i_size=5, j_size=5))],
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
    'reference_volume': [_Fresh(lambda: pv.ImageData(dimensions=(5, 5, 5)))],
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
    'source': [_Fresh(lambda: pv.PointSet(np.array([[0.0, 0.0, 0.0]])))],
    'source_center': [(0.0, 0.0, 0.0)],
    'source_radius': [0.5],
    'spacing': [0.3],
    'start_position': [(0.0, 0.0, 0.0)],
    'step_length': [0.2],
    'surface': [_Fresh(_closed_surface)],
    'target': [_Fresh(_sample_target)],
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


def _literal_options(annotation):
    """Return the options of a ``Literal`` annotation, or ``None``."""
    match = _LITERAL_PATTERN.search(str(annotation))
    return None if match is None else ast.literal_eval(f'[{match.group(1)}]')


def _kwarg_variants(parameter):
    """Yield values to try for a single keyword parameter."""
    if parameter.name in _SKIP_KWARGS:
        return
    if parameter.name in _KWARG_VALUES:
        yield from _KWARG_VALUES[parameter.name]
        return
    if isinstance(parameter.default, bool):
        yield not parameter.default
        return
    options = _literal_options(parameter.annotation)
    if options is not None:
        yield from (option for option in options if option != parameter.default)


def _call_variants(func):
    """Yield ``(keyword, kwargs)`` pairs exercising each keyword of ``func`` in turn."""
    parameters = list(inspect.signature(func).parameters.values())[1:]
    # Filters which can modify the input must be told not to
    base = {'inplace': False} if any(p.name == 'inplace' for p in parameters) else {}
    yield None, base
    for parameter in parameters:
        if parameter.default is inspect.Parameter.empty:
            continue
        for value in _kwarg_variants(parameter):
            yield parameter.name, {**base, parameter.name: value}


_FILTER_CLASSES = (
    CompositeFilters,
    DataObjectFilters,
    DataSetFilters,
    ImageDataFilters,
    PolyDataFilters,
    RectilinearGridFilters,
    StructuredGridFilters,
    UnstructuredGridFilters,
)


def _filters():
    """Return every public filter of every filter class, keyed by class and name."""
    found = {}
    for cls in _FILTER_CLASSES:
        for name in sorted(vars(cls)):
            if not name.startswith('_') and name not in (_PLOTTING_FILTERS | _DEPRECATED_FILTERS):
                found[f'{cls.__name__}.{name}'] = getattr(cls, name)
    return found


FILTERS = _filters()

MESH_KINDS = [
    'poly',
    'unstructured',
    'image',
    'rectilinear',
    'structured',
    'pointset',
    'multiblock',
]
#: What each arrangement of data arrays puts on a mesh, and how a failure describes it.
DATA_MODES = {
    'point': 'five point arrays, point scalars and vectors active',
    'cell': 'five cell arrays, cell scalars and vectors active',
    'both': 'five point and five cell arrays, all four active',
    'single_point': 'one point array, nothing active',
    'single_cell': 'one cell array, nothing active',
    'single_vector': 'one three-component point array, nothing active',
}

#: The keywords are swept over these modes only, to keep the sweep's runtime in hand.
KEYWORD_DATA_MODES = ['both', 'single_point', 'single_vector']

#: Modes whose mesh carries one array and no active arrays, so a default has to be resolved.
UNSET_DATA_MODES = ['single_point', 'single_cell', 'single_vector']


def _output_digest(result):
    """Return a comparable digest of whatever a filter returned."""
    if isinstance(result, pv.MultiBlock):
        return {f'block {index}': _output_digest(block) for index, block in enumerate(result)}
    if isinstance(result, pv.DataSet):
        return _fingerprint(result)
    if isinstance(result, tuple):
        return {f'return value {index}': _output_digest(item) for index, item in enumerate(result)}
    return {'returned': repr(result)}


def _flatten(digest, prefix=''):
    """Return a nested output digest as one flat dict."""
    flat = {}
    for key, value in digest.items():
        label = f'{prefix}{key} '
        if isinstance(value, dict):
            flat.update(_flatten(value, label))
        else:
            flat[f'{prefix}{key}'] = value
    return flat


def _output_or_error(mesh, name, args, kwargs):
    """Return the flat digest of a filter's output, or the error it raised."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            return _flatten(_output_digest(getattr(mesh, name)(*args, **kwargs)))
        except Exception as error:  # noqa: BLE001  - the filter does not apply to this mesh
            return {'raised': type(error).__name__}


def _run(mesh, name, args, kwargs):
    """Call a filter, returning whether it ran on this mesh."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            getattr(mesh, name)(*args, **kwargs)
        except Exception:  # noqa: BLE001  - the filter does not apply to this mesh
            return False
    return True


@pytest.fixture(autouse=True)
def _quiet_vtk():
    """Silence the VTK errors raised by filters which do not apply to a mesh."""
    pv.vtk_verbosity('off')
    yield
    pv.vtk_verbosity('info')


def _override_mesh(name, mode, default):
    """Return the special input a filter needs for this mode, or ``None`` if it has none."""
    if name not in _MESH_OVERRIDES:
        return default
    try:
        return _MESH_OVERRIDES[name](mode)
    except Exception:  # noqa: BLE001  - this mode cannot build that input
        return None


def _call_arguments(name, keyword_variant=()):
    """Return the positional and keyword arguments for one call of ``name``."""
    args = _POSITIONAL_ARGS[name]() if name in _POSITIONAL_ARGS else ()
    merged = {**_REQUIRED_KWARGS.get(name, {}), **dict(keyword_variant)}
    kwargs = {
        key: value.build() if isinstance(value, _Fresh) else value for key, value in merged.items()
    }
    return args, kwargs


def _fail(key, problem, reports, ran):
    """Fail with one readable block per call which broke the property."""
    pytest.fail(
        f'{key} {problem} in {len(reports)} of {ran} calls:\n\n' + '\n\n'.join(reports),
        pytrace=False,
    )


@pytest.mark.parametrize('key', list(FILTERS))
def test_filter_does_not_modify_input(key):
    """A filter leaves its input's arrays, active arrays, points and cells alone, even on error."""
    func = FILTERS[key]
    name = func.__name__
    reports = []
    ran = 0
    for mode in DATA_MODES:
        for kind in MESH_KINDS:
            template = _override_mesh(name, mode, _make_mesh(kind, mode))
            if template is None or not hasattr(template, name):
                continue
            for keyword, keyword_variant in _call_variants(func):
                if keyword is not None and mode not in KEYWORD_DATA_MODES:
                    continue
                if _crashes_vtk(kind, name, keyword):
                    continue
                mesh = template.copy()
                args, kwargs = _call_arguments(name, keyword_variant)
                before = _fingerprint(mesh)
                if _run(mesh, name, args, kwargs):
                    ran += 1
                changes = _changes(before, _fingerprint(mesh))
                if changes:  # pragma: no cover -- failure path
                    reports.append(_report(kind, mode, name, args, kwargs, changes))
    assert ran, f'{key} never ran; the test meshes or arguments no longer apply'
    if reports:  # pragma: no cover -- failure path
        _fail(key, 'modified its input', reports, ran)


#: These re-mesh the whole attribute table, carrying the input's active scalars to the output.
_ACTIVE_SCALARS_PASSTHROUGH = frozenset({'cells_to_points', 'points_to_cells'})

_DEFAULT_SCALARS_FILTERS = [
    key
    for key, func in FILTERS.items()
    if 'scalars' in inspect.signature(func).parameters
    and func.__name__ not in _ACTIVE_SCALARS_PASSTHROUGH
]


def _activated(mesh):
    """Return a copy of ``mesh`` with its default scalars made active."""
    activated = mesh.copy()
    set_default_active_scalars(activated)
    return activated


@pytest.mark.parametrize('key', _DEFAULT_SCALARS_FILTERS)
def test_filter_output_does_not_depend_on_active_scalars(key):
    """A filter returns the same output whether or not its default array was already active."""
    name = FILTERS[key].__name__
    reports = []
    ran = 0
    for mode in UNSET_DATA_MODES:
        for kind in MESH_KINDS:
            template = _override_mesh(name, mode, _make_mesh(kind, mode))
            if template is None or isinstance(template, pv.MultiBlock):
                continue
            if not hasattr(template, name):
                continue
            activated = _activated(template)
            args, kwargs = _call_arguments(name)
            as_is = _output_or_error(template.copy(), name, args, kwargs)
            preactivated = _output_or_error(activated, name, args, kwargs)
            if 'raised' in as_is and as_is == preactivated:
                continue  # the filter does not apply to this mesh
            ran += 1
            changes = _changes(as_is, preactivated)
            if changes:  # pragma: no cover -- failure path
                reports.append(_report(kind, mode, name, args, kwargs, changes))
    assert ran, f'{key} never ran; the test meshes or arguments no longer apply'
    if reports:  # pragma: no cover -- failure path
        _fail(key, 'returns a different output once its default array is active', reports, ran)


def test_failure_names_the_call_and_every_entry_which_changed():
    """A failure report names the call which broke the property and what it did to the mesh."""
    mesh = _make_mesh('poly', 'single_point')
    before = _fingerprint(mesh)
    mesh.point_data['extra'] = np.arange(mesh.n_points, dtype=float)
    mesh.set_active_scalars('extra')
    changes = _changes(before, _fingerprint(mesh))
    report = _report('poly', 'single_point', 'sample', (pv.Sphere(),), {'tolerance': 0.5}, changes)

    with pytest.raises(pytest.fail.Exception) as excinfo:
        _fail('DataSetFilters.sample', 'modified its input', [report], 4)

    message = str(excinfo.value)
    assert message.startswith('DataSetFilters.sample modified its input in 1 of 4 calls:')
    assert f'  poly mesh, {DATA_MODES["single_point"]}' in message
    assert "    _make_mesh('poly', 'single_point').sample(<PolyData>, tolerance=0.5)" in message
    assert "      point data arrays: +'extra:" in message
    assert "      point data active scalars: None -> 'extra'" in message


def test_failure_names_reordered_arrays_as_reordered():
    """Arrays which only changed order are reported as reordered, not as added and removed."""
    change = _describe_change('cell data arrays', ('a', 'b'), ('b', 'a'))
    assert change == 'cell data arrays: reordered'
