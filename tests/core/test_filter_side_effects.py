"""Check that dataset and dataobject filters never modify their input."""

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
from pyvista.core.filters.data_object import DataObjectFilters
from pyvista.core.filters.data_set import DataSetFilters

# Filters that open a plot rather than return a mesh.
_PLOTTING_FILTERS = frozenset(
    {'plot_over_circular_arc', 'plot_over_circular_arc_normal', 'plot_over_line'}
)

# Keywords which modify the input by design, or which cannot affect it.
_SKIP_KWARGS = frozenset(
    {'figsize', 'figure', 'fname', 'inplace', 'progress_bar', 'show', 'title', 'ylabel'}
)

# Combinations which crash VTK, unrelated to side effects.
_CRASHES = frozenset({('pointset', 'streamlines_from_source', 'interpolator_type')})

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
    else:
        mesh = pv.PointSet(pv.Sphere(theta_resolution=8, phi_resolution=8).points)
        # PointSet has no cells, so cell arrays cannot exist
        mode = {'cell': 'point', 'both': 'point', 'single_cell': 'single_point'}.get(mode, mode)
    return _mesh_arrays(mesh, mode)


def _array_digest(array):
    """Digest a VTK array's name, type, shape and contents."""
    return (
        array.GetName(),
        array.GetDataTypeAsString(),
        array.GetNumberOfComponents(),
        array.GetNumberOfTuples(),
        hashlib.sha256(np.ascontiguousarray(pv.convert_array(array)).tobytes()).hexdigest(),
    )


def _attributes_digest(attributes):
    """Digest every array of a :vtk:`vtkDataSetAttributes` and its active-attribute slots."""
    arrays = tuple(
        _array_digest(attributes.GetAbstractArray(i))
        for i in range(attributes.GetNumberOfArrays())
    )
    active = []
    for i in range(_vtk.vtkDataSetAttributes.NUM_ATTRIBUTES):
        attribute = attributes.GetAbstractAttribute(i)
        active.append(None if attribute is None else attribute.GetName())
    return arrays, tuple(active)


def _cells_digest(cell_array):
    """Digest a :vtk:`vtkCellArray`'s offsets and connectivity."""
    if cell_array is None:
        return None
    return (
        _array_digest(cell_array.GetOffsetsArray()),
        _array_digest(cell_array.GetConnectivityArray()),
    )


def fingerprint(mesh):
    """Return a hashable digest of everything a filter could modify on ``mesh``."""
    parts: list[Any] = [
        type(mesh).__name__,
        mesh.n_points,
        mesh.n_cells,
        _attributes_digest(mesh.GetPointData()),
        _attributes_digest(mesh.GetCellData()),
        tuple(
            _array_digest(mesh.GetFieldData().GetAbstractArray(i))
            for i in range(mesh.GetFieldData().GetNumberOfArrays())
        ),
        mesh.GetExtent() if isinstance(mesh, _vtk.vtkImageData) else None,
        mesh.bounds,
    ]
    if isinstance(mesh, _vtk.vtkPointSet):
        points = mesh.GetPoints()
        parts.append(None if points is None else _array_digest(points.GetData()))
    if isinstance(mesh, _vtk.vtkPolyData):
        parts += [
            _cells_digest(getattr(mesh, f'Get{name}')())
            for name in ('Verts', 'Lines', 'Polys', 'Strips')
        ]
    if isinstance(mesh, _vtk.vtkUnstructuredGrid):
        parts.append(_cells_digest(mesh.GetCells()))
    if isinstance(mesh, _vtk.vtkRectilinearGrid):
        parts += [
            _array_digest(getattr(mesh, f'Get{axis}Coordinates')()) for axis in ('X', 'Y', 'Z')
        ]
    # Names of the arrays which are read back as bool or complex live on the dataset
    for names in (mesh._association_bitarray_names, mesh._association_complex_names):
        parts.append(
            tuple(sorted((key, tuple(sorted(value))) for key, value in names.items() if value))
        )
    return tuple(parts)


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
    'sample_over_circular_arc': dict(pointa=(-1, 0, 0), pointb=(1, 0, 0), center=(0, 0, 0)),
    'sample_over_circular_arc_normal': dict(center=(0, 0, 0)),
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
    if match is None:
        return None
    try:
        return ast.literal_eval(f'[{match.group(1)}]')
    except (SyntaxError, ValueError):
        return None


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


def _filters():
    """Return every public dataobject and dataset filter."""
    found = {}
    for cls in (DataObjectFilters, DataSetFilters):
        for name in sorted(vars(cls)):
            if not name.startswith('_') and name not in _PLOTTING_FILTERS:
                found[name] = getattr(cls, name)
    return found


FILTERS = _filters()

MESH_KINDS = ['poly', 'unstructured', 'image', 'rectilinear', 'structured', 'pointset']
DATA_MODES = ['point', 'cell', 'both', 'single_point', 'single_cell', 'single_vector']


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


@pytest.mark.parametrize('name', list(FILTERS))
def test_filter_does_not_modify_input(name):
    func = FILTERS[name]
    ran = 0
    for mode in DATA_MODES:
        for kind in MESH_KINDS:
            template = _make_mesh(kind, mode)
            if not hasattr(template, name):
                continue
            for keyword, kwargs in _call_variants(func):
                if (kind, name, keyword) in _CRASHES:
                    continue
                mesh = template.copy()
                args = _POSITIONAL_ARGS[name]() if name in _POSITIONAL_ARGS else ()
                call_kwargs = {
                    key: value.build() if isinstance(value, _Fresh) else value
                    for key, value in {**_REQUIRED_KWARGS.get(name, {}), **kwargs}.items()
                }
                before = fingerprint(mesh)
                if not _run(mesh, name, args, call_kwargs):
                    continue
                ran += 1
                assert fingerprint(mesh) == before, (
                    f'{type(mesh).__name__}.{name}() modified its input '
                    f'({kind} mesh, {mode} data, {keyword}={call_kwargs.get(keyword)!r})'
                )
    assert ran, f'{name} never ran; the test meshes or arguments no longer apply'
