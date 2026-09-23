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

The filters, arguments and special meshes live in ``filter_side_effects_cases.py``, whose
docstring says what to do when a new filter or keyword makes this module fail.
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
from tests.core import filter_side_effects_cases as cases

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
        f'_override_mesh({name!r}, {mode!r}, None)'
        if name in cases.MESH_OVERRIDES
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


def _literal_options(annotation):
    """Return the options of a ``Literal`` annotation, or ``None``."""
    match = _LITERAL_PATTERN.search(str(annotation))
    return None if match is None else ast.literal_eval(f'[{match.group(1)}]')


def _kwarg_variants(parameter):
    """Yield values to try for a single keyword parameter."""
    if parameter.name in cases.SKIP_KWARGS:
        return
    if parameter.name in cases.KWARG_VALUES:
        yield from cases.KWARG_VALUES[parameter.name]
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
            if not name.startswith('_') and name not in (
                cases.PLOTTING_FILTERS | cases.DEPRECATED_FILTERS
            ):
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
    """Call a filter, returning the error it raised, or ``None`` if it ran."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            getattr(mesh, name)(*args, **kwargs)
        except Exception as error:  # noqa: BLE001  - the filter does not apply to this mesh
            return error
    return None


@pytest.fixture(autouse=True)
def _quiet_vtk():
    """Silence the VTK errors raised by filters which do not apply to a mesh."""
    pv.vtk_verbosity('off')
    yield
    pv.vtk_verbosity('info')


def _override_mesh(name, mode, default):
    """Return the special input a filter needs for this mode, or ``None`` if it has none."""
    if name not in cases.MESH_OVERRIDES:
        return default
    try:
        return cases.MESH_OVERRIDES[name](lambda mesh: _mesh_arrays(mesh, mode))
    except Exception:  # noqa: BLE001  - this mode cannot build that input
        return None


def _call_arguments(name, keyword_variant=()):
    """Return the positional and keyword arguments for one call of ``name``."""
    args = cases.POSITIONAL_ARGS[name]() if name in cases.POSITIONAL_ARGS else ()
    merged = {**cases.REQUIRED_KWARGS.get(name, {}), **dict(keyword_variant)}
    kwargs = {
        key: value.build() if isinstance(value, cases.Fresh) else value
        for key, value in merged.items()
    }
    return args, kwargs


#: Where the filters, arguments and meshes the sweep uses are listed.
CASES_FILE = 'tests/core/filter_side_effects_cases.py'

#: What to look at when a filter modifies its input.
MODIFIED_HINT = (
    'Hint: a filter must leave `self` unchanged. Resolve array names with the non-mutating '
    'helpers in pyvista/core/utilities/arrays.py (`_active_scalars_input`, '
    '`_default_scalars_input` and their vector twins) and give VTK the shallow copy they '
    'return. Run any call above on its own to reproduce it.'
)

#: What to look at when a filter's output depends on the input's active scalars.
ACTIVE_SCALARS_HINT = (
    "Hint: the filter reads the input's active scalars instead of resolving its default "
    'array itself; use `_default_scalars_input` from pyvista/core/utilities/arrays.py. If '
    'the filter carries the active scalars to its output by design, add it to '
    'ACTIVE_SCALARS_PASSTHROUGH in ' + CASES_FILE + '.'
)


class SweepSetupError(Exception):
    """The sweep could not exercise a filter, which is a problem with the test, not the filter."""


def _fail(key, problem, reports, ran, hint):
    """Fail with one readable block per call which broke the property, then a hint."""
    pytest.fail(
        f'{key} {problem} in {len(reports)} of {ran} calls:\n\n'
        + '\n\n'.join(reports)
        + f'\n\n{hint}',
        pytrace=False,
    )


def _first_line(error):
    """Return the first line of an error's message, shortened."""
    return next(iter(str(error).splitlines()), '')[:120]


def _fail_setup(key, errors):
    """Raise because a filter never ran, naming the first errors it raised."""
    seen = '\n'.join(
        f'  {where}: {type(error).__name__}: {_first_line(error)}'
        for where, error in list(errors.items())[:5]
    )
    msg = (
        f'{key} raised on every test mesh, so the sweep checked nothing. This is a problem '
        f'with the test setup, not a side effect. First errors, by mesh kind and data mode:\n'
        f'{seen}\n\n'
        f'Fix: in {CASES_FILE}, give the filter its required arguments in POSITIONAL_ARGS or '
        f'REQUIRED_KWARGS, or a mesh it accepts in MESH_OVERRIDES.'
    )
    raise SweepSetupError(msg)


@pytest.mark.parametrize('key', list(FILTERS))
def test_filter_does_not_modify_input(key):
    """A filter leaves its input's arrays, active arrays, points and cells alone, even on error."""
    func = FILTERS[key]
    name = func.__name__
    reports = []
    errors = {}
    ran = 0
    for mode in DATA_MODES:
        for kind in MESH_KINDS:
            template = _override_mesh(name, mode, _make_mesh(kind, mode))
            if template is None or not hasattr(template, name):
                continue
            for keyword, keyword_variant in _call_variants(func):
                if keyword is not None and mode not in KEYWORD_DATA_MODES:
                    continue
                if (kind, name, keyword) in cases.CRASHES_VTK:
                    continue
                mesh = template.copy()
                args, kwargs = _call_arguments(name, keyword_variant)
                before = _fingerprint(mesh)
                error = _run(mesh, name, args, kwargs)
                if error is None:
                    ran += 1
                else:
                    errors.setdefault(f'{kind}/{mode}', error)
                changes = _changes(before, _fingerprint(mesh))
                if changes:  # pragma: no cover -- failure path
                    reports.append(_report(kind, mode, name, args, kwargs, changes))
    if not ran:  # pragma: no cover -- failure path
        _fail_setup(key, errors)
    if reports:  # pragma: no cover -- failure path
        _fail(key, 'modified its input', reports, ran, MODIFIED_HINT)


def _unvaried_keywords():
    """Return the filter keywords the sweep has no value for, with the filters using each."""
    unvaried = {}
    for key, func in FILTERS.items():
        for parameter in list(inspect.signature(func).parameters.values())[1:]:
            if (
                parameter.default is not inspect.Parameter.empty
                and parameter.name not in cases.SKIP_KWARGS
                and next(_kwarg_variants(parameter), None) is None
            ):
                unvaried.setdefault(parameter.name, []).append(key)
    return unvaried


def _check_keyword_setup(unvaried, listed):
    """Raise if a keyword lacks values without being listed, or is listed but has values."""
    new = sorted(set(unvaried) - listed)
    if new:
        lines = '\n'.join(f'  {name}, used by {", ".join(unvaried[name])}' for name in new)
        msg = (
            f'The sweep has no values to try for these keywords, so it only ever calls them '
            f'with their defaults:\n{lines}\n\n'
            f'Fix: in {CASES_FILE}, add values to KWARG_VALUES, or add the name to '
            f'SKIP_KWARGS if the keyword cannot affect the input.'
        )
        raise SweepSetupError(msg)
    stale = sorted(listed - set(unvaried))
    if stale:
        msg = (
            f'These keywords are listed in UNVARIED_KWARGS but are varied now, or no filter '
            f'uses them any more: {stale}\n\nFix: remove them from UNVARIED_KWARGS in '
            f'{CASES_FILE}.'
        )
        raise SweepSetupError(msg)


def test_setup_every_keyword_has_values():
    """Every filter keyword has values to try, or is listed as skipped or not yet varied."""
    _check_keyword_setup(_unvaried_keywords(), cases.UNVARIED_KWARGS)


_DEFAULT_SCALARS_FILTERS = [
    key
    for key, func in FILTERS.items()
    if 'scalars' in inspect.signature(func).parameters
    and func.__name__ not in cases.ACTIVE_SCALARS_PASSTHROUGH
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
    errors = {}
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
                errors.setdefault(f'{kind}/{mode}', _run(template.copy(), name, args, kwargs))
                continue  # the filter does not apply to this mesh
            ran += 1
            changes = _changes(as_is, preactivated)
            if changes:  # pragma: no cover -- failure path
                reports.append(_report(kind, mode, name, args, kwargs, changes))
    if not ran:  # pragma: no cover -- failure path
        _fail_setup(key, errors)
    if reports:  # pragma: no cover -- failure path
        _fail(
            key,
            'returns a different output once its default array is active',
            reports,
            ran,
            ACTIVE_SCALARS_HINT,
        )


def test_failure_names_the_call_and_every_entry_which_changed():
    """A failure report names the call which broke the property and what it did to the mesh."""
    mesh = _make_mesh('poly', 'single_point')
    before = _fingerprint(mesh)
    mesh.point_data['extra'] = np.arange(mesh.n_points, dtype=float)
    mesh.set_active_scalars('extra')
    changes = _changes(before, _fingerprint(mesh))
    report = _report('poly', 'single_point', 'sample', (pv.Sphere(),), {'tolerance': 0.5}, changes)

    with pytest.raises(pytest.fail.Exception) as excinfo:
        _fail('DataSetFilters.sample', 'modified its input', [report], 4, MODIFIED_HINT)

    message = str(excinfo.value)
    assert message.endswith(MODIFIED_HINT)
    assert message.startswith('DataSetFilters.sample modified its input in 1 of 4 calls:')
    assert f'  poly mesh, {DATA_MODES["single_point"]}' in message
    assert "    _make_mesh('poly', 'single_point').sample(<PolyData>, tolerance=0.5)" in message
    assert "      point data arrays: +'extra:" in message
    assert "      point data active scalars: None -> 'extra'" in message


def test_failure_names_reordered_arrays_as_reordered():
    """Arrays which only changed order are reported as reordered, not as added and removed."""
    change = _describe_change('cell data arrays', ('a', 'b'), ('b', 'a'))
    assert change == 'cell data arrays: reordered'


def test_setup_failure_names_the_errors_and_the_tables_to_edit():
    """A filter which never ran is reported as a setup problem, with its errors and the fix."""
    errors = {'poly/both': TypeError("missing 1 required positional argument: 'target'")}
    with pytest.raises(SweepSetupError, match='problem with the test setup') as excinfo:
        _fail_setup('DataSetFilters.sample', errors)
    message = str(excinfo.value)
    assert "  poly/both: TypeError: missing 1 required positional argument: 'target'" in message
    assert 'POSITIONAL_ARGS' in message


def test_setup_failure_names_new_and_stale_keywords():
    """A keyword without values, or one listed needlessly, is reported with the fix."""
    unvaried = {'order': ['ImageDataFilters.low_pass']}
    with pytest.raises(
        SweepSetupError, match=re.escape('order, used by ImageDataFilters.low_pass')
    ):
        _check_keyword_setup(unvaried, frozenset())
    with pytest.raises(SweepSetupError, match='remove them from UNVARIED_KWARGS'):
        _check_keyword_setup({}, frozenset({'order'}))
