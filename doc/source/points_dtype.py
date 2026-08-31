"""Measure which filters generate double-precision points.

Support belongs to the VTK algorithm a filter runs rather than to PyVista, and it moves
between VTK releases and backends, so it is measured by running the filter instead of
read off a list. ``_templates/autosummary/class.rst`` marks each filter in a class's
``Filters`` section with the result, through the ``points_dtype_mark`` Jinja global
``conf.py`` registers.
"""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING
from typing import Any
import warnings

import numpy as np

import pyvista as pv
from pyvista.examples import cells

if TYPE_CHECKING:
    from collections.abc import Callable

YES_MARK = ':material-regular:`check;1.3em;sd-text-success`'
NO_MARK = ':material-regular:`close;1.3em;sd-text-error`'

_AXIS = np.arange(4.0)

#: A sample per dataset class that inherits a filter mixin, keyed by the name of the
#: class whose page is being generated.
SAMPLES: dict[str, Callable[[], Any]] = {
    'PolyData': lambda: pv.Sphere().points_to_double(),
    'UnstructuredGrid': cells.Hexahedron,
    'StructuredGrid': lambda: pv.StructuredGrid(*np.meshgrid(_AXIS, _AXIS, _AXIS, indexing='ij')),
    'PointSet': lambda: pv.PointSet(np.random.default_rng(0).random((30, 3))),
    'ImageData': lambda: pv.ImageData(dimensions=(5, 5, 5)),
    'RectilinearGrid': lambda: pv.RectilinearGrid(_AXIS, _AXIS, _AXIS),
    'ExplicitStructuredGrid': lambda: pv.StructuredGrid(
        *np.meshgrid(_AXIS, _AXIS, _AXIS, indexing='ij')
    ).cast_to_explicit_structured_grid(),
    'MultiBlock': lambda: pv.MultiBlock([pv.Sphere().points_to_double()]),
}


def _partner(mesh):
    """Return a surface overlapping ``mesh``, for the filters that take a second mesh.

    Sized and placed from the sample so that it genuinely intersects it, which the
    clipping and boolean filters need in order to produce any output at all.
    """
    x, y, z = mesh.center
    offset = (x + mesh.length / 6, y, z)
    return pv.Sphere(center=offset, radius=mesh.length / 3).points_to_double()


#: Sample arguments for the filters that require one, so they can be measured too. A
#: filter that needs an argument and is not listed here is left unmarked rather than
#: guessed at; ``tests/doc/test_points_dtype.py`` fails when a new one appears.
ARGS: dict[str, Callable[[Any], tuple[Any, ...]]] = {
    'align': lambda mesh: (mesh.copy(),),
    'boolean_difference': lambda mesh: (_partner(mesh),),
    'boolean_intersection': lambda mesh: (_partner(mesh),),
    'boolean_union': lambda mesh: (_partner(mesh),),
    'clip_slab': lambda mesh: (mesh.length / 4,),
    'clip_surface': lambda mesh: (_partner(mesh),),
    'compute_implicit_distance': lambda mesh: (_partner(mesh),),
    'concatenate': lambda mesh: (mesh.copy(),),
    'contour_banded': lambda _: (5,),
    'decimate': lambda _: (0.5,),
    'decimate_pro': lambda _: (0.5,),
    'extract_cells': lambda mesh: (range(max(mesh.n_cells // 2, 1)),),
    'extract_points': lambda mesh: (range(max(mesh.n_points // 2, 1)),),
    'extract_subset': lambda _: ((0, 2, 0, 2, 0, 2),),
    'extrude': lambda _: ((0.0, 0.0, 1.0),),
    'fill_holes': lambda mesh: (mesh.length / 4,),
    'flip_normal': lambda _: ((0.0, 0.0, 1.0),),
    'geodesic': lambda mesh: (0, mesh.n_points - 1),
    'image_threshold': lambda _: (0.5,),
    'interpolate': lambda mesh: (_partner(mesh),),
    'merge': lambda mesh: (_partner(mesh),),
    'partition': lambda _: (2,),
    'reflect': lambda _: ((1.0, 0.0, 0.0),),
    'rotate': lambda _: (np.eye(3),),
    'rotate_vector': lambda _: ((1.0, 0.0, 0.0), 30.0),
    'rotate_x': lambda _: (30.0,),
    'rotate_y': lambda _: (30.0,),
    'rotate_z': lambda _: (30.0,),
    'sample': lambda mesh: (_partner(mesh),),
    'sample_over_line': lambda mesh: (mesh.bounds[::2], mesh.bounds[1::2]),
    'scale': lambda _: (2.0,),
    'select_enclosed_points': lambda mesh: (_partner(mesh),),
    'subdivide': lambda _: (1,),
    'transform': lambda _: (np.eye(4),),
    'translate': lambda _: ((1.0, 1.0, 1.0),),
}


def sample(class_name: str):  # numpydoc ignore=RT01
    """Return a sample of ``class_name`` carrying point data for a filter to act on."""
    mesh = SAMPLES[class_name]()
    blocks = list(mesh.recursive_iterator()) if isinstance(mesh, pv.MultiBlock) else [mesh]
    for block in blocks:
        block['scalars'] = np.arange(block.n_points, dtype=float)
        block['vectors'] = np.ones((block.n_points, 3))
    return mesh


def _inplace_kwarg(method) -> dict[str, bool]:
    """Return ``inplace=False`` for the filters that take it.

    Those mid-deprecation raise unless it is passed explicitly, and measuring wants the
    new dataset either way.
    """
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return {}
    return {'inplace': False} if 'inplace' in parameters else {}


@functools.cache
def delivers_double(class_name: str, name: str) -> bool | None:  # numpydoc ignore=RT01
    """Return whether ``class_name.name`` generates double points.

    ``None`` where the filter does not run on the sample, or returns no points, so
    nothing was measured.
    """
    pv.global_config.points_dtype = 'float64'
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            mesh = sample(class_name)
            method = getattr(mesh, name)
            args = ARGS[name](mesh) if name in ARGS else ()
            output = method(*args, **_inplace_kwarg(method))
        points = getattr(output, 'points', None)
        if isinstance(output, pv.MultiBlock):
            points = next(
                (b.points for b in output.recursive_iterator(skip_none=True) if b.n_points),
                None,
            )
        if points is None or not points.size:
            return None
    except Exception:  # noqa: BLE001
        # A filter that cannot run on the sample says nothing about its support
        return None
    finally:
        pv.global_config.points_dtype = None
    return not any(issubclass(w.category, pv.PrecisionWarning) for w in caught)


def points_dtype_mark(class_name: str, label: str) -> str:  # numpydoc ignore=RT01
    """Return the double-precision mark for one filter on one dataset class.

    Empty where nothing was measured, so the filter is left unmarked.
    """
    if class_name not in SAMPLES:
        return ''
    verdict = delivers_double(class_name, label.rsplit('.', 1)[-1])
    if verdict is None:
        return ''
    return YES_MARK if verdict else NO_MARK
