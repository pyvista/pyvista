"""Benchmarks for the layers every attribute access and object passes through."""

from __future__ import annotations

import operator

import pyvista as pv


def _touch_attributes(mesh):
    """Read a spread of attributes through the forwarding guard."""
    return (
        mesh.n_points,
        mesh.n_cells,
        mesh.bounds,
        mesh.center,
        mesh.array_names,
        mesh.n_arrays,
    )


def _build_and_discard():
    """Construct a dataset and let it fall out of scope."""
    return pv.PolyData().n_points


def test_attribute_forwarding(sphere, benchmark):
    """Read six attributes through the snake-case forwarding guard."""
    assert benchmark(_touch_attributes, sphere)


def test_frozen_setattr(mutable_sphere, benchmark):
    """Reassign an existing attribute on a frozen object."""
    names = mutable_sphere._association_bitarray_names
    benchmark(setattr, mutable_sphere, '_association_bitarray_names', names)
    assert mutable_sphere._association_bitarray_names is names


def test_celltype_attribute(benchmark):
    """Read a member of the cell type enum."""
    assert benchmark(operator.attrgetter('HEXAHEDRON'), pv.CellType)


def test_empty_polydata(benchmark):
    """Construct an empty polydata."""
    assert benchmark(pv.PolyData).n_points == 0


def test_dataobject_lifecycle(benchmark):
    """Construct a dataset and immediately discard it."""
    assert benchmark(_build_and_discard) == 0
