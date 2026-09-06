"""Typing cases for :func:`pyvista.read`."""

from __future__ import annotations

from pathlib import Path

from type_assert import assert_types

import pyvista as pv
from pyvista import DataSet
from pyvista import MultiBlock
from pyvista import examples


def a_class() -> type[pv.UnstructuredGrid]:
    """Return the class to read into, so `cls` binds the return type variable."""
    return pv.UnstructuredGrid


# Without `cls` the reader picks the type, so only the union is known
assert_types(pv.read(examples.hexbeamfile), DataSet | MultiBlock)
assert_types(pv.read(Path(examples.hexbeamfile)), DataSet | MultiBlock)
assert_types(pv.read([examples.hexbeamfile, examples.spherefile]), DataSet | MultiBlock)
assert_types(pv.read(examples.hexbeamfile, cls=None), DataSet | MultiBlock)

# `cls` names the type that comes back
assert_types(pv.read(examples.hexbeamfile, cls=pv.UnstructuredGrid), pv.UnstructuredGrid)
assert_types(pv.read(examples.spherefile, cls=pv.PolyData), pv.PolyData)
assert_types(pv.read(examples.hexbeamfile, cls=a_class()), pv.UnstructuredGrid)
assert_types(pv.read(examples.spherefile, cls=pv.PolyData, validate=True), pv.PolyData)
