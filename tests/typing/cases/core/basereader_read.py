"""Typing cases for :meth:`pyvista.BaseReader.read`."""

from __future__ import annotations

from pathlib import Path
import tempfile

from type_assert import assert_types

import pyvista as pv
from pyvista import examples
from tests.typing.meshes import image
from tests.typing.meshes import multiblock
from tests.typing.meshes import poly
from tests.typing.meshes import rectilinear
from tests.typing.meshes import unstructured


def a_file(mesh: pv.DataObject, suffix: str) -> str:
    """Write ``mesh`` to a temporary file and return its path."""
    path = Path(tempfile.mkdtemp()) / f'mesh{suffix}'
    mesh.save(path)
    return str(path)


def a_reader() -> pv.BaseReader[pv.ImageData]:
    """Return a reader typed only as the generic base class."""
    return pv.XMLImageDataReader(a_file(image(), '.vti'))


# Each subclass binds the type variable, so `read` names one concrete class
assert_types(pv.XMLPolyDataReader(a_file(poly(), '.vtp')).read(), pv.PolyData)
assert_types(pv.XMLImageDataReader(a_file(image(), '.vti')).read(), pv.ImageData)
assert_types(pv.XMLRectilinearGridReader(a_file(rectilinear(), '.vtr')).read(), pv.RectilinearGrid)
assert_types(pv.XMLUnstructuredGridReader(a_file(unstructured(), '.vtu')).read(), pv.UnstructuredGrid)
assert_types(pv.XMLMultiBlockDataReader(a_file(multiblock(), '.vtm')).read(), pv.MultiBlock)

# A non-XML reader binds it the same way
assert_types(pv.PLYReader(examples.spherefile).read(), pv.PolyData)

# A format that allows several outputs binds the common supertype
assert_types(pv.VTKDataSetReader(examples.hexbeamfile).read(), pv.DataSet)

# The `validate` keyword does not widen the result
assert_types(pv.XMLPolyDataReader(a_file(poly(), '.vtp')).read(validate=True), pv.PolyData)
assert_types(pv.XMLMultiBlockDataReader(a_file(multiblock(), '.vtm')).read(validate=None), pv.MultiBlock)

# The base class itself carries the binding
assert_types(a_reader().read(), pv.ImageData)
