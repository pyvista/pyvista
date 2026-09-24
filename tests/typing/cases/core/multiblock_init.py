"""Typing cases for :meth:`pyvista.MultiBlock.__init__`."""

from __future__ import annotations

from pathlib import Path
import tempfile

from type_assert import assert_types

import pyvista as pv
from pyvista import DataSet
from pyvista import MultiBlock
from pyvista import PolyData
from pyvista import _vtk
from tests.typing.meshes import image
from tests.typing.meshes import multiblock
from tests.typing.meshes import multiblock_poly
from tests.typing.meshes import poly


def a_file() -> str:
    """Write a composite to a temporary file and return its path."""
    path = Path(tempfile.mkdtemp()) / 'mesh.vtm'
    multiblock().save(path)
    return str(path)


def a_vtk_composite() -> _vtk.vtkMultiBlockDataSet:
    """Return a composite typed only as the VTK class."""
    return _vtk.vtkMultiBlockDataSet()


# The blocks passed in name the block type
assert_types(pv.MultiBlock([poly()]), MultiBlock[PolyData])
assert_types(pv.MultiBlock((poly(), poly())), MultiBlock[PolyData])
assert_types(pv.MultiBlock({'mesh': poly()}), MultiBlock[PolyData])
assert_types(pv.MultiBlock([poly(), None]), MultiBlock[PolyData | None])
assert_types(pv.MultiBlock([poly(), image()]), MultiBlock[DataSet])
assert_types(pv.MultiBlock([multiblock_poly()]), MultiBlock[MultiBlock[PolyData]])

# Copying keeps the block type
assert_types(pv.MultiBlock(multiblock_poly()), MultiBlock[PolyData])
assert_types(pv.MultiBlock(multiblock_poly(), deep=True), MultiBlock[PolyData])

# With no blocks to read it from, the block type stays open
assert_types(pv.MultiBlock(), MultiBlock)
assert_types(pv.MultiBlock([]), MultiBlock)
assert_types(pv.MultiBlock(a_vtk_composite()), MultiBlock)
assert_types(pv.MultiBlock(a_file()), MultiBlock)
assert_types(pv.MultiBlock(Path(a_file())), MultiBlock)

# Reader options reach `pyvista.read`
assert_types(pv.MultiBlock(a_file(), force_ext='.vtm'), MultiBlock)

# A declared block type is kept, so blocks of a subclass widen to it
declared: MultiBlock[DataSet] = pv.MultiBlock([poly()])
assert_types(declared, MultiBlock[DataSet])
