"""Typing cases for :func:`pyvista.merge`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import image
from tests.typing.meshes import multiblock
from tests.typing.meshes import multiblock_poly
from tests.typing.meshes import poly
from tests.typing.meshes import unstructured


def datasets() -> pv.MultiBlock[pv.DataSet]:
    """Return a composite declared to hold only datasets."""
    return pv.MultiBlock([poly(), image()])


# Any sequence of datasets gives back a `DataSet`, whatever the merged result is
assert_types(pv.merge([poly()]), pv.DataSet)
assert_types(pv.merge([poly(), poly()]), pv.DataSet)
assert_types(pv.merge([poly(), unstructured()]), pv.DataSet)
assert_types(pv.merge((poly(), image())), pv.DataSet)

# Any `MultiBlock` is accepted, whatever its blocks are declared to be
assert_types(pv.merge(datasets()), pv.DataSet)
assert_types(pv.merge(multiblock()), pv.DataSet)
assert_types(pv.merge(multiblock_poly()), pv.DataSet)

# The keywords do not change what comes back
assert_types(pv.merge([poly(), poly()], merge_points=False), pv.DataSet)
assert_types(pv.merge([poly(), poly()], progress_bar=True), pv.DataSet)
