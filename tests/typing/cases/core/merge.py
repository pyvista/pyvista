"""Typing cases for :func:`pyvista.merge`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import image
from tests.typing.meshes import multiblock
from tests.typing.meshes import poly
from tests.typing.meshes import unstructured

# Any sequence of datasets gives back a `DataSet`, whatever the merged result is
assert_types(pv.merge([poly()]), pv.DataSet)
assert_types(pv.merge([poly(), poly()]), pv.DataSet)
assert_types(pv.merge([poly(), unstructured()]), pv.DataSet)
assert_types(pv.merge((poly(), image())), pv.DataSet)

# A `MultiBlock` is a `MutableSequence`, so it is a sequence of datasets too
assert_types(pv.merge(multiblock()), pv.DataSet)

# The keywords do not change what comes back
assert_types(pv.merge([poly(), poly()], merge_points=False), pv.DataSet)
assert_types(pv.merge([poly(), poly()], progress_bar=True), pv.DataSet)
