"""Typing cases for :func:`pyvista.examples.get_example`."""

from __future__ import annotations

from type_assert import assert_types
from type_assert import skip_runtime

import pyvista as pv
from pyvista import examples
from pyvista.core.utilities.reader import PLYReader
from pyvista.core.utilities.reader import VTKDataSetReader
from pyvista.core.utilities.reader import XMLPolyDataReader
from pyvista.core.utilities.reader import XMLUnstructuredGridReader
from pyvista.examples import Example


def a_name() -> str:
    """Return an example name typed only as ``str``, so no ``Literal`` overload applies."""
    return 'ant'


def missing(name: str) -> bool:
    """Return whether the files of example ``name`` are not available locally."""
    try:
        examples.get_example(name, download=False)
    except FileNotFoundError:
        return True
    return False


# A literal name resolves the dataset and reader types
assert_types(examples.get_example('ant'), Example[pv.PolyData, tuple[PLYReader]])
assert_types(examples.get_example('ant').load(), pv.PolyData)
assert_types(examples.get_example('ant').readers, tuple[PLYReader])
assert_types(examples.get_example('uniform').load(), pv.ImageData)
assert_types(examples.get_example('uniform').readers, tuple[VTKDataSetReader])
assert_types(examples.get_example('structured').load(), pv.StructuredGrid)
assert_types(examples.get_example('structured').readers, tuple[()])
with skip_runtime(missing('electronics_cooling'), reason='not downloaded'):
    assert_types(examples.get_example('electronics_cooling').load(), pv.MultiBlock[pv.PolyData | pv.UnstructuredGrid])
    assert_types(examples.get_example('electronics_cooling').readers, tuple[XMLPolyDataReader, XMLUnstructuredGridReader])

# A function resolves the dataset type, and a plain string resolves neither
assert_types(examples.get_example(examples.load_ant).load(), pv.PolyData)
assert_types(examples.get_example(a_name()), Example)
