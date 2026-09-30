"""Typing cases for :meth:`pyvista.DataSetFilters.surface_indices`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types
from typing_extensions import Never

from tests.typing.meshes import explicit_structured
from tests.typing.meshes import image
from tests.typing.meshes import pointset
from tests.typing.meshes import poly
from tests.typing.meshes import rectilinear
from tests.typing.meshes import structured
from tests.typing.meshes import unstructured

SKIP_RUNTIME = {
    'pointset().surface_indices()': 'a `PointSet` has no cells, so the call raises',
}


assert_types(poly().surface_indices(), NDArray[np.signedinteger])
assert_types(image().surface_indices(), NDArray[np.signedinteger])
assert_types(rectilinear().surface_indices(), NDArray[np.signedinteger])
assert_types(structured().surface_indices(), NDArray[np.signedinteger])
assert_types(unstructured().surface_indices(), NDArray[np.signedinteger])
assert_types(explicit_structured().surface_indices(), NDArray[np.signedinteger])

assert_types(pointset().surface_indices(), Never)  # pragma: no cover
