"""Typing cases for :meth:`pyvista.DataSetFilters.contour`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types
from typing_extensions import Never

import pyvista as pv
from tests.typing.meshes import explicit_structured
from tests.typing.meshes import image
from tests.typing.meshes import pointset
from tests.typing.meshes import poly
from tests.typing.meshes import rectilinear
from tests.typing.meshes import structured
from tests.typing.meshes import unstructured
from tests.typing.meshes import with_arrays

if TYPE_CHECKING:
    from numpy.typing import NDArray

SKIP_RUNTIME = {
    "with_arrays(pointset()).contour(scalars='s')": 'contouring a `PointSet` is not supported, so the call raises',
}


assert_types(with_arrays(poly()).contour(scalars='s'), pv.PolyData)
assert_types(with_arrays(image()).contour(scalars='s'), pv.PolyData)
assert_types(with_arrays(rectilinear()).contour(scalars='s'), pv.PolyData)
assert_types(with_arrays(structured()).contour(scalars='s'), pv.PolyData)
assert_types(with_arrays(unstructured()).contour(scalars='s'), pv.PolyData)
assert_types(with_arrays(explicit_structured()).contour(scalars='s'), pv.PolyData)


def float_list_scalars() -> list[float]:
    """Return one float per point of the shared image."""
    return [float(value) for value in range(image().n_points)]


def int64_scalars() -> NDArray[np.int64]:
    """Return one int64 value per point of the shared image."""
    return np.arange(image().n_points, dtype=np.int64)


assert_types(image().contour(scalars=float_list_scalars()), pv.PolyData)
assert_types(image().contour(scalars=int64_scalars()), pv.PolyData)

assert_types(with_arrays(pointset()).contour(scalars='s'), Never)  # pragma: no cover
