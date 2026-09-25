"""Typing cases for :attr:`pyvista.ImageData.index_to_physical_matrix`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

from tests.typing.meshes import image

assert_types(image().index_to_physical_matrix, npt.NDArray[np.float64])
