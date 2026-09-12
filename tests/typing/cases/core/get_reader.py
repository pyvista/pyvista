"""Typing cases for :func:`pyvista.get_reader`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from type_assert import assert_types

import pyvista as pv
from pyvista import examples

# The reader class is chosen from the extension at runtime, so the result stays `Any`
assert_types(pv.get_reader(examples.channelsfile), Any)
assert_types(pv.get_reader(Path(examples.channelsfile)), Any)
assert_types(pv.get_reader(examples.hexbeamfile), Any)
assert_types(pv.get_reader(examples.hexbeamfile, '.vtk'), Any)
assert_types(pv.get_reader(examples.spherefile, force_ext='.ply'), Any)

# Members of the chosen subclass are reachable, without a cast or an ignore
assert_types(pv.get_reader(examples.channelsfile).number_point_arrays, Any)
assert_types(pv.get_reader(examples.channelsfile).read(), Any)
