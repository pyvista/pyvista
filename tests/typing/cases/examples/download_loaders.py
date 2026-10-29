"""Typing cases for the ``load`` and ``texture`` overloads of the example loaders."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista import examples
from pyvista.examples import planets

# One file, read as a dataset
assert_types(examples.download_bunny(), pv.PolyData)
assert_types(examples.download_bunny(load=True), pv.PolyData)
assert_types(examples.download_bunny(load=False), str)

# Several files, read as a composite
assert_types(examples.download_electronics_cooling(), pv.MultiBlock[pv.PolyData | pv.UnstructuredGrid])
assert_types(examples.download_electronics_cooling(load=False), tuple[str, ...])

# A texture, whose files are one path or several
assert_types(examples.download_masonry_texture(), pv.Texture)
assert_types(examples.download_masonry_texture(load=False), str)
assert_types(examples.download_sky_box_cube_map(), pv.Texture)
assert_types(examples.download_sky_box_cube_map(load=False), tuple[str, ...])

# A planet image, returned as a texture on request
assert_types(planets.download_mars_surface(), pv.ImageData)
assert_types(planets.download_mars_surface(texture=False), pv.ImageData)
assert_types(planets.download_mars_surface(load=False), str)
assert_types(planets.download_mars_surface(texture=True, load=False), str)
assert_types(planets.download_mars_surface(texture=True), pv.Texture)
