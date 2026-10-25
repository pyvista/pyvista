"""Typing cases for the ``load`` and ``texture`` overloads of the example loaders."""

from __future__ import annotations

import importlib.util

from type_assert import assert_types
from type_assert import skip_runtime

import pyvista as pv
from pyvista import _vtk
from pyvista import examples
from pyvista.examples import planets

HAS_RENDERING = importlib.util.find_spec(f'{_vtk._VTK_ROOT}.vtkRenderingCore') is not None


def missing(name: str) -> bool:
    """Return whether the files of example ``name`` are not available locally."""
    try:
        examples.get_example(name, download=False)
    except FileNotFoundError:
        return True
    return False


# One file, read as a dataset
with skip_runtime(missing('bunny'), reason='not downloaded'):
    assert_types(examples.download_bunny(), pv.PolyData)
    assert_types(examples.download_bunny(load=True), pv.PolyData)
    assert_types(examples.download_bunny(load=False), str)

# Several files, read as a composite
with skip_runtime(missing('electronics_cooling'), reason='not downloaded'):
    assert_types(examples.download_electronics_cooling(), pv.MultiBlock[pv.PolyData | pv.UnstructuredGrid])
    assert_types(examples.download_electronics_cooling(load=False), tuple[str, ...])

# A texture, whose files are one path or several
with skip_runtime(missing('masonry_texture') or not HAS_RENDERING, reason='not downloaded, or no rendering'):
    assert_types(examples.download_masonry_texture(), pv.Texture)
with skip_runtime(missing('masonry_texture'), reason='not downloaded'):
    assert_types(examples.download_masonry_texture(load=False), str)
with skip_runtime(missing('sky_box_cube_map') or not HAS_RENDERING, reason='not downloaded, or no rendering'):
    assert_types(examples.download_sky_box_cube_map(), pv.Texture)
with skip_runtime(missing('sky_box_cube_map'), reason='not downloaded'):
    assert_types(examples.download_sky_box_cube_map(load=False), tuple[str, ...])

# A planet image, returned as a texture on request
with skip_runtime(missing('mars_surface'), reason='not downloaded'):
    assert_types(planets.download_mars_surface(), pv.ImageData)
    assert_types(planets.download_mars_surface(texture=False), pv.ImageData)
    assert_types(planets.download_mars_surface(load=False), str)
    assert_types(planets.download_mars_surface(texture=True, load=False), str)
with skip_runtime(missing('mars_surface') or not HAS_RENDERING, reason='not downloaded, or no rendering'):
    assert_types(planets.download_mars_surface(texture=True), pv.Texture)
