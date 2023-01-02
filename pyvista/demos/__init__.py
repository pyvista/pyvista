"""PyVista Demos."""
from pyvista.demos.demos import (
    glyphs,
    plot_glyphs,
    plot_datasets,
    orientation_cube,
    orientation_plotter,
    plot_wave,
    plot_ants_plane,
    plot_beam,
)
from pyvista.demos.logo import logo_atomized, logo_basic, logo_letters, logo_voxel, plot_logo


# __all__ only left for mypy --strict to work when pyvista is a dependency
__all__ = [
    'glyphs',
    'logo_atomized',
    'logo_basic',
    'logo_letters',
    'logo_voxel',
    'orientation_cube',
    'orientation_plotter',
    'plot_ants_plane',
    'plot_beam',
    'plot_datasets',
    'plot_glyphs',
    'plot_logo',
    'plot_wave',
]
