import pytest

from pyvista.plotting.colors import get_cmap_safe

COLORMAPS = []

try:
    import matplotlib

    COLORMAPS.append('Greys')
    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False

try:
    import cmocean  # noqa: F401

    COLORMAPS.append('algae')
except ImportError:
    pass


try:
    import colorcet  # noqa: F401

    COLORMAPS.append('fire')
except:
    pass


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='Requires matplotlib')
@pytest.mark.parametrize("cmap", COLORMAPS)
def test_get_cmap_safe(cmap):
    assert isinstance(get_cmap_safe(cmap), matplotlib.colors.LinearSegmentedColormap)
