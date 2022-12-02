import pyvista as pv
from pyvista import examples


def test_volume(verify_image_cache):
    pl = pv.Plotter()
    vol = examples.download_knee_full()
    actor = pl.add_volume(vol, cmap="bone", opacity="sigmoid")
    actor.mapper.lookup_table.cmap = "viridis"
    assert actor.mapper.lookup_table.cmap.name == "viridis"
    actor.prop.SetShade(True)
    assert actor.prop.GetShade()
    pl.show()
