import pytest
from PIL import Image, ImageSequence
import numpy as np

import pyvista

HAS_IMAGEIO = True
try:
    import imageio  # noqa: F401
except ModuleNotFoundError:
    HAS_IMAGEIO = False


@pytest.fixture()
def gif_file(tmpdir):
    filename = str(tmpdir.join('sample.gif'))

    pl = pyvista.Plotter(window_size=(300, 200))
    pl.open_gif(filename, palettesize=16, fps=1)

    mesh = pyvista.Sphere()
    opacity = mesh.points[:, 0]
    opacity -= opacity.min()
    opacity /= opacity.max()
    for color in ['red', 'blue', 'green']:
        pl.clear()
        pl.background_color = 'w'
        pl.add_mesh(mesh, color=color, opacity=opacity)
        pl.camera_position = 'xy'
        pl.write_frame()

    pl.close()
    return filename


@pytest.mark.skipif(not HAS_IMAGEIO, reason="Requires imageio")
def test_gif_reader(gif_file):
    reader = pyvista.get_reader(gif_file)
    assert isinstance(reader, pyvista.GIFReader)
    assert reader.path == gif_file
    reader.show_progress()

    grid = reader.read()
    assert grid.n_arrays == 3

    img = Image.open(gif_file)
    new_grid = pyvista.UniformGrid(dimensions=(img.size[0], img.size[1], 1))

    # load each frame to the grid
    for i, frame in enumerate(ImageSequence.Iterator(img)):
        data = np.array(frame.convert('RGB').getdata(), dtype=np.uint8)
        data_name = f'frame{i}'
        new_grid.point_data.set_array(data, data_name)
        assert np.allclose(grid[data_name], new_grid[data_name])
