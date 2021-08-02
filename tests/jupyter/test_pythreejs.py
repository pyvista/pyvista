"""Test optional pythreejs functionality"""

import pytest

try:
    import pythreejs
except:
    pytestmark = pytest.mark.skip

import pyvista


def test_export_to_html(cube, tmpdir):
    filename = str(tmpdir.mkdir("tmpdir").join(f'tmp.html'))

    pl = pyvista.Plotter()
    pl.add_mesh(cube)
    pl.export_html(filename)

    raw = open(filename).read()
    assert 'jupyter-threejs' in raw

    # expect phong material
    assert '"model_name": "MeshPhongMaterialModel"' in raw

    # at least a single instance of lighting
    assert 'DirectionalLightModel' in raw
