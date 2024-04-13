from pathlib import Path
import subprocess

import pytest

THIS_PATH = Path(__file__).parent.absolute()


@pytest.mark.parametrize(
    "server_path",
    [
        "../advanced/contour.py",
        "../advanced/custom_ui.py",
        "../basic/actor_color.py",
        "../basic/algorithm.py",
        "../basic/file_viewer.py",
        "../basic/mesh_scalars.py",
        "../basic/multi_views.py",
        "../basic/PyVistaLocalView.py",
        "../basic/PyVistaRemoteLocalView.py",
        "../basic/PyVistaRemoteView.py",
        "../basic/ui_template.py",
        "../validation/many_actors.py",
    ],
)
def test_serve(server_path):
    returncode = subprocess.run(
        ["python", Path(THIS_PATH) / server_path, "--serve", "-t", "1"],
    ).returncode
    assert returncode == 0
