import pyvista


def test_extrude():
    arc = pyvista.CircularArc([-1, 0, 0], [1, 0, 0], [0, 0, 0])
    mesh = pyvista.extrude(arc, [0, 0, 1])
    mesh.plot()
