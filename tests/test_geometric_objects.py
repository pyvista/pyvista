import numpy as np

import vtki


def test_cylinder():
    surf = vtki.Cylinder([0, 10, 0], [1, 1, 1], 1, 5)
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_arrow():
    surf = vtki.Arrow([0, 0, 0], [1, 1, 1])
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_sphere():
    surf = vtki.Sphere()
    assert np.any(surf.points)
    assert np.any(surf.faces)


def test_plane():
    surf = vtki.Plane()
    assert np.any(surf.points)
    assert np.any(surf.faces)
