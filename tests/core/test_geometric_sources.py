import numpy as np
import pytest

import pyvista as pv


def test_cone_source():
    algo = pv.ConeSource()
    assert np.array_equal(algo.center, (0.0, 0.0, 0.0))
    assert np.array_equal(algo.direction, (1.0, 0.0, 0.0))
    assert algo.height == 1.0
    assert algo.radius == 0.5
    assert algo.capping
    assert algo.resolution == 6
    with pytest.raises(ValueError):
        algo = pv.ConeSource(angle=0.0, radius=0.0)
    algo = pv.ConeSource(angle=0.0)
    assert algo.angle == 0.0
    algo = pv.ConeSource(radius=0.0)
    assert algo.radius == 0.0
