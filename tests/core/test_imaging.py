from __future__ import annotations

import os

import numpy as np
from numpy.testing import assert_allclose
import pytest

import pyvista as pv


def test_perlin_noise():
    amp = 10
    freq = (3, 2, 1)
    phase = (0, 0, 0)
    perlin = pv.perlin_noise(amp, freq, phase)

    assert_allclose(perlin.GetAmplitude(), amp)
    assert_allclose(perlin.GetFrequency(), freq)
    assert_allclose(perlin.GetPhase(), phase)


@pytest.mark.parametrize(
    'dtype',
    [
        np.float64,
        np.float32,
        np.int64,
        np.uint64,
        np.int32,
        np.uint32,
        np.int16,
        np.uint16,
        np.int8,
        np.uint8,
    ],
)
def test_sample_function(dtype):
    perlin = pv.perlin_noise(0.1, (1, 1, 1), (0, 0, 0))
    bounds = (0, 2, 0, 1, -4, 4)
    dim = (5, 10, 20)
    scalar_arr_name = 'my_scalars'

    if os.name == 'nt' and dtype in [np.int64, np.uint64]:
        with pytest.raises(ValueError):  # noqa: PT011
            pv.sample_function(perlin, output_type=dtype)
    else:
        mesh = pv.sample_function(
            perlin,
            bounds=bounds,
            dim=dim,
            compute_normals=False,
            output_type=dtype,
            scalar_arr_name=scalar_arr_name,
        )

        assert_allclose(mesh.dimensions, dim)
        assert_allclose(mesh.bounds, bounds)
        assert mesh[scalar_arr_name].dtype == dtype
