"""Contains PyVista mappings from vtkmodules.vtkCommonDataModel."""
from typing import Sequence

from pyvista import _vtk


def perlin_noise(amplitude, freq: Sequence[float], phase: Sequence[float]):
    """Return the implicit function that implements Perlin noise.

    Uses ``vtk.vtkPerlinNoise`` and computes a Perlin noise field as
    an implicit function. ``vtk.vtkPerlinNoise`` is a concrete
    implementation of ``vtk.vtkImplicitFunction``. Perlin noise,
    originally described by Ken Perlin, is a non-periodic and
    continuous noise function useful for modeling real-world objects.

    The amplitude and frequency of the noise pattern are
    adjustable. This implementation of Perlin noise is derived closely
    from Greg Ward's version in Graphics Gems II.

    Parameters
    ----------
    amplitude : float
        Amplitude of the noise function.

        ``amplitude`` can be negative. The noise function varies
        randomly between ``-|Amplitude|`` and
        ``|Amplitude|``. Therefore the range of values is
        ``2*|Amplitude|`` large. The initial amplitude is 1.

    freq : Sequence[float, float, float]
        The frequency, or physical scale, of the noise function
        (higher is finer scale).

        The frequency can be adjusted per axis, or the same for all axes.

    phase : Sequence[float, float, float]
        Set/get the phase of the noise function.

        This parameter can be used to shift the noise function within
        space (perhaps to avoid a beat with a noise pattern at another
        scale). Phase tends to repeat about every unit, so a phase of
        0.5 is a half-cycle shift.

    Examples
    --------
    Create a perlin noise function with an amplitude of 0.1, frequency
    for all axes of 1, and a phase of 0 for all axes.

    >>> import pyvista
    >>> noise = perlin_noise(0.1, (1, 1, 1), (0, 0, 0))

    Apply the perlin noise function to 

    """
    noise = _vtk.vtkPerlinNoise()
    noise.SetAmplitude(amplitude)
    noise.SetFrequency(freq)
    noise.SetPhase(phase)
    return noise
