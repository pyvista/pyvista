"""Common functions."""
import os
import sys
from typing import Sequence

import numpy as np

import pyvista
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

    Returns
    -------
    vtk.vtkPerlinNoise
        Instance of ``vtk.vtkPerlinNoise`` to a Perlin noise field as an
        implicit function. Use with :func:`pyvista.sample_function()
        <pyvista.utilities.common.sample_function>`.

    Examples
    --------
    Create a Perlin noise function with an amplitude of 0.1, frequency
    for all axes of 1, and a phase of 0 for all axes.

    >>> import pyvista
    >>> noise = pyvista.perlin_noise(0.1, (1, 1, 1), (0, 0, 0))

    Sample Perlin noise over a structured grid and plot it.

    >>> grid = pyvista.sample_function(noise, [0, 5, 0, 5, 0, 5])
    >>> grid.plot()

    """
    noise = _vtk.vtkPerlinNoise()
    noise.SetAmplitude(amplitude)
    noise.SetFrequency(freq)
    noise.SetPhase(phase)
    return noise


def sample_function(
    function: _vtk.vtkImplicitFunction,
    bounds: Sequence[float] = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
    dim: Sequence[int] = (50, 50, 50),
    compute_normals: bool = False,
    output_type: np.dtype = np.double,  # type: ignore
    capping: bool = False,
    cap_value: float = sys.float_info.max,
    scalar_arr_name: str = "scalars",
    normal_arr_name: str = "normals",
    progress_bar: bool = False,
):
    """Sample an implicit function over a structured point set.

    Uses ``vtk.vtkSampleFunction``

    This method evaluates an implicit function and normals at each
    point in a ``vtk.vtkStructuredPoints``. The user can specify the
    sample dimensions and location in space to perform the sampling.

    To create closed surfaces (in conjunction with the
    vtkContourFilter), capping can be turned on to set a particular
    value on the boundaries of the sample space.

    Parameters
    ----------
    function : vtk.vtkImplicitFunction
        Implicit function to evaluate.  For example, the function
        generated from :func:`pyvista.perlin_noise`.

    bounds : length 6 sequence, default: (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        Specify the bounds in the format of:

        - ``(xmin, xmax, ymin, ymax, zmin, zmax)``.

    dim : length 3 sequence, default: (50, 50, 50)
        Dimensions of the data on which to sample in the format of
        ``(xdim, ydim, zdim)``.

    compute_normals : bool, default: False
        Enable or disable the computation of normals.

    output_type : np.dtype, default: numpy.double
        Set the output scalar type.  One of the following:

        - ``np.float64``
        - ``np.float32``
        - ``np.int64``
        - ``np.uint64``
        - ``np.int32``
        - ``np.uint32``
        - ``np.int16``
        - ``np.uint16``
        - ``np.int8``
        - ``np.uint8``

    capping : bool, default: False
        Enable or disable capping. If capping is enabled, then the outer
        boundaries of the structured point set are set to cap value. This can
        be used to ensure surfaces are closed.

    cap_value : float, default: sys.float_info.max
        Capping value used with the ``capping`` parameter.

    scalar_arr_name : str, default: "scalars"
        Set the scalar array name for this data set.

    normal_arr_name : str, default: "normals"
        Set the normal array name for this data set.

    progress_bar : bool, default: False
        Display a progress bar to indicate progress.

    Returns
    -------
    pyvista.UniformGrid
        Uniform grid with sampled data.

    Examples
    --------
    Sample Perlin noise over a structured grid in 3D.

    >>> import pyvista
    >>> noise = pyvista.perlin_noise(0.1, (1, 1, 1), (0, 0, 0))
    >>> grid = pyvista.sample_function(noise, [0, 3.0, -0, 1.0, 0, 1.0],
    ...                                dim=(60, 20, 20))
    >>> grid.plot(cmap='gist_earth_r', show_scalar_bar=False, show_edges=True)

    Sample Perlin noise in 2D and plot it.

    >>> noise = pyvista.perlin_noise(0.1, (5, 5, 5), (0, 0, 0))
    >>> surf = pyvista.sample_function(noise, dim=(200, 200, 1))
    >>> surf.plot()

    See :ref:`perlin_noise_2d_example` for a full example using this function.

    """
    # internal import to avoide circular dependency
    from pyvista.core.filters import _update_alg

    samp = _vtk.vtkSampleFunction()
    samp.SetImplicitFunction(function)
    samp.SetSampleDimensions(dim)
    samp.SetModelBounds(bounds)
    samp.SetComputeNormals(compute_normals)
    samp.SetCapping(capping)
    samp.SetCapValue(cap_value)
    samp.SetNormalArrayName(normal_arr_name)
    samp.SetScalarArrayName(scalar_arr_name)

    if output_type == np.float64:
        samp.SetOutputScalarTypeToDouble()
    elif output_type == np.float32:
        samp.SetOutputScalarTypeToFloat()
    elif output_type == np.int64:
        if os.name == 'nt':
            raise ValueError('This function on Windows only supports int32 or smaller')
        samp.SetOutputScalarTypeToLong()
    elif output_type == np.uint64:
        if os.name == 'nt':
            raise ValueError('This function on Windows only supports int32 or smaller')
        samp.SetOutputScalarTypeToUnsignedLong()
    elif output_type == np.int32:
        samp.SetOutputScalarTypeToInt()
    elif output_type == np.uint32:
        samp.SetOutputScalarTypeToUnsignedInt()
    elif output_type == np.int16:
        samp.SetOutputScalarTypeToShort()
    elif output_type == np.uint16:
        samp.SetOutputScalarTypeToUnsignedShort()
    elif output_type == np.int8:
        samp.SetOutputScalarTypeToChar()
    elif output_type == np.uint8:
        samp.SetOutputScalarTypeToUnsignedChar()
    else:
        raise ValueError(f'Invalid output_type {output_type}')

    _update_alg(samp, progress_bar=progress_bar, message='Sampling')
    return pyvista.wrap(samp.GetOutput())
