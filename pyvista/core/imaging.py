"""Contains PyVista mappings from vtkmodules.vtkImagingHybrid."""
from typing import Sequence
import sys, os

import numpy as np

from pyvista import _vtk, wrap
from .filters import _get_output, _update_alg


def sample_function(function: _vtk.vtkImplicitFunction,
                    bounds: Sequence[float] = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
                    dim: Sequence[int] = (50, 50, 50),
                    compute_normals: bool = False,
                    output_type: np.dtype = np.double,
                    capping: bool = False,
                    cap_value: float = sys.float_info.max,
                    scalar_arr_name: str = "scalars",
                    normal_arr_name: str = "normals",
                    progress_bar: bool = False):
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
        generated from ``perlin_noise``.

    bounds : length 6 sequence
        Specify the bounds in the format of:

        - ``(xmin, xmax, ymin, ymax, zmin, zmax)``

        Defaults to ``(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)``.

    dim : length 3 sequence
        Dimensions of the data on which to sample in the format of
        ``(xdim, ydim, zdim)``.  Defaults to ``(50, 50, 50)``.

    compute_normals : bool, optional
        Enable or disable the computation of normals.  Default ``False``.

    output_type : np.dtype, optional
        Set the output scalar type.  Defaults to ``np.double``.  One
        of the following:

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

    capping : bool, optional
        Enable or disable capping.  Default ``False``.  If capping is
        enabled, then the outer boundaries of the structured point set
        are set to cap value. This can be used to ensure surfaces are
        closed.

    cap_value : float, optional
        Capping value used with the ``capping`` parameter.

    scalar_arr_name : str, optional
        Set the scalar array name for this data set.  Defaults to
        ``"scalars"``

    normal_arr_name : str, optional
        Set the normal array name for this data set.  Defaults to
        ``"normals"``

    progress_bar : bool, optional
        Display a progress bar to indicate progress.  Default
        ``False``.

    Examples
    --------
    Sample perlin noise over a structured grid in 3D.

    >>> import pyvista
    >>> noise = pyvista.perlin_noise(0.1, (1, 1, 1), (0, 0, 0))
    >>> grid = pyvista.sample_function(noise, [0, 3.0, -0, 1.0, 0, 1.0],
    ...                                dim=(60, 20, 20))
    >>> out.plot(cmap='gist_earth_r', show_scalar_bar=False,
    ...          show_edges=True)  # doctest:+SKIP

    Sample perlin noise in 2D and plot it.

    >>> noise = pyvista.perlin_noise(0.1, (5, 5, 5), (0, 0, 0))
    >>> surf = pyvista.sample_function(noise, dim=(200, 200, 1))
    >>> surf.plot()    # doctest:+SKIP

    """
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
    return wrap(samp.GetOutput())
