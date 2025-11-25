"""Module managing parametric objects."""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING

import numpy as np

import pyvista as pv
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core import _validation
from pyvista.core import _vtk_core as _vtk

from .geometric_sources import translate
from .helpers import wrap

if TYPE_CHECKING:
    from pyvista import PolyData
    from pyvista.core._typing_core import MatrixLike
    from pyvista.core._typing_core import VectorLike


def Spline(
    points: VectorLike[float] | MatrixLike[float],
    n_points: int | None = None,
    *,
    closed: bool = False,
    parameterize_by: str = 'length',
    boundary_constraints: tuple[str] | str = "clamped",
    boundary_values: tuple[float] | float | None = 0.0,
    **kwargs,
) -> PolyData:
    """Create a spline from points.

    Parameters
    ----------
    points : numpy.ndarray
        Array of points to build a spline out of. Array must be 3D and
        directionally ordered.

    n_points : int, optional
        Number of points to interpolate along the points array. Defaults to
        ``points.shape[0]``.

    closed : bool, default: False
        Close the spline if ``True`` (both ends are joined). Not closed by default.

    parameterize_by : str, default: 'length'
        Parametrize spline by length or point index.

    boundary_constraints : Tuple[str] | str, optional, default: ('clamped', 'clamped')
        Derivative constraint type at both boundaries of the spline.
        Can be set by a single string (both ends) or a tuple of length equal to 2.
        Each value be one of:
        - 'finite_difference': The first derivative at the left(right) most point is determined
          from the line defined from the first(last) two points. (Default)
        - 'clamped': Default: the first derivative at the left(right) most point is set to
          Left(Right) value.
        - 'second': The second derivative at the left(right) most point is set to
          Left(Right) value.
        - 'scaled_second': The second derivative at left(right) most points is
          Left(Right) value times second derivative at first interior point.

    boundary_values : Tuple[float] | float | None, optional, default: (0.0, 0.0)
        Values of derivative at both ends of the spline.
        Can be set by a single float, or a tuple of floats or None (see below).
        Has to be None for each end with boundary constraint type 'finite_difference'.

    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        Line mesh of spline.

    See Also
    --------
    :ref:`distance_along_spline_example`

    Examples
    --------
    Construct a spline.

    >>> import numpy as np
    >>> import pyvista as pv
    >>> theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    >>> z = np.linspace(-2, 2, 100)
    >>> r = z**2 + 1
    >>> x = r * np.sin(theta)
    >>> y = r * np.cos(theta)
    >>> points = np.column_stack((x, y, z))
    >>> spline = pv.Spline(points, 1000)
    >>> spline.plot(
    ...     render_lines_as_tubes=True,
    ...     line_width=10,
    ...     show_scalar_bar=False,
    ... )

    """
    points_ = _validation.validate_arrayNx3(points, name='points')
    spline_function = _vtk.vtkParametricSpline()
    spline_function.SetPoints(pv.vtk_points(points_, deep=False))
    if closed:
        spline_function.ClosedOn()
    else:
        spline_function.ClosedOff()
    if parameterize_by == 'length':
        spline_function.ParameterizeByLengthOn()
    elif parameterize_by == 'index':
        spline_function.ParameterizeByLengthOff()
    else:  # pragma: no cover
        msg = f'Invalid parametrization of points {parameterize_by}'
        raise ValueError(msg)
    # handle single argument for constraint and values at both ends
    if type(boundary_constraints) is str:
        boundary_constraints = (boundary_constraints, boundary_constraints)
    if type(boundary_values) is float or boundary_values is None:
        boundary_values = (boundary_values, boundary_values)
    if len(boundary_constraints) != 2:
        msg = 'Invalid size for boundary constraints'
        raise ValueError(msg)
    if len(boundary_values) != 2:
        msg = 'Invalid size for boundary values'
        raise ValueError(msg)
    _boundary_types_dict = {
        'finite_difference': 0,
        'clamped': 1,
        'second': 2,
        'scaled_second': 3,
    }
    for incr, (constraint, value) in enumerate(
        zip(boundary_constraints,
            boundary_values,
            strict=True)
        ):
        if constraint in _boundary_types_dict.keys():
            if incr == 0:
                spline_function.SetLeftConstraint(_boundary_types_dict[constraint])
            else:
                spline_function.SetRightConstraint(_boundary_types_dict[constraint])
        else:  # pragma: no cover
            msg = f'Invalid boundary constraint {constraint}'
            raise ValueError(msg)
        if (value is not None and constraint == 'finite_difference'):  # pragma: no cover
            msg = f'''finite difference not compatible with
            boundary value {value} (should be None)'''
            raise ValueError(msg)
        elif value is not None:
            if incr == 0:
                spline_function.SetLeftValue(value)
            else:
                spline_function.SetRightValue(value)

    # get interpolation density
    u_res = n_points
    if u_res is None:
        u_res = points_.shape[0]
    u_res -= 1
    spline = surface_from_para(spline_function, u_res=u_res, **kwargs)
    return spline.compute_arc_length()


@_deprecate_positional_args(allowed=['points'])
def KochanekSpline(  # noqa: PLR0917
    points: VectorLike[float] | MatrixLike[float],
    tension: VectorLike[float] | None = None,
    bias: VectorLike[float] | None = None,
    continuity: VectorLike[float] | None = None,
    n_points: int | None = None,
) -> PolyData:
    """Create a Kochanek spline from points.

    Parameters
    ----------
    points : array_like[float]
        Array of points to build a Kochanek spline out of.  Array must
        be 3D and directionally ordered.

    tension : sequence[float], default: [0.0, 0.0, 0.0]
        Changes the length of the tangent vector.

    bias : sequence[float], default: [0.0, 0.0, 0.0]
        Primarily changes the direction of the tangent vector.

    continuity : sequence[float], default: [0.0, 0.0, 0.0]
        Changes the sharpness in change between tangents.

    n_points : int, optional
        Number of points on the spline. Defaults to ``points.shape[0]``.

    Returns
    -------
    pyvista.PolyData
        Kochanek spline.

    Examples
    --------
    Construct a Kochanek spline.

    >>> import numpy as np
    >>> import pyvista as pv
    >>> theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    >>> z = np.linspace(-2, 2, 100)
    >>> r = z**2 + 1
    >>> x = r * np.sin(theta)
    >>> y = r * np.cos(theta)
    >>> points = np.column_stack((x, y, z))
    >>> kochanek_spline = pv.KochanekSpline(points, n_points=6)
    >>> kochanek_spline.plot(line_width=4, color='k')

    See :ref:`create_kochanek_spline_example` for an additional example.

    """
    if tension is None:
        tension = np.array([0.0, 0.0, 0.0])
    tension_ = _validation.validate_arrayN(tension, must_be_in_range=[-1.0, 1.0], name='tension')

    if bias is None:
        bias = np.array([0.0, 0.0, 0.0])
    bias_ = _validation.validate_arrayN(bias, must_be_in_range=[-1.0, 1.0], name='bias')

    if continuity is None:
        continuity = np.array([0.0, 0.0, 0.0])
    continuity_ = _validation.validate_arrayN(
        continuity, must_be_in_range=[-1.0, 1.0], name='continuity'
    )

    points_ = _validation.validate_arrayNx3(points, name='points')
    spline_function = _vtk.vtkParametricSpline()
    spline_function.SetPoints(pv.vtk_points(points_, deep=False))

    # set Kochanek spline for each direction
    xspline = _vtk.vtkKochanekSpline()
    yspline = _vtk.vtkKochanekSpline()
    zspline = _vtk.vtkKochanekSpline()
    xspline.SetDefaultBias(bias_[0])
    yspline.SetDefaultBias(bias_[1])
    zspline.SetDefaultBias(bias_[2])
    xspline.SetDefaultTension(tension_[0])
    yspline.SetDefaultTension(tension_[1])
    zspline.SetDefaultTension(tension_[2])
    xspline.SetDefaultContinuity(continuity_[0])
    yspline.SetDefaultContinuity(continuity_[1])
    zspline.SetDefaultContinuity(continuity_[2])
    spline_function.SetXSpline(xspline)
    spline_function.SetYSpline(yspline)
    spline_function.SetZSpline(zspline)

    # get interpolation density
    u_res = n_points
    if u_res is None:
        u_res = points_.shape[0]

    u_res -= 1
    spline = surface_from_para(spline_function, u_res=u_res)
    return spline.compute_arc_length()


def ParametricBohemianDome(
    a: float | None = None,
    b: float | None = None,
    c: float | None = None,
    **kwargs,
) -> PolyData:
    """Generate a Bohemian dome surface.

    Parameters
    ----------
    a : float, default: 0.5
        Bohemian dome surface parameter a.

    b : float, default: 1.5
        Bohemian dome surface parameter b.

    c : float, default: 1.0
        Bohemian dome surface parameter c.

    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricBohemianDome surface.

    Examples
    --------
    Create a ParametricBohemianDome mesh.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricBohemianDome()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricBohemianDome()
    if a is not None:
        parametric_function.SetA(a)
    if b is not None:
        parametric_function.SetB(b)
    if c is not None:
        parametric_function.SetC(c)

    center = kwargs.pop('center', [0.0, 0.0, 0.0])
    direction = kwargs.pop('direction', [1.0, 0.0, 0.0])
    kwargs.setdefault('clean', True)
    surf = surface_from_para(parametric_function, **kwargs)
    translate(surf, center, direction)
    return surf


def ParametricBour(**kwargs) -> PolyData:
    """Generate Bour's minimal surface.

    Parameters
    ----------
    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricBour surface.

    Examples
    --------
    Create a ParametricBour mesh.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricBour()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricBour()

    center = kwargs.pop('center', [0.0, 0.0, 0.0])
    direction = kwargs.pop('direction', [1.0, 0.0, 0.0])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricBoy(zscale: float | None = None, **kwargs) -> PolyData:
    """Generate Boy's surface.

    This is a model of the projective plane without singularities.  It
    was found by Werner Boy on assignment from David Hilbert.

    For further information about this surface, please consult the
    technical description "Parametric surfaces" in the
    "VTK Technical Documents" section in the VTK.org web pages.

    Parameters
    ----------
    zscale : float, optional
        The scale factor for the z-coordinate.

    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricBoy surface.

    Examples
    --------
    Create a ParametricBoy mesh.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricBoy()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricBoy()
    if zscale is not None:
        parametric_function.SetZScale(zscale)

    center = kwargs.pop('center', [0.0, 0.0, 0.0])
    direction = kwargs.pop('direction', [1.0, 0.0, 0.0])
    kwargs.setdefault('clean', True)
    surf = surface_from_para(parametric_function, **kwargs)
    translate(surf, center, direction)
    return surf


def ParametricCatalanMinimal(**kwargs) -> PolyData:
    """Generate Catalan's minimal surface.

    ParametricCatalanMinimal generates Catalan's minimal surface
    parametrically. This minimal surface contains the cycloid as a
    geodesic.

    Parameters
    ----------
    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricCatalanMinimal surface.

    Examples
    --------
    Create a ParametricCatalanMinimal mesh.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricCatalanMinimal()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricCatalanMinimal()

    center = kwargs.pop('center', [0.0, 0.0, 0.0])
    direction = kwargs.pop('direction', [1.0, 0.0, 0.0])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricConicSpiral(  # noqa: PLR0917
    a: float | None = None,
    b: float | None = None,
    c: float | None = None,
    n: float | None = None,
    **kwargs,
) -> PolyData:
    """Generate conic spiral surfaces that resemble sea-shells.

    ParametricConicSpiral generates conic spiral surfaces. These can
    resemble sea shells, or may look like a torus "eating" its own
    tail.

    Parameters
    ----------
    a : float, default: 0.2
        The scale factor.

    b : float, default: 1
        The A function coefficient.
        See the definition in Parametric surfaces referred to above.

    c : float, default: 0.1
        The B function coefficient.
        See the definition in Parametric surfaces referred to above.

    n : float, default: 2
        The C function coefficient.
        See the definition in Parametric surfaces referred to above.

    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricConicSpiral surface.

    Examples
    --------
    Create a ParametricConicSpiral mesh.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricConicSpiral()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricConicSpiral()
    if a is not None:
        parametric_function.SetA(a)

    if b is not None:
        parametric_function.SetB(b)

    if c is not None:
        parametric_function.SetC(c)

    if n is not None:
        parametric_function.SetN(n)

    center = kwargs.pop('center', [0.0, 0.0, 0.0])
    direction = kwargs.pop('direction', [1.0, 0.0, 0.0])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricCrossCap(**kwargs) -> PolyData:
    """Generate a cross-cap.

    ParametricCrossCap generates a cross-cap which is a non-orientable
    self-intersecting single-sided surface.  This is one possible
    image of a projective plane in three-space.

    Parameters
    ----------
    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricCrossCap surface.

    Examples
    --------
    Create a ParametricCrossCap mesh.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricCrossCap()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricCrossCap()

    center = kwargs.pop('center', [0.0, 0.0, 0.0])
    direction = kwargs.pop('direction', [1.0, 0.0, 0.0])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricDini(a: float | None = None, b: float | None = None, **kwargs) -> PolyData:
    """Generate Dini's surface.

    ParametricDini generates Dini's surface.  Dini's surface is a
    surface that possesses constant negative Gaussian curvature

    Parameters
    ----------
    a : float, default: 1.0
        The scale factor.  See the definition in Parametric surfaces
        referred to above.

    b : float, default: 0.2
        The scale factor.  See the definition in Parametric surfaces
        referred to above.

    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricDini surface.

    Examples
    --------
    Create a ParametricDini mesh.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricDini()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricDini()
    if a is not None:
        parametric_function.SetA(a)

    if b is not None:
        parametric_function.SetB(b)

    center = kwargs.pop('center', [0.0, 0.0, 0.0])
    direction = kwargs.pop('direction', [1.0, 0.0, 0.0])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricEllipsoid(
    xradius: float | None = None,
    yradius: float | None = None,
    zradius: float | None = None,
    **kwargs,
) -> PolyData:
    """Generate an ellipsoid.

    ParametricEllipsoid generates an ellipsoid.  If all the radii are
    the same, we have a sphere.  An oblate spheroid occurs if RadiusX
    = RadiusY > RadiusZ.  Here the Z-axis forms the symmetry axis. To
    a first approximation, this is the shape of the earth.  A prolate
    spheroid occurs if RadiusX = RadiusY < RadiusZ.

    Parameters
    ----------
    xradius : float, default: 1.0
        The scaling factor for the x-axis.

    yradius : float, default: 1.0
        The scaling factor for the y-axis.

    zradius : float, default: 1.0
        The scaling factor for the z-axis.

    **kwargs : dict, optional
        See :func:`surface_from_para` and :func:`parametric_keywords`
        for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricEllipsoid surface.

    Examples
    --------
    Create a ParametricEllipsoid mesh.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricEllipsoid()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricEllipsoid()
    parametric_keywords(
        parametric_function,
        min_u=kwargs.pop('min_u', 0),
        max_u=kwargs.pop('max_u', 2 * pi),
        min_v=kwargs.pop('min_v', 0.0),
        max_v=kwargs.pop('max_v', pi),
        join_u=kwargs.pop('join_u', False),
        join_v=kwargs.pop('join_v', False),
        twist_u=kwargs.pop('twist_u', False),
        twist_v=kwargs.pop('twist_v', False),
        clockwise=kwargs.pop('clockwise', True),
    )

    if xradius is not None:
        parametric_function.SetXRadius(xradius)

    if yradius is not None:
        parametric_function.SetYRadius(yradius)

    if zradius is not None:
        parametric_function.SetZRadius(zradius)

    center = kwargs.pop('center', [0.0, 0.0, 0.0])
    direction = kwargs.pop('direction', [1.0, 0.0, 0.0])
    kwargs.setdefault('clean', True)
    surf = surface_from_para(parametric_function, **kwargs)
    translate(surf, center, direction)
    return surf


def ParametricEnneper(**kwargs) -> PolyData:
    """Generate Enneper's surface.

    ParametricEnneper generates Enneper's surface.  Enneper's surface
    is a self-intersecting minimal surface possessing constant
    negative Gaussian curvature.

    Parameters
    ----------
    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricEnneper surface.

    Examples
    --------
    Create a ParametricEnneper mesh.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricEnneper()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricEnneper()

    center = kwargs.pop('center', [0.0, 0.0, 0.0])
    direction = kwargs.pop('direction', [1.0, 0.0, 0.0])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricFigure8Klein(radius: float | None = None, **kwargs) -> PolyData:
    """Generate a figure-8 Klein bottle.

    ParametricFigure8Klein generates a figure-8 Klein bottle.  A Klein
    bottle is a closed surface with no interior and only one surface.
    It is unrealisable in 3 dimensions without intersecting surfaces.

    Parameters
    ----------
    radius : float, default: 1.0
        The radius of the bottle.

    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricFigure8Klein surface.

    Examples
    --------
    Create a ParametricFigure8Klein mesh.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricFigure8Klein()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricFigure8Klein()
    if radius is not None:
        parametric_function.SetRadius(radius)

    center = kwargs.pop('center', [0.0, 0.0, 0.0])
    direction = kwargs.pop('direction', [1.0, 0.0, 0.0])
    kwargs.setdefault('clean', True)
    surf = surface_from_para(parametric_function, **kwargs)
    translate(surf, center, direction)
    return surf


def ParametricHenneberg(**kwargs) -> PolyData:
    """Generate Henneberg's minimal surface.

    Parameters
    ----------
    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricHenneberg surface.

    Examples
    --------
    Create a ParametricHenneberg mesh.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricHenneberg()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricHenneberg()

    center = kwargs.pop('center', [0.0, 0.0, 0.0])
    direction = kwargs.pop('direction', [1.0, 0.0, 0.0])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricKlein(**kwargs) -> PolyData:
    """Generate a "classical" representation of a Klein bottle.

    ParametricKlein generates a "classical" representation of a Klein
    bottle.  A Klein bottle is a closed surface with no interior and only one
    surface.  It is unrealisable in 3 dimensions without intersecting
    surfaces.

    Parameters
    ----------
    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricKlein surface.

    Examples
    --------
    Create a ParametricKlein mesh.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricKlein()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricKlein()

    center = kwargs.pop('center', [0.0, 0.0, 0.0])
    direction = kwargs.pop('direction', [1.0, 0.0, 0.0])
    kwargs.setdefault('clean', True)
    surf = surface_from_para(parametric_function, **kwargs)
    translate(surf, center, direction)
    return surf


def ParametricKuen(deltav0: float | None = None, **kwargs) -> PolyData:
    """Generate Kuens' surface.

    ParametricKuen generates Kuens' surface. This surface has a constant
    negative Gaussian curvature.

    Parameters
    ----------
    deltav0 : float, default: 0.05
        The value to use when ``V == 0``.
        This has the best appearance with the default settings.
        Setting it to a value less than 0.05 extrapolates the surface
        towards a pole in the -z direction.
        Setting it to 0 retains the pole whose z-value is -inf.

    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricKuen surface.

    Examples
    --------
    Create a ParametricKuen mesh.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricKuen()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricKuen()
    if deltav0 is not None:
        parametric_function.SetDeltaV0(deltav0)

    center = kwargs.pop('center', [0.0, 0.0, 0.0])
    direction = kwargs.pop('direction', [1.0, 0.0, 0.0])
    kwargs.setdefault('clean', True)
    surf = surface_from_para(parametric_function, **kwargs)
    translate(surf, center, direction)
    return surf


def ParametricMobius(radius: float | None = None, **kwargs) -> PolyData:
    """Generate a Mobius strip.

    Parameters
    ----------
    radius : float, default: 1.0
        The radius of the Mobius strip.

    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricMobius surface.

    Examples
    --------
    Create a ParametricMobius mesh.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricMobius()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricMobius()
    if radius is not None:
        parametric_function.SetRadius(radius)

    center = kwargs.pop('center', [0.0, 0.0, 0.0])
    direction = kwargs.pop('direction', [1.0, 0.0, 0.0])
    surf = surface_from_para(parametric_function, **kwargs)
    translate(surf, center, direction)
    return surf


def ParametricPluckerConoid(n: int | None = None, **kwargs) -> PolyData:
    """Generate Plucker's conoid surface.

    ParametricPluckerConoid generates Plucker's conoid surface
    parametrically.  Plucker's conoid is a ruled surface, named after
    Julius Plucker. It is possible to set the number of folds in this
    class via the parameter 'n'.

    Parameters
    ----------
    n : int, optional
        This is the number of folds in the conoid.

    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricPluckerConoid surface.

    Examples
    --------
    Create a ParametricPluckerConoid mesh.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricPluckerConoid()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricPluckerConoid()
    if n is not None:
        parametric_function.SetN(n)

    center = kwargs.pop('center', [0.0, 0.0, 0.0])
    direction = kwargs.pop('direction', [1.0, 0.0, 0.0])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricPseudosphere(**kwargs) -> PolyData:
    """Generate a pseudosphere.

    ParametricPseudosphere generates a parametric pseudosphere. The
    pseudosphere is generated as a surface of revolution of the
    tractrix about it's asymptote, and is a surface of constant
    negative Gaussian curvature.

    Parameters
    ----------
    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricPseudosphere surface.

    Examples
    --------
    Create a ParametricPseudosphere mesh.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricPseudosphere()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricPseudosphere()

    center = kwargs.pop('center', [0.0, 0.0, 0.0])
    direction = kwargs.pop('direction', [1.0, 0.0, 0.0])
    kwargs.setdefault('clean', True)
    surf = surface_from_para(parametric_function, **kwargs)
    translate(surf, center, direction)
    return surf


@_deprecate_positional_args
def ParametricRandomHills(  # noqa: PLR0917
    number_of_hills: int | None = None,
    hill_x_variance: float | None = None,
    hill_y_variance: float | None = None,
    hill_amplitude: float | None = None,
    random_seed: int | None = None,
    x_variance_scale_factor: float | None = None,
    y_variance_scale_factor: float | None = None,
    amplitude_scale_factor: float | None = None,
    **kwargs,
) -> PolyData:
    """Generate a surface covered with randomly placed hills.

    ParametricRandomHills generates a surface covered with randomly
    placed hills. Hills will vary in shape and height since the
    presence of nearby hills will contribute to the shape and height
    of a given hill.  An option is provided for placing hills on a
    regular grid on the surface.  In this case the hills will all have
    the same shape and height.

    Parameters
    ----------
    number_of_hills : int, default: 30
        The number of hills.

    hill_x_variance : float, default: 2.5
        The hill variance in the x-direction.

    hill_y_variance : float, default: 2.5
        The hill variance in the y-direction.

    hill_amplitude : float, default: 2
        The hill amplitude (height).

    random_seed : int, default: 1
        The Seed for the random number generator,
        a value of 1 will initialize the random number generator,
        a negative value will initialize it with the system time.

    x_variance_scale_factor : float, default: 13
        The scaling factor for the variance in the x-direction.

    y_variance_scale_factor : float, default: 13
        The scaling factor for the variance in the y-direction.

    amplitude_scale_factor : float, default: 13
        The scaling factor for the amplitude.

    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricRandomHills surface.

    Examples
    --------
    Create a ParametricRandomHills mesh.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricRandomHills()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricRandomHills()
    if number_of_hills is not None:
        parametric_function.SetNumberOfHills(number_of_hills)

    if hill_x_variance is not None:
        parametric_function.SetHillXVariance(hill_x_variance)

    if hill_y_variance is not None:
        parametric_function.SetHillYVariance(hill_y_variance)

    if hill_amplitude is not None:
        parametric_function.SetHillAmplitude(hill_amplitude)

    if random_seed is not None:
        parametric_function.SetRandomSeed(random_seed)

    if x_variance_scale_factor is not None:
        parametric_function.SetXVarianceScaleFactor(x_variance_scale_factor)

    if y_variance_scale_factor is not None:
        parametric_function.SetYVarianceScaleFactor(y_variance_scale_factor)

    if amplitude_scale_factor is not None:
        parametric_function.SetAmplitudeScaleFactor(amplitude_scale_factor)

    center = kwargs.pop('center', [0.0, 0.0, 0.0])
    direction = kwargs.pop('direction', [1.0, 0.0, 0.0])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricRoman(radius: float | None = None, **kwargs) -> PolyData:
    """Generate Steiner's Roman Surface.

    Parameters
    ----------
    radius : float, default: 1
        The radius.

    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricRoman surface.

    Examples
    --------
    Create a ParametricRoman mesh.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricRoman()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricRoman()
    if radius is not None:
        parametric_function.SetRadius(radius)

    center = kwargs.pop('center', [0.0, 0.0, 0.0])
    direction = kwargs.pop('direction', [1.0, 0.0, 0.0])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


@_deprecate_positional_args(allowed=['xradius', 'yradius', 'zradius'])
def ParametricSuperEllipsoid(  # noqa: PLR0917
    xradius: float | None = None,
    yradius: float | None = None,
    zradius: float | None = None,
    n1: float | None = None,
    n2: float | None = None,
    **kwargs,
) -> PolyData:
    """Generate a superellipsoid.

    ParametricSuperEllipsoid generates a superellipsoid.  A superellipsoid
    is a versatile primitive that is controlled by two parameters n1 and
    n2. As special cases it can represent a sphere, square box, and closed
    cylindrical can.

    Parameters
    ----------
    xradius : float, default: 1
        The scaling factor for the x-axis.

    yradius : float, default: 1
        The scaling factor for the y-axis.

    zradius : float, default: 1
        The scaling factor for the z-axis.

    n1 : float, default: 1
        The "squareness" parameter in the z-axis.

    n2 : float, default: 1
        The "squareness" parameter in the x-y plane.

    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricSuperEllipsoid surface.

    See Also
    --------
    pyvista.ParametricSuperToroid :
        Toroidal equivalent of ParametricSuperEllipsoid.
    pyvista.Superquadric :
        Geometric object with additional parameters.

    Examples
    --------
    Create a ParametricSuperEllipsoid surface that looks like a box
    with smooth edges.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricSuperEllipsoid(n1=0.02, n2=0.02)
    >>> mesh.plot(color='w', smooth_shading=True)

    Create one that looks like a spinning top.

    >>> mesh = pv.ParametricSuperEllipsoid(n1=4, n2=0.5)
    >>> mesh.plot(color='w', smooth_shading=True, cpos='xz')

    """
    parametric_function = _vtk.vtkParametricSuperEllipsoid()
    if xradius is not None:
        parametric_function.SetXRadius(xradius)

    if yradius is not None:
        parametric_function.SetYRadius(yradius)

    if zradius is not None:
        parametric_function.SetZRadius(zradius)

    if n1 is not None:
        parametric_function.SetN1(n1)

    if n2 is not None:
        parametric_function.SetN2(n2)

    center = kwargs.pop('center', [0.0, 0.0, 0.0])
    direction = kwargs.pop('direction', [1.0, 0.0, 0.0])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


@_deprecate_positional_args
def ParametricSuperToroid(  # noqa: PLR0917
    ringradius: float | None = None,
    crosssectionradius: float | None = None,
    xradius: float | None = None,
    yradius: float | None = None,
    zradius: float | None = None,
    n1: float | None = None,
    n2: float | None = None,
    **kwargs,
) -> PolyData:
    """Generate a supertoroid.

    ParametricSuperToroid generates a supertoroid.  Essentially a
    supertoroid is a torus with the sine and cosine terms raised to a power.
    A supertoroid is a versatile primitive that is controlled by four
    parameters r0, r1, n1 and n2. r0, r1 determine the type of torus whilst
    the value of n1 determines the shape of the torus ring and n2 determines
    the shape of the cross section of the ring. It is the different values of
    these powers which give rise to a family of 3D shapes that are all
    basically toroidal in shape.

    Parameters
    ----------
    ringradius : float, default: 1
        The radius from the center to the middle of the ring of the
        supertoroid.

    crosssectionradius : float, default: 0.5
        The radius of the cross section of ring of the supertoroid.

    xradius : float, default: 1
        The scaling factor for the x-axis.

    yradius : float, default: 1
        The scaling factor for the y-axis.

    zradius : float, default: 1
        The scaling factor for the z-axis.

    n1 : float, default: 1
        The shape of the torus ring.

    n2 : float, default: 1
        The shape of the cross section of the ring.

    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricSuperToroid surface.

    See Also
    --------
    pyvista.ParametricSuperEllipsoid :
        Ellipsoidal equivalent of ParametricSuperToroid.
    pyvista.Superquadric :
        Geometric object with additional parameters.

    Examples
    --------
    Create a ParametricSuperToroid mesh.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricSuperToroid(n1=2, n2=0.3)
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricSuperToroid()
    if ringradius is not None:
        parametric_function.SetRingRadius(ringradius)

    if crosssectionradius is not None:
        parametric_function.SetCrossSectionRadius(crosssectionradius)

    if xradius is not None:
        parametric_function.SetXRadius(xradius)

    if yradius is not None:
        parametric_function.SetYRadius(yradius)

    if zradius is not None:
        parametric_function.SetZRadius(zradius)

    if n1 is not None:
        parametric_function.SetN1(n1)

    if n2 is not None:
        parametric_function.SetN2(n2)

    center = kwargs.pop('center', [0.0, 0.0, 0.0])
    direction = kwargs.pop('direction', [1.0, 0.0, 0.0])
    kwargs.setdefault('clean', True)
    surf = surface_from_para(parametric_function, **kwargs)
    translate(surf, center, direction)
    return surf


def ParametricTorus(
    ringradius: float | None = None, crosssectionradius: float | None = None, **kwargs
) -> PolyData:
    """Generate a torus.

    Parameters
    ----------
    ringradius : float, default: 1.0
        The radius from the center to the middle of the ring of the
        torus.

    crosssectionradius : float, default: 0.5
        The radius of the cross section of ring of the torus.

    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricTorus surface.

    Examples
    --------
    Create a ParametricTorus mesh.

    >>> import pyvista as pv
    >>> mesh = pv.ParametricTorus()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricTorus()
    if ringradius is not None:
        parametric_function.SetRingRadius(ringradius)

    if crosssectionradius is not None:
        parametric_function.SetCrossSectionRadius(crosssectionradius)

    center = kwargs.pop('center', [0.0, 0.0, 0.0])
    direction = kwargs.pop('direction', [1.0, 0.0, 0.0])
    kwargs.setdefault('clean', True)
    surf = surface_from_para(parametric_function, **kwargs)
    translate(surf, center, direction)
    return surf


@_deprecate_positional_args(allowed=['parametric_function'])
def parametric_keywords(  # noqa: PLR0917
    parametric_function: _vtk.vtkParametricFunction,
    min_u: float = 0.0,
    max_u: float = 2 * pi,
    min_v: float = 0.0,
    max_v: float = 2 * pi,
    join_u: bool = False,  # noqa: FBT001, FBT002
    join_v: bool = False,  # noqa: FBT001, FBT002
    twist_u: bool = False,  # noqa: FBT001, FBT002
    twist_v: bool = False,  # noqa: FBT001, FBT002
    clockwise: bool = True,  # noqa: FBT001, FBT002
) -> None:
    """Apply keyword arguments to a parametric function.

    Parameters
    ----------
    parametric_function : :vtk:`vtkParametricFunction`
        Parametric function to generate mesh from.

    min_u : float, optional
        The minimum u-value.

    max_u : float, optional
        The maximum u-value.

    min_v : float, optional
        The minimum v-value.

    max_v : float, optional
        The maximum v-value.

    join_u : bool, optional
        Joins the first triangle strip to the last one with a twist in
        the u direction.

    join_v : bool, optional
        Joins the first triangle strip to the last one with a twist in
        the v direction.

    twist_u : bool, optional
        Joins the first triangle strip to the last one with a twist in
        the u direction.

    twist_v : bool, optional
        Joins the first triangle strip to the last one with a twist in
        the v direction.

    clockwise : bool, optional
        Determines the ordering of the vertices forming the triangle
        strips.

    """
    parametric_function.SetMinimumU(min_u)
    parametric_function.SetMaximumU(max_u)
    parametric_function.SetMinimumV(min_v)
    parametric_function.SetMaximumV(max_v)
    parametric_function.SetJoinU(join_u)
    parametric_function.SetJoinV(join_v)
    parametric_function.SetTwistU(twist_u)
    parametric_function.SetTwistV(twist_v)
    parametric_function.SetClockwiseOrdering(clockwise)


@_deprecate_positional_args(allowed=['parametric_function'])
def surface_from_para(  # noqa: PLR0917
    parametric_function: _vtk.vtkParametricFunction,
    u_res: int = 100,
    v_res: int = 100,
    w_res: int = 100,
    clean: bool = False,  # noqa: FBT001, FBT002
    texture_coordinates: bool = False,  # noqa: FBT001, FBT002
) -> PolyData:
    """Construct a mesh from a parametric function.

    Parameters
    ----------
    parametric_function : :vtk:`vtkParametricFunction`
        Parametric function to generate mesh from.

    u_res : int, default: 100
        Resolution in the u direction.

    v_res : int, default: 100
        Resolution in the v direction.

    w_res : int, default: 100
        Resolution in the w direction.

    clean : bool, default: False
        Clean and merge duplicate points to avoid "creases" when
        plotting with smooth shading.

    texture_coordinates : bool, default: False
        The generation of texture coordinates.
        This is off by default. Note that this is only applicable
        to parametric surfaces whose parametric dimension is 2.
        Note that texturing may fail in some cases.

    Returns
    -------
    pyvista.PolyData
        Surface from the parametric function.

    """
    # convert to a mesh
    para_source = _vtk.vtkParametricFunctionSource()
    para_source.SetParametricFunction(parametric_function)
    para_source.SetUResolution(u_res)
    para_source.SetVResolution(v_res)
    para_source.SetWResolution(w_res)
    para_source.SetGenerateTextureCoordinates(texture_coordinates)
    para_source.Update()
    surf = wrap(para_source.GetOutput())
    if clean:
        surf = surf.clean(
            tolerance=1e-7,  # determined experimentally
            absolute=False,
            lines_to_points=False,
            polys_to_lines=False,
            strips_to_polys=False,
        )
    return surf
