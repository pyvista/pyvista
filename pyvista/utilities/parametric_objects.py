"""Module managing parametric objects."""

from math import pi

import numpy as np

import pyvista
from pyvista import _vtk
from pyvista.utilities.errors import check_valid_vector

from .geometric_objects import translate


def Spline(points, n_points=None):
    """Create a spline from points.

    Parameters
    ----------
    points : np.ndarray
        Array of points to build a spline out of.  Array must be 3D
        and directionally ordered.

    n_points : int, optional
        Number of points to interpolate along the points array.

    Returns
    -------
    pyvista.PolyData
        Line mesh of spline.

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
    >>> spline.plot(render_lines_as_tubes=True, line_width=10, show_scalar_bar=False)

    """
    spline_function = _vtk.vtkParametricSpline()
    spline_function.SetPoints(pyvista.vtk_points(points, False))

    # get interpolation density
    u_res = n_points
    if u_res is None:
        u_res = points.shape[0]

    u_res -= 1
    spline = surface_from_para(spline_function, u_res)
    return spline.compute_arc_length()


def KochanekSpline(points, tension=None, bias=None, continuity=None, n_points=None):
    """Create a Kochanek spline from points.

    Parameters
    ----------
    points : sequence
        Array of points to build a Kochanek spline out of.  Array must
        be 3D and directionally ordered.

    tension : sequence, optional
        Changes the length of the tangent vector.  Defaults to ``[0.0,
        0.0, 0.0]``.

    bias : sequence, optional
        Primarily changes the direction of the tangent vector.
        Defaults to ``[0.0, 0.0, 0.0]``.

    continuity : sequence, optional
        Changes the sharpness in change between tangents.  Defaults to
        ``[0.0, 0.0, 0.0]``.

    n_points : int, optional
        Number of points on the spline.  Defaults to the number of
        points in ``points``.

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
    >>> r = z ** 2 + 1
    >>> x = r * np.sin(theta)
    >>> y = r * np.cos(theta)
    >>> points = np.column_stack((x, y, z))
    >>> kochanek_spline = pv.KochanekSpline(points, n_points=6)
    >>> kochanek_spline.plot(line_width=4, color="k")

    See :ref:`create_kochanek_spline_example` for an additional example.

    """
    if tension is None:
        tension = np.array([0.0, 0.0, 0.0])
    check_valid_vector(tension, "tension")
    if not np.all(np.abs(tension) <= 1.0):
        raise ValueError("The absolute value of all values of the tension array elements must be <= 1.0 ")

    if bias is None:
        bias = np.array([0.0, 0.0, 0.0])
    check_valid_vector(bias, "bias")
    if not np.all(np.abs(bias) <= 1.0):
        raise ValueError("The absolute value of all values of the bias array elements must be <= 1.0 ")

    if continuity is None:
        continuity = np.array([0.0, 0.0, 0.0])
    check_valid_vector(continuity, "continuity")
    if not np.all(np.abs(continuity) <= 1.0):
        raise ValueError(
            "The absolute value of all values continuity array elements must be <= 1.0 "
        )

    spline_function = _vtk.vtkParametricSpline()
    spline_function.SetPoints(pyvista.vtk_points(points, False))

    # set Kochanek spline for each direction
    xspline = _vtk.vtkKochanekSpline()
    yspline = _vtk.vtkKochanekSpline()
    zspline = _vtk.vtkKochanekSpline()
    xspline.SetDefaultBias(bias[0])
    yspline.SetDefaultBias(bias[1])
    zspline.SetDefaultBias(bias[2])
    xspline.SetDefaultTension(tension[0])
    yspline.SetDefaultTension(tension[1])
    zspline.SetDefaultTension(tension[2])
    xspline.SetDefaultContinuity(continuity[0])
    yspline.SetDefaultContinuity(continuity[1])
    zspline.SetDefaultContinuity(continuity[2])
    spline_function.SetXSpline(xspline)
    spline_function.SetYSpline(yspline)
    spline_function.SetZSpline(zspline)

    # get interpolation density
    u_res = n_points
    if u_res is None:
        u_res = points.shape[0]

    u_res -= 1
    spline = surface_from_para(spline_function, u_res)
    return spline.compute_arc_length()

def ParametricBohemianDome(a=None, **kwargs):
    """Generate a Bohemian dome surface.

    Parameters
    ----------
    a : float, optional
        Bohemian dome surface parameter.

    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricBohemianDome surface.

    Examples
    --------
    Create a ParametricBohemianDome mesh.

    >>> import pyvista
    >>> mesh = pyvista.ParametricBohemianDome()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricBohemianDome()
    if a is not None:
        parametric_function.SetA(a)

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    kwargs.setdefault('clean', True)
    surf = surface_from_para(parametric_function, **kwargs)
    translate(surf, center, direction)
    return surf


def ParametricBour(**kwargs):
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

    >>> import pyvista
    >>> mesh = pyvista.ParametricBour()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricBour()

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricBoy(zscale=None, **kwargs):
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

    >>> import pyvista
    >>> mesh = pyvista.ParametricBoy()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricBoy()
    if zscale is not None:
        parametric_function.SetZScale(zscale)

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    kwargs.setdefault('clean', True)
    surf = surface_from_para(parametric_function, **kwargs)
    translate(surf, center, direction)
    return surf


def ParametricCatalanMinimal(**kwargs):
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

    >>> import pyvista
    >>> mesh = pyvista.ParametricCatalanMinimal()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricCatalanMinimal()

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricConicSpiral(a=None, b=None, c=None, n=None, **kwargs):
    """Generate conic spiral surfaces that resemble sea-shells.

    ParametricConicSpiral generates conic spiral surfaces. These can
    resemble sea shells, or may look like a torus "eating" its own
    tail.

    Parameters
    ----------
    a : float, optional
        The scale factor.
        Default is 0.2.

    b : float, optional
        The A function coefficient.
        See the definition in Parametric surfaces referred to above.
        Default is 1.

    c : float, optional
        The B function coefficient.
        See the definition in Parametric surfaces referred to above.
        Default is 0.1.

    n : float, optional
        The C function coefficient.
        See the definition in Parametric surfaces referred to above.
        Default is 2.

    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricConicSpiral surface.

    Examples
    --------
    Create a ParametricConicSpiral mesh.

    >>> import pyvista
    >>> mesh = pyvista.ParametricConicSpiral()
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

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricCrossCap(**kwargs):
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

    >>> import pyvista
    >>> mesh = pyvista.ParametricCrossCap()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricCrossCap()

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricDini(a=None, b=None, **kwargs):
    """Generate Dini's surface.

    ParametricDini generates Dini's surface.  Dini's surface is a
    surface that possesses constant negative Gaussian curvature

    Parameters
    ----------
    a : float, optional
        The scale factor.  See the definition in Parametric surfaces
        referred to above.  Default is 1.

    b : float, optional
        The scale factor.  See the definition in Parametric surfaces
        referred to above.  Default is 0.2.

    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricDini surface.

    Examples
    --------
    Create a ParametricDini mesh.

    >>> import pyvista
    >>> mesh = pyvista.ParametricDini()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricDini()
    if a is not None:
        parametric_function.SetA(a)

    if b is not None:
        parametric_function.SetB(b)

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricEllipsoid(xradius=None, yradius=None, zradius=None,
                        **kwargs):
    """Generate an ellipsoid.

    ParametricEllipsoid generates an ellipsoid.  If all the radii are
    the same, we have a sphere.  An oblate spheroid occurs if RadiusX
    = RadiusY > RadiusZ.  Here the Z-axis forms the symmetry axis. To
    a first approximation, this is the shape of the earth.  A prolate
    spheroid occurs if RadiusX = RadiusY < RadiusZ.

    Parameters
    ----------
    xradius : float, optional
        The scaling factor for the x-axis. Default is 1.

    yradius : float, optional
        The scaling factor for the y-axis. Default is 1.

    zradius : float, optional
        The scaling factor for the z-axis. Default is 1.

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

    >>> import pyvista
    >>> mesh = pyvista.ParametricEllipsoid()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricEllipsoid()
    parametric_keywords(parametric_function, min_u=kwargs.pop("min_u", 0),
                        max_u=kwargs.pop("max_u", 2*pi),
                        min_v=kwargs.pop("min_v", 0.0),
                        max_v=kwargs.pop("max_v", pi),
                        join_u=kwargs.pop("join_u", False),
                        join_v=kwargs.pop("join_v", False),
                        twist_u=kwargs.pop("twist_u", False),
                        twist_v=kwargs.pop("twist_v", False),
                        clockwise=kwargs.pop("clockwise", True),)

    if xradius is not None:
        parametric_function.SetXRadius(xradius)

    if yradius is not None:
        parametric_function.SetYRadius(yradius)

    if zradius is not None:
        parametric_function.SetZRadius(zradius)

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    kwargs.setdefault('clean', True)
    surf = surface_from_para(parametric_function, **kwargs)
    translate(surf, center, direction)
    return surf


def ParametricEnneper(**kwargs):
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

    >>> import pyvista
    >>> mesh = pyvista.ParametricEnneper()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricEnneper()

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricFigure8Klein(radius=None, **kwargs):
    """Generate a figure-8 Klein bottle.

    ParametricFigure8Klein generates a figure-8 Klein bottle.  A Klein
    bottle is a closed surface with no interior and only one surface.
    It is unrealisable in 3 dimensions without intersecting surfaces.

    Parameters
    ----------
    radius : float, optional
        The radius of the bottle. Default is 1.
    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricFigure8Klein surface.

    Examples
    --------
    Create a ParametricFigure8Klein mesh.

    >>> import pyvista
    >>> mesh = pyvista.ParametricFigure8Klein()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricFigure8Klein()
    if radius is not None:
        parametric_function.SetRadius(radius)

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    kwargs.setdefault('clean', True)
    surf = surface_from_para(parametric_function, **kwargs)
    translate(surf, center, direction)
    return surf


def ParametricHenneberg(**kwargs):
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

    >>> import pyvista
    >>> mesh = pyvista.ParametricHenneberg()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricHenneberg()

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricKlein(**kwargs):
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

    >>> import pyvista
    >>> mesh = pyvista.ParametricKlein()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricKlein()

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    kwargs.setdefault('clean', True)
    surf = surface_from_para(parametric_function, **kwargs)
    translate(surf, center, direction)
    return surf


def ParametricKuen(deltav0=None, **kwargs):
    """Generate Kuens' surface.

    ParametricKuen generates Kuens' surface. This surface has a constant
    negative gaussian curvature.

    Parameters
    ----------
    deltav0 : float, optional
        The value to use when ``V == 0``.
        Default is 0.05, giving the best appearance with the default settings.
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

    >>> import pyvista
    >>> mesh = pyvista.ParametricKuen()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricKuen()
    if deltav0 is not None:
        parametric_function.SetDeltaV0(deltav0)

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    kwargs.setdefault('clean', True)
    surf = surface_from_para(parametric_function, **kwargs)
    translate(surf, center, direction)
    return surf


def ParametricMobius(radius=None, **kwargs):
    """Generate a Mobius strip.

    Parameters
    ----------
    radius : float, optional
        The radius of the Mobius strip. Default is 1.

    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricMobius surface.

    Examples
    --------
    Create a ParametricMobius mesh.

    >>> import pyvista
    >>> mesh = pyvista.ParametricMobius()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricMobius()
    if radius is not None:
        parametric_function.SetRadius(radius)

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    surf = surface_from_para(parametric_function, **kwargs)
    translate(surf, center, direction)
    return surf


def ParametricPluckerConoid(n=None, **kwargs):
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

    >>> import pyvista
    >>> mesh = pyvista.ParametricPluckerConoid()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricPluckerConoid()
    if n is not None:
        parametric_function.SetN(n)

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricPseudosphere(**kwargs):
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

    >>> import pyvista
    >>> mesh = pyvista.ParametricPseudosphere()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricPseudosphere()

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    kwargs.setdefault('clean', True)
    surf = surface_from_para(parametric_function, **kwargs)
    translate(surf, center, direction)
    return surf


def ParametricRandomHills(numberofhills=None, hillxvariance=None,
                          hillyvariance=None, hillamplitude=None,
                          randomseed=None, xvariancescalefactor=None,
                          yvariancescalefactor=None,
                          amplitudescalefactor=None, **kwargs):
    """Generate a surface covered with randomly placed hills.

    ParametricRandomHills generates a surface covered with randomly
    placed hills. Hills will vary in shape and height since the
    presence of nearby hills will contribute to the shape and height
    of a given hill.  An option is provided for placing hills on a
    regular grid on the surface.  In this case the hills will all have
    the same shape and height.

    Parameters
    ----------
    numberofhills : int, optional
        The number of hills.
        Default is 30.

    hillxvariance : float, optional
        The hill variance in the x-direction.
        Default is 2.5.

    hillyvariance : float, optional
        The hill variance in the y-direction.
        Default is 2.5.

    hillamplitude : float, optional
        The hill amplitude (height).
        Default is 2.

    randomseed : int, optional
        The Seed for the random number generator,
        a value of 1 will initialize the random number generator,
        a negative value will initialize it with the system time.
        Default is 1.

    xvariancescalefactor : float, optional
        The scaling factor for the variance in the x-direction.
        Default is 13.

    yvariancescalefactor : float, optional
        The scaling factor for the variance in the y-direction.
        Default is 13.

    amplitudescalefactor : float, optional
        The scaling factor for the amplitude.
        Default is 13.

    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricRandomHills surface.

    Examples
    --------
    Create a ParametricRandomHills mesh.

    >>> import pyvista
    >>> mesh = pyvista.ParametricRandomHills()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricRandomHills()
    if numberofhills is not None:
        parametric_function.SetNumberOfHills(numberofhills)

    if hillxvariance is not None:
        parametric_function.SetHillXVariance(hillxvariance)

    if hillyvariance is not None:
        parametric_function.SetHillYVariance(hillyvariance)

    if hillamplitude is not None:
        parametric_function.SetHillAmplitude(hillamplitude)

    if randomseed is not None:
        parametric_function.SetRandomSeed(randomseed)

    if xvariancescalefactor is not None:
        parametric_function.SetXVarianceScaleFactor(xvariancescalefactor)

    if yvariancescalefactor is not None:
        parametric_function.SetYVarianceScaleFactor(yvariancescalefactor)

    if amplitudescalefactor is not None:
        parametric_function.SetAmplitudeScaleFactor(amplitudescalefactor)

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricRoman(radius=None, **kwargs):
    """Generate Steiner's Roman Surface.

    Parameters
    ----------
    radius : float, optional
        The radius. Default is 1.

    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricRoman surface.

    Examples
    --------
    Create a ParametricRoman mesh.

    >>> import pyvista
    >>> mesh = pyvista.ParametricRoman()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricRoman()
    if radius is not None:
        parametric_function.SetRadius(radius)

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricSuperEllipsoid(xradius=None, yradius=None, zradius=None,
                             n1=None, n2=None, **kwargs):
    """Generate a superellipsoid.

    ParametricSuperEllipsoid generates a superellipsoid.  A superellipsoid
    is a versatile primitive that is controlled by two parameters n1 and
    n2. As special cases it can represent a sphere, square box, and closed
    cylindrical can.

    Parameters
    ----------
    xradius : float, optional
        The scaling factor for the x-axis. Default is 1.

    yradius : float, optional
        The scaling factor for the y-axis. Default is 1.

    zradius : float, optional
        The scaling factor for the z-axis. Default is 1.

    n1 : float, optional
        The "squareness" parameter in the z axis.  Default is 1.

    n2 : float, optional
        The "squareness" parameter in the x-y plane. Default is 1.

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

    >>> import pyvista
    >>> mesh = pyvista.ParametricSuperEllipsoid(n1=0.02, n2=0.02)
    >>> mesh.plot(color='w', smooth_shading=True)

    Create one that looks like a spinning top.

    >>> mesh = pyvista.ParametricSuperEllipsoid(n1=4, n2=0.5)
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

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricSuperToroid(ringradius=None, crosssectionradius=None,
                          xradius=None, yradius=None, zradius=None,
                          n1=None, n2=None, **kwargs):
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
    ringradius : float, optional
        The radius from the center to the middle of the ring of the
        supertoroid. Default is 1.

    crosssectionradius : float, optional
        The radius of the cross section of ring of the supertoroid.
        Default = 0.5.

    xradius : float, optional
        The scaling factor for the x-axis. Default is 1.

    yradius : float, optional
        The scaling factor for the y-axis. Default is 1.

    zradius : float, optional
        The scaling factor for the z-axis. Default is 1.

    n1 : float, optional
        The shape of the torus ring.  Default is 1.

    n2 : float, optional
        The shape of the cross section of the ring. Default is 1.

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

    >>> import pyvista
    >>> mesh = pyvista.ParametricSuperToroid(n1=2, n2=0.3)
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

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    kwargs.setdefault('clean', True)
    surf = surface_from_para(parametric_function, **kwargs)
    translate(surf, center, direction)
    return surf


def ParametricTorus(ringradius=None, crosssectionradius=None, **kwargs):
    """Generate a torus.

    Parameters
    ----------
    ringradius : float, optional
        The radius from the center to the middle of the ring of the
        torus. Default is 1.0.

    crosssectionradius : float, optional
        The radius of the cross section of ring of the torus. Default is 0.5.

    **kwargs : dict, optional
        See :func:`surface_from_para` for additional keyword arguments.

    Returns
    -------
    pyvista.PolyData
        ParametricTorus surface.

    Examples
    --------
    Create a ParametricTorus mesh.

    >>> import pyvista
    >>> mesh = pyvista.ParametricTorus()
    >>> mesh.plot(color='w', smooth_shading=True)

    """
    parametric_function = _vtk.vtkParametricTorus()
    if ringradius is not None:
        parametric_function.SetRingRadius(ringradius)

    if crosssectionradius is not None:
        parametric_function.SetCrossSectionRadius(crosssectionradius)

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    kwargs.setdefault('clean', True)
    surf = surface_from_para(parametric_function, **kwargs)
    translate(surf, center, direction)
    return surf


def parametric_keywords(parametric_function, min_u=0, max_u=2*pi,
                        min_v=0.0, max_v=2*pi, join_u=False, join_v=False,
                        twist_u=False, twist_v=False, clockwise=True):
    """Apply keyword arguments to a parametric function.

    Parameters
    ----------
    parametric_function : vtk.vtkParametricFunction
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


def surface_from_para(parametric_function, u_res=100, v_res=100, w_res=100,
                      clean=False):
    """Construct a mesh from a parametric function.

    Parameters
    ----------
    parametric_function : vtk.vtkParametricFunction
        Parametric function to generate mesh from.

    u_res : int, optional
        Resolution in the u direction.

    v_res : int, optional
        Resolution in the v direction.

    w_res : int, optional
        Resolution in the w direction.

    clean : bool, optional
        Clean and merge duplicate points to avoid "creases" when
        plotting with smooth shading.

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
    para_source.Update()
    surf = pyvista.wrap(para_source.GetOutput())
    if clean:
        surf = surf.clean(
            tolerance=1E-7,  # determined experimentally
            absolute=False,
            lines_to_points=False,
            polys_to_lines=False,
            strips_to_polys=False,
        )
    return surf
