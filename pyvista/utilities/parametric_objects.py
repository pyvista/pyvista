"""Module managing parametric objects."""

from math import pi

import vtk

import pyvista
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

    Return
    ------
    spline : pyvista.PolyData
        Line mesh of spline.

    Examples
    --------
    Construct a spline

    >>> import numpy as np
    >>> import pyvista as pv
    >>> theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    >>> z = np.linspace(-2, 2, 100)
    >>> r = z**2 + 1
    >>> x = r * np.sin(theta)
    >>> y = r * np.cos(theta)
    >>> points = np.column_stack((x, y, z))
    >>> spline = pv.Spline(points, 1000)

    """
    spline_function = vtk.vtkParametricSpline()
    spline_function.SetPoints(pyvista.vtk_points(points, False))

    # get interpolation density
    u_res = n_points
    if u_res is None:
        u_res = points.shape[0]

    u_res -= 1
    spline = surface_from_para(spline_function, u_res)
    return spline.compute_arc_length()


def ParametricBohemianDome(a=None, **kwargs):
    """Generate a Bohemian dome.

    Parameters
    ----------
    a : double, optional
        Construct a Bohemian dome surface with the following parameters:

    vtkGetMacro(A, double);

    Return
    ------
    surf : pyvista.PolyData
        ParametricBohemianDome surface

    Examples
    --------
    Create a ParametricBohemianDome mesh

    >>> import pyvista
    >>> mesh = pyvista.ParametricBohemianDome()
    >>> mesh.plot(color='w', smooth_shading=True)  # doctest:+SKIP

    """
    parametric_function = vtk.vtkParametricBohemianDome()
    if a is not None:
        parametric_function.SetA(a)

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricBour(**kwargs):
    """Generate Bour's minimal surface.

    Return
    ------
    surf : pyvista.PolyData
        ParametricBour surface

    Examples
    --------
    Create a ParametricBour mesh

    >>> import pyvista
    >>> mesh = pyvista.ParametricBour()
    >>> mesh.plot(color='w', smooth_shading=True)  # doctest:+SKIP

    """
    parametric_function = vtk.vtkParametricBour()

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricBoy(zscale=None, **kwargs):
    """Generate Boy's surface.

    ParametricBoy generates Boy's surface.
    This is a Model of the projective plane without singularities.
    It was found by Werner Boy on assignment from David Hilbert.

    For further information about this surface, please consult the
    technical description "Parametric surfaces" in the
    "VTK Technical Documents" section in the VTK.org web pages.

    Parameters
    ----------
    zscale : double, optional
        The scale factor for the z-coordinate.
      Default is 18, giving a nice shape.

    Return
    ------
    surf : pyvista.PolyData
        ParametricBoy surface

    Examples
    --------
    Create a ParametricBoy mesh

    >>> import pyvista
    >>> mesh = pyvista.ParametricBoy()
    >>> mesh.plot(color='w', smooth_shading=True)  # doctest:+SKIP

    """
    parametric_function = vtk.vtkParametricBoy()
    if zscale is not None:
        parametric_function.SetZScale(zscale)

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricCatalanMinimal(**kwargs):
    """Generate Catalan's minimal surface.

    ParametricCatalanMinimal generates Catalan's minimal surface
    parametrically. This minimal surface contains the cycloid as a
    geodesic.

    Returns
    -------
    surf : pyvista.PolyData
        ParametricCatalanMinimal surface

    Example
    -------
    Create a ParametricCatalanMinimal mesh

    >>> import pyvista
    >>> mesh = pyvista.ParametricCatalanMinimal()
    >>> mesh.plot(color='w', smooth_shading=True)  # doctest:+SKIP

    """
    parametric_function = vtk.vtkParametricCatalanMinimal()

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
    a : double, optional
        The scale factor.
        Default is 0.2

    b : double, optional
        The A function coefficient.
        See the definition in Parametric surfaces referred to above.
        Default is 1.

    c : double, optional
        The B function coefficient.
        See the definition in Parametric surfaces referred to above.
        Default is 0.1.

    n : double, optional
        The C function coefficient.
        See the definition in Parametric surfaces referred to above.
        Default is 2.

    Return
    ------
    surf : pyvista.PolyData
        ParametricConicSpiral surface

    Examples
    --------
    Create a ParametricConicSpiral mesh

    >>> import pyvista
    >>> mesh = pyvista.ParametricConicSpiral()
    >>> mesh.plot(color='w', smooth_shading=True)  # doctest:+SKIP

    """
    parametric_function = vtk.vtkParametricConicSpiral()
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

    Return
    ------
    surf : pyvista.PolyData
        ParametricCrossCap surface

    Examples
    --------
    Create a ParametricCrossCap mesh

    >>> import pyvista
    >>> mesh = pyvista.ParametricCrossCap()
    >>> mesh.plot(color='w', smooth_shading=True)  # doctest:+SKIP

    """
    parametric_function = vtk.vtkParametricCrossCap()

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
    a : double, optional
        The scale factor.
        See the definition in Parametric surfaces referred to above.
        Default is 1.

    b : double, optional
        The scale factor.
        See the definition in Parametric surfaces referred to above.
        Default is 0.2

    Return
    ------
    surf : pyvista.PolyData
        ParametricDini surface

    Examples
    --------
    Create a ParametricDini mesh

    >>> import pyvista
    >>> mesh = pyvista.ParametricDini()
    >>> mesh.plot(color='w', smooth_shading=True)  # doctest:+SKIP

    """
    parametric_function = vtk.vtkParametricDini()
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
    xradius : double, optional
        The scaling factor for the x-axis. Default is 1.

    yradius : double, optional
        The scaling factor for the y-axis. Default is 1.

    zradius : double, optional
        The scaling factor for the z-axis. Default is 1.

    Return
    ------
    surf : pyvista.PolyData
        ParametricEllipsoid surface

    Examples
    --------
    Create a ParametricEllipsoid mesh

    >>> import pyvista
    >>> mesh = pyvista.ParametricEllipsoid()
    >>> mesh.plot(color='w', smooth_shading=True)  # doctest:+SKIP

    """
    parametric_function = vtk.vtkParametricEllipsoid()
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
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricEnneper(**kwargs):
    """Generate Enneper's surface.

    ParametricEnneper generates Enneper's surface.
    Enneper's surface is a a self-intersecting minimal surface
    possessing constant negative Gaussian curvature

    Return
    ------
    surf : pyvista.PolyData
        ParametricEnneper surface

    Examples
    --------
    Create a ParametricEnneper mesh

    >>> import pyvista
    >>> mesh = pyvista.ParametricEnneper()
    >>> mesh.plot(color='w', smooth_shading=True)  # doctest:+SKIP

    """
    parametric_function = vtk.vtkParametricEnneper()

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
    radius : double, optional
        The radius of the bottle. Default is 1.

    Return
    ------
    surf : pyvista.PolyData
        ParametricFigure8Klein surface

    Examples
    --------
    Create a ParametricFigure8Klein mesh

    >>> import pyvista
    >>> mesh = pyvista.ParametricFigure8Klein()
    >>> mesh.plot(color='w', smooth_shading=True)  # doctest:+SKIP

    """
    parametric_function = vtk.vtkParametricFigure8Klein()
    if radius is not None:
        parametric_function.SetRadius(radius)

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricHenneberg(**kwargs):
    """Generate Henneberg's minimal surface.

    Return
    ------
    surf : pyvista.PolyData
        ParametricHenneberg surface

    Examples
    --------
    Create a ParametricHenneberg mesh

    >>> import pyvista
    >>> mesh = pyvista.ParametricHenneberg()
    >>> mesh.plot(color='w', smooth_shading=True)  # doctest:+SKIP

    """
    parametric_function = vtk.vtkParametricHenneberg()

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

    Return
    ------
    surf : pyvista.PolyData
        ParametricKlein surface

    Examples
    --------
    Create a ParametricKlein mesh

    >>> import pyvista
    >>> mesh = pyvista.ParametricKlein()
    >>> mesh.plot(color='w', smooth_shading=True)  # doctest:+SKIP

    """
    parametric_function = vtk.vtkParametricKlein()

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricKuen(deltav0=None, **kwargs):
    """Generate Kuens' surface.

    ParametricKuen generates Kuens' surface. This surface has a constant
    negative gaussian curvature.

    Parameters
    ----------
    deltav0 : double, optional
        The value to use when V == 0.
        Default is 0.05, giving the best appearance with the default settings.
        Setting it to a value less than 0.05 extrapolates the surface
        towards a pole in the -z direction.
        Setting it to 0 retains the pole whose z-value is -inf.

    Return
    ------
    surf : pyvista.PolyData
        ParametricKuen surface

    Examples
    --------
    Create a ParametricKuen mesh

    >>> import pyvista
    >>> mesh = pyvista.ParametricKuen()
    >>> mesh.plot(color='w', smooth_shading=True)  # doctest:+SKIP

    """
    parametric_function = vtk.vtkParametricKuen()
    if deltav0 is not None:
        parametric_function.SetDeltaV0(deltav0)

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricMobius(radius=None, **kwargs):
    """Generate a Mobius strip.

    Parameters
    ----------
    radius : double, optional
        The radius of the Mobius strip. Default is 1.

    Return
    ------
    surf : pyvista.PolyData
        ParametricMobius surface

    Examples
    --------
    Create a ParametricMobius mesh

    >>> import pyvista
    >>> mesh = pyvista.ParametricMobius()
    >>> mesh.plot(color='w', smooth_shading=True)  # doctest:+SKIP

    """
    parametric_function = vtk.vtkParametricMobius()
    if radius is not None:
        parametric_function.SetRadius(radius)

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricPluckerConoid(n=None, **kwargs):
    """Generate Plucker's conoid surface.

    ParametricPluckerConoid generates Plucker's conoid surface parametrically.
    Plucker's conoid is a ruled surface, named after Julius Plucker. It is
    possible to set the number of folds in this class via the parameter 'N'.

    Parameters
    ----------
    n : int, optional
        This is the number of folds in the conoid.

    vtkGetMacro(N, int);

    Return
    ------
    surf : pyvista.PolyData
        ParametricPluckerConoid surface

    Examples
    --------
    Create a ParametricPluckerConoid mesh

    >>> import pyvista
    >>> mesh = pyvista.ParametricPluckerConoid()
    >>> mesh.plot(color='w', smooth_shading=True)  # doctest:+SKIP

    """
    parametric_function = vtk.vtkParametricPluckerConoid()
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

    Return
    ------
    surf : pyvista.PolyData
        ParametricPseudosphere surface

    Examples
    --------
    Create a ParametricPseudosphere mesh

    >>> import pyvista
    >>> mesh = pyvista.ParametricPseudosphere()
    >>> mesh.plot(color='w', smooth_shading=True)  # doctest:+SKIP

    """
    parametric_function = vtk.vtkParametricPseudosphere()

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
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

    hillxvariance : double, optional
        The hill variance in the x-direction.
        Default is 2.5.

    hillyvariance : double, optional
        The hill variance in the y-direction.
        Default is 2.5.

    hillamplitude : double, optional
        The hill amplitude (height).
        Default is 2.

    randomseed : int, optional
        The Seed for the random number generator,
        a value of 1 will initialize the random number generator,
        a negative value will initialize it with the system time.
        Default is 1.

    xvariancescalefactor : double, optional
        The scaling factor for the variance in the x-direction.
        Default is 13.

    yvariancescalefactor : double, optional
        The scaling factor for the variance in the y-direction.
        Default is 13.

    amplitudescalefactor : double, optional
        The scaling factor for the amplitude.
        Default is 13.

    Return
    ------
    surf : pyvista.PolyData
        ParametricRandomHills surface

    Examples
    --------
    Create a ParametricRandomHills mesh

    >>> import pyvista
    >>> mesh = pyvista.ParametricRandomHills()
    >>> mesh.plot(color='w', smooth_shading=True)  # doctest:+SKIP

    """
    parametric_function = vtk.vtkParametricRandomHills()
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
    radius : double, optional
        The radius. Default is 1.

    Return
    ------
    surf : pyvista.PolyData
        ParametricRoman surface

    Examples
    --------
    Create a ParametricRoman mesh

    >>> import pyvista
    >>> mesh = pyvista.ParametricRoman()
    >>> mesh.plot(color='w', smooth_shading=True)  # doctest:+SKIP

    """
    parametric_function = vtk.vtkParametricRoman()
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
    xradius : double, optional
        The scaling factor for the x-axis. Default is 1.

    yradius : double, optional
        The scaling factor for the y-axis. Default is 1.

    zradius : double, optional
        The scaling factor for the z-axis. Default is 1.

    n1 : double, optional
        The "squareness" parameter in the z axis.  Default is 1.

    n2 : double, optional
        The "squareness" parameter in the x-y plane. Default is 1.

    Return
    ------
    surf : pyvista.PolyData
        ParametricSuperEllipsoid surface

    Examples
    --------
    Create a ParametricSuperEllipsoid mesh

    >>> import pyvista
    >>> mesh = pyvista.ParametricSuperEllipsoid()
    >>> mesh.plot(color='w', smooth_shading=True)  # doctest:+SKIP

    """
    parametric_function = vtk.vtkParametricSuperEllipsoid()
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
    ringradius : double, optional
        The radius from the center to the middle of the ring of the
      supertoroid. Default is 1.

    crosssectionradius : double, optional
        The radius of the cross section of ring of the supertoroid.
      Default = 0.5.

    xradius : double, optional
        The scaling factor for the x-axis. Default is 1.

    yradius : double, optional
        The scaling factor for the y-axis. Default is 1.

    zradius : double, optional
        The scaling factor for the z-axis. Default is 1.

    n1 : double, optional
        The shape of the torus ring.  Default is 1.

    n2 : double, optional
        The shape of the cross section of the ring. Default is 1.

    Return
    ------
    surf : pyvista.PolyData
        ParametricSuperToroid surface

    Examples
    --------
    Create a ParametricSuperToroid mesh

    >>> import pyvista
    >>> mesh = pyvista.ParametricSuperToroid()
    >>> mesh.plot(color='w', smooth_shading=True)  # doctest:+SKIP

    """
    parametric_function = vtk.vtkParametricSuperToroid()
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
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def ParametricTorus(ringradius=None, crosssectionradius=None, **kwargs):
    """Generate a torus.

    Parameters
    ----------
    ringradius : double, optional
        The radius from the center to the middle of the ring of the
        torus. Default is 1.0.

    crosssectionradius : double, optional
        The radius of the cross section of ring of the torus. Default is 0.5.

    Return
    ------
    surf : pyvista.PolyData
        ParametricTorus surface

    Examples
    --------
    Create a ParametricTorus mesh

    >>> import pyvista
    >>> mesh = pyvista.ParametricTorus()
    >>> mesh.plot(color='w', smooth_shading=True)  # doctest:+SKIP

    """
    parametric_function = vtk.vtkParametricTorus()
    if ringradius is not None:
        parametric_function.SetRingRadius(ringradius)

    if crosssectionradius is not None:
        parametric_function.SetCrossSectionRadius(crosssectionradius)

    center = kwargs.pop('center', [0., 0., 0.])
    direction = kwargs.pop('direction', [1., 0., 0.])
    surf = surface_from_para(parametric_function, **kwargs)

    translate(surf, center, direction)

    return surf


def parametric_keywords(parametric_function, min_u=0, max_u=2*pi,
                        min_v=0.0, max_v=2*pi, join_u=False, join_v=False,
                        twist_u=False, twist_v=False, clockwise=True):
    """Apply keyword arguments to a parametric function.

    Parameters
    ----------
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
        joins the first triangle strip to the last one with a twist in
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


def surface_from_para(parametric_function, u_res=100, v_res=100,
                      w_res=100):
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

    """
    # convert to a mesh
    para_source = vtk.vtkParametricFunctionSource()
    para_source.SetParametricFunction(parametric_function)
    para_source.SetUResolution(u_res)
    para_source.SetVResolution(v_res)
    para_source.SetWResolution(w_res)
    para_source.Update()
    return pyvista.wrap(para_source.GetOutput())
