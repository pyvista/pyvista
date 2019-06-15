"""
.. _ref_parametric_example:

Geometric Parametric Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating parametric objects
"""
import pyvista as pv
from math import pi

###############################################################################
# This example demonstrates how to plot parametric objects using pyvista
# Create a supertoroid

supertoroid = pv.ParametricSuperToroid(n1=0.5)
supertoroid.plot(color='tan', smooth_shading=True)

################################################################################
# Create a parametric ellipsoid

# Ellipsoid with a long x axis
ellipsoid = pv.ParametricEllipsoid(10, 5, 5)
ellipsoid.plot(color='tan')



################################################################################
# Create a partial parametric ellipsoid

# cool plotting direction
cpos = [(21.9930, 21.1810, -30.3780),
        (-1.1640, -1.3098, -0.1061),
        (0.8498, -0.2515, 0.4631)]


# half ellipsoid
part_ellipsoid = pv.ParametricEllipsoid(10, 5, 5, max_v=pi/2)
part_ellipsoid.plot(color='tan', smooth_shading=True, cpos=cpos)


################################################################################
# Create a pseudosphere

pseudosphere = pv.ParametricPseudosphere()
pseudosphere.plot(color='tan', smooth_shading=True)

###############################################################################
# Create a bohemiandome

bohemiandome = pv.ParametricBohemianDome()
bohemiandome.plot(color='tan')

###############################################################################
# Create a bour

bour = pv.ParametricBour()
bour.plot(color='tan')

###############################################################################
# Create a boy

boy = pv.ParametricBoy()
boy.plot(color='tan')

###############################################################################
# Create a catalanminimal

catalanminimal = pv.ParametricCatalanMinimal()
catalanminimal.plot(color='tan')

###############################################################################
# Create a conicspiral

conicspiral = pv.ParametricConicSpiral()
conicspiral.plot(color='tan')

###############################################################################
# Create a crosscap

crosscap = pv.ParametricCrossCap()
crosscap.plot(color='tan')

###############################################################################
# Create a dini

dini = pv.ParametricDini()
dini.plot(color='tan')

###############################################################################
# Create a enneper

enneper = pv.ParametricEnneper()
enneper.plot(color='tan')

###############################################################################
# Create a figure8klein

figure8klein = pv.ParametricFigure8Klein()
figure8klein.plot(color='tan')

###############################################################################
# Create a henneberg

henneberg = pv.ParametricHenneberg()
henneberg.plot(color='tan')

###############################################################################
# Create a klein

klein = pv.ParametricKlein()
klein.plot(color='tan')

###############################################################################
# Create a kuen

kuen = pv.ParametricKuen()
kuen.plot(color='tan')

###############################################################################
# Create a mobius

mobius = pv.ParametricMobius()
mobius.plot(color='tan')

###############################################################################
# Create a pluckerconoid

pluckerconoid = pv.ParametricPluckerConoid()
pluckerconoid.plot(color='tan')

###############################################################################
# Create a pseudosphere

pseudosphere = pv.ParametricPseudosphere()
pseudosphere.plot(color='tan')

###############################################################################
# Create a randomhills

randomhills = pv.ParametricRandomHills()
randomhills.plot(color='tan')

###############################################################################
# Create a roman

roman = pv.ParametricRoman()
roman.plot(color='tan')

###############################################################################
# Create a superellipsoid

superellipsoid = pv.ParametricSuperEllipsoid(n1=0.1, n2=2)
superellipsoid.plot(color='tan')

###############################################################################
# Create a torus

torus = pv.ParametricTorus()
torus.plot(color='tan')
