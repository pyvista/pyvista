"""
.. _create_parametric_geometric_objects_example:

Parametric Geometric Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating parametric objects from :ref:`parametric_api`.
"""

# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pyvista",
# ]
# ///

from __future__ import annotations

from math import pi

# sphinx_gallery_thumbnail_number = 12
import pyvista as pv

# %%
# This example demonstrates how to plot parametric objects using pyvista
#
# Supertoroid
# +++++++++++

supertoroid = pv.ParametricSuperToroid(n1=0.5)
supertoroid.plot(color='lightblue', smooth_shading=True)

# %%
# Parametric Ellipsoid
# ++++++++++++++++++++

# Ellipsoid with a long x axis
ellipsoid = pv.ParametricEllipsoid(10, 5, 5)
ellipsoid.plot(color='lightblue')


# %%
# Partial Parametric Ellipsoid
# ++++++++++++++++++++++++++++

# cool plotting direction
cpos = pv.CameraPosition(
    position=(21.9930, 21.1810, -30.3780),
    focal_point=(-1.1640, -1.3098, -0.1061),
    viewup=(0.8498, -0.2515, 0.4631),
)


# half ellipsoid
part_ellipsoid = pv.ParametricEllipsoid(10, 5, 5, max_v=pi / 2)
part_ellipsoid.plot(color='lightblue', smooth_shading=True, cpos=cpos)


# %%
# Pseudosphere
# ++++++++++++

pseudosphere = pv.ParametricPseudosphere()
pseudosphere.plot(color='lightblue', smooth_shading=True)

# %%
# Bohemian Dome
# +++++++++++++


bohemiandome = pv.ParametricBohemianDome()
bohemiandome.plot(color='lightblue')

# %%
# Bour
# ++++

bour = pv.ParametricBour()
bour.plot(color='lightblue')

# %%
# Boy's Surface
# +++++++++++++

boy = pv.ParametricBoy()
boy.plot(color='lightblue')

# %%
# Catalan Minimal
# +++++++++++++++

catalanminimal = pv.ParametricCatalanMinimal()
catalanminimal.plot(color='lightblue')

# %%
# Conic Spiral
# ++++++++++++

conicspiral = pv.ParametricConicSpiral()
conicspiral.plot(color='lightblue')

# %%
# Cross Cap
# +++++++++

crosscap = pv.ParametricCrossCap()
crosscap.plot(color='lightblue')

# %%
# Dini
# ++++

dini = pv.ParametricDini()
dini.plot(color='lightblue')

# %%
# Enneper
# +++++++

enneper = pv.ParametricEnneper()
enneper.plot(cpos='yz')

# %%
# Figure-8 Klein
# ++++++++++++++

figure8klein = pv.ParametricFigure8Klein()
figure8klein.plot()

# %%
# Henneberg
# +++++++++

henneberg = pv.ParametricHenneberg()
henneberg.plot(color='lightblue')

# %%
# Klein
# +++++

klein = pv.ParametricKlein()
klein.plot(color='lightblue')

# %%
# Kuen
# ++++

kuen = pv.ParametricKuen()
kuen.plot(color='lightblue')

# %%
# Mobius
# ++++++

mobius = pv.ParametricMobius()
mobius.plot(color='lightblue')

# %%
# Plucker Conoid
# ++++++++++++++

pluckerconoid = pv.ParametricPluckerConoid()
pluckerconoid.plot(color='lightblue')


# %%
# Random Hills
# ++++++++++++

randomhills = pv.ParametricRandomHills()
randomhills.plot(color='lightblue')

# %%
# Roman
# +++++

roman = pv.ParametricRoman()
roman.plot(color='lightblue')

# %%
# Super Ellipsoid
# +++++++++++++++

superellipsoid = pv.ParametricSuperEllipsoid(n1=0.1, n2=2)
superellipsoid.plot(color='lightblue')

# %%
# Torus
# +++++

torus = pv.ParametricTorus()
torus.plot(color='lightblue')

# %%
# Circular Arc
# ++++++++++++

pointa = [-1, 0, 0]
pointb = [0, 1, 0]
center = [0, 0, 0]
resolution = 100

arc = pv.CircularArc(pointa=pointa, pointb=pointb, center=center, resolution=resolution)

pl = pv.Plotter()
pl.add_mesh(arc, color='k', line_width=4)
pl.show_bounds()
pl.view_xy()
pl.show()


# %%
# Extruded Half Arc
# +++++++++++++++++

pointa = [-1, 0, 0]
pointb = [1, 0, 0]
center = [0, 0, 0]
resolution = 100

arc = pv.CircularArc(pointa=pointa, pointb=pointb, center=center, resolution=resolution)
poly = arc.extrude([0, 0, 1])
poly.plot(color='lightblue', cpos='iso', show_edges=True)
# %%
# .. tags:: load
