Parametric Objects
------------------

These objects represent surfaces that are parametrised by a set of independent
variables. Some of these are impossible to represent (correctly or at all)
using implicit functions, such as the Mobius strip.

The following functions can be used to create parametric surfaces. To
see additional examples, see :ref:`ref_parametric_example`.

.. currentmodule:: pyvista

.. autosummary::
   :toctree: _autosummary

   ParametricBohemianDome
   ParametricBour
   ParametricBoy
   ParametricDini
   ParametricCatalanMinimal
   ParametricConicSpiral
   ParametricCrossCap
   ParametricEllipsoid
   ParametricEnneper
   ParametricFigure8Klein
   ParametricHenneberg
   ParametricKlein
   ParametricKuen
   ParametricMobius
   ParametricPluckerConoid
   ParametricPseudosphere
   ParametricRandomHills
   ParametricRoman
   ParametricSuperEllipsoid
   ParametricSuperToroid
   ParametricTorus

These functions support building parametric surfaces:

.. autosummary::
   :toctree: _autosummary

   parametric_keywords
   surface_from_para
