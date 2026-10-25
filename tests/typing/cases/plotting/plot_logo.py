"""Typing cases for :func:`pyvista.demos.plot_logo`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista import demos

assert_types(demos.plot_logo(just_return_plotter=True), pv.Plotter)
