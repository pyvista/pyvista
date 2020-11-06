"""
Spline Widget
~~~~~~~~~~~~~


A spline widget can be enabled and disabled by the
:func:`pyvista.WidgetHelper.add_spline_widget` and
:func:`pyvista.WidgetHelper.clear_spline_widgets` methods respectively.
This widget allows users to interactively create a poly line (spline) through
a scene and use that spline.

A common task with splines is to slice a volumetric dataset using an irregular
path. To do this, we have added a convenient helper method which leverages the
:func:`pyvista.DataSetFilters.slice_along_line` filter named
:func:`pyvista.WidgetHelper.add_mesh_slice_spline`.
"""
