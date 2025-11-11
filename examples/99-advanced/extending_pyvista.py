"""
.. _extending_pyvista_example:

Extending PyVista
~~~~~~~~~~~~~~~~~

A :class:`pyvista.DataSet`, such as :class:`pyvista.PolyData`, can be extended
by users.  For example, if the user wants to keep track of the location of the
maximum point in the (1, 0, 1) direction on the mesh.

There are two methods by which users can handle subclassing.  One is directly managing
the types objects.  This may require checking types during filter
operations.

The second is automatic managing of types.  Users can control whether user defined
classes are nearly always used for particular types of DataSets.

.. note::
    This is for advanced usage only.  Automatic managing of types
    will not work in all situations, in particular when a builtin dataset is directly
    instantiated.  See examples below.

"""

from __future__ import annotations

import numpy as np
import vtk

import pyvista as pv

pv.set_plot_theme('document')

# %%
# A user defined subclass of :class:`pyvista.PolyData`, ``FooData`` is defined.
# It includes a property to keep track of the point on the mesh that is
# furthest along in the (1, 0, 1) direction.


class FooData(pv.PolyData):  # noqa: D101
    @property
    def max_point(self):
        """Returns index of point that is furthest along (1, 0, 1) direction."""
        return np.argmax(np.dot(self.points, (1.0, 0.0, 1.0)))


# %%
# Directly Managing Types
# +++++++++++++++++++++++
#
# Now a ``foo_sphere`` object is created of type ``FooData``.
# The index of the point and location of the point of interest can be obtained
# directly. The sphere has a radius of 0.5, so the maximum extent in the
# direction (1, 0, 1) is :math:`0.5\sqrt{0.5}\approx0.354`
#

foo_sphere = FooData(pv.Sphere(theta_resolution=100, phi_resolution=100))
print('Original foo sphere:')
print(f'Type: {type(foo_sphere)}')
print(f'Maximum point index: {foo_sphere.max_point}')
print(f'Location of maximum point: {foo_sphere.points[foo_sphere.max_point, :]}')

# %%
# Using an inplace operation like :func:`~pyvista.DataObjectFilters.rotate_y` does not
# affect the type of the object.

foo_sphere.rotate_y(90, inplace=True)
print('\nRotated foo sphere:')
print(f'Type: {type(foo_sphere)}')
print(f'Maximum point index: {foo_sphere.max_point}')
print(f'Location of maximum point: {foo_sphere.points[foo_sphere.max_point, :]}')

# %%
# However, filter operations can return different ``DataSet`` types including
# ones that differ from the original type.  In this case, the
# :func:`decimate <pyvista.PolyDataFilters.decimate>` method returns a
# :class:`pyvista.PolyData` object.

print('\nDecimated foo sphere:')
decimated_foo_sphere = foo_sphere.decimate(0.5)
print(f'Type: {type(decimated_foo_sphere)}')

# %%
# It is now required to explicitly wrap the object into ``FooData``.

decimated_foo_sphere = FooData(foo_sphere.decimate(0.5))
print(f'Type: {type(decimated_foo_sphere)}')
print(f'Maximum point index: {decimated_foo_sphere.max_point}')
print(f'Location of maximum point: {foo_sphere.points[foo_sphere.max_point, :]}')

# %%
# Automatically Managing Types
# ++++++++++++++++++++++++++++
#
# The default :class:`pyvista.DataSet` type can be set using ``pyvista._wrappers``.
# In general, it is best to use this method when it is expected to primarily
# use the user defined class.
#
# In this example, all objects that would have been created as
# :class:`pyvista.PolyData` would now be created as a ``FooData`` object. Note,
# that the key is the underlying vtk object.

pv._wrappers['vtkPolyData'] = FooData

# %%
# It is no longer necessary to specifically wrap :class:`pyvista.PolyData`
# objects to obtain a ``FooData`` object.

foo_sphere = pv.Sphere(theta_resolution=100, phi_resolution=100)
print('Original foo sphere:')
print(f'Type: {type(foo_sphere)}')
print(f'Maximum point index: {foo_sphere.max_point}')
print(f'Location of maximum point: {foo_sphere.points[foo_sphere.max_point, :]}')

# %%
# Using an inplace operation like :func:`~pyvista.DataObjectFilters.rotate_y` does not
# affect the type of the object.

foo_sphere.rotate_y(90, inplace=True)
print('\nRotated foo sphere:')
print(f'Type: {type(foo_sphere)}')
print(f'Maximum point index: {foo_sphere.max_point}')
print(f'Location of maximum point: {foo_sphere.points[foo_sphere.max_point, :]}')

# %%
# Filter operations that return :class:`pyvista.PolyData` now return
# ``FooData``

print('\nDecimated foo sphere:')
decimated_foo_sphere = foo_sphere.decimate(0.5)
print(f'Type: {type(decimated_foo_sphere)}')
print(f'Maximum point index: {decimated_foo_sphere.max_point}')
print(f'Location of maximum point: {foo_sphere.points[foo_sphere.max_point, :]}')

# %%
# Users can still create a native :class:`pyvista.PolyData` object, but
# using this method may incur unintended consequences.  In this case,
# it is recommended to use the directly managing types method.

poly_object = pv.PolyData(vtk.vtkPolyData())
print(f'Type: {type(poly_object)}')
# catch error
try:
    poly_object.rotate_y(90, inplace=True)
except TypeError:
    print('This operation fails')

# %%
# Usage of ``pyvista._wrappers`` may require resetting the default value
# to avoid leaking the setting into cases where it is unused.

pv._wrappers['vtkPolyData'] = pv.PolyData

# %%
# For instances where a localized usage is preferred, a tear-down method is
# recommended.  One example is a ``try...finally`` block.

try:
    pv._wrappers['vtkPolyData'] = FooData
    # some operation that sometimes raises an error
finally:
    pv._wrappers['vtkPolyData'] = pv.PolyData
