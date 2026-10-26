r"""
.. _actor_transformations_example:

Actor Transformations
~~~~~~~~~~~~~~~~~~~~~

Understand how the transform properties of an actor place it in a scene.

The properties are :attr:`~pyvista.Prop3D.position`,
:attr:`~pyvista.Prop3D.orientation`, :attr:`~pyvista.Prop3D.scale`,
:attr:`~pyvista.Prop3D.origin`, and :attr:`~pyvista.Prop3D.user_matrix`. Every
:class:`~pyvista.Prop3D`, such as :class:`~pyvista.Actor` or
:class:`~pyvista.AxesAssembly`, combines these five properties into a single
transformation. A point ``x`` of the actor's mesh is moved to

.. math::

    x' = M \left( R \, S \, (x - o) + o + p \right)

where :math:`o` is the ``origin``, :math:`p` is the ``position``, :math:`S` is the
``scale``, :math:`R` is the rotation defined by the ``orientation``, and :math:`M` is the
``user_matrix``. In words, the actor is:

#. Translated so that its ``origin`` is at the world origin.
#. Scaled by ``scale``.
#. Rotated by ``orientation``: about the y-axis first, then the x-axis, then the z-axis.
#. Translated back by ``origin``, and then by ``position``.
#. Transformed by ``user_matrix``.

The ``origin`` is the point that stays fixed while scaling and rotating, and
``position`` is an offset applied after them. This means the point ``origin`` of
the mesh ends up at ``origin + position``, and not at ``position``.

"""

# sphinx_gallery_start_ignore
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

import numpy as np
import pyvista as pv

# %%
# Define Helpers
# ==============
# Define a function which labels the ``origin``, the ``position``, and
# ``origin + position`` in a scene. Points at the same location share one label.
# A grey :class:`~pyvista.AxesAssembly` marks the world axes.

MARKER_COLORS = {'origin': 'purple', 'position': 'orange', 'origin + position': 'teal'}


def add_markers(pl, origin, position):
    points = {
        'origin': np.array(origin, dtype=float),
        'position': np.array(position, dtype=float),
    }
    points['origin + position'] = points['origin'] + points['position']

    groups = {}
    for name, point in points.items():
        groups.setdefault(tuple(point.round(6)), []).append(name)
    for point, names in groups.items():
        pl.add_point_labels(
            [point],
            [', '.join(names)],
            point_color=MARKER_COLORS[names[0]],
            point_size=20,
            render_points_as_spheres=True,
            font_size=14,
            always_visible=True,
        )

    world_axes = pv.AxesAssembly(
        x_color='lightgray',
        y_color='lightgray',
        z_color='lightgray',
        label_color='gray',
        label_size=30,
    )
    pl.add_actor(world_axes)


# %%
# Transform Properties One at a Time
# ==================================
# Apply each property on its own to an :class:`~pyvista.AxesAssembly`.
#
# - ``position`` moves the axes.
# - ``orientation`` rotates the axes about the ``origin``.
# - ``scale`` scales the axes about the ``origin``.
# - ``origin`` on its own does nothing. It only matters when combined with
#   ``orientation`` or ``scale``.

properties = {
    'position': dict(position=(2, 1, 0)),
    'orientation': dict(orientation=(0, 0, 45)),
    'scale': dict(scale=(2, 1, 1)),
    'origin': dict(origin=(1, 1, 0)),
}

pl = pv.Plotter(shape=(2, 2))
for index, (name, kwargs) in enumerate(properties.items()):
    pl.subplot(*divmod(index, 2))
    pl.add_text(f'{name}={kwargs[name]}', font_size=12)
    pl.add_actor(pv.AxesAssembly(**kwargs))
    add_markers(
        pl,
        origin=kwargs.get('origin', (0, 0, 0)),
        position=kwargs.get('position', (0, 0, 0)),
    )
    pl.view_xy()
pl.link_views()
pl.show()

# %%
# The Origin Is a Pivot, the Position Is an Offset
# ================================================
# Set both ``origin`` and ``position`` to ``(0, 0, 5)`` and rotate two axes with the
# same ``orientation``. One axes is scaled by 3. The axes do not end up at
# ``(0, 0, 5)``, and scaling moves them even further away.
#
# To see why, a black line connects each axes to its ``origin``, the point
# ``(0, 0, 5)`` in the axes' own coordinates. This point is the pivot for scaling and
# rotation, and it ends up at ``origin + position``, which is ``(0, 0, 10)``.
# The line is a regular mesh actor with the same transform properties as the axes.

center = (0, 0, 5)
orientation = (30, 20, 45)


def add_axes_and_pivot_line(pl, scale, labels, **kwargs):
    transform_properties = dict(orientation=orientation, scale=scale, **kwargs)
    pl.add_actor(pv.AxesAssembly(labels=labels, label_size=30, **transform_properties))

    origin = transform_properties.get('origin', (0, 0, 0))
    line = pl.add_mesh(pv.Line((0, 0, 0), origin), color='black', line_width=3)
    for name, value in transform_properties.items():
        setattr(line, name, value)


pl = pv.Plotter()
pl.add_text(f'origin=position={center}', font_size=12)
add_axes_and_pivot_line(
    pl, scale=1, labels=('X1', 'Y1', 'Z1'), origin=center, position=center
)
add_axes_and_pivot_line(
    pl, scale=3, labels=('X3', 'Y3', 'Z3'), origin=center, position=center
)
add_markers(pl, origin=center, position=center)
pl.show()

# %%
# For axes, the point which should stay fixed is the axes' own origin
# ``(0, 0, 0)``. So leave ``origin`` at its default and use ``position`` alone to move
# the axes. Both axes now pivot about their own origin and end up at ``(0, 0, 5)``.

pl = pv.Plotter()
pl.add_text(f'origin=(0, 0, 0), position={center}', font_size=12)
add_axes_and_pivot_line(pl, scale=1, labels=('X1', 'Y1', 'Z1'), position=center)
add_axes_and_pivot_line(pl, scale=3, labels=('X3', 'Y3', 'Z3'), position=center)
add_markers(pl, origin=(0, 0, 0), position=center)
pl.show()

# %%
# Place a Mesh at a Point
# =======================
# A mesh is usually not centered at ``(0, 0, 0)``. To rotate a mesh about its own
# center and move that center to a ``target`` point, set ``origin`` to the mesh's
# center and ``position`` to ``target - origin``.

cone = pv.Cone(center=(3, 0, 0), direction=(1, 0, 0), height=2, resolution=32)
target = np.array((0, 3, 2))

pl = pv.Plotter()
pl.add_mesh(cone, color='lightgray', opacity=0.5)
actor = pl.add_mesh(cone, color='tan')
actor.origin = cone.center
actor.position = target - cone.center
actor.orientation = (0, 0, 90)
add_markers(pl, origin=actor.origin, position=actor.position)
pl.show()

# %%
# Scale Before Rotation
# =====================
# Scaling is applied before rotation. With a non-uniform scale, the order matters:
# rotating first and scaling second shears the axes so that they are no longer at
# right angles.
#
# Compare an :class:`~pyvista.AxesAssembly` transformed with properties to two
# :class:`~pyvista.Transform` objects applied with ``user_matrix``.
# :class:`~pyvista.Transform` applies its operations in the order they are added.

scale = (3, 1, 1)
angle = 45

scale_then_rotate = pv.Transform().scale(scale).rotate_z(angle)
rotate_then_scale = pv.Transform().rotate_z(angle).scale(scale)

pl = pv.Plotter(shape=(1, 3))
pl.add_text(f'scale={scale}\norientation=(0, 0, {angle})', font_size=10)
pl.add_actor(pv.AxesAssembly(scale=scale, orientation=(0, 0, angle)))
pl.subplot(0, 1)
pl.add_text('Transform()\n.scale().rotate_z()', font_size=10)
pl.add_actor(pv.AxesAssembly(user_matrix=scale_then_rotate.matrix))
pl.subplot(0, 2)
pl.add_text('Transform()\n.rotate_z().scale()', font_size=10)
pl.add_actor(pv.AxesAssembly(user_matrix=rotate_then_scale.matrix))
pl.link_views()
pl.view_xy()
pl.show()

# %%
# Rotation Order
# ==============
# The ``orientation`` angles are given in x-y-z order, but the rotations are applied
# about the y-axis first, then the x-axis, then the z-axis. Rotating about the x-axis
# first gives a different result.

orientation = (60, 45, 0)

pl = pv.Plotter(shape=(1, 3))
pl.add_text(f'orientation={orientation}', font_size=10)
pl.add_actor(pv.AxesAssembly(orientation=orientation))
pl.subplot(0, 1)
pl.add_text('Transform()\n.rotate_y(45).rotate_x(60)', font_size=10)
transform = pv.Transform().rotate_y(orientation[1]).rotate_x(orientation[0])
pl.add_actor(pv.AxesAssembly(user_matrix=transform.matrix))
pl.subplot(0, 2)
pl.add_text('Transform()\n.rotate_x(60).rotate_y(45)', font_size=10)
transform = pv.Transform().rotate_x(orientation[0]).rotate_y(orientation[1])
pl.add_actor(pv.AxesAssembly(user_matrix=transform.matrix))
pl.link_views()
pl.show()

# %%
# User Matrix Last
# ================
# The ``user_matrix`` is applied after all other properties, so it also transforms the
# ``position``. Here, a 90 degree rotation about the z-axis moves the axes from
# ``position`` to ``(0, 2, 0)``.

position = (2, 0, 0)
rotation = pv.Transform().rotate_z(90)

pl = pv.Plotter()
pl.add_actor(pv.AxesAssembly(position=position, labels=('X1', 'Y1', 'Z1')))
pl.add_actor(
    pv.AxesAssembly(
        position=position, user_matrix=rotation.matrix, labels=('X2', 'Y2', 'Z2')
    )
)
add_markers(pl, origin=(0, 0, 0), position=position)
pl.view_xy()
pl.show()

# %%
# Equivalent Transforms
# =====================
# Build the same transformation in different ways and compare the results. Each
# scene shows two halves of a sphere: the red top half is transformed with all five
# properties, and the blue bottom half is transformed with an equivalent
# :class:`~pyvista.Transform`. When the two transformations are the same, the halves
# join to form a closed shape.

position = (-0.5, -0.5, 1)
orientation = (60, 20, 10)
scale = (1.5, 2, 2.5)
origin = (2, 1.5, 1)
user_matrix = pv.Transform().rotate_vector((1, 2, 3), 30).translate(1, -2, 0).matrix

hemisphere_top = pv.Sphere(start_phi=0, end_phi=90)
hemisphere_bottom = pv.Sphere(start_phi=90, end_phi=180)


def make_top_actor():
    actor = pv.Actor(mapper=pv.DataSetMapper(hemisphere_top))
    actor.position = position
    actor.orientation = orientation
    actor.scale = scale
    actor.origin = origin
    actor.user_matrix = user_matrix
    actor.prop.color = 'red'
    return actor


# %%
# Write out each step of the transformation explicitly.

explicit = (
    pv.Transform()
    .translate(-np.array(origin))
    .scale(scale)
    .rotate_y(orientation[1])
    .rotate_x(orientation[0])
    .rotate_z(orientation[2])
    .translate(origin)
    .translate(position)
    .compose(user_matrix)
)

# %%
# Use :attr:`~pyvista.Transform.point` instead of the translations to and from
# ``origin``. Scaling and rotations are then applied about this point. The point is
# cleared before composing the user matrix so that it is not also applied about the
# point.

compact = pv.Transform(point=origin)
compact.scale(scale)
compact.rotate_y(orientation[1])
compact.rotate_x(orientation[0])
compact.rotate_z(orientation[2])
compact.translate(position)
compact.point = None
compact.compose(user_matrix)

# %%
# Apply the rotations in x-y-z order instead. The halves no longer line up.

wrong_order = (
    pv.Transform(point=origin)
    .scale(scale)
    .rotate_x(orientation[0])
    .rotate_y(orientation[1])
    .rotate_z(orientation[2])
    .translate(position)
)
wrong_order.point = None
wrong_order.compose(user_matrix)

# %%
# Plot the comparisons. The bottom half is transformed with
# :meth:`Transform.apply() <pyvista.Transform.apply>`, which sets the actor's
# ``user_matrix``. In the third scene, the mesh itself is transformed instead of the
# actor, which gives the same result.

pl = pv.Plotter(shape=(2, 2))
scenes = {
    'Explicit steps': explicit,
    'Transform(point=origin)': compact,
    'Transformed mesh': compact,
    'Wrong rotation order': wrong_order,
}
for index, (title, transform) in enumerate(scenes.items()):
    pl.subplot(*divmod(index, 2))
    pl.add_text(title, font_size=12)
    pl.add_actor(make_top_actor())
    if title == 'Transformed mesh':
        pl.add_mesh(hemisphere_bottom.transform(transform, inplace=False), color='blue')
    else:
        bottom_actor = pv.Actor(mapper=pv.DataSetMapper(hemisphere_bottom))
        bottom_actor.prop.color = 'blue'
        pl.add_actor(transform.apply(bottom_actor, 'replace'))
pl.link_views()
pl.show()

# %%
# Recommendation
# ==============
# Setting ``position``, ``orientation``, ``scale``, and ``origin`` together is easy to
# get wrong, because the result depends on the fixed order shown above. Building a
# single :class:`~pyvista.Transform` makes the order explicit. Its 4x4
# :attr:`~pyvista.Transform.matrix` can be set as the ``user_matrix`` of any actor, and
# the same transform can also be applied to a mesh.
#
# .. seealso::
#
#     :class:`~pyvista.Prop3D`
#         Base class with the transform properties of all actors.
#     :class:`~pyvista.Transform`
#         Describe linear transformations via a 4x4 matrix.
#     :ref:`axes_objects_example`
#         Overview of axes objects.
#
# .. tags:: plot
