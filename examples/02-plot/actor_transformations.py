r"""
.. _actor_transformations_example:

Actor Transformations
~~~~~~~~~~~~~~~~~~~~~

Place actors with a pivot, a rotation, a scale and an offset, and know where they end up.

Every :class:`~pyvista.Prop3D`, such as an :class:`~pyvista.Actor` or an
:class:`~pyvista.AxesAssembly`, has five transform properties:
:attr:`~pyvista.Prop3D.origin`, :attr:`~pyvista.Prop3D.scale`,
:attr:`~pyvista.Prop3D.orientation`, :attr:`~pyvista.Prop3D.position`, and
:attr:`~pyvista.Prop3D.user_matrix`. They work like the pivot, scale, rotation, and
location of an object in a 3D modelling tool. A point :math:`x` of the mesh is drawn at

.. math::

    x' = M \left( R \, S \, (x - o) + o + p \right)

The ``origin`` :math:`o` is the pivot. The mesh is scaled by :math:`S` and rotated by
:math:`R` about the pivot, moved by the ``position`` :math:`p`, and finally transformed by
the ``user_matrix`` :math:`M`. Two consequences follow directly:

- Rotation and scaling do not move the pivot. The pivot ends up at ``origin + position``.
- The ``position`` is an offset, not a destination. Only when ``origin`` is zero does the
  mesh origin land at ``position``.

"""

# sphinx_gallery_start_ignore
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

import numpy as np
import pyvista as pv
from pyvista import examples

# sphinx_gallery_thumbnail_number = 6

# %%
# Define a helper which adds a unit grid on the floor and small world axes at the
# origin, and frames the camera on the scene without the floor.


def add_context(pl, *, view='iso', bounds=None):
    world_axes = pv.AxesAssembly(
        shaft_radius=0.01,
        tip_radius=0.04,
        x_color='gray',
        y_color='gray',
        z_color='gray',
        show_labels=False,
    )
    pl.add_actor(world_axes)
    floor = pv.Plane(i_size=400, j_size=400, i_resolution=400, j_resolution=400)
    floor_actor = pl.add_mesh(floor, style='wireframe', color='lightgray', lighting=False)
    floor_actor.use_bounds = False
    if view == 'top':
        pl.view_vector((0, -1, 1.5), viewup=(0, 1, 0), bounds=bounds)
    else:
        pl.view_vector((1, 1, 1), viewup=(0, 0, 1), bounds=bounds)


# %%
# Rotate About the Pivot
# ======================
# Start with a cone which does not sit at the world origin, and rotate it 90 degrees
# about the z-axis. With the default ``origin``, the cone swings around the world
# origin. With ``origin`` set to the cone's center, it spins in place. The
# untransformed cone is shown in grey, and the pivot is marked with a sphere.

cone = pv.Cone(center=(3, 0, 0.5), direction=(1, 0, 0), height=2, radius=0.5)
cases = {
    'origin=(0, 0, 0)': (0, 0, 0),
    'origin=cone.center': cone.center,
}

pl = pv.Plotter(shape=(1, 2))
for index, (title, origin) in enumerate(cases.items()):
    pl.subplot(0, index)
    pl.add_text(title, font_size=12)
    pl.add_mesh(cone, color='lightgray', opacity=0.4)
    actor = pl.add_mesh(cone, color='tan')
    actor.origin = origin
    actor.orientation = (0, 0, 90)
    pl.add_mesh(pv.Sphere(radius=0.15, center=origin), color='purple')
    add_context(pl)
pl.link_views()
pl.show()

# %%
# Scaling works the same way. It is applied about the pivot, so a cone scaled with the
# default ``origin`` moves away from the world origin as well as growing.

pl = pv.Plotter(shape=(1, 2))
for index, (title, origin) in enumerate(cases.items()):
    pl.subplot(0, index)
    pl.add_text(title, font_size=12)
    pl.add_mesh(cone, color='lightgray', opacity=0.4)
    actor = pl.add_mesh(cone, color='tan')
    actor.origin = origin
    actor.scale = 2
    pl.add_mesh(pv.Sphere(radius=0.15, center=origin), color='purple')
    add_context(pl)
pl.link_views()
pl.show()

# %%
# Position Is an Offset
# =====================
# A common mistake is to set ``origin`` and ``position`` to the same point, expecting
# the actor to pivot about that point and end up there. Here two
# :class:`~pyvista.AxesAssembly` are meant to sit at the orange target ``(0, 0, 5)``.
# On the left, ``origin`` and ``position`` are both the target, so the axes pivot about
# their own ``(0, 0, 5)`` point, which lands at ``origin + position = (0, 0, 10)``. The
# axes themselves land elsewhere, and scaling pushes them further. On the right,
# ``origin`` is left at zero, so both axes pivot about their own origin and land on the
# target.

target = (0, 0, 5)
orientation = (30, 20, 45)
cases = {
    'origin=target, position=target': dict(origin=target, position=target),
    'origin=(0, 0, 0), position=target': dict(position=target),
}

pl = pv.Plotter(shape=(1, 2))
for index, (title, kwargs) in enumerate(cases.items()):
    pl.subplot(0, index)
    pl.add_text(title, font_size=12)
    pl.add_actor(
        pv.AxesAssembly(orientation=orientation, labels=('1', '2', '3'), **kwargs)
    )
    pl.add_actor(
        pv.AxesAssembly(
            orientation=orientation, scale=3, labels=('X', 'Y', 'Z'), **kwargs
        )
    )
    pl.add_mesh(pv.Sphere(radius=0.3, center=target), color='orange', label='target')
    pl.add_legend(face='circle')
    add_context(pl)
pl.link_views()
pl.show()

# %%
# To rotate a mesh about its own center `and` move that center to a target, set
# ``origin`` to the center and ``position`` to ``target - origin``.

# %%
# The Same Transform, Written Out
# ===============================
# The properties are implicit: the rotation order and the pivot are fixed and hidden.
# :class:`~pyvista.Transform` makes every step explicit. Build the transform which
# matches an actor with all five properties set.

position = (-0.5, -0.5, 1)
orientation = (60, 20, 10)
scale = (1.5, 2, 2.5)
origin = (2, 1.5, 1)
user_matrix = pv.Transform().rotate_vector((1, 2, 3), 30).translate(1, -2, 0).matrix

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
# Note the rotation order: ``orientation`` is given in x-y-z order, but the rotations are
# applied about y, then x, then z. :attr:`~pyvista.Transform.point` replaces the two
# translations to and from the pivot. The point is cleared before composing the user
# matrix, which is not applied about the pivot.

compact = pv.Transform(point=origin)
compact.scale(scale)
compact.rotate_y(orientation[1])
compact.rotate_x(orientation[0])
compact.rotate_z(orientation[2])
compact.translate(position)
compact.point = None
compact.compose(user_matrix)

# %%
# Compare the two by transforming the two halves of a sphere: the red top half through
# the actor's properties, the blue bottom half through the transform with
# :meth:`Transform.apply() <pyvista.Transform.apply>`, which sets its ``user_matrix``.
# The halves form a closed sphere only when the transforms are the same. Applying the
# rotations in x-y-z order instead splits them apart.

wrong_order = pv.Transform(point=origin)
wrong_order.scale(scale)
wrong_order.rotate_x(orientation[0])
wrong_order.rotate_y(orientation[1])
wrong_order.rotate_z(orientation[2])
wrong_order.translate(position)
wrong_order.point = None
wrong_order.compose(user_matrix)

hemisphere_top = pv.Sphere(start_phi=0, end_phi=90)
hemisphere_bottom = pv.Sphere(start_phi=90, end_phi=180)
cases = {'Rotated y, x, z': compact, 'Rotated x, y, z': wrong_order}

pl = pv.Plotter(shape=(1, 2))
for index, (title, transform) in enumerate(cases.items()):
    pl.subplot(0, index)
    pl.add_text(title, font_size=12)
    top_actor = pl.add_mesh(hemisphere_top, color='red')
    top_actor.position = position
    top_actor.orientation = orientation
    top_actor.scale = scale
    top_actor.origin = origin
    top_actor.user_matrix = user_matrix
    bottom_actor = pl.add_mesh(hemisphere_bottom, color='blue')
    transform.apply(bottom_actor, 'replace', copy=False)
    add_context(pl)
pl.link_views()
pl.show()

# %%
# Local and Global Frames
# =======================
# Meshes from different sources come in different coordinate systems. These three
# sharks are different sizes, lie along different axes, and one is off-center. Each
# mesh has a natural local frame, which :meth:`~pyvista.DataSetFilters.align_xyz` finds
# from the :func:`~pyvista.principal_axes` of its points. The returned matrix is the
# rotation into that frame, after centering. Show each shark with its local frame.

sharks = {
    'hammerhead': examples.download_shark(),
    'great white': examples.download_great_white_shark(),
    'grey nurse': examples.download_grey_nurse_shark(),
}
colors = {'hammerhead': 'tomato', 'great white': 'steelblue', 'grey nurse': 'goldenrod'}
seeds = {
    'hammerhead': dict(axis_0_direction='-x', axis_2_direction='z'),
    'great white': dict(axis_0_direction='z', axis_2_direction='-x'),
    'grey nurse': dict(axis_0_direction='-x', axis_2_direction='z'),
}
frames = {}
for name, shark in sharks.items():
    _, matrix = shark.align_xyz(return_matrix=True, **seeds[name])
    frames[name] = matrix[:3, :3]

pl = pv.Plotter(shape=(1, 3))
for index, (name, shark) in enumerate(sharks.items()):
    pl.subplot(0, index)
    pl.add_text(name, font_size=12)
    pl.add_mesh(shark, color=colors[name])
    local_frame = pv.Transform().scale(shark.length / 3).rotate(frames[name].T)
    local_frame.translate(shark.center)
    pl.add_actor(pv.AxesAssembly(user_matrix=local_frame.matrix))
    pl.view_isometric()
pl.show()

# %%
# Line the sharks up in the global frame, all the same length and swimming the same
# way. Each one pivots about its own center, is rotated by its local frame and scaled
# to a common length, and is offset to a slot on the grid. The meshes are not modified.
# The grid is 1 unit, so the hammerhead, which is 167 units long in its file, is now 10.

length = 10


def line_up(pl, name, slot, **kwargs):
    shark = sharks[name]
    actor = pl.add_mesh(shark, **kwargs)
    actor.origin = shark.center
    actor.rotation_from(frames[name])
    actor.scale = length / shark.length
    actor.position = np.array(slot) - shark.center
    return actor


pl = pv.Plotter()
for index, name in enumerate(sharks):
    line_up(pl, name, slot=(0, 12 * (index - 1), 3), color=colors[name], label=name)
pl.add_legend(face='circle')
add_context(pl)
pl.show()

# %%
# .. seealso::
#
#     :class:`~pyvista.Prop3D`
#         The transform properties shared by all actors.
#     :class:`~pyvista.Transform`
#         Explicit transformations as a 4x4 matrix.
#     :ref:`icp_registration_example`
#         Apply a recovered registration to an actor with ``user_matrix``.
#     :ref:`axes_objects_example`
#         Overview of axes objects.
#
# .. tags:: plot
