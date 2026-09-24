"""
.. _slice_oblique_anatomy_example:

Slicing Oblique Anatomy
~~~~~~~~~~~~~~~~~~~~~~~

Put a scan on a structure's own axes and slice it there.

A structure which runs across the scan axes is awkward to read in the slices the scanner
produced. :meth:`~pyvista.ImageDataFilters.reslice` puts the scan on a grid aligned to
the structure, and :meth:`~pyvista.DataObjectFilters.slice_orthogonal` and
:meth:`~pyvista.ImageDataFilters.slice_index` then cut it squarely.

See :ref:`slice_example` for the slicing filters on an axis-aligned volume, and
:ref:`reslice_example` for what ``reslice`` does on its own.

"""

import numpy as np

# sphinx_gallery_thumbnail_number = 3
import pyvista as pv
from pyvista import examples

# %%
# Load a whole-body CT with its segmentations and take the left scapula, which lies at
# an angle within the axial plane.

dataset = examples.download_whole_body_ct_male()
ct = dataset['ct']
scapula = dataset['segmentations']['scapula_left']

# %%
# Project the foreground of the bone's mask onto the axial plane and fit a line to it
# there. Seeding ``init_direction`` pins the sign, so the fitted direction cannot come
# back reversed.

foreground = scapula.points[scapula.active_scalars > 0]
center = foreground.mean(axis=0)

in_plane = foreground.copy()
in_plane[:, 2] = center[2]
line, _, direction = pv.fit_line_to_points(in_plane, init_direction='x', return_meta=True)

# %%
# Draw the line over the axial slice it was fitted in. It crosses the slice at an angle,
# and so does the bone underneath it.

index = round((center[2] - ct.origin[2]) / ct.spacing[2])
axial = ct.slice_index(k=index)
axial_mask = scapula.slice_index(k=index)
axial_mask['scapula'] = (np.asarray(axial_mask.active_scalars) > 0).astype(float) * 0.3

# sphinx_gallery_start_ignore
# the interactive scene renders blank, so keep the static figure
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

pl = pv.Plotter()
pl.add_mesh(axial, cmap='bone', clim=[-200, 900], show_scalar_bar=False, lighting=False)
pl.add_mesh(
    axial_mask.translate((0, 0, 0.5)), color='orange', opacity='scapula', lighting=False
)
pl.add_mesh(line.translate((0, 0, 1)), color='magenta', line_width=6)
pl.add_legend(
    [['scapula', 'orange'], ['fitted axis', 'magenta']],
    bcolor='w',
    loc='lower left',
    size=(0.28, 0.12),
    face='none',
)
pl.view_xy()
pl.camera.tight()
pl.show()

# %%
# Because the fit is confined to that plane, a single rotation about the scan axis is
# enough to bring the line onto the image's x axis.

angle = np.degrees(np.arctan2(direction[1], direction[0]))

plane = pv.ImageData(dimensions=(320, 220, 1), spacing=(1.0, 1.0, 1.0))


def centered(grid, rotation):
    """Return the transform which rotates the scapula and centers it in ``grid``."""
    return pv.Transform().translate(-center).rotate_z(rotation).translate(grid.center)


def masked(grid, rotation):
    """Return the bone's mask on ``grid``, carrying its opacity as an array."""
    mask = scapula.reslice(
        grid, 'nearest', transform=centered(grid, rotation), background_value=0
    )
    mask['scapula'] = (np.asarray(mask.active_scalars) > 0).astype(float) * 0.3
    return mask


# %%
# Reslice twice from that one grid, once without a rotation and once with it, and put
# the line and the bone's mask through the same transforms so they can be compared
# against. The mask is a label image, so it takes ``'nearest'``.

along_scan = ct.reslice(
    plane, 'linear', transform=centered(plane, 0.0), background_value=-1000
)
along_bone = ct.reslice(
    plane, 'linear', transform=centered(plane, -angle), background_value=-1000
)

line_scan = line.transform(centered(plane, 0.0), inplace=False)
line_bone = line.transform(centered(plane, -angle), inplace=False)

mask_scan = masked(plane, 0.0)
mask_bone = masked(plane, -angle)

# %%
# The rotated grid samples along the bone. The bone itself comes out level, and so does
# the line it was fitted to.

# sphinx_gallery_start_ignore
# the interactive scene renders blank, so keep the static figure
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

pl = pv.Plotter(shape=(1, 2))
panels = [
    (along_scan, mask_scan, line_scan, 'scan axes'),
    (along_bone, mask_bone, line_bone, 'bone axis'),
]
for index, (image, mask, axis, label) in enumerate(panels):
    pl.subplot(0, index)
    pl.add_mesh(
        image, cmap='bone', clim=[-200, 900], show_scalar_bar=False, lighting=False
    )
    pl.add_mesh(
        mask.translate((0, 0, 0.5)), color='orange', opacity='scapula', lighting=False
    )
    pl.add_mesh(axis.translate((0, 0, 1)), color='magenta', line_width=6)
    pl.add_text(label, font_size=10)
    pl.view_xy()
    pl.camera.tight()
pl.subplot(0, 0)
pl.add_legend(
    [['scapula', 'orange'], ['fitted axis', 'magenta']],
    bcolor='w',
    loc='lower left',
    size=(0.4, 0.14),
    face='none',
)
pl.show()

# %%
# The same rotation applied to a volume rather than a single plane gives a bone-aligned
# block of the scan. Reslice a generous one, then trim it to the bone with
# :meth:`~pyvista.ImageDataFilters.crop`, which works in index space and so needs the
# reslice to have happened first. See :ref:`crop_labeled_example` for that filter on
# its own.

volume = pv.ImageData(dimensions=(200, 160, 280), spacing=(1.0, 1.0, 1.0))
block = ct.reslice(
    volume, 'linear', transform=centered(volume, -angle), background_value=-1000
)
bone = masked(volume, -angle)

block = block.crop(mask=bone, padding=10)
bone = bone.crop(extent=block.extent)

# %%
# The trimmed block is axis-aligned, so
# :meth:`~pyvista.DataObjectFilters.slice_orthogonal` cuts the bone squarely along all
# three planes.

slices = block.slice_orthogonal()
mask_slices = bone.slice_orthogonal()

# %%
# View each plane face-on. ``XZ`` shows the blade with its spine and the glenoid, and
# ``XY`` is a cross-section through it. None of these planes cut the bone this way in
# the axes the scanner produced.

# sphinx_gallery_start_ignore
# the interactive scene renders a blank panel rather than the three planes
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

pl = pv.Plotter(shape=(1, 3), window_size=[1000, 620])
for index, name in enumerate(['XY', 'XZ', 'YZ']):
    pl.subplot(0, index)
    pl.add_mesh(
        slices[name], cmap='bone', clim=[-200, 900], show_scalar_bar=False, lighting=False
    )
    pl.add_mesh(mask_slices[name], color='orange', opacity='scapula', lighting=False)
    pl.add_text(name, font_size=10)
    pl.camera.tight(view=name.lower(), adjust_render_window=False, padding=0.1)
pl.subplot(0, 0)
pl.add_legend(
    [['scapula', 'orange']], bcolor='w', loc='lower left', size=(0.5, 0.09), face='none'
)
pl.show()

# %%
# ``slice_orthogonal`` cuts at the block's center. To cut elsewhere, use
# :meth:`~pyvista.ImageDataFilters.slice_index`, which takes an index along each axis
# and returns :class:`~pyvista.ImageData`. Step a series of ``XZ`` planes through the
# blade and lay them side by side with :meth:`~pyvista.ImageDataFilters.concatenate`.

indices = [38, 41, 44, 47, 50]
planes = [block.slice_index(j=index, rebase_coordinates=True) for index in indices]
masks = [bone.slice_index(j=index, rebase_coordinates=True) for index in indices]

strip = planes[0].concatenate(planes[1:], 'x')
strip_mask = masks[0].concatenate(masks[1:], 'x')

# %%
# The blade fans out as the planes move forward, and the last of them is the one
# ``slice_orthogonal`` chose.

# sphinx_gallery_start_ignore
# the interactive scene renders blank, so keep the static figure
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

pl = pv.Plotter(window_size=[1200, 420])
pl.add_mesh(strip, cmap='bone', clim=[-200, 900], show_scalar_bar=False, lighting=False)
pl.add_mesh(strip_mask, color='orange', opacity='scapula', lighting=False)
pl.camera.tight(view='xz', padding=0.05)
pl.show()

# %%
# .. tags:: filter
