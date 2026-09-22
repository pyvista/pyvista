"""
.. _resample_reslice_example:

Resample and Reslice Images
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare the ``resample`` and ``reslice`` image filters.

:meth:`~pyvista.ImageDataFilters.resample` and :meth:`~pyvista.ImageDataFilters.reslice`
both create a new image by interpolating an existing one, but they answer different
questions. ``resample`` changes how densely an image is sampled and leaves it where it
is. ``reslice`` samples the image at the points of a second image, so the two images end
up on a common grid.

See :meth:`~pyvista.ImageDataFilters.resample` for changing an image's resolution and
for the interpolation and anti-aliasing options both filters share.

"""

import numpy as np

# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
from pyvista import examples

# %%
# Reslice
# +++++++
#
# Load two photographs. The bird is smaller than the gourds, and both start at the
# origin with unit spacing, so the bird covers the lower left corner of the region the
# gourds cover.

gourds = examples.download_gourds()
bird = examples.download_bird()
print(bird.dimensions, gourds.dimensions)

# %%
# ``reslice`` samples the bird at the points of the gourds image. The bird keeps its
# size and position, and the reference points which fall outside it take
# ``background_value``.

resliced = bird.reslice(gourds, 'linear', background_value=0)

# %%
# Plot the result with the outline of the gourds image in red. The bird fills the
# corner of the grid it occupies and the rest is background.

pl = pv.Plotter()
pl.add_mesh(resliced, rgba=True, lighting=False)
pl.add_mesh(gourds.outline(), color='red', line_width=3)
pl.view_xy()
pl.camera.tight()
pl.show()

# %%
# Position Matters
# ++++++++++++++++
#
# The difference is clearest on an image coarse enough to see every sample. Generate a
# small Mandelbrot set, and a reference which covers part of it at a finer spacing.

mandelbrot = pv.ImageMandelbrotSource(
    whole_extent=(0, 23, 0, 17, 0, 0), maxiter=25
).output
reference = pv.ImageData(
    dimensions=(17, 17, 1), spacing=(0.05, 0.05, 1.0), origin=(-1.0, -0.4, 0.0)
)

# %%
# ``reslice`` returns that region of the image, sampled at the reference's points.
# ``resample`` returns the whole image squeezed into the reference's geometry.

resliced = mandelbrot.reslice(reference, 'linear')
resampled = mandelbrot.resample(reference_image=reference, interpolation='linear')

# %%
# The region can also be cropped out of the image by hand and the crop resampled to the
# reference's spacing. :meth:`~pyvista.ImageDataFilters.crop` works in index space, so
# the reference's bounds have to be converted into the image's indices first.

lo = np.floor((np.array(reference.bounds[::2]) - mandelbrot.origin) / mandelbrot.spacing)
hi = np.ceil((np.array(reference.bounds[1::2]) - mandelbrot.origin) / mandelbrot.spacing)
cropped = mandelbrot.crop(extent=np.column_stack([lo, hi]).astype(int).ravel())
sample_rate = np.array(cropped.spacing) / reference.spacing
cropped_resampled = cropped.resample(
    (*sample_rate[:2], 1.0), 'linear', extend_border=False
)

# %%
# Plot each output over the image it came from. Use
# :meth:`~pyvista.ImageDataFilters.points_to_cells` to draw the samples as
# :attr:`~pyvista.CellType.PIXEL` cells with their edges showing, and outline the
# reference region in red. The resliced samples continue the picture around them,
# because that is where they were taken. The resampled ones are the whole set shrunk
# into the frame. The cropped ones carry the right picture but not the right grid.

clim = mandelbrot.get_data_range()

outputs = [
    ('reslice', resliced),
    ('resample', resampled),
    ('crop+resample', cropped_resampled),
]
pl = pv.Plotter(shape=(1, 3))
for index, (label, output) in enumerate(outputs):
    pl.subplot(0, index)
    for voxels in [mandelbrot.points_to_cells(), output.points_to_cells()]:
        pl.add_mesh(
            voxels, clim=clim, show_edges=True, lighting=False, show_scalar_bar=False
        )
    pl.add_mesh(reference.points_to_cells().outline(), color='red', line_width=4)
    pl.add_text(label, font_size=10)
    pl.view_xy()
    pl.camera.tight()
pl.show()

# %%
# ``reslice`` and ``resample`` both carry the reference's geometry, since that is what
# ``reference_image`` asks for. Only their values differ: ``reslice`` read the image at
# the reference's points, while ``resample`` stretched the whole image onto them. The
# crop can only land on whole input voxels, so it covers more than the reference asked
# for and its spacing cannot match either.

print(resliced.origin, resliced.spacing)
print(resampled.origin, resampled.spacing)
print(cropped_resampled.origin, cropped_resampled.spacing)

# %%
# The crop can be made to reproduce ``reslice`` exactly, but only once every condition
# it would otherwise take care of is met:
#
# #. The region has to begin and end on whole input voxels, so the reference cannot be
#    placed freely.
# #. Those voxel corners have to be converted into physical coordinates by hand to
#    build the reference.
# #. The output has to be sized to the region by hand, through either ``dimensions``
#    or ``spacing``.
# #. ``extend_border`` has to be disabled, so the output keeps the crop's point bounds
#    rather than its cell bounds.

extent = (7, 14, 5, 12, 0, 0)
dimensions = (17, 17, 1)
step = np.array(mandelbrot.spacing[:2])
lo = np.array(mandelbrot.origin[:2]) + step * extent[:4:2]
hi = np.array(mandelbrot.origin[:2]) + step * extent[1:4:2]
aligned = pv.ImageData(
    dimensions=dimensions,
    spacing=(*(hi - lo) / (np.array(dimensions[:2]) - 1), 1.0),
    origin=(*lo, 0.0),
)

by_reslice = mandelbrot.reslice(aligned, 'linear')
by_crop = mandelbrot.crop(extent=extent).resample(
    dimensions=dimensions, interpolation='linear', extend_border=False
)

# %%
# The two agree exactly. ``reslice`` does nothing the other filters cannot; it does it
# from the reference alone, which is the whole of its value.

print(np.allclose(by_reslice.bounds, by_crop.bounds))
print(np.allclose(by_reslice.active_scalars, by_crop.active_scalars))

# %%
# Transform or Reslice
# ++++++++++++++++++++
#
# Rotating an image with :meth:`~pyvista.DataObjectFilters.transform` and reslicing it
# through the same rotation are different operations. ``transform`` moves the image and
# leaves the values alone, recording the rotation in the image's
# :attr:`~pyvista.ImageData.direction_matrix`. ``reslice`` interpolates the values onto
# the reference's points, so the output keeps the reference's geometry.

rotate = pv.Transform().rotate_z(30)

moved = gourds.transform(rotate, inplace=False)
resliced = gourds.reslice(gourds, 'linear', transform=rotate, background_value=0)

# %%
# The moved image carries the rotation in the matrix which maps its indices to physical
# space. The resliced one is still on the axes it started on, and only its values changed.

print(moved.index_to_physical_matrix.round(3))
print(resliced.index_to_physical_matrix.round(3))

# %%
# Plot both with the outline of the original image in red. ``transform`` carries the
# picture out of that frame, while ``reslice`` fills the frame and writes
# ``background_value`` wherever the rotated image does not reach it.

# sphinx_gallery_start_ignore
# two full-resolution photographs push the interactive scene past the size limit
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

pl = pv.Plotter(shape=(1, 2))
for index, (image, label) in enumerate([(moved, 'transform'), (resliced, 'reslice')]):
    pl.subplot(0, index)
    pl.add_mesh(image, rgba=True, lighting=False)
    pl.add_mesh(gourds.outline(), color='red', line_width=3)
    pl.add_text(label, font_size=10)
    pl.view_xy()
    pl.camera.tight()
pl.show()

# %%
# Beyond a Matrix
# +++++++++++++++
#
# ``transform`` is limited to what an image's geometry can hold: an origin, a spacing
# and an orthogonal :attr:`~pyvista.ImageData.direction_matrix`. ``reslice`` resamples
# the values instead, so it also accepts transformations no image geometry could
# express. :class:`~pyvista.ThinPlateSplineTransform` bends space so that one set of
# points lands on another.
#
# Build an image with a curved structure running across it.

WIDTH, HEIGHT, MIDDLE = 121, 81, 40.0


def centerline(position):
    """Return the height of the curved structure at each ``position``."""
    return MIDDLE + 14.0 * np.sin(2 * np.pi * np.asarray(position) / (WIDTH - 1))


curved = pv.ImageData(dimensions=(WIDTH, HEIGHT, 1))
horizontal, vertical = curved.points[:, 0], curved.points[:, 1]
curved['scan'] = np.exp(-((vertical - centerline(horizontal)) ** 2) / (2 * 5.0**2))

# %%
# Map points along the centerline onto a straight line, pinning the top and bottom
# edges so the warp stays put where there is nothing to straighten.

samples = np.linspace(0, WIDTH - 1, 13)
source = [(sample, centerline(sample), 0.0) for sample in samples]
target = [(sample, MIDDLE, 0.0) for sample in samples]
for sample in np.linspace(0, WIDTH - 1, 7):
    for edge in (0.0, HEIGHT - 1.0):
        source.append((sample, edge, 0.0))
        target.append((sample, edge, 0.0))

warp = pv.ThinPlateSplineTransform(source, target)
straightened = curved.reslice(curved, 'linear', transform=warp)

# %%
# The structure now runs along a single row. ``transform`` could not have done this:
# it would have to keep the image's samples on a regular grid.

pl = pv.Plotter(shape=(1, 2))
panels = [(curved, 'curved'), (straightened, 'straightened')]
for index, (image, label) in enumerate(panels):
    pl.subplot(0, index)
    pl.add_mesh(image, cmap='bone', clim=[0, 1], show_scalar_bar=False, lighting=False)
    pl.add_text(label, font_size=10)
    pl.view_xy()
    pl.camera.tight(padding=0.1)
pl.show()

# %%
# Oblique Anatomy
# +++++++++++++++
#
# A structure which runs across the scan axes is awkward to read in the slices the
# scanner produced. Load a whole-body CT with its segmentations and take the left
# scapula, which lies at an angle within the axial plane.

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

axial = ct.slice_index(k=round((center[2] - ct.origin[2]) / ct.spacing[2]))

pl = pv.Plotter()
pl.add_mesh(axial, cmap='bone', clim=[-200, 900], show_scalar_bar=False, lighting=False)
pl.add_mesh(line.translate((0, 0, 1)), color='magenta', line_width=6)
pl.view_xy()
pl.camera.tight()
pl.show()

# %%
# Because the fit is confined to that plane, a single rotation about the scan axis is
# enough to bring the line onto the image's x axis.

angle = np.degrees(np.arctan2(direction[1], direction[0]))
rotation = pv.Transform().rotate_z(-angle).matrix[:3, :3]

reference = pv.ImageData(dimensions=(320, 220, 1), spacing=(1.0, 1.0, 1.0))


def centered(rot):
    """Return the matrix which centers the scapula in the reference under ``rot``."""
    matrix = np.eye(4)
    matrix[:3, :3] = rot
    matrix[:3, 3] = np.array(reference.center) - rot @ center
    return matrix


# %%
# Reslice twice from that one grid, once without a rotation and once with it, and put
# the line through the same transforms so it can be compared against.

along_scan = ct.reslice(
    reference, 'linear', transform=centered(np.eye(3)), background_value=-1000
)
along_bone = ct.reslice(
    reference, 'linear', transform=centered(rotation), background_value=-1000
)

line_scan = line.transform(centered(np.eye(3)), inplace=False)
line_bone = line.transform(centered(rotation), inplace=False)

# %%
# The rotated grid samples along the bone, and the line it was fitted to comes out level.

pl = pv.Plotter(shape=(1, 2))
panels = [(along_scan, line_scan, 'scan axes'), (along_bone, line_bone, 'bone axis')]
for index, (image, overlay, label) in enumerate(panels):
    pl.subplot(0, index)
    pl.add_mesh(
        image, cmap='bone', clim=[-200, 900], show_scalar_bar=False, lighting=False
    )
    pl.add_mesh(overlay.translate((0, 0, 1)), color='magenta', line_width=6)
    pl.add_text(label, font_size=10)
    pl.view_xy()
    pl.camera.tight()
pl.show()

# %%
# .. tags:: filter
