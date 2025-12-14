"""
.. _compare_threshold_filters_example:

Compare threshold filters
~~~~~~~~~~~~~~~~~~~~~~~~~

Multiple filters exist to exclude scalar values or to highlight them for values of interest.
The goal of this example is to compare some of these filters to show how each can be used.
"""
from __future__ import annotations

import numpy as np

# sphinx_gallery_thumbnail_number = 4
import pyvista as pv
from pyvista import examples


# %%
# Volume Data to Volume Data
# ++++++++++++++++++++++++++
# There are a few filters like :func:`pyvista.DataSetFilters.clip_scalar`
method_map = {'threshold': 0, 'image_threshold': 1}
step = -80
values = (155, 580)
volume = pv.examples.download_carotid()
thresholded_vol = volume.threshold(values)
thresholded_vol = thresholded_vol.translate([step, 0, 0]) #.ORIGIN[:, 0] += 80
image_thresholded_vol = volume.image_threshold(values, in_value=580)
image_thresholded_vol = image_thresholded_vol.translate([2 * step, 0, 0]) #.ORIGIN[:, 0] += 80

pl = pv.Plotter()
pl.add_volume(volume)
pl.add_volume(thresholded_vol)
pl.add_volume(image_thresholded_vol)
# pl.add_mesh(thresholded_vol,  color = "g", style='volume')

pl.show()
print(pl.camera_position)
# %%
# Volume Data to Unstructured Grid
# ++++++++++++++++++++++++++++++++
# There are a few filters like :func:`pyvista.DataSetFilters.clip_scalar`
method_map = {'threshold': 0, 'extract_values': 1, "clip_scalar": 2}

volume = pv.examples.download_carotid()
step = -80
values = (155, 580)
mesh = pv.UnstructuredGrid()
print(mesh)
pl = pv.Plotter()
clipped_vol = volume.clip_scalar(value=values)
clipped_vol.cell_data["method"] = np.full(
    (clipped_vol.number_of_cells), 
    method_map["clip_scalar"]
)
mesh += clipped_vol
print(mesh)

extracted_values_vol = volume.extract_values(values)
extracted_values_vol.cell_data["method"] = np.full(
    (extracted_values_vol.number_of_cells), 
    method_map["extract_values"]
)
extracted_values_vol.points[:, 0] += step
mesh += extracted_values_vol
print(mesh)

thresholded_vol = volume.threshold(values)
thresholded_vol.cell_data["method"] = np.full(
    (thresholded_vol.number_of_cells), 
    method_map["threshold"]
)
thresholded_vol.points[:, 0] += step * 2
mesh += thresholded_vol
print(mesh)
mesh.set_active_scalars("method")
colored_mesh, color_map = mesh.color_labels(output_scalars='method', return_dict=True)
legend_map = dict(zip(method_map.keys(), color_map.values(), strict=True))

pl.add_mesh(colored_mesh, style='wireframe', rgb=True)
pl.add_legend(legend_map)

# pl.add_mesh(thresholded_vol,  color = "g", style='wireframe')
# pl.add_mesh(extracted_values_vol,  color = "b", style='wireframe')
pl.show()
print(pl.camera_position)
pl.close()
# %%
# Unstructured Grid to Unstructured Grid
# ++++++++++++++++++++++++++++++++++++++
# There are a few filters like :func:`pyvista.DataSetFilters.clip_scalar`
values = (-1, 1)
step = -5
surface = examples.download_foot_bones()
surface.point_data["z"] = surface.points[:, 2]
pl = pv.Plotter()
thresholded_surf = surface.threshold(values)
clipped_surf = surface.clip_scalar(value=values)
clipped_surf.points[:, 2] += step
pl.add_mesh(thresholded_surf, color = "r", style='wireframe')
pl.add_mesh(clipped_surf,  color = "g", style='wireframe')
pl.show()

# %%
# .. tags:: load
